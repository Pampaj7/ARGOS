#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from scripts.temporal_refinement.lib.losses import edge_aware_smoothness
from scripts.temporal_refinement.lib.models import AdaptiveMotionFusionRefiner
from scripts.temporal_refinement.lib.training import colorize


class AdaptiveFusionClipDataset(Dataset):
    def __init__(
        self,
        cache_root: Path,
        sequence_ids: list[str],
        sequence_length: int,
        crop_size: tuple[int, int],
        random_crop: bool,
        disp_norm: float = 128.0,
    ):
        self.cache_root = cache_root
        self.sequence_length = sequence_length
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.disp_norm = float(disp_norm)
        with (cache_root / "index.csv").open() as f:
            rows = list(csv.DictReader(f))
        wanted = set(sequence_ids)
        by_seq: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            if row["sequence_id"] in wanted:
                by_seq.setdefault(row["sequence_id"], []).append(row)
        self.clips: list[list[dict[str, str]]] = []
        for _seq, seq_rows in sorted(by_seq.items()):
            seq_rows = sorted(seq_rows, key=lambda r: int(r["center_frame_id"]))
            for start in range(0, len(seq_rows) - sequence_length + 1):
                self.clips.append(seq_rows[start : start + sequence_length])
        if not self.clips:
            raise RuntimeError(f"No clips found for {sequence_ids}")

    def __len__(self) -> int:
        return len(self.clips)

    def _load_disp(self, rel: str, y: int, x: int, h: int, w: int) -> np.ndarray:
        arr = np.load(self.cache_root / rel, mmap_mode="r")
        return np.asarray(arr[y : y + h, x : x + w], dtype=np.float32)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        rows = self.clips[idx]
        h0, w0 = int(rows[0]["height"]), int(rows[0]["width"])
        ch, cw = self.crop_size
        if self.random_crop:
            y, x = random.randint(0, h0 - ch), random.randint(0, w0 - cw)
        else:
            y, x = (h0 - ch) // 2, (w0 - cw) // 2

        rgbs, raw, spatial, sav = [], [], [], []
        for row in rows:
            rgb = cv2.imread(str(self.cache_root / row["rgb_center_path"]), cv2.IMREAD_COLOR)
            if rgb is None:
                raise RuntimeError(f"Could not read {row['rgb_center_path']}")
            rgb = cv2.cvtColor(rgb[y : y + ch, x : x + cw], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rgbs.append(torch.from_numpy(rgb).permute(2, 0, 1))
            raw.append(torch.from_numpy(self._load_disp(row["s2m2_s512_t_path"], y, x, ch, cw)).unsqueeze(0))
            spatial.append(torch.from_numpy(self._load_disp(row["s2m2_l736_t_path"], y, x, ch, cw)).unsqueeze(0))
            sav.append(torch.from_numpy(self._load_disp(row["sav_t_path"], y, x, ch, cw)).unsqueeze(0))

        return {
            "rgb": torch.stack(rgbs).float(),
            "raw": torch.stack(raw).float(),
            "spatial": torch.stack(spatial).float(),
            "sav": torch.stack(sav).float(),
            "sequence_id": rows[0]["sequence_id"],
        }


def ddp_setup() -> tuple[bool, int, int, int]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        torch.cuda.set_device(local)
        dist.init_process_group("nccl")
    else:
        rank = 0
        local = 0
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
    return world > 1, rank, local, world


def is_main() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def forward_clip(model, batch: dict, disp_norm: float) -> dict[str, torch.Tensor]:
    rgb = batch["rgb"]
    raw = batch["raw"]
    sav = batch["sav"]
    b, t, _c, h, w = rgb.shape
    hidden = None
    fused, alphas, resets, residuals = [], [], [], []
    prev_fused = raw[:, 0]
    prev_rgb = rgb[:, 0]
    for i in range(t):
        raw_i = raw[:, i]
        if i == 0:
            warped_prev = raw_i
            raw_diff = torch.zeros_like(raw_i)
            flow_mag = torch.zeros_like(raw_i)
            flow_valid = torch.ones_like(raw_i)
        else:
            warped_prev = prev_fused
            raw_diff = torch.abs(raw_i - warped_prev) / disp_norm
            flow_mag = torch.mean(torch.abs(rgb[:, i] - prev_rgb), dim=1, keepdim=True)
            flow_valid = (raw_diff < 0.25).float()
        x = torch.cat(
            [
                rgb[:, i],
                raw_i / disp_norm,
                warped_prev / disp_norm,
                raw_diff,
                flow_mag,
                flow_valid,
            ],
            dim=1,
        )
        fused_i, alpha_i, reset_i, residual_i, hidden = model(x, raw_i, warped_prev, hidden)
        fused.append(fused_i)
        alphas.append(alpha_i)
        resets.append(reset_i)
        residuals.append(residual_i)
        prev_fused = fused_i
        prev_rgb = rgb[:, i]
    return {
        "fused": torch.stack(fused, dim=1),
        "alpha": torch.stack(alphas, dim=1),
        "reset": torch.stack(resets, dim=1),
        "residual": torch.stack(residuals, dim=1),
        "raw": raw,
        "spatial": batch["spatial"],
        "sav": sav,
        "rgb": rgb,
    }


def reset_target_from_clip(out: dict[str, torch.Tensor], args) -> torch.Tensor:
    raw = out["raw"]
    rgb = out["rgb"]
    raw_diff = torch.abs(raw[:, 1:] - raw[:, :-1]) / args.disp_norm
    rgb_diff = torch.mean(torch.abs(rgb[:, 1:] - rgb[:, :-1]), dim=2, keepdim=False)
    rgb_diff = rgb_diff.unsqueeze(2)
    return ((raw_diff > args.reset_disp_threshold) | (rgb_diff > args.reset_rgb_threshold)).float()


def balanced_bce_prob(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pos = target.mean().clamp(1e-4, 1.0 - 1e-4)
    pos_weight = (1.0 - pos) / pos
    weight = torch.where(target > 0.5, pos_weight, torch.ones_like(target))
    return F.binary_cross_entropy(prob.float().clamp(1e-4, 1.0 - 1e-4), target.float(), weight=weight)


def loss_for_outputs(out: dict[str, torch.Tensor], args, epoch: int) -> tuple[torch.Tensor, dict[str, float]]:
    fused, raw, spatial, sav = out["fused"], out["raw"], out["spatial"], out["sav"]
    residual, alpha, reset = out["residual"], out["alpha"], out["reset"]
    loss_spatial = F.smooth_l1_loss(fused, spatial)
    loss_sav = F.smooth_l1_loss(fused, sav)
    loss_raw = F.smooth_l1_loss(fused, raw)
    if fused.shape[1] > 1:
        loss_motion = F.smooth_l1_loss(fused[:, 1:] - fused[:, :-1], sav[:, 1:] - sav[:, :-1])
        invalid_proxy = reset_target_from_clip(out, args)
        with torch.autocast("cuda", enabled=False):
            loss_reset = balanced_bce_prob(reset[:, 1:], invalid_proxy)
        reset_target_prevalence = float(invalid_proxy.detach().mean().cpu())
    else:
        loss_motion = fused.new_tensor(0.0)
        loss_reset = fused.new_tensor(0.0)
        reset_target_prevalence = 0.0
    loss_res = torch.mean(torch.abs(residual))
    alpha_prior_scale = max(0.0, 1.0 - float(epoch - 1) / max(1, args.alpha_prior_epochs))
    loss_alpha = alpha_prior_scale * F.smooth_l1_loss(alpha, torch.full_like(alpha, args.alpha_prior))
    edge = torch.stack([edge_aware_smoothness(residual[:, i], out["rgb"][:, i]) for i in range(residual.shape[1])]).mean()
    loss = (
        args.spatial_weight * loss_spatial
        + args.sav_weight * loss_sav
        + args.motion_weight * loss_motion
        + args.raw_weight * loss_raw
        + args.res_weight * loss_res
        + args.alpha_weight * loss_alpha
        + args.edge_weight * edge
        + args.reset_weight * loss_reset
    )
    return loss, {
        "loss": float(loss.detach().cpu()),
        "loss_spatial": float(loss_spatial.detach().cpu()),
        "loss_sav": float(loss_sav.detach().cpu()),
        "loss_motion": float(loss_motion.detach().cpu()),
        "loss_raw": float(loss_raw.detach().cpu()),
        "loss_res": float(loss_res.detach().cpu()),
        "loss_alpha": float(loss_alpha.detach().cpu()),
        "loss_edge": float(edge.detach().cpu()),
        "loss_reset": float(loss_reset.detach().cpu()),
        "alpha_prior_scale": float(alpha_prior_scale),
        "reset_target_prevalence": reset_target_prevalence,
    }


def mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({k for r in rows for k in r})
    return {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}


@torch.no_grad()
def evaluate(model, loader, device: torch.device, args) -> dict[str, float]:
    model.eval()
    rows = []
    for batch in loader:
        out = forward_clip(model, to_device(batch, device), args.disp_norm)
        fused, raw, spatial, sav = out["fused"], out["raw"], out["spatial"], out["sav"]
        ema = fixed_ema(raw, alpha=0.50)
        target = reset_target_from_clip(out, args) if fused.shape[1] > 1 else torch.zeros_like(out["reset"][:, 1:])
        row = {
            "raw_to_spatial_mae": float(torch.mean(torch.abs(raw - spatial)).cpu()),
            "ema_to_spatial_mae": float(torch.mean(torch.abs(ema - spatial)).cpu()),
            "fused_to_spatial_mae": float(torch.mean(torch.abs(fused - spatial)).cpu()),
            "raw_to_sav_mae": float(torch.mean(torch.abs(raw - sav)).cpu()),
            "ema_to_sav_mae": float(torch.mean(torch.abs(ema - sav)).cpu()),
            "fused_to_sav_mae": float(torch.mean(torch.abs(fused - sav)).cpu()),
            "alpha_mean": float(out["alpha"].mean().cpu()),
            "alpha_std": float(out["alpha"].std().cpu()),
            "alpha_min": float(out["alpha"].min().cpu()),
            "alpha_max": float(out["alpha"].max().cpu()),
            "alpha_hist_0_20": float((out["alpha"] < 0.2).float().mean().cpu()),
            "alpha_hist_20_40": float(((out["alpha"] >= 0.2) & (out["alpha"] < 0.4)).float().mean().cpu()),
            "alpha_hist_40_60": float(((out["alpha"] >= 0.4) & (out["alpha"] < 0.6)).float().mean().cpu()),
            "alpha_hist_60_80": float(((out["alpha"] >= 0.6) & (out["alpha"] < 0.8)).float().mean().cpu()),
            "alpha_hist_80_100": float((out["alpha"] >= 0.8).float().mean().cpu()),
            "reset_mean": float(out["reset"].mean().cpu()),
            "reset_std": float(out["reset"].std().cpu()),
            "reset_min": float(out["reset"].min().cpu()),
            "reset_max": float(out["reset"].max().cpu()),
            "reset_target_prevalence": float(target.mean().cpu()) if fused.shape[1] > 1 else 0.0,
            "residual_abs_mean": float(torch.mean(torch.abs(out["residual"])).cpu()),
        }
        if fused.shape[1] > 1:
            row.update(
                {
                    "raw_temporal_mae": float(torch.mean(torch.abs(raw[:, 1:] - raw[:, :-1])).cpu()),
                    "ema_temporal_mae": float(torch.mean(torch.abs(ema[:, 1:] - ema[:, :-1])).cpu()),
                    "fused_temporal_mae": float(torch.mean(torch.abs(fused[:, 1:] - fused[:, :-1])).cpu()),
                    "sav_temporal_mae": float(torch.mean(torch.abs(sav[:, 1:] - sav[:, :-1])).cpu()),
                    "ema_teacher_delta_mae": float(torch.mean(torch.abs((ema[:, 1:] - ema[:, :-1]) - (sav[:, 1:] - sav[:, :-1]))).cpu()),
                    "teacher_delta_mae": float(torch.mean(torch.abs((fused[:, 1:] - fused[:, :-1]) - (sav[:, 1:] - sav[:, :-1]))).cpu()),
                }
            )
        rows.append(row)
    return mean_rows(rows)


def fixed_ema(raw: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    filtered = [raw[:, 0]]
    prev = raw[:, 0]
    for t in range(1, raw.shape[1]):
        prev = alpha * raw[:, t] + (1.0 - alpha) * prev
        filtered.append(prev)
    return torch.stack(filtered, dim=1)


def tensor_l2_norm(tensors: list[torch.Tensor]) -> float:
    total = 0.0
    for tensor in tensors:
        total += float(torch.sum(tensor.detach().float() ** 2).cpu())
    return total ** 0.5


def gradient_stats(model: torch.nn.Module) -> dict[str, float]:
    groups: dict[str, list[torch.Tensor]] = {"enc": [], "gru": [], "dec": [], "head": [], "all": []}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        groups["all"].append(grad)
        key = "head" if "head" in name else "gru" if "gru" in name else "dec" if "dec" in name else "enc"
        groups[key].append(grad)
    return {f"grad_norm_{key}": tensor_l2_norm(vals) if vals else 0.0 for key, vals in groups.items()}


def parameter_update_norm(before: dict[str, torch.Tensor], model: torch.nn.Module) -> float:
    diffs = []
    for name, param in model.named_parameters():
        if name in before:
            diffs.append((param.detach().float().cpu() - before[name]).reshape(-1))
    return float(torch.linalg.vector_norm(torch.cat(diffs)).item()) if diffs else 0.0


@torch.no_grad()
def save_reference_images(model, dataset: AdaptiveFusionClipDataset, out_dir: Path, device: torch.device, args) -> list[str]:
    model.eval()
    ref_dir = out_dir / "reference_images"
    ref_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx in np.linspace(0, len(dataset) - 1, min(4, len(dataset)), dtype=int):
        batch = dataset[int(idx)]
        collated = {k: v.unsqueeze(0) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = forward_clip(model, to_device(collated, device), args.disp_norm)
        ema = fixed_ema(out["raw"], alpha=0.50)
        t = out["fused"].shape[1] // 2
        rgb = (out["rgb"][0, t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        raw = out["raw"][0, t, 0].cpu().numpy()
        ema_t = ema[0, t, 0].cpu().numpy()
        fused = out["fused"][0, t, 0].cpu().numpy()
        sav = out["sav"][0, t, 0].cpu().numpy()
        alpha = out["alpha"][0, t, 0].cpu().numpy()
        reset = out["reset"][0, t, 0].cpu().numpy()
        residual = out["residual"][0, t, 0].cpu().numpy()
        td = np.abs(out["fused"][0, t, 0].cpu().numpy() - out["fused"][0, max(0, t - 1), 0].cpu().numpy())
        vmax = float(np.nanpercentile(np.concatenate([raw.ravel(), ema_t.ravel(), fused.ravel(), sav.ravel()]), 99))
        tiles = [
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            colorize(raw, vmax),
            colorize(ema_t, vmax),
            colorize(fused, vmax),
            colorize(sav, vmax),
            colorize(alpha, 1.0, cv2.COLORMAP_VIRIDIS),
            colorize(reset, 1.0, cv2.COLORMAP_MAGMA),
            colorize(residual, args.residual_clamp_px, cv2.COLORMAP_TURBO),
            colorize(td, 8.0, cv2.COLORMAP_MAGMA),
        ]
        labels = ["RGB", "raw S2M2-S", "EMA0.50", "adaptive", "SAV", "alpha", "reset", "residual", "temp err"]
        small = []
        for tile, label in zip(tiles, labels):
            tile = cv2.resize(tile, (180, 126), interpolation=cv2.INTER_AREA)
            cv2.putText(tile, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            small.append(tile)
        path = ref_dir / f"adaptive_fusion_ref_{int(idx):04d}.png"
        cv2.imwrite(str(path), np.concatenate(small, axis=1))
        paths.append(str(path))
    return paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-root", type=Path, default=Path("results/03_temporal_refinement/cache/temporal_refinement_cache/large_v3_s2m2s512_fast"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=240)
    p.add_argument("--sequence-length", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--crop-height", type=int, default=384)
    p.add_argument("--crop-width", type=int, default=640)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--hidden-channels", type=int, default=128)
    p.add_argument("--residual-clamp-px", type=float, default=1.5)
    p.add_argument("--disp-norm", type=float, default=128.0)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--spatial-weight", type=float, default=0.25)
    p.add_argument("--sav-weight", type=float, default=0.30)
    p.add_argument("--motion-weight", type=float, default=0.30)
    p.add_argument("--raw-weight", type=float, default=0.10)
    p.add_argument("--res-weight", type=float, default=0.10)
    p.add_argument("--alpha-weight", type=float, default=0.02)
    p.add_argument("--alpha-prior", type=float, default=0.5)
    p.add_argument("--alpha-prior-epochs", type=int, default=5)
    p.add_argument("--edge-weight", type=float, default=0.04)
    p.add_argument("--reset-weight", type=float, default=0.03)
    p.add_argument("--reset-disp-threshold", type=float, default=0.05)
    p.add_argument("--reset-rgb-threshold", type=float, default=0.06)
    p.add_argument("--abort-patience", type=int, default=3)
    p.add_argument("--debug-output-names", action="store_true")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def list_sequences(cache_root: Path) -> list[str]:
    with (cache_root / "index.csv").open() as f:
        return sorted({r["sequence_id"] for r in csv.DictReader(f)})


def write_yaml(path: Path, args, split: dict, world: int, batch: int) -> None:
    lines = [
        f"cache_root: {args.cache_root}",
        f"out_dir: {args.out_dir}",
        "model: AdaptiveMotionFusionRefiner",
        "backbone: frozen_s2m2_s512_cache",
        "flow_source: proxy_no_learned_temporal_flow_found",
        f"ddp_world_size: {world}",
        f"batch_size_per_gpu: {batch}",
        f"effective_batch_size: {batch * world}",
        f"sequence_length: {args.sequence_length}",
        f"crop_size: [{args.crop_height}, {args.crop_width}]",
        f"epochs: {args.epochs}",
        f"base_channels: {args.base_channels}",
        f"hidden_channels: {args.hidden_channels}",
        f"residual_clamp_px: {args.residual_clamp_px}",
        "split:",
        f"  train: {split['train']}",
        f"  val: {split['val']}",
        f"  test_held_out: {split['test']}",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    ddp, rank, local_rank, world = ddp_setup()
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this overnight training")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()

    seqs = list_sequences(args.cache_root)
    split = {"test": [seqs[-1]], "val": [seqs[-2]], "train": seqs[:-2]}
    if args.smoke:
        args.epochs = 1
        args.eval_every = 1
        args.save_every = 1

    train_ds = AdaptiveFusionClipDataset(args.cache_root, split["train"], args.sequence_length, (args.crop_height, args.crop_width), True, args.disp_norm)
    val_ds = AdaptiveFusionClipDataset(args.cache_root, split["val"], args.sequence_length, (args.crop_height, args.crop_width), False, args.disp_norm)
    train_sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True) if ddp else None
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    model = AdaptiveMotionFusionRefiner(8, args.base_channels, args.hidden_channels, args.residual_clamp_px).to(device)
    if ddp:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs - args.warmup_epochs), eta_min=args.lr * 0.05)

    train_log_name = "debug_train_log.csv" if args.debug_output_names else "train_log.csv"
    val_log_name = "debug_validation.csv" if args.debug_output_names else "validation_summary.csv"
    report_name = "debug_report.md" if args.debug_output_names else "report.md"

    if is_main():
        args.out_dir.mkdir(parents=True, exist_ok=True)
        if not args.debug_output_names:
            (args.out_dir / "checkpoints").mkdir(exist_ok=True)
            write_yaml(args.out_dir / "config.yaml", args, split, world, args.batch_size)
    log_fields = [
        "epoch", "seconds", "lr", "train_loss", "train_loss_spatial", "train_loss_sav", "train_loss_motion",
        "train_loss_raw", "train_loss_res", "train_loss_alpha", "train_loss_edge", "train_loss_reset",
        "train_alpha_prior_scale", "train_reset_target_prevalence",
        "grad_norm_all", "grad_norm_enc", "grad_norm_gru", "grad_norm_dec", "grad_norm_head", "param_update_norm",
        "raw_to_spatial_mae", "fused_to_spatial_mae", "raw_to_sav_mae", "fused_to_sav_mae",
        "ema_to_spatial_mae", "ema_to_sav_mae",
        "raw_temporal_mae", "ema_temporal_mae", "fused_temporal_mae", "sav_temporal_mae",
        "ema_teacher_delta_mae", "teacher_delta_mae",
        "alpha_mean", "alpha_std", "alpha_min", "alpha_max",
        "alpha_hist_0_20", "alpha_hist_20_40", "alpha_hist_40_60", "alpha_hist_60_80", "alpha_hist_80_100",
        "reset_mean", "reset_std", "reset_min", "reset_max", "reset_target_prevalence",
        "residual_abs_mean", "peak_vram_mb", "abort_reason",
    ]
    if is_main():
        with (args.out_dir / train_log_name).open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writeheader()
        with (args.out_dir / val_log_name).open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=["epoch", *log_fields[22:-2]]).writeheader()
        print(f"split={split}", flush=True)
        print(f"ddp_world={world} batch_per_gpu={args.batch_size} effective_batch={args.batch_size * world}", flush=True)

    best_geo = best_temp = best_pareto = float("inf")
    start = time.time()
    last_val: dict[str, float] = {}
    previous_validation_signature: tuple[float, ...] | None = None
    worse_than_baselines_count = 0
    abort_reason = ""
    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        rows = []
        grad_rows = []
        before_params = {name: param.detach().float().cpu().clone() for name, param in (model.module if hasattr(model, "module") else model).named_parameters()}
        t0 = time.time()
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = forward_clip(model, to_device(batch, device), args.disp_norm)
                loss, parts = loss_for_outputs(out, args, epoch)
            if not torch.isfinite(loss):
                abort_reason = "nan_or_inf_loss"
                break
            loss.backward()
            grads = gradient_stats(model.module if hasattr(model, "module") else model)
            grad_rows.append(grads)
            if grads["grad_norm_all"] < 1e-10:
                abort_reason = "near_zero_gradient_norm"
                break
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            opt.step()
            rows.append(parts)
        if abort_reason:
            print(f"ABORT {abort_reason}", flush=True)
        if epoch > args.warmup_epochs:
            sched.step()
        seconds = time.time() - t0
        train = mean_rows(rows)
        grad_mean = mean_rows(grad_rows)
        update_norm = parameter_update_norm(before_params, model.module if hasattr(model, "module") else model)
        if update_norm < 1e-12 and not abort_reason:
            abort_reason = "near_zero_parameter_update_norm"
        if is_main() and (epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs):
            last_val = evaluate(model.module if hasattr(model, "module") else model, val_loader, device, args)
            sig = tuple(round(float(last_val[k]), 10) for k in ["fused_to_spatial_mae", "fused_to_sav_mae", "fused_temporal_mae", "teacher_delta_mae"])
            if previous_validation_signature == sig and epoch > 1 and not abort_reason:
                abort_reason = "validation_outputs_identical_after_real_eval"
            previous_validation_signature = sig
            adaptive_worse = (
                last_val["fused_to_spatial_mae"] > min(last_val["raw_to_spatial_mae"], last_val["ema_to_spatial_mae"])
                and last_val["fused_to_sav_mae"] > min(last_val["raw_to_sav_mae"], last_val["ema_to_sav_mae"])
                and last_val["teacher_delta_mae"] > last_val["ema_teacher_delta_mae"]
            )
            worse_than_baselines_count = worse_than_baselines_count + 1 if adaptive_worse else 0
            if worse_than_baselines_count >= args.abort_patience and not abort_reason:
                abort_reason = "adaptive_worse_than_raw_and_ema_for_3_validations"
            if (last_val["alpha_mean"] < 0.02 or last_val["alpha_mean"] > 0.98 or last_val["alpha_std"] < 1e-4) and not abort_reason:
                abort_reason = "alpha_collapse"
            if (last_val["reset_mean"] < 1e-4 or last_val["reset_mean"] > 0.9999 or last_val["reset_std"] < 1e-5) and not abort_reason:
                abort_reason = "reset_collapse"
            with (args.out_dir / val_log_name).open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["epoch", *log_fields[22:-2]], extrasaction="ignore")
                writer.writerow({"epoch": epoch, **last_val})
        if ddp:
            dist.barrier()
        if is_main():
            val = last_val
            peak = float(torch.cuda.max_memory_allocated() / 1024**2)
            ckpt = {
                "epoch": epoch,
                "model_state_dict": (model.module if hasattr(model, "module") else model).state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "args": vars(args),
                "split": split,
                "validation": val,
            }
            if not args.debug_output_names:
                torch.save(ckpt, args.out_dir / "checkpoints" / "latest.pt")
            geo = val.get("fused_to_spatial_mae", float("inf"))
            temp = val.get("teacher_delta_mae", float("inf"))
            pareto = geo + temp
            if geo < best_geo:
                best_geo = geo
                if not args.debug_output_names:
                    torch.save(ckpt, args.out_dir / "checkpoints" / "best_geometric.pt")
            if temp < best_temp:
                best_temp = temp
                if not args.debug_output_names:
                    torch.save(ckpt, args.out_dir / "checkpoints" / "best_temporal.pt")
            if pareto < best_pareto:
                best_pareto = pareto
                if not args.debug_output_names:
                    torch.save(ckpt, args.out_dir / "checkpoints" / "best_pareto.pt")
            if not args.debug_output_names and args.save_every and epoch % args.save_every == 0:
                torch.save(ckpt, args.out_dir / "checkpoints" / f"epoch_{epoch:04d}.pt")
            row = {
                "epoch": epoch,
                "seconds": seconds,
                "lr": opt.param_groups[0]["lr"],
                **{f"train_{k}": v for k, v in train.items()},
                **grad_mean,
                "param_update_norm": update_norm,
                **val,
                "peak_vram_mb": peak,
                "abort_reason": abort_reason,
            }
            with (args.out_dir / train_log_name).open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=log_fields, extrasaction="ignore")
                writer.writerow(row)
            print(
                f"epoch={epoch:03d} sec={seconds:.1f} "
                f"geo={geo:.4f} temp={temp:.4f} "
                f"fused_td={val.get('fused_temporal_mae', float('nan')):.4f} "
                f"alpha={val.get('alpha_mean', float('nan')):.3f}±{val.get('alpha_std', float('nan')):.3f} "
                f"grad={grad_mean.get('grad_norm_all', float('nan')):.3e} update={update_norm:.3e} "
                f"peak_vram={peak:.0f}MB abort={abort_reason}",
                flush=True,
            )
        if abort_reason:
            break
        if ddp:
            dist.barrier()

    if is_main():
        refs = save_reference_images(model.module if hasattr(model, "module") else model, val_ds, args.out_dir, device, args)
        summary = {
            "runtime_seconds": time.time() - start,
            "peak_vram_mb": float(torch.cuda.max_memory_allocated() / 1024**2),
            "best_checkpoints": {
                "geometric": "checkpoints/best_geometric.pt",
                "temporal": "checkpoints/best_temporal.pt",
                "pareto": "checkpoints/best_pareto.pt",
            },
            "last_validation": last_val,
            "reference_images": refs,
        }
        title = "Adaptive Motion-Aware Fusion Debug Run" if args.debug_output_names else "Adaptive Motion-Aware Fusion Overnight Run"
        summary["abort_reason"] = abort_reason
        (args.out_dir / report_name).write_text("# " + title + "\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n")
        print(json.dumps(summary, indent=2), flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
