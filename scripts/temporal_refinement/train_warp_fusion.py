#!/usr/bin/env python3
"""
Causal Warped Fusion Refiner - Training Script

Architecture:
- Frozen RAFT optical flow
- CausalWarpedFusionRefiner (ConvGRU at 1/8 resolution)
- Motion-compensated loss

Validation runs progressively over full sequences without resets.
"""

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

from scripts.temporal_refinement.lib.flow import FrozenRAFT, flow_confidence, warp_disp
from scripts.temporal_refinement.lib.training import colorize
from scripts.temporal_refinement.lib.warp_fusion_losses import warp_fusion_loss
from scripts.temporal_refinement.lib.warp_fusion_model import CausalWarpedFusionRefiner


# ──────────────────────────────────────────────────────────────────────────────
# Datasets
# ──────────────────────────────────────────────────────────────────────────────

def _load_disp(cache_root: Path, rel: str, y: int, x: int, h: int, w: int) -> np.ndarray:
    arr = np.load(cache_root / rel, mmap_mode="r")
    return np.asarray(arr[y : y + h, x : x + w], dtype=np.float32)


class AdaptiveFusionClipDataset(Dataset):
    """Training dataset: extracts short overlapping clips and applies random crops."""
    def __init__(
        self,
        cache_root: Path,
        sequence_ids: list[str],
        sequence_length: int,
        crop_size: tuple[int, int],
        random_crop: bool,
    ):
        self.cache_root = cache_root
        self.sequence_length = sequence_length
        self.crop_size = crop_size
        self.random_crop = random_crop
        
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
            rgb = cv2.cvtColor(rgb[y : y + ch, x : x + cw], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rgbs.append(torch.from_numpy(rgb).permute(2, 0, 1))
            raw.append(torch.from_numpy(_load_disp(self.cache_root, row["s2m2_s512_t_path"], y, x, ch, cw)).unsqueeze(0))
            spatial.append(torch.from_numpy(_load_disp(self.cache_root, row["s2m2_l736_t_path"], y, x, ch, cw)).unsqueeze(0))
            sav.append(torch.from_numpy(_load_disp(self.cache_root, row["sav_t_path"], y, x, ch, cw)).unsqueeze(0))

        return {
            "rgb": torch.stack(rgbs).float(),
            "raw": torch.stack(raw).float(),
            "spatial": torch.stack(spatial).float(),
            "sav": torch.stack(sav).float(),
            "sequence_id": rows[0]["sequence_id"],
        }


class ProgressiveSequenceDataset(Dataset):
    """Validation dataset: yields entire sequences without cropping. BPTT is ignored."""
    def __init__(self, cache_root: Path, sequence_ids: list[str]):
        self.cache_root = cache_root
        with (cache_root / "index.csv").open() as f:
            rows = list(csv.DictReader(f))
        
        wanted = set(sequence_ids)
        by_seq: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            if row["sequence_id"] in wanted:
                by_seq.setdefault(row["sequence_id"], []).append(row)
                
        self.sequences = []
        for _seq, seq_rows in sorted(by_seq.items()):
            self.sequences.append(sorted(seq_rows, key=lambda r: int(r["center_frame_id"])))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        rows = self.sequences[idx]
        h0, w0 = int(rows[0]["height"]), int(rows[0]["width"])
        
        rgbs, raw, spatial, sav = [], [], [], []
        for row in rows:
            rgb = cv2.imread(str(self.cache_root / row["rgb_center_path"]), cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rgbs.append(torch.from_numpy(rgb).permute(2, 0, 1))
            raw.append(torch.from_numpy(_load_disp(self.cache_root, row["s2m2_s512_t_path"], 0, 0, h0, w0)).unsqueeze(0))
            spatial.append(torch.from_numpy(_load_disp(self.cache_root, row["s2m2_l736_t_path"], 0, 0, h0, w0)).unsqueeze(0))
            sav.append(torch.from_numpy(_load_disp(self.cache_root, row["sav_t_path"], 0, 0, h0, w0)).unsqueeze(0))

        return {
            "rgb": torch.stack(rgbs).float(),
            "raw": torch.stack(raw).float(),
            "spatial": torch.stack(spatial).float(),
            "sav": torch.stack(sav).float(),
            "sequence_id": rows[0]["sequence_id"],
        }


# ──────────────────────────────────────────────────────────────────────────────
# Forward Sequence Driver
# ──────────────────────────────────────────────────────────────────────────────

def forward_sequence(
    refiner: nn.Module,
    flow_model: FrozenRAFT,
    batch: dict[str, torch.Tensor],
    disp_norm: float = 128.0,
    bptt_steps: int = 0,
) -> dict[str, torch.Tensor]:
    """Process an entire clip/sequence frame-by-frame."""
    rgb = batch["rgb"]
    raw = batch["raw"]
    
    B, T, _, H, W = rgb.shape
    device = rgb.device
    
    hidden = None
    
    fused_list, alpha_list, reset_list, residual_list = [], [], [], []
    flow_list, conf_list, occ_list, warped_prev_list = [], [], [], []
    
    prev_fused = raw[:, 0]
    prev_rgb = rgb[:, 0]
    
    for t in range(T):
        rgb_t = rgb[:, t]
        raw_t = raw[:, t]
        
        if t == 0:
            flow_fwd = torch.zeros(B, 2, H, W, device=device)
            conf = torch.ones(B, 1, H, W, device=device)
            occ = torch.zeros(B, 1, H, W, device=device)
            warped_prev = raw_t
        else:
            with torch.no_grad():
                # Extract RAFT flow
                flow_fwd = flow_model(prev_rgb, rgb_t)
                flow_bwd = flow_model(rgb_t, prev_rgb)
                conf, occ = flow_confidence(flow_fwd, flow_bwd)
                
            warped_prev = warp_disp(prev_fused, flow_fwd)
            
        raw_diff = torch.abs(raw_t - warped_prev)
        flow_mag = torch.linalg.vector_norm(flow_fwd, dim=1, keepdim=True) / 20.0
        
        x = torch.cat([
            rgb_t,
            raw_t / disp_norm,
            warped_prev / disp_norm,
            raw_diff / disp_norm,
            torch.clamp(flow_mag, 0.0, 1.0),
            conf,
            occ,
        ], dim=1)
        
        fused_t, alpha_t, reset_t, residual_t, hidden = refiner(
            x, raw_t, warped_prev, hidden
        )
        
        fused_list.append(fused_t)
        alpha_list.append(alpha_t)
        reset_list.append(reset_t)
        residual_list.append(residual_t)
        flow_list.append(flow_fwd)
        conf_list.append(conf)
        occ_list.append(occ)
        warped_prev_list.append(warped_prev)
        
        prev_fused = fused_t
        prev_rgb = rgb_t
        
        # Optional Truncated BPTT
        if bptt_steps > 0 and (t + 1) % bptt_steps == 0 and hidden is not None:
            hidden = hidden.detach()
            prev_fused = prev_fused.detach()

    return {
        "fused": torch.stack(fused_list, dim=1),
        "alpha": torch.stack(alpha_list, dim=1),
        "reset": torch.stack(reset_list, dim=1),
        "residual": torch.stack(residual_list, dim=1),
        "flow": torch.stack(flow_list, dim=1),
        "confidence": torch.stack(conf_list, dim=1),
        "occlusion": torch.stack(occ_list, dim=1),
        "warped_prev": torch.stack(warped_prev_list, dim=1),
        "raw": raw,
        "spatial": batch["spatial"],
        "sav": batch["sav"],
        "rgb": rgb,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation & Utils
# ──────────────────────────────────────────────────────────────────────────────

def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}

def mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({k for r in rows for k in r})
    return {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}

def fixed_ema(raw: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    filtered = [raw[:, 0]]
    prev = raw[:, 0]
    for t in range(1, raw.shape[1]):
        prev = alpha * raw[:, t] + (1.0 - alpha) * prev
        filtered.append(prev)
    return torch.stack(filtered, dim=1)

@torch.no_grad()
def evaluate_sequences(refiner, flow_model, loader, device, args) -> dict[str, float]:
    refiner.eval()
    flow_model.eval()
    rows = []
    
    for batch in loader:
        b = to_device(batch, device)
        out = forward_sequence(refiner, flow_model, b, args.disp_norm, bptt_steps=0)
        
        fused, raw, spatial, sav = out["fused"], out["raw"], out["spatial"], out["sav"]
        ema = fixed_ema(raw, alpha=0.50)
        
        row = {
            "raw_to_spatial_mae": float(torch.mean(torch.abs(raw - spatial)).cpu()),
            "ema_to_spatial_mae": float(torch.mean(torch.abs(ema - spatial)).cpu()),
            "fused_to_spatial_mae": float(torch.mean(torch.abs(fused - spatial)).cpu()),
            
            "raw_to_sav_mae": float(torch.mean(torch.abs(raw - sav)).cpu()),
            "ema_to_sav_mae": float(torch.mean(torch.abs(ema - sav)).cpu()),
            "fused_to_sav_mae": float(torch.mean(torch.abs(fused - sav)).cpu()),
            
            "alpha_mean": float(out["alpha"].mean().cpu()),
            "alpha_std": float(out["alpha"].std().cpu()),
            "alpha_p05": float(np.percentile(out["alpha"].cpu().numpy(), 5)),
            "alpha_p50": float(torch.median(out["alpha"]).cpu()),
            "alpha_p95": float(np.percentile(out["alpha"].cpu().numpy(), 95)),
            
            "reset_mean": float(out["reset"].mean().cpu()),
            "residual_abs_mean": float(torch.mean(torch.abs(out["residual"])).cpu()),
        }
        
        if fused.shape[1] > 1:
            # Temporal consistency metrics
            flow_flat = out["flow"][:, 1:].reshape(-1, 2, raw.shape[-2], raw.shape[-1])
            
            def warped_temporal_mae(disp: torch.Tensor) -> float:
                prev = disp[:, :-1].reshape(-1, 1, raw.shape[-2], raw.shape[-1])
                curr = disp[:, 1:].reshape(-1, 1, raw.shape[-2], raw.shape[-1])
                w_prev = warp_disp(prev, flow_flat)
                return float(torch.mean(torch.abs(curr - w_prev)).cpu())

            row.update({
                "raw_temporal_mae": warped_temporal_mae(raw),
                "ema_temporal_mae": warped_temporal_mae(ema),
                "fused_temporal_mae": warped_temporal_mae(fused),
                "sav_temporal_mae": warped_temporal_mae(sav),
            })
            
        rows.append(row)
        
    return mean_rows(rows)


@torch.no_grad()
def save_reference_images(refiner, flow_model, dataset, out_dir, device, args, epoch: int) -> None:
    refiner.eval()
    flow_model.eval()
    ref_dir = out_dir / "reference_images"
    ref_dir.mkdir(parents=True, exist_ok=True)
    
    # Run through the first sequence to find the target frames
    batch = dataset[0]
    b = to_device({k: v.unsqueeze(0) if torch.is_tensor(v) else v for k, v in batch.items()}, device)
    out = forward_sequence(refiner, flow_model, b, args.disp_norm, bptt_steps=0)
    
    T = out["fused"].shape[1]
    if T < 2: return
    
    flow_mags = [float(torch.mean(torch.linalg.vector_norm(out["flow"][0, t], dim=0))) for t in range(1, T)]
    occ_means = [float(torch.mean(out["occlusion"][0, t])) for t in range(1, T)]
    
    t_normal = 1 + int(np.argmin(np.abs(np.array(flow_mags) - np.median(flow_mags))))
    t_strong = 1 + int(np.argmax(flow_mags))
    t_occ = 1 + int(np.argmax(occ_means))
    
    ema = fixed_ema(out["raw"], alpha=0.50)
    
    for cond, t in [("normal", t_normal), ("strong_motion", t_strong), ("occlusion", t_occ)]:
        rgb = (out["rgb"][0, t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        raw = out["raw"][0, t, 0].cpu().numpy()
        ema_t = ema[0, t, 0].cpu().numpy()
        fused = out["fused"][0, t, 0].cpu().numpy()
        warped = out["warped_prev"][0, t, 0].cpu().numpy()
        sav = out["sav"][0, t, 0].cpu().numpy()
        alpha = out["alpha"][0, t, 0].cpu().numpy()
        reset = out["reset"][0, t, 0].cpu().numpy()
        occ = out["occlusion"][0, t, 0].cpu().numpy()
        
        # Merge reset and occlusion visually
        reset_occ = np.maximum(reset, occ)
        
        w_prev = warp_disp(out["fused"][:, t-1], out["flow"][:, t])
        td = np.abs(out["fused"][0, t, 0].cpu().numpy() - w_prev[0, 0].cpu().numpy())
        geo_err = np.abs(fused - sav)
        
        vmax = float(np.nanpercentile(raw, 99))
        
        tiles = [
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            colorize(raw, vmax),
            colorize(ema_t, vmax),
            colorize(warped, vmax),
            colorize(fused, vmax),
            colorize(sav, vmax),
            colorize(alpha, 1.0, cv2.COLORMAP_VIRIDIS),
            colorize(reset_occ, 1.0, cv2.COLORMAP_MAGMA),
            colorize(geo_err, 8.0, cv2.COLORMAP_TURBO),
            colorize(td, 8.0, cv2.COLORMAP_MAGMA),
        ]
        labels = ["RGB", "raw S2M2-S", "EMA0.50", "warped prev", "adaptive", "SAV GT", "alpha", "reset/occ", "geom error", "temp err"]
        
        small = []
        for tile, label in zip(tiles, labels):
            tile = cv2.resize(tile, (200, 140), interpolation=cv2.INTER_AREA)
            cv2.putText(tile, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            small.append(tile)
            
        path = ref_dir / f"epoch_{epoch:02d}_{cond}.png"
        cv2.imwrite(str(path), np.concatenate(small, axis=1))


# ──────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-root", type=Path, default=Path("results/03_temporal_refinement/cache/temporal_refinement_cache/large_v3_s2m2s512_fast"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--sequence-length", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--crop-height", type=int, default=384)
    p.add_argument("--crop-width", type=int, default=640)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--bptt-steps", type=int, default=8)
    p.add_argument("--residual-clamp-px", type=float, default=1.5)
    p.add_argument("--disp-norm", type=float, default=128.0)
    p.add_argument("--raft-checkpoint", type=str, default="external/frame_stereo_repos/RAFT/models/raft-things.pth")
    p.add_argument("--sav-weight", type=float, default=0.40)
    p.add_argument("--spatial-weight", type=float, default=0.15)
    p.add_argument("--raw-fidelity-weight", type=float, default=0.10)
    p.add_argument("--motion-comp-weight", type=float, default=0.25)
    p.add_argument("--residual-l1-weight", type=float, default=0.08)
    p.add_argument("--edge-weight", type=float, default=0.04)
    p.add_argument("--alpha-prior-weight", type=float, default=0.02)
    p.add_argument("--alpha-prior-decay-epochs", type=int, default=20)
    args = p.parse_args()

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    ddp = world > 1
    
    if ddp:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")

    device = torch.device(f"cuda:{local_rank}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Models
    raft_path = Path(args.raft_checkpoint)
    flow_model = FrozenRAFT(checkpoint=raft_path if raft_path.exists() else None).to(device)
    refiner = CausalWarpedFusionRefiner(residual_clamp_px=args.residual_clamp_px).to(device)
    
    if ddp:
        refiner = DistributedDataParallel(refiner, device_ids=[local_rank], output_device=local_rank)

    opt = torch.optim.AdamW(refiner.parameters(), lr=args.lr, weight_decay=1e-4)

    start_epoch = 1
    checkpoint_path = args.out_dir / "checkpoint_latest.pth"
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if ddp:
            refiner.module.load_state_dict(ckpt["model"])
        else:
            refiner.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        if rank == 0:
            print(f"Resumed from epoch {ckpt['epoch']}")

    # Datasets
    with (args.cache_root / "index.csv").open() as f:
        seqs = sorted({r["sequence_id"] for r in csv.DictReader(f)})
        
    train_seqs = [s for s in seqs if "test_dataset_8" in s]
    val_seqs = [s for s in seqs if "test_dataset_9" in s]
    train_ds = AdaptiveFusionClipDataset(args.cache_root, train_seqs, args.sequence_length, (args.crop_height, args.crop_width), random_crop=True)
    val_ds = ProgressiveSequenceDataset(args.cache_root, val_seqs)
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
        sampler=DistributedSampler(train_ds) if ddp else None, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0, shuffle=False)

    if rank == 0:
        print(f"Starting debug training on {len(train_seqs)} train seqs, {len(val_seqs)} val seqs.")

    log_fields = [
        "epoch", "seconds", "loss", "loss_sav", "loss_spatial", "loss_raw", "loss_motion", 
        "loss_residual", "loss_alpha_prior", "loss_reset", "loss_anti_collapse",
        "alpha_mean", "alpha_std", "alpha_p05", "alpha_p50", "alpha_p95", 
        "reset_mean", "residual_abs_mean", "peak_vram_mb"
    ]
    
    if rank == 0 and start_epoch == 1:
        with (args.out_dir / "debug_train_log.csv").open("w") as f:
            csv.DictWriter(f, fieldnames=log_fields).writeheader()

    history = []
    for epoch in range(start_epoch, args.epochs + 1):
        if ddp: train_loader.sampler.set_epoch(epoch)
        refiner.train()
        torch.cuda.reset_peak_memory_stats()
        
        t0 = time.time()
        epoch_rows = []
        
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            b = to_device(batch, device)
            
            # Forward with Truncated BPTT + Autocast (BF16)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = forward_sequence(refiner, flow_model, b, args.disp_norm, bptt_steps=args.bptt_steps)
            
            loss, metrics = warp_fusion_loss(
                out, epoch,
                sav_weight=args.sav_weight,
                spatial_weight=args.spatial_weight,
                raw_fidelity_weight=args.raw_fidelity_weight,
                motion_comp_weight=args.motion_comp_weight,
                residual_l1_weight=args.residual_l1_weight,
                edge_weight=args.edge_weight,
                alpha_prior_weight=args.alpha_prior_weight,
                alpha_prior_decay_epochs=args.alpha_prior_decay_epochs
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
            opt.step()
            epoch_rows.append(metrics)

        seconds = time.time() - t0
        train_metrics = mean_rows(epoch_rows)
        peak = float(torch.cuda.max_memory_allocated() / 1024**2)
        
        abort = False
        # Validation
        if rank == 0:
            val_metrics = evaluate_sequences(refiner.module if ddp else refiner, flow_model, val_loader, device, args)
            
            row = {
                "epoch": epoch, "seconds": seconds, "peak_vram_mb": peak,
                **{k: train_metrics.get(k, 0.0) for k in log_fields if k in train_metrics}
            }
            with (args.out_dir / "debug_train_log.csv").open("a") as f:
                csv.DictWriter(f, fieldnames=log_fields, extrasaction="ignore").writerow(row)
                
            val_keys = list(val_metrics.keys())
            if epoch == 1:
                with (args.out_dir / "debug_validation.csv").open("w") as f:
                    csv.DictWriter(f, fieldnames=["epoch"] + val_keys).writeheader()
            with (args.out_dir / "debug_validation.csv").open("a") as f:
                csv.DictWriter(f, fieldnames=["epoch"] + val_keys, extrasaction="ignore").writerow({"epoch": epoch, **val_metrics})

            print(f"epoch={epoch:02d} sec={seconds:.1f} loss={train_metrics['loss']:.4f} "
                  f"alpha={val_metrics['alpha_mean']:.3f} "
                  f"val_geo={val_metrics['fused_to_sav_mae']:.4f} "
                  f"val_temp={val_metrics.get('fused_temporal_mae', 0.0):.4f} "
                  f"peak_vram={peak:.0f}MB", flush=True)

            save_reference_images(refiner.module if ddp else refiner, flow_model, val_ds, args.out_dir, device, args, epoch)
            
            # Check warnings
            m = val_metrics
            history.append(m)
            
            if m["alpha_mean"] < 0.05:
                print("WARNING: alpha mean fell below 0.05 (potential collapse).")
                    
            if epoch > 1 and m.get("fused_temporal_mae", 0) >= m.get("ema_temporal_mae", 100):
                print("WARNING: motion-compensated temporal error worse than EMA.")
                
            torch.save({
                "model": (refiner.module if ddp else refiner).state_dict(),
                "optimizer": opt.state_dict(),
                "epoch": epoch
            }, checkpoint_path)

    if ddp:
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
