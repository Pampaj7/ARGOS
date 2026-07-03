#!/usr/bin/env python3
"""Train safe gated tiny refiner on full S2M2->GT low-res targets."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_tiny_refiner_v1_full_gt import (
    DEFAULT_TARGETS_ROOT,
    DISP_SCALE,
    charbonnier,
    colorize,
    finite_mean,
    load_samples,
    load_shards,
    masked_mean,
    parse_bool,
    unwrap,
    write_csv,
)


DEFAULT_OUTPUT_ROOT = Path("results/03_temporal_refinement/training/tiny_refiner_v1_2_safe_gated")


class GatedDataset(Dataset):
    def __init__(self, samples, shards):
        self.samples = samples
        self.shards = shards

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        shard = self.shards[sample.target_path]
        raw = shard["raw_disp"][sample.offset].astype(np.float32)
        gt = shard["gt_disp"][sample.offset].astype(np.float32)
        valid = shard["valid_mask"][sample.offset].astype(np.float32)
        delta = shard["delta_disp_gt_minus_raw"][sample.offset].astype(np.float32)
        gx = np.zeros_like(raw, dtype=np.float32)
        gy = np.zeros_like(raw, dtype=np.float32)
        gx[:, 1:] = raw[:, 1:] - raw[:, :-1]
        gy[1:, :] = raw[1:, :] - raw[:-1, :]
        x = np.stack([raw / DISP_SCALE, valid, gx / DISP_SCALE, gy / DISP_SCALE], axis=0).astype(np.float32)
        return {
            "x": torch.from_numpy(x),
            "raw": torch.from_numpy(raw[None]),
            "gt": torch.from_numpy(gt[None]),
            "delta": torch.from_numpy(delta[None]),
            "valid": torch.from_numpy(valid[None]),
            "sequence_id": sample.sequence_id,
            "frame_id": sample.frame_id,
        }


class TinyRefinerV11Gated(nn.Module):
    def __init__(self, in_channels: int = 4, hidden: int = 48, gate_bias_init: float = -2.0):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_channels, hidden, 3, padding=1), nn.GroupNorm(8, hidden), nn.SiLU(inplace=True)]
        for _ in range(4):
            layers += [nn.Conv2d(hidden, hidden, 3, padding=1), nn.GroupNorm(8, hidden), nn.SiLU(inplace=True)]
        layers.append(nn.Conv2d(hidden, 2, 1))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias[1].fill_(gate_bias_init)

    def forward(self, x: torch.Tensor, residual_scale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.net(x)
        residual = residual_scale * torch.tanh(out[:, :1])
        logit = out[:, 1:2]
        gate = torch.sigmoid(logit)
        applied = gate * residual
        return applied, residual, gate, logit


def bad_rate(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor, tau: float) -> float:
    return float(masked_mean((torch.abs(pred - gt) > tau).float(), valid).detach().cpu() * 100.0)


def gate_auc(gate: torch.Tensor, good: torch.Tensor, bad: torch.Tensor) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        mask = ((good + bad) > 0).detach().cpu().numpy().reshape(-1)
        if not mask.any():
            return float("nan")
        y = bad.detach().cpu().numpy().reshape(-1)[mask]
        p = gate.detach().cpu().numpy().reshape(-1)[mask]
        if y.size > 200_000:
            idx = np.linspace(0, y.size - 1, 200_000, dtype=np.int64)
            y = y[idx]
            p = p[idx]
        return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    except Exception:
        return float("nan")


def batch_metrics(raw, refined, gt, applied, residual, gate, delta, valid, good_threshold_px: float, bad_threshold_px: float) -> dict[str, float]:
    raw_err = torch.abs(raw - gt)
    refined_err = torch.abs(refined - gt)
    good = valid * (raw_err < good_threshold_px).float()
    bad = valid * (raw_err >= bad_threshold_px).float()
    raw_mae = float(masked_mean(raw_err, valid).detach().cpu())
    refined_mae = float(masked_mean(refined_err, valid).detach().cpu())
    raw_bad3 = bad_rate(raw, gt, valid, 3.0)
    refined_bad3 = bad_rate(refined, gt, valid, 3.0)
    abs_applied = torch.abs(applied)
    return {
        "raw_disp_mae_vs_gt": raw_mae,
        "refined_disp_mae_vs_gt": refined_mae,
        "raw_bad_1px": bad_rate(raw, gt, valid, 1.0),
        "refined_bad_1px": bad_rate(refined, gt, valid, 1.0),
        "raw_bad_3px": raw_bad3,
        "refined_bad_3px": refined_bad3,
        "relative_mae_improvement": (raw_mae - refined_mae) / raw_mae if raw_mae > 1e-8 else float("nan"),
        "relative_bad3_improvement": (raw_bad3 - refined_bad3) / raw_bad3 if raw_bad3 > 1e-8 else float("nan"),
        "residual_l1": float(masked_mean(torch.abs(applied - delta), valid).detach().cpu()),
        "gate_mean": float(masked_mean(gate, valid).detach().cpu()),
        "gate_raw_good_mean": float(masked_mean(gate, good).detach().cpu()) if float(good.sum()) > 0 else float("nan"),
        "gate_raw_bad_mean": float(masked_mean(gate, bad).detach().cpu()) if float(bad.sum()) > 0 else float("nan"),
        "gate_auc": gate_auc(gate, good, bad),
        "correction_magnitude_mean": float(masked_mean(abs_applied, valid).detach().cpu()),
        "correction_magnitude_median": float(torch.median(abs_applied[valid > 0]).detach().cpu()) if bool((valid > 0).any()) else float("nan"),
        "preserve_error_raw_good": float(masked_mean(torch.abs(refined - raw), good).detach().cpu()) if float(good.sum()) > 0 else float("nan"),
    }


def loss_batch(model, batch, args, device) -> tuple[torch.Tensor, dict[str, float]]:
    x = batch["x"].to(device)
    raw = batch["raw"].to(device)
    gt = batch["gt"].to(device)
    delta = batch["delta"].to(device)
    valid = batch["valid"].to(device)
    applied, residual, gate, logit = model(x, args.residual_scale)
    refined = raw + applied
    raw_err = torch.abs(raw - gt)
    good = valid * (raw_err < args.good_threshold_px).float()
    bad = valid * (raw_err >= args.bad_threshold_px).float()
    primary = masked_mean(charbonnier(refined - gt), valid)
    residual_target = masked_mean(charbonnier(applied - delta), valid)
    gate_l1 = masked_mean(gate, valid)
    preserve = masked_mean(torch.abs(applied), good) if float(good.sum()) > 0 else primary.new_tensor(0.0)
    bad_loss = masked_mean(charbonnier(refined - gt), bad) if float(bad.sum()) > 0 else primary.new_tensor(0.0)
    gate_mask = (good + bad).clamp(0, 1)
    gate_target = bad
    gate_bce = masked_mean(nn.functional.binary_cross_entropy_with_logits(logit, gate_target, reduction="none"), gate_mask)
    loss = (
        args.main_weight * primary
        + args.residual_weight * residual_target
        + args.gate_bce_weight * gate_bce
        + args.gate_l1_weight * gate_l1
        + args.preserve_weight * preserve
        + 0.25 * bad_loss
    )
    metrics = {
        "loss": float(loss.detach().cpu()),
        "primary_loss": float(primary.detach().cpu()),
        "residual_target_loss": float(residual_target.detach().cpu()),
        "gate_l1": float(gate_l1.detach().cpu()),
        "gate_bce": float(gate_bce.detach().cpu()),
        "good_preserve_loss": float(preserve.detach().cpu()),
        "bad_loss": float(bad_loss.detach().cpu()),
    }
    return loss, metrics


def train_one_epoch(model, loader, optimizer, args, device) -> dict[str, float]:
    model.train()
    rows: list[dict[str, float]] = []
    for batch in loader:
        loss, metrics = loss_batch(model, batch, args, device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        rows.append(metrics)
    return {k: finite_mean([r[k] for r in rows]) for k in rows[0]}


@torch.no_grad()
def evaluate(model, loader, args, device, split: str) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    rows: list[dict[str, float]] = []
    seq_rows: dict[str, list[dict[str, float]]] = {}
    for batch in loader:
        x = batch["x"].to(device)
        raw = batch["raw"].to(device)
        gt = batch["gt"].to(device)
        delta = batch["delta"].to(device)
        valid = batch["valid"].to(device)
        applied, residual, gate, _logit = model(x, args.residual_scale)
        refined = raw + applied
        metrics = batch_metrics(raw, refined, gt, applied, residual, gate, delta, valid, args.good_threshold_px, args.bad_threshold_px)
        rows.append(metrics)
        for seq in set(batch["sequence_id"]):
            idx = torch.tensor([s == seq for s in batch["sequence_id"]], device=device, dtype=torch.bool)
            sm = batch_metrics(raw[idx], refined[idx], gt[idx], applied[idx], residual[idx], gate[idx], delta[idx], valid[idx], args.good_threshold_px, args.bad_threshold_px)
            seq_rows.setdefault(str(seq), []).append(sm)
    out = {"split": split, "frames": len(loader.dataset)}
    for key in rows[0]:
        out[key] = finite_mean([r[key] for r in rows])
    seq_out = [
        {"split": split, "sequence_id": seq, "frames": sum(1 for s in loader.dataset.samples if s.sequence_id == seq), **{k: finite_mean([r[k] for r in vals]) for k in vals[0]}}
        for seq, vals in sorted(seq_rows.items())
    ]
    return out, seq_out


@torch.no_grad()
def write_diagnostics(model, dataset: GatedDataset, out_dir: Path, args, device, prefix: str, count: int = 4) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for idx in np.linspace(0, max(0, len(dataset) - 1), min(count, len(dataset)), dtype=int):
        item = dataset[int(idx)]
        applied, residual, gate, _logit = model(item["x"][None].to(device), args.residual_scale)
        raw = item["raw"][0].numpy()
        gt = item["gt"][0].numpy()
        valid = item["valid"][0].numpy()
        applied_np = applied[0, 0].cpu().numpy()
        residual_np = residual[0, 0].cpu().numpy()
        gate_np = gate[0, 0].cpu().numpy()
        refined = raw + applied_np
        raw_err = np.abs(raw - gt)
        refined_err = np.abs(refined - gt)
        vmax = float(np.percentile(gt[valid > 0], 98)) if np.any(valid > 0) else 64.0
        tiles = [
            colorize(raw, vmax),
            colorize(refined, vmax),
            colorize(gt, vmax),
            colorize(raw_err, 16.0, cv2.COLORMAP_MAGMA),
            colorize(refined_err, 16.0, cv2.COLORMAP_MAGMA),
            colorize(np.clip(raw_err - refined_err, -8.0, 8.0), 8.0, cv2.COLORMAP_TURBO),
            colorize(np.abs(residual_np), args.residual_scale, cv2.COLORMAP_MAGMA),
            colorize(np.abs(applied_np), args.residual_scale, cv2.COLORMAP_MAGMA),
            colorize(gate_np, 1.0, cv2.COLORMAP_VIRIDIS),
            colorize(valid, 1.0, cv2.COLORMAP_VIRIDIS),
        ]
        cv2.imwrite(str(out_dir / f"{prefix}_{item['sequence_id']}_{item['frame_id']}.png"), np.concatenate(tiles, axis=1))


def make_loader(ds, args, shuffle: bool) -> DataLoader:
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets-root", type=Path, default=DEFAULT_TARGETS_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--main-weight", type=float, default=1.0)
    p.add_argument("--residual-weight", type=float, default=0.25)
    p.add_argument("--gate-bce-weight", type=float, default=0.5)
    p.add_argument("--preserve-weight", type=float, default=0.5)
    p.add_argument("--gate-l1-weight", type=float, default=0.05)
    p.add_argument("--good-threshold-px", type=float, default=1.0)
    p.add_argument("--bad-threshold-px", type=float, default=3.0)
    p.add_argument("--residual-scale", type=float, default=3.0)
    p.add_argument("--gate-bias-init", type=float, default=-4.0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--overwrite", nargs="?", const=True, default=False, type=parse_bool)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(args.output_root)
        shutil.rmtree(args.output_root)
    (args.output_root / "checkpoints").mkdir(parents=True)
    (args.output_root / "diagnostics").mkdir()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    splits, by_split = load_samples(args.targets_root, args.max_frames)
    all_samples = by_split["train"] + by_split["val"] + by_split["test"]
    shards = load_shards(all_samples)
    datasets = {split: GatedDataset(samples, shards) for split, samples in by_split.items()}
    loaders = {split: make_loader(ds, args, split == "train") for split, ds in datasets.items()}
    model = TinyRefinerV11Gated(gate_bias_init=args.gate_bias_init).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best = float("inf")
    best_epoch = 0
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    run_log = [
        f"device={device}",
        f"cuda_device_count={torch.cuda.device_count() if device.type == 'cuda' else 0}",
        f"params={params}",
        f"splits={json.dumps(splits)}",
        f"frames={{'train': {len(datasets['train'])}, 'val': {len(datasets['val'])}, 'test': {len(datasets['test'])}}}",
        f"shards_loaded={len(shards)}",
        f"residual_scale={args.residual_scale} gate_bias_init={args.gate_bias_init}",
    ]
    start = time.perf_counter()
    (args.output_root / "run.log").write_text("\n".join(run_log) + "\n")
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, loaders["train"], optimizer, args, device)
        val_metrics, _seq = evaluate(model, loaders["val"], args, device, "val")
        train_rows.append({"epoch": epoch, **train_metrics})
        val_rows.append({"epoch": epoch, **val_metrics})
        score = val_metrics["refined_disp_mae_vs_gt"]
        if score < best:
            best = score
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": unwrap(model).state_dict(),
                    "args": vars(args),
                    "splits": splits,
                    "input_channels": 4,
                    "parameter_count": params,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "score": score,
                },
                args.output_root / "checkpoints" / "best.pt",
            )
        run_log.append(
            f"epoch={epoch} train_loss={train_metrics['loss']:.6f} val_raw={val_metrics['raw_disp_mae_vs_gt']:.6f} "
            f"val_refined={val_metrics['refined_disp_mae_vs_gt']:.6f} val_bad3={val_metrics['raw_bad_3px']:.3f}->{val_metrics['refined_bad_3px']:.3f} "
            f"gate={val_metrics['gate_mean']:.4f}/{val_metrics['gate_raw_good_mean']:.4f}/{val_metrics['gate_raw_bad_mean']:.4f} auc={val_metrics['gate_auc']:.4f}"
        )
        (args.output_root / "run.log").write_text("\n".join(run_log) + "\n")
        if epoch - best_epoch >= args.patience:
            run_log.append(f"early_stop epoch={epoch} best_epoch={best_epoch} patience={args.patience}")
            (args.output_root / "run.log").write_text("\n".join(run_log) + "\n")
            break

    ckpt = torch.load(args.output_root / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    unwrap(model).load_state_dict(ckpt["model_state_dict"])
    train_metrics_eval, train_seq = evaluate(model, loaders["train"], args, device, "train")
    val_metrics, val_seq = evaluate(model, loaders["val"], args, device, "val")
    test_metrics, test_seq = evaluate(model, loaders["test"], args, device, "test")
    write_csv(args.output_root / "train_log.csv", train_rows)
    write_csv(args.output_root / "val_metrics.csv", val_rows)
    write_csv(args.output_root / "test_metrics.csv", [test_metrics])
    write_csv(args.output_root / "sequence_metrics.csv", train_seq + val_seq + test_seq)
    write_diagnostics(model, datasets["val"], args.output_root / "diagnostics", args, device, "val")
    write_diagnostics(model, datasets["test"], args.output_root / "diagnostics", args, device, "test")
    summary = {
        "targets_root": str(args.targets_root),
        "output_root": str(args.output_root),
        "params": params,
        "best_epoch": ckpt["epoch"],
        "elapsed_seconds": time.perf_counter() - start,
        "frames": {k: len(v) for k, v in datasets.items()},
        "splits": splits,
        "train": train_metrics_eval,
        "val": val_metrics,
        "test": test_metrics,
        "no_teacher_inference": True,
    }
    (args.output_root / "aggregate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_root / "README.md").write_text(
        f"""# Tiny Refiner v1.2 Safe Gated

Conservative gated residual refiner trained on full S2M2->GT low-resolution targets from `{args.targets_root}`.
No S2M2, SAV, RAFT, DINO, oracle, or teacher inference was run during training.

- Parameters: `{params}`
- Best epoch: `{ckpt['epoch']}`
- Split by sequence: `{splits}`
- Frames: `{summary['frames']}`
- Residual scale: `{args.residual_scale}`
- Gate bias init: `{args.gate_bias_init}`
- Test raw MAE: `{test_metrics['raw_disp_mae_vs_gt']:.6f}`
- Test refined MAE: `{test_metrics['refined_disp_mae_vs_gt']:.6f}`
- Test raw/refined bad-3px: `{test_metrics['raw_bad_3px']:.6f}` -> `{test_metrics['refined_bad_3px']:.6f}`
- Test gate mean good/bad: `{test_metrics['gate_raw_good_mean']:.6f}` / `{test_metrics['gate_raw_bad_mean']:.6f}`
"""
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
