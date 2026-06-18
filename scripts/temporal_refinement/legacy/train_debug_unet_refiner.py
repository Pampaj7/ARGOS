#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.temporal_refinement.lib.datasets import TemporalRefinementCacheDataset, split_sample_ids
from scripts.temporal_refinement.lib.losses import refiner_loss
from scripts.temporal_refinement.lib.metrics import gt_metrics, teacher_metrics
from scripts.temporal_refinement.lib.models import TinyUNetRefiner


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def colorize(x: np.ndarray, vmax=None, cmap=cv2.COLORMAP_TURBO) -> np.ndarray:
    x = x.astype(np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros((*x.shape, 3), dtype=np.uint8)
    if vmax is None:
        lo = float(np.nanpercentile(x[finite], 1))
        hi = float(np.nanpercentile(x[finite], 99))
    else:
        lo, hi = 0.0, float(vmax)
    if hi <= lo:
        return np.zeros((*x.shape, 3), dtype=np.uint8)
    y = (np.clip((x - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(y, cmap)


def save_config(args, train_ids, val_ids):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    text = f"""cache_root: {args.cache_root}
output_root: {args.out_dir}
model: TinyUNetRefiner
input_channels: 8
base_channels: {args.base_channels}
residual_clamp_px: {args.residual_clamp_px}
crop_size: [{args.crop_h}, {args.crop_w}]
epochs: {args.epochs}
batch_size: {args.batch_size}
learning_rate: {args.lr}
optimizer: AdamW
loss:
  teacher_smooth_l1: {args.teacher_weight}
  temporal_window_median: {args.temporal_weight}
  residual_l1: {args.residual_l1_weight}
  edge_aware_smoothness: {args.smoothness_weight}
train_ids: {train_ids}
val_ids: {val_ids}
"""
    (args.out_dir / "config.yaml").write_text(text)


def run_model(model, batch, device):
    x = batch["input"].to(device)
    s2m2 = batch["s2m2_center"].to(device)
    teacher = batch["teacher"].to(device)
    delta = model(x)
    refined = torch.clamp(s2m2 + delta, min=0.0)
    return x, s2m2, teacher, delta, refined


def train_one_epoch(model, loader, optimizer, scaler, device, use_amp, args):
    model.train()
    totals = []
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            x, _s2m2, teacher, delta, refined = run_model(model, batch, device)
            loss, parts = refiner_loss(
                refined,
                teacher,
                delta,
                x[:, :3],
                disp_window=x[:, 3:],
                teacher_weight=args.teacher_weight,
                temporal_weight=args.temporal_weight,
                residual_l1_weight=args.residual_l1_weight,
                smoothness_weight=args.smoothness_weight,
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        parts["loss"] = float(loss.detach().cpu())
        totals.append(parts)
    return mean_dict(totals)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    rows = []
    for batch in loader:
        x, s2m2, teacher, delta, refined = run_model(model, batch, device)
        row = teacher_metrics(s2m2, refined, teacher, delta)
        if "gt_disp" in batch:
            row.update(
                gt_metrics(
                    refined,
                    batch["gt_disp"].to(device),
                    batch["gt_depth"].to(device),
                    batch["valid_mask"].to(device),
                    batch["fx"].to(device),
                    batch["baseline_mm"].to(device),
                )
            )
        rows.append(row)
    return mean_dict(rows)


@torch.no_grad()
def temporal_consistency_metrics(model, dataset, device, source_sequence="consecutive32"):
    model.eval()
    frames = []
    for idx in range(len(dataset)):
        batch = dataset[idx]
        if batch["source_sequence"] != source_sequence:
            continue
        collated = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                collated[key] = value.unsqueeze(0)
            else:
                collated[key] = [value]
        _x, s2m2, teacher, _delta, refined = run_model(model, collated, device)
        frames.append(
            {
                "sample_id": int(batch["sample_id"]),
                "s2m2": s2m2.detach().cpu(),
                "teacher": teacher.detach().cpu(),
                "refined": refined.detach().cpu(),
            }
        )
    frames = sorted(frames, key=lambda row: row["sample_id"])
    rows = []
    for prev, cur in zip(frames[:-1], frames[1:]):
        if cur["sample_id"] != prev["sample_id"] + 1:
            continue
        rows.append(
            {
                "s2m2_temporal_diff": float(torch.mean(torch.abs(cur["s2m2"] - prev["s2m2"]))),
                "teacher_temporal_diff": float(torch.mean(torch.abs(cur["teacher"] - prev["teacher"]))),
                "refined_temporal_diff": float(torch.mean(torch.abs(cur["refined"] - prev["refined"]))),
            }
        )
    out = mean_dict(rows)
    out["pair_count"] = float(len(rows))
    if "s2m2_temporal_diff" in out and "refined_temporal_diff" in out:
        out["refined_minus_s2m2_temporal_diff"] = out["refined_temporal_diff"] - out["s2m2_temporal_diff"]
    return out


def mean_dict(rows):
    if not rows:
        return {}
    keys = sorted({k for r in rows for k in r})
    out = {}
    for key in keys:
        vals = [r[key] for r in rows if key in r and np.isfinite(r[key])]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    return out


def append_log(path, row, header_written):
    keys = [
        "epoch",
        "train_loss",
        "train_loss_teacher",
        "train_loss_temporal",
        "train_loss_residual_l1",
        "train_loss_smooth",
        "train_teacher_mae_before",
        "train_teacher_mae_after",
        "val_teacher_mae_before",
        "val_teacher_mae_after",
        "val_residual_mean",
        "val_residual_std",
        "val_residual_min",
        "val_residual_max",
        "val_gt_disp_mae",
        "val_gt_depth_mae",
        "val_gt_bad_2mm",
    ]
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if not header_written:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in keys})


@torch.no_grad()
def save_qualitative(model, dataset, out_dir, device, max_items=5):
    qdir = out_dir / "qualitative"
    qdir.mkdir(parents=True, exist_ok=True)
    model.eval()
    chosen = np.linspace(0, len(dataset) - 1, min(max_items, len(dataset)), dtype=int)
    for idx in chosen:
        batch = dataset[int(idx)]
        collated = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                collated[key] = value.unsqueeze(0)
            else:
                collated[key] = [value]
        x, s2m2, teacher, delta, refined = run_model(model, collated, device)
        rgb = (x[0, :3].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        s2m2_np = s2m2[0, 0].detach().cpu().numpy()
        teacher_np = teacher[0, 0].detach().cpu().numpy()
        refined_np = refined[0, 0].detach().cpu().numpy()
        delta_np = delta[0, 0].detach().cpu().numpy()
        valid = np.isfinite(s2m2_np) & np.isfinite(teacher_np)
        vmax = float(np.nanpercentile(np.concatenate([s2m2_np[valid], teacher_np[valid], refined_np[valid]]), 99))
        tiles = [
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            colorize(s2m2_np, vmax),
            colorize(teacher_np, vmax),
            colorize(refined_np, vmax),
            colorize(np.abs(s2m2_np - teacher_np), 8.0, cv2.COLORMAP_MAGMA),
            colorize(np.abs(refined_np - teacher_np), 8.0, cv2.COLORMAP_MAGMA),
            colorize(delta_np, 4.0, cv2.COLORMAP_VIRIDIS),
        ]
        labels = ["RGB", "S2M2", "Teacher", "Refined", "|S2M2-T|", "|Ref-T|", "delta"]
        small = []
        for tile, label in zip(tiles, labels):
            tile = cv2.resize(tile, (180, 120), interpolation=cv2.INTER_AREA)
            cv2.putText(tile, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            small.append(tile)
        cv2.imwrite(str(qdir / f"sample_{int(batch['sample_id']):06d}.png"), np.concatenate(small, axis=1))


def write_report(args, metrics, best_epoch):
    temporal = metrics.get("temporal_consecutive32_fixed_crop", {})
    comparison = metrics.get("v1_comparison", {})
    lines = [
        f"# {args.report_title}",
        "",
        "Debug/overfit experiment for the first ARGOS temporal residual refiner. S2M2 and StereoAnyVideo are frozen; only the Tiny U-Net refiner is trained.",
        "",
        f"Cache: `{args.cache_root}`",
        f"Best epoch: `{best_epoch}`",
        "",
        "## Loss",
        "",
        f"- teacher SmoothL1 weight: `{args.teacher_weight}`",
        f"- temporal window median weight: `{args.temporal_weight}`",
        f"- residual L1 weight: `{args.residual_l1_weight}`",
        f"- edge-aware smoothness weight: `{args.smoothness_weight}`",
        "",
        "## Metrics",
        "",
        f"- train teacher MAE before: `{metrics['train'].get('teacher_mae_before', float('nan')):.4f}`",
        f"- train teacher MAE after: `{metrics['train'].get('teacher_mae_after', float('nan')):.4f}`",
        f"- val teacher MAE before: `{metrics['val'].get('teacher_mae_before', float('nan')):.4f}`",
        f"- val teacher MAE after: `{metrics['val'].get('teacher_mae_after', float('nan')):.4f}`",
        f"- val residual mean/std: `{metrics['val'].get('residual_mean', float('nan')):.4f}` / `{metrics['val'].get('residual_std', float('nan')):.4f}`",
        f"- val residual min/max: `{metrics['val'].get('residual_min', float('nan')):.4f}` / `{metrics['val'].get('residual_max', float('nan')):.4f}`",
        "",
        "GT metrics are reported only if the validation crop includes the single GT sample.",
        "",
        f"- val GT disp MAE: `{metrics['val'].get('gt_disp_mae', float('nan')):.4f}`",
        f"- val GT depth MAE: `{metrics['val'].get('gt_depth_mae', float('nan')):.4f}`",
        f"- val GT Bad-2mm: `{metrics['val'].get('gt_bad_2mm', float('nan')):.2f}%`",
        "",
        "## Temporal Consistency",
        "",
        "Measured on fixed validation-size center crops from consecutive32 samples.",
        "",
        f"- pair count: `{temporal.get('pair_count', float('nan')):.0f}`",
        f"- S2M2 temporal diff: `{temporal.get('s2m2_temporal_diff', float('nan')):.4f}` px",
        f"- refined temporal diff: `{temporal.get('refined_temporal_diff', float('nan')):.4f}` px",
        f"- StereoAnyVideo teacher temporal diff: `{temporal.get('teacher_temporal_diff', float('nan')):.4f}` px",
        f"- refined minus S2M2 temporal diff: `{temporal.get('refined_minus_s2m2_temporal_diff', float('nan')):.4f}` px",
        "",
        "## V1 Comparison",
        "",
        f"- teacher MAE after change: `{comparison.get('val_teacher_mae_after_delta', float('nan')):.4f}` px",
        f"- refined temporal diff change: `{comparison.get('refined_temporal_diff_delta', float('nan')):.4f}` px",
        f"- residual std change: `{comparison.get('val_residual_std_delta', float('nan')):.4f}`",
        f"- GT depth MAE change: `{comparison.get('val_gt_depth_mae_delta', float('nan')):.4f}` mm",
        "",
        "Negative temporal-diff change means v2 is smoother than v1. Positive teacher-MAE or GT-depth change means degradation relative to v1.",
        "",
        "## Notes",
        "",
        "- This is a debug experiment, not final training.",
        "- Full images are cropped to avoid memory blow-up.",
        "- CUDA is used when available, but the debug crop and batch size are intentionally too small to saturate a large GPU.",
        "- The checkpoint is a local artifact and should not be committed unless explicitly requested.",
    ]
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=Path("results/03_temporal_refinement/cache/debug_v1"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/temporal_refinement_debug_unet_v1"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--crop-h", type=int, default=256)
    parser.add_argument("--crop-w", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--residual-clamp-px", type=float, default=4.0)
    parser.add_argument("--teacher-weight", type=float, default=1.0)
    parser.add_argument("--temporal-weight", type=float, default=0.0)
    parser.add_argument("--residual-l1-weight", type=float, default=0.05)
    parser.add_argument("--smoothness-weight", type=float, default=0.05)
    parser.add_argument("--v1-metrics", type=Path, default=Path("results/temporal_refinement_debug_unet_v1/metrics.json"))
    parser.add_argument("--report-title", default="Tiny U-Net Temporal Refiner Debug V1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    train_ids, val_ids = split_sample_ids(args.cache_root, val_count=5)
    save_config(args, train_ids, val_ids)

    train_ds = TemporalRefinementCacheDataset(args.cache_root, train_ids, (args.crop_h, args.crop_w), random_crop=True)
    train_eval_ds = TemporalRefinementCacheDataset(args.cache_root, train_ids, (args.crop_h, args.crop_w), random_crop=False)
    val_ds = TemporalRefinementCacheDataset(args.cache_root, val_ids, (args.crop_h, args.crop_w), random_crop=False)
    all_eval_ds = TemporalRefinementCacheDataset(args.cache_root, train_ids + val_ids, (args.crop_h, args.crop_w), random_crop=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=1, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = TinyUNetRefiner(in_channels=8, base_channels=args.base_channels, residual_clamp_px=args.residual_clamp_px).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    use_amp = (device.type == "cuda") and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    log_path = args.out_dir / "train_log.csv"
    if log_path.exists():
        log_path.unlink()

    best_val = float("inf")
    best_epoch = -1
    header_written = False
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, use_amp, args)
        train_metrics = evaluate(model, train_eval_loader, device)
        val_metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_loss.items()},
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        append_log(log_path, row, header_written)
        header_written = True
        val_score = val_metrics.get("teacher_mae_after", float("inf"))
        if val_score < best_val:
            best_val = val_score
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_teacher_mae_after": best_val,
                    "train_ids": train_ids,
                    "val_ids": val_ids,
                },
                args.out_dir / "checkpoints" / "best.pt",
            )
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:03d} train_after={train_metrics.get('teacher_mae_after', float('nan')):.4f} "
                f"val_after={val_metrics.get('teacher_mae_after', float('nan')):.4f}",
                flush=True,
            )

    ckpt = torch.load(args.out_dir / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    final_metrics = {
        "train": evaluate(model, train_eval_loader, device),
        "val": evaluate(model, val_loader, device),
        "temporal_consecutive32_fixed_crop": temporal_consistency_metrics(model, all_eval_ds, device),
        "best_epoch": best_epoch,
        "train_ids": train_ids,
        "val_ids": val_ids,
    }
    if args.v1_metrics.exists():
        v1 = json.loads(args.v1_metrics.read_text())
        final_metrics["v1_comparison"] = {
            "val_teacher_mae_after_delta": final_metrics["val"].get("teacher_mae_after", float("nan")) - v1["val"].get("teacher_mae_after", float("nan")),
            "refined_temporal_diff_delta": final_metrics["temporal_consecutive32_fixed_crop"].get("refined_temporal_diff", float("nan"))
            - v1["temporal_consecutive32_fixed_crop"].get("refined_temporal_diff", float("nan")),
            "val_residual_std_delta": final_metrics["val"].get("residual_std", float("nan")) - v1["val"].get("residual_std", float("nan")),
            "val_gt_depth_mae_delta": final_metrics["val"].get("gt_depth_mae", float("nan")) - v1["val"].get("gt_depth_mae", float("nan")),
        }
    (args.out_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2) + "\n")
    save_qualitative(model, val_ds, args.out_dir, device)
    write_report(args, final_metrics, best_epoch)


if __name__ == "__main__":
    main()
