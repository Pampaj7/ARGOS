#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from scripts.temporal_refinement.lib.datasets import PairedTemporalRefinementCacheDataset, TemporalRefinementCacheDataset, split_sample_ids
from scripts.temporal_refinement.lib.losses import edge_aware_smoothness
from scripts.temporal_refinement.lib.metrics import gt_metrics, teacher_metrics
from scripts.temporal_refinement.lib.models import TinyUNetRefiner
from scripts.temporal_refinement.legacy.train_debug_unet_refiner import colorize, evaluate, mean_dict, run_model, temporal_consistency_metrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_pair(model, batch, device):
    prev = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch["prev"].items()}
    cur = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch["cur"].items()}
    x_prev, s_prev, t_prev, d_prev, r_prev = run_model(model, prev, device)
    x_cur, s_cur, t_cur, d_cur, r_cur = run_model(model, cur, device)
    return (x_prev, s_prev, t_prev, d_prev, r_prev), (x_cur, s_cur, t_cur, d_cur, r_cur)


def pair_loss(model, batch, device, args):
    prev, cur = run_pair(model, batch, device)
    x_prev, _s_prev, t_prev, d_prev, r_prev = prev
    x_cur, _s_cur, t_cur, d_cur, r_cur = cur
    abs_loss = F.smooth_l1_loss(r_prev, t_prev) + F.smooth_l1_loss(r_cur, t_cur)
    delta_loss = F.smooth_l1_loss(r_cur - r_prev, t_cur - t_prev)
    residual = torch.mean(torch.abs(d_prev)) + torch.mean(torch.abs(d_cur))
    edge = edge_aware_smoothness(d_prev, x_prev[:, :3]) + edge_aware_smoothness(d_cur, x_cur[:, :3])
    prev_anchor_mask = torch.abs(_s_prev - t_prev) < args.anchor_threshold_px
    cur_anchor_mask = torch.abs(_s_cur - t_cur) < args.anchor_threshold_px
    anchor_terms = []
    if prev_anchor_mask.any():
        anchor_terms.append(F.smooth_l1_loss(r_prev[prev_anchor_mask], _s_prev[prev_anchor_mask]))
    if cur_anchor_mask.any():
        anchor_terms.append(F.smooth_l1_loss(r_cur[cur_anchor_mask], _s_cur[cur_anchor_mask]))
    anchor = torch.stack(anchor_terms).mean() if anchor_terms else r_cur.new_tensor(0.0)
    anchor_ratio = 0.5 * (prev_anchor_mask.float().mean() + cur_anchor_mask.float().mean())
    loss = (
        args.abs_weight * abs_loss
        + args.delta_weight * delta_loss
        + args.residual_l1_weight * residual
        + args.edge_weight * edge
        + args.anchor_weight * anchor
    )
    return loss, {
        "loss_abs": float(abs_loss.detach().cpu()),
        "loss_delta": float(delta_loss.detach().cpu()),
        "loss_residual_l1": float(residual.detach().cpu()),
        "loss_edge": float(edge.detach().cpu()),
        "loss_anchor": float(anchor.detach().cpu()),
        "anchor_ratio": float(anchor_ratio.detach().cpu()),
    }


def train_one_epoch(model, loader, optimizer, scaler, device, use_amp, args):
    model.train()
    rows = []
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss, parts = pair_loss(model, batch, device, args)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        parts["loss"] = float(loss.detach().cpu())
        rows.append(parts)
    return mean_dict(rows)


@torch.no_grad()
def temporal_delta_metrics(model, dataset, device):
    model.eval()
    rows = []
    for batch in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
        prev, cur = run_pair(model, batch, device)
        _xp, s_prev, t_prev, _dp, r_prev = prev
        _xc, s_cur, t_cur, _dc, r_cur = cur
        rows.append(
            {
                "s2m2_temporal_diff": float(torch.mean(torch.abs(s_cur - s_prev)).cpu()),
                "teacher_temporal_diff": float(torch.mean(torch.abs(t_cur - t_prev)).cpu()),
                "refined_temporal_diff": float(torch.mean(torch.abs(r_cur - r_prev)).cpu()),
                "temporal_delta_mae": float(torch.mean(torch.abs((r_cur - r_prev) - (t_cur - t_prev))).cpu()),
                "anchor_mask_ratio": float((torch.abs(s_cur - t_cur) < 1.0).float().mean().cpu()),
            }
        )
    return mean_dict(rows) | {"pair_count": float(len(rows))}


def evaluate_single(model, sample_ids, args, device):
    ds = TemporalRefinementCacheDataset(args.cache_root, sample_ids, (args.crop_h, args.crop_w), random_crop=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    return evaluate(model, loader, device)


@torch.no_grad()
def save_pair_qualitative(model, dataset, out_dir, device, max_items=5):
    qdir = out_dir / "qualitative"
    qdir.mkdir(parents=True, exist_ok=True)
    chosen = np.linspace(0, len(dataset) - 1, min(max_items, len(dataset)), dtype=int)
    for idx in chosen:
        batch = dataset[int(idx)]
        collated = {"prev": {}, "cur": {}}
        for side in ("prev", "cur"):
            for key, value in batch[side].items():
                collated[side][key] = value.unsqueeze(0) if torch.is_tensor(value) else [value]
        prev, cur = run_pair(model, collated, device)
        x_cur, s_cur, t_cur, _d_cur, r_cur = cur
        _x_prev, s_prev, t_prev, _d_prev, r_prev = prev
        rgb = (x_cur[0, :3].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        s = s_cur[0, 0].cpu().numpy()
        t = t_cur[0, 0].cpu().numpy()
        r = r_cur[0, 0].cpu().numpy()
        vmax = float(np.nanpercentile(np.concatenate([s.ravel(), t.ravel(), r.ravel()]), 99))
        tiles = [
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            colorize(s, vmax),
            colorize(t, vmax),
            colorize(r, vmax),
            colorize(np.abs(s - t), 8.0, cv2.COLORMAP_MAGMA),
            colorize(np.abs(r - t), 8.0, cv2.COLORMAP_MAGMA),
            colorize(np.abs(s - s_prev[0, 0].cpu().numpy()), 6.0, cv2.COLORMAP_MAGMA),
            colorize(np.abs(t - t_prev[0, 0].cpu().numpy()), 6.0, cv2.COLORMAP_MAGMA),
            colorize(np.abs(r - r_prev[0, 0].cpu().numpy()), 6.0, cv2.COLORMAP_MAGMA),
        ]
        labels = ["RGB_t", "S2M2_t", "Teacher_t", "Refined_t", "|S-T|", "|R-T|", "td S2M2", "td T", "td R"]
        small = []
        for tile, label in zip(tiles, labels):
            tile = cv2.resize(tile, (160, 110), interpolation=cv2.INTER_AREA)
            cv2.putText(tile, label, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            small.append(tile)
        cv2.imwrite(str(qdir / f"pair_{int(batch['cur']['sample_id']):06d}.png"), np.concatenate(small, axis=1))


def write_config(args, train_ids, val_ids):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "config.yaml").write_text(
        f"""cache_root: {args.cache_root}
output_root: {args.out_dir}
model: TinyUNetRefiner
train_mode: consecutive_pair_teacher_delta
crop_size: [{args.crop_h}, {args.crop_w}]
epochs: {args.epochs}
batch_size: {args.batch_size}
learning_rate: {args.lr}
loss:
  abs_teacher: {args.abs_weight}
  teacher_delta: {args.delta_weight}
  residual_l1: {args.residual_l1_weight}
  edge_aware_smoothness: {args.edge_weight}
  s2m2_anchor: {args.anchor_weight}
  anchor_threshold_px: {args.anchor_threshold_px}
eval_every: {args.eval_every}
save_every: {args.save_every}
num_workers: {args.num_workers}
train_ids: {train_ids}
val_ids: {val_ids}
"""
    )


def write_report(args, metrics):
    v1 = metrics["comparisons"]["v1"]
    v2 = metrics["comparisons"]["v2"]
    lines = [
        "# Tiny U-Net Temporal Refiner Debug V3 Teacher Delta Loss",
        "",
        "True consecutive-frame temporal distillation debug run.",
        "",
        f"Best epoch: `{metrics['best_epoch']}`",
        f"Peak VRAM allocated: `{metrics.get('peak_vram_mb', float('nan')):.1f} MB`",
        f"Runtime seconds: `{metrics.get('runtime_seconds', float('nan')):.1f}`",
        "",
        "## Metrics",
        "",
        f"- val teacher MAE before: `{metrics['val']['teacher_mae_before']:.4f}`",
        f"- val teacher MAE after: `{metrics['val']['teacher_mae_after']:.4f}`",
        f"- refined temporal diff: `{metrics['temporal']['refined_temporal_diff']:.4f}` px",
        f"- teacher temporal diff: `{metrics['temporal']['teacher_temporal_diff']:.4f}` px",
        f"- S2M2 temporal diff: `{metrics['temporal']['s2m2_temporal_diff']:.4f}` px",
        f"- temporal delta MAE: `{metrics['temporal']['temporal_delta_mae']:.4f}` px",
        f"- val residual std: `{metrics['val']['residual_std']:.4f}`",
        f"- anchor mask ratio: `{metrics['temporal'].get('anchor_mask_ratio', float('nan')):.4f}`",
        f"- val GT depth MAE: `{metrics['val'].get('gt_depth_mae', float('nan')):.4f}` mm",
        f"- val GT Bad-2mm: `{metrics['val'].get('gt_bad_2mm', float('nan')):.2f}%`",
        "",
        "## Comparison",
        "",
        f"- versus V1 temporal diff delta: `{metrics['temporal']['refined_temporal_diff'] - v1['temporal_consecutive32_fixed_crop']['refined_temporal_diff']:.4f}` px",
        f"- versus V2 temporal diff delta: `{metrics['temporal']['refined_temporal_diff'] - v2['temporal_consecutive32_fixed_crop']['refined_temporal_diff']:.4f}` px",
        f"- versus V1 teacher MAE delta: `{metrics['val']['teacher_mae_after'] - v1['val']['teacher_mae_after']:.4f}` px",
        f"- versus V2 teacher MAE delta: `{metrics['val']['teacher_mae_after'] - v2['val']['teacher_mae_after']:.4f}` px",
        "",
        "Negative deltas are improvements.",
    ]
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-root", type=Path, default=Path("results/temporal_refinement_cache/debug_v1"))
    p.add_argument("--out-dir", type=Path, default=Path("results/temporal_refinement_debug_unet_v3_teacher_delta_loss"))
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--crop-h", "--crop-height", type=int, default=256)
    p.add_argument("--crop-w", "--crop-width", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--abs-weight", type=float, default=0.7)
    p.add_argument("--delta-weight", type=float, default=0.3)
    p.add_argument("--residual-l1-weight", type=float, default=0.05)
    p.add_argument("--edge-weight", type=float, default=0.03)
    p.add_argument("--anchor-weight", type=float, default=0.0)
    p.add_argument("--anchor-threshold-px", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--amp", dest="amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--v1-metrics", type=Path, default=Path("results/temporal_refinement_debug_unet_v1/metrics.json"))
    p.add_argument("--v2-metrics", type=Path, default=Path("results/temporal_refinement_debug_unet_v2_temporal_loss/metrics.json"))
    return p.parse_args()


def main():
    args = parse_args()
    start_time = time.time()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "checkpoints").mkdir(exist_ok=True)
    train_ids, val_ids = split_sample_ids(args.cache_root, val_count=5)
    if args.max_train_samples:
        train_ids = train_ids[: args.max_train_samples]
    if args.max_val_samples:
        val_ids = val_ids[: args.max_val_samples]
    write_config(args, train_ids, val_ids)
    train_pairs = PairedTemporalRefinementCacheDataset(args.cache_root, train_ids, (args.crop_h, args.crop_w), random_crop=True)
    eval_pairs = PairedTemporalRefinementCacheDataset(args.cache_root, train_ids + val_ids, (args.crop_h, args.crop_w), random_crop=False)
    train_loader = DataLoader(train_pairs, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model = TinyUNetRefiner(in_channels=8, base_channels=16, residual_clamp_px=4.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_epoch = 1
    if args.resume is not None and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            opt.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp)
    log_path = args.out_dir / "train_log.csv"
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_loss_abs", "train_loss_delta", "train_loss_residual_l1", "train_loss_edge", "train_loss_anchor", "train_anchor_ratio", "val_teacher_mae_after", "temporal_delta_mae", "refined_temporal_diff"])
        writer.writeheader()
    best = float("inf")
    best_epoch = -1
    last_val = {}
    last_temporal = {}
    for epoch in range(start_epoch, args.epochs + 1):
        train = train_one_epoch(model, train_loader, opt, scaler, device, device.type == "cuda" and args.amp, args)
        should_eval = epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs
        if should_eval:
            val = evaluate_single(model, val_ids, args, device)
            temporal = temporal_delta_metrics(model, eval_pairs, device)
            last_val = val
            last_temporal = temporal
        else:
            val = last_val
            temporal = last_temporal
        score = temporal.get("temporal_delta_mae", float("inf")) + 0.25 * val.get("teacher_mae_after", float("inf"))
        if score < best:
            best = score
            best_epoch = epoch
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(), "epoch": epoch, "score": score}, args.out_dir / "checkpoints" / "best.pt")
        if args.save_every and epoch % args.save_every == 0:
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(), "epoch": epoch, "score": score}, args.out_dir / "checkpoints" / f"epoch_{epoch:04d}.pt")
        with log_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_loss_abs", "train_loss_delta", "train_loss_residual_l1", "train_loss_edge", "train_loss_anchor", "train_anchor_ratio", "val_teacher_mae_after", "temporal_delta_mae", "refined_temporal_diff"])
            writer.writerow({"epoch": epoch, **{f"train_{k}": v for k, v in train.items()}, "val_teacher_mae_after": val.get("teacher_mae_after", ""), "temporal_delta_mae": temporal.get("temporal_delta_mae", ""), "refined_temporal_diff": temporal.get("refined_temporal_diff", "")})
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:03d} val_after={val.get('teacher_mae_after', float('nan')):.4f} td_mae={temporal.get('temporal_delta_mae', float('nan')):.4f} ref_td={temporal.get('refined_temporal_diff', float('nan')):.4f}", flush=True)
    ckpt = torch.load(args.out_dir / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    final = {
        "train": evaluate_single(model, train_ids, args, device),
        "val": evaluate_single(model, val_ids, args, device),
        "temporal": temporal_delta_metrics(model, eval_pairs, device),
        "best_epoch": best_epoch,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "comparisons": {"v1": json.loads(args.v1_metrics.read_text()), "v2": json.loads(args.v2_metrics.read_text())},
        "runtime_seconds": time.time() - start_time,
        "peak_vram_mb": float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == "cuda" else 0.0,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(final, indent=2) + "\n")
    save_pair_qualitative(model, eval_pairs, args.out_dir, device)
    write_report(args, final)


if __name__ == "__main__":
    main()
