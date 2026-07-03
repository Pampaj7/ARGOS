#!/usr/bin/env python3
"""v3.3b: hard-negative residual-only fine-tuning on the two pathological clips.

Starts from the v3.2c checkpoint. Detector and trunk frozen (residual head only by
default). Batch mix: 50% full-GT safety crops, 25% selected-oracle crops from the four
well-behaved clips, 25% hard-negative crops from the two pathological clips
(high_temporal_flicker / high_boundary_error). Hard negatives are pixels v3.2c turns
from raw-good (<3px) into refined-Bad3 with p_bad >= 0.7; hard positives are pixels
where the oracle clearly beats raw. Threshold stays 0.7 throughout.
No S2M2/SAV/RAFT/DINO inference is run.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "eval_scripts"))

from train_tiny_refiner_v1_full_gt import (  # noqa: E402
    DEFAULT_TARGETS_ROOT,
    charbonnier,
    finite_mean,
    load_shards,
    masked_mean,
    parse_bool,
    write_csv,
)
from train_tiny_refiner_v3_1_staged_abstention import (  # noqa: E402
    DEFAULT_BALANCED_SPLIT,
    AbstentionCropRefiner,
    BalancedCropDataset,
    FullFrameDataset,
    evaluate,
    load_samples_with_split,
    make_features_from_raws,
    set_requires_grad,
    unwrap,
    write_csv_union,
)
from train_tiny_refiner_v3_2_hybrid_oracle import (  # noqa: E402
    Clip,
    OracleCropDataset,
    hybrid_loss_batch,
    load_clips,
    make_loader,
    row_at,
    selected_frame_rows,
)
from calibrate_v3_3_failure_mode_thresholds import (  # noqa: E402
    PATHOLOGICAL_MODES,
    aggregate,
    frame_metrics_for_threshold,
)
from evaluate_v3_1_on_selected_oracle_clips import diagnostic, predict_clip  # noqa: E402


DEFAULT_BASE_CHECKPOINT = Path("results/03_temporal_refinement/training/tiny_refiner_v3_2c_hybrid_oracle_freeze_detector_long/checkpoints/best.pt")
DEFAULT_ORACLE_TARGETS = Path("results/03_temporal_refinement/evaluation/distillation_targets_selected_clips")
DEFAULT_OUTPUT_ROOT = Path("results/03_temporal_refinement/training/tiny_refiner_v3_3b_hard_negative_residual")
EVAL_THRESHOLD = 0.7
SWEEP_THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95, 1.01)
BASELINES = {
    "v3.1": {"selected_mae": 11.0421, "gap_pct": 6.22, "new_bad3_fm": 4.79, "modified": 52.27},
    "v3.2c": {"selected_mae": 11.0054, "gap_pct": 7.03, "new_bad3_fm": 5.36, "modified": 18.43, "patho_new_bad3": 15.77, "clean_new_bad3": 0.89},
    "v3.3_threshold_only": {"selected_mae": 11.1062, "gap_pct": 4.80, "new_bad3_fm": 2.63, "patho_new_bad3": 6.69},
}


def mine_hard_masks(clip: Clip, p_bad: np.ndarray, residual: np.ndarray, args: argparse.Namespace) -> dict[str, np.ndarray]:
    """Hard negatives: raw-good(<3px) pixels v3.2c flips to Bad-3 with p_bad>=0.7.
    Hard positives: raw-bad pixels where the oracle clearly beats raw."""
    hard = (p_bad >= EVAL_THRESHOLD).astype(np.float32)
    refined = clip.raws + hard * residual
    raw_err = np.abs(clip.raws - clip.gts)
    ref_err = np.abs(refined - clip.gts)
    oracle_err = np.abs(clip.oracle - clip.gts)
    valid = clip.valids > 0
    hard_neg = valid & (raw_err < 3.0) & (ref_err >= 3.0) & (p_bad >= EVAL_THRESHOLD)
    hard_pos = valid & (raw_err >= 3.0) & (oracle_err + args.oracle_margin_px < raw_err)
    return {"hard_neg": hard_neg, "hard_pos": hard_pos}


class HardNegativeCropDataset(Dataset):
    """Crops from the two pathological clips, biased toward mined hard-negative pixels."""

    def __init__(self, clips: list[Clip], masks: dict[str, dict[str, np.ndarray]], args: argparse.Namespace, crops_per_epoch: int):
        self.clips = clips
        self.masks = masks
        self.args = args
        self.crops_per_epoch = crops_per_epoch
        # oversample frames that actually contain hard negatives
        self.frame_pool = []
        for ci, c in enumerate(clips):
            hn = masks[c.clip_id]["hard_neg"]
            for fi in range(len(c.frame_ids)):
                weight = 3 if hn[fi].any() else 1
                self.frame_pool.extend([(ci, fi)] * weight)
        self.rng = random.Random(7777)

    def __len__(self) -> int:
        return self.crops_per_epoch

    def __getitem__(self, idx: int) -> dict[str, Any]:
        a = self.args
        ci, fi = self.frame_pool[self.rng.randrange(len(self.frame_pool))]
        clip = self.clips[ci]
        hn = self.masks[clip.clip_id]["hard_neg"][fi]
        hp = self.masks[clip.clip_id]["hard_pos"][fi]
        h, w = clip.raws.shape[1:]
        s = min(a.crop_size, h, w)
        best = (-1.0, 0, 0)
        for _ in range(a.crop_candidate_tries):
            y = self.rng.randint(0, max(0, h - s))
            x = self.rng.randint(0, max(0, w - s))
            score = float(hn[y : y + s, x : x + s].mean()) + 0.3 * float(hp[y : y + s, x : x + s].mean())
            if score >= best[0]:
                best = (score, y, x)
        _, y, x = best
        ys, xs = slice(y, y + s), slice(x, x + s)
        ids = [max(0, fi - i) for i in range(a.context_frames)]
        raws = clip.raws[ids, ys, xs]
        valids = clip.valids[ids, ys, xs]
        xfeat, _e, _v = make_features_from_raws(raws, valids)
        raw = raws[0]
        gt = clip.gts[fi, ys, xs]
        raw_err = np.abs(raw - gt)
        return {
            "x": torch.from_numpy(xfeat),
            "raw": torch.from_numpy(raw[None]),
            "gt": torch.from_numpy(gt[None]),
            "delta": torch.from_numpy((clip.oracle[fi, ys, xs] - raw)[None].astype(np.float32)),
            "valid": torch.from_numpy(valids[0][None]),
            "hard_neg": torch.from_numpy(hn[ys, xs].astype(np.float32)[None]),
            "hard_pos": torch.from_numpy(hp[ys, xs].astype(np.float32)[None]),
            "below3": torch.from_numpy((raw_err < 3.0).astype(np.float32)[None]),
        }


def hard_negative_loss_batch(model: nn.Module, batch: dict[str, torch.Tensor], args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    x = batch["x"].to(device, non_blocking=True)
    raw = batch["raw"].to(device, non_blocking=True)
    gt = batch["gt"].to(device, non_blocking=True)
    delta = batch["delta"].to(device, non_blocking=True)
    valid = batch["valid"].to(device, non_blocking=True)
    hard_neg = batch["hard_neg"].to(device, non_blocking=True) * valid
    hard_pos = batch["hard_pos"].to(device, non_blocking=True) * valid
    below3 = batch["below3"].to(device, non_blocking=True) * valid
    _logit, p_bad, residual = model(x, args.residual_scale)
    refined = raw + p_bad * residual
    zero = residual.sum() * 0.0
    target = delta.clamp(-args.residual_scale, args.residual_scale)
    pos_loss = masked_mean(charbonnier(residual - target), hard_pos) if float(hard_pos.sum()) > 0 else zero
    hn_preserve = masked_mean(torch.abs(refined - raw), hard_neg) if float(hard_neg.sum()) > 0 else zero
    shrink = masked_mean(torch.abs(residual), hard_neg) if float(hard_neg.sum()) > 0 else zero
    nb3 = masked_mean(torch.relu(torch.abs(refined - gt) - 3.0), below3) if float(below3.sum()) > 0 else zero
    anchor = masked_mean(torch.clamp(charbonnier(refined - gt), max=args.robust_loss_clip_px), valid)
    loss = (
        args.oracle_positive_weight * pos_loss
        + args.hard_negative_weight * hn_preserve
        + args.shrinkage_weight * shrink
        + args.new_bad3_weight * nb3
        + args.full_weight * anchor
    )
    return loss, {
        "hardneg_loss": float(loss.detach().cpu()),
        "hardneg_pos_loss": float(pos_loss.detach().cpu()),
        "hardneg_preserve_loss": float(hn_preserve.detach().cpu()),
        "hardneg_shrink_loss": float(shrink.detach().cpu()),
        "hardneg_nb3_loss": float(nb3.detach().cpu()),
        "hardneg_anchor_loss": float(anchor.detach().cpu()),
        "hardneg_frac": float(masked_mean(hard_neg, valid).detach().cpu()),
    }


def train_one_epoch(model: nn.Module, loaders: dict[str, DataLoader], optimizer: torch.optim.Optimizer, args: argparse.Namespace, device: torch.device, rng: random.Random) -> dict[str, float]:
    model.train()
    order = [name for name, loader in loaders.items() for _ in range(len(loader))]
    rng.shuffle(order)
    iters = {name: iter(loader) for name, loader in loaders.items()}
    rows: list[dict[str, float]] = []
    for source in order:
        try:
            batch = next(iters[source])
        except StopIteration:
            continue
        if source == "hardneg":
            loss, metrics = hard_negative_loss_batch(model, batch, args, device)
        else:
            loss, metrics = hybrid_loss_batch(model, batch, args, device, source)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        rows.append(metrics)
    keys = {k for r in rows for k in r}
    return {k: finite_mean([r[k] for r in rows if k in r]) for k in sorted(keys)}


@torch.no_grad()
def eval_selected_groups(model: nn.Module, clips: list[Clip], args: argparse.Namespace, device: torch.device, thresholds: tuple[float, ...] = (EVAL_THRESHOLD,)) -> tuple[dict[float, dict[str, dict[str, float]]], dict[str, tuple[np.ndarray, np.ndarray]]]:
    model.eval()
    predictions = {}
    for clip in clips:
        _, p_bad, residual, _ = predict_clip(model, clip.raws, clip.valids, args, device, args.residual_scale, EVAL_THRESHOLD)
        predictions[clip.clip_id] = (p_bad, residual)
    out: dict[float, dict[str, dict[str, float]]] = {}
    for t in thresholds:
        frames_all, frames_patho, frames_clean = [], [], []
        for clip in clips:
            p_bad, residual = predictions[clip.clip_id]
            frames = frame_metrics_for_threshold(clip, p_bad, residual, t)
            frames_all.extend(frames)
            (frames_patho if clip.failure_mode in PATHOLOGICAL_MODES else frames_clean).extend(frames)
        out[t] = {"all": aggregate(frames_all), "patho": aggregate(frames_patho), "clean": aggregate(frames_clean)}
    return out, predictions


def score_epoch(sel: dict[str, dict[str, float]], fg_val: dict[str, Any]) -> float:
    return (
        sel["all"]["refined_mae"]
        + 0.02 * max(0.0, sel["patho"]["new_bad3_frame_mean_pct"] - 8.0)
        + 1.0 * max(0.0, sel["clean"]["new_bad3_frame_mean_pct"] - 1.2)
        + 20.0 * max(0.0, fg_val["refined_hard_mae"] - fg_val["raw_mae"])
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets-root", type=Path, default=DEFAULT_TARGETS_ROOT)
    p.add_argument("--oracle-targets-root", type=Path, default=DEFAULT_ORACLE_TARGETS)
    p.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--balanced-split-json", type=Path, default=DEFAULT_BALANCED_SPLIT)
    p.add_argument("--context-frames", type=int, default=4)
    p.add_argument("--crop-size", type=int, default=96)
    p.add_argument("--crop-candidate-tries", type=int, default=12)
    p.add_argument("--crops-per-epoch", type=int, default=60000)
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--full-gt-batch-ratio", type=float, default=0.50)
    p.add_argument("--oracle-batch-ratio", type=float, default=0.25)
    p.add_argument("--hard-negative-batch-ratio", type=float, default=0.25)
    p.add_argument("--residual-lr", type=float, default=1e-4)
    p.add_argument("--backbone-lr", type=float, default=0.0)
    p.add_argument("--detector-lr", type=float, default=0.0)
    p.add_argument("--num-workers", type=int, default=24)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--residual-scale", type=float, default=3.0)
    p.add_argument("--bad-threshold-px", type=float, default=3.0)
    p.add_argument("--good-threshold-px", type=float, default=1.0)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--oracle-min-improvement-px", type=float, default=1.0)
    p.add_argument("--oracle-margin-px", type=float, default=1.0)
    p.add_argument("--oracle-hard-only", nargs="?", const=True, default=False, type=parse_bool)
    p.add_argument("--full-weight", type=float, default=0.5)
    p.add_argument("--detector-weight", type=float, default=0.0)
    p.add_argument("--residual-weight", type=float, default=0.5)
    p.add_argument("--preserve-weight", type=float, default=1.0)
    p.add_argument("--new-bad3-weight", type=float, default=2.0)
    p.add_argument("--sparsity-weight", type=float, default=0.02)
    p.add_argument("--hard-negative-weight", type=float, default=2.0)
    p.add_argument("--oracle-positive-weight", type=float, default=1.0)
    p.add_argument("--shrinkage-weight", type=float, default=0.25)
    p.add_argument("--robust-loss-clip-px", type=float, default=10.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--early-stop-patience", type=int, default=5)
    p.add_argument("--max-auc-pixels", type=int, default=200000)
    p.add_argument("--diagnostics-per-clip", type=int, default=2)
    p.add_argument("--overwrite", nargs="?", const=True, default=False, type=parse_bool)
    args = p.parse_args()
    total = args.full_gt_batch_ratio + args.oracle_batch_ratio + args.hard_negative_batch_ratio
    args.full_gt_batch_ratio /= total
    args.oracle_batch_ratio /= total
    args.hard_negative_batch_ratio /= total
    return args


def main() -> int:
    args = parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(args.output_root)
        shutil.rmtree(args.output_root)
    (args.output_root / "checkpoints").mkdir(parents=True)
    (args.output_root / "diagnostics").mkdir()
    sys.stdout = (args.output_root / "stdout.log").open("w", buffering=1)
    sys.stderr = (args.output_root / "stderr.log").open("w", buffering=1)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    start = time.perf_counter()

    ckpt = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    input_channels = int(ckpt.get("input_channels", args.context_frames * 2 + 8))
    model = AbstentionCropRefiner(input_channels).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    params = sum(p.numel() for p in model.parameters())

    clips = load_clips(args.oracle_targets_root, args)
    patho_clips = [c for c in clips if c.failure_mode in PATHOLOGICAL_MODES]
    clean_clips = [c for c in clips if c.failure_mode not in PATHOLOGICAL_MODES]

    # mine hard negatives/positives from frozen v3.2c behavior (detector frozen -> masks stay valid)
    model.eval()
    masks: dict[str, dict[str, np.ndarray]] = {}
    with torch.no_grad():
        for clip in patho_clips:
            _, p_bad, residual, _ = predict_clip(model, clip.raws, clip.valids, args, device, args.residual_scale, EVAL_THRESHOLD)
            masks[clip.clip_id] = mine_hard_masks(clip, p_bad, residual, args)
    hn_pixels = int(sum(m["hard_neg"].sum() for m in masks.values()))
    hp_pixels = int(sum(m["hard_pos"].sum() for m in masks.values()))

    splits, by_split = load_samples_with_split(args.targets_root, args.balanced_split_json, args.max_frames)
    shards = load_shards(by_split["train"] + by_split["val"] + by_split["test"])
    gt_crops = int(round(args.crops_per_epoch * args.full_gt_batch_ratio))
    oracle_crops = int(round(args.crops_per_epoch * args.oracle_batch_ratio))
    hn_crops = args.crops_per_epoch - gt_crops - oracle_crops
    gt_args = argparse.Namespace(**{**vars(args), "crops_per_epoch": gt_crops})
    gt_ds = BalancedCropDataset(by_split["train"], shards, gt_args)
    oracle_ds = OracleCropDataset(clean_clips, args, oracle_crops)
    hn_ds = HardNegativeCropDataset(patho_clips, masks, args, hn_crops)
    full_ds = {split: FullFrameDataset(samples, shards, args.context_frames) for split, samples in by_split.items()}
    loaders = {
        "gt": make_loader(gt_ds, args.batch_size, args.num_workers, True, args.prefetch_factor),
        "oracle": make_loader(oracle_ds, args.batch_size, max(2, args.num_workers // 4), True, args.prefetch_factor),
        "hardneg": make_loader(hn_ds, args.batch_size, max(2, args.num_workers // 4), True, args.prefetch_factor),
    }
    eval_loaders = {split: make_loader(ds, args.eval_batch_size, max(0, args.num_workers // 2), False, args.prefetch_factor) for split, ds in full_ds.items()}

    # freeze everything except residual head (optional tiny backbone LR)
    m = unwrap(model)
    set_requires_grad(m.trunk, args.backbone_lr > 0)
    set_requires_grad(m.bad_head, False)
    set_requires_grad(m.residual_head, True)
    groups = [{"params": list(m.residual_head.parameters()), "lr": args.residual_lr}]
    if args.backbone_lr > 0:
        groups.append({"params": list(m.trunk.parameters()), "lr": args.backbone_lr})
    optimizer = torch.optim.AdamW(groups)

    run_lines = [
        f"device={device} params={params} base={args.base_checkpoint} base_epoch={ckpt.get('epoch')}",
        f"clips={len(clips)} patho={len(patho_clips)} clean={len(clean_clips)}",
        f"hard_neg_pixels={hn_pixels} hard_pos_pixels={hp_pixels}",
        f"crops gt={gt_crops} oracle={oracle_crops} hardneg={hn_crops} batch={args.batch_size}",
        f"lrs residual={args.residual_lr} backbone={args.backbone_lr} detector=frozen",
        f"weights hn={args.hard_negative_weight} pos={args.oracle_positive_weight} nb3={args.new_bad3_weight} shrink={args.shrinkage_weight}",
        f"frames={{'train': {len(full_ds['train'])}, 'val': {len(full_ds['val'])}, 'test': {len(full_ds['test'])}}}",
    ]
    (args.output_root / "run.log").write_text("\n".join(run_lines) + "\n")

    rng = random.Random(555)
    train_rows: list[dict[str, Any]] = []
    best_score = float("inf")
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        metrics = train_one_epoch(model, loaders, optimizer, args, device, rng)
        sel_by_thr, _ = eval_selected_groups(model, clips, args, device)
        sel = sel_by_thr[EVAL_THRESHOLD]
        fg_rows, _ = evaluate(model, eval_loaders["val"], args, device, "val", per_sequence=False)
        fg = row_at(fg_rows, EVAL_THRESHOLD)
        score = score_epoch(sel, fg)
        train_rows.append({
            "epoch": epoch, "score": score, **metrics,
            "sel_all_mae": sel["all"]["refined_mae"], "sel_all_gap": sel["all"]["oracle_gap_recovered_pct"],
            "sel_all_new_bad3": sel["all"]["new_bad3_frame_mean_pct"],
            "sel_patho_new_bad3": sel["patho"]["new_bad3_frame_mean_pct"], "sel_patho_mae": sel["patho"]["refined_mae"],
            "sel_clean_new_bad3": sel["clean"]["new_bad3_frame_mean_pct"], "sel_clean_mae": sel["clean"]["refined_mae"],
            "fullgt_val_raw": fg["raw_mae"], "fullgt_val_refined": fg["refined_hard_mae"],
        })
        if score < best_score:
            best_score = score
            best_epoch = epoch
            torch.save({
                "model_state_dict": unwrap(model).state_dict(),
                "args": vars(args), "splits": splits, "input_channels": input_channels,
                "parameter_count": params, "epoch": epoch, "threshold": EVAL_THRESHOLD,
                "selected_metrics": sel, "full_gt_val_metrics": fg,
            }, args.output_root / "checkpoints" / "best.pt")
        run_lines.append(
            f"epoch={epoch} score={score:.4f} sel_mae={sel['all']['refined_mae']:.4f} gap={sel['all']['oracle_gap_recovered_pct']:.2f}% "
            f"patho_nb3={sel['patho']['new_bad3_frame_mean_pct']:.2f}% clean_nb3={sel['clean']['new_bad3_frame_mean_pct']:.2f}% "
            f"mod={sel['all']['modified_pct']:.2f}% fullgt_val={fg['raw_mae']:.4f}->{fg['refined_hard_mae']:.4f}"
        )
        (args.output_root / "run.log").write_text("\n".join(run_lines) + "\n")
        if epoch - best_epoch >= args.early_stop_patience:
            run_lines.append(f"early_stop epoch={epoch} best_epoch={best_epoch}")
            (args.output_root / "run.log").write_text("\n".join(run_lines) + "\n")
            break

    best = torch.load(args.output_root / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    unwrap(model).load_state_dict(best["model_state_dict"])

    sel_by_thr, predictions = eval_selected_groups(model, clips, args, device, SWEEP_THRESHOLDS)
    sel = sel_by_thr[EVAL_THRESHOLD]
    frame_rows = selected_frame_rows(clips, predictions, EVAL_THRESHOLD, args)
    fg_final: dict[str, dict[str, Any]] = {}
    for split in ("val", "test"):
        rows, _ = evaluate(model, eval_loaders[split], args, device, split, per_sequence=False)
        fg_final[split] = row_at(rows, EVAL_THRESHOLD)

    write_csv_union(args.output_root / "train_log.csv", train_rows)
    write_csv_union(args.output_root / "selected_oracle_metrics.csv", frame_rows)
    write_csv(args.output_root / "selected_pathological_metrics.csv", [{"threshold": t, **sel_by_thr[t]["patho"]} for t in SWEEP_THRESHOLDS])
    write_csv(args.output_root / "selected_clean_metrics.csv", [{"threshold": t, **sel_by_thr[t]["clean"]} for t in SWEEP_THRESHOLDS])
    write_csv(args.output_root / "full_gt_val_metrics.csv", [fg_final["val"]])
    write_csv(args.output_root / "full_gt_test_metrics.csv", [fg_final["test"]])
    write_csv(args.output_root / "threshold_sweep_after_training.csv", [{"threshold": t, **{f"all_{k}": v for k, v in sel_by_thr[t]["all"].items()}, **{f"patho_{k}": v for k, v in sel_by_thr[t]["patho"].items()}, **{f"clean_{k}": v for k, v in sel_by_thr[t]["clean"].items()}} for t in SWEEP_THRESHOLDS])

    for clip in patho_clips + clean_clips[:1]:
        p_bad, residual = predictions[clip.clip_id]
        hard = (p_bad >= EVAL_THRESHOLD).astype(np.float32)
        refined = clip.raws + hard * residual
        for i in np.linspace(0, len(clip.frame_ids) - 1, min(args.diagnostics_per_clip, len(clip.frame_ids)), dtype=int):
            diagnostic(
                args.output_root / "diagnostics" / f"{clip.clip_id}_{clip.frame_ids[i]}.png",
                clip.raws[i], refined[i], clip.gts[i], clip.oracle[i],
                clip.sav[i] if clip.sav is not None else None,
                clip.valids[i], p_bad[i], hard[i],
            )

    v32c = BASELINES["v3.2c"]
    success = {
        "patho_new_bad3_below_8pct": bool(sel["patho"]["new_bad3_frame_mean_pct"] < 8.0),
        "patho_new_bad3_below_v32c": bool(sel["patho"]["new_bad3_frame_mean_pct"] < v32c["patho_new_bad3"]),
        "gap_at_least_6_5pct": bool(sel["all"]["oracle_gap_recovered_pct"] >= 6.5),
        "selected_mae_at_most_11_03": bool(sel["all"]["refined_mae"] <= 11.03),
        "clean_not_harmed": bool(sel["clean"]["new_bad3_frame_mean_pct"] <= 1.2 and sel["clean"]["refined_mae"] <= sel_by_thr[EVAL_THRESHOLD]["clean"]["raw_mae"] + 0.05),
        "full_gt_test_beats_raw": bool(fg_final["test"]["refined_hard_mae"] < fg_final["test"]["raw_mae"]),
    }
    summary = {
        "base_checkpoint": str(args.base_checkpoint),
        "output_root": str(args.output_root),
        "params": params,
        "best_epoch": best["epoch"],
        "threshold": EVAL_THRESHOLD,
        "elapsed_seconds": time.perf_counter() - start,
        "hard_neg_pixels_mined": hn_pixels,
        "hard_pos_pixels_mined": hp_pixels,
        "selected_all": sel["all"],
        "selected_pathological": sel["patho"],
        "selected_clean": sel["clean"],
        "full_gt_val": fg_final["val"],
        "full_gt_test": fg_final["test"],
        "baselines": BASELINES,
        "success_criteria": success,
        "no_teacher_inference": True,
    }
    (args.output_root / "aggregate_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")

    (args.output_root / "README.md").write_text(f"""# Tiny Refiner v3.3b: Hard-Negative Residual-Only Fine-Tune

Starts from v3.2c (`{args.base_checkpoint}`), detector and trunk frozen, residual head
only. Batch mix: `{args.full_gt_batch_ratio:.0%}` full-GT, `{args.oracle_batch_ratio:.0%}` oracle crops
(4 clean clips), `{args.hard_negative_batch_ratio:.0%}` hard-negative crops (2 pathological clips).
Hard negatives mined from frozen v3.2c behavior: `{hn_pixels}` raw-good->refined-Bad3 pixels
with p_bad>=0.7; `{hp_pixels}` oracle-beneficial hard positives kept under oracle supervision.
Threshold fixed at `0.7`. No S2M2/SAV/RAFT/DINO inference.

## Result (selected clips, frame-mean, threshold 0.7)

| Metric | v3.1 | v3.2c | v3.3 thr-only | v3.3b |
|---|---:|---:|---:|---:|
| Selected MAE | 11.0421 | 11.0054 | 11.1062 | **{sel['all']['refined_mae']:.4f}** |
| Oracle gap | 6.22% | 7.03% | 4.80% | **{sel['all']['oracle_gap_recovered_pct']:.2f}%** |
| New-Bad3 frame-mean | 4.79% | 5.36% | 2.63% | **{sel['all']['new_bad3_frame_mean_pct']:.2f}%** |
| Patho 2-clip new-Bad3 | — | 15.77% | 6.69% | **{sel['patho']['new_bad3_frame_mean_pct']:.2f}%** |
| Clean 4-clip new-Bad3 | — | 0.89% | 0.89% | **{sel['clean']['new_bad3_frame_mean_pct']:.2f}%** |
| Modified pixels | 52.27% | 18.43% | — | **{sel['all']['modified_pct']:.2f}%** |

Full-GT test: raw `{fg_final['test']['raw_mae']:.4f}` -> refined `{fg_final['test']['refined_hard_mae']:.4f}`,
Bad-3 `{fg_final['test']['raw_bad3']:.3f}` -> `{fg_final['test']['refined_hard_bad3']:.3f}`.

Success criteria: `{json.dumps(success)}`

Best epoch: `{best['epoch']}`. Threshold sweep after training in
`threshold_sweep_after_training.csv`; per-frame metrics in `selected_oracle_metrics.csv`.
""")
    print(json.dumps(summary["success_criteria"], indent=2))
    print(json.dumps({"selected_all": sel["all"], "patho": sel["patho"], "clean": sel["clean"]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
