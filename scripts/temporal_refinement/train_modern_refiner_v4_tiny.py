#!/usr/bin/env python3
"""Train v4_tiny modern refiner: staged detector -> residual+damping -> hybrid oracle.

Stage 1: detector pretraining on full-GT crops (residual/damping heads frozen at
identity). Stage 2: residual + damping training on full-GT crops with the bad head at
low LR. Stage 3: hybrid fine-tune — 50% full-GT, 25% oracle crops (4 clean clips),
25% hard-negative crops (2 pathological clips), with explicit damping supervision:
damping pushed toward 0 on mined hard negatives, free on oracle-beneficial positives.

refined = raw + p_bad * residual, where the v4 residual output already includes
damping * scale * tanh(.). Threshold fixed at 0.7. No S2M2/SAV/RAFT/DINO inference.
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
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "models"))
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
    BalancedCropDataset,
    FullFrameDataset,
    evaluate,
    focal_bce,
    load_samples_with_split,
    set_requires_grad,
    unwrap,
    write_csv_union,
)
from train_tiny_refiner_v3_2_hybrid_oracle import (  # noqa: E402
    OracleCropDataset,
    load_clips,
    make_loader,
    row_at,
    selected_frame_rows,
)
from train_tiny_refiner_v3_3b_hard_negative import (  # noqa: E402
    HardNegativeCropDataset,
    eval_selected_groups,
    mine_hard_masks,
    score_epoch,
)
from calibrate_v3_3_failure_mode_thresholds import PATHOLOGICAL_MODES  # noqa: E402
from evaluate_v3_1_on_selected_oracle_clips import predict_clip  # noqa: E402
from modern_refiner_v4 import v4_tiny  # noqa: E402


DEFAULT_ORACLE_TARGETS = Path("results/03_temporal_refinement/evaluation/distillation_targets_selected_clips")
DEFAULT_OUTPUT_ROOT = Path("results/03_temporal_refinement/training/modern_refiner_v4_tiny")
EVAL_THRESHOLD = 0.7
SWEEP_THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95, 1.01)
BASELINES = {
    "v3.1": {"selected_mae": 11.0421, "gap_pct": 6.22, "new_bad3_fm": 4.79, "modified": 52.27},
    "v3.2c": {"selected_mae": 11.0054, "gap_pct": 7.03, "patho_new_bad3": 15.77, "clean_new_bad3": 0.89, "full_gt_test_mae": 4.6145},
    "v3.3_threshold_only": {"selected_mae": 11.1062, "gap_pct": 4.80, "patho_new_bad3": 6.69},
    "v3.3b": {"selected_mae": 11.0059, "gap_pct": 7.02, "patho_new_bad3": 15.25},
}


class V4EvalAdapter(nn.Module):
    """Exposes the v3 3-tuple interface (bad_logit, p_bad, residual) over the v4 model."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, residual_scale: float):
        return self.model(x, residual_scale)[:3]


def v4_forward(model: nn.Module, x: torch.Tensor, scale: float):
    out = model(x, scale)
    return out  # (bad_logit, p_bad, residual_damped, damping)


def detector_loss_batch(model: nn.Module, batch: dict[str, torch.Tensor], args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    x = batch["x"].to(device, non_blocking=True)
    valid = batch["valid"].to(device, non_blocking=True)
    raw_bad = batch["raw_bad"].to(device, non_blocking=True) * valid
    bad_logit, p_bad, _res, _damp = v4_forward(model, x, args.residual_scale)
    loss = focal_bce(bad_logit, raw_bad, valid, args.focal_gamma)
    return loss, {"det_loss": float(loss.detach().cpu()), "p_bad_mean": float(masked_mean(p_bad, valid).detach().cpu())}


def residual_loss_batch(model: nn.Module, batch: dict[str, torch.Tensor], args: argparse.Namespace, device: torch.device, source: str) -> tuple[torch.Tensor, dict[str, float]]:
    x = batch["x"].to(device, non_blocking=True)
    raw = batch["raw"].to(device, non_blocking=True)
    gt = batch["gt"].to(device, non_blocking=True)
    delta = batch["delta"].to(device, non_blocking=True)
    valid = batch["valid"].to(device, non_blocking=True)
    raw_bad = batch["raw_bad"].to(device, non_blocking=True) * valid
    raw_good = batch["raw_good"].to(device, non_blocking=True) * valid
    sup = batch["sup"].to(device, non_blocking=True) * valid if "sup" in batch else raw_bad
    bad_logit, p_bad, residual, damping = v4_forward(model, x, args.residual_scale)
    refined = raw + p_bad * residual
    raw_err = torch.abs(raw - gt)
    zero = residual.sum() * 0.0
    full_loss = masked_mean(torch.clamp(charbonnier(refined - gt), max=args.robust_loss_clip_px), valid)
    det_loss = focal_bce(bad_logit, raw_bad, valid, args.focal_gamma)
    target = delta.clamp(-args.residual_scale, args.residual_scale)
    res_loss = masked_mean(charbonnier(residual - target), sup) if float(sup.sum()) > 0 else zero
    preserve = masked_mean(torch.abs(p_bad * residual), raw_good) if float(raw_good.sum()) > 0 else zero
    below3 = valid * (raw_err < args.bad_threshold_px).float()
    new_bad3 = masked_mean(torch.relu(torch.abs(refined - gt) - args.bad_threshold_px), below3) if float(below3.sum()) > 0 else zero
    damp_good = masked_mean(damping, raw_good) if float(raw_good.sum()) > 0 else zero
    loss = (
        args.full_weight * full_loss
        + args.detector_weight * det_loss
        + args.residual_weight * res_loss
        + args.preserve_weight * preserve
        + args.new_bad3_weight * new_bad3
        + args.damping_good_weight * damp_good
    )
    return loss, {
        f"{source}_loss": float(loss.detach().cpu()),
        f"{source}_full": float(full_loss.detach().cpu()),
        f"{source}_res": float(res_loss.detach().cpu()),
        f"{source}_preserve": float(preserve.detach().cpu()),
        f"{source}_nb3": float(new_bad3.detach().cpu()),
        f"{source}_damp_good": float(damp_good.detach().cpu()),
    }


def hardneg_loss_batch(model: nn.Module, batch: dict[str, torch.Tensor], args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    x = batch["x"].to(device, non_blocking=True)
    raw = batch["raw"].to(device, non_blocking=True)
    gt = batch["gt"].to(device, non_blocking=True)
    delta = batch["delta"].to(device, non_blocking=True)
    valid = batch["valid"].to(device, non_blocking=True)
    hard_neg = batch["hard_neg"].to(device, non_blocking=True) * valid
    hard_pos = batch["hard_pos"].to(device, non_blocking=True) * valid
    below3 = batch["below3"].to(device, non_blocking=True) * valid
    _logit, p_bad, residual, damping = v4_forward(model, x, args.residual_scale)
    refined = raw + p_bad * residual
    zero = residual.sum() * 0.0
    target = delta.clamp(-args.residual_scale, args.residual_scale)
    pos_loss = masked_mean(charbonnier(residual - target), hard_pos) if float(hard_pos.sum()) > 0 else zero
    hn_preserve = masked_mean(torch.abs(refined - raw), hard_neg) if float(hard_neg.sum()) > 0 else zero
    # explicit damping supervision: close the aggressiveness valve on hard negatives
    damp_neg = masked_mean(damping, hard_neg) if float(hard_neg.sum()) > 0 else zero
    nb3 = masked_mean(torch.relu(torch.abs(refined - gt) - 3.0), below3) if float(below3.sum()) > 0 else zero
    anchor = masked_mean(torch.clamp(charbonnier(refined - gt), max=args.robust_loss_clip_px), valid)
    loss = (
        args.oracle_positive_weight * pos_loss
        + args.hard_negative_weight * hn_preserve
        + args.damping_neg_weight * damp_neg
        + args.new_bad3_weight * nb3
        + args.full_weight * anchor
    )
    return loss, {
        "hardneg_loss": float(loss.detach().cpu()),
        "hardneg_pos": float(pos_loss.detach().cpu()),
        "hardneg_preserve": float(hn_preserve.detach().cpu()),
        "hardneg_damp_neg": float(damp_neg.detach().cpu()),
        "hardneg_nb3": float(nb3.detach().cpu()),
        "damping_hard_neg_mean": float(masked_mean(damping, hard_neg).detach().cpu()) if float(hard_neg.sum()) > 0 else float("nan"),
        "damping_hard_pos_mean": float(masked_mean(damping, hard_pos).detach().cpu()) if float(hard_pos.sum()) > 0 else float("nan"),
    }


def train_one_epoch(model: nn.Module, loaders: dict[str, DataLoader], optimizer: torch.optim.Optimizer, args: argparse.Namespace, device: torch.device, rng: random.Random, stage: int) -> dict[str, float]:
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
        if stage == 1:
            loss, metrics = detector_loss_batch(model, batch, args, device)
        elif source == "hardneg":
            loss, metrics = hardneg_loss_batch(model, batch, args, device)
        else:
            loss, metrics = residual_loss_batch(model, batch, args, device, source)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        rows.append(metrics)
    keys = {k for r in rows for k in r}
    return {k: finite_mean([r[k] for r in rows if k in r]) for k in sorted(keys)}


def make_optimizer(model: nn.Module, args: argparse.Namespace, stage: int) -> torch.optim.Optimizer:
    m = unwrap(model)
    heads_res = list(m.residual_head.parameters()) + list(m.damping_head.parameters())
    body = [p for name, p in m.named_parameters() if not name.startswith(("bad_head", "residual_head", "damping_head"))]
    if stage == 1:
        set_requires_grad(m, True)
        for p in heads_res:
            p.requires_grad = False
        return torch.optim.AdamW([
            {"params": body, "lr": args.lr},
            {"params": list(m.bad_head.parameters()), "lr": args.lr},
        ])
    for p in m.parameters():
        p.requires_grad = True
    lr = args.lr if stage == 2 else args.stage3_lr
    return torch.optim.AdamW([
        {"params": body, "lr": lr},
        {"params": heads_res, "lr": lr},
        {"params": list(m.bad_head.parameters()), "lr": args.detector_lr},
    ])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets-root", type=Path, default=DEFAULT_TARGETS_ROOT)
    p.add_argument("--oracle-targets-root", type=Path, default=DEFAULT_ORACLE_TARGETS)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--balanced-split-json", type=Path, default=DEFAULT_BALANCED_SPLIT)
    p.add_argument("--context-frames", type=int, default=4)
    p.add_argument("--crop-size", type=int, default=96)
    p.add_argument("--crop-candidate-tries", type=int, default=12)
    p.add_argument("--crops-per-epoch", type=int, default=60000)
    p.add_argument("--stage1-epochs", type=int, default=6)
    p.add_argument("--stage2-epochs", type=int, default=10)
    p.add_argument("--stage3-epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--full-gt-batch-ratio", type=float, default=0.50)
    p.add_argument("--oracle-batch-ratio", type=float, default=0.25)
    p.add_argument("--hard-negative-batch-ratio", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--stage3-lr", type=float, default=1e-4)
    p.add_argument("--detector-lr", type=float, default=5e-5)
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
    p.add_argument("--detector-weight", type=float, default=0.2)
    p.add_argument("--residual-weight", type=float, default=0.5)
    p.add_argument("--preserve-weight", type=float, default=1.0)
    p.add_argument("--new-bad3-weight", type=float, default=2.0)
    p.add_argument("--damping-good-weight", type=float, default=0.1)
    p.add_argument("--damping-neg-weight", type=float, default=1.0)
    p.add_argument("--hard-negative-weight", type=float, default=4.0)
    p.add_argument("--oracle-positive-weight", type=float, default=1.0)
    p.add_argument("--sparsity-weight", type=float, default=0.02)
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


def save_ckpt(path: Path, model: nn.Module, args: argparse.Namespace, splits: dict, epoch: int, stage: int, extra: dict[str, Any]) -> None:
    torch.save({
        "model_state_dict": unwrap(model).state_dict(),
        "args": vars(args), "splits": splits, "input_channels": args.context_frames * 2 + 8,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "epoch": epoch, "stage": stage, "threshold": EVAL_THRESHOLD, **extra,
    }, path)


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

    input_channels = args.context_frames * 2 + 8
    core = v4_tiny(input_channels, args.residual_scale).to(device)
    model: nn.Module = core
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(core)
    adapter = V4EvalAdapter(model)
    params = sum(p.numel() for p in core.parameters())

    splits, by_split = load_samples_with_split(args.targets_root, args.balanced_split_json, args.max_frames)
    shards = load_shards(by_split["train"] + by_split["val"] + by_split["test"])
    clips = load_clips(args.oracle_targets_root, args)
    patho_clips = [c for c in clips if c.failure_mode in PATHOLOGICAL_MODES]
    clean_clips = [c for c in clips if c.failure_mode not in PATHOLOGICAL_MODES]
    full_ds = {split: FullFrameDataset(samples, shards, args.context_frames) for split, samples in by_split.items()}
    eval_loaders = {split: make_loader(ds, args.eval_batch_size, max(0, args.num_workers // 2), False, args.prefetch_factor) for split, ds in full_ds.items()}

    gt_args_full = argparse.Namespace(**{**vars(args), "crops_per_epoch": args.crops_per_epoch})
    gt_full_loader = make_loader(BalancedCropDataset(by_split["train"], shards, gt_args_full), args.batch_size, args.num_workers, True, args.prefetch_factor)

    run_lines = [
        f"device={device} gpus={torch.cuda.device_count() if device.type == 'cuda' else 0} params={params}",
        f"model=v4_tiny input_channels={input_channels} residual_scale={args.residual_scale}",
        f"frames={{'train': {len(full_ds['train'])}, 'val': {len(full_ds['val'])}, 'test': {len(full_ds['test'])}}}",
        f"clips={len(clips)} patho={len(patho_clips)} clean={len(clean_clips)}",
        f"stages epochs={args.stage1_epochs}/{args.stage2_epochs}/{args.stage3_epochs} batch={args.batch_size} crops={args.crops_per_epoch}",
    ]
    (args.output_root / "run.log").write_text("\n".join(run_lines) + "\n")

    def log(line: str) -> None:
        run_lines.append(line)
        (args.output_root / "run.log").write_text("\n".join(run_lines) + "\n")

    rng = random.Random(555)
    train_rows: list[dict[str, Any]] = []
    stage_summaries: dict[str, Any] = {}

    # ---------- Stage 1: detector pretraining ----------
    optimizer = make_optimizer(model, args, 1)
    for epoch in range(1, args.stage1_epochs + 1):
        metrics = train_one_epoch(model, {"gt": gt_full_loader}, optimizer, args, device, rng, stage=1)
        fg_rows, _ = evaluate(adapter, eval_loaders["val"], args, device, "val", per_sequence=False)
        fg = row_at(fg_rows, EVAL_THRESHOLD)
        train_rows.append({"stage": 1, "epoch": epoch, **metrics, "val_auc": fg["bad_pixel_auc"], "val_ap": fg["bad_pixel_ap"]})
        log(f"stage=1 epoch={epoch} det_loss={metrics['det_loss']:.5f} p_bad={metrics['p_bad_mean']:.4f} val_auc={fg['bad_pixel_auc']:.4f} val_ap={fg['bad_pixel_ap']:.4f}")
    save_ckpt(args.output_root / "checkpoints" / "stage1_detector.pt", model, args, splits, args.stage1_epochs, 1, {"val_auc": fg["bad_pixel_auc"], "val_ap": fg["bad_pixel_ap"]})
    stage_summaries["stage1"] = {"epochs": args.stage1_epochs, "val_auc": fg["bad_pixel_auc"], "val_ap": fg["bad_pixel_ap"]}

    # ---------- Stage 2: residual + damping on full-GT ----------
    optimizer = make_optimizer(model, args, 2)
    best2 = float("inf")
    for epoch in range(1, args.stage2_epochs + 1):
        metrics = train_one_epoch(model, {"gt": gt_full_loader}, optimizer, args, device, rng, stage=2)
        fg_rows, _ = evaluate(adapter, eval_loaders["val"], args, device, "val", per_sequence=False)
        fg = row_at(fg_rows, EVAL_THRESHOLD)
        score = fg["refined_hard_mae"] + 20.0 * max(0.0, fg["refined_hard_mae"] - fg["raw_mae"])
        train_rows.append({"stage": 2, "epoch": epoch, **metrics, "val_raw": fg["raw_mae"], "val_refined": fg["refined_hard_mae"], "val_new_bad3": fg["new_bad3_from_raw_good_pct"], "val_modified": fg["fraction_modified_pct"]})
        if score < best2:
            best2 = score
            save_ckpt(args.output_root / "checkpoints" / "stage2_fullgt.pt", model, args, splits, epoch, 2, {"val_metrics": fg})
        log(f"stage=2 epoch={epoch} val={fg['raw_mae']:.4f}->{fg['refined_hard_mae']:.4f} nb3={fg['new_bad3_from_raw_good_pct']:.3f}% mod={fg['fraction_modified_pct']:.2f}% auc={fg['bad_pixel_auc']:.4f}")
    ck2 = torch.load(args.output_root / "checkpoints" / "stage2_fullgt.pt", map_location=device, weights_only=False)
    unwrap(model).load_state_dict(ck2["model_state_dict"])
    stage_summaries["stage2"] = {"epochs": args.stage2_epochs, "best_epoch": ck2["epoch"], "val_metrics": {k: v for k, v in ck2["val_metrics"].items() if isinstance(v, (int, float, str))}}

    # ---------- Stage 3: hybrid oracle + pathological damping-aware fine-tune ----------
    with torch.no_grad():
        masks = {}
        for clip in patho_clips:
            _, p_bad, residual, _m = predict_clip(adapter, clip.raws, clip.valids, argparse.Namespace(batch_size=32, context_frames=args.context_frames), device, args.residual_scale, EVAL_THRESHOLD)
            masks[clip.clip_id] = mine_hard_masks(clip, p_bad, residual, args)
    hn_px = int(sum(m["hard_neg"].sum() for m in masks.values()))
    hp_px = int(sum(m["hard_pos"].sum() for m in masks.values()))
    log(f"stage=3 mining hard_neg={hn_px} hard_pos={hp_px} (from current stage-2 model)")

    gt_crops = int(round(args.crops_per_epoch * args.full_gt_batch_ratio))
    oracle_crops = int(round(args.crops_per_epoch * args.oracle_batch_ratio))
    hn_crops = args.crops_per_epoch - gt_crops - oracle_crops
    gt_args = argparse.Namespace(**{**vars(args), "crops_per_epoch": gt_crops})
    loaders3 = {
        "gt": make_loader(BalancedCropDataset(by_split["train"], shards, gt_args), args.batch_size, args.num_workers, True, args.prefetch_factor),
        "oracle": make_loader(OracleCropDataset(clean_clips, args, oracle_crops), args.batch_size, max(2, args.num_workers // 4), True, args.prefetch_factor),
        "hardneg": make_loader(HardNegativeCropDataset(patho_clips, masks, args, hn_crops), args.batch_size, max(2, args.num_workers // 4), True, args.prefetch_factor),
    }
    optimizer = make_optimizer(model, args, 3)
    best_score = float("inf")
    best_epoch = 0
    for epoch in range(1, args.stage3_epochs + 1):
        metrics = train_one_epoch(model, loaders3, optimizer, args, device, rng, stage=3)
        sel_by_thr, _ = eval_selected_groups(adapter, clips, argparse.Namespace(batch_size=32, context_frames=args.context_frames, residual_scale=args.residual_scale), device)
        sel = sel_by_thr[EVAL_THRESHOLD]
        fg_rows, _ = evaluate(adapter, eval_loaders["val"], args, device, "val", per_sequence=False)
        fg = row_at(fg_rows, EVAL_THRESHOLD)
        score = score_epoch(sel, fg)
        train_rows.append({
            "stage": 3, "epoch": epoch, "score": score, **metrics,
            "sel_all_mae": sel["all"]["refined_mae"], "sel_all_gap": sel["all"]["oracle_gap_recovered_pct"],
            "sel_patho_new_bad3": sel["patho"]["new_bad3_frame_mean_pct"], "sel_clean_new_bad3": sel["clean"]["new_bad3_frame_mean_pct"],
            "sel_modified": sel["all"]["modified_pct"], "fullgt_val_refined": fg["refined_hard_mae"],
        })
        if score < best_score:
            best_score = score
            best_epoch = epoch
            save_ckpt(args.output_root / "checkpoints" / "best.pt", model, args, splits, epoch, 3, {"selected_metrics": sel, "full_gt_val_metrics": fg})
        log(
            f"stage=3 epoch={epoch} score={score:.4f} sel_mae={sel['all']['refined_mae']:.4f} gap={sel['all']['oracle_gap_recovered_pct']:.2f}% "
            f"patho_nb3={sel['patho']['new_bad3_frame_mean_pct']:.2f}% clean_nb3={sel['clean']['new_bad3_frame_mean_pct']:.2f}% "
            f"mod={sel['all']['modified_pct']:.2f}% damp_hn={metrics.get('damping_hard_neg_mean', float('nan')):.3f} damp_hp={metrics.get('damping_hard_pos_mean', float('nan')):.3f} "
            f"fullgt_val={fg['raw_mae']:.4f}->{fg['refined_hard_mae']:.4f}"
        )
        if epoch - best_epoch >= args.early_stop_patience:
            log(f"early_stop stage=3 epoch={epoch} best_epoch={best_epoch}")
            break
    if best_epoch == 0:
        save_ckpt(args.output_root / "checkpoints" / "best.pt", model, args, splits, 0, 3, {})
    best = torch.load(args.output_root / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    unwrap(model).load_state_dict(best["model_state_dict"])
    stage_summaries["stage3"] = {"epochs_run": epoch, "best_epoch": best_epoch, "hard_neg_pixels": hn_px, "hard_pos_pixels": hp_px}

    # ---------- Final evaluation ----------
    eval_args = argparse.Namespace(batch_size=32, context_frames=args.context_frames, residual_scale=args.residual_scale)
    sel_by_thr, predictions = eval_selected_groups(adapter, clips, eval_args, device, SWEEP_THRESHOLDS)
    sel = sel_by_thr[EVAL_THRESHOLD]
    frame_rows = selected_frame_rows(clips, predictions, EVAL_THRESHOLD, args)
    fg_final = {}
    for split in ("val", "test"):
        rows, _ = evaluate(adapter, eval_loaders[split], args, device, split, per_sequence=False)
        fg_final[split] = row_at(rows, EVAL_THRESHOLD)

    # damping analysis on pathological clips (final model, stage-3 masks)
    from train_tiny_refiner_v3_1_staged_abstention import make_features_from_raws

    damping_rows = []
    with torch.no_grad():
        for clip in patho_clips:
            xs = []
            for i in range(len(clip.frame_ids)):
                ids = [max(0, i - k) for k in range(args.context_frames)]
                xf, _e, _v = make_features_from_raws(clip.raws[ids], clip.valids[ids])
                xs.append(xf)
            damp_maps = []
            for s_ in range(0, len(xs), 32):
                xb = torch.from_numpy(np.stack(xs[s_ : s_ + 32])).to(device)
                _l, _p, _r, d = v4_forward(model, xb, args.residual_scale)
                damp_maps.append(d[:, 0].detach().cpu().numpy())
            damp = np.concatenate(damp_maps)
            hn = masks[clip.clip_id]["hard_neg"]
            hp = masks[clip.clip_id]["hard_pos"]
            valid = clip.valids > 0
            damping_rows.append({
                "clip_id": clip.clip_id,
                "failure_mode": clip.failure_mode,
                "damping_mean_valid": float(damp[valid].mean()),
                "damping_mean_hard_neg": float(damp[hn].mean()) if hn.any() else float("nan"),
                "damping_mean_hard_pos": float(damp[hp].mean()) if hp.any() else float("nan"),
            })
    write_csv(args.output_root / "damping_analysis.csv", damping_rows)

    write_csv_union(args.output_root / "train_log.csv", train_rows)
    write_csv_union(args.output_root / "selected_oracle_metrics.csv", frame_rows)
    write_csv(args.output_root / "selected_pathological_metrics.csv", [{"threshold": t, **sel_by_thr[t]["patho"]} for t in SWEEP_THRESHOLDS])
    write_csv(args.output_root / "selected_clean_metrics.csv", [{"threshold": t, **sel_by_thr[t]["clean"]} for t in SWEEP_THRESHOLDS])
    write_csv(args.output_root / "full_gt_val_metrics.csv", [fg_final["val"]])
    write_csv(args.output_root / "full_gt_test_metrics.csv", [fg_final["test"]])
    write_csv(args.output_root / "threshold_sweep_after_training.csv", [{"threshold": t, **{f"all_{k}": v for k, v in sel_by_thr[t]["all"].items()}, **{f"patho_{k}": v for k, v in sel_by_thr[t]["patho"].items()}, **{f"clean_{k}": v for k, v in sel_by_thr[t]["clean"].items()}} for t in SWEEP_THRESHOLDS])
    (args.output_root / "stage_summaries.json").write_text(json.dumps(stage_summaries, indent=2, default=str) + "\n")

    # quick runtime check (batched fp32, matches benchmark protocol)
    x_bench = torch.randn(32, input_channels, 256, 320, device=device)
    with torch.no_grad():
        for _ in range(10):
            v4_forward(model, x_bench, args.residual_scale)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            v4_forward(model, x_bench, args.residual_scale)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ms_per_frame = 1000.0 * (time.perf_counter() - t0) / (50 * 32)

    v32c = BASELINES["v3.2c"]
    success = {
        "selected_mae_close_to_v32c": bool(sel["all"]["refined_mae"] <= v32c["selected_mae"] + 0.03),
        "gap_at_least_6_8pct": bool(sel["all"]["oracle_gap_recovered_pct"] >= 6.8),
        "patho_new_bad3_below_v32c": bool(sel["patho"]["new_bad3_frame_mean_pct"] < v32c["patho_new_bad3"]),
        "patho_new_bad3_below_8pct": bool(sel["patho"]["new_bad3_frame_mean_pct"] < 8.0),
        "clean_not_harmed": bool(sel["clean"]["new_bad3_frame_mean_pct"] <= 1.2),
        "full_gt_test_beats_raw": bool(fg_final["test"]["refined_hard_mae"] < fg_final["test"]["raw_mae"]),
        "runtime_below_3ms": bool(ms_per_frame < 3.0),
    }
    summary = {
        "output_root": str(args.output_root),
        "model": "v4_tiny",
        "params": params,
        "best_stage3_epoch": best["epoch"],
        "threshold": EVAL_THRESHOLD,
        "elapsed_seconds": time.perf_counter() - start,
        "runtime_ms_per_frame_batched_fp32": round(ms_per_frame, 4),
        "stage_summaries": stage_summaries,
        "selected_all": sel["all"],
        "selected_pathological": sel["patho"],
        "selected_clean": sel["clean"],
        "full_gt_val": fg_final["val"],
        "full_gt_test": fg_final["test"],
        "damping_analysis": damping_rows,
        "baselines": BASELINES,
        "success_criteria": success,
        "no_teacher_inference": True,
    }
    (args.output_root / "aggregate_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")

    (args.output_root / "README.md").write_text(f"""# Modern Refiner v4_tiny: Staged Training

v4_tiny ({params:,} params) trained in 3 stages: detector pretrain (full-GT), residual+damping
(full-GT), hybrid oracle + pathological damping-aware fine-tune (50/25/25 GT/oracle/hard-neg).
Threshold 0.7 throughout. Hard negatives mined from the stage-2 model itself; damping pushed
toward 0 on hard negatives. No S2M2/SAV/RAFT/DINO inference.

## Final comparison (selected clips, frame-mean, threshold 0.7)

| Metric | v3.1 | v3.2c | v3.3 thr | v3.3b | v4_tiny |
|---|---:|---:|---:|---:|---:|
| Selected MAE | 11.0421 | 11.0054 | 11.1062 | 11.0059 | **{sel['all']['refined_mae']:.4f}** |
| Oracle gap | 6.22% | 7.03% | 4.80% | 7.02% | **{sel['all']['oracle_gap_recovered_pct']:.2f}%** |
| Patho new-Bad3 | — | 15.77% | 6.69% | 15.25% | **{sel['patho']['new_bad3_frame_mean_pct']:.2f}%** |
| Clean new-Bad3 | — | 0.89% | 0.89% | 0.85% | **{sel['clean']['new_bad3_frame_mean_pct']:.2f}%** |
| New-Bad3 frame-mean | 4.79% | 5.36% | 2.63% | 5.18% | **{sel['all']['new_bad3_frame_mean_pct']:.2f}%** |
| Modified pixels | 52.27% | 18.43% | — | 18.43% | **{sel['all']['modified_pct']:.2f}%** |

Full-GT test: raw `{fg_final['test']['raw_mae']:.4f}` -> refined `{fg_final['test']['refined_hard_mae']:.4f}`,
Bad-3 `{fg_final['test']['raw_bad3']:.3f}` -> `{fg_final['test']['refined_hard_bad3']:.3f}`, detector AUC `{fg_final['test']['bad_pixel_auc']:.4f}`.
Runtime: `{ms_per_frame:.3f}` ms/frame batched fp32 (target <3).

Damping sanity (`damping_analysis.csv`): mean damping on hard negatives vs hard positives per
pathological clip — lower on negatives = the damping head learned aggressiveness control.

Success criteria: `{json.dumps(success)}`
Stage summaries: `stage_summaries.json`. Threshold sweep: `threshold_sweep_after_training.csv`.
""")
    print(json.dumps(success, indent=2))
    print(json.dumps({"selected_all": sel["all"], "patho": sel["patho"], "clean": sel["clean"], "runtime_ms": ms_per_frame}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
