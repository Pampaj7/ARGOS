#!/usr/bin/env python3
"""Train the EGBM-Refiner (experimental branch): staged detector -> residual/damping ->
hybrid oracle + pathological hard-negative fine-tune.

EGBM's forward already applies gating/damping internally: refined = raw + residual
(unlike v3/v4 where refined = raw + p_bad * residual). All loss/eval code here is
written against that convention directly -- reusing v3's evaluate()/predict_clip() would
double-gate the output and silently produce wrong numbers, so this script does not call
them. Data loading (BalancedCropDataset, OracleCropDataset, HardNegativeCropDataset) is
reused as-is since it does not depend on the model's gating convention.

No S2M2/SAV/RAFT/DINO inference. Threshold/abstention is entirely internal to the model
(dynamic per-region threshold head + mixture-of-experts identity route), so there is no
external threshold sweep for EGBM -- metrics are reported at the network's own decision.
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
    focal_bce,
    load_samples_with_split,
    unwrap,
    write_csv_union,
)
from train_tiny_refiner_v3_2_hybrid_oracle import OracleCropDataset, load_clips, make_loader  # noqa: E402
from train_tiny_refiner_v3_3b_hard_negative import HardNegativeCropDataset, mine_hard_masks  # noqa: E402
from calibrate_v3_3_failure_mode_thresholds import PATHOLOGICAL_MODES  # noqa: E402
from evaluate_v3_1_on_selected_oracle_clips import auc_ap  # noqa: E402
from experimental_refiner_vx import egbm_refiner  # noqa: E402


DEFAULT_ORACLE_TARGETS = Path("results/03_temporal_refinement/evaluation/distillation_targets_selected_clips")
DEFAULT_OUTPUT_ROOT = Path("results/03_temporal_refinement/training/experimental_refiner_vx_training")
BASELINES = {
    "v3.1": {"selected_mae": 11.0421, "gap_pct": 6.22, "new_bad3_fm": 4.79, "modified": 52.27},
    "v3.2c": {"selected_mae": 11.0054, "gap_pct": 7.03, "patho_new_bad3": 15.77, "clean_new_bad3": 0.89, "full_gt_test_mae": 4.6145},
    "v3.3_threshold_only": {"selected_mae": 11.1062, "gap_pct": 4.80, "patho_new_bad3": 6.69},
    "v3.3b": {"selected_mae": 11.0059, "gap_pct": 7.02, "patho_new_bad3": 15.25},
    "v4_tiny": {"note": "trained separately, see modern_refiner_v4_tiny/aggregate_summary.json"},
}


def detector_loss_batch(model: nn.Module, batch: dict[str, torch.Tensor], args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    x = batch["x"].to(device, non_blocking=True)
    valid = batch["valid"].to(device, non_blocking=True)
    raw_bad = batch["raw_bad"].to(device, non_blocking=True) * valid
    bad_logit, p_bad, _res, _diag = model(x, args.residual_scale)
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
    bad_logit, _p_bad, residual, diag = model(x, args.residual_scale)
    refined = raw + residual  # EGBM convention: gating already applied inside `residual`
    raw_err = torch.abs(raw - gt)
    zero = residual.sum() * 0.0
    full_loss = masked_mean(torch.clamp(charbonnier(refined - gt), max=args.robust_loss_clip_px), valid)
    det_loss = focal_bce(bad_logit, raw_bad, valid, args.focal_gamma)
    target = delta.clamp(-args.residual_scale, args.residual_scale)
    res_loss = masked_mean(charbonnier(residual - target), sup) if float(sup.sum()) > 0 else zero
    preserve = masked_mean(torch.abs(refined - raw), raw_good) if float(raw_good.sum()) > 0 else zero
    below3 = valid * (raw_err < args.bad_threshold_px).float()
    new_bad3 = masked_mean(torch.relu(torch.abs(refined - gt) - args.bad_threshold_px), below3) if float(below3.sum()) > 0 else zero
    damp_good = masked_mean(diag["damping"], raw_good) if float(raw_good.sum()) > 0 else zero
    loss = (
        args.full_weight * full_loss
        + args.detector_weight * det_loss
        + args.residual_weight * res_loss
        + args.preserve_weight * preserve
        + args.new_bad3_weight * new_bad3
        + args.damping_good_weight * damp_good
    )
    return loss, {
        f"{source}_loss": float(loss.detach().cpu()), f"{source}_full": float(full_loss.detach().cpu()),
        f"{source}_res": float(res_loss.detach().cpu()), f"{source}_preserve": float(preserve.detach().cpu()),
        f"{source}_nb3": float(new_bad3.detach().cpu()), f"{source}_damp_good": float(damp_good.detach().cpu()),
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
    _logit, _p_bad, residual, diag = model(x, args.residual_scale)
    refined = raw + residual
    zero = residual.sum() * 0.0
    target = delta.clamp(-args.residual_scale, args.residual_scale)
    pos_loss = masked_mean(charbonnier(residual - target), hard_pos) if float(hard_pos.sum()) > 0 else zero
    hn_preserve = masked_mean(torch.abs(refined - raw), hard_neg) if float(hard_neg.sum()) > 0 else zero
    # damping supervision: explicitly close the valve on mined hard negatives
    damp_neg = masked_mean(diag["damping"], hard_neg) if float(hard_neg.sum()) > 0 else zero
    # router supervision: push identity-expert mass up on hard negatives (last channel)
    id_route = masked_mean(1.0 - diag["router_weights"][:, -1:], hard_neg) if float(hard_neg.sum()) > 0 else zero
    nb3 = masked_mean(torch.relu(torch.abs(refined - gt) - 3.0), below3) if float(below3.sum()) > 0 else zero
    anchor = masked_mean(torch.clamp(charbonnier(refined - gt), max=args.robust_loss_clip_px), valid)
    loss = (
        args.oracle_positive_weight * pos_loss
        + args.hard_negative_weight * hn_preserve
        + args.damping_neg_weight * damp_neg
        + args.router_identity_weight * id_route
        + args.new_bad3_weight * nb3
        + args.full_weight * anchor
    )
    return loss, {
        "hardneg_loss": float(loss.detach().cpu()), "hardneg_pos": float(pos_loss.detach().cpu()),
        "hardneg_preserve": float(hn_preserve.detach().cpu()), "hardneg_damp_neg": float(damp_neg.detach().cpu()),
        "hardneg_id_route": float(id_route.detach().cpu()), "hardneg_nb3": float(nb3.detach().cpu()),
        "damping_hard_neg_mean": float(masked_mean(diag["damping"], hard_neg).detach().cpu()) if float(hard_neg.sum()) > 0 else float("nan"),
        "damping_hard_pos_mean": float(masked_mean(diag["damping"], hard_pos).detach().cpu()) if float(hard_pos.sum()) > 0 else float("nan"),
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


@torch.no_grad()
def full_gt_eval(model: nn.Module, loader: DataLoader, device: torch.device, bad_threshold_px: float, max_auc_pixels: int = 200_000) -> dict[str, float]:
    model.eval()
    n = raw_abs = ref_abs = raw_bad3 = ref_bad3 = new_bad3 = raw_good_n = modified = 0.0
    labels, scores, left = [], [], max_auc_pixels
    for batch in loader:
        x = batch["x"].to(device)
        raw = batch["raw"].to(device)
        gt = batch["gt"].to(device)
        valid = batch["valid"].to(device)
        bad_logit, p_bad, residual, _diag = model(x, 3.0)
        refined = raw + residual
        raw_err = torch.abs(raw - gt)
        ref_err = torch.abs(refined - gt)
        v = valid > 0
        good = v & (raw_err < 1.0)
        rb3 = v & (raw_err > bad_threshold_px)
        fb3 = v & (ref_err > bad_threshold_px)
        n += float(v.sum()); raw_abs += float(raw_err[v].sum()); ref_abs += float(ref_err[v].sum())
        raw_bad3 += float(rb3.sum()); ref_bad3 += float(fb3.sum())
        new_bad3 += float((good & fb3).sum()); raw_good_n += float(good.sum())
        modified += float((torch.abs(residual) > 0.01)[v].sum())
        if left > 0:
            y = (rb3.detach().cpu().numpy().reshape(-1)).astype(np.uint8)
            m = v.detach().cpu().numpy().reshape(-1)
            p = p_bad.detach().cpu().numpy().reshape(-1)
            idx = np.flatnonzero(m)
            if idx.size:
                take = min(left, idx.size, 20_000)
                pick = np.linspace(0, idx.size - 1, take, dtype=np.int64)
                labels.append(y[idx[pick]]); scores.append(p[idx[pick]]); left -= take
    auc, ap = auc_ap(np.concatenate(scores), np.concatenate(labels)) if labels else (float("nan"), float("nan"))
    n = max(n, 1.0)
    return {
        "raw_mae": raw_abs / n, "refined_mae": ref_abs / n,
        "raw_bad3": 100.0 * raw_bad3 / n, "refined_bad3": 100.0 * ref_bad3 / n,
        "new_bad3_from_raw_good_pct": 100.0 * new_bad3 / max(raw_good_n, 1.0),
        "modified_pct": 100.0 * modified / n, "detector_auc": auc, "detector_ap": ap,
    }


@torch.no_grad()
def predict_clip_egbm(model: nn.Module, clip, args: argparse.Namespace, device: torch.device) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    from train_tiny_refiner_v3_1_staged_abstention import make_features_from_raws

    refined_all, p_bad_all, damp_all = [], [], []
    n = len(clip.frame_ids)
    for s in range(0, n, args.eval_clip_batch):
        e = min(n, s + args.eval_clip_batch)
        xs = []
        for i in range(s, e):
            ids = [max(0, i - k) for k in range(args.context_frames)]
            xf, _e2, _v2 = make_features_from_raws(clip.raws[ids], clip.valids[ids])
            xs.append(xf)
        xb = torch.from_numpy(np.stack(xs)).to(device)
        _logit, p_bad, residual, diag = model(xb, args.residual_scale)
        refined_all.append((torch.from_numpy(clip.raws[s:e]).to(device) + residual[:, 0]).cpu().numpy())
        p_bad_all.append(p_bad[:, 0].cpu().numpy())
        damp_all.append(diag["damping"][:, 0].cpu().numpy())
    return np.concatenate(refined_all), np.concatenate(p_bad_all), {"damping": np.concatenate(damp_all)}


def frame_metrics_egbm(clip, refined: np.ndarray) -> list[dict[str, float]]:
    rows = []
    for i in range(len(clip.frame_ids)):
        valid = clip.valids[i] > 0
        n = max(int(valid.sum()), 1)
        raw_err = np.abs(clip.raws[i] - clip.gts[i])
        ref_err = np.abs(refined[i] - clip.gts[i])
        oracle_err = np.abs(clip.oracle[i] - clip.gts[i])
        good = valid & (raw_err < 1.0)
        n_good = max(int(good.sum()), 1)
        modified = valid & (np.abs(refined[i] - clip.raws[i]) > 0.01)
        rows.append({
            "raw_mae": float(raw_err[valid].mean()) if valid.any() else float("nan"),
            "refined_mae": float(ref_err[valid].mean()) if valid.any() else float("nan"),
            "oracle_mae": float(oracle_err[valid].mean()) if valid.any() else float("nan"),
            "raw_bad3": 100.0 * float((raw_err[valid] > 3.0).sum()) / n,
            "refined_bad3": 100.0 * float((ref_err[valid] > 3.0).sum()) / n,
            "new_bad3_pct": 100.0 * float((ref_err[good] >= 3.0).sum()) / n_good,
            "new_bad3_pixels": float((ref_err[good] >= 3.0).sum()),
            "raw_good_pixels": float(good.sum()),
            "modified_pct": 100.0 * float(modified.sum()) / n,
        })
    return rows


def aggregate_frames(frames: list[dict[str, float]]) -> dict[str, float]:
    def fmean(key: str) -> float:
        vals = [f[key] for f in frames if math.isfinite(f[key])]
        return float(np.mean(vals)) if vals else float("nan")

    raw, ref, oracle = fmean("raw_mae"), fmean("refined_mae"), fmean("oracle_mae")
    good = sum(f["raw_good_pixels"] for f in frames)
    return {
        "frames": len(frames), "raw_mae": raw, "refined_mae": ref, "oracle_mae": oracle,
        "oracle_gap_recovered_pct": 100.0 * (raw - ref) / (raw - oracle) if raw > oracle else float("nan"),
        "raw_bad3": fmean("raw_bad3"), "refined_bad3": fmean("refined_bad3"),
        "new_bad3_frame_mean_pct": fmean("new_bad3_pct"),
        "new_bad3_pixel_weighted_pct": 100.0 * sum(f["new_bad3_pixels"] for f in frames) / max(good, 1.0),
        "modified_pct": fmean("modified_pct"),
    }


def score_epoch(sel: dict[str, dict[str, float]], fg: dict[str, float]) -> float:
    return (
        sel["all"]["refined_mae"]
        + 0.02 * max(0.0, sel["patho"]["new_bad3_frame_mean_pct"] - 8.0)
        + 1.0 * max(0.0, sel["clean"]["new_bad3_frame_mean_pct"] - 1.2)
        + 20.0 * max(0.0, fg["refined_mae"] - fg["raw_mae"])
    )


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
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--eval-batch-size", type=int, default=48)
    p.add_argument("--eval-clip-batch", type=int, default=16)
    p.add_argument("--full-gt-batch-ratio", type=float, default=0.50)
    p.add_argument("--oracle-batch-ratio", type=float, default=0.25)
    p.add_argument("--hard-negative-batch-ratio", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--stage3-lr", type=float, default=1e-4)
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
    p.add_argument("--router-identity-weight", type=float, default=0.5)
    p.add_argument("--hard-negative-weight", type=float, default=4.0)
    p.add_argument("--oracle-positive-weight", type=float, default=1.0)
    p.add_argument("--robust-loss-clip-px", type=float, default=10.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--early-stop-patience", type=int, default=5)
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
        "model_state_dict": unwrap(model).state_dict(), "args": vars(args), "splits": splits,
        "input_channels": args.context_frames * 2 + 8, "parameter_count": sum(p.numel() for p in model.parameters()),
        "epoch": epoch, "stage": stage, **extra,
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
    # ponytail: cudnn.benchmark off + single-GPU (no DataParallel) after repeated
    # "illegal memory access" crashes in backward on this model. DataParallel gathering a
    # dict (diagnostics) across replicas combined with varying batch-tail shapes under
    # cudnn autotune looked like the trigger; both are cheap to disable, revisit if
    # multi-GPU throughput is needed later.
    start = time.perf_counter()

    input_channels = args.context_frames * 2 + 8
    core = egbm_refiner(input_channels, args.residual_scale).to(device)
    model: nn.Module = core
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
        f"model=egbm_refiner input_channels={input_channels} residual_scale={args.residual_scale}",
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

    # ---------- Stage 1: detector pretraining (residual/damping/expert/router heads stay zero-init) ----------
    optimizer = torch.optim.AdamW(core.parameters(), lr=args.lr)
    for epoch in range(1, args.stage1_epochs + 1):
        metrics = train_one_epoch(model, {"gt": gt_full_loader}, optimizer, args, device, rng, stage=1)
        fg = full_gt_eval(model, eval_loaders["val"], device, args.bad_threshold_px)
        train_rows.append({"stage": 1, "epoch": epoch, **metrics, "val_auc": fg["detector_auc"], "val_ap": fg["detector_ap"]})
        log(f"stage=1 epoch={epoch} det_loss={metrics['det_loss']:.5f} p_bad={metrics['p_bad_mean']:.4f} val_auc={fg['detector_auc']:.4f} val_ap={fg['detector_ap']:.4f}")
    save_ckpt(args.output_root / "checkpoints" / "stage1_detector.pt", model, args, splits, args.stage1_epochs, 1, {"val_auc": fg["detector_auc"], "val_ap": fg["detector_ap"]})
    stage_summaries["stage1"] = {"epochs": args.stage1_epochs, "val_auc": fg["detector_auc"], "val_ap": fg["detector_ap"]}

    # ---------- Stage 2: residual + damping on full-GT ----------
    optimizer = torch.optim.AdamW(core.parameters(), lr=args.lr)
    best2 = float("inf")
    for epoch in range(1, args.stage2_epochs + 1):
        metrics = train_one_epoch(model, {"gt": gt_full_loader}, optimizer, args, device, rng, stage=2)
        fg = full_gt_eval(model, eval_loaders["val"], device, args.bad_threshold_px)
        score = fg["refined_mae"] + 20.0 * max(0.0, fg["refined_mae"] - fg["raw_mae"])
        train_rows.append({"stage": 2, "epoch": epoch, **metrics, "val_raw": fg["raw_mae"], "val_refined": fg["refined_mae"], "val_new_bad3": fg["new_bad3_from_raw_good_pct"], "val_modified": fg["modified_pct"]})
        if score < best2:
            best2 = score
            save_ckpt(args.output_root / "checkpoints" / "stage2_fullgt.pt", model, args, splits, epoch, 2, {"val_metrics": fg})
        log(f"stage=2 epoch={epoch} val={fg['raw_mae']:.4f}->{fg['refined_mae']:.4f} nb3={fg['new_bad3_from_raw_good_pct']:.3f}% mod={fg['modified_pct']:.2f}% auc={fg['detector_auc']:.4f}")
    ck2 = torch.load(args.output_root / "checkpoints" / "stage2_fullgt.pt", map_location=device, weights_only=False)
    unwrap(model).load_state_dict(ck2["model_state_dict"])
    stage_summaries["stage2"] = {"epochs": args.stage2_epochs, "best_epoch": ck2["epoch"], "val_metrics": ck2["val_metrics"]}

    # ---------- Stage 3: hybrid oracle + pathological damping/router-aware fine-tune ----------
    with torch.no_grad():
        masks = {}
        for clip in patho_clips:
            refined, p_bad, _diag = predict_clip_egbm(model, clip, args, device)
            residual = refined - clip.raws
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
    optimizer = torch.optim.AdamW(core.parameters(), lr=args.stage3_lr)
    best_score = float("inf")
    best_epoch = 0
    epoch = 0
    for epoch in range(1, args.stage3_epochs + 1):
        metrics = train_one_epoch(model, loaders3, optimizer, args, device, rng, stage=3)
        pred = {c.clip_id: predict_clip_egbm(model, c, args, device) for c in clips}
        sel: dict[str, dict[str, float]] = {}
        for name, group in (("all", clips), ("patho", patho_clips), ("clean", clean_clips)):
            frames = [f for c in group for f in frame_metrics_egbm(c, pred[c.clip_id][0])]
            sel[name] = aggregate_frames(frames)
        fg = full_gt_eval(model, eval_loaders["val"], device, args.bad_threshold_px)
        score = score_epoch(sel, fg)
        train_rows.append({
            "stage": 3, "epoch": epoch, "score": score, **metrics,
            "sel_all_mae": sel["all"]["refined_mae"], "sel_all_gap": sel["all"]["oracle_gap_recovered_pct"],
            "sel_patho_new_bad3": sel["patho"]["new_bad3_frame_mean_pct"], "sel_clean_new_bad3": sel["clean"]["new_bad3_frame_mean_pct"],
            "sel_modified": sel["all"]["modified_pct"], "fullgt_val_refined": fg["refined_mae"],
        })
        if score < best_score:
            best_score = score
            best_epoch = epoch
            save_ckpt(args.output_root / "checkpoints" / "best.pt", model, args, splits, epoch, 3, {"selected_metrics": sel, "full_gt_val_metrics": fg})
        log(
            f"stage=3 epoch={epoch} score={score:.4f} sel_mae={sel['all']['refined_mae']:.4f} gap={sel['all']['oracle_gap_recovered_pct']:.2f}% "
            f"patho_nb3={sel['patho']['new_bad3_frame_mean_pct']:.2f}% clean_nb3={sel['clean']['new_bad3_frame_mean_pct']:.2f}% "
            f"mod={sel['all']['modified_pct']:.2f}% damp_hn={metrics.get('damping_hard_neg_mean', float('nan')):.3f} damp_hp={metrics.get('damping_hard_pos_mean', float('nan')):.3f} "
            f"fullgt_val={fg['raw_mae']:.4f}->{fg['refined_mae']:.4f}"
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
    pred = {c.clip_id: predict_clip_egbm(model, c, args, device) for c in clips}
    sel: dict[str, dict[str, float]] = {}
    frame_rows: list[dict[str, Any]] = []
    for name, group in (("all", clips), ("patho", patho_clips), ("clean", clean_clips)):
        frames = []
        for c in group:
            fr = frame_metrics_egbm(c, pred[c.clip_id][0])
            for i, row in enumerate(fr):
                row2 = {"clip_id": c.clip_id, "sequence_id": c.sequence_id, "frame_id": c.frame_ids[i], "dominant_failure_mode": c.failure_mode, **row}
                if name == "all":
                    frame_rows.append(row2)
            frames.extend(fr)
        sel[name] = aggregate_frames(frames)
    fg_final = {}
    for split in ("val", "test"):
        fg_final[split] = full_gt_eval(model, eval_loaders[split], device, args.bad_threshold_px)

    # damping analysis on pathological clips
    damping_rows = []
    for clip in patho_clips:
        _refined, _p, diag = pred[clip.clip_id]
        hn, hp = masks[clip.clip_id]["hard_neg"], masks[clip.clip_id]["hard_pos"]
        valid = clip.valids > 0
        damp = diag["damping"]
        damping_rows.append({
            "clip_id": clip.clip_id, "failure_mode": clip.failure_mode,
            "damping_mean_valid": float(damp[valid].mean()),
            "damping_mean_hard_neg": float(damp[hn].mean()) if hn.any() else float("nan"),
            "damping_mean_hard_pos": float(damp[hp].mean()) if hp.any() else float("nan"),
        })
    write_csv(args.output_root / "damping_analysis.csv", damping_rows)

    write_csv_union(args.output_root / "train_log.csv", train_rows)
    write_csv_union(args.output_root / "selected_oracle_metrics.csv", frame_rows)
    write_csv(args.output_root / "full_gt_val_metrics.csv", [fg_final["val"]])
    write_csv(args.output_root / "full_gt_test_metrics.csv", [fg_final["test"]])

    # runtime check (batched fp32, matches benchmark protocol)
    x_bench = torch.randn(32, input_channels, 256, 320, device=device)
    with torch.no_grad():
        for _ in range(10):
            model(x_bench, args.residual_scale)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            model(x_bench, args.residual_scale)
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
        "full_gt_test_beats_raw": bool(fg_final["test"]["refined_mae"] < fg_final["test"]["raw_mae"]),
        "runtime_below_25ms": bool(ms_per_frame < 25.0),
    }
    summary = {
        "output_root": str(args.output_root), "model": "egbm_refiner", "params": params,
        "best_stage3_epoch": best["epoch"], "elapsed_seconds": time.perf_counter() - start,
        "runtime_ms_per_frame_batched_fp32": round(ms_per_frame, 4),
        "stage_summaries": stage_summaries,
        "selected_all": sel["all"], "selected_pathological": sel["patho"], "selected_clean": sel["clean"],
        "full_gt_val": fg_final["val"], "full_gt_test": fg_final["test"],
        "damping_analysis": damping_rows, "baselines": BASELINES, "success_criteria": success,
        "no_teacher_inference": True,
    }
    (args.output_root / "aggregate_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")

    (args.output_root / "README.md").write_text(f"""# EGBM-Refiner (Experimental): Staged Training Result

Trained from scratch (identity init), same 3-stage recipe as v4_tiny: detector pretrain,
residual+damping on full-GT, hybrid oracle (50% full-GT / 25% oracle-clean / 25%
hard-negative on the 2 pathological clips) with explicit damping and router-identity
supervision on mined hard negatives. `refined = raw + residual` (EGBM's own convention;
gating/damping already applied internally, no external threshold sweep).

## Final comparison (selected clips, frame-mean)

| Metric | v3.1 | v3.2c | v3.3 thr | v3.3b | EGBM |
|---|---:|---:|---:|---:|---:|
| Selected MAE | 11.0421 | 11.0054 | 11.1062 | 11.0059 | **{sel['all']['refined_mae']:.4f}** |
| Oracle gap | 6.22% | 7.03% | 4.80% | 7.02% | **{sel['all']['oracle_gap_recovered_pct']:.2f}%** |
| Patho new-Bad3 | — | 15.77% | 6.69% | 15.25% | **{sel['patho']['new_bad3_frame_mean_pct']:.2f}%** |
| Clean new-Bad3 | — | 0.89% | 0.89% | 0.85% | **{sel['clean']['new_bad3_frame_mean_pct']:.2f}%** |
| New-Bad3 frame-mean | 4.79% | 5.36% | 2.63% | 5.18% | **{sel['all']['new_bad3_frame_mean_pct']:.2f}%** |
| Modified pixels | 52.27% | 18.43% | — | 18.43% | **{sel['all']['modified_pct']:.2f}%** |

Full-GT test: raw `{fg_final['test']['raw_mae']:.4f}` -> refined `{fg_final['test']['refined_mae']:.4f}`,
Bad-3 `{fg_final['test']['raw_bad3']:.3f}` -> `{fg_final['test']['refined_bad3']:.3f}`,
detector AUC `{fg_final['test']['detector_auc']:.4f}`. Runtime: `{ms_per_frame:.3f}` ms/frame batched fp32.

Success criteria: `{json.dumps(success)}`

Damping sanity (`damping_analysis.csv`): mean damping on hard negatives vs hard positives
per pathological clip.
""")
    print(json.dumps(success, indent=2))
    print(json.dumps({"selected_all": sel["all"], "patho": sel["patho"], "clean": sel["clean"], "runtime_ms": ms_per_frame}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
