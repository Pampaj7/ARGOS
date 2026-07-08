#!/usr/bin/env python3
"""Train EGBM-v2-CARE: change-aware reliability encoding over the EGBM-v1 recipe.

Same 3-stage protocol, splits, datasets and safety losses as EGBM-v1
(train_experimental_refiner_vx.py), plus:
  * warm start from the EGBM-v1 best checkpoint (all shape-matching tensors),
  * a predictive-memory loss (z_hat vs stop-grad z) in every stage,
  * light CARE auxiliary losses in stages 2-3 built ONLY from available proxies:
      - artifact_like  ^ on mined hard negatives, v on hard positives
      - boundary_change ^ where spatial edge AND temporal diff are both high
      - real/occlusion  ^ on coherent large temporal changes (pooled dt mask)
      - stable_predictable ^ where temporal diff is small
      - small entropy bonus against class collapse
    Proxy limitations are documented in the README; aux weights are small so
    refinement losses dominate.

Single GPU, no DataParallel, cudnn.benchmark off, expandable_segments recommended
(EGBM-v1 crash lessons). No S2M2/SAV/RAFT/DINO inference.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
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
    make_features_from_raws,
    unwrap,
    write_csv_union,
)
from train_tiny_refiner_v3_2_hybrid_oracle import OracleCropDataset, load_clips, make_loader  # noqa: E402
from train_tiny_refiner_v3_3b_hard_negative import HardNegativeCropDataset, mine_hard_masks  # noqa: E402
from train_experimental_refiner_vx import (  # noqa: E402
    aggregate_frames,
    frame_metrics_egbm,
    full_gt_eval,
    predict_clip_egbm,
    score_epoch,
)
from calibrate_v3_3_failure_mode_thresholds import PATHOLOGICAL_MODES  # noqa: E402
from egbm_v2_care_refiner import CARE_CLASSES, N_CARE, egbm_v2_care, load_v1_warm_start  # noqa: E402


DEFAULT_ORACLE_TARGETS = Path("results/03_temporal_refinement/evaluation/distillation_targets_selected_clips")
DEFAULT_OUTPUT_ROOT = Path("results/03_temporal_refinement/training/egbm_v2_care")
DEFAULT_V1_CHECKPOINT = Path("results/03_temporal_refinement/training/experimental_refiner_vx_training/checkpoints/best.pt")
DISP_SCALE = 64.0
BASELINES = {
    "v3.2c": {"selected_mae": 11.0054, "gap_pct": 7.03, "patho_new_bad3": 15.77, "clean_new_bad3": 0.89, "full_gt_test_mae": 4.6145},
    "v4_tiny": {"selected_mae": 11.0669, "gap_pct": 5.67, "patho_new_bad3": 0.33, "full_gt_test_mae": 4.7763},
    "SOG": {"selected_mae": 11.0909, "gap_pct": 5.14, "patho_new_bad3": 5.77, "full_gt_test_mae": 4.6221},
    "EGBM_v1": {"selected_mae": 10.4032, "gap_pct": 20.37, "patho_new_bad3": 1.30, "clean_new_bad3": 0.81, "full_gt_test_mae": 4.5226, "runtime_ms": 6.25, "flicker_damp_hn": 0.387, "flicker_damp_hp": 0.733},
}


# --------------------------------------------------------------------------------------
# CARE auxiliary losses (proxy supervision only; see module docstring for limitations)
# --------------------------------------------------------------------------------------

def care_proxy_masks(x: torch.Tensor, out_hw: tuple[int, int]) -> dict[str, torch.Tensor]:
    """Low-res proxy masks from the feature stack itself (no invented labels)."""
    dt = x[:, 8:9] * DISP_SCALE       # |raw_t - raw_{t-1}| px
    edge = x[:, 15:16] * DISP_SCALE   # gradient magnitude px
    boundary_change = ((edge > 2.0) & (dt > 1.0)).float()
    coherent_change = (F.avg_pool2d((dt > 2.0).float(), 9, stride=1, padding=4) > 0.6).float()
    stable = (dt < 0.5).float()
    pool = lambda m: F.adaptive_avg_pool2d(m, out_hw)
    return {"boundary_change": pool(boundary_change), "coherent_change": pool(coherent_change), "stable": pool(stable)}


def care_aux_losses(diag: dict[str, torch.Tensor], x: torch.Tensor, args: argparse.Namespace, hard_neg: torch.Tensor | None = None, hard_pos: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    care = diag["care_probs"]  # (B,5,h,w): artifact, real, occl, boundary, stable
    hw = care.shape[-2:]
    zero = care.sum() * 0.0
    # prediction loss (train the forecaster; stop-grad targets so the encoder stays free)
    pred = zero
    for z_hat, z in diag["pred_pairs"]:
        pred = pred + charbonnier(z_hat - z.detach()).mean()
    pred = pred / max(len(diag["pred_pairs"]), 1)
    masks = care_proxy_masks(x, hw)
    art = care[:, 0:1]
    real_occl = care[:, 1:2] + care[:, 2:3]
    boundary = care[:, 3:4]
    stable_p = care[:, 4:5]
    l_boundary = masked_mean(1.0 - boundary, masks["boundary_change"]) if float(masks["boundary_change"].sum()) > 0 else zero
    l_real = masked_mean(1.0 - real_occl, masks["coherent_change"]) if float(masks["coherent_change"].sum()) > 0 else zero
    l_stable = masked_mean(1.0 - stable_p, masks["stable"]) if float(masks["stable"].sum()) > 0 else zero
    l_art = zero
    if hard_neg is not None and float(hard_neg.sum()) > 0:
        hn = F.adaptive_avg_pool2d(hard_neg, hw)
        l_art = l_art + masked_mean(1.0 - art, hn)
    if hard_pos is not None and float(hard_pos.sum()) > 0:
        hp = F.adaptive_avg_pool2d(hard_pos, hw)
        l_art = l_art + masked_mean(art, hp)
    entropy = -(care.clamp_min(1e-8).log() * care).sum(dim=1).mean()
    loss = (
        args.pred_weight * pred
        + args.care_artifact_weight * l_art
        + args.care_boundary_weight * l_boundary
        + args.care_realchange_weight * l_real
        + args.care_stable_weight * l_stable
        - args.care_entropy_weight * entropy
    )
    return loss, {
        "aux_pred": float(pred.detach().cpu()),
        "aux_artifact": float(l_art.detach().cpu()),
        "aux_boundary": float(l_boundary.detach().cpu()),
        "aux_realchange": float(l_real.detach().cpu()),
        "aux_stable": float(l_stable.detach().cpu()),
        "care_entropy": float(entropy.detach().cpu()),
    }


# --------------------------------------------------------------------------------------
# Stage losses (EGBM-v1 semantics + CARE aux)
# --------------------------------------------------------------------------------------

def detector_loss_batch(model, batch, args, device):
    x = batch["x"].to(device, non_blocking=True)
    valid = batch["valid"].to(device, non_blocking=True)
    raw_bad = batch["raw_bad"].to(device, non_blocking=True) * valid
    bad_logit, p_bad, _r, diag = model(x, args.residual_scale)
    det = focal_bce(bad_logit, raw_bad, valid, args.focal_gamma)
    aux, am = care_aux_losses(diag, x, args)
    loss = det + aux
    return loss, {"det_loss": float(det.detach().cpu()), "p_bad_mean": float(masked_mean(p_bad, valid).detach().cpu()), **am}


def residual_loss_batch(model, batch, args, device, source):
    x = batch["x"].to(device, non_blocking=True)
    raw = batch["raw"].to(device, non_blocking=True)
    gt = batch["gt"].to(device, non_blocking=True)
    delta = batch["delta"].to(device, non_blocking=True)
    valid = batch["valid"].to(device, non_blocking=True)
    raw_bad = batch["raw_bad"].to(device, non_blocking=True) * valid
    raw_good = batch["raw_good"].to(device, non_blocking=True) * valid
    sup = batch["sup"].to(device, non_blocking=True) * valid if "sup" in batch else raw_bad
    bad_logit, _p_bad, residual, diag = model(x, args.residual_scale)
    refined = raw + residual
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
    aux, am = care_aux_losses(diag, x, args)
    loss = (
        args.full_weight * full_loss + args.detector_weight * det_loss + args.residual_weight * res_loss
        + args.preserve_weight * preserve + args.new_bad3_weight * new_bad3 + args.damping_good_weight * damp_good
        + aux
    )
    return loss, {
        f"{source}_loss": float(loss.detach().cpu()), f"{source}_full": float(full_loss.detach().cpu()),
        f"{source}_res": float(res_loss.detach().cpu()), f"{source}_preserve": float(preserve.detach().cpu()),
        f"{source}_nb3": float(new_bad3.detach().cpu()), **am,
    }


def hardneg_loss_batch(model, batch, args, device):
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
    damp_neg = masked_mean(diag["damping"], hard_neg) if float(hard_neg.sum()) > 0 else zero
    id_route = masked_mean(1.0 - diag["router_weights"][:, -1:], hard_neg) if float(hard_neg.sum()) > 0 else zero
    nb3 = masked_mean(torch.relu(torch.abs(refined - gt) - 3.0), below3) if float(below3.sum()) > 0 else zero
    anchor = masked_mean(torch.clamp(charbonnier(refined - gt), max=args.robust_loss_clip_px), valid)
    aux, am = care_aux_losses(diag, x, args, hard_neg=hard_neg, hard_pos=hard_pos)
    loss = (
        args.oracle_positive_weight * pos_loss + args.hard_negative_weight * hn_preserve
        + args.damping_neg_weight * damp_neg + args.router_identity_weight * id_route
        + args.new_bad3_weight * nb3 + args.full_weight * anchor + aux
    )
    return loss, {
        "hardneg_loss": float(loss.detach().cpu()), "hardneg_pos": float(pos_loss.detach().cpu()),
        "hardneg_preserve": float(hn_preserve.detach().cpu()),
        "damping_hard_neg_mean": float(masked_mean(diag["damping"], hard_neg).detach().cpu()) if float(hard_neg.sum()) > 0 else float("nan"),
        "damping_hard_pos_mean": float(masked_mean(diag["damping"], hard_pos).detach().cpu()) if float(hard_pos.sum()) > 0 else float("nan"),
        **am,
    }


def train_one_epoch(model, loaders, optimizer, args, device, rng, stage):
    model.train()
    order = [name for name, loader in loaders.items() for _ in range(len(loader))]
    rng.shuffle(order)
    iters = {name: iter(loader) for name, loader in loaders.items()}
    rows = []
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
def eval_selected(model, clips, args, device):
    model.eval()
    eval_args = argparse.Namespace(context_frames=args.context_frames, residual_scale=args.residual_scale, eval_clip_batch=args.eval_clip_batch)
    preds = {c.clip_id: predict_clip_egbm(model, c, eval_args, device)[0] for c in clips}
    out = {}
    for name, group in (("all", clips), ("patho", [c for c in clips if c.failure_mode in PATHOLOGICAL_MODES]), ("clean", [c for c in clips if c.failure_mode not in PATHOLOGICAL_MODES])):
        frames = [f for c in group for f in frame_metrics_egbm(c, preds[c.clip_id])]
        out[name] = aggregate_frames(frames)
    return out, preds


@torch.no_grad()
def care_full_frame_maps(model, clip, args, device):
    """Per-frame low-res care probs / surprise / damping maps for a clip."""
    n = len(clip.frame_ids)
    care_all, sur_all, damp_all = [], [], []
    for s_ in range(0, n, args.eval_clip_batch):
        e_ = min(n, s_ + args.eval_clip_batch)
        xs = []
        for i in range(s_, e_):
            ids = [max(0, i - k) for k in range(args.context_frames)]
            xf, _e2, _v2 = make_features_from_raws(clip.raws[ids], clip.valids[ids])
            xs.append(xf)
        xb = torch.from_numpy(np.stack(xs)).to(device)
        _l, _p, _r, diag = model(xb, args.residual_scale)
        care_all.append(diag["care_probs"].cpu().numpy())
        sur = diag["surprise"].abs().mean(dim=1, keepdim=True)
        sur_all.append(sur.cpu().numpy())
        # damping is full-resolution; pool to the CARE low-res grid so masks line up
        damp_all.append(F.adaptive_avg_pool2d(diag["damping"], sur.shape[-2:]).cpu().numpy())
    return np.concatenate(care_all), np.concatenate(sur_all), np.concatenate(damp_all)


def pool_mask(mask: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    t = torch.from_numpy(mask.astype(np.float32))[:, None]
    return F.adaptive_avg_pool2d(t, hw).numpy()[:, 0]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets-root", type=Path, default=DEFAULT_TARGETS_ROOT)
    p.add_argument("--oracle-targets-root", type=Path, default=DEFAULT_ORACLE_TARGETS)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--balanced-split-json", type=Path, default=DEFAULT_BALANCED_SPLIT)
    p.add_argument("--v1-checkpoint", type=Path, default=DEFAULT_V1_CHECKPOINT)
    p.add_argument("--warm-start", nargs="?", const=True, default=True, type=parse_bool)
    p.add_argument("--context-frames", type=int, default=4)
    p.add_argument("--crop-size", type=int, default=96)
    p.add_argument("--crop-candidate-tries", type=int, default=12)
    p.add_argument("--crops-per-epoch", type=int, default=60000)
    p.add_argument("--stage1-epochs", type=int, default=6)
    p.add_argument("--stage2-epochs", type=int, default=10)
    p.add_argument("--stage3-epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128)
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
    p.add_argument("--pred-weight", type=float, default=0.10)
    p.add_argument("--care-artifact-weight", type=float, default=0.20)
    p.add_argument("--care-boundary-weight", type=float, default=0.05)
    p.add_argument("--care-realchange-weight", type=float, default=0.05)
    p.add_argument("--care-stable-weight", type=float, default=0.05)
    p.add_argument("--care-entropy-weight", type=float, default=0.01)
    p.add_argument("--robust-loss-clip-px", type=float, default=10.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--early-stop-patience", type=int, default=5)
    p.add_argument("--diagnostics-per-clip", type=int, default=2)
    p.add_argument("--fresh", nargs="?", const=True, default=False, type=parse_bool)
    args = p.parse_args()
    total = args.full_gt_batch_ratio + args.oracle_batch_ratio + args.hard_negative_batch_ratio
    args.full_gt_batch_ratio /= total
    args.oracle_batch_ratio /= total
    args.hard_negative_batch_ratio /= total
    return args


def save_ckpt(path, model, args, splits, epoch, stage, extra):
    torch.save({
        "model_state_dict": unwrap(model).state_dict(), "args": vars(args), "splits": splits,
        "input_channels": args.context_frames * 2 + 8, "parameter_count": sum(p.numel() for p in model.parameters()),
        "epoch": epoch, "stage": stage, **extra,
    }, path)


def main() -> int:
    args = parse_args()
    if args.fresh and args.output_root.exists():
        shutil.rmtree(args.output_root)
    (args.output_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (args.output_root / "diagnostics").mkdir(exist_ok=True)
    sys.stdout = (args.output_root / "stdout.log").open("w", buffering=1)
    sys.stderr = (args.output_root / "stderr.log").open("w", buffering=1)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    start = time.perf_counter()

    model = egbm_v2_care(16, args.residual_scale).to(device)
    params = sum(p.numel() for p in model.parameters())
    warm_loaded = warm_total = 0
    if args.warm_start and args.v1_checkpoint.exists():
        warm_loaded, warm_total = load_v1_warm_start(model, str(args.v1_checkpoint))

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
        f"device={device} params={params} model=egbm_v2_care",
        f"warm_start={args.warm_start} v1_ckpt={args.v1_checkpoint} loaded_tensors={warm_loaded}/{warm_total}",
        f"frames={{'train': {len(full_ds['train'])}, 'val': {len(full_ds['val'])}, 'test': {len(full_ds['test'])}}}",
        f"clips={len(clips)} patho={len(patho_clips)} clean={len(clean_clips)}",
        f"stages epochs={args.stage1_epochs}/{args.stage2_epochs}/{args.stage3_epochs} batch={args.batch_size} crops={args.crops_per_epoch}",
        f"aux weights pred={args.pred_weight} art={args.care_artifact_weight} bnd={args.care_boundary_weight} real={args.care_realchange_weight} stable={args.care_stable_weight} ent={args.care_entropy_weight}",
    ]
    (args.output_root / "run.log").write_text("\n".join(run_lines) + "\n")

    def log(line: str) -> None:
        run_lines.append(line)
        (args.output_root / "run.log").write_text("\n".join(run_lines) + "\n")

    rng = random.Random(555)
    train_rows: list[dict[str, Any]] = []
    stage_summaries: dict[str, Any] = {}

    # ---------- Stage 1 ----------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    fg = {}
    for epoch in range(1, args.stage1_epochs + 1):
        metrics = train_one_epoch(model, {"gt": gt_full_loader}, optimizer, args, device, rng, stage=1)
        fg = full_gt_eval(model, eval_loaders["val"], device, args.bad_threshold_px)
        train_rows.append({"stage": 1, "epoch": epoch, **metrics, "val_auc": fg["detector_auc"], "val_ap": fg["detector_ap"]})
        log(f"stage=1 epoch={epoch} det_loss={metrics['det_loss']:.5f} aux_pred={metrics.get('aux_pred', float('nan')):.5f} ent={metrics.get('care_entropy', float('nan')):.3f} val_auc={fg['detector_auc']:.4f} val_ap={fg['detector_ap']:.4f}")
    save_ckpt(args.output_root / "checkpoints" / "stage1_detector.pt", model, args, splits, args.stage1_epochs, 1, {"val_auc": fg.get("detector_auc")})
    stage_summaries["stage1"] = {"epochs": args.stage1_epochs, "val_auc": fg.get("detector_auc"), "val_ap": fg.get("detector_ap")}

    # ---------- Stage 2 ----------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best2 = float("inf")
    for epoch in range(1, args.stage2_epochs + 1):
        metrics = train_one_epoch(model, {"gt": gt_full_loader}, optimizer, args, device, rng, stage=2)
        fg = full_gt_eval(model, eval_loaders["val"], device, args.bad_threshold_px)
        score = fg["refined_mae"] + 20.0 * max(0.0, fg["refined_mae"] - fg["raw_mae"])
        train_rows.append({"stage": 2, "epoch": epoch, **metrics, "val_raw": fg["raw_mae"], "val_refined": fg["refined_mae"], "val_new_bad3": fg["new_bad3_from_raw_good_pct"], "val_modified": fg["modified_pct"]})
        if score < best2:
            best2 = score
            save_ckpt(args.output_root / "checkpoints" / "stage2_fullgt.pt", model, args, splits, epoch, 2, {"val_metrics": fg})
        log(f"stage=2 epoch={epoch} val={fg['raw_mae']:.4f}->{fg['refined_mae']:.4f} nb3={fg['new_bad3_from_raw_good_pct']:.3f}% mod={fg['modified_pct']:.2f}% auc={fg['detector_auc']:.4f} aux_pred={metrics.get('aux_pred', float('nan')):.5f} ent={metrics.get('care_entropy', float('nan')):.3f}")
    ck2 = torch.load(args.output_root / "checkpoints" / "stage2_fullgt.pt", map_location=device, weights_only=False)
    unwrap(model).load_state_dict(ck2["model_state_dict"])
    stage_summaries["stage2"] = {"epochs": args.stage2_epochs, "best_epoch": ck2["epoch"], "val_metrics": ck2["val_metrics"]}

    # ---------- Stage 3 ----------
    eval_args = argparse.Namespace(context_frames=args.context_frames, residual_scale=args.residual_scale, eval_clip_batch=args.eval_clip_batch, batch_size=args.eval_clip_batch)
    with torch.no_grad():
        masks = {}
        for clip in patho_clips:
            refined, p_bad, _diag = predict_clip_egbm(model, clip, eval_args, device)
            masks[clip.clip_id] = mine_hard_masks(clip, p_bad, refined - clip.raws, args)
    hn_px = int(sum(m["hard_neg"].sum() for m in masks.values()))
    hp_px = int(sum(m["hard_pos"].sum() for m in masks.values()))
    log(f"stage=3 mining hard_neg={hn_px} hard_pos={hp_px}")

    gt_crops = int(round(args.crops_per_epoch * args.full_gt_batch_ratio))
    oracle_crops = int(round(args.crops_per_epoch * args.oracle_batch_ratio))
    hn_crops = args.crops_per_epoch - gt_crops - oracle_crops
    gt_args = argparse.Namespace(**{**vars(args), "crops_per_epoch": gt_crops})
    loaders3 = {
        "gt": make_loader(BalancedCropDataset(by_split["train"], shards, gt_args), args.batch_size, args.num_workers, True, args.prefetch_factor),
        "oracle": make_loader(OracleCropDataset(clean_clips, args, oracle_crops), args.batch_size, max(2, args.num_workers // 4), True, args.prefetch_factor),
        "hardneg": make_loader(HardNegativeCropDataset(patho_clips, masks, args, hn_crops), args.batch_size, max(2, args.num_workers // 4), True, args.prefetch_factor),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.stage3_lr)
    best_score = float("inf")
    best_epoch = 0
    epoch = 0
    for epoch in range(1, args.stage3_epochs + 1):
        metrics = train_one_epoch(model, loaders3, optimizer, args, device, rng, stage=3)
        sel, _ = eval_selected(model, clips, args, device)
        fg = full_gt_eval(model, eval_loaders["val"], device, args.bad_threshold_px)
        score = score_epoch(sel, {"raw_mae": fg["raw_mae"], "refined_mae": fg["refined_mae"]})
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
            f"ent={metrics.get('care_entropy', float('nan')):.3f} fullgt_val={fg['raw_mae']:.4f}->{fg['refined_mae']:.4f}"
        )
        if epoch - best_epoch >= args.early_stop_patience:
            log(f"early_stop stage=3 epoch={epoch} best_epoch={best_epoch}")
            break
    if best_epoch == 0:
        save_ckpt(args.output_root / "checkpoints" / "best.pt", model, args, splits, 0, 3, {})
    best = torch.load(args.output_root / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    unwrap(model).load_state_dict(best["model_state_dict"])
    stage_summaries["stage3"] = {"epochs_run": epoch, "best_epoch": best_epoch, "hard_neg_pixels": hn_px, "hard_pos_pixels": hp_px}

    # ---------- Final evaluation + CARE analyses ----------
    sel, preds = eval_selected(model, clips, args, device)
    frame_rows = []
    for c in clips:
        for i, f in enumerate(frame_metrics_egbm(c, preds[c.clip_id])):
            frame_rows.append({"clip_id": c.clip_id, "sequence_id": c.sequence_id, "frame_id": c.frame_ids[i], "dominant_failure_mode": c.failure_mode, **f})
    fg_final = {split: full_gt_eval(model, eval_loaders[split], device, args.bad_threshold_px) for split in ("val", "test")}

    # threshold sweep over the model's base gate threshold
    sweep_rows = []
    orig_thr = float(unwrap(model).base_threshold)
    for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
        unwrap(model).base_threshold.fill_(thr)
        s_thr, _ = eval_selected(model, clips, args, device)
        sweep_rows.append({"base_threshold": thr, **{f"all_{k}": v for k, v in s_thr["all"].items()}, **{f"patho_{k}": v for k, v in s_thr["patho"].items()}, **{f"clean_{k}": v for k, v in s_thr["clean"].items()}})
    unwrap(model).base_threshold.fill_(orig_thr)
    write_csv(args.output_root / "threshold_sweep.csv", sweep_rows)

    # CARE + damping + surprise analyses on all clips
    damping_rows, care_rows, surprise_rows = [], [], []
    for clip in clips:
        care, sur, damp = care_full_frame_maps(model, clip, args, device)
        hw = sur.shape[-2:]
        valid_lr = pool_mask(clip.valids > 0, hw) > 0.5
        dt = np.abs(clip.raws - np.roll(clip.raws, 1, axis=0))
        dt[0] = 0
        row_s: dict[str, Any] = {
            "clip_id": clip.clip_id, "failure_mode": clip.failure_mode,
            "surprise_mean": float(sur[:, 0][valid_lr].mean()),
            "surprise_p95": float(np.percentile(sur[:, 0][valid_lr], 95)),
            "corr_surprise_vs_temporal_diff": float(np.corrcoef(sur[:, 0][valid_lr], pool_mask(dt, hw)[valid_lr])[0, 1]),
        }
        row_c: dict[str, Any] = {"clip_id": clip.clip_id, "failure_mode": clip.failure_mode}
        for k, name in enumerate(CARE_CLASSES):
            row_c[f"{name}_mean"] = float(care[:, k][valid_lr].mean())
        ent = -(np.clip(care, 1e-8, 1) * np.log(np.clip(care, 1e-8, 1))).sum(axis=1)
        row_c["entropy_mean"] = float(ent[valid_lr].mean())
        if clip.clip_id in masks:
            hn_lr = pool_mask(masks[clip.clip_id]["hard_neg"], hw) > 0.05
            hp_lr = pool_mask(masks[clip.clip_id]["hard_pos"], hw) > 0.05
            row_s["surprise_hard_neg"] = float(sur[:, 0][hn_lr & valid_lr].mean()) if (hn_lr & valid_lr).any() else float("nan")
            row_s["surprise_hard_pos"] = float(sur[:, 0][hp_lr & valid_lr].mean()) if (hp_lr & valid_lr).any() else float("nan")
            for k, name in enumerate(CARE_CLASSES):
                row_c[f"{name}_on_hard_neg"] = float(care[:, k][hn_lr & valid_lr].mean()) if (hn_lr & valid_lr).any() else float("nan")
                row_c[f"{name}_on_hard_pos"] = float(care[:, k][hp_lr & valid_lr].mean()) if (hp_lr & valid_lr).any() else float("nan")
            damping_rows.append({
                "clip_id": clip.clip_id, "failure_mode": clip.failure_mode,
                "damping_mean_valid": float(damp[:, 0][valid_lr].mean()),
                "damping_mean_hard_neg": float(damp[:, 0][hn_lr & valid_lr].mean()) if (hn_lr & valid_lr).any() else float("nan"),
                "damping_mean_hard_pos": float(damp[:, 0][hp_lr & valid_lr].mean()) if (hp_lr & valid_lr).any() else float("nan"),
                "v1_flicker_reference_hn_hp": "0.387/0.733" if clip.failure_mode == "high_temporal_flicker" else "",
            })
        care_rows.append(row_c)
        surprise_rows.append(row_s)
    write_csv_union(args.output_root / "care_change_type_analysis.csv", care_rows)
    write_csv_union(args.output_root / "care_surprise_analysis.csv", surprise_rows)
    write_csv_union(args.output_root / "damping_analysis.csv", damping_rows)

    # diagnostics plots
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [r["clip_id"][:22] for r in surprise_rows]
    ax.bar(names, [r["surprise_mean"] for r in surprise_rows])
    ax.set_ylabel("mean surprise")
    ax.set_title("CARE surprise by clip/failure mode")
    plt.xticks(rotation=30, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(args.output_root / "diagnostics" / "surprise_by_clip.png", dpi=120)
    plt.close(fig)
    if damping_rows:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(damping_rows))
        ax.bar(x - 0.2, [r["damping_mean_hard_neg"] for r in damping_rows], width=0.4, label="hard neg")
        ax.bar(x + 0.2, [r["damping_mean_hard_pos"] for r in damping_rows], width=0.4, label="hard pos")
        ax.set_xticks(x)
        ax.set_xticklabels([r["failure_mode"] for r in damping_rows], fontsize=8)
        ax.axhline(0.387, color="red", ls="--", lw=1, label="v1 flicker hn (0.387)")
        ax.set_ylabel("damping")
        ax.set_title("v2-CARE damping separation (patho clips)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output_root / "diagnostics" / "damping_separation.png", dpi=120)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(care_rows))
    for k, name in enumerate(CARE_CLASSES):
        ax.plot(x, [r[f"{name}_mean"] for r in care_rows], marker="o", label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([r["clip_id"][:16] for r in care_rows], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("mean class prob")
    ax.set_title("CARE change-type distribution by clip")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.output_root / "diagnostics" / "care_class_distribution.png", dpi=120)
    plt.close(fig)

    write_csv_union(args.output_root / "train_log.csv", train_rows)
    write_csv_union(args.output_root / "selected_oracle_metrics.csv", frame_rows)
    write_csv(args.output_root / "full_gt_val_metrics.csv", [fg_final["val"]])
    write_csv(args.output_root / "full_gt_test_metrics.csv", [fg_final["test"]])
    write_csv(args.output_root / "pathological_metrics.csv", [sel["patho"]])
    write_csv(args.output_root / "clean_metrics.csv", [sel["clean"]])

    comparison = [
        {"model": "v3.2c", **BASELINES["v3.2c"]},
        {"model": "v4_tiny", **BASELINES["v4_tiny"]},
        {"model": "SOG", **BASELINES["SOG"]},
        {"model": "EGBM_v1", **{k: v for k, v in BASELINES["EGBM_v1"].items() if not k.startswith("flicker")}},
        {"model": "EGBM_v2_CARE", "selected_mae": sel["all"]["refined_mae"], "gap_pct": sel["all"]["oracle_gap_recovered_pct"],
         "patho_new_bad3": sel["patho"]["new_bad3_frame_mean_pct"], "clean_new_bad3": sel["clean"]["new_bad3_frame_mean_pct"],
         "full_gt_test_mae": fg_final["test"]["refined_mae"]},
    ]
    write_csv_union(args.output_root / "final_comparison_table.csv", comparison)
    latex = "\\begin{tabular}{lccccc}\n\\toprule\nModel & Sel. MAE & Oracle gap & Patho nB3 & Clean nB3 & Full-GT test \\\\\n\\midrule\n"
    for r in comparison:
        latex += f"{r['model'].replace('_', '-')} & {r.get('selected_mae', float('nan')):.4f} & {r.get('gap_pct', float('nan')):.2f}\\% & {r.get('patho_new_bad3', float('nan')):.2f}\\% & {r.get('clean_new_bad3', float('nan')) if r.get('clean_new_bad3') is not None else float('nan'):.2f}\\% & {r.get('full_gt_test_mae', float('nan')):.4f} \\\\\n"
    latex += "\\bottomrule\n\\end{tabular}\n"
    (args.output_root / "final_comparison_table_latex.tex").write_text(latex)

    v1 = BASELINES["EGBM_v1"]
    success = {
        "strong_selected_mae_below_v1": bool(sel["all"]["refined_mae"] < v1["selected_mae"]),
        "strong_gap_above_v1": bool(sel["all"]["oracle_gap_recovered_pct"] > v1["gap_pct"]),
        "strong_patho_new_bad3_at_most_1_30": bool(sel["patho"]["new_bad3_frame_mean_pct"] <= 1.30),
        "strong_clean_new_bad3_at_most_1": bool(sel["clean"]["new_bad3_frame_mean_pct"] <= 1.0),
        "strong_full_gt_test_at_most_v1": bool(fg_final["test"]["refined_mae"] <= v1["full_gt_test_mae"]),
        "min_full_gt_test_beats_raw_and_v32c": bool(fg_final["test"]["refined_mae"] < 4.6145),
        "excellent_gap_at_least_25": bool(sel["all"]["oracle_gap_recovered_pct"] >= 25.0),
    }
    flicker_damp = next((r for r in damping_rows if r["failure_mode"] == "high_temporal_flicker"), {})
    summary = {
        "model": "egbm_v2_care",
        "output_root": str(args.output_root),
        "params": params,
        "warm_start_tensors": f"{warm_loaded}/{warm_total}",
        "best_stage3_epoch": best["epoch"],
        "elapsed_seconds": time.perf_counter() - start,
        "stage_summaries": stage_summaries,
        "selected_all": sel["all"],
        "selected_pathological": sel["patho"],
        "selected_clean": sel["clean"],
        "full_gt_val": fg_final["val"],
        "full_gt_test": fg_final["test"],
        "flicker_damping_v2": flicker_damp,
        "flicker_damping_v1_reference": {"hard_neg": 0.387, "hard_pos": 0.733},
        "baselines": BASELINES,
        "success_criteria": success,
        "no_teacher_inference": True,
    }
    (args.output_root / "aggregate_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(success, indent=2))
    print(json.dumps({"selected_all": sel["all"], "patho": sel["patho"], "clean": sel["clean"], "flicker_damping": flicker_damp}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
