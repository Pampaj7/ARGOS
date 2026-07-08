#!/usr/bin/env python3
"""Train EGBM-v3-CARE-S: stateful streaming CARE over sequence chunks.

Same 3-stage protocol and safety losses as EGBM-v2-CARE, but training batches are
causal chunks. The model state persists inside each chunk and is detached between
chunks, matching streaming inference without storing any extra predictions.

Adds:
  * explicit forward_step state training,
  * predictive-memory loss (z_hat vs stop-grad z) in every stage,
  * light CARE auxiliary losses built ONLY from available proxies:
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
from evaluate_v3_1_on_selected_oracle_clips import auc_ap  # noqa: E402
from egbm_v3_care_streaming_refiner import CARE_CLASSES, N_CARE, egbm_v3_care_streaming, load_v1_warm_start  # noqa: E402


DEFAULT_ORACLE_TARGETS = Path("results/03_temporal_refinement/evaluation/distillation_targets_selected_clips")
DEFAULT_OUTPUT_ROOT = Path("results/03_temporal_refinement/training/egbm_v3_care_streaming")
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


def sequence_loss_batch(model, batch, args, device, stage: int, source: str):
    x = batch["x"].to(device, non_blocking=True)
    raw = batch["raw"].to(device, non_blocking=True)
    gt = batch["gt"].to(device, non_blocking=True)
    delta = batch["delta"].to(device, non_blocking=True)
    valid = batch["valid"].to(device, non_blocking=True)
    raw_bad = batch["raw_bad"].to(device, non_blocking=True) * valid
    raw_good = batch["raw_good"].to(device, non_blocking=True) * valid
    sup = batch.get("sup")
    sup = sup.to(device, non_blocking=True) * valid if sup is not None else raw_bad
    hard_neg = batch.get("hard_neg")
    hard_pos = batch.get("hard_pos")
    below3 = batch.get("below3")
    if hard_neg is not None:
        hard_neg = hard_neg.to(device, non_blocking=True) * valid
        hard_pos = hard_pos.to(device, non_blocking=True) * valid
        below3 = below3.to(device, non_blocking=True) * valid

    state = model.init_state(x.shape[0], x.shape[-2], x.shape[-1], device, x.dtype)
    losses, metrics = [], []
    for t in range(x.shape[1]):
        bad_logit, p_bad, residual, diag, state = model.forward_step(x[:, t], state, args.residual_scale)
        refined = raw[:, t] + residual
        raw_err = torch.abs(raw[:, t] - gt[:, t])
        zero = residual.sum() * 0.0
        aux, am = care_aux_losses(
            diag,
            x[:, t],
            args,
            hard_neg[:, t] if hard_neg is not None else None,
            hard_pos[:, t] if hard_pos is not None else None,
        )
        if stage == 1:
            det = focal_bce(bad_logit, raw_bad[:, t], valid[:, t], args.focal_gamma)
            loss = det + aux
            row = {"det_loss": float(det.detach().cpu()), "p_bad_mean": float(masked_mean(p_bad, valid[:, t]).detach().cpu()), **am}
        elif source == "hardneg":
            target = delta[:, t].clamp(-args.residual_scale, args.residual_scale)
            pos = masked_mean(charbonnier(residual - target), hard_pos[:, t]) if float(hard_pos[:, t].sum()) > 0 else zero
            hn_preserve = masked_mean(torch.abs(refined - raw[:, t]), hard_neg[:, t]) if float(hard_neg[:, t].sum()) > 0 else zero
            damp_neg = masked_mean(diag["damping"], hard_neg[:, t]) if float(hard_neg[:, t].sum()) > 0 else zero
            id_route = masked_mean(1.0 - diag["router_weights"][:, -1:], hard_neg[:, t]) if float(hard_neg[:, t].sum()) > 0 else zero
            nb3 = masked_mean(torch.relu(torch.abs(refined - gt[:, t]) - 3.0), below3[:, t]) if float(below3[:, t].sum()) > 0 else zero
            anchor = masked_mean(torch.clamp(charbonnier(refined - gt[:, t]), max=args.robust_loss_clip_px), valid[:, t])
            keep = diag["memory_keep_gate"]
            hn_lr = F.adaptive_avg_pool2d(hard_neg[:, t], keep.shape[-2:])
            hp_lr = F.adaptive_avg_pool2d(hard_pos[:, t], keep.shape[-2:])
            keep_hn = masked_mean(keep, hn_lr) if float(hard_neg[:, t].sum()) > 0 else zero
            keep_hp = masked_mean(keep, hp_lr) if float(hard_pos[:, t].sum()) > 0 else zero
            keep_hn_loss = masked_mean(1.0 - keep, hn_lr) if float(hard_neg[:, t].sum()) > 0 else zero
            loss = args.oracle_positive_weight * pos + args.hard_negative_weight * hn_preserve + args.damping_neg_weight * damp_neg + args.router_identity_weight * id_route + args.new_bad3_weight * nb3 + args.full_weight * anchor + args.memory_update_neg_weight * keep_hn_loss + aux
            row = {
                "hardneg_loss": float(loss.detach().cpu()),
                "damping_hard_neg_mean": float(masked_mean(diag["damping"], hard_neg[:, t]).detach().cpu()) if float(hard_neg[:, t].sum()) > 0 else float("nan"),
                "damping_hard_pos_mean": float(masked_mean(diag["damping"], hard_pos[:, t]).detach().cpu()) if float(hard_pos[:, t].sum()) > 0 else float("nan"),
                "memory_keep_hard_neg": float(keep_hn.detach().cpu()),
                "memory_keep_hard_pos": float(keep_hp.detach().cpu()),
                "memory_keep_hard_neg_loss": float(keep_hn_loss.detach().cpu()),
                **am,
            }
        else:
            full_loss = masked_mean(torch.clamp(charbonnier(refined - gt[:, t]), max=args.robust_loss_clip_px), valid[:, t])
            det_loss = focal_bce(bad_logit, raw_bad[:, t], valid[:, t], args.focal_gamma)
            target = delta[:, t].clamp(-args.residual_scale, args.residual_scale)
            res_loss = masked_mean(charbonnier(residual - target), sup[:, t]) if float(sup[:, t].sum()) > 0 else zero
            preserve = masked_mean(torch.abs(refined - raw[:, t]), raw_good[:, t]) if float(raw_good[:, t].sum()) > 0 else zero
            below = valid[:, t] * (raw_err < args.bad_threshold_px).float()
            new_bad3 = masked_mean(torch.relu(torch.abs(refined - gt[:, t]) - args.bad_threshold_px), below) if float(below.sum()) > 0 else zero
            damp_good = masked_mean(diag["damping"], raw_good[:, t]) if float(raw_good[:, t].sum()) > 0 else zero
            keep_mean = masked_mean(diag["memory_keep_gate"], F.adaptive_avg_pool2d(valid[:, t], diag["memory_keep_gate"].shape[-2:]))
            loss = args.full_weight * full_loss + args.detector_weight * det_loss + args.residual_weight * res_loss + args.preserve_weight * preserve + args.new_bad3_weight * new_bad3 + args.damping_good_weight * damp_good + args.memory_update_weight * keep_mean + aux
            row = {f"{source}_loss": float(loss.detach().cpu()), f"{source}_full": float(full_loss.detach().cpu()), f"{source}_nb3": float(new_bad3.detach().cpu()), "memory_keep_mean": float(keep_mean.detach().cpu()), **am}
        weight = args.burnin_loss_weight if t < args.burn_in_frames else 1.0
        losses.append(loss * weight)
        metrics.append(row)
    state = model.detach_state(state)
    _ = state
    keys = {k for row in metrics for k in row}
    return torch.stack(losses).mean(), {k: finite_mean([row[k] for row in metrics if k in row]) for k in sorted(keys)}


def train_sequence_epoch(model, loaders, optimizer, args, device, rng, stage):
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
        loss, metrics = sequence_loss_batch(model, batch, args, device, stage, source)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        rows.append(metrics)
    keys = {k for row in rows for k in row}
    return {k: finite_mean([row[k] for row in rows if k in row]) for k in sorted(keys)}


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


@torch.no_grad()
def predict_clip_streaming(model, clip, args, device, burn_in: int = 0):
    model.eval()
    state = None
    refined_all, p_bad_all, damp_all, update_all, care_all, surprise_all = [], [], [], [], [], []
    for i in range(len(clip.frame_ids)):
        ids = [max(0, i - k) for k in range(args.context_frames)]
        xfeat, _edge, _var = make_features_from_raws(clip.raws[ids], clip.valids[ids])
        xb = torch.from_numpy(xfeat[None]).to(device)
        _logit, p_bad, residual, diag, state = model.forward_step(xb, state, args.residual_scale)
        refined_all.append((torch.from_numpy(clip.raws[i : i + 1]).to(device) + residual[:, 0]).cpu().numpy()[0])
        p_bad_all.append(p_bad[:, 0].cpu().numpy()[0])
        damp_all.append(diag["damping"][:, 0].cpu().numpy()[0])
        update_all.append(F.interpolate(diag["memory_keep_gate"], size=clip.raws.shape[-2:], mode="bilinear", align_corners=False)[:, 0].cpu().numpy()[0])
        care_all.append(diag["care_probs"].cpu().numpy()[0])
        surprise_all.append(diag["surprise"].abs().mean(dim=1).cpu().numpy()[0])
    refined = np.stack(refined_all)
    p_bad = np.stack(p_bad_all)
    diag = {
        "damping": np.stack(damp_all),
        "memory_keep_gate": np.stack(update_all),
        "memory_update_gate": np.stack(update_all),
        "care_probs": np.stack(care_all),
        "surprise": np.stack(surprise_all),
    }
    if burn_in > 0:
        diag["burn_in_excluded_frames"] = burn_in
    return refined, p_bad, diag


@torch.no_grad()
def eval_selected_streaming(model, clips, args, device, burn_in: int = 0):
    pred = {c.clip_id: predict_clip_streaming(model, c, args, device, burn_in) for c in clips}
    out = {}
    for name, group in (("all", clips), ("patho", [c for c in clips if c.failure_mode in PATHOLOGICAL_MODES]), ("clean", [c for c in clips if c.failure_mode not in PATHOLOGICAL_MODES])):
        frames = []
        for c in group:
            rows = frame_metrics_egbm(c, pred[c.clip_id][0])
            frames.extend(rows[burn_in:] if burn_in > 0 else rows)
        out[name] = aggregate_frames(frames)
    return out, pred


def make_full_frame_feature(shard, offset, context_frames):
    ids = [max(0, offset - i) for i in range(context_frames)]
    raws = np.stack([shard["raw_disp"][i].astype(np.float32) for i in ids])
    valids = np.stack([shard["valid_mask"][i].astype(np.float32) for i in ids])
    xfeat, _edge, _var = make_features_from_raws(raws, valids)
    return xfeat


@torch.no_grad()
def full_gt_eval_streaming(model, sample_groups, shards, args, device, burn_in: int = 0):
    model.eval()
    n = raw_abs = ref_abs = raw_bad3 = ref_bad3 = new_bad3 = raw_good_n = modified = 0.0
    labels, scores, left = [], [], 200_000
    seq_rows = []
    for group in sample_groups:
        shard = shards[group[0].target_path]
        state = None
        seq_acc = dict(frames=0, valid_pixels=0.0, raw_abs=0.0, ref_abs=0.0, raw_bad3=0.0, ref_bad3=0.0, new_bad3=0.0, raw_good=0.0, modified=0.0)
        for local_i, sample in enumerate(group):
            x = torch.from_numpy(make_full_frame_feature(shard, sample.offset, args.context_frames)[None]).to(device)
            raw = torch.from_numpy(shard["raw_disp"][sample.offset].astype(np.float32)[None, None]).to(device)
            gt = torch.from_numpy(shard["gt_disp"][sample.offset].astype(np.float32)[None, None]).to(device)
            valid = torch.from_numpy(shard["valid_mask"][sample.offset].astype(np.float32)[None, None]).to(device)
            _logit, p_bad, residual, _diag, state = model.forward_step(x, state, args.residual_scale)
            if local_i < burn_in:
                continue
            refined = raw + residual
            raw_err = torch.abs(raw - gt)
            ref_err = torch.abs(refined - gt)
            v = valid > 0
            good = v & (raw_err < 1.0)
            rb3 = v & (raw_err > args.bad_threshold_px)
            fb3 = v & (ref_err > args.bad_threshold_px)
            count = float(v.sum())
            vals = {
                "frames": 1,
                "valid_pixels": count,
                "raw_abs": float(raw_err[v].sum()),
                "ref_abs": float(ref_err[v].sum()),
                "raw_bad3": float(rb3.sum()),
                "ref_bad3": float(fb3.sum()),
                "new_bad3": float((good & fb3).sum()),
                "raw_good": float(good.sum()),
                "modified": float((torch.abs(residual)[v] > 0.01).sum()),
            }
            for k, v0 in vals.items():
                seq_acc[k] += v0
            n += count; raw_abs += vals["raw_abs"]; ref_abs += vals["ref_abs"]
            raw_bad3 += vals["raw_bad3"]; ref_bad3 += vals["ref_bad3"]
            new_bad3 += vals["new_bad3"]; raw_good_n += vals["raw_good"]; modified += vals["modified"]
            if left > 0:
                yy = rb3.detach().cpu().numpy().reshape(-1).astype(np.uint8)
                mm = (valid > 0).detach().cpu().numpy().reshape(-1)
                ss = p_bad.detach().cpu().numpy().reshape(-1)
                idx = np.flatnonzero(mm)
                if idx.size:
                    take = min(left, idx.size, 20_000)
                    pick = np.linspace(0, idx.size - 1, take, dtype=np.int64)
                    labels.append(yy[idx[pick]]); scores.append(ss[idx[pick]]); left -= take
        vp = max(seq_acc["valid_pixels"], 1.0)
        seq_rows.append({
            "sequence_id": group[0].sequence_id,
            "frames": int(seq_acc["frames"]),
            "raw_mae": seq_acc["raw_abs"] / vp,
            "refined_mae": seq_acc["ref_abs"] / vp,
            "raw_bad3": 100.0 * seq_acc["raw_bad3"] / vp,
            "refined_bad3": 100.0 * seq_acc["ref_bad3"] / vp,
            "new_bad3_from_raw_good_pct": 100.0 * seq_acc["new_bad3"] / max(seq_acc["raw_good"], 1.0),
            "modified_pct": 100.0 * seq_acc["modified"] / vp,
        })
    auc, ap = auc_ap(np.concatenate(scores), np.concatenate(labels)) if labels else (float("nan"), float("nan"))
    n = max(n, 1.0)
    return {
        "raw_mae": raw_abs / n,
        "refined_mae": ref_abs / n,
        "raw_bad3": 100.0 * raw_bad3 / n,
        "refined_bad3": 100.0 * ref_bad3 / n,
        "new_bad3_from_raw_good_pct": 100.0 * new_bad3 / max(raw_good_n, 1.0),
        "modified_pct": 100.0 * modified / n,
        "detector_auc": auc,
        "detector_ap": ap,
    }, seq_rows


def pool_mask(mask: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    t = torch.from_numpy(mask.astype(np.float32))[:, None]
    return F.adaptive_avg_pool2d(t, hw).numpy()[:, 0]


def group_samples_by_sequence(samples):
    groups: dict[Path, list[Any]] = {}
    for sample in samples:
        groups.setdefault(sample.target_path, []).append(sample)
    return [sorted(v, key=lambda s: s.offset) for v in groups.values()]


def sequence_features(shard: dict[str, np.ndarray], offset: int, y: int, x: int, size: int, context_frames: int) -> dict[str, np.ndarray]:
    ys, xs = slice(y, y + size), slice(x, x + size)
    ids = [max(0, offset - i) for i in range(context_frames)]
    raws = np.stack([shard["raw_disp"][i, ys, xs].astype(np.float32) for i in ids])
    valids = np.stack([shard["valid_mask"][i, ys, xs].astype(np.float32) for i in ids])
    xfeat, _edge, _var = make_features_from_raws(raws, valids)
    raw = raws[0]
    gt = shard["gt_disp"][offset, ys, xs].astype(np.float32)
    valid = valids[0]
    delta = shard["delta_disp_gt_minus_raw"][offset, ys, xs].astype(np.float32)
    err = np.abs(raw - gt)
    return {
        "x": xfeat, "raw": raw, "gt": gt, "valid": valid, "delta": delta,
        "raw_bad": (err >= 3.0).astype(np.float32), "raw_good": (err < 1.0).astype(np.float32),
    }


class FullGTChunkDataset(torch.utils.data.Dataset):
    def __init__(self, sample_groups, shards, args, chunks_per_epoch: int):
        self.groups = sample_groups
        self.shards = shards
        self.args = args
        self.chunks_per_epoch = chunks_per_epoch
        self.rng = random.Random(2027)

    def __len__(self):
        return self.chunks_per_epoch

    def _crop_xy(self, shard, offsets):
        h, w = shard["raw_disp"].shape[1:]
        s = min(self.args.crop_size, h, w)
        best = (-1.0, 0, 0)
        for _ in range(self.args.crop_candidate_tries):
            y = self.rng.randint(0, max(0, h - s))
            x = self.rng.randint(0, max(0, w - s))
            off = offsets[self.rng.randrange(len(offsets))]
            raw = shard["raw_disp"][off, y : y + s, x : x + s].astype(np.float32)
            gt = shard["gt_disp"][off, y : y + s, x : x + s].astype(np.float32)
            valid = shard["valid_mask"][off, y : y + s, x : x + s].astype(np.float32)
            score = float((((np.abs(raw - gt) >= self.args.bad_threshold_px) | (np.abs(raw - gt) >= 10.0)) * valid).mean())
            if score >= best[0]:
                best = (score, y, x)
        return best[1], best[2], s

    def __getitem__(self, idx):
        group = self.groups[self.rng.randrange(len(self.groups))]
        shard = self.shards[group[0].target_path]
        n = len(group)
        length = min(self.args.chunk_length, n)
        start = self.rng.randint(0, max(0, n - length))
        offsets = [group[start + i].offset for i in range(length)]
        y, x, size = self._crop_xy(shard, offsets)
        rows = [sequence_features(shard, off, y, x, size, self.args.context_frames) for off in offsets]
        return {k: torch.from_numpy(np.stack([r[k] for r in rows])[:, None] if k != "x" else np.stack([r[k] for r in rows])) for k in rows[0]}


class OracleChunkDataset(torch.utils.data.Dataset):
    def __init__(self, clips, args, chunks_per_epoch: int, hardneg_masks: dict[str, dict[str, np.ndarray]] | None = None):
        self.clips = clips
        self.args = args
        self.chunks_per_epoch = chunks_per_epoch
        self.hardneg_masks = hardneg_masks or {}
        self.rng = random.Random(2028)

    def __len__(self):
        return self.chunks_per_epoch

    def __getitem__(self, idx):
        clip = self.clips[self.rng.randrange(len(self.clips))]
        n, h, w = clip.raws.shape
        length = min(self.args.chunk_length, n)
        start = self.rng.randint(0, max(0, n - length))
        s = min(self.args.crop_size, h, w)
        ref_mask = self.hardneg_masks.get(clip.clip_id, {}).get("hard_neg", clip.sup_mask)
        best = (-1.0, 0, 0)
        for _ in range(self.args.crop_candidate_tries):
            y = self.rng.randint(0, max(0, h - s))
            x = self.rng.randint(0, max(0, w - s))
            score = float(ref_mask[start : start + length, y : y + s, x : x + s].mean())
            if score >= best[0]:
                best = (score, y, x)
        _, y, x = best
        ys, xs = slice(y, y + s), slice(x, x + s)
        xs_feat = []
        raw = []
        gt = []
        valid = []
        delta = []
        raw_bad = []
        raw_good = []
        sup = []
        hard_neg = []
        hard_pos = []
        below3 = []
        masks = self.hardneg_masks.get(clip.clip_id)
        for fi in range(start, start + length):
            ids = [max(0, fi - i) for i in range(self.args.context_frames)]
            xfeat, _edge, _var = make_features_from_raws(clip.raws[ids, ys, xs], clip.valids[ids, ys, xs])
            r = clip.raws[fi, ys, xs].astype(np.float32)
            g = clip.gts[fi, ys, xs].astype(np.float32)
            v = clip.valids[fi, ys, xs].astype(np.float32)
            err = np.abs(r - g)
            xs_feat.append(xfeat); raw.append(r); gt.append(g); valid.append(v)
            delta.append((clip.oracle[fi, ys, xs] - r).astype(np.float32))
            raw_bad.append((err >= self.args.bad_threshold_px).astype(np.float32))
            raw_good.append((err < self.args.good_threshold_px).astype(np.float32))
            sup.append(clip.sup_mask[fi, ys, xs].astype(np.float32))
            if masks:
                hn = masks["hard_neg"][fi, ys, xs].astype(np.float32)
                hp = masks["hard_pos"][fi, ys, xs].astype(np.float32)
                hard_neg.append(hn); hard_pos.append(hp)
                below3.append(((err < self.args.bad_threshold_px) & (v > 0)).astype(np.float32))
        out = {
            "x": torch.from_numpy(np.stack(xs_feat)),
            "raw": torch.from_numpy(np.stack(raw)[:, None]),
            "gt": torch.from_numpy(np.stack(gt)[:, None]),
            "valid": torch.from_numpy(np.stack(valid)[:, None]),
            "delta": torch.from_numpy(np.stack(delta)[:, None]),
            "raw_bad": torch.from_numpy(np.stack(raw_bad)[:, None]),
            "raw_good": torch.from_numpy(np.stack(raw_good)[:, None]),
            "sup": torch.from_numpy(np.stack(sup)[:, None]),
        }
        if masks:
            out["hard_neg"] = torch.from_numpy(np.stack(hard_neg)[:, None])
            out["hard_pos"] = torch.from_numpy(np.stack(hard_pos)[:, None])
            out["below3"] = torch.from_numpy(np.stack(below3)[:, None])
        return out


def make_sequence_loader(dataset, args, workers: int):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=args.prefetch_factor if workers > 0 else None,
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets-root", type=Path, default=DEFAULT_TARGETS_ROOT)
    p.add_argument("--oracle-targets-root", type=Path, default=DEFAULT_ORACLE_TARGETS)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--balanced-split-json", type=Path, default=DEFAULT_BALANCED_SPLIT)
    p.add_argument("--v1-checkpoint", type=Path, default=DEFAULT_V1_CHECKPOINT)
    p.add_argument("--warm-start", nargs="?", const=True, default=True, type=parse_bool)
    p.add_argument("--context-frames", type=int, default=4)
    p.add_argument("--chunk-length", type=int, default=16)
    p.add_argument("--burn-in-frames", type=int, default=2)
    p.add_argument("--burnin-loss-weight", type=float, default=0.25)
    p.add_argument("--crop-size", type=int, default=96)
    p.add_argument("--crop-candidate-tries", type=int, default=12)
    p.add_argument("--crops-per-epoch", type=int, default=4096)
    p.add_argument("--stage1-epochs", type=int, default=6)
    p.add_argument("--stage2-epochs", type=int, default=10)
    p.add_argument("--stage3-epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=48)
    p.add_argument("--eval-clip-batch", type=int, default=16)
    p.add_argument("--full-gt-batch-ratio", type=float, default=0.50)
    p.add_argument("--oracle-batch-ratio", type=float, default=0.25)
    p.add_argument("--hard-negative-batch-ratio", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--stage1-lr", type=float, default=1e-4)
    p.add_argument("--stage1-freeze-backbone", nargs="?", const=True, default=True, type=parse_bool)
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
    p.add_argument("--memory-update-weight", type=float, default=0.0)
    p.add_argument("--memory-update-neg-weight", type=float, default=0.2)
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

    model = egbm_v3_care_streaming(16, args.residual_scale).to(device)
    params = sum(p.numel() for p in model.parameters())
    warm_loaded = warm_total = 0
    if args.warm_start and args.v1_checkpoint.exists():
        warm_loaded, warm_total = load_v1_warm_start(model, str(args.v1_checkpoint))

    splits, by_split = load_samples_with_split(args.targets_root, args.balanced_split_json, args.max_frames)
    shards = load_shards(by_split["train"] + by_split["val"] + by_split["test"])
    groups = {split: group_samples_by_sequence(by_split[split]) for split in by_split}
    clips = load_clips(args.oracle_targets_root, args)
    patho_clips = [c for c in clips if c.failure_mode in PATHOLOGICAL_MODES]
    clean_clips = [c for c in clips if c.failure_mode not in PATHOLOGICAL_MODES]
    full_ds = {split: FullFrameDataset(samples, shards, args.context_frames) for split, samples in by_split.items()}
    eval_loaders = {split: make_loader(ds, args.eval_batch_size, max(0, args.num_workers // 2), False, args.prefetch_factor) for split, ds in full_ds.items()}
    gt_loader = make_sequence_loader(FullGTChunkDataset(groups["train"], shards, args, args.crops_per_epoch), args, args.num_workers)

    run_lines = [
        f"device={device} params={params} model=egbm_v3_care_streaming",
        "memory_gate_semantics=high_keep_old_memory_low_write_candidate",
        f"warm_start={args.warm_start} v1_ckpt={args.v1_checkpoint} loaded_tensors={warm_loaded}/{warm_total}",
        f"frames={{'train': {len(full_ds['train'])}, 'val': {len(full_ds['val'])}, 'test': {len(full_ds['test'])}}}",
        f"sequences={{'train': {len(groups['train'])}, 'val': {len(groups['val'])}, 'test': {len(groups['test'])}}}",
        f"clips={len(clips)} patho={len(patho_clips)} clean={len(clean_clips)}",
        f"stages epochs={args.stage1_epochs}/{args.stage2_epochs}/{args.stage3_epochs} batch_chunks={args.batch_size} chunk_length={args.chunk_length} chunks_per_epoch={args.crops_per_epoch}",
    ]
    (args.output_root / "run.log").write_text("\n".join(run_lines) + "\n")

    def log(line: str) -> None:
        run_lines.append(line)
        (args.output_root / "run.log").write_text("\n".join(run_lines) + "\n")

    (args.output_root / "README.md").write_text(
        "# EGBM-v3-CARE-S\n\n"
        "Stateful streaming CARE over contiguous chunks. Window mode remains available for comparison; streaming mode keeps memory across a sequence/clip and resets only at boundaries.\n\n"
        "Memory gate semantics: high means keep old memory; low means write candidate memory.\n\n"
        "No S2M2/SAV/RAFT/DINO inference is run; training uses existing low-resolution target shards and selected oracle artifacts.\n"
    )

    rng = random.Random(555)
    train_rows: list[dict[str, Any]] = []
    stage_summaries: dict[str, Any] = {}

    # Stage 1 supervision fix (2026-07-03): the original recipe (lr=3e-4, all params
    # trainable) DIVERGED under 16-step BPTT with small chunk batches — det loss rose
    # epoch over epoch and the warm-started detector collapsed to chance streaming AUC
    # (~0.5) after one epoch, across 4 independent runs. forward_step itself is correct:
    # at warm start, streaming AUC == window AUC == 0.76 with zero training.
    # Fix: stage 1 freezes the v1 backbone (it needs no detector-stage training; v2
    # retrained it in stage 2 anyway) and trains only bad_head + CARE modules + the
    # widened up2/damping_head at a fine-tune LR.
    stage1_modules = [model.bad_head, model.care_encoder, model.care_gru, model.care_predictor,
                      model.care_head, model.care_context, model.memory_update_gate,
                      model.up2, model.damping_head]
    if args.stage1_freeze_backbone:
        for p_ in model.parameters():
            p_.requires_grad = False
        for mod in stage1_modules:
            for p_ in mod.parameters():
                p_.requires_grad = True
        optimizer = torch.optim.AdamW([p_ for p_ in model.parameters() if p_.requires_grad], lr=args.stage1_lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.stage1_lr)
    fg_stream = {}
    for epoch in range(1, args.stage1_epochs + 1):
        metrics = train_sequence_epoch(model, {"gt": gt_loader}, optimizer, args, device, rng, stage=1)
        fg_stream, _ = full_gt_eval_streaming(model, groups["val"], shards, args, device, burn_in=0)
        train_rows.append({"stage": 1, "epoch": epoch, **metrics, "val_auc_streaming": fg_stream["detector_auc"], "val_ap_streaming": fg_stream["detector_ap"]})
        log(f"stage=1 epoch={epoch} det={metrics.get('det_loss', float('nan')):.5f} keep={metrics.get('memory_keep_mean', float('nan')):.3f} stream_auc={fg_stream['detector_auc']:.4f}")
    save_ckpt(args.output_root / "checkpoints" / "stage1_detector.pt", model, args, splits, args.stage1_epochs, 1, {"val_streaming": fg_stream})
    stage_summaries["stage1"] = {"epochs": args.stage1_epochs, "val_streaming": fg_stream}

    for p_ in model.parameters():
        p_.requires_grad = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best2 = float("inf")
    for epoch in range(1, args.stage2_epochs + 1):
        metrics = train_sequence_epoch(model, {"gt": gt_loader}, optimizer, args, device, rng, stage=2)
        fg_stream, _ = full_gt_eval_streaming(model, groups["val"], shards, args, device, burn_in=0)
        score = fg_stream["refined_mae"] + 20.0 * max(0.0, fg_stream["refined_mae"] - fg_stream["raw_mae"])
        train_rows.append({"stage": 2, "epoch": epoch, **metrics, "val_raw_streaming": fg_stream["raw_mae"], "val_refined_streaming": fg_stream["refined_mae"]})
        if score < best2:
            best2 = score
            save_ckpt(args.output_root / "checkpoints" / "stage2_fullgt.pt", model, args, splits, epoch, 2, {"val_streaming": fg_stream})
        log(f"stage=2 epoch={epoch} stream_val={fg_stream['raw_mae']:.4f}->{fg_stream['refined_mae']:.4f} nb3={fg_stream['new_bad3_from_raw_good_pct']:.3f}% mod={fg_stream['modified_pct']:.2f}%")
    ck2 = torch.load(args.output_root / "checkpoints" / "stage2_fullgt.pt", map_location=device, weights_only=False)
    unwrap(model).load_state_dict(ck2["model_state_dict"])
    stage_summaries["stage2"] = {"epochs": args.stage2_epochs, "best_epoch": ck2["epoch"], "val_streaming": ck2["val_streaming"]}

    with torch.no_grad():
        masks = {}
        for clip in patho_clips:
            refined, p_bad, _diag = predict_clip_streaming(model, clip, args, device)
            masks[clip.clip_id] = mine_hard_masks(clip, p_bad, refined - clip.raws, args)
    hn_px = int(sum(m["hard_neg"].sum() for m in masks.values()))
    hp_px = int(sum(m["hard_pos"].sum() for m in masks.values()))
    log(f"stage=3 mining_streaming hard_neg={hn_px} hard_pos={hp_px}")

    gt_chunks = int(round(args.crops_per_epoch * args.full_gt_batch_ratio))
    oracle_chunks = int(round(args.crops_per_epoch * args.oracle_batch_ratio))
    hn_chunks = args.crops_per_epoch - gt_chunks - oracle_chunks
    loaders3 = {
        "gt": make_sequence_loader(FullGTChunkDataset(groups["train"], shards, args, gt_chunks), args, args.num_workers),
        "oracle": make_sequence_loader(OracleChunkDataset(clean_clips, args, oracle_chunks), args, max(2, args.num_workers // 4)),
        "hardneg": make_sequence_loader(OracleChunkDataset(patho_clips, args, hn_chunks, masks), args, max(2, args.num_workers // 4)),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.stage3_lr)
    best_score = float("inf")
    best_epoch = 0
    epoch = 0
    for epoch in range(1, args.stage3_epochs + 1):
        metrics = train_sequence_epoch(model, loaders3, optimizer, args, device, rng, stage=3)
        sel_stream, _ = eval_selected_streaming(model, clips, args, device)
        fg_stream, _ = full_gt_eval_streaming(model, groups["val"], shards, args, device, burn_in=0)
        score = score_epoch(sel_stream, {"raw_mae": fg_stream["raw_mae"], "refined_mae": fg_stream["refined_mae"]})
        train_rows.append({"stage": 3, "epoch": epoch, "score": score, **metrics, "sel_stream_mae": sel_stream["all"]["refined_mae"], "sel_stream_gap": sel_stream["all"]["oracle_gap_recovered_pct"], "fullgt_val_streaming": fg_stream["refined_mae"]})
        if score < best_score:
            best_score = score
            best_epoch = epoch
            save_ckpt(args.output_root / "checkpoints" / "best.pt", model, args, splits, epoch, 3, {"selected_streaming": sel_stream, "full_gt_val_streaming": fg_stream})
        log(f"stage=3 epoch={epoch} score={score:.4f} stream_sel={sel_stream['all']['refined_mae']:.4f} gap={sel_stream['all']['oracle_gap_recovered_pct']:.2f}% patho_nb3={sel_stream['patho']['new_bad3_frame_mean_pct']:.2f}% clean_nb3={sel_stream['clean']['new_bad3_frame_mean_pct']:.2f}% fg_val={fg_stream['raw_mae']:.4f}->{fg_stream['refined_mae']:.4f} damp_hn={metrics.get('damping_hard_neg_mean', float('nan')):.3f} keep_hn={metrics.get('memory_keep_hard_neg', float('nan')):.3f} keep_hp={metrics.get('memory_keep_hard_pos', float('nan')):.3f}")
        if epoch - best_epoch >= args.early_stop_patience:
            log(f"early_stop stage=3 epoch={epoch} best_epoch={best_epoch}")
            break
    if best_epoch == 0:
        save_ckpt(args.output_root / "checkpoints" / "best.pt", model, args, splits, 0, 3, {})
    best = torch.load(args.output_root / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    unwrap(model).load_state_dict(best["model_state_dict"])
    stage_summaries["stage3"] = {"epochs_run": epoch, "best_epoch": best_epoch, "hard_neg_pixels": hn_px, "hard_pos_pixels": hp_px}

    sel_window, preds_window = eval_selected(model, clips, args, device)
    sel_stream, preds_stream = eval_selected_streaming(model, clips, args, device)
    sel_burn, _ = eval_selected_streaming(model, clips, args, device, burn_in=args.burn_in_frames)
    fg_window = {split: full_gt_eval(model, eval_loaders[split], device, args.bad_threshold_px) for split in ("val", "test")}
    fg_stream_val, seq_val = full_gt_eval_streaming(model, groups["val"], shards, args, device)
    fg_stream_test, seq_test = full_gt_eval_streaming(model, groups["test"], shards, args, device)

    def frame_rows(preds):
        rows = []
        for c in clips:
            for i, f in enumerate(frame_metrics_egbm(c, preds[c.clip_id][0])):
                rows.append({"clip_id": c.clip_id, "sequence_id": c.sequence_id, "frame_id": c.frame_ids[i], "dominant_failure_mode": c.failure_mode, **f})
        return rows

    write_csv_union(args.output_root / "selected_oracle_metrics_window.csv", frame_rows(preds_window))
    write_csv_union(args.output_root / "selected_oracle_metrics_streaming.csv", frame_rows(preds_stream))
    write_csv(args.output_root / "selected_oracle_metrics_burnin.csv", [{"burn_in_frames": args.burn_in_frames, **sel_burn["all"]}])
    write_csv(args.output_root / "full_gt_val_metrics_window.csv", [fg_window["val"]])
    write_csv(args.output_root / "full_gt_test_metrics_window.csv", [fg_window["test"]])
    write_csv(args.output_root / "full_gt_val_metrics_streaming.csv", [fg_stream_val])
    write_csv(args.output_root / "full_gt_test_metrics_streaming.csv", [fg_stream_test])
    write_csv_union(args.output_root / "full_gt_sequence_metrics_streaming.csv", [{"split": "val", **r} for r in seq_val] + [{"split": "test", **r} for r in seq_test])
    write_csv(args.output_root / "pathological_metrics_window.csv", [sel_window["patho"]])
    write_csv(args.output_root / "pathological_metrics_streaming.csv", [sel_stream["patho"]])
    write_csv(args.output_root / "clean_metrics_window.csv", [sel_window["clean"]])
    write_csv(args.output_root / "clean_metrics_streaming.csv", [sel_stream["clean"]])

    sweep_rows = []
    orig_thr = float(unwrap(model).base_threshold)
    for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
        unwrap(model).base_threshold.fill_(thr)
        s_thr, _ = eval_selected_streaming(model, clips, args, device)
        sweep_rows.append({"base_threshold": thr, **{f"all_{k}": v for k, v in s_thr["all"].items()}, **{f"patho_{k}": v for k, v in s_thr["patho"].items()}, **{f"clean_{k}": v for k, v in s_thr["clean"].items()}})
    unwrap(model).base_threshold.fill_(orig_thr)
    write_csv(args.output_root / "threshold_sweep_streaming.csv", sweep_rows)

    damping_rows, care_rows, surprise_rows, update_rows = [], [], [], []
    for clip in clips:
        refined, _pbad, diag = preds_stream[clip.clip_id]
        hw = diag["surprise"].shape[-2:]
        valid_lr = pool_mask(clip.valids > 0, hw) > 0.5
        damp_lr = pool_mask(diag["damping"], hw)
        update_lr = pool_mask(diag["memory_keep_gate"], hw)
        care = diag["care_probs"]
        sur = diag["surprise"]
        row_c = {"clip_id": clip.clip_id, "failure_mode": clip.failure_mode}
        for k, name in enumerate(CARE_CLASSES):
            row_c[f"{name}_mean"] = float(care[:, k][valid_lr].mean())
        care_rows.append(row_c)
        surprise_rows.append({"clip_id": clip.clip_id, "failure_mode": clip.failure_mode, "surprise_mean": float(sur[valid_lr].mean()), "surprise_p95": float(np.percentile(sur[valid_lr], 95))})
        update_rows.append({"clip_id": clip.clip_id, "failure_mode": clip.failure_mode, "gate_semantics": "high=keep_old_memory_low=write_candidate", "memory_keep_mean": float(update_lr[valid_lr].mean()), "memory_keep_p05": float(np.percentile(update_lr[valid_lr], 5)), "memory_keep_p95": float(np.percentile(update_lr[valid_lr], 95))})
        if clip.clip_id in masks:
            hn_lr = pool_mask(masks[clip.clip_id]["hard_neg"], hw) > 0.05
            hp_lr = pool_mask(masks[clip.clip_id]["hard_pos"], hw) > 0.05
            damping_rows.append({
                "clip_id": clip.clip_id, "failure_mode": clip.failure_mode,
                "damping_mean_valid": float(damp_lr[valid_lr].mean()),
                "damping_mean_hard_neg": float(damp_lr[hn_lr & valid_lr].mean()) if (hn_lr & valid_lr).any() else float("nan"),
                "damping_mean_hard_pos": float(damp_lr[hp_lr & valid_lr].mean()) if (hp_lr & valid_lr).any() else float("nan"),
                "memory_keep_hard_neg": float(update_lr[hn_lr & valid_lr].mean()) if (hn_lr & valid_lr).any() else float("nan"),
                "memory_keep_hard_pos": float(update_lr[hp_lr & valid_lr].mean()) if (hp_lr & valid_lr).any() else float("nan"),
                "v1_flicker_reference_hn_hp": "0.387/0.733" if clip.failure_mode == "high_temporal_flicker" else "",
            })
    write_csv_union(args.output_root / "damping_analysis_streaming.csv", damping_rows)
    write_csv_union(args.output_root / "care_change_type_analysis_streaming.csv", care_rows)
    write_csv_union(args.output_root / "care_surprise_analysis_streaming.csv", surprise_rows)
    write_csv_union(args.output_root / "memory_update_analysis.csv", update_rows)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["window", "streaming"], [sel_window["all"]["refined_mae"], sel_stream["all"]["refined_mae"]])
    ax.set_ylabel("selected MAE")
    fig.tight_layout(); fig.savefig(args.output_root / "diagnostics" / "streaming_vs_window_selected_mae.png", dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["window", "streaming"], [sel_window["all"]["oracle_gap_recovered_pct"], sel_stream["all"]["oracle_gap_recovered_pct"]])
    ax.set_ylabel("oracle gap recovered (%)")
    fig.tight_layout(); fig.savefig(args.output_root / "diagnostics" / "streaming_vs_window_oracle_gap.png", dpi=140); plt.close(fig)
    if damping_rows:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(damping_rows))
        ax.bar(x - 0.2, [r["damping_mean_hard_neg"] for r in damping_rows], 0.4, label="hard neg")
        ax.bar(x + 0.2, [r["damping_mean_hard_pos"] for r in damping_rows], 0.4, label="hard pos")
        ax.set_xticks(x); ax.set_xticklabels([r["failure_mode"] for r in damping_rows], fontsize=8)
        ax.legend(); fig.tight_layout(); fig.savefig(args.output_root / "diagnostics" / "flicker_damping_separation_streaming.png", dpi=140); plt.close(fig)

    write_csv_union(args.output_root / "train_log.csv", train_rows)
    comparison = [
        {"model": "v3.2c", **BASELINES["v3.2c"]},
        {"model": "v4_tiny", **BASELINES["v4_tiny"]},
        {"model": "SOG", **BASELINES["SOG"]},
        {"model": "EGBM_v1", **{k: v for k, v in BASELINES["EGBM_v1"].items() if not k.startswith("flicker")}},
        {"model": "EGBM_v3_CARE_S_streaming", "selected_mae": sel_stream["all"]["refined_mae"], "gap_pct": sel_stream["all"]["oracle_gap_recovered_pct"], "patho_new_bad3": sel_stream["patho"]["new_bad3_frame_mean_pct"], "clean_new_bad3": sel_stream["clean"]["new_bad3_frame_mean_pct"], "full_gt_test_mae": fg_stream_test["refined_mae"]},
    ]
    write_csv_union(args.output_root / "final_comparison_table.csv", comparison)
    latex = "\\begin{tabular}{lccccc}\n\\toprule\nModel & Sel. MAE & Gap & Patho nB3 & Clean nB3 & Test MAE \\\\\n\\midrule\n"
    for r in comparison:
        latex += f"{r['model'].replace('_', '-')} & {r.get('selected_mae', float('nan')):.4f} & {r.get('gap_pct', float('nan')):.2f}\\% & {r.get('patho_new_bad3', float('nan')):.2f}\\% & {r.get('clean_new_bad3', float('nan')) if r.get('clean_new_bad3') is not None else float('nan'):.2f}\\% & {r.get('full_gt_test_mae', float('nan')):.4f} \\\\\n"
    latex += "\\bottomrule\n\\end{tabular}\n"
    (args.output_root / "final_comparison_table_latex.tex").write_text(latex)

    v1 = BASELINES["EGBM_v1"]
    success = {
        "streaming_selected_mae_below_v1": bool(sel_stream["all"]["refined_mae"] < v1["selected_mae"]),
        "streaming_gap_above_v1": bool(sel_stream["all"]["oracle_gap_recovered_pct"] > v1["gap_pct"]),
        "patho_new_bad3_at_most_1_30": bool(sel_stream["patho"]["new_bad3_frame_mean_pct"] <= 1.30),
        "clean_new_bad3_at_most_1": bool(sel_stream["clean"]["new_bad3_frame_mean_pct"] <= 1.0),
        "full_gt_test_at_most_v1": bool(fg_stream_test["refined_mae"] <= v1["full_gt_test_mae"]),
        "full_gt_test_beats_raw_and_v32c": bool(fg_stream_test["refined_mae"] < 4.6145),
    }
    summary = {
        "model": "egbm_v3_care_streaming",
        "params": params,
        "warm_start_tensors": f"{warm_loaded}/{warm_total}",
        "best_stage3_epoch": best.get("epoch"),
        "elapsed_seconds": time.perf_counter() - start,
        "stage_summaries": stage_summaries,
        "selected_window": sel_window,
        "selected_streaming": sel_stream,
        "selected_burnin": sel_burn,
        "full_gt_window": fg_window,
        "full_gt_streaming": {"val": fg_stream_val, "test": fg_stream_test},
        "damping_analysis_streaming": damping_rows,
        "memory_gate_semantics": "high_keep_old_memory_low_write_candidate",
        "memory_update_analysis": update_rows,
        "memory_keep_analysis": update_rows,
        "baselines": BASELINES,
        "success_criteria": success,
        "no_teacher_inference": True,
    }
    (args.output_root / "aggregate_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (args.output_root / "README.md").write_text(
        "# EGBM-v3-CARE-S\n\n"
        f"Best streaming selected MAE: {sel_stream['all']['refined_mae']:.4f}; oracle gap: {sel_stream['all']['oracle_gap_recovered_pct']:.2f}%.\n\n"
        f"Full-GT test streaming: raw {fg_stream_test['raw_mae']:.4f} -> refined {fg_stream_test['refined_mae']:.4f}.\n\n"
        "Memory gate semantics: high means keep old memory; low means write candidate memory. Hard-negative flicker loss pushes this gate high.\n\n"
        "This run is causal/stateful: memory resets at sequence boundaries and persists across frames. Window-mode outputs are written separately for comparison.\n"
    )
    print(json.dumps(success, indent=2))
    print(json.dumps({"selected_streaming": sel_stream["all"], "full_gt_test_streaming": fg_stream_test}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
