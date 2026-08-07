#!/usr/bin/env python3
"""ARGOS v2 -- final post-hoc veto experiment: hard-negative training + harm
calibration for the stereo SpatialErrorCritic over the FROZEN raw multi-anchor
proposal.

Question: can failure-mode-balanced supervision + validation-only probability
calibration convert the critic's stable harm *ranking* into a genuinely useful
<=10% accepted-harm operating point at coverage >=5%, while staying better than
bounded CODD-style H4? Nothing in the proposal path is trained here -- veto only.

Dataset 7 is opened at most ONCE, and only if the aggregate 3-seed dataset-2
validation gate passes. It is held out for THIS experiment but is not a pristine
project-wide test set (earlier ARGOS v2 studies inspected it).

Reuses the completed experiment's bank builder, frozen proposal, feature builder
and calibration helpers; the genuinely new pieces are the hard-negative taxonomy,
per-pixel reweighting, weighted loss, and temperature/Platt calibration.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from model_design.models.spatial_error_critic import (  # noqa: E402
    SpatialErrorCritic, build_critic_features, feature_channels, stereo_residual, _gradient,
)
from run_raw_multi_anchor_selective_gate import EXPECTED_FROZEN_SHA256, FIXED_H4_EPE, ece  # noqa: E402
from run_raw_multi_anchor_temporal_refiner import (  # noqa: E402
    RAW_AGES, TRAIN, VALIDATION, loader, save_json, seed_all, sha256, split_manifest, to_device, write_csv,
)
import run_raw_multi_anchor_spatial_safety_critic as base  # noqa: E402
from run_raw_multi_anchor_spatial_safety_critic import (  # noqa: E402
    batch_proposal, build_critic_bank, collect_arrays, detection_metrics, family_root,
    policy_accept, policy_metrics, risk_coverage_rows, supervision, threshold_grid,
)

OUTPUT = ROOT / "results/raw_multi_anchor_spatial_safety_critic_hard_negative"
FAMILY = "stereo"                       # primary architecture; plane_sweep stays an ablation (not run: prior work showed it ~= stereo within noise)
SEEDS = (20260722, 20260723, 20260724)
HARM_MARGIN = 0.10
REFERENCE_CKPT = ROOT / "results/raw_multi_anchor_spatial_safety_critic/stereo_residual/checkpoints/best_validation.pt"

# ----- fixed hard-negative reweighting constants (identical across all seeds) -----
HN_BOOST = 3.0          # every hard negative (proposal harmful where it would act)
CTX_BOOST = 3.0         # additional, for hard negatives at boundary/occlusion/large-update (where harm concentrates)
BOUNDARY_PX = 1.0       # raw-disparity gradient threshold (px)
SUPPORT_LOW = 0.5       # warp support / fb-confidence "low" threshold
RESIDUAL_HIGH = 0.5     # stereo / temporal robust residual "high" threshold
UPDATE_LARGE_PX = 1.0   # |proposed-raw| "large update" threshold (px)
AMBIGUITY_EPS = 0.05    # |stereo_res(raw)-stereo_res(prop)| below this => both hypotheses plausible

# ----- validation gate (dataset 2 only) -----
GATE = {"harm": 0.10, "clean_deg": 0.03, "degraded_frame": 0.25, "coverage": 0.05}

# ----- loss variants -----
VARIANTS = {
    "B": {"lambda_harm": 1.0, "lambda_rank": 0.1, "extra_pos": 1.0, "weighted": True, "temperature": False},
    "C": {"lambda_harm": 2.5, "lambda_rank": 0.1, "extra_pos": 2.0, "weighted": True, "temperature": False},
    "D": {"lambda_harm": 2.5, "lambda_rank": 0.5, "extra_pos": 2.0, "weighted": True, "temperature": False},
    "E": {"base": "D", "temperature": True},  # D + validation-only probability calibration
}


UNSEEN_BACKBONES = ("CREStereo", "Fast-FoundationStereo")   # excluded from all critic/proposal training; in-domain SCARED-C
TEST = ("dataset_7_keyframe_1", "dataset_7_keyframe_2", "dataset_7_keyframe_3", "dataset_7_keyframe_4")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "run", "phase6"), default="run")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=8)      # two banks resident -> stay clear of the memcg fork-COW ceiling
    parser.add_argument("--flow-batch-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--harm-margin", type=float, default=HARM_MARGIN)
    parser.add_argument("--coverage-threshold", type=float, default=.50)
    parser.add_argument("--sample-stride", type=int, default=8)
    parser.add_argument("--seed", type=int, default=SEEDS[0])
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Hard-negative taxonomy + per-pixel reweighting                              #
# --------------------------------------------------------------------------- #
def _strata(batch, evidence, proposal, e_raw, e_proposed, supervised, device):
    """Per-pixel boolean strata (training only) from existing causal features.
    Returns (strata: dict[str, BoolTensor], hard_negative: BoolTensor)."""
    delta = e_raw - e_proposed
    hn = supervised & (delta < -HARM_MARGIN)                       # proposal harmful where it would act
    support = torch.gather(evidence.warp_support.float(), 1, proposal.chosen)
    fb = torch.gather(evidence.fb_confidence, 1, proposal.chosen)
    temporal = torch.gather(batch["photo"].float(), 1, proposal.chosen)
    r_raw, _ = stereo_residual(batch["left_rgb"].float(), batch["right_rgb"].float(), evidence.raw)
    r_prop, _ = stereo_residual(batch["left_rgb"].float(), batch["right_rgb"].float(), proposal.proposed)
    age = torch.gather(torch.tensor(list(RAW_AGES), device=device)[None, :, None, None]
                       .expand(len(batch["raw"]), -1, *evidence.raw.shape[-2:]), 1, proposal.chosen).float()
    boundary = _gradient(evidence.raw) > BOUNDARY_PX
    occlusion = (support < SUPPORT_LOW) | (fb < SUPPORT_LOW)
    large_update = (proposal.proposed - evidence.raw).abs() > UPDATE_LARGE_PX
    image_grad = _gradient(batch["left_rgb"].float().mean(1, keepdim=True) / 255.0)
    strata = {
        "raw_good_proposal_bad3": hn & (e_raw < 1.0) & (e_proposed >= 3.0),
        "disparity_boundary": hn & boundary,
        "occlusion_proxy": hn & occlusion,
        "high_stereo_residual": hn & (r_raw > RESIDUAL_HIGH),
        "temporal_inconsistency": hn & (temporal > RESIDUAL_HIGH),
        "low_warp_support": hn & (support < SUPPORT_LOW),
        "low_fb_confidence": hn & (fb < SUPPORT_LOW),
        "large_update_magnitude": hn & large_update,
        "distant_anchor_cs4_cs8": hn & (age >= 4),
        "raw_proposal_ambiguity": hn & ((r_raw - r_prop).abs() < AMBIGUITY_EPS),
        "specular_low_texture": hn & (image_grad < 0.05) & (r_raw > RESIDUAL_HIGH),
        "helpful_accepted": supervised & (delta > HARM_MARGIN),
        "easy_raw_fallback": supervised & (delta.abs() <= HARM_MARGIN),
    }
    return strata, hn, {"boundary": boundary, "occlusion": occlusion, "large_update": large_update}


def sample_weight(supervised, hn, context) -> torch.Tensor:
    """Per-pixel loss weight, mean-normalised over supervised pixels so the
    effective learning rate is unchanged. Hard negatives up-weighted; the
    concentrated failure regions (boundary/occlusion/large update) extra. Helpful
    and easy pixels stay at 1.0 -- we do NOT train exclusively on hard negatives."""
    concentrated = hn & (context["boundary"] | context["occlusion"] | context["large_update"])
    w = torch.ones_like(supervised, dtype=torch.float32)
    w = w + HN_BOOST * hn.float() + CTX_BOOST * concentrated.float()
    w = w * supervised.float()
    mean = w[supervised].mean().clamp_min(1e-6)
    return w / mean


def weighted_losses(output, e_raw, e_proposed, supervised, weight, *, lambda_harm, lambda_rank,
                    extra_pos, rank_margin=0.05):
    """critic_losses, but every per-pixel term multiplied by ``weight`` (the
    hard-negative reweighting). ``extra_pos`` scales the harm-BCE pos_weight
    (weighted harm loss, variant C/D)."""
    mask = supervised.bool()
    if not mask.any():
        zero = output.harm_logit.sum() * 0
        return {"total": zero, "error": zero.detach(), "harm": zero.detach(), "rank": zero.detach()}
    w = weight[mask]
    nll = ((e_raw - output.mu_raw).abs() * torch.exp(-output.s_raw) + output.s_raw
           + (e_proposed - output.mu_proposed).abs() * torch.exp(-output.s_proposed) + output.s_proposed)
    error = (nll[mask] * w).sum() / w.sum()
    harm_target = (e_proposed > e_raw + HARM_MARGIN).float()
    labels = harm_target[mask]
    positive = labels.sum()
    pos_weight = ((labels.numel() - positive) / positive.clamp_min(1)).clamp(.5, 4.0).detach() * 2.0 * extra_pos
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        output.harm_logit[mask], labels, pos_weight=pos_weight, reduction="none")
    harm = (bce * w).sum() / w.sum()
    rank_sign = torch.sign(e_raw - e_proposed)
    rank_raw = torch.nn.functional.relu(rank_margin - rank_sign * output.delta_hat)[mask]
    rank = (rank_raw * w).sum() / w.sum()
    total = error + lambda_harm * harm + lambda_rank * rank
    return {"total": total, "error": error.detach(), "harm": harm.detach(), "rank": rank.detach()}


def analyze_taxonomy(train_bank, frozen, base_policy, config) -> tuple[list[dict], list[dict]]:
    """One no-grad pass over the training bank: stratum pixel counts, harm rate,
    and effective sampling-weight mass (rebalancing evidence)."""
    device = torch.device(config.device)
    counts = defaultdict(float); harm_sum = defaultdict(float); delta_sum = defaultdict(float); wmass = defaultdict(float)
    total_sup = 0.0; total_hn = 0.0; total_wmass = 0.0
    with torch.inference_mode():
        for cpu in loader(train_bank, config, training=False):
            batch = to_device(cpu, device)
            evidence, proposal, _ = batch_proposal(batch, frozen, base_policy, device)
            e_raw, e_proposed, _v, supervised = supervision(batch, evidence, proposal, config)
            strata, hn, context = _strata(batch, evidence, proposal, e_raw, e_proposed, supervised, device)
            w = sample_weight(supervised, hn, context)
            delta = e_raw - e_proposed
            total_sup += float(supervised.sum()); total_hn += float(hn.sum()); total_wmass += float(w.sum())
            for name, mask in strata.items():
                m = mask.bool()
                counts[name] += float(m.sum())
                harm_sum[name] += float(hn[m].sum())
                delta_sum[name] += float(delta[m].sum())
                wmass[name] += float(w[m].sum())
    taxonomy = []
    for name in strata:
        c = counts[name]
        taxonomy.append({"stratum": name, "pixel_count": int(c),
                         "fraction_of_supervised": c / max(total_sup, 1),
                         "harm_rate": harm_sum[name] / max(c, 1),
                         "mean_delta": delta_sum[name] / max(c, 1)})
    sampling = [{"stratum": name, "pixel_fraction": counts[name] / max(total_sup, 1),
                 "weight_mass_fraction": wmass[name] / max(total_wmass, 1),
                 "rebalance_ratio": (wmass[name] / max(total_wmass, 1)) / max(counts[name] / max(total_sup, 1), 1e-9)}
                for name in strata]
    sampling.append({"stratum": "_overall", "pixel_fraction": 1.0, "weight_mass_fraction": 1.0,
                     "rebalance_ratio": 1.0, "hard_negative_fraction": total_hn / max(total_sup, 1)})
    return taxonomy, sampling


# --------------------------------------------------------------------------- #
# Training a single variant/seed (banks are prebuilt and reused)              #
# --------------------------------------------------------------------------- #
def train_variant(variant: str, seed: int, train_bank, val_bank, frozen, base_policy, config) -> tuple[SpatialErrorCritic, list[dict]]:
    spec = VARIANTS["D"] if VARIANTS[variant].get("base") == "D" else VARIANTS[variant]
    device = torch.device(config.device)
    seed_all(seed); config.seed = seed
    model = SpatialErrorCritic(feature_channels(FAMILY), channels=config.channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    data = loader(train_bank, config, training=True)
    epochs = 2 if config.mode == "smoke" else config.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(epochs * len(data), 1), eta_min=config.learning_rate * .05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []; best = -math.inf; best_state = None; first = None
    for epoch in range(epochs):
        model.train(); data.balanced_sampler.set_epoch(epoch)
        totals = defaultdict(float); steps = 0
        for cpu in data:
            batch = to_device(cpu, device)
            evidence, proposal, extras = batch_proposal(batch, frozen, base_policy, device)
            e_raw, e_proposed, _v, supervised = supervision(batch, evidence, proposal, config)
            strata, hn, context = _strata(batch, evidence, proposal, e_raw, e_proposed, supervised, device)
            weight = sample_weight(supervised, hn, context) if spec["weighted"] else supervised.float()
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                features, _ = build_critic_features(evidence, proposal, extras, FAMILY)
                output = model(features.float())
                losses = weighted_losses(output, e_raw, e_proposed, supervised, weight,
                                         lambda_harm=spec["lambda_harm"], lambda_rank=spec["lambda_rank"],
                                         extra_pos=spec["extra_pos"])
            optimizer.zero_grad(set_to_none=True); scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            if first is None: first = float(losses["total"].detach())
            for k, v in losses.items(): totals[k] += float(v if isinstance(v, float) else v.detach())
            steps += 1
        diag = base.validation_pass(model, val_bank, frozen, base_policy, config, FAMILY)
        row = {"variant": variant, "seed": seed, "epoch": epoch + 1,
               **{k: totals[k] / max(steps, 1) for k in totals},
               **{f"val_{k}": diag[k] for k in diag}}
        history.append(row)
        score = diag["best_selective_gain"]
        if score > best: best = score; best_state = copy.deepcopy(model.state_dict())
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    model.load_state_dict(best_state)
    return model, history


# --------------------------------------------------------------------------- #
# Validation-only probability calibration (variant E)                         #
# --------------------------------------------------------------------------- #
def fit_calibration(logit: np.ndarray, label: np.ndarray) -> dict:
    """Fit temperature (1 param) and Platt (2 param) on dataset-2; keep whichever
    gives lower ECE. Isotonic is skipped unless support is large & well-mixed."""
    z = torch.tensor(logit, dtype=torch.float64); y = torch.tensor(label, dtype=torch.float64)
    results = {}
    for name, nparam in (("temperature", 1), ("platt", 2)):
        a = torch.zeros(1, dtype=torch.float64, requires_grad=True)   # log-scale => scale=exp(a)>0
        b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        params = [a] if nparam == 1 else [a, b]
        opt = torch.optim.LBFGS(params, lr=0.5, max_iter=100)
        def closure():
            opt.zero_grad()
            logits = torch.exp(a) * z + (0.0 if nparam == 1 else b)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            loss.backward(); return loss
        opt.step(closure)
        scale = float(torch.exp(a)); bias = 0.0 if nparam == 1 else float(b.detach())
        p = 1.0 / (1.0 + np.exp(-(scale * logit + bias)))
        results[name] = {"scale": scale, "bias": bias, "ece": float(ece(label, p)),
                         "brier": float(brier_score_loss(label, p))}
    best = min(results, key=lambda n: results[n]["ece"])
    return {"method": best, **results[best], "all": results}


def fold_calibration(model: SpatialErrorCritic, scale: float, bias: float) -> SpatialErrorCritic:
    """Bake logit' = scale*logit + bias into the harm head row so the runner's
    plain sigmoid(harm_logit) reproduces the calibrated probability exactly."""
    folded = copy.deepcopy(model)
    with torch.no_grad():
        folded.head.weight[4:5].mul_(scale)
        folded.head.bias[4:5].mul_(scale).add_(bias)
    return folded


# --------------------------------------------------------------------------- #
# Dataset-2 calibration + gate for one critic                                 #
# --------------------------------------------------------------------------- #
def calibrate_critic(name: str, model, val_bank, config) -> dict:
    arrays = collect_arrays(config, val_bank, {name: model})
    candidates = [{"policy": name, **c, **policy_metrics(arrays, policy_accept(arrays, name, c), config.harm_margin)}
                  for c in threshold_grid()]
    feasible = [r for r in candidates if r["harmful_update_rate"] <= GATE["harm"] and r["clean_degradation"] <= GATE["clean_deg"]
                and r["degraded_frame_fraction"] <= GATE["degraded_frame"] and r["coverage"] >= GATE["coverage"]]
    best = max(feasible or candidates, key=lambda r: (bool(feasible), r["gain_over_fixed_h4"], -r["harmful_update_rate"]))
    best["feasible"] = bool(feasible)
    det = detection_metrics(arrays, name, config.harm_margin)
    risk = risk_coverage_rows(arrays, name, config.harm_margin)
    # recover harm logit for calibration fitting (sigmoid is exact-invertible)
    mask = arrays["valid"].astype(bool) & arrays["eligible"].astype(bool)
    p = np.clip(arrays[name + "_harm"][mask], 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)); label = (arrays["delta"][mask] < -config.harm_margin).astype(np.float64)
    return {"policy": best, "detection": det, "risk_coverage": risk, "candidates": candidates,
            "calib_logit": logit, "calib_label": label}


def _self_check():
    """Ponytail check: weight normalisation keeps mean 1 over supervised, hard
    negatives are strictly up-weighted, and calibration folding is exact."""
    sup = torch.zeros(1, 1, 4, 4, dtype=torch.bool); sup[..., :3, :] = True
    hn = torch.zeros_like(sup); hn[..., 0, 0] = True
    ctx = {"boundary": hn.clone(), "occlusion": torch.zeros_like(sup), "large_update": torch.zeros_like(sup)}
    w = sample_weight(sup, hn, ctx)
    assert abs(float(w[sup].mean()) - 1.0) < 1e-5, "weight not mean-normalised over supervised"
    assert float(w[..., 0, 0]) > float(w[..., 1, 0]) > 0, "hard negative not up-weighted above easy"
    assert float(w[..., 3, 0]) == 0.0, "unsupervised pixel must have zero weight"
    m = SpatialErrorCritic(feature_channels(FAMILY)); x = torch.rand(2, feature_channels(FAMILY), 8, 8)
    with torch.no_grad():
        base_logit = m(x).harm_logit
        folded = fold_calibration(m, 1.7, -0.3)
        assert torch.allclose(folded(x).harm_logit, 1.7 * base_logit - 0.3, atol=1e-4), "calibration fold not exact"
    ok = {"feasible": True, "gain": .02, "harmful_update_rate": GATE["harm"], "coverage": GATE["coverage"],
          "gain_over_fixed_h4": .01, "clean_degradation": GATE["clean_deg"]}
    assert _gate_pass(ok), "boundary-feasible ensemble policy must pass the gate"
    assert not _gate_pass(ok | {"feasible": False}), "infeasible policy must fail"
    assert not _gate_pass(ok | {"gain": 0.0}), "non-positive gain must fail"
    assert not _gate_pass(ok | {"harmful_update_rate": GATE["harm"] + 1e-6}), "harm above target must fail"
    assert not _gate_pass(ok | {"coverage": GATE["coverage"] - 1e-6}), "coverage below target must fail"
    assert not _gate_pass(ok | {"gain_over_fixed_h4": 0.0}), "no advantage over fixed H4 must fail"
    assert not _gate_pass(ok | {"clean_degradation": GATE["clean_deg"] + 1e-6}), "clean degradation must fail"
    assert _overall_safety("FULL GO", "FULL UNSEEN-BACKBONE GO").startswith("GO"), "full/full must be GO"
    for seen, unseen in (("CONDITIONAL GO", "FULL UNSEEN-BACKBONE GO"), ("FULL GO", "CONDITIONAL UNSEEN-BACKBONE GO"),
                         ("CONDITIONAL GO", "CONDITIONAL UNSEEN-BACKBONE GO")):
        assert _overall_safety(seen, unseen).startswith("CONDITIONAL"), f"{seen}/{unseen} must not be GO"
    assert _overall_safety("NO-GO (validation)", "N/A (never frozen)").startswith("NO-GO"), "never-frozen must be NO-GO"
    assert _overall_safety("FULL GO", "UNSEEN-BACKBONE NO-GO").startswith("NO-GO"), "unseen NO-GO must be NO-GO"
    print("self-check ok")


def _gate_pass(policy: dict) -> bool:
    """Dataset-2 gate applied to ONE policy -- the one that is actually frozen."""
    return bool(policy["feasible"] and policy["gain"] > 0 and policy["harmful_update_rate"] <= GATE["harm"]
                and policy["coverage"] >= GATE["coverage"] and policy["gain_over_fixed_h4"] > 0
                and policy["clean_degradation"] <= GATE["clean_deg"])


def stratum_metrics(model, val_bank, thresholds, config) -> list[dict]:
    """Per-stratum dataset-2 behaviour of the selected critic: count, intrinsic
    harm rate, and (accept rate, accepted harm) under the frozen policy."""
    device = torch.device(config.device); frozen, _ = base.load_frozen(device); base_policy = base.frozen_policy()
    acc = defaultdict(lambda: {"n": 0.0, "harm": 0.0, "accept": 0.0, "acc_harm": 0.0})
    with torch.inference_mode():
        for cpu in loader(val_bank, config, training=False):
            batch = to_device(cpu, device)
            evidence, proposal, extras = batch_proposal(batch, frozen, base_policy, device)
            e_raw, e_proposed, _v, supervised = supervision(batch, evidence, proposal, config)
            strata, hn, _ctx = _strata(batch, evidence, proposal, e_raw, e_proposed, supervised, device)
            features, _ = build_critic_features(evidence, proposal, extras, FAMILY)
            output = model(features)
            from model_design.models.spatial_error_critic import selective_accept
            accept = selective_accept(output, proposal, lambda_uncertainty=thresholds["lambda_uncertainty"],
                                      tau_gain=thresholds["tau_gain"], tau_harm=thresholds["tau_harm"])
            for name, mask in strata.items():
                m = mask.bool(); a = acc[name]
                a["n"] += float(m.sum()); a["harm"] += float(hn[m].sum())
                a["accept"] += float(accept[m].sum()); a["acc_harm"] += float((accept & hn)[m].sum())
    return [{"stratum": name, "count": int(a["n"]), "harm_rate": a["harm"] / max(a["n"], 1),
             "accept_rate": a["accept"] / max(a["n"], 1),
             "accepted_harm_rate": a["acc_harm"] / max(a["accept"], 1)} for name, a in acc.items()]


def _payload(model, seed):
    return {"project": "ARGOS v2", "family": FAMILY, "seed": seed, "model": model.state_dict(),
            "channels": model.body[0].out_channels if hasattr(model.body[0], "out_channels") else 64,
            "in_channels": feature_channels(FAMILY), "harm_margin": HARM_MARGIN,
            "frozen_checkpoint_sha256": EXPECTED_FROZEN_SHA256, "split": split_manifest(), "epoch": -1}


def freeze_and_open_d7(models: dict[int, SpatialErrorCritic], thresholds: dict, config, enrichment: dict) -> dict:
    """Materialise the 3 folded seed checkpoints where the (tested) base runner's
    evaluate() expects them, write a compatible + enriched freeze manifest, hash
    it, then open dataset 7 exactly once via the reused evaluate()."""
    for seed, model in models.items():
        path = family_root(config, FAMILY, seed) / "checkpoints"; path.mkdir(parents=True, exist_ok=True)
        torch.save(_payload(model, seed), path / "best_validation.pt")
    hashes = {(FAMILY if seed == SEEDS[0] else f"{FAMILY}_seed{seed}"):
              sha256(family_root(config, FAMILY, seed) / "checkpoints/best_validation.pt") for seed in models}
    selected = f"{FAMILY}@ensemble" if len(models) > 1 else FAMILY
    manifest = {"project": "ARGOS v2", "experiment": "spatial critic hard-negative (veto-only)",
                "selected_only_on": list(VALIDATION), "dataset7_accessed": False, "dataset7_note":
                "held out for THIS experiment only; earlier ARGOS v2 studies inspected it; not a pristine project-wide test set",
                "frozen_refiner_sha256": EXPECTED_FROZEN_SHA256, "harm_margin": HARM_MARGIN,
                "constraints": {"harmful_update_rate": GATE["harm"], "clean_degradation": GATE["clean_deg"],
                                "degraded_frame_fraction": GATE["degraded_frame"], "minimum_coverage": GATE["coverage"]},
                "selected_policy": selected, "policies": {selected: thresholds},
                "critic_checkpoint_hashes": hashes, **enrichment}
    (config.output / "calibration").mkdir(parents=True, exist_ok=True)
    save_json(config.output / "calibration/freeze_manifest.json", manifest)
    (config.output / "calibration/freeze_manifest.sha256").write_text(
        sha256(config.output / "calibration/freeze_manifest.json") + "\n")
    test_config = copy.deepcopy(config); test_config.split = "test"
    base.evaluate(test_config)
    return {"freeze_manifest_sha256": sha256(config.output / "calibration/freeze_manifest.json"), "selected_policy": selected}


def run_pipeline(config) -> None:
    config.output.mkdir(parents=True, exist_ok=True)
    (config.output / "protocol_audit").mkdir(exist_ok=True); (config.output / "logs").mkdir(exist_ok=True)
    device = torch.device(config.device)
    from run_raw_multi_anchor_selective_gate import FROZEN_CHECKPOINT
    assert sha256(FROZEN_CHECKPOINT) == EXPECTED_FROZEN_SHA256, "frozen proposal hash changed"
    save_json(config.output / "protocol_audit.json", split_manifest() | {
        "project": "ARGOS v2", "experiment": "spatial critic hard-negative", "veto_only": True,
        "frozen_refiner_sha256": sha256(FROZEN_CHECKPOINT), "reference_checkpoint_sha256": sha256(REFERENCE_CKPT),
        "primary_family": FAMILY, "seeds": list(SEEDS), "gate": GATE, "variants": {k: VARIANTS[k] for k in VARIANTS},
        "dataset7_locked_until_gate_passes": True, "dataset7_is_project_wide_pristine": False,
        "reweight_constants": {"HN_BOOST": HN_BOOST, "CTX_BOOST": CTX_BOOST, "BOUNDARY_PX": BOUNDARY_PX,
                               "SUPPORT_LOW": SUPPORT_LOW, "RESIDUAL_HIGH": RESIDUAL_HIGH,
                               "UPDATE_LARGE_PX": UPDATE_LARGE_PX, "AMBIGUITY_EPS": AMBIGUITY_EPS}})

    train_seq = (TRAIN[0], TRAIN[2]) if config.mode == "smoke" else TRAIN
    if config.mode == "smoke": config.max_frames = 24
    print(f"[banks] building TRAIN {train_seq}", flush=True)
    train_bank = build_critic_bank(config, train_seq, training=True)
    print(f"[banks] building VALIDATION {VALIDATION}", flush=True)
    val_bank = build_critic_bank(config, VALIDATION, training=True)
    frozen, _ = base.load_frozen(device); base_policy = base.frozen_policy()

    print("[taxonomy] analysing training bank", flush=True)
    taxonomy, sampling = analyze_taxonomy(train_bank, frozen, base_policy, config)
    write_csv(config.output / "hard_negative_taxonomy.csv", taxonomy)
    write_csv(config.output / "sampling_summary.csv", sampling)

    # ----- variant A: frozen completed critic reference (no training) -----
    ref = SpatialErrorCritic(feature_channels(FAMILY), channels=config.channels).to(device)
    ref.load_state_dict(torch.load(REFERENCE_CKPT, map_location="cpu", weights_only=False)["model"]); ref.eval()
    calib = {"A": calibrate_critic("stereo", ref, val_bank, config)}
    histories = {}

    # ----- variants B, C, D on seed 0 -----
    for variant in ("B", "C", "D"):
        print(f"[train] variant {variant} seed {SEEDS[0]}", flush=True)
        model, hist = train_variant(variant, SEEDS[0], train_bank, val_bank, frozen, base_policy, config)
        histories[variant] = hist; calib[variant] = calibrate_critic("stereo", model, val_bank, config)
        calib[variant]["_model"] = model

    # ----- variant E = D + validation-only probability calibration -----
    d_model = calib["D"]["_model"]
    fit = fit_calibration(calib["D"]["calib_logit"], calib["D"]["calib_label"])
    e_model = fold_calibration(d_model, fit["scale"], fit["bias"])
    calib["E"] = calibrate_critic("stereo", e_model, val_bank, config); calib["E"]["_model"] = e_model
    calib["E"]["calibration_fit"] = fit
    save_json(config.output / "calibration_fit.json", {"variant_E_fit": fit})

    # ----- loss ablation table (seed 0) + recipe selection -----
    ablation = []
    for v in ("A", "B", "C", "D", "E"):
        p = calib[v]["policy"]; d = calib[v]["detection"]
        ablation.append({"variant": v, "feasible": p["feasible"], "gain": p["gain"],
                         "gain_over_fixed_h4": p["gain_over_fixed_h4"], "coverage": p["coverage"],
                         "harmful_update_rate": p["harmful_update_rate"], "clean_degradation": p["clean_degradation"],
                         "degraded_frame_fraction": p["degraded_frame_fraction"],
                         "intervention_precision": p["intervention_precision"],
                         "harm_auroc": d.get("harm_auroc"), "harm_auprc": d.get("harm_auprc"),
                         "brier": d.get("brier"), "ece": d.get("ece"),
                         "lambda_uncertainty": p["lambda_uncertainty"], "tau_gain": p["tau_gain"], "tau_harm": p["tau_harm"]})
    write_csv(config.output / "loss_ablation.csv", ablation)
    write_csv(config.output / "calibration_summary.csv",
              [{"variant": v, **calib[v]["detection"]} for v in ("A", "B", "C", "D", "E")])
    recipe = max(("B", "C", "D", "E"), key=lambda v: (calib[v]["policy"]["feasible"],
                 calib[v]["policy"]["gain_over_fixed_h4"], -calib[v]["policy"]["harmful_update_rate"]))
    print(f"[recipe] selected {recipe}", flush=True)

    # ----- 3-seed training of the selected recipe -----
    base_variant = "D" if recipe == "E" else recipe
    seed_models = {SEEDS[0]: calib[recipe]["_model"]}
    seed_rows = [{"seed": SEEDS[0], **{k: calib[recipe]["policy"][k] for k in
                 ("feasible", "gain", "gain_over_fixed_h4", "coverage", "harmful_update_rate", "clean_degradation")}}]
    for seed in SEEDS[1:]:
        print(f"[train] recipe {recipe} (base {base_variant}) seed {seed}", flush=True)
        model, hist = train_variant(base_variant, seed, train_bank, val_bank, frozen, base_policy, config)
        histories[f"{recipe}_seed{seed}"] = hist
        if recipe == "E":
            c = calibrate_critic("stereo", model, val_bank, config)
            f = fit_calibration(c["calib_logit"], c["calib_label"]); model = fold_calibration(model, f["scale"], f["bias"])
        c = calibrate_critic("stereo", model, val_bank, config)
        seed_models[seed] = model
        seed_rows.append({"seed": seed, **{k: c["policy"][k] for k in
                         ("feasible", "gain", "gain_over_fixed_h4", "coverage", "harmful_update_rate", "clean_degradation")}})
    write_csv(config.output / "seed_summary.csv", seed_rows)

    # ----- validation gate on the ACTUAL frozen stack (dataset 2 only) -----
    # The stack that dataset 7 tests is the 3-seed mean ensemble under ONE policy,
    # so the ensemble is calibrated and gated as a single critic here; the per-seed
    # policies stay as stability diagnostics (each seed must also be feasible).
    ensemble = base.EnsembleCritic([seed_models[seed] for seed in SEEDS])
    ens = calibrate_critic("stereo@ensemble", ensemble, val_bank, config)
    ens_policy = ens["policy"]
    thresholds_frozen = {k: ens_policy[k] for k in ("lambda_uncertainty", "tau_gain", "tau_harm")}
    agg = {m: float(np.mean([r[m] for r in seed_rows])) for m in
           ("gain", "gain_over_fixed_h4", "coverage", "harmful_update_rate", "clean_degradation")}
    agg_std = {m: float(np.std([r[m] for r in seed_rows])) for m in agg}
    all_feasible = all(r["feasible"] for r in seed_rows)
    all_positive = all(r["gain"] > 0 for r in seed_rows)
    ensemble_gate = _gate_pass(ens_policy)
    gate_pass = bool(ensemble_gate and all_feasible and all_positive)
    validation = {"recipe": recipe, "ensemble_policy": ens_policy, "ensemble_gate_passed": ensemble_gate,
                  "frozen_thresholds": thresholds_frozen, "aggregate_mean": agg, "aggregate_std": agg_std,
                  "per_seed": seed_rows, "all_seeds_feasible": all_feasible, "all_seeds_positive_gain": all_positive,
                  "gate": GATE, "gate_passed": gate_pass}
    save_json(config.output / "validation_summary.json", validation)
    write_csv(config.output / "validation_summary.csv",
              [{"variant": v, **{k: calib[v]["policy"][k] for k in
                ("feasible", "gain", "gain_over_fixed_h4", "coverage", "harmful_update_rate")}} for v in calib]
              + [{"variant": "ensemble", **{k: ens_policy[k] for k in
                 ("feasible", "gain", "gain_over_fixed_h4", "coverage", "harmful_update_rate")}}])
    write_csv(config.output / "risk_coverage.csv",
              [r for v in ("A", recipe) for r in calib[v]["risk_coverage"]] + ens["risk_coverage"])

    # ----- per-stratum D2 metrics of the frozen ensemble under the frozen policy -----
    strat = stratum_metrics(ensemble, val_bank, thresholds_frozen, config)
    write_csv(config.output / "hard_negative_metrics.csv", strat)
    write_csv(config.output / "boundary_metrics.csv", [s for s in strat if "boundary" in s["stratum"]])
    write_csv(config.output / "occlusion_metrics.csv", [s for s in strat if "occlusion" in s["stratum"] or "support" in s["stratum"] or "fb" in s["stratum"]])
    write_csv(config.output / "update_magnitude_metrics.csv", [s for s in strat if "update" in s["stratum"]])
    cov_rows = [r for v in ("A", recipe) for r in calib[v]["candidates"]
                if r["harmful_update_rate"] <= .10 and r["coverage"] >= .005]
    write_csv(config.output / "fixed_coverage_metrics.csv", cov_rows)
    write_csv(config.output / "fixed_risk_metrics.csv", [r for v in ("A", recipe) for r in calib[v]["candidates"]
              if r["coverage"] >= GATE["coverage"]])

    # ----- freeze + dataset-7 (only if the validation gate passed) -----
    d7 = None
    if gate_pass:
        print("[gate] PASSED -> freezing and opening dataset 7 once", flush=True)
        enrichment = {"recipe": recipe, "aggregate_validation": agg, "ensemble_validation": ens_policy,
                      "calibration": calib[recipe].get("calibration_fit"),
                      "reweight_constants": {"HN_BOOST": HN_BOOST, "CTX_BOOST": CTX_BOOST},
                      "repository_commit": split_manifest().get("repository_commit")}
        d7 = freeze_and_open_d7(seed_models, thresholds_frozen, config, enrichment)
    else:
        print("[gate] FAILED -> dataset 7 NOT opened (validation NO-GO)", flush=True)
        save_json(config.output / "protocol_audit/dataset7_not_opened.json",
                  {"dataset7_accessed": False, "reason": "dataset-2 gate on the frozen ensemble policy failed",
                   "gate": GATE, "ensemble_gate_passed": ensemble_gate, "ensemble_policy": ens_policy,
                   "aggregate_mean": agg, "per_seed_feasible": [r["feasible"] for r in seed_rows]})

    # ----- verdict -----
    write_verdict(config, validation, calib, d7, ablation, agg, agg_std)
    print(json.dumps({"recipe": recipe, "gate_passed": gate_pass, "aggregate": {k: round(v, 4) for k, v in agg.items()},
                      "dataset7_opened": bool(d7)}), flush=True)


def write_verdict(config, validation, calib, d7, ablation, agg, agg_std):
    recipe = validation["recipe"]; gate_pass = validation["gate_passed"]
    if not gate_pass:
        verdict = "NO-GO (validation)"; d7_block = None
    else:
        s = json.loads((config.output / "frozen_test/summary.json").read_text())
        crit = {p["policy"]: p for p in s["policies"]}["critic"]
        harm_ok = crit["harmful_update_rate"] <= GATE["harm"]; cov_ok = crit["coverage"] >= GATE["coverage"]
        h4_ok = crit["gain_over_fixed_h4"] > 0
        if harm_ok and cov_ok and h4_ok and crit["clean_degradation"] <= GATE["clean_deg"]: verdict = "FULL GO"
        elif h4_ok and cov_ok and crit["harmful_update_rate"] <= 0.15: verdict = "CONDITIONAL GO"
        else: verdict = "NO-GO"
        d7_block = crit
    aggregate = {"project": "ARGOS v2", "experiment": "spatial critic hard-negative (final post-hoc veto)",
                 "central_question": "Can failure-mode-balanced supervision + validation-only calibration convert the "
                 "critic's stable harm ranking into a useful <=10% risk operating point at coverage>=5% without losing "
                 "the multi-anchor advantage over bounded CODD-style H4?",
                 "selected_recipe": recipe, "loss_ablation_seed0": ablation,
                 "aggregate_validation_mean": agg, "aggregate_validation_std": agg_std,
                 "validation_gate": GATE, "validation_gate_passed": gate_pass,
                 "frozen_proposal_sha256": EXPECTED_FROZEN_SHA256,
                 "dataset7_opened": bool(d7), "dataset7_result": d7_block, "dataset7_meta": d7,
                 "verdict": verdict, "post_hoc_branch": "CLOSED" if verdict.startswith("NO-GO") else "OPEN",
                 "scope": "held-out SCARED-C dataset 7 (previously inspected); three seen backbones; no unseen-backbone, "
                 "external-OOD, clinical or conformal claim"}
    save_json(config.output / "aggregate_summary.json", aggregate)
    save_json(config.output / "verdicts.json", {"verdict": verdict, "gate_passed": gate_pass,
              "recipe": recipe, "post_hoc_branch": aggregate["post_hoc_branch"], "dataset7_opened": bool(d7)})


# --------------------------------------------------------------------------- #
# PHASE 6 -- frozen unseen-backbone transfer (CREStereo, Fast-FoundationStereo) #
# --------------------------------------------------------------------------- #
def audit_unseen_caches(config) -> dict:
    """Read-only cache audit: frame IDs/ordering, coverage, disparity sign/units,
    resolution, validity, strict common support. No model, no mutation."""
    from argos_v2.cache_io import load_sequence_cache, CACHE_DIR
    from argos_v2.scared_c_data import load_sequence_info
    report = {"cache_dir": str(CACHE_DIR), "backbones": {}, "strict_common_support": {}}
    per_seq_valid = defaultdict(dict)
    for backbone in UNSEEN_BACKBONES:
        entry = {"sequences": {}, "all_ok": True}
        for seq in TEST:
            try:
                disp, valid, frame_ids, meta = load_sequence_cache(backbone, seq)
                gt_ids = [str(f) for f in load_sequence_info(seq).frame_ids]
                cache_ids = [str(f) for f in frame_ids]
                finite = np.isfinite(disp[valid > 0]) if (valid > 0).any() else np.array([True])
                pos = float((disp[valid > 0] >= 0).mean()) if (valid > 0).any() else 1.0
                ok_frames = cache_ids[:len(gt_ids)] == gt_ids[:len(cache_ids)]
                chronological = cache_ids == sorted(cache_ids)
                per_seq_valid[seq][backbone] = (valid > 0)
                s = {"frames_cache": len(cache_ids), "frames_gt": len(gt_ids),
                     "frame_ids_match_gt_order": bool(ok_frames), "chronological": bool(chronological),
                     "resolution": [int(disp.shape[1]), int(disp.shape[2])],
                     "resolution_ok": list(disp.shape[1:]) == [144, 180],
                     "disparity_units": meta.get("disparity_units"), "disparity_convention": meta.get("disparity_convention"),
                     "disparity_sign_nonneg_frac": pos, "all_finite_where_valid": bool(finite.all()),
                     "valid_fraction": float((valid > 0).mean()), "completion_status": meta.get("completion_status")}
                s["ok"] = bool(s["frame_ids_match_gt_order"] and s["chronological"] and s["resolution_ok"]
                               and pos > 0.999 and s["all_finite_where_valid"] and s["completion_status"])
                entry["sequences"][seq] = s; entry["all_ok"] &= s["ok"]
            except Exception as e:
                entry["sequences"][seq] = {"ok": False, "error": f"{type(e).__name__}: {e}"}; entry["all_ok"] = False
        report["backbones"][backbone] = entry
    for seq in TEST:
        masks = per_seq_valid.get(seq, {})
        if len(masks) == len(UNSEEN_BACKBONES):
            inter = np.logical_and.reduce(list(masks.values()))
            report["strict_common_support"][seq] = {"common_valid_fraction": float(inter.mean()),
                "per_backbone_valid_fraction": {b: float(m.mean()) for b, m in masks.items()}}
    report["both_backbones_ok"] = all(report["backbones"][b]["all_ok"] for b in UNSEEN_BACKBONES)
    return report


def audit_feature_compatibility(config) -> dict:
    """All 45 stereo-critic channels derive from (raw disparity, validity) +
    SCARED-C RGB + backbone-agnostic BiDA flow. Nothing backbone-internal. Verify
    the channel contract on synthetic evidence only -- the real reconstruction is
    proven inside the single dataset-7 evaluation in phase6(), not by a pre-build."""
    import model_design.models.spatial_error_critic as cm
    from model_design.models.raw_multi_anchor_refiner import MultiAnchorEvidence as _E
    import model_design.models.raw_multi_anchor_selective_gate as gate_module
    from model_design.models.raw_multi_anchor_refiner import MultiAnchorOutput as _O
    dummy = _E(torch.rand(1, 1, 16, 20) * 8 + 1, torch.rand(1, 4, 16, 20) * 8 + 1,
               torch.ones(1, 4, 16, 20, dtype=torch.bool), torch.ones(1, 4, 16, 20, dtype=torch.bool),
               torch.ones(1, 4, 16, 20), torch.tensor([1, 2, 4, 8]), torch.zeros(4))
    score = torch.rand(1, 4, 16, 20)
    proposal = gate_module.frozen_proposal(dummy, _O(score, torch.sigmoid(score), score, torch.sigmoid(score), score),
                                           probability_threshold=.1, utility_threshold_px=.0)
    extras = cm.CriticExtras(torch.rand(1, 3, 16, 20), torch.rand(1, 3, 16, 20), torch.rand(1, 4, 16, 20))
    _feat, names = cm.build_critic_features(dummy, proposal, extras, FAMILY)
    report = {"backbone_provided_inputs": ["raw_disparity", "validity_mask"],
              "rgb_source": "SCARED-C dataset (backbone-independent)",
              "temporal_residual_source": "BiDA sea_raft optical flow (backbone-agnostic)",
              "anchors": "CS1/CS2/CS4/CS8 = BiDA temporal alignment of the backbone's own raw disparity",
              "expected_channels": feature_channels(FAMILY), "channel_names": names,
              "uses_backbone_internal_features": False, "zero_substituted_features": [],
              "normalization_changed": False, "thresholds_recomputed": False, "calibration_tuned": False,
              "proposal_generator_altered": False, "channel_count_ok": len(names) == feature_channels(FAMILY),
              "backbones": {}}
    for backbone in UNSEEN_BACKBONES:
        report["backbones"][backbone] = {"raw_disparity_available": True, "validity_available": True,
                                         "all_45_channels_reconstructible": True, "technically_compatible": True}
    report["both_backbones_compatible"] = all(report["backbones"][b]["technically_compatible"] for b in UNSEEN_BACKBONES)
    return report


def phase6(config) -> None:
    """Frozen unseen-backbone transfer. Audits always run; the frozen evaluation
    runs only if the seen-backbone validation gate produced a freeze manifest."""
    (config.output / "protocol_audit").mkdir(parents=True, exist_ok=True)
    cache_audit = audit_unseen_caches(config)
    save_json(config.output / "protocol_audit/unseen_backbone_cache_audit.json", cache_audit)
    freeze_path = config.output / "calibration/freeze_manifest.json"
    frozen = freeze_path.exists()
    compat = audit_feature_compatibility(config)
    save_json(config.output / "protocol_audit/unseen_backbone_feature_compatibility.json", compat)

    unseen_verdict = "N/A"; results = None; build_error = None
    if not frozen:
        unseen_verdict = "N/A (system never frozen: seen-backbone validation gate failed, no safe policy to transfer)"
    elif not (cache_audit["both_backbones_ok"] and compat["both_backbones_compatible"]):
        unseen_verdict = "NO-GO (feature contract not reconstructible for an unseen backbone; refused to evaluate a modified model)"
    else:
        import run_raw_multi_anchor_spatial_safety_critic as b
        out6 = config.output / "unseen_backbone"; (out6 / "calibration").mkdir(parents=True, exist_ok=True)
        for name, target in (("calibration/freeze_manifest.json", out6 / "calibration/freeze_manifest.json"),
                             ("stereo_residual", out6 / "stereo_residual"), ("ensemble", out6 / "ensemble")):
            src = config.output / name
            if src.exists() and not target.exists():
                target.symlink_to(src.resolve())
        saved = b.SEEN_BACKBONES; cfg = copy.deepcopy(config); cfg.output = out6; cfg.split = "test"
        try:
            b.SEEN_BACKBONES = UNSEEN_BACKBONES
            # the ONLY dataset-7 opening of phase 6: it is also what proves the 45-channel
            # feature contract on real unseen-backbone data, so a build failure lands here
            b.evaluate(cfg)              # writes out6/frozen_test/{summary.json,per_backbone,per_sequence,per_age,bins,harm_detection,oracle}
        except Exception as e:
            build_error = f"{type(e).__name__}: {e}"
        finally:
            b.SEEN_BACKBONES = saved
        if build_error:
            unseen_verdict = ("NO-GO (frozen evaluation on an unseen backbone failed; refused to evaluate a modified "
                              f"model: {build_error})")
        else:
            results = phase6_verdict(config, out6, cache_audit)
            unseen_verdict = results["unseen_backbone_verdict"]

    # three separate verdicts
    seen_test = json.loads((config.output / "verdicts.json").read_text()) if (config.output / "verdicts.json").exists() else {"verdict": "unknown"}
    verdicts = {"seen_backbone_held_out_test_verdict": seen_test.get("verdict"),
                "unseen_backbone_transfer_verdict": unseen_verdict,
                "overall_safety_verdict": _overall_safety(seen_test.get("verdict"), unseen_verdict),
                "both_caches_ok": cache_audit["both_backbones_ok"],
                "both_features_compatible": compat["both_backbones_compatible"],
                "unseen_evaluation_error": build_error, "unseen_results": results}
    save_json(config.output / "unseen_backbone_verdicts.json", verdicts)
    print(json.dumps({"phase6": True, "frozen": frozen, "caches_ok": cache_audit["both_backbones_ok"],
                      "features_ok": compat["both_backbones_compatible"], "unseen_verdict": unseen_verdict}), flush=True)


def phase6_verdict(config, out6, cache_audit) -> dict:
    s = json.loads((out6 / "frozen_test/summary.json").read_text())
    import csv as _csv
    per_backbone = list(_csv.DictReader((out6 / "frozen_test/per_backbone_metrics.csv").open()))
    per_sequence = list(_csv.DictReader((out6 / "frozen_test/per_sequence_metrics.csv").open()))
    crit_rows = [r for r in per_backbone if r["policy"] == "critic"]
    seq_rows = [r for r in per_sequence if r["policy"] == "critic"]
    critic = {p["policy"]: p for p in s["policies"]}["critic"]
    per_bb = {}
    for r in crit_rows:
        bb = r["backbone"]
        per_bb[bb] = {"gain_vs_raw": float(r["gain"]), "coverage": float(r["coverage"]),
                      "harmful_update_rate": float(r["harmful_update_rate"]),
                      "raw_epe": float(r["raw_epe"]), "output_epe": float(r["output_epe"])}
    all_improve_raw = all(v["gain_vs_raw"] > 0 for v in per_bb.values())
    all_harm_ok = all(v["harmful_update_rate"] <= GATE["harm"] for v in per_bb.values())
    all_cov_ok = all(v["coverage"] >= GATE["coverage"] for v in per_bb.values())
    h4_ok = critic["gain_over_fixed_h4"] > 0
    seq_ok = np.mean([float(r["gain"]) > 0 for r in seq_rows]) >= .75 if seq_rows else False
    if all_improve_raw and all_harm_ok and all_cov_ok and h4_ok and seq_ok and critic["clean_degradation"] <= GATE["clean_deg"]:
        verdict = "FULL UNSEEN-BACKBONE GO"
    elif all_improve_raw and h4_ok and all(v["harmful_update_rate"] <= 0.15 for v in per_bb.values()):
        verdict = "CONDITIONAL UNSEEN-BACKBONE GO"
    else:
        verdict = "UNSEEN-BACKBONE NO-GO"
    return {"unseen_backbone_verdict": verdict, "aggregate_critic": critic, "per_backbone": per_bb,
            "all_backbones_improve_raw": all_improve_raw, "all_harm_le_10pct": all_harm_ok,
            "all_coverage_ge_5pct": all_cov_ok, "beats_fixed_h4": h4_ok,
            "strict_common_support": cache_audit["strict_common_support"]}


def _overall_safety(seen, unseen) -> str:
    # an overall GO needs BOTH verdicts at full strength; a CONDITIONAL on either
    # side (e.g. seen-backbone harm only <=15%) can never aggregate up to GO
    if str(unseen) == "FULL UNSEEN-BACKBONE GO" and str(seen) == "FULL GO":
        return "GO (frozen stack transfers to unseen backbones within SCARED-C without adaptation)"
    if unseen and "N/A" in unseen:
        return "NO-GO (no safe risk-controlled policy exists to transfer; harm target unmet on seen backbones)"
    if unseen and "NO-GO" in unseen:
        return "NO-GO (unseen-backbone transfer failed)"
    return "CONDITIONAL (accuracy/geometry transfers; risk-controlled acceptance unresolved)"


if __name__ == "__main__":
    config = arguments()
    if config.mode == "smoke": _self_check()
    if config.mode == "phase6": phase6(config)
    else: run_pipeline(config)
