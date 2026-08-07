#!/usr/bin/env python3
"""Train/calibrate/evaluate the ARGOS v2 spatial comparative error critic.

Veto-only post-hoc adapter over the frozen raw multi-anchor soft proposal.
Configurations are cumulative evidence families: geometry (C), temporal (D),
stereo (E, primary), plane_sweep (F, controlled ablation), plus a 3-seed
ensemble (G) of the validation-selected family.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, roc_auc_score
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from argos_v2.cache_io import load_sequence_cache  # noqa: E402
from model_design.data.temporal_pair_dataset import SEEN_BACKBONES  # noqa: E402
from model_design.external_components.bidavideo import BiDAFlowInferenceAdapter, temporal_disparity_evidence  # noqa: E402
from model_design.models.raw_multi_anchor_selective_gate import apply_veto, frozen_proposal  # noqa: E402
from model_design.models.spatial_error_critic import (  # noqa: E402
    FAMILIES, CriticExtras, SpatialErrorCritic, build_critic_features, critic_losses,
    feature_channels, selective_accept,
)
from run_raw_multi_anchor_selective_gate import (  # noqa: E402
    EXPECTED_FROZEN_SHA256, FIXED_H4_EPE, FROZEN_CHECKPOINT, RAW_BANK_ORACLE_GAIN, ece,
    frozen_policy, load_frozen,
)
from run_raw_multi_anchor_temporal_refiner import (  # noqa: E402
    RAW_AGES, TEST, TRAIN, VALIDATION, AnchorBankDataset, _aligned_gt_age1,
    _load_sequence_inputs, clean, evidence_from_batch, loader, save_json, seed_all,
    sha256, split_manifest, to_device, write_csv,
)
from run_hybrid_temporal_memory_oracle_audit import CHECKPOINT as CORRECTED_CHECKPOINT, fused_history  # noqa: E402
from run_hybrid_temporal_memory_oracle_audit import load_model as load_corrected_model  # noqa: E402
from run_ppmstereo_validation import infer_age_flows, rgb_tensor  # noqa: E402
from argos_v2.scared_c_data import load_frame_lr, load_sequence_info  # noqa: E402


OUTPUT = ROOT / "results/raw_multi_anchor_spatial_safety_critic"
HARM_MARGINS = (0.0, 0.05, 0.10)
ENSEMBLE_SEEDS = (20260722, 20260723, 20260724)
FAMILY_DIRS = {"geometry": "geometry_only", "temporal": "temporal_residual",
               "stereo": "stereo_residual", "plane_sweep": "plane_sweep_ablation"}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "overfit", "smoke", "train", "calibrate", "evaluate", "summarize"), required=True)
    parser.add_argument("--families", default="geometry,temporal,stereo")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=min(48, os.cpu_count() or 8))
    parser.add_argument("--flow-batch-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--harm-margin", type=float, default=.10)
    parser.add_argument("--lambda-harm", type=float, default=1.0)
    parser.add_argument("--lambda-rank", type=float, default=.1)
    parser.add_argument("--coverage-threshold", type=float, default=.50)
    parser.add_argument("--seed", type=int, default=ENSEMBLE_SEEDS[0])
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--sample-stride", type=int, default=8)
    return parser.parse_args()


def family_root(config: argparse.Namespace, family: str, seed: int | None = None) -> Path:
    root = config.output / FAMILY_DIRS[family]
    if seed is not None and seed != ENSEMBLE_SEEDS[0]:
        root = config.output / "ensemble" / f"{family}_seed{seed}"
    return root


@torch.inference_mode()
def _align_raw_anchors_with_photo(raw, valid, images, current_indices, flows, device, batch_size):
    """Identical alignment path to the validated refiner bank, additionally
    retaining the robust temporal photometric residual per age."""
    stores = {key: [] for key in ("candidates", "valid", "support", "fb", "photo")}
    for start in range(0, len(current_indices), batch_size):
        query = current_indices[start:start + batch_size]; per_age = defaultdict(list)
        for age in RAW_AGES:
            positions = np.arange(start, start + len(query))
            current = torch.from_numpy(np.stack([raw[index] for index in query]))[:, None].float().to(device)
            source = torch.from_numpy(np.stack([raw[index - age] for index in query]))[:, None].float().to(device)
            current_valid = torch.from_numpy(np.stack([valid[index] for index in query]))[:, None].bool().to(device)
            source_valid = torch.from_numpy(np.stack([valid[index - age] for index in query]))[:, None].bool().to(device)
            evidence = temporal_disparity_evidence(
                current, source, torch.from_numpy(flows[age][0][positions]).float().to(device),
                torch.from_numpy(flows[age][1][positions]).float().to(device),
                current_valid=current_valid, past_valid=source_valid,
                current_rgb=torch.stack([images[index] for index in query]),
                past_rgb=torch.stack([images[index - age] for index in query]),
            )
            per_age["candidates"].append(evidence.aligned_past_disparity[:, 0].cpu().numpy())
            per_age["valid"].append(evidence.aligned_validity[:, 0].cpu().numpy())
            per_age["support"].append(evidence.warp_support[:, 0].cpu().numpy())
            per_age["fb"].append(evidence.forward_backward_confidence[:, 0].cpu().numpy())
            per_age["photo"].append(evidence.photometric_residual[:, 0].cpu().numpy())
        for key in stores: stores[key].append(np.stack(per_age[key], axis=1))
    return {key: np.concatenate(values, axis=0) for key, values in stores.items()}


def build_critic_bank(config: argparse.Namespace, sequences: tuple[str, ...], *, training: bool) -> AnchorBankDataset:
    """Full-resolution (144x180) bank with left/right RGB and per-age temporal
    photometric residuals. Full frames instead of crops: the stereo residual
    needs complete rows, the frames are tiny, and RAM is plentiful."""
    device = torch.device(config.device); adapter = BiDAFlowInferenceAdapter("sea_raft", device=device)
    assert not any(parameter.requires_grad for parameter in adapter.model.parameters())
    corrected_model = None
    if not training:
        corrected_model, _ = load_corrected_model(device)
    fields = defaultdict(list); metadata = []
    for sequence in sequences:
        info, frame_ids, images, gt, coverage = _load_sequence_inputs(sequence, device, config.max_frames)
        if len(frame_ids) <= 8: continue
        right_images = torch.stack([rgb_tensor(load_frame_lr(info, frame_id)[1], device) for frame_id in frame_ids])
        left_images = torch.stack(images)
        current_indices = list(range(8, len(frame_ids)))
        flows, _latency, _peak = infer_age_flows(adapter, images, current_indices, list(RAW_AGES), config.flow_batch_size, device)
        gt_memory, gt_memory_valid = _aligned_gt_age1(gt, coverage, current_indices, flows, device, config.flow_batch_size)
        for backbone in SEEN_BACKBONES:
            raw_mmap, valid_mmap, cache_ids, _ = load_sequence_cache(backbone, sequence)
            if [str(value) for value in cache_ids[:len(frame_ids)]] != frame_ids: raise RuntimeError(f"frame-ID mismatch {backbone}/{sequence}")
            raw = np.asarray(raw_mmap[:len(frame_ids)], dtype=np.float32); valid = np.asarray(valid_mmap[:len(frame_ids)]) > 0
            maps = _align_raw_anchors_with_photo(raw, valid, images, current_indices, flows, device, config.flow_batch_size)
            if corrected_model is not None:
                history_indices = list(range(1, len(frame_ids)))
                history_flows, _, _ = infer_age_flows(adapter, images, history_indices, [1], config.flow_batch_size, device)
                corrected = fused_history(corrected_model, adapter, images, raw, valid, history_flows[1], device)
                del history_flows
            else:
                corrected = None
            for position, index in enumerate(current_indices):
                fields["raw"].append(raw[index][None].astype(np.float16))
                fields["candidates"].append(maps["candidates"][position].astype(np.float16))
                fields["candidate_valid"].append(maps["valid"][position].astype(np.uint8))
                fields["warp_support"].append(maps["support"][position].astype(np.uint8))
                fields["fb_confidence"].append(maps["fb"][position].astype(np.float16))
                fields["photo"].append(maps["photo"][position].astype(np.float16))
                fields["left_rgb"].append(left_images[index].cpu().numpy().astype(np.uint8))
                fields["right_rgb"].append(right_images[index].cpu().numpy().astype(np.uint8))
                fields["gt"].append(gt[index][None].astype(np.float16))
                fields["gt_coverage"].append(coverage[index][None].astype(np.float16))
                fields["raw_valid"].append(valid[index][None].astype(np.uint8))
                fields["gt_memory"].append(gt_memory[position][None].astype(np.float16))
                fields["gt_memory_valid"].append(gt_memory_valid[position][None].astype(np.uint8))
                if corrected is not None:
                    fields["fixed_h4_output"].append(corrected[index][None].astype(np.float16))
                metadata.append({"backbone": backbone, "sequence": sequence, "frame_id": frame_ids[index],
                                 "frame_index": index, "ages": list(RAW_AGES), "provenance": [0.0] * 4})
        del images, left_images, right_images, gt, coverage, flows
        if device.type == "cuda": torch.cuda.empty_cache()
        print(f"CRITIC BANK sequence={sequence} examples={len(metadata)}", flush=True)
    tensors = {key: torch.from_numpy(np.stack(values)) for key, values in fields.items()}
    return AnchorBankDataset(tensors, metadata)


def batch_proposal(batch: dict, frozen, base_policy: dict, device: torch.device):
    evidence = evidence_from_batch(batch, list(RAW_AGES), [0.0] * 4)
    with torch.no_grad():
        output = frozen(evidence)
        proposal = frozen_proposal(evidence, output, probability_threshold=base_policy["probability_threshold"],
                                  utility_threshold_px=base_policy["utility_threshold_px"])
    extras = CriticExtras(left_rgb=batch["left_rgb"].float(), right_rgb=batch["right_rgb"].float(),
                          temporal_residual=batch["photo"].float())
    return evidence, proposal, extras


def supervision(batch: dict, evidence, proposal, config: argparse.Namespace):
    gt = batch["gt"].float()
    e_raw = (evidence.raw - gt).abs()
    e_proposed = (proposal.proposed - gt).abs()
    valid = (batch["gt_coverage"].float() > config.coverage_threshold) & batch["raw_valid"].bool()
    return e_raw, e_proposed, valid, valid & proposal.eligible


@torch.inference_mode()
def validation_pass(model, dataset, frozen, base_policy, config, family: str) -> dict:
    """Sampled validation diagnostics plus the best constrained selective gain."""
    device = torch.device(config.device); model.eval(); values = defaultdict(list)
    for cpu in loader(dataset, config, training=False):
        batch = to_device(cpu, device)
        evidence, proposal, extras = batch_proposal(batch, frozen, base_policy, device)
        features, _ = build_critic_features(evidence, proposal, extras, family)
        output = model(features)
        e_raw, e_proposed, _valid, supervised = supervision(batch, evidence, proposal, config)
        flat = supervised.flatten().nonzero().flatten()[::max(config.sample_stride, 1)]
        take = lambda tensor: tensor.flatten()[flat].float().cpu().numpy()
        values["delta"].append(take(e_raw - e_proposed)); values["delta_hat"].append(take(output.delta_hat))
        values["sigma"].append(take(output.sigma_delta)); values["harm_prob"].append(take(output.harm_probability))
        values["e_raw"].append(take(e_raw)); values["mu_raw"].append(take(output.mu_raw))
        values["e_prop"].append(take(e_proposed)); values["mu_prop"].append(take(output.mu_proposed))
    arrays = {key: np.concatenate(items) for key, items in values.items()}
    label = arrays["delta"] < -config.harm_margin
    result = {"harm_auroc": float(roc_auc_score(label, arrays["harm_prob"])) if label.any() and not label.all() else math.nan,
              "harm_auprc": float(average_precision_score(label, arrays["harm_prob"])) if label.any() else math.nan,
              "raw_error_mae": float(mean_absolute_error(arrays["e_raw"], arrays["mu_raw"])),
              "proposal_error_mae": float(mean_absolute_error(arrays["e_prop"], arrays["mu_prop"])),
              "delta_mae": float(mean_absolute_error(arrays["delta"], arrays["delta_hat"])),
              "delta_spearman": float(spearmanr(arrays["delta"], arrays["delta_hat"]).statistic),
              "sample_count": int(len(label)), "harm_fraction": float(label.mean())}
    # Checkpoint selection needs a smooth per-epoch signal. A hard-constrained
    # (harm<=10%, coverage>=1%) argmax is a step function that is frequently
    # infeasible early in training (ties at 0.0 forever, so the strict
    # score>best comparison in train_families would freeze on epoch 1). Track
    # the constrained safe operating point for diagnostics, but select
    # checkpoints on the unconstrained net-utility optimum (coverage>=0.5%
    # only), which improves monotonically as delta_hat/harm calibrate.
    unconstrained = (-math.inf, 0.0, 0.0)
    constrained = (-math.inf, 0.0, 0.0)
    for lam in (0.0, 0.5, 1.0, 2.0):
        lcb = arrays["delta_hat"] - lam * arrays["sigma"]
        for tau_gain in (-.05, 0.0, .025, .05, .10, .20):
            for tau_harm in (.2, .3, .4, .5, .6, .7):
                accepted = (lcb > tau_gain) & (arrays["harm_prob"] < tau_harm)
                coverage = float(accepted.mean())
                if coverage < .005: continue
                harm = float(label[accepted].mean()); gain = float((arrays["delta"] * accepted).mean())
                if gain > unconstrained[0]: unconstrained = (gain, harm, coverage)
                if harm <= .10 and coverage >= .01 and gain > constrained[0]: constrained = (gain, harm, coverage)
    result.update({"best_selective_gain": unconstrained[0] if math.isfinite(unconstrained[0]) else 0.0,
                   "best_selective_harm": unconstrained[1], "best_selective_coverage": unconstrained[2],
                   "best_constrained_gain": constrained[0] if math.isfinite(constrained[0]) else 0.0,
                   "best_constrained_harm": constrained[1], "best_constrained_coverage": constrained[2]})
    return result


def train_families(config: argparse.Namespace, tiny: str | None = None) -> None:
    seed_all(config.seed); families = [name for name in config.families.split(",") if name]
    for family in families:
        if family not in FAMILIES: raise ValueError(f"unknown family {family}")
    sequences = (TRAIN[0],) if tiny else TRAIN
    original = config.max_frames
    if tiny: config.max_frames = 20
    train_data = build_critic_bank(config, sequences, training=True)
    if tiny == "overfit": validation_data = train_data
    else:
        validation_data = build_critic_bank(config, (VALIDATION[0],) if tiny else VALIDATION, training=True)
    config.max_frames = original
    device = torch.device(config.device); frozen, _state = load_frozen(device); base_policy = frozen_policy()
    for family in families:
        seed_all(config.seed)
        root = family_root(config, family, config.seed); (root / "checkpoints").mkdir(parents=True, exist_ok=True)
        model = SpatialErrorCritic(feature_channels(family), channels=config.channels).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        epochs = 40 if tiny == "overfit" else 6 if tiny == "smoke" else config.epochs
        data = loader(train_data, config, training=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(epochs * len(data), 1), eta_min=config.learning_rate * .05)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        history = []; best = -math.inf; first = None; started = time.perf_counter()
        for epoch in range(epochs):
            model.train(); data.balanced_sampler.set_epoch(epoch)
            totals = defaultdict(float); steps = 0
            for cpu in data:
                batch = to_device(cpu, device)
                evidence, proposal, extras = batch_proposal(batch, frozen, base_policy, device)
                e_raw, e_proposed, _valid, supervised = supervision(batch, evidence, proposal, config)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    features, _ = build_critic_features(evidence, proposal, extras, family)
                    output = model(features.float())
                    losses = critic_losses(output, e_raw, e_proposed, supervised, harm_margin=config.harm_margin,
                                           lambda_harm=config.lambda_harm, lambda_rank=config.lambda_rank)
                optimizer.zero_grad(set_to_none=True); scaler.scale(losses["total"]).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); scaler.step(optimizer); scaler.update(); scheduler.step()
                if first is None: first = float(losses["total"].detach())
                for key, value in losses.items(): totals[key] += float(value if isinstance(value, float) else value.detach())
                steps += 1
            diagnostics = validation_pass(model, validation_data, frozen, base_policy, config, family)
            row = {"family": family, "seed": config.seed, "epoch": epoch + 1,
                   **{key: value / max(steps, 1) for key, value in totals.items()},
                   **{f"validation_{key}": value for key, value in diagnostics.items()},
                   "learning_rate": scheduler.get_last_lr()[0]}
            history.append(row); score = diagnostics["best_selective_gain"]
            payload = {"project": "ARGOS v2", "family": family, "seed": config.seed, "model": model.state_dict(),
                       "channels": config.channels, "in_channels": feature_channels(family),
                       "harm_margin": config.harm_margin, "frozen_checkpoint_sha256": EXPECTED_FROZEN_SHA256,
                       "split": split_manifest(), "epoch": epoch + 1}
            torch.save(payload, root / "checkpoints/last.pt")
            if score > best: best = score; torch.save(payload, root / "checkpoints/best_validation.pt")
            print(json.dumps(clean(row)), flush=True)
        write_csv(root / "training_history.csv", history)
        summary = {"family": family, "seed": config.seed,
                   "parameters": sum(parameter.numel() for parameter in model.parameters()),
                   "initial_loss": first, "final_loss": history[-1]["total"],
                   "best_validation_selective_gain": best, "wall_time_s": time.perf_counter() - started}
        if tiny:
            best_auprc = max(row["validation_harm_auprc"] for row in history if math.isfinite(row["validation_harm_auprc"]))
            required = .55 if tiny == "overfit" else .90
            summary["best_validation_harm_auprc"] = best_auprc
            summary["passed"] = bool(math.isfinite(history[-1]["total"]) and (history[-1]["total"] < float(first) * required))
            save_json(root / f"{tiny}_summary.json", summary)
            if not summary["passed"]: raise RuntimeError(f"{family} {tiny} loss reduction insufficient")
        else: save_json(root / "training_summary.json", summary)


def load_critic(config: argparse.Namespace, family: str, seed: int, device: torch.device):
    path = family_root(config, family, seed) / "checkpoints/best_validation.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = SpatialErrorCritic(int(state["in_channels"]), channels=int(state["channels"])).to(device)
    model.load_state_dict(state["model"]); model.eval()
    return model, path


class EnsembleCritic:
    """Mean-aggregated 3-seed ensemble with epistemic variance folded into sigma."""

    def __init__(self, models: list[SpatialErrorCritic]) -> None:
        self.models = models

    def __call__(self, features: torch.Tensor):
        outputs = [model(features) for model in self.models]
        from model_design.models.spatial_error_critic import CriticOutput
        mu_raw = torch.stack([o.mu_raw for o in outputs]).mean(0)
        mu_prop = torch.stack([o.mu_proposed for o in outputs]).mean(0)
        s_raw = torch.stack([o.s_raw for o in outputs]).mean(0)
        s_prop = torch.stack([o.s_proposed for o in outputs]).mean(0)
        harm_logit = torch.stack([o.harm_logit for o in outputs]).mean(0)
        delta_stack = torch.stack([o.delta_hat for o in outputs])
        epistemic = delta_stack.var(0, unbiased=False)
        aleatoric_sq = torch.exp(2 * s_raw) + torch.exp(2 * s_prop)
        # fold epistemic variance into an effective log scale so sigma_delta
        # becomes sqrt(aleatoric^2 + epistemic^2) through the standard property
        total = torch.sqrt(aleatoric_sq + epistemic)
        s_effective = torch.log(total.clamp_min(1e-6) / math.sqrt(2.0))
        return CriticOutput(mu_raw, mu_prop, s_effective, s_effective, harm_logit, torch.sigmoid(harm_logit))

    def eval(self): [model.eval() for model in self.models]; return self


@torch.inference_mode()
def collect_arrays(config: argparse.Namespace, dataset, critics: dict[str, object]) -> dict[str, np.ndarray]:
    """Sampled per-pixel arrays for calibration/diagnostics across critics."""
    device = torch.device(config.device); frozen, _ = load_frozen(device); base_policy = frozen_policy()
    values = defaultdict(list); offset = 0
    for cpu in loader(dataset, config, training=False):
        batch = to_device(cpu, device)
        evidence, proposal, extras = batch_proposal(batch, frozen, base_policy, device)
        e_raw, e_proposed, valid, _sup = supervision(batch, evidence, proposal, config)
        gray = batch["left_rgb"].float().mean(1, keepdim=True) / 255.0
        dx = torch.nn.functional.pad(evidence.raw[..., 1:] - evidence.raw[..., :-1], (0, 1, 0, 0))
        dy = torch.nn.functional.pad(evidence.raw[..., 1:, :] - evidence.raw[..., :-1, :], (0, 0, 0, 1))
        from model_design.models.spatial_error_critic import stereo_residual
        r_stereo_raw, _ = stereo_residual(batch["left_rgb"].float(), batch["right_rgb"].float(), evidence.raw)
        chosen_age = torch.gather(torch.tensor(list(RAW_AGES), device=device)[None, :, None, None]
                                  .expand(len(batch["raw"]), -1, *evidence.raw.shape[-2:]), 1, proposal.chosen)
        tensors = {"valid": valid, "eligible": proposal.eligible, "delta": e_raw - e_proposed,
                   "e_raw": e_raw, "e_prop": e_proposed, "age": chosen_age.float(),
                   "update": (proposal.proposed - evidence.raw).abs(),
                   "fb": torch.gather(evidence.fb_confidence, 1, proposal.chosen),
                   "temporal_residual": torch.gather(batch["photo"].float(), 1, proposal.chosen),
                   "stereo_residual": r_stereo_raw, "gradient": torch.sqrt(dx.square() + dy.square()),
                   "witness": evidence.available.sum(1, keepdim=True).float()}
        for name, critic in critics.items():
            features, _ = build_critic_features(evidence, proposal, extras, name.split("@")[0])
            output = critic(features)
            tensors[name + "_delta_hat"] = output.delta_hat; tensors[name + "_sigma"] = output.sigma_delta
            tensors[name + "_harm"] = output.harm_probability
            tensors[name + "_mu_raw"] = output.mu_raw; tensors[name + "_mu_prop"] = output.mu_proposed
        stride = max(config.sample_stride, 1); pixels = evidence.raw.shape[-2] * evidence.raw.shape[-1]
        selected = np.arange(0, len(batch["raw"]) * pixels, stride, dtype=np.int64)
        for key, tensor in tensors.items(): values[key].append(tensor.reshape(-1)[selected].cpu().numpy())
        values["frame_group"].append((selected // pixels + offset).astype(np.int32))
        offset += len(batch["raw"])
    return {key: np.concatenate(items) for key, items in values.items()}


def policy_metrics(arrays: dict[str, np.ndarray], accepted: np.ndarray, margin: float) -> dict:
    valid = arrays["valid"].astype(bool); accepted = accepted & valid
    delta = arrays["delta"]; raw_error = arrays["e_raw"]
    harmful = accepted & (delta < -margin); helpful = accepted & (delta > margin)
    clean = valid & (raw_error <= .10); clean_deg = accepted & clean & (delta < -margin)
    frame_ids = arrays["frame_group"]; group_count = int(frame_ids.max()) + 1 if len(frame_ids) else 0
    frame_valid = np.bincount(frame_ids, weights=valid.astype(np.float64), minlength=group_count)
    frame_net = np.bincount(frame_ids, weights=(-delta * accepted * valid), minlength=group_count)
    frame_delta = np.divide(frame_net, frame_valid, out=np.zeros_like(frame_net), where=frame_valid > 0)
    gain = float((delta * accepted)[valid].mean()); coverage = float(accepted[valid].mean()); changed = max(int(accepted.sum()), 1)
    return {"gain": gain, "output_epe": float(raw_error[valid].mean() - gain), "coverage": coverage,
            "exact_raw_fallback_frequency": 1 - coverage,
            "intervention_precision": float(helpful.sum() / changed), "harmful_update_rate": float(harmful.sum() / changed),
            "harmful_update_rate_all_valid": float(harmful.sum() / max(int(valid.sum()), 1)),
            "clean_degradation": float(clean_deg.sum() / max(int(clean.sum()), 1)),
            "degraded_frame_fraction": float(np.mean(frame_delta > 0)),
            "worst_frame_degradation": float(frame_delta.max(initial=0.0)),
            "gain_over_fixed_h4": float(FIXED_H4_EPE - (raw_error[valid].mean() - gain)),
            "mean_accepted_update": float(arrays["update"][accepted].mean()) if accepted.any() else 0.0,
            "valid_count": int(valid.sum()), "accepted_count": int(accepted.sum())}


def policy_accept(arrays: dict[str, np.ndarray], name: str, candidate: dict) -> np.ndarray:
    eligible = arrays["eligible"].astype(bool)
    if name == "ungated": return eligible
    lcb = arrays[name + "_delta_hat"] - candidate["lambda_uncertainty"] * arrays[name + "_sigma"]
    return eligible & (lcb > candidate["tau_gain"]) & (arrays[name + "_harm"] < candidate["tau_harm"])


def threshold_grid():
    for lam in (0.0, 0.5, 1.0, 2.0):
        for tau_gain in (-.05, 0.0, .025, .05, .10, .20, .50):
            for tau_harm in (.2, .3, .4, .5, .6, .7, .8):
                yield {"lambda_uncertainty": lam, "tau_gain": tau_gain, "tau_harm": tau_harm}


def detection_metrics(arrays: dict[str, np.ndarray], name: str, margin: float) -> dict:
    mask = arrays["valid"].astype(bool) & arrays["eligible"].astype(bool)
    delta = arrays["delta"][mask]; probability = arrays[name + "_harm"][mask]
    label = delta < -margin
    row = {"model": name, "margin": margin, "sample_count": int(mask.sum()), "harm_fraction": float(label.mean())}
    if label.any() and not label.all():
        row.update({"harm_auroc": float(roc_auc_score(label, probability)),
                    "harm_auprc": float(average_precision_score(label, probability)),
                    "brier": float(brier_score_loss(label, probability)), "ece": float(ece(label, probability))})
    row.update({"raw_error_mae": float(mean_absolute_error(arrays["e_raw"][mask], arrays[name + "_mu_raw"][mask])),
                "proposal_error_mae": float(mean_absolute_error(arrays["e_prop"][mask], arrays[name + "_mu_prop"][mask])),
                "delta_mae": float(mean_absolute_error(delta, arrays[name + "_delta_hat"][mask])),
                "delta_spearman": float(spearmanr(delta, arrays[name + "_delta_hat"][mask]).statistic)})
    return row


def risk_coverage_rows(arrays: dict[str, np.ndarray], name: str, margin: float) -> list[dict]:
    mask = arrays["valid"].astype(bool) & arrays["eligible"].astype(bool)
    delta = arrays["delta"][mask]
    score = arrays[name + "_delta_hat"][mask] - arrays[name + "_sigma"][mask]
    order = np.argsort(score)[::-1]; rows = []
    for coverage in (.01, .02, .05, .10, .20, .30, .50, 1.0):
        count = max(1, int(math.ceil(coverage * len(score)))); chosen = order[:count]
        rows.append({"model": name, "eligible_coverage": coverage, "sample_count": count,
                     "harmful_fraction": float((delta[chosen] < -margin).mean()),
                     "mean_gain": float(delta[chosen].mean()),
                     "selective_risk": float((-delta[chosen]).clip(min=0).mean())})
    risk = np.cumsum(-np.sort(-score) * 0 + (delta[order] < -margin)) / np.arange(1, len(delta) + 1)
    rows.append({"model": name, "eligible_coverage": "aurc", "sample_count": len(delta),
                 "harmful_fraction": float(risk.mean()), "mean_gain": float(delta.mean()), "selective_risk": float(risk.mean())})
    return rows


def calibrate(config: argparse.Namespace) -> None:
    device = torch.device(config.device)
    families = [name for name in config.families.split(",") if name]
    dataset = build_critic_bank(config, VALIDATION, training=True)
    critics: dict[str, object] = {}
    for family in families:
        model, _ = load_critic(config, family, ENSEMBLE_SEEDS[0], device)
        critics[family] = model
    ensemble_family = None
    if all((family_root(config, family, seed) / "checkpoints/best_validation.pt").exists()
           for family in families[-1:] for seed in ENSEMBLE_SEEDS[1:]):
        ensemble_family = families[-1]
        critics[ensemble_family + "@ensemble"] = EnsembleCritic(
            [load_critic(config, ensemble_family, seed, device)[0] for seed in ENSEMBLE_SEEDS])
    arrays = collect_arrays(config, dataset, critics)
    rows = []; policies = {}; detection = []; risk = []
    for name in critics:
        for margin in HARM_MARGINS:
            detection.append(detection_metrics(arrays, name, margin))
        risk.extend(risk_coverage_rows(arrays, name, config.harm_margin))
        candidates = []
        for candidate in threshold_grid():
            metrics = policy_metrics(arrays, policy_accept(arrays, name, candidate), config.harm_margin)
            record = {"policy": name, **candidate, **metrics}; rows.append(record); candidates.append(record)
        feasible = [row for row in candidates if row["harmful_update_rate"] <= .10 and row["clean_degradation"] <= .03
                    and row["degraded_frame_fraction"] <= .25 and row["coverage"] >= .005]
        policies[name] = max(feasible or candidates, key=lambda row: (row["gain"], row["intervention_precision"]))
        policies[name]["feasible"] = bool(feasible)
    rows.append({"policy": "ungated", **policy_metrics(arrays, policy_accept(arrays, "ungated", {}), config.harm_margin)})
    selected_name = max((name for name in critics), key=lambda name: (policies[name]["feasible"], policies[name]["gain"]))
    checkpoint_hashes = {family: sha256(family_root(config, family, ENSEMBLE_SEEDS[0]) / "checkpoints/best_validation.pt")
                         for family in families}
    if ensemble_family:
        for seed in ENSEMBLE_SEEDS[1:]:
            checkpoint_hashes[f"{ensemble_family}_seed{seed}"] = sha256(
                family_root(config, ensemble_family, seed) / "checkpoints/best_validation.pt")
    freeze = {"project": "ARGOS v2", "experiment": "spatial comparative error critic",
              "selected_only_on": list(VALIDATION), "dataset7_accessed": False,
              "frozen_refiner_sha256": EXPECTED_FROZEN_SHA256, "harm_margin": config.harm_margin,
              "constraints": {"harmful_update_rate": .10, "clean_degradation": .03,
                              "degraded_frame_fraction": .25, "minimum_coverage": .005},
              "selected_policy": selected_name, "policies": policies,
              "critic_checkpoint_hashes": checkpoint_hashes}
    write_csv(config.output / "validation_summary.csv", rows)
    write_csv(config.output / "risk_coverage.csv", risk)
    write_csv(config.output / "harm_detection_metrics.csv", detection)
    fixed = []
    for name in critics:
        for target in (.05, .10, .15):
            feasible = [row for row in rows if row.get("policy") == name and row["harmful_update_rate"] <= target and row["coverage"] >= .005]
            if feasible: fixed.append({"harm_target": target, **max(feasible, key=lambda row: row["gain"])})
    write_csv(config.output / "fixed_risk_operating_points.csv", fixed)
    save_json(config.output / "calibration/frozen_policy.json", freeze)
    save_json(config.output / "calibration/freeze_manifest.json", freeze)
    write_csv(config.output / "checkpoint_selection.csv",
              [{"policy": name, **policies[name]} for name in critics])
    print(json.dumps(clean({"selected": selected_name, "policy": policies[selected_name]})), flush=True)


BIN_SPECS = {
    "gradient": ((0, .25), (.25, .75), (.75, 2.0), (2.0, float("inf"))),
    "temporal_residual": ((0, .1), (.1, .3), (.3, .6), (.6, 1.01)),
    "stereo_residual": ((0, .1), (.1, .3), (.3, .6), (.6, 1.01)),
    "update": ((0, .05), (.05, .25), (.25, 1.0), (1.0, float("inf"))),
    "fb": ((0, .25), (.25, .5), (.5, .75), (.75, 1.01)),
    "witness": ((1, 2), (2, 3), (3, 4), (4, 5)),
}


@torch.inference_mode()
def evaluate(config: argparse.Namespace) -> None:
    freeze_path = config.output / "calibration/freeze_manifest.json"
    if config.split == "test" and not freeze_path.exists():
        raise RuntimeError("dataset 7 locked until complete validation policy freeze")
    freeze = json.loads(freeze_path.read_text())
    device = torch.device(config.device)
    selected = freeze["selected_policy"]; family = selected.split("@")[0]
    for name, digest in freeze["critic_checkpoint_hashes"].items():
        if "@" in name: continue
        seed = ENSEMBLE_SEEDS[0]
        if "_seed" in name: family_name, seed = name.split("_seed"); seed = int(seed)
        else: family_name = name
        path = family_root(config, family_name, seed) / "checkpoints/best_validation.pt"
        if sha256(path) != digest: raise RuntimeError(f"critic checkpoint hash mismatch: {name}")
    if selected.endswith("@ensemble"):
        critic = EnsembleCritic([load_critic(config, family, seed, device)[0] for seed in ENSEMBLE_SEEDS])
    else:
        critic, _ = load_critic(config, family, ENSEMBLE_SEEDS[0], device)
    thresholds = freeze["policies"][selected]
    sequences = TEST if config.split == "test" else VALIDATION
    dataset = build_critic_bank(config, sequences, training=False)
    frozen, _ = load_frozen(device); base_policy = frozen_policy()
    rows = []; offset = 0; runtime = []; peak = 0
    diag = defaultdict(list)
    for cpu in loader(dataset, config, training=False):
        batch = to_device(cpu, device)
        evidence, proposal, extras = batch_proposal(batch, frozen, base_policy, device)
        if device.type == "cuda": torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        tick = time.perf_counter()
        features, _ = build_critic_features(evidence, proposal, extras, family)
        output = critic(features)
        accept = selective_accept(output, proposal, lambda_uncertainty=thresholds["lambda_uncertainty"],
                                  tau_gain=thresholds["tau_gain"], tau_harm=thresholds["tau_harm"])
        if device.type == "cuda":
            torch.cuda.synchronize(); peak = max(peak, torch.cuda.max_memory_allocated())
        runtime.append((time.perf_counter() - tick) / len(batch["raw"]))
        e_raw, e_proposed, valid, _sup = supervision(batch, evidence, proposal, config)
        chosen_age = torch.gather(torch.tensor(list(RAW_AGES), device=device)[None, :, None, None]
                                  .expand(len(batch["raw"]), -1, *evidence.raw.shape[-2:]), 1, proposal.chosen)
        for method, accepted in (("ungated", proposal.eligible), ("critic", accept)):
            prediction, applied = apply_veto(evidence.raw, proposal, accepted)
            error = (prediction - batch["gt"].float()).abs()
            delta = e_raw - error; changed = applied & valid
            gt_delta = batch["gt"].float() - batch["gt_memory"].float()
            temporal = valid & batch["gt_memory_valid"].bool()
            raw_tepe = ((evidence.raw - evidence.candidates[:, 0:1]) - gt_delta).abs()
            output_tepe = ((prediction - evidence.candidates[:, 0:1]) - gt_delta).abs()
            for local in range(len(batch["raw"])):
                s = slice(local, local + 1); meta = dataset.metadata[offset + local]
                v = valid[s]; ch = changed[s]; err = error[s]; er = e_raw[s]; dl = delta[s]
                clean_mask = v & (er <= .10)
                row = {"method": method, **{k: meta[k] for k in ("backbone", "sequence", "frame_id", "frame_index")},
                       "valid_count": int(v.sum()), "raw_error_sum": float(er[v].sum()), "output_error_sum": float(err[v].sum()),
                       "fixed_h4_error_sum": float((batch["fixed_h4_output"][s].float() - batch["gt"][s].float()).abs()[v].sum()),
                       "changed_count": int(ch.sum()), "helpful_count": int((ch & (dl > .10)).sum()),
                       "harmful_count": int((ch & (dl < -.10)).sum()), "clean_count": int(clean_mask.sum()),
                       "clean_degradation_count": int((ch & clean_mask & (dl < -.10)).sum()),
                       "frame_degradation": float(err[v].mean() - er[v].mean()) if v.any() else 0.0,
                       "temporal_count": int(temporal[s].sum()), "raw_tepe_sum": float(raw_tepe[s][temporal[s]].sum()),
                       "output_tepe_sum": float(output_tepe[s][temporal[s]].sum()),
                       "raw_teper_sum": float((raw_tepe[s] / (gt_delta[s].abs() + 1e-3))[temporal[s]].sum()),
                       "output_teper_sum": float((output_tepe[s] / (gt_delta[s].abs() + 1e-3))[temporal[s]].sum()),
                       **{f"accepted_age_{age}": int((ch[0] & (chosen_age[s][0] == age)).sum()) for age in RAW_AGES}}
                rows.append(row)
        stride = max(config.sample_stride, 1); pixels = evidence.raw.shape[-2] * evidence.raw.shape[-1]
        flat = np.arange(0, len(batch["raw"]) * pixels, stride, dtype=np.int64)
        take = lambda tensor: tensor.reshape(-1)[flat].float().cpu().numpy()
        diag["valid"].append(take(valid)); diag["eligible"].append(take(proposal.eligible))
        diag["accept"].append(take(accept)); diag["delta"].append(take(e_raw - e_proposed))
        diag["e_raw"].append(take(e_raw)); diag["e_prop"].append(take(e_proposed))
        diag["delta_hat"].append(take(output.delta_hat)); diag["sigma"].append(take(output.sigma_delta))
        diag["harm_prob"].append(take(output.harm_probability))
        diag["mu_raw"].append(take(output.mu_raw)); diag["mu_prop"].append(take(output.mu_proposed))
        diag["update"].append(take((proposal.proposed - evidence.raw).abs())); diag["age"].append(take(chosen_age.float()))
        diag["fb"].append(take(torch.gather(evidence.fb_confidence, 1, proposal.chosen)))
        diag["temporal_residual"].append(take(torch.gather(batch["photo"].float(), 1, proposal.chosen)))
        from model_design.models.spatial_error_critic import stereo_residual, _gradient
        diag["stereo_residual"].append(take(stereo_residual(batch["left_rgb"].float(), batch["right_rgb"].float(), evidence.raw)[0]))
        diag["gradient"].append(take(_gradient(evidence.raw)))
        diag["witness"].append(take(evidence.available.sum(1, keepdim=True).float()))
        diag["frame_group"].append((flat // pixels + offset).astype(np.int32))
        offset += len(batch["raw"])
    destination = config.output / ("frozen_test" if config.split == "test" else "validation_eval")
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / "frame_metrics.csv", rows)
    arrays = {key: np.concatenate(items) for key, items in diag.items()}
    arrays["e_raw"] = arrays["e_raw"]; arrays[selected + "_delta_hat"] = arrays["delta_hat"]
    arrays[selected + "_sigma"] = arrays["sigma"]; arrays[selected + "_harm"] = arrays["harm_prob"]
    arrays[selected + "_mu_raw"] = arrays["mu_raw"]; arrays[selected + "_mu_prop"] = arrays["mu_prop"]
    summaries = []
    for method in ("ungated", "critic"):
        method_rows = [row for row in rows if row["method"] == method]
        count = sum(row["valid_count"] for row in method_rows); raw_sum = sum(row["raw_error_sum"] for row in method_rows)
        out_sum = sum(row["output_error_sum"] for row in method_rows); fixed = sum(row["fixed_h4_error_sum"] for row in method_rows)
        changed = sum(row["changed_count"] for row in method_rows); harmful = sum(row["harmful_count"] for row in method_rows)
        helpful = sum(row["helpful_count"] for row in method_rows); clean_count = sum(row["clean_count"] for row in method_rows)
        clean_deg = sum(row["clean_degradation_count"] for row in method_rows)
        temporal = sum(row["temporal_count"] for row in method_rows)
        frame = np.asarray([row["frame_degradation"] for row in method_rows]); frame = frame[np.isfinite(frame)]
        ungated_gain = sum(r["raw_error_sum"] - r["output_error_sum"] for r in rows if r["method"] == "ungated")
        summary = {"policy": method, "split": config.split, "family": selected, "valid_count": count,
                   "raw_epe": raw_sum / count, "output_epe": out_sum / count, "gain": (raw_sum - out_sum) / count,
                   "fixed_h4_epe_common_support": fixed / count, "gain_over_fixed_h4": (fixed - out_sum) / count,
                   "raw_bank_oracle_recovery": ((raw_sum - out_sum) / count) / RAW_BANK_ORACLE_GAIN,
                   "convex_oracle_recovery": ((raw_sum - out_sum) / count) / .1549843363093545,
                   "ungated_gain_retained": (raw_sum - out_sum) / max(ungated_gain, 1e-12),
                   "coverage": changed / count, "exact_raw_fallback_frequency": 1 - changed / count,
                   "intervention_precision": helpful / max(changed, 1), "harmful_update_rate": harmful / max(changed, 1),
                   "harmful_update_rate_all_valid": harmful / count, "clean_degradation": clean_deg / max(clean_count, 1),
                   "degraded_frame_fraction": float((frame > 0).mean()), "worst_frame_degradation": float(frame.max(initial=0.0)),
                   "raw_tepe": sum(r["raw_tepe_sum"] for r in method_rows) / max(temporal, 1),
                   "output_tepe": sum(r["output_tepe_sum"] for r in method_rows) / max(temporal, 1),
                   "raw_teper": sum(r["raw_teper_sum"] for r in method_rows) / max(temporal, 1),
                   "output_teper": sum(r["output_teper_sum"] for r in method_rows) / max(temporal, 1),
                   **{f"accepted_age_{age}_fraction": sum(r[f"accepted_age_{age}"] for r in method_rows) / max(changed, 1) for age in RAW_AGES}}
        summaries.append(summary)
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[("backbone", row["method"], row["backbone"])]["rows"].append(row)
        grouped[("sequence", row["method"], row["sequence"])]["rows"].append(row)
    per_group = []
    for (kind, method, value), item in grouped.items():
        items = item["rows"]; count = sum(r["valid_count"] for r in items)
        raw_sum = sum(r["raw_error_sum"] for r in items); out_sum = sum(r["output_error_sum"] for r in items)
        changed = sum(r["changed_count"] for r in items); harmful = sum(r["harmful_count"] for r in items)
        per_group.append({"group": kind, "policy": method, kind: value, "raw_epe": raw_sum / count,
                          "output_epe": out_sum / count, "gain": (raw_sum - out_sum) / count,
                          "coverage": changed / count, "harmful_update_rate": harmful / max(changed, 1)})
    write_csv(destination / "per_backbone_metrics.csv", [row for row in per_group if row["group"] == "backbone"])
    write_csv(destination / "per_sequence_metrics.csv", [row for row in per_group if row["group"] == "sequence"])
    per_age = []
    for method in ("ungated", "critic"):
        method_rows = [row for row in rows if row["method"] == method]
        total = sum(sum(r[f"accepted_age_{age}"] for r in method_rows) for age in RAW_AGES)
        for age in RAW_AGES:
            accepted_count = sum(r[f"accepted_age_{age}"] for r in method_rows)
            per_age.append({"policy": method, "age": age, "accepted_count": accepted_count,
                            "accepted_fraction": accepted_count / max(total, 1)})
    write_csv(destination / "per_age_metrics.csv", per_age)
    detection = [detection_metrics(arrays, selected, margin) for margin in HARM_MARGINS]
    write_csv(destination / "harm_detection_metrics.csv", detection)
    write_csv(destination / "risk_coverage.csv", risk_coverage_rows(arrays, selected, config.harm_margin))
    valid = arrays["valid"].astype(bool); eligible = arrays["eligible"].astype(bool); delta = arrays["delta"]
    oracle_rows = []
    learned_coverage = float((arrays["accept"].astype(bool) & valid).mean())
    oracle_specs = [("oracle_reject_all_harmful", eligible & (delta >= -config.harm_margin)),
                    ("oracle_positive_gain_only", eligible & (delta > 0)),
                    ("oracle_true_delta_threshold", eligible & (delta > .10))]
    order = np.argsort(-np.where(eligible & valid, delta, -np.inf))
    budget = int(learned_coverage * valid.sum())
    matched = np.zeros_like(eligible); matched[order[:max(budget, 1)]] = True
    oracle_specs.append(("oracle_at_learned_coverage", matched & eligible))
    for label, accepted in oracle_specs:
        oracle_rows.append({"oracle": label, "note": "diagnostic ceiling; never used for selection",
                            **policy_metrics(arrays, accepted, config.harm_margin)})
    write_csv(destination / "oracle_rejection_ceiling.csv", oracle_rows)
    bins = []
    accept_mask = arrays["accept"].astype(bool)
    for condition, edges in BIN_SPECS.items():
        for low, high in edges:
            mask = valid & eligible & (arrays[condition] >= low) & (arrays[condition] < high)
            bins.append({"condition": condition, "bin": f"{low}-{high}", "count": int(mask.sum()),
                         "mean_delta": float(delta[mask].mean()) if mask.any() else 0.0,
                         "harmful_fraction": float((delta[mask] < -config.harm_margin).mean()) if mask.any() else 0.0,
                         "accept_fraction": float(accept_mask[mask].mean()) if mask.any() else 0.0,
                         "accepted_harm": float((delta[mask & accept_mask] < -config.harm_margin).mean()) if (mask & accept_mask).any() else 0.0})
    write_csv(destination / "residual_bin_analysis.csv", [b for b in bins if b["condition"] != "update"])
    write_csv(destination / "update_magnitude_analysis.csv", [b for b in bins if b["condition"] == "update"])
    summary_payload = {"project": "ARGOS v2", "split": config.split, "selected_policy": selected,
                       "thresholds": thresholds, "policies": summaries, "harm_detection": detection,
                       "oracle_ceilings": oracle_rows,
                       "runtime_ms_per_frame": 1000 * float(np.mean(runtime)),
                       "peak_gpu_memory_mb": peak / 2**20}
    write_csv(destination / "summary.csv", summaries)
    save_json(destination / "summary.json", summary_payload)
    if config.split == "test":
        save_json(config.output / "protocol_audit/test_opened.json",
                  {"opened_after_freeze": True, "freeze_manifest_sha256": sha256(freeze_path), "sequences": list(TEST)})


def audit(config: argparse.Namespace) -> None:
    config.output.mkdir(parents=True, exist_ok=True)
    manifest = split_manifest() | {
        "project": "ARGOS v2", "experiment": "spatial comparative error critic (veto-only)",
        "frozen_refiner_checkpoint": str(FROZEN_CHECKPOINT), "frozen_refiner_sha256": sha256(FROZEN_CHECKPOINT),
        "expected_sha256": EXPECTED_FROZEN_SHA256, "corrected_h4_sha256": sha256(CORRECTED_CHECKPOINT),
        "dataset7_locked_until_freeze": True, "critic_can_open": False, "critic_can_change_candidate": False,
        "critic_can_change_weight": False, "critic_output_enters_anchor_bank": False,
        "gt_in_inference_features": False, "future_frames_used": False,
        "families": {family: feature_channels(family) for family in FAMILIES},
    }
    save_json(config.output / "protocol_audit/protocol_audit.json", manifest)
    save_json(config.output / "protocol_audit.json", manifest)
    save_json(config.output / "frozen_checkpoint_hashes.json",
              {"raw_multi_anchor_refiner": sha256(FROZEN_CHECKPOINT), "corrected_h4": sha256(CORRECTED_CHECKPOINT)})
    from model_design.models.raw_multi_anchor_refiner import MultiAnchorEvidence  # noqa: F401
    device = torch.device("cpu")
    example_counts = {}
    for family in FAMILIES:
        model = SpatialErrorCritic(feature_channels(family), channels=config.channels)
        example_counts[family] = sum(parameter.numel() for parameter in model.parameters())
    save_json(config.output / "parameter_count.json", example_counts)
    save_json(config.output / "architecture_summary.json", {
        "model": "SpatialErrorCritic", "channels": config.channels, "dilations": [1, 2, 4, 2, 1],
        "normalization": "GroupNorm(8)", "activation": "SiLU", "receptive_field_px": 43,
        "heads": ["mu_raw", "mu_proposed", "s_raw", "s_proposed", "harm_logit"],
        "parameter_budget_max": 1_500_000, "parameters": example_counts})
    import model_design.models.spatial_error_critic as critic_module
    from model_design.models.raw_multi_anchor_refiner import MultiAnchorEvidence as _E
    import model_design.models.raw_multi_anchor_selective_gate as gate_module
    dummy_evidence = _E(torch.rand(1, 1, 16, 20) * 8 + 1, torch.rand(1, 4, 16, 20) * 8 + 1,
                        torch.ones(1, 4, 16, 20, dtype=torch.bool), torch.ones(1, 4, 16, 20, dtype=torch.bool),
                        torch.ones(1, 4, 16, 20), torch.tensor([1, 2, 4, 8]), torch.zeros(4))
    from model_design.models.raw_multi_anchor_refiner import MultiAnchorOutput as _O
    score = torch.rand(1, 4, 16, 20)
    dummy_proposal = gate_module.frozen_proposal(
        dummy_evidence, _O(score, torch.sigmoid(score), score, torch.sigmoid(score), score),
        probability_threshold=.1, utility_threshold_px=.0)
    extras = critic_module.CriticExtras(torch.rand(1, 3, 16, 20), torch.rand(1, 3, 16, 20), torch.rand(1, 4, 16, 20))
    schema = {}
    for family in FAMILIES:
        _features, names = critic_module.build_critic_features(dummy_evidence, dummy_proposal, extras, family)
        schema[family] = names
    save_json(config.output / "feature_schema.json", {
        "families_cumulative": True, "contains_gt": False, "contains_backbone_identity": False,
        "normalization": "fixed physical clipping; no dataset-7 statistics", "features": schema})


def summarize(config: argparse.Namespace) -> None:
    test = json.loads((config.output / "frozen_test/summary.json").read_text())
    policies = {row["policy"]: row for row in test["policies"]}
    critic = policies["critic"]; ungated = policies["ungated"]
    prior_gate = json.loads((ROOT / "results/raw_multi_anchor_selective_gate/aggregate_summary.json").read_text())
    validation_rows = list(csv.DictReader((config.output / "validation_summary.csv").open()))
    freeze = json.loads((config.output / "calibration/frozen_policy.json").read_text())
    per_backbone = list(csv.DictReader((config.output / "frozen_test/per_backbone_metrics.csv").open()))
    per_sequence = list(csv.DictReader((config.output / "frozen_test/per_sequence_metrics.csv").open()))
    per_age = list(csv.DictReader((config.output / "frozen_test/per_age_metrics.csv").open()))
    detection = test["harm_detection"]
    distant = sum(float(row["accepted_fraction"]) for row in per_age
                  if row["policy"] == "critic" and row["age"] in ("4", "8"))
    gates = {"beats_fixed_h4": critic["gain_over_fixed_h4"] > 0,
             "harm_le_10pct": critic["harmful_update_rate"] <= .10,
             "clean_le_3pct": critic["clean_degradation"] <= .03,
             "degraded_frames_le_25pct": critic["degraded_frame_fraction"] <= .25,
             "nontrivial_coverage": critic["coverage"] >= .005,
             "meaningful_gain_retained": critic["ungated_gain_retained"] >= .5,
             "harm_separability_beats_scalar_gate": any(
                 row.get("harm_auroc", 0) and row["margin"] == .10 and row["harm_auroc"] > .62 for row in detection),
             "all_seen_backbones_positive": all(float(row["gain"]) > 0 for row in per_backbone if row["policy"] == "critic"),
             "nearly_all_sequences_positive": np.mean([float(row["gain"]) > 0 for row in per_sequence if row["policy"] == "critic"]) >= .75,
             "distant_anchors_authorized": distant >= .05}
    if all(gates.values()): verdict = "GO"
    elif (gates["harm_le_10pct"] and gates["clean_le_3pct"] and gates["degraded_frames_le_25pct"]
          and critic["gain"] > 0): verdict = "CONDITIONAL GO"
    else: verdict = "NO-GO"
    aggregate = {"project": "ARGOS v2", "experiment": "spatial comparative error critic",
                 "frozen_ungated": ungated, "spatial_critic": critic,
                 "previous_scalar_gate": prior_gate["best_validation_selected_test_policy"],
                 "previous_scalar_gate_harm_detection": prior_gate["harm_detection"],
                 "harm_detection": detection, "selected_policy": freeze["selected_policy"],
                 "frozen_thresholds": freeze["policies"][freeze["selected_policy"]],
                 "promotion_gates": gates,
                 "geometry_verdict": "GO" if critic["gain"] > 0 else "NO-GO",
                 "safety_verdict": "GO" if gates["harm_le_10pct"] and gates["clean_le_3pct"] and gates["degraded_frames_le_25pct"] else "NO-GO",
                 "verdict": verdict,
                 "runtime_ms_per_frame": test["runtime_ms_per_frame"], "peak_gpu_memory_mb": test["peak_gpu_memory_mb"],
                 "scope": "held-out SCARED-C dataset 7 (previously inspected by earlier ARGOS v2 studies); "
                          "three training-seen backbones; no unseen-backbone, external-OOD, clinical or conformal claim"}
    save_json(config.output / "aggregate_summary.json", aggregate)
    save_json(config.output / "verdicts.json", {"verdict": verdict, "gates": gates,
              "geometry_verdict": aggregate["geometry_verdict"], "safety_verdict": aggregate["safety_verdict"]})
    save_json(config.output / "frozen_reference/frozen_reference_summary.json",
              {"frozen_ungated_multi_anchor": ungated,
               "previous_scalar_gate": prior_gate["best_validation_selected_test_policy"],
               "fixed_h4_epe_common_support": FIXED_H4_EPE})
    ablation = []
    for name in sorted({row["policy"] for row in validation_rows if row.get("policy") and row["policy"] != "ungated"}):
        best = max((row for row in validation_rows if row["policy"] == name), key=lambda row: float(row["gain"]))
        ablation.append({"configuration": name, "split": "validation", **best})
    write_csv(config.output / "ablation_summary.csv", ablation)
    training = []
    for family in FAMILIES:
        for seed in ENSEMBLE_SEEDS:
            path = family_root(config, family, seed) / "training_history.csv"
            if path.exists(): training.extend(csv.DictReader(path.open()))
    write_csv(config.output / "training_summary.csv", training)
    for source, target in (("frozen_test/summary.csv", "frozen_test_summary.csv"),
                           ("frozen_test/per_backbone_metrics.csv", "per_backbone_metrics.csv"),
                           ("frozen_test/per_sequence_metrics.csv", "per_sequence_metrics.csv"),
                           ("frozen_test/per_age_metrics.csv", "per_age_metrics.csv"),
                           ("frozen_test/harm_detection_metrics.csv", "calibration_metrics.csv"),
                           ("frozen_test/harm_detection_metrics.csv", "comparative_error_metrics.csv"),
                           ("frozen_test/residual_bin_analysis.csv", "residual_bin_analysis.csv"),
                           ("frozen_test/update_magnitude_analysis.csv", "update_magnitude_analysis.csv"),
                           ("frozen_test/oracle_rejection_ceiling.csv", "oracle_rejection_ceiling.csv"),
                           ("frozen_test/summary.csv", "intervention_metrics.csv")):
        path = config.output / source
        if path.exists(): (config.output / target).write_bytes(path.read_bytes())
    write_csv(config.output / "runtime_summary.csv", [{"policy": "critic",
              "runtime_ms_per_frame": test["runtime_ms_per_frame"], "peak_gpu_memory_mb": test["peak_gpu_memory_mb"],
              "parameters": json.loads((config.output / "parameter_count.json").read_text()).get(freeze["selected_policy"].split("@")[0])}])
    (config.output / "paper_ready_tables.tex").write_text(
        "\\begin{tabular}{lrrrrr}\\toprule\nPolicy & EPE & Gain & Coverage & Harm & Clean deg. \\\\\n" +
        "\n".join(f"{row['policy'].replace('_', ' ')} & {row['output_epe']:.4f} & {row['gain']:.4f} & "
                  f"{row['coverage']:.3f} & {row['harmful_update_rate']:.3f} & {row['clean_degradation']:.3f} \\\\"
                  for row in test["policies"]) + "\n\\bottomrule\\end{tabular}\n")
    (config.output / "README.md").write_text(
        "# ARGOS v2 raw multi-anchor spatial safety critic\n\n"
        f"Frozen dataset-7 verdict: **{verdict}**. Selected policy `{freeze['selected_policy']}`; "
        f"EPE {critic['output_epe']:.5f}, gain {critic['gain']:.5f}, gain over fixed H=4 "
        f"{critic['gain_over_fixed_h4']:.5f}, accepted harm {100 * critic['harmful_update_rate']:.2f}%, "
        f"clean degradation {100 * critic['clean_degradation']:.2f}%, coverage {100 * critic['coverage']:.2f}%, "
        f"ungated gain retained {100 * critic['ungated_gain_retained']:.1f}%.\n\n"
        "Veto-only post-hoc spatial critic over the frozen multi-anchor proposal. Dataset 7 was "
        "previously inspected by earlier ARGOS v2 studies. No unseen-backbone, external-dataset OOD, "
        "clinical-safety or formal conformal claim is made.\n")


def main() -> None:
    config = arguments(); audit(config)
    if config.mode == "audit": return
    if config.mode == "overfit": train_families(config, "overfit")
    elif config.mode == "smoke": train_families(config, "smoke")
    elif config.mode == "train": train_families(config)
    elif config.mode == "calibrate": calibrate(config)
    elif config.mode == "evaluate": evaluate(config)
    elif config.mode == "summarize": summarize(config)


if __name__ == "__main__": main()
