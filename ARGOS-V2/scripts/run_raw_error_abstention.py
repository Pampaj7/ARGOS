#!/usr/bin/env python3
"""Train/evaluate the ARGOS v2 raw-error detector and frozen-A2 abstention."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))
sys.path.insert(0, str(V2_ROOT / "scripts"))

from argos_v2.sequences import accepted_sequences  # noqa: E402
from model_design.data.raw_error_dataset import (  # noqa: E402
    CALIBRATION_SEQUENCES,
    TEST_SEQUENCES,
    RawErrorDataset,
    raw_error_targets,
)
from model_design.data.temporal_pair_dataset import (  # noqa: E402
    PRIMARY_UNSEEN_BACKBONE,
    SEEN_BACKBONES,
    TemporalPairDataset,
)
from model_design.external_components.bidavideo import BiDAFlowInferenceAdapter  # noqa: E402
from model_design.losses.raw_error_losses import RawErrorLossConfig, raw_error_losses  # noqa: E402
from model_design.models.abstention import (  # noqa: E402
    OperatingMode,
    authorization_mask,
    authorized_update,
    calibrated_probability,
    fit_temperature,
)
from model_design.models.learned_t1_refiner import LearnedT1Refiner  # noqa: E402
from model_design.models.raw_error_detector import (  # noqa: E402
    ARCHITECTURES,
    RECEPTIVE_FIELDS,
    RawErrorDetector,
    RawErrorEvidence,
)
from run_learned_t1_refiner import build_evidence  # noqa: E402


A2_CHECKPOINT = V2_ROOT / "results/learned_t1_refiner/ablations/A2/checkpoints/best_validation.pt"
COVERAGE_THRESHOLDS = (0.05, 0.25, 0.50, 0.90)
ERROR_THRESHOLDS = (0.25, 0.50, 1.00, 3.00)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "train", "evaluate", "unseen"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="s2")
    parser.add_argument("--loss-mode", choices=("a0", "a1", "a2", "a3", "a4"), default="a4")
    parser.add_argument("--channels", type=int, default=24)
    parser.add_argument("--epsilon", type=float, default=0.50)
    parser.add_argument("--indifference-band", type=float, default=0.10)
    parser.add_argument("--false-positive-cost", type=float, default=3.0)
    parser.add_argument("--coverage-threshold", type=float, default=0.50)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8))
    parser.add_argument("--max-train-pairs", type=int, default=256)
    parser.add_argument("--max-validation-pairs", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sample-pixels-per-frame", type=int, default=2048)
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_json(value):
    if isinstance(value, dict): return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean_json(v) for v in value]
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(""); return
    keys = list(rows[0])
    for row in rows[1:]:
        keys.extend(key for key in row if key not in keys)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def atomic_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary); temporary.replace(path)


def loader(dataset, args, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0, drop_last=False,
        generator=torch.Generator().manual_seed(args.seed),
    )


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def load_a2(device: torch.device) -> LearnedT1Refiner:
    payload = torch.load(A2_CHECKPOINT, map_location="cpu", weights_only=False)
    model = LearnedT1Refiner("A2", tau_px=float(payload.get("tau_px", 3.0)))
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model


@torch.no_grad()
def detector_evidence(a2, batch: dict, evidence: dict) -> tuple[RawErrorEvidence, object]:
    proposal = a2(batch["raw"], evidence, batch["current_rgb"])
    return RawErrorEvidence(
        raw=batch["raw"].detach(), raw_valid=batch["raw_valid"].detach(),
        aligned=evidence["aligned_past_disparity"].detach(),
        aligned_valid=evidence["aligned_validity"].detach(),
        warp_support=evidence["warp_support"].detach(),
        forward_backward_error=evidence["forward_backward_error"].detach(),
        forward_backward_confidence=evidence["forward_backward_confidence"].detach(),
        photometric_residual=evidence["photometric_residual"].detach(),
        flow_magnitude=evidence["flow_magnitude"].detach(),
        a2_update=proposal.update.detach(), a2_error_gate=proposal.g_error.detach(),
        a2_memory_gate=proposal.c_memory.detach(),
    ), proposal


def loss_config(args) -> RawErrorLossConfig:
    return RawErrorLossConfig(mode=args.loss_mode, false_positive_cost=args.false_positive_cost)


def binary_metrics(probability: np.ndarray, label: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score
    if not label.size:
        return {k: float("nan") for k in ("auroc", "average_precision", "brier", "ece")}
    auroc = float(roc_auc_score(label, probability)) if np.unique(label).size > 1 else float("nan")
    ap = float(average_precision_score(label, probability)) if label.sum() else float("nan")
    ece = 0.0
    for low in np.linspace(0, 1, 11)[:-1]:
        selected = (probability >= low) & (probability < low + 0.1)
        if selected.any():
            ece += selected.mean() * abs(probability[selected].mean() - label[selected].mean())
    return {"auroc": auroc, "average_precision": ap,
            "brier": float(np.mean((probability - label) ** 2)), "ece": float(ece)}


def correlation(left: np.ndarray, right: np.ndarray, rank: bool = False) -> float:
    if left.size < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12: return float("nan")
    if rank:
        from scipy.stats import spearmanr
        return float(spearmanr(left, right).statistic)
    return float(np.corrcoef(left, right)[0, 1])


@torch.no_grad()
def validate(model, a2, adapter, data_loader, device, args) -> dict[str, float]:
    model.eval(); values = defaultdict(float); arrays = defaultdict(list); batches = 0
    for cpu in data_loader:
        batch = to_device(cpu, device); evidence, _ = build_evidence(adapter, batch)
        detector_input, _ = detector_evidence(a2, batch, evidence)
        output = model(detector_input)
        target = raw_error_targets(batch, epsilon_px=args.epsilon,
            indifference_band_px=args.indifference_band,
            coverage_threshold=args.coverage_threshold)
        losses = raw_error_losses(output, target, loss_config(args))
        for key, value in losses.items(): values[key] += float(value.detach())
        index = target.classification_valid.flatten().nonzero().flatten()[::64]
        arrays["p"].append(output.probability.flatten()[index].cpu().numpy())
        arrays["y"].append(target.label.flatten()[index].cpu().numpy())
        index_r = target.regression_valid.flatten().nonzero().flatten()[::64]
        arrays["mu"].append(output.mu.flatten()[index_r].cpu().numpy())
        arrays["err"].append(target.error.flatten()[index_r].cpu().numpy())
        arrays["sigma"].append(output.sigma.flatten()[index_r].cpu().numpy())
        batches += 1
    summary = {f"loss_{k}": v / max(batches, 1) for k, v in values.items()}
    p, y = (np.concatenate(arrays[k]) if arrays[k] else np.array([]) for k in ("p", "y"))
    mu, err, sigma = (np.concatenate(arrays[k]) if arrays[k] else np.array([]) for k in ("mu", "err", "sigma"))
    summary.update(binary_metrics(p, y)); summary.update({
        "regression_mae": float(np.mean(np.abs(mu - err))) if err.size else float("nan"),
        "pearson": correlation(mu, err), "spearman": correlation(mu, err, True),
        "uncertainty_error_correlation": correlation(sigma, np.abs(mu - err)),
    })
    return summary


def split_manifest(args) -> dict:
    all_sequences = accepted_sequences()
    held_out = set(CALIBRATION_SEQUENCES) | set(TEST_SEQUENCES)
    return {
        "seed": args.seed, "training_backbones": list(SEEN_BACKBONES),
        "train_sequences": [s for s in all_sequences if s not in held_out],
        "calibration_sequences": list(CALIBRATION_SEQUENCES),
        "seen_test_sequences": list(TEST_SEQUENCES),
        "unseen_policy": "Fast-FoundationStereo and CREStereo rejected before loading",
        "causal_pair": "t-1 -> t", "primary_coverage": args.coverage_threshold,
    }


def train(args) -> None:
    seed_all(args.seed); device = torch.device(args.device); manifest = split_manifest(args)
    smoke = args.mode == "smoke"
    smoke_sequence = "dataset_3_keyframe_1"
    if smoke_sequence not in manifest["train_sequences"]:
        smoke_sequence = manifest["train_sequences"][0]
    train_sequences = [smoke_sequence] if smoke else manifest["train_sequences"]
    train_backbones = [SEEN_BACKBONES[0]] if smoke else list(SEEN_BACKBONES)
    train_pairs = 24 if smoke else args.max_train_pairs
    val_pairs = 8 if smoke else args.max_validation_pairs
    epochs = 25 if smoke else args.epochs
    train_set = RawErrorDataset(train_backbones, train_sequences, coverage_threshold=args.coverage_threshold,
        max_pairs_per_sequence=train_pairs, random_clip_start=True, seed=args.seed)
    val_set = RawErrorDataset(train_backbones, list(CALIBRATION_SEQUENCES), coverage_threshold=args.coverage_threshold,
        max_pairs_per_sequence=val_pairs, random_clip_start=False, seed=args.seed)
    model = RawErrorDetector(args.architecture, channels=args.channels).to(device)
    a2 = load_a2(device); adapter = BiDAFlowInferenceAdapter("sea_raft", device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history_path = args.output / "training_history.csv"; history = []
    if args.resume and history_path.exists() and not smoke:
        with history_path.open(newline="") as handle:
            history = list(csv.DictReader(handle))
    best = float("inf"); start_epoch = 0
    last_path = args.output / "checkpoints/final.pt"
    if args.resume and last_path.exists() and not smoke:
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]); best = float(state["best_validation_loss"])
    initial = None; global_step = 0
    for epoch in range(start_epoch, epochs):
        model.train(); sums = defaultdict(float); batches = 0
        for cpu in loader(train_set, args, True):
            batch = to_device(cpu, device); evidence, _ = build_evidence(adapter, batch)
            detector_input, _ = detector_evidence(a2, batch, evidence)
            output = model(detector_input)
            target = raw_error_targets(batch, epsilon_px=args.epsilon,
                indifference_band_px=args.indifference_band,
                coverage_threshold=args.coverage_threshold)
            losses = raw_error_losses(output, target, loss_config(args))
            if initial is None: initial = float(losses["total"].detach())
            optimizer.zero_grad(set_to_none=True); losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            for key, value in losses.items(): sums[key] += float(value.detach())
            batches += 1; global_step += 1
            if args.steps and global_step >= args.steps: break
        metrics = validate(model, a2, adapter, loader(val_set, args, False), device, args)
        row = {"epoch": epoch + 1, **{f"train_{k}": v / max(batches, 1) for k, v in sums.items()}, **metrics}
        history.append(row); write_csv(history_path, history)
        score = metrics["loss_total"]
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1,
            "best_validation_loss": min(best, score), "architecture": args.architecture,
            "channels": args.channels, "config": vars(args), "split_manifest": manifest,
            "a2_checkpoint": str(A2_CHECKPOINT), "loss_config": asdict(loss_config(args))}
        atomic_checkpoint(last_path, payload)
        if score < best:
            best = score; payload["best_validation_loss"] = best
            atomic_checkpoint(args.output / "checkpoints/best_validation.pt", payload)
        print(json.dumps(clean_json(row)), flush=True)
        if args.steps and global_step >= args.steps: break
    save_json(args.output / "config.json", vars(args)); save_json(args.output / "split_manifest.json", manifest)
    if smoke:
        final = history[-1]["train_total"]
        result = {"initial_loss": initial, "final_train_loss": final,
            "loss_reduction_fraction": (initial - final) / max(initial, 1e-8),
            "gate_probability_changed": abs(history[-1]["train_probability_mean"] - 0.1192) > 1e-3,
            "finite": bool(math.isfinite(final)), "passed": bool(final < 0.65 * initial)}
        save_json(args.output / "smoke_summary.json", result); print(json.dumps(result, indent=2))
        if not result["passed"]: raise RuntimeError("overfit smoke did not reduce loss by 35%")


def sample_batch(output, proposal, batch, evidence, args) -> dict[str, np.ndarray]:
    target = raw_error_targets(batch, epsilon_px=args.epsilon,
        indifference_band_px=args.indifference_band, coverage_threshold=args.coverage_threshold)
    valid = target.regression_valid & evidence["aligned_validity"] & evidence["warp_support"]
    indices = valid.flatten().nonzero().flatten()
    if indices.numel() > args.sample_pixels_per_frame * batch["raw"].shape[0]:
        pick = torch.linspace(0, indices.numel() - 1,
            args.sample_pixels_per_frame * batch["raw"].shape[0], device=indices.device).long()
        indices = indices[pick]
    def take(value): return value.flatten()[indices].float().cpu().numpy()
    return {"logits": take(output.logits), "p": take(output.probability), "mu": take(output.mu),
        "sigma": take(output.sigma), "error": take(target.error), "label": take(target.label),
        "coverage": take(batch["gt_coverage"]),
        "class_valid": take(target.classification_valid).astype(bool),
        "proposal_update": take(proposal.update),
        "a2_error": take((proposal.disparity - batch["gt"]).abs()),
        "aligned_valid": np.ones(indices.numel(), bool), "warp_support": np.ones(indices.numel(), bool)}


@torch.no_grad()
def collect_samples(model, a2, adapter, dataset, device, args) -> dict[str, np.ndarray]:
    collected = defaultdict(list); model.eval()
    for cpu in loader(dataset, args, False):
        batch = to_device(cpu, device); evidence, _ = build_evidence(adapter, batch)
        inp, proposal = detector_evidence(a2, batch, evidence); output = model(inp)
        for key, value in sample_batch(output, proposal, batch, evidence, args).items(): collected[key].append(value)
    return {key: np.concatenate(value) for key, value in collected.items()}


def sweep(samples: dict, temperature: float) -> tuple[list[dict], dict[str, OperatingMode]]:
    probability = 1 / (1 + np.exp(-samples["logits"] / temperature))
    rows = []
    for p_threshold in (0.50, 0.65, 0.80, 0.90, 0.95):
        for mu_threshold in (0.25, 0.50, 1.00):
            for sigma_threshold in (0.25, 0.50, 1.00, 2.00):
                apply = ((probability >= p_threshold) & (samples["mu"] >= mu_threshold)
                    & (samples["sigma"] <= sigma_threshold) & (np.abs(samples["proposal_update"]) <= 3))
                raw_error = samples["error"]; refined_error = np.abs(
                    np.sign(samples["proposal_update"]) * 0 + 0)  # allocated below
                # The exact refined error is available as A2 error wherever its update is authorized.
                refined_error = np.where(apply, samples["a2_error"], raw_error)
                changed = apply & (np.abs(samples["proposal_update"]) > 0.05)
                clean = raw_error <= 0.5; harmful = changed & (samples["a2_error"] > raw_error + 0.02)
                beneficial = changed & (samples["a2_error"] + 0.02 < raw_error)
                a2_gain = raw_error.mean() - samples["a2_error"].mean()
                gain = raw_error.mean() - refined_error.mean()
                rows.append({"probability_threshold": p_threshold, "error_threshold_px": mu_threshold,
                    "uncertainty_threshold_px": sigma_threshold, "intervention_coverage": changed.mean(),
                    "intervention_precision": beneficial.sum() / max(changed.sum(), 1),
                    "false_update_rate": (changed & clean).sum() / max(clean.sum(), 1),
                    "clean_pixel_degradation": (harmful & clean).sum() / max(clean.sum(), 1),
                    "epe_gain": gain, "retained_a2_gain": gain / max(a2_gain, 1e-8)})
    def choose(name, fp, clean, minimum_coverage):
        eligible = [r for r in rows if r["false_update_rate"] <= fp
            and r["clean_pixel_degradation"] <= clean and r["intervention_coverage"] >= minimum_coverage]
        score = ((lambda r: (r["intervention_precision"], r["epe_gain"])) if name == "ultra_safe"
                 else (lambda r: (r["epe_gain"], r["intervention_precision"])))
        selected = max(eligible, key=score, default=None)
        if selected is None:
            selected = min(rows, key=lambda r: (r["false_update_rate"] + r["clean_pixel_degradation"], -r["epe_gain"]))
        return OperatingMode(name, selected["probability_threshold"], selected["error_threshold_px"],
            selected["uncertainty_threshold_px"])
    modes = {"ultra_safe": choose("ultra_safe", .10, .05, .002),
        "balanced": choose("balanced", .15, .10, .005),
        "high_coverage": max((OperatingMode("high_coverage", r["probability_threshold"],
            r["error_threshold_px"], r["uncertainty_threshold_px"]) for r in rows if r["epe_gain"] >= 0),
            key=lambda m: next(x["intervention_coverage"] for x in rows if x["probability_threshold"] == m.probability_threshold and x["error_threshold_px"] == m.error_threshold_px and x["uncertainty_threshold_px"] == m.uncertainty_threshold_px),
            default=OperatingMode("high_coverage", .5, .25, 2.0))}
    return rows, modes


def map_metrics(prediction, raw, gt, valid, boundary, update) -> dict[str, float]:
    error = (prediction - gt).abs(); raw_error = (raw - gt).abs(); selected = valid.bool()
    changed = selected & (update.abs() > .05); clean = selected & (raw_error <= .5)
    harmful = changed & (error > raw_error + .02); helpful = changed & (error + .02 < raw_error)
    raw_good3 = selected & (raw_error <= 3)
    return {"valid_count": int(selected.sum()), "clean_count": int(clean.sum()),
        "changed_count": int(changed.sum()), "helpful_count": int(helpful.sum()),
        "harmful_count": int(harmful.sum()), "false_update_count": int((changed & clean).sum()),
        "clean_degradation_count": int((harmful & clean).sum()), "raw_good3_count": int(raw_good3.sum()),
        "epe": float(error[selected].mean()),
        "raw_epe": float(raw_error[selected].mean()), "refined_minus_raw_epe": float((error-raw_error)[selected].mean()),
        "bad1": float((error[selected] > 1).float().mean()), "bad3": float((error[selected] > 3).float().mean()),
        "boundary_epe": float(error[selected & boundary].mean()) if (selected & boundary).any() else float("nan"),
        "new_bad3": float((error[raw_good3] > 3).float().mean()) if raw_good3.any() else 0.0,
        "intervention_coverage": float(changed.sum() / selected.sum().clamp_min(1)),
        "intervention_precision": float(helpful.sum() / changed.sum().clamp_min(1)),
        "false_update_rate": float((changed & clean).sum() / clean.sum().clamp_min(1)),
        "clean_pixel_degradation": float((harmful & clean).sum() / clean.sum().clamp_min(1)),
        "mean_update_magnitude_clean": float(update[clean].abs().mean()) if clean.any() else 0.0,
        "mean_gain_modified": float((raw_error-error)[changed].mean()) if changed.any() else 0.0,
        "mean_loss_false_modified": float((error-raw_error)[harmful].mean()) if harmful.any() else 0.0}


def boundary_mask_tensor(gt: torch.Tensor) -> torch.Tensor:
    """Three-pixel cache-grid band around disparity gradients above one pixel."""
    dx = torch.nn.functional.pad((gt[..., 1:] - gt[..., :-1]).abs(), (0, 1, 0, 0))
    dy = torch.nn.functional.pad((gt[..., 1:, :] - gt[..., :-1, :]).abs(), (0, 0, 0, 1))
    edge = (dx > 1.0) | (dy > 1.0)
    return torch.nn.functional.max_pool2d(edge.float(), 3, stride=1, padding=1).bool()


@torch.no_grad()
def evaluate_seen(model, a2, adapter, dataset, modes, temperature, device, args) -> tuple[list[dict], dict]:
    rows = []; model.eval(); start = time.perf_counter()
    for cpu in loader(dataset, args, False):
        batch = to_device(cpu, device); evidence, _ = build_evidence(adapter, batch)
        inp, proposal = detector_evidence(a2, batch, evidence); detector = model(inp)
        heuristic_gate = .5 * evidence["forward_backward_confidence"] * (1-evidence["photometric_residual"]) * evidence["aligned_validity"].float()
        heuristic = batch["raw"] + heuristic_gate * (evidence["aligned_past_disparity"]-batch["raw"])
        predictions = {"raw": batch["raw"], "original_a2": proposal.disparity, "heuristic_bida_gate": heuristic}
        updates = {"raw": torch.zeros_like(batch["raw"]), "original_a2": proposal.update,
            "heuristic_bida_gate": heuristic-batch["raw"]}
        for name, mode in modes.items():
            authorization = authorization_mask(detector, mode=mode, temperature=temperature,
                aligned_valid=evidence["aligned_validity"], warp_support=evidence["warp_support"],
                proposal_update=proposal.update)
            predictions[f"authorized_{name}"] = authorized_update(batch["raw"], proposal.update, authorization)
            updates[f"authorized_{name}"] = torch.where(authorization, proposal.update, torch.zeros_like(proposal.update))
        oracle = (proposal.disparity-batch["gt"]).abs() < (batch["raw"]-batch["gt"]).abs()
        predictions["oracle_authorized_a2"] = authorized_update(batch["raw"], proposal.update, oracle)
        updates["oracle_authorized_a2"] = torch.where(oracle, proposal.update, torch.zeros_like(proposal.update))
        boundary = boundary_mask_tensor(batch["gt"])
        for threshold in COVERAGE_THRESHOLDS:
            common = ((batch["gt_coverage"] > threshold) & batch["raw_valid"].bool()
                & evidence["aligned_validity"] & evidence["warp_support"])
            for method, prediction in predictions.items():
                for index in range(batch["raw"].shape[0]):
                    row = {"backbone": batch["backbone"][index], "sequence": batch["sequence"][index],
                        "frame_id": batch["current_frame_id"][index], "coverage_threshold": threshold,
                        "method": method, **map_metrics(prediction[index:index+1], batch["raw"][index:index+1],
                            batch["gt"][index:index+1], common[index:index+1], boundary[index:index+1],
                            updates[method][index:index+1])}
                    rows.append(row)
    runtime = {"total_evaluation_seconds": time.perf_counter()-start,
        "detector_parameters": sum(p.numel() for p in model.parameters()),
        "a2_parameters": sum(p.numel() for p in a2.parameters())}
    return rows, runtime


def aggregate_rows(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows: groups[(row["backbone"], row["sequence"], row["coverage_threshold"], row["method"])].append(row)
    output = []
    for key, group in groups.items():
        count = sum(r["valid_count"] for r in group)
        weighted = lambda metric: sum(r[metric]*r["valid_count"] for r in group) / max(count, 1)
        clean_count=sum(r["clean_count"] for r in group); changed_count=sum(r["changed_count"] for r in group)
        raw_good3_count=sum(r["raw_good3_count"] for r in group)
        degradation = np.array([r["refined_minus_raw_epe"] for r in group])
        output.append({"backbone": key[0], "sequence": key[1], "coverage_threshold": key[2], "method": key[3],
            "frames": len(group), "valid_count": count, "clean_count":clean_count,
            "changed_count":changed_count,
            **{m: weighted(m) for m in ("epe","raw_epe","bad1","bad3","boundary_epe")},
            "new_bad3":sum(r["new_bad3"]*r["raw_good3_count"] for r in group)/max(raw_good3_count,1),
            "intervention_coverage":changed_count/max(count,1),
            "intervention_precision":sum(r["helpful_count"] for r in group)/max(changed_count,1),
            "false_update_rate":sum(r["false_update_count"] for r in group)/max(clean_count,1),
            "clean_pixel_degradation":sum(r["clean_degradation_count"] for r in group)/max(clean_count,1),
            "mean_update_magnitude_clean":sum(r["mean_update_magnitude_clean"]*r["clean_count"] for r in group)/max(clean_count,1),
            "percentage_frames_worsened": float((degradation > 0).mean()),
            "worst_frame_degradation": float(degradation.max()),
            "p95_frame_degradation": float(np.quantile(degradation, .95)),
            "catastrophic_tail_p99": float(np.quantile(degradation, .99)),
            "clean_frame_degradation": float((degradation[np.array([r["raw_epe"] <= 1 for r in group])] > 0).mean()) if any(r["raw_epe"] <= 1 for r in group) else float("nan")})
    return output


def risk_rows(samples, temperature) -> list[dict]:
    probability = 1/(1+np.exp(-samples["logits"]/temperature)); risk = np.abs(samples["mu"]-samples["error"])
    confidence = 1/(samples["sigma"]+1e-6); order = np.argsort(-confidence); rows=[]
    for coverage in (.01,.05,.10,.20,.50,1.0):
        take=order[:max(1,int(order.size*coverage))]; predicted=probability[take]>=.5
        true=samples["error"][take]>.5
        rows.append({"coverage":coverage,"regression_risk_mae":float(risk[take].mean()),
            "error_precision":float(true[predicted].mean()) if predicted.any() else 1.0,
            "mean_uncertainty":float(samples["sigma"][take].mean())})
    return rows


def precision_coverage_rows(samples, temperature) -> list[dict]:
    probability = 1/(1+np.exp(-samples["logits"]/temperature))
    score = probability * (samples["mu"]/(samples["mu"]+1)) / (samples["sigma"]+1e-3)
    order = np.argsort(-score); rows=[]
    helpful = samples["a2_error"] + .02 < samples["error"]
    raw_wrong = samples["error"] > .5
    for coverage in (.01,.05,.10,.20,.50,1.0):
        take=order[:max(1,int(order.size*coverage))]
        rows.append({"coverage":coverage,"raw_error_precision":float(raw_wrong[take].mean()),
            "a2_helpful_precision":float(helpful[take].mean()),
            "mean_available_gain":float((samples["error"][take]-samples["a2_error"][take]).mean())})
    return rows


def detector_sensitivity_rows(samples, temperature) -> list[dict]:
    probability=1/(1+np.exp(-samples["logits"]/temperature)); rows=[]
    for coverage in COVERAGE_THRESHOLDS:
        coverage_valid=samples["coverage"]>coverage
        for epsilon in ERROR_THRESHOLDS:
            valid=coverage_valid & (np.abs(samples["error"]-epsilon)>.1)
            label=(samples["error"]>epsilon).astype(np.float32)
            metric=binary_metrics(probability[valid],label[valid])
            rows.append({"coverage_threshold":coverage,"error_threshold_px":epsilon,
                "count":int(valid.sum()),"prevalence":float(label[valid].mean()) if valid.any() else float("nan"),
                **metric,"regression_mae":float(np.mean(np.abs(samples["mu"][coverage_valid]-samples["error"][coverage_valid])))})
    return rows


def evaluate(args) -> None:
    seed_all(args.seed); device=torch.device(args.device)
    checkpoint=args.checkpoint or args.output/"checkpoints/best_validation.pt"
    state=torch.load(checkpoint,map_location="cpu",weights_only=False)
    args.architecture=state["architecture"]; args.channels=int(state["channels"])
    frozen_config = state.get("config", {})
    for name in ("epsilon", "indifference_band", "coverage_threshold", "loss_mode", "false_positive_cost"):
        if name in frozen_config:
            setattr(args, name, frozen_config[name])
    model=RawErrorDetector(args.architecture,channels=args.channels).to(device)
    model.load_state_dict(state["model"]); model.eval()
    a2=load_a2(device); adapter=BiDAFlowInferenceAdapter("sea_raft",device=device)
    calibration=RawErrorDataset(SEEN_BACKBONES,CALIBRATION_SEQUENCES,coverage_threshold=args.coverage_threshold,
        max_pairs_per_sequence=args.max_validation_pairs,random_clip_start=False,seed=args.seed)
    samples=collect_samples(model,a2,adapter,calibration,device,args)
    temperature=fit_temperature(torch.from_numpy(samples["logits"]),torch.from_numpy(samples["label"]),
        torch.from_numpy(samples["class_valid"]),split="validation")
    sweep_rows,modes=sweep(samples,temperature)
    save_json(args.output/"operating_modes.json",{"temperature":temperature,
        "modes":{k:v.as_dict() for k,v in modes.items()},"provenance":"dataset_7_keyframe_1/2 only"})
    write_csv(args.output/"abstention_sweep.csv",sweep_rows)
    p=1/(1+np.exp(-samples["logits"]/temperature)); class_valid=samples["class_valid"]
    detector_summary={**binary_metrics(p[class_valid],samples["label"][class_valid]),
        "raw_error_mae":float(np.mean(np.abs(samples["mu"]-samples["error"]))),
        "pearson":correlation(samples["mu"],samples["error"]),
        "spearman":correlation(samples["mu"],samples["error"],True),
        "uncertainty_error_correlation":correlation(samples["sigma"],np.abs(samples["mu"]-samples["error"])),
        "temperature":temperature}
    save_json(args.output/"detector_metrics.json",detector_summary)
    write_csv(args.output/"risk_coverage.csv",risk_rows(samples,temperature))
    write_csv(args.output/"precision_coverage.csv",precision_coverage_rows(samples,temperature))
    save_json(args.output/"calibration_metrics.json",{
        "temperature":temperature,"brier_after":detector_summary["brier"],
        "ece_after":detector_summary["ece"],"fit_split":list(CALIBRATION_SEQUENCES)})
    test=RawErrorDataset(SEEN_BACKBONES,TEST_SEQUENCES,coverage_threshold=args.coverage_threshold,
        max_pairs_per_sequence=args.max_validation_pairs,random_clip_start=False,seed=args.seed)
    frame_rows,runtime=evaluate_seen(model,a2,adapter,test,modes,temperature,device,args)
    original_coverage=args.coverage_threshold
    args.coverage_threshold=.05
    sensitivity_samples=collect_samples(model,a2,adapter,test,device,args)
    args.coverage_threshold=original_coverage
    write_csv(args.output/"detector_metrics.csv",detector_sensitivity_rows(sensitivity_samples,temperature))
    sequence_rows=aggregate_rows(frame_rows); write_csv(args.output/"frame_metrics.csv",frame_rows)
    write_csv(args.output/"sequence_metrics.csv",sequence_rows); save_json(args.output/"runtime_summary.json",runtime)
    primary=[r for r in sequence_rows if r["coverage_threshold"]==.5 and r["method"]=="authorized_balanced"]
    per_backbone=[]
    for backbone in SEEN_BACKBONES:
        group=[r for r in primary if r["backbone"]==backbone]; total=sum(r["valid_count"] for r in group)
        weighted=lambda metric:sum(r[metric]*r["valid_count"] for r in group)/max(total,1)
        clean_total=sum(r["clean_count"] for r in group)
        safety=lambda metric:sum(r[metric]*r["clean_count"] for r in group)/max(clean_total,1)
        per_backbone.append({"backbone":backbone,"epe":weighted("epe"),"raw_epe":weighted("raw_epe"),
            "epe_gain":weighted("raw_epe")-weighted("epe"),"false_update_rate":safety("false_update_rate"),
            "clean_pixel_degradation":safety("clean_pixel_degradation"),"intervention_coverage":weighted("intervention_coverage")})
    write_csv(args.output/"per_backbone_metrics.csv",per_backbone)
    improved=sum(r["epe_gain"]>0 for r in per_backbone); aggregate_total=sum(r["valid_count"] for r in primary)
    w=lambda metric:sum(r[metric]*r["valid_count"] for r in primary)/max(aggregate_total,1)
    clean_total=sum(r["clean_count"] for r in primary)
    safety=lambda metric:sum(r[metric]*r["clean_count"] for r in primary)/max(clean_total,1)
    go=improved>=2 and safety("false_update_rate")<=.15 and safety("clean_pixel_degradation")<=.10 and (w("raw_epe")-w("epe"))>0
    summary={"detector":detector_summary,"balanced_seen":{m:w(m) for m in ("epe","raw_epe","false_update_rate","clean_pixel_degradation","intervention_coverage")},
        "seen_backbones_improved":improved,"promotion":"GO" if go else "NO-GO",
        "unseen_evaluated":False,"reason":"Unseen remains untouched unless seen promotion criteria pass."}
    summary["balanced_seen"]["false_update_rate"]=safety("false_update_rate")
    summary["balanced_seen"]["clean_pixel_degradation"]=safety("clean_pixel_degradation")
    unseen_path=args.output/"unseen_fast_foundation_complete.json"
    if unseen_path.exists():
        summary["unseen_evaluated"]=True; summary["unseen"]=json.loads(unseen_path.read_text())
        summary["reason"]="Frozen one-shot unseen result; no post-unseen tuning."
    save_json(args.output/"aggregate_summary.json",summary); save_json(args.output/"safety_summary.json",summary["balanced_seen"])
    save_json(args.output/"config.json",vars(args)); save_json(args.output/"split_manifest.json",split_manifest(args))
    print(json.dumps(clean_json(summary),indent=2))


def evaluate_unseen_once(args) -> None:
    """Apply already-frozen settings once; no calibration or selection is possible here."""
    completion = args.output / "unseen_fast_foundation_complete.json"
    if completion.exists():
        raise RuntimeError(f"one-shot unseen evaluation already completed: {completion}")
    seen_summary = json.loads((args.output / "aggregate_summary.json").read_text())
    if seen_summary.get("promotion") != "GO":
        raise RuntimeError("seen GO is required before unseen evaluation")
    frozen = json.loads((args.output / "operating_modes.json").read_text())
    modes = {name: OperatingMode(**value) for name, value in frozen["modes"].items()}
    temperature = float(frozen["temperature"])
    device = torch.device(args.device)
    checkpoint = args.checkpoint or args.output / "checkpoints/best_validation.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = RawErrorDetector(state["architecture"], channels=int(state["channels"])).to(device)
    model.load_state_dict(state["model"]); model.eval()
    a2 = load_a2(device); adapter = BiDAFlowInferenceAdapter("sea_raft", device=device)
    dataset = TemporalPairDataset(
        [PRIMARY_UNSEEN_BACKBONE], list(TEST_SEQUENCES),
        coverage_threshold=float(state["config"]["coverage_threshold"]),
        max_pairs_per_sequence=args.max_validation_pairs,
        random_clip_start=False, seed=args.seed,
    )
    rows, runtime = evaluate_seen(model, a2, adapter, dataset, modes, temperature, device, args)
    sequences = aggregate_rows(rows)
    write_csv(args.output / "unseen_frame_metrics.csv", rows)
    write_csv(args.output / "unseen_sequence_metrics.csv", sequences)
    primary = [r for r in sequences if r["coverage_threshold"] == .5 and r["method"] == "authorized_balanced"]
    total = sum(r["valid_count"] for r in primary)
    weighted = lambda metric: sum(r[metric] * r["valid_count"] for r in primary) / max(total, 1)
    clean_total = sum(r["clean_count"] for r in primary)
    safety = lambda metric: sum(r[metric] * r["clean_count"] for r in primary) / max(clean_total, 1)
    result = {
        "backbone": PRIMARY_UNSEEN_BACKBONE,
        "sequences": list(TEST_SEQUENCES),
        "checkpoint": str(checkpoint),
        "operating_modes_source": str(args.output / "operating_modes.json"),
        "epe": weighted("epe"), "raw_epe": weighted("raw_epe"),
        "epe_gain": weighted("raw_epe") - weighted("epe"),
        "false_update_rate": safety("false_update_rate"),
        "clean_pixel_degradation": safety("clean_pixel_degradation"),
        "intervention_coverage": weighted("intervention_coverage"),
        "runtime": runtime, "tuning_after_unseen": False,
    }
    save_json(completion, result)
    print(json.dumps(clean_json(result), indent=2))


def main() -> None:
    args=parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    if args.mode in {"smoke","train"}: train(args)
    elif args.mode == "evaluate": evaluate(args)
    else: evaluate_unseen_once(args)


if __name__ == "__main__": main()
