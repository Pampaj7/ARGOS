#!/usr/bin/env python3
"""Train and evaluate the minimal ARGOS v2 frozen-A2 applicability detector."""
from __future__ import annotations

import argparse
import csv
import hashlib
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
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

V2_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(V2_ROOT), str(V2_ROOT / "scripts")]

from argos_v2.sequences import accepted_sequences  # noqa: E402
from model_design.data.proposal_utility_dataset import (  # noqa: E402
    ProposalUtilityDataset,
    proposal_utility_targets,
    stratified_training_targets,
)
from model_design.data.raw_error_dataset import (  # noqa: E402
    CALIBRATION_SEQUENCES,
    TEST_SEQUENCES,
)
from model_design.data.temporal_pair_dataset import (  # noqa: E402
    SEEN_BACKBONES,
    TemporalPairDataset,
)
from model_design.external_components.bidavideo import (  # noqa: E402
    BiDAFlowInferenceAdapter,
    SEA_RAFT_CHECKPOINT,
)
from model_design.losses.proposal_utility_losses import (  # noqa: E402
    ProposalUtilityLossConfig,
    proposal_utility_losses,
)
from model_design.models.abstention import (  # noqa: E402
    OperatingMode,
    authorization_mask,
    calibrated_probability,
)
from model_design.models.proposal_applicability_detector import (  # noqa: E402
    RECEPTIVE_FIELDS,
    VARIANTS,
    ProposalApplicabilityDetector,
    ProposalEvidence,
    apply_frozen_proposal,
    proposal_authorization_mask,
)
from model_design.models.raw_error_detector import RawErrorDetector  # noqa: E402
from run_learned_t1_refiner import build_evidence  # noqa: E402
from run_raw_error_abstention import (  # noqa: E402
    A2_CHECKPOINT,
    aggregate_rows,
    boundary_mask_tensor,
    detector_evidence,
    load_a2,
    map_metrics,
)


P0_CHECKPOINT = V2_ROOT / "results/raw_error_abstention/full/checkpoints/best_validation.pt"
P0_MODES = V2_ROOT / "results/raw_error_abstention/full/operating_modes.json"
COVERAGE_THRESHOLDS = (0.05, 0.25, 0.50, 0.90)
EXPECTED_HASHES = {
    "a2": "6cd29277397001333ef3ce630b2f3bc04ec393cdc72e65aa5eb087afd3b389ea",
    "raw_error_detector": "78b1bb6cf809dc76448222e41e3bcfafb754bc9b7b6629edcdfa2e1a33444e67",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "train", "calibrate", "evaluate", "unseen"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--variant", choices=VARIANTS, default="P4")
    parser.add_argument("--channels", type=int, default=24)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--coverage-threshold", type=float, default=0.50)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8))
    parser.add_argument("--max-train-pairs", type=int, default=256)
    parser.add_argument("--max-validation-pairs", type=int, default=160)
    parser.add_argument("--sample-pixels-per-frame", type=int, default=2048)
    parser.add_argument("--training-pixels-per-batch", type=int, default=32768)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backbones", nargs="+", default=[])
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(""); return
    keys: list[str] = []
    for row in rows:
        keys.extend(key for key in row if key not in keys)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def atomic_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary); temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_artifacts() -> dict:
    hashes = {
        "a2": sha256(A2_CHECKPOINT),
        "raw_error_detector": sha256(P0_CHECKPOINT),
        "sea_raft": sha256(SEA_RAFT_CHECKPOINT),
        "bida_source": sha256(V2_ROOT / "model_design/external_components/bidavideo.py"),
    }
    for name, expected in EXPECTED_HASHES.items():
        if hashes[name] != expected:
            raise RuntimeError(f"frozen {name} hash mismatch: {hashes[name]}")
    return hashes


def split_manifest(args) -> dict:
    held_out = set(CALIBRATION_SEQUENCES) | set(TEST_SEQUENCES)
    return {
        "seed": args.seed,
        "train_sequences": [sequence for sequence in accepted_sequences() if sequence not in held_out],
        "calibration_sequences": list(CALIBRATION_SEQUENCES),
        "final_seen_sequences": list(TEST_SEQUENCES),
        "training_backbones": list(SEEN_BACKBONES),
        "unseen_backbones": ["Fast-FoundationStereo", "CREStereo"],
        "causal_pair": "t-1 -> t",
        "primary_coverage_threshold": args.coverage_threshold,
        "evaluation_coverage_thresholds": list(COVERAGE_THRESHOLDS),
        "unseen_and_ood_policy": "not loaded before frozen seen promotion",
    }


def loader(dataset, args, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
        generator=torch.Generator().manual_seed(args.seed), drop_last=False,
    )


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def load_p0(device: torch.device):
    state = torch.load(P0_CHECKPOINT, map_location="cpu", weights_only=False)
    model = RawErrorDetector(state["architecture"], channels=int(state["channels"]))
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval().requires_grad_(False)
    policy = json.loads(P0_MODES.read_text())
    mode = OperatingMode(**policy["modes"]["balanced"])
    return model, float(policy["temperature"]), mode


@torch.no_grad()
def frozen_proposal_evidence(a2, batch: dict, evidence: dict) -> tuple[ProposalEvidence, object]:
    proposal = a2(batch["raw"], evidence, batch["current_rgb"])
    return ProposalEvidence(
        raw=batch["raw"].detach(), aligned=evidence["aligned_past_disparity"].detach(),
        proposal=proposal.disparity.detach(), update=proposal.update.detach(),
        a2_error_gate=proposal.g_error.detach(), a2_memory_gate=proposal.c_memory.detach(),
        a2_delta=proposal.delta.detach(), raw_valid=batch["raw_valid"].detach(),
        aligned_valid=evidence["aligned_validity"].detach(),
        warp_support=evidence["warp_support"].detach(),
        flow_magnitude=evidence["flow_magnitude"].detach(),
        photometric_residual=evidence["photometric_residual"].detach(),
        forward_backward_error=evidence["forward_backward_error"].detach(),
        forward_backward_confidence=evidence["forward_backward_confidence"].detach(),
    ), proposal


def loss_config(variant: str) -> ProposalUtilityLossConfig:
    if variant in {"P1", "P2"}:
        return ProposalUtilityLossConfig()
    if variant == "P3":
        return ProposalUtilityLossConfig(heteroscedastic_weight=0.05)
    return ProposalUtilityLossConfig(
        heteroscedastic_weight=0.05,
        classification_weight=0.20,
        harmful_as_helpful_weight=0.50,
    )


def binary_metrics(score: np.ndarray, label: np.ndarray) -> dict:
    if not label.size or np.unique(label).size < 2:
        return {"auroc": None, "average_precision": None}
    return {
        "auroc": float(roc_auc_score(label, score)),
        "average_precision": float(average_precision_score(label, score)),
    }


def correlation(left: np.ndarray, right: np.ndarray, rank: bool = False) -> float | None:
    if left.size < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12: return None
    if rank:
        from scipy.stats import spearmanr
        return float(spearmanr(left, right).statistic)
    return float(np.corrcoef(left, right)[0, 1])


@torch.no_grad()
def validate(model, a2, adapter, data_loader, device, args) -> dict:
    model.eval(); sums = defaultdict(float); batches = 0; arrays = defaultdict(list)
    for cpu in data_loader:
        batch = to_device(cpu, device); evidence, _ = build_evidence(adapter, batch)
        inputs, proposal = frozen_proposal_evidence(a2, batch, evidence); output = model(inputs)
        target = proposal_utility_targets(
            batch, proposal.disparity, aligned_valid=evidence["aligned_validity"],
            warp_support=evidence["warp_support"], epsilon_px=args.epsilon,
            coverage_threshold=args.coverage_threshold,
        )
        losses = proposal_utility_losses(output, target, loss_config(args.variant))
        for key, value in losses.items(): sums[key] += float(value.detach())
        indices = target.regression_valid.flatten().nonzero().flatten()[::64]
        arrays["utility"].append(target.utility.flatten()[indices].cpu().numpy())
        arrays["pred"].append(output.utility.flatten()[indices].cpu().numpy())
        arrays["sigma"].append(output.sigma.flatten()[indices].cpu().numpy())
        score = output.utility if output.class_probability is None else output.class_probability[:, 2:3]
        arrays["score"].append(score.flatten()[indices].cpu().numpy())
        arrays["label"].append(target.helpful.flatten()[indices].cpu().numpy())
        batches += 1
    values = {k: v / max(batches, 1) for k, v in sums.items()}
    joined = {k: np.concatenate(v) if v else np.array([]) for k, v in arrays.items()}
    values.update(binary_metrics(joined["score"], joined["label"]))
    values.update({
        "utility_mae": float(np.mean(np.abs(joined["pred"] - joined["utility"]))) if joined["utility"].size else None,
        "utility_pearson": correlation(joined["pred"], joined["utility"]),
        "utility_spearman": correlation(joined["pred"], joined["utility"], True),
        "uncertainty_error_correlation": correlation(joined["sigma"], np.abs(joined["pred"] - joined["utility"])),
    })
    return values


def train(args) -> None:
    seed_all(args.seed); device = torch.device(args.device); smoke = args.mode == "smoke"
    manifest = split_manifest(args); train_sequences = manifest["train_sequences"]
    if smoke:
        train_sequences = ["dataset_3_keyframe_1"]
    backbones = ["S2M2-S"] if smoke else list(SEEN_BACKBONES)
    train_set = ProposalUtilityDataset(
        backbones, train_sequences, coverage_threshold=args.coverage_threshold,
        max_pairs_per_sequence=24 if smoke else args.max_train_pairs,
        random_clip_start=True, seed=args.seed,
    )
    val_set = ProposalUtilityDataset(
        backbones, list(CALIBRATION_SEQUENCES), coverage_threshold=args.coverage_threshold,
        max_pairs_per_sequence=8 if smoke else args.max_validation_pairs,
        random_clip_start=False, seed=args.seed,
    )
    epochs = 25 if smoke else args.epochs
    model = ProposalApplicabilityDetector(args.variant, channels=args.channels).to(device)
    a2 = load_a2(device); adapter = BiDAFlowInferenceAdapter("sea_raft", device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_loader = loader(train_set, args, True)
    validation_loader = loader(val_set, args, False)
    history_path = args.output / "training_history.csv"; history: list[dict] = []
    start_epoch = 0; best = -math.inf; global_step = 0
    final_path = args.output / "checkpoints/final.pt"
    if args.resume and final_path.exists() and not smoke:
        state = torch.load(final_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]); best = float(state["best_validation_ap"])
        if history_path.exists():
            history = list(csv.DictReader(history_path.open()))
    initial_loss = None
    for epoch in range(start_epoch, epochs):
        model.train(); sums = defaultdict(float); batches = 0
        for cpu in train_loader:
            batch = to_device(cpu, device); evidence, _ = build_evidence(adapter, batch)
            inputs, proposal = frozen_proposal_evidence(a2, batch, evidence); output = model(inputs)
            target = proposal_utility_targets(
                batch, proposal.disparity, aligned_valid=evidence["aligned_validity"],
                warp_support=evidence["warp_support"], epsilon_px=args.epsilon,
                coverage_threshold=args.coverage_threshold,
            )
            target = stratified_training_targets(
                target, proposal.update, batch["gt"], maximum_pixels=args.training_pixels_per_batch,
            )
            losses = proposal_utility_losses(output, target, loss_config(args.variant))
            if initial_loss is None: initial_loss = float(losses["total"].detach())
            optimizer.zero_grad(set_to_none=True); losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            for key, value in losses.items(): sums[key] += float(value.detach())
            batches += 1; global_step += 1
            if args.steps and global_step >= args.steps: break
        metrics = validate(model, a2, adapter, validation_loader, device, args)
        row = {"epoch": epoch + 1, **{f"train_{k}": v / max(batches, 1) for k, v in sums.items()},
               **{f"validation_{k}": v for k, v in metrics.items()}}
        history.append(row); write_csv(history_path, history)
        score = float(metrics.get("average_precision") or -math.inf)
        payload = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1,
            "best_validation_ap": max(best, score), "variant": args.variant, "channels": args.channels,
            "config": vars(args), "loss_config": asdict(loss_config(args.variant)),
            "split_manifest": manifest, "frozen_hashes": verify_frozen_artifacts(),
        }
        atomic_checkpoint(final_path, payload)
        if score > best:
            best = score; payload["best_validation_ap"] = best
            atomic_checkpoint(args.output / "checkpoints/best_validation.pt", payload)
        print(json.dumps(clean(row)), flush=True)
        if args.steps and global_step >= args.steps: break
    save_json(args.output / "config.json", vars(args))
    save_json(args.output / "split_manifest.json", manifest)
    save_json(args.output / "parameter_summary.json", {
        "variant": args.variant, "parameters": sum(p.numel() for p in model.parameters()),
        "receptive_field": RECEPTIVE_FIELDS[args.variant], "trainable_component": "proposal applicability detector only",
    })
    if smoke:
        final_loss = float(history[-1]["train_total"])
        result = {
            "initial_loss": initial_loss, "final_loss": final_loss,
            "loss_reduction_fraction": (initial_loss - final_loss) / max(abs(initial_loss), 1e-8),
            "utility_correlation": history[-1]["validation_utility_spearman"],
            "finite": math.isfinite(final_loss),
            "passed": math.isfinite(final_loss) and final_loss < initial_loss * .75,
        }
        save_json(args.output / "smoke_summary.json", result); print(json.dumps(result, indent=2))
        if not result["passed"]: raise RuntimeError("proposal applicability smoke did not overfit")


def load_model(args, device: torch.device):
    checkpoint = args.checkpoint or args.output / "checkpoints/best_validation.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = ProposalApplicabilityDetector(state["variant"], channels=int(state["channels"]))
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()
    return model, state, checkpoint


@torch.no_grad()
def collect_calibration(model, a2, p0, p0_temperature, p0_mode, adapter, dataset, device, args) -> dict[str, np.ndarray]:
    arrays = defaultdict(list)
    for cpu in loader(dataset, args, False):
        batch = to_device(cpu, device); evidence, _ = build_evidence(adapter, batch)
        inputs, proposal = frozen_proposal_evidence(a2, batch, evidence); output = model(inputs)
        p0_input, _ = detector_evidence(a2, batch, evidence); p0_output = p0(p0_input)
        p0_auth = authorization_mask(
            p0_output, mode=p0_mode, temperature=p0_temperature,
            aligned_valid=evidence["aligned_validity"], warp_support=evidence["warp_support"],
            proposal_update=proposal.update,
        )
        target = proposal_utility_targets(
            batch, proposal.disparity, aligned_valid=evidence["aligned_validity"],
            warp_support=evidence["warp_support"], epsilon_px=args.epsilon,
            coverage_threshold=args.coverage_threshold,
        )
        indices = target.regression_valid.flatten().nonzero().flatten()
        limit = args.sample_pixels_per_frame * batch["raw"].shape[0]
        if indices.numel() > limit:
            positions = torch.linspace(0, indices.numel() - 1, limit, device=device).long()
            indices = indices[positions]
        take = lambda value: value.flatten()[indices].float().cpu().numpy()
        score = output.utility if output.class_probability is None else output.class_probability[:, 2:3]
        for key, value in {
            "utility": target.utility, "predicted_utility": output.utility,
            "sigma": output.sigma, "score": score,
            "raw_error": target.raw_error, "proposal_error": target.proposal_error,
            "update": proposal.update, "p0_score": calibrated_probability(p0_output.logits, p0_temperature),
            "p0_authorized": p0_auth,
        }.items(): arrays[key].append(take(value))
        arrays["true_class"].append(take(target.classes).astype(np.int64))
        if output.class_logits is not None:
            arrays["predicted_class"].append(take(output.class_logits.argmax(1, keepdim=True)).astype(np.int64))
    return {key: np.concatenate(value) for key, value in arrays.items()}


def decision_metrics(samples: dict, authorization: np.ndarray, epsilon: float) -> dict:
    changed = authorization & (np.abs(samples["update"]) > .05)
    helpful = samples["utility"] > epsilon
    harmful = samples["utility"] < -epsilon
    clean = samples["raw_error"] <= .5
    clean_harm = changed & clean & (samples["utility"] < -.02)
    gain = float(np.mean(samples["utility"] * authorization))
    old_gain = float(np.mean(samples["utility"] * samples["p0_authorized"].astype(bool)))
    return {
        "intervention_coverage": float(changed.mean()),
        "intervention_precision": float((changed & helpful).sum() / max(changed.sum(), 1)),
        "intervention_recall": float((changed & helpful).sum() / max(helpful.sum(), 1)),
        "false_update_rate": float((changed & clean).sum() / max(clean.sum(), 1)),
        "clean_pixel_degradation": float(clean_harm.sum() / max(clean.sum(), 1)),
        "harmful_proposal_acceptance_rate": float((changed & harmful).sum() / max(harmful.sum(), 1)),
        "epe_gain": gain,
        "retained_existing_authorized_gain": gain / max(old_gain, 1e-8),
        "selected_candidate_regret": float(np.mean(np.maximum(samples["utility"], 0) - samples["utility"] * authorization)),
    }


def calibrate(args) -> None:
    seed_all(args.seed); device = torch.device(args.device)
    model, state, checkpoint = load_model(args, device); args.variant = state["variant"]
    a2 = load_a2(device); p0, p0_temperature, p0_mode = load_p0(device)
    adapter = BiDAFlowInferenceAdapter("sea_raft", device=device)
    dataset = ProposalUtilityDataset(
        SEEN_BACKBONES, CALIBRATION_SEQUENCES, coverage_threshold=args.coverage_threshold,
        max_pairs_per_sequence=args.max_validation_pairs, random_clip_start=False, seed=args.seed,
    )
    samples = collect_calibration(model, a2, p0, p0_temperature, p0_mode, adapter, dataset, device, args)
    sigma_thresholds = (999.0,) if args.variant in {"P1", "P2"} else (.10, .25, .50, 1.0, 2.0)
    rows = []
    require_class = args.variant == "P4"
    for margin in (0.0, .02, .05, .10, .25):
        for sigma in sigma_thresholds:
            authorize = (samples["predicted_utility"] > margin) & (samples["sigma"] < sigma)
            if require_class:
                authorize &= samples["predicted_class"] == 2
            rows.append({"utility_margin_px": margin, "uncertainty_threshold_px": sigma,
                         "require_helpful_class": require_class,
                         **decision_metrics(samples, authorize, args.epsilon)})
    eligible = [r for r in rows if r["false_update_rate"] < .05 and r["clean_pixel_degradation"] < .03
                and r["intervention_coverage"] >= .002]
    balanced = max(eligible, key=lambda r: (r["epe_gain"], r["intervention_precision"]), default=None)
    safe_candidates = [r for r in rows if r["false_update_rate"] < .02 and r["clean_pixel_degradation"] < .01
                       and r["intervention_coverage"] >= .001]
    safe = max(safe_candidates, key=lambda r: (r["intervention_precision"], r["epe_gain"]), default=balanced)
    if balanced is None:
        balanced = min(rows, key=lambda r: (r["false_update_rate"] + r["clean_pixel_degradation"], -r["epe_gain"]))
    selected = {"balanced": balanced, "safe": safe or balanced}
    selected_coverage = balanced["intervention_coverage"]
    update_threshold = float(np.quantile(np.abs(samples["update"]), max(0, 1 - selected_coverage)))
    nontrivial_update_prevalence = float((np.abs(samples["update"]) > .05).mean())
    # Intervention coverage excludes proposals below 0.05 px. Correct for that
    # prevalence so the random authorization is coverage-matched on calibration.
    random_probability = min(1.0, selected_coverage / max(nontrivial_update_prevalence, 1e-8))
    selected["matched_baselines"] = {
        "random_probability": random_probability,
        "update_magnitude_threshold_px": update_threshold,
        "calibration_intervention_coverage": selected_coverage,
        "calibration_nontrivial_update_prevalence": nontrivial_update_prevalence,
    }
    helpful = samples["utility"] > args.epsilon
    metrics = {
        "variant": args.variant, "epsilon_px": args.epsilon,
        "sample_count": int(helpful.size), "helpful_prevalence": float(helpful.mean()),
        "proposal_predictor": binary_metrics(samples["score"], helpful),
        "raw_error_detector_proxy": binary_metrics(samples["p0_score"], helpful),
        "utility_mae": float(np.mean(np.abs(samples["predicted_utility"] - samples["utility"]))),
        "utility_pearson": correlation(samples["predicted_utility"], samples["utility"]),
        "utility_spearman": correlation(samples["predicted_utility"], samples["utility"], True),
        "uncertainty_error_correlation": correlation(samples["sigma"], np.abs(samples["predicted_utility"] - samples["utility"])),
    }
    if "predicted_class" in samples:
        metrics["three_class_macro_f1"] = float(f1_score(samples["true_class"], samples["predicted_class"], average="macro"))
        matrix = confusion_matrix(samples["true_class"], samples["predicted_class"], labels=(0, 1, 2))
        write_csv(args.output / "candidate_confusion_matrix.csv", [
            {"true_class": name, "pred_harmful": int(matrix[i, 0]), "pred_indifferent": int(matrix[i, 1]), "pred_helpful": int(matrix[i, 2])}
            for i, name in enumerate(("harmful", "indifferent", "helpful"))
        ])
    curves = []
    confidence = samples["predicted_utility"] / np.maximum(samples["sigma"], 1e-3)
    order = np.argsort(-confidence)
    for coverage in (.01, .05, .10, .20, .50, 1.0):
        take = order[:max(1, int(order.size * coverage))]
        curves.append({
            "coverage": coverage,
            "helpful_precision": float(helpful[take].mean()),
            "mean_true_utility": float(samples["utility"][take].mean()),
            "mean_absolute_utility_error": float(np.abs(samples["predicted_utility"][take] - samples["utility"][take]).mean()),
        })
    write_csv(args.output / "calibration_sweep.csv", rows)
    write_csv(args.output / "coverage_risk_gain.csv", curves)
    save_json(args.output / "calibration_metrics.json", metrics)
    save_json(args.output / "operating_points.json", selected)
    save_json(args.output / "frozen_candidate_manifest.json", {
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "variant": args.variant, "channels": state["channels"], "epsilon_px": args.epsilon,
        "operating_points": selected, "selected_on": list(CALIBRATION_SEQUENCES),
        "frozen_dependency_hashes": verify_frozen_artifacts(),
        "unseen_loaded": False,
    })
    print(json.dumps(clean({"metrics": metrics, "selected": selected}), indent=2))


def aggregate_overall(rows: list[dict], *, threshold: float = .5) -> tuple[list[dict], dict]:
    sequence = aggregate_rows(rows)
    primary = [row for row in sequence if float(row["coverage_threshold"]) == threshold]
    primary_frames = [row for row in rows if float(row["coverage_threshold"]) == threshold]
    frames_by_method: dict[str, list[dict]] = defaultdict(list)
    for row in primary_frames:
        frames_by_method[row["method"]].append(row)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in primary: groups[(row["backbone"], row["method"])].append(row)
    per_backbone = []
    for (backbone, method), members in groups.items():
        count = sum(int(r["valid_count"]) for r in members)
        weighted = lambda key: sum(float(r[key]) * int(r["valid_count"]) for r in members) / max(count, 1)
        per_backbone.append({"backbone": backbone, "method": method, "valid_count": count,
                             **{key: weighted(key) for key in ("epe", "raw_epe", "bad1", "bad3", "boundary_epe",
                                  "intervention_coverage", "intervention_precision", "false_update_rate",
                                  "clean_pixel_degradation", "new_bad3")}})
    methods: dict[str, list[dict]] = defaultdict(list)
    for row in primary: methods[row["method"]].append(row)
    overall = {}
    for method, members in methods.items():
        count = sum(int(r["valid_count"]) for r in members)
        weighted = lambda key: sum(float(r[key]) * int(r["valid_count"]) for r in members) / max(count, 1)
        frame_deltas = np.array([
            float(r["refined_minus_raw_epe"]) for r in frames_by_method[method]
        ])
        overall[method] = {
            "valid_count": count,
            **{key: weighted(key) for key in ("epe", "raw_epe", "bad1", "bad3", "boundary_epe",
                 "intervention_coverage", "intervention_precision", "false_update_rate",
                 "clean_pixel_degradation", "new_bad3", "mean_update_magnitude_clean")},
            "gain": weighted("raw_epe") - weighted("epe"),
            "frames_worsened_fraction": float((frame_deltas > 0).mean()),
            "worst_frame_degradation": float(frame_deltas.max()),
            "p95_frame_degradation": float(np.quantile(frame_deltas, .95)),
        }
    return sequence, {"per_backbone": per_backbone, "overall": overall}


@torch.no_grad()
def evaluate_dataset(model, a2, p0, p0_temperature, p0_mode, adapter, dataset, operation, device, args):
    rows = []; start = time.perf_counter(); detector_ms = 0.0; arrays = defaultdict(list)
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    random_generator = torch.Generator(device=device).manual_seed(args.seed)
    for cpu in loader(dataset, args, False):
        batch = to_device(cpu, device); evidence, _ = build_evidence(adapter, batch)
        inputs, proposal = frozen_proposal_evidence(a2, batch, evidence)
        if device.type == "cuda": torch.cuda.synchronize(device)
        tick = time.perf_counter(); output = model(inputs)
        if device.type == "cuda": torch.cuda.synchronize(device)
        detector_ms += (time.perf_counter() - tick) * 1000
        p0_input, _ = detector_evidence(a2, batch, evidence); p0_output = p0(p0_input)
        p0_auth = authorization_mask(
            p0_output, mode=p0_mode, temperature=p0_temperature,
            aligned_valid=evidence["aligned_validity"], warp_support=evidence["warp_support"],
            proposal_update=proposal.update,
        )
        proposal_auth = proposal_authorization_mask(
            output, inputs, utility_margin_px=float(operation["utility_margin_px"]),
            uncertainty_threshold_px=float(operation["uncertainty_threshold_px"]),
            require_helpful_class=bool(operation["require_helpful_class"]),
        )
        utility = (batch["raw"] - batch["gt"]).abs() - (proposal.disparity - batch["gt"]).abs()
        oracle = utility > args.epsilon
        random_auth = torch.rand(proposal_auth.shape, generator=random_generator, device=device) < float(operation["random_probability"])
        magnitude_auth = proposal.update.abs() >= float(operation["update_magnitude_threshold_px"])
        primary_common = ((batch["gt_coverage"] > args.coverage_threshold) & batch["raw_valid"].bool()
                          & evidence["aligned_validity"].bool() & evidence["warp_support"].bool())
        indices = primary_common.flatten().nonzero().flatten()
        limit = args.sample_pixels_per_frame * batch["raw"].shape[0]
        if indices.numel() > limit:
            positions = torch.linspace(0, indices.numel() - 1, limit, device=device).long()
            indices = indices[positions]
        take = lambda value: value.flatten()[indices].float().cpu().numpy()
        score = output.utility if output.class_probability is None else output.class_probability[:, 2:3]
        raw_error = (batch["raw"] - batch["gt"]).abs()
        proposal_error = (proposal.disparity - batch["gt"]).abs()
        for key, value in {
            "utility": utility, "predicted_utility": output.utility, "sigma": output.sigma,
            "score": score, "raw_error": raw_error, "proposal_error": proposal_error,
            "update": proposal.update, "p0_score": calibrated_probability(p0_output.logits, p0_temperature),
            "p0_authorized": p0_auth, "proposal_authorized": proposal_auth,
        }.items(): arrays[key].append(take(value))
        true_class = torch.ones_like(utility, dtype=torch.long)
        true_class[utility < -args.epsilon] = 0; true_class[utility > args.epsilon] = 2
        arrays["true_class"].append(take(true_class).astype(np.int64))
        if output.class_logits is not None:
            arrays["predicted_class"].append(take(output.class_logits.argmax(1, keepdim=True)).astype(np.int64))
        predictions = {
            "raw": batch["raw"],
            "a2_unconditional": proposal.disparity,
            "raw_error_authorized_a2": apply_frozen_proposal(batch["raw"], proposal.disparity, p0_auth),
            "proposal_authorized_a2": apply_frozen_proposal(batch["raw"], proposal.disparity, proposal_auth),
            "oracle_proposal_authorization": apply_frozen_proposal(batch["raw"], proposal.disparity, oracle),
            "random_matched_coverage": apply_frozen_proposal(batch["raw"], proposal.disparity, random_auth),
            "update_magnitude_matched_coverage": apply_frozen_proposal(batch["raw"], proposal.disparity, magnitude_auth),
        }
        authorizations = {
            "raw": torch.zeros_like(proposal_auth), "a2_unconditional": torch.ones_like(proposal_auth),
            "raw_error_authorized_a2": p0_auth, "proposal_authorized_a2": proposal_auth,
            "oracle_proposal_authorization": oracle, "random_matched_coverage": random_auth,
            "update_magnitude_matched_coverage": magnitude_auth,
        }
        boundary = boundary_mask_tensor(batch["gt"])
        for threshold in COVERAGE_THRESHOLDS:
            common = ((batch["gt_coverage"] > threshold) & batch["raw_valid"].bool()
                      & evidence["aligned_validity"].bool() & evidence["warp_support"].bool())
            for method, prediction in predictions.items():
                update = torch.where(authorizations[method], proposal.update, torch.zeros_like(proposal.update))
                for index in range(batch["raw"].shape[0]):
                    rows.append({
                        "backbone": batch["backbone"][index], "sequence": batch["sequence"][index],
                        "frame_id": batch["current_frame_id"][index], "coverage_threshold": threshold,
                        "method": method,
                        **map_metrics(
                            prediction[index:index+1], batch["raw"][index:index+1], batch["gt"][index:index+1],
                            common[index:index+1], boundary[index:index+1], update[index:index+1],
                        ),
                    })
    frames = len(dataset)
    joined = {key: np.concatenate(value) for key, value in arrays.items()}
    return rows, {
        "frames": frames, "total_seconds": time.perf_counter() - start,
        "detector_latency_ms_per_frame": detector_ms / max(frames, 1),
        "detector_parameters": sum(p.numel() for p in model.parameters()),
        "a2_parameters": sum(p.numel() for p in a2.parameters()),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
    }, joined


def evaluate(args, unseen: bool = False) -> None:
    seed_all(args.seed); device = torch.device(args.device)
    model, state, checkpoint = load_model(args, device); args.variant = state["variant"]
    frozen_path = args.output / "frozen_candidate_manifest.json"
    if not frozen_path.exists(): raise RuntimeError("calibration must freeze an operating point before evaluation")
    frozen = json.loads(frozen_path.read_text())
    if sha256(checkpoint) != frozen["checkpoint_sha256"]: raise RuntimeError("selected checkpoint hash changed")
    if verify_frozen_artifacts() != frozen["frozen_dependency_hashes"]: raise RuntimeError("frozen dependency hash changed")
    points = frozen["operating_points"]; point = points["balanced"] | points["matched_baselines"]
    if unseen:
        seen_summary_path = args.output / "aggregate_summary.json"
        if not seen_summary_path.exists() or not json.loads(seen_summary_path.read_text()).get("seen_promotion_passed"):
            raise RuntimeError("unseen backbones cannot be loaded before seen promotion passes")
        backbones = args.backbones
        if not backbones or any(b not in {"Fast-FoundationStereo", "CREStereo"} for b in backbones):
            raise ValueError("--backbones must contain only frozen unseen backbones")
        marker = args.output / "unseen_complete.json"
        if marker.exists(): raise RuntimeError("unseen evaluation already completed; no post-hoc rerun")
        dataset = TemporalPairDataset(
            backbones, TEST_SEQUENCES, coverage_threshold=args.coverage_threshold,
            max_pairs_per_sequence=args.max_validation_pairs, random_clip_start=False, seed=args.seed,
        )
        prefix = "unseen_"
    else:
        dataset = ProposalUtilityDataset(
            SEEN_BACKBONES, TEST_SEQUENCES, coverage_threshold=args.coverage_threshold,
            max_pairs_per_sequence=args.max_validation_pairs, random_clip_start=False, seed=args.seed,
        )
        prefix = ""
    a2 = load_a2(device); p0, p0_temperature, p0_mode = load_p0(device)
    adapter = BiDAFlowInferenceAdapter("sea_raft", device=device)
    rows, runtime, samples = evaluate_dataset(model, a2, p0, p0_temperature, p0_mode, adapter, dataset, point, device, args)
    sequence, summary = aggregate_overall(rows)
    write_csv(args.output / f"{prefix}frame_metrics.csv", rows)
    write_csv(args.output / f"{prefix}sequence_metrics.csv", sequence)
    write_csv(args.output / f"{prefix}per_backbone.csv", summary["per_backbone"])
    save_json(args.output / f"{prefix}runtime_summary.json", runtime)
    detector_rows = []
    for epsilon in (.05, .10, .25, .50):
        label = samples["utility"] > epsilon
        detector_rows.append({
            "epsilon_px": epsilon, "helpful_prevalence": float(label.mean()),
            **{f"proposal_{k}": v for k, v in binary_metrics(samples["score"], label).items()},
            **{f"p0_proxy_{k}": v for k, v in binary_metrics(samples["p0_score"], label).items()},
        })
    write_csv(args.output / f"{prefix}detector_metrics.csv", detector_rows)
    if "predicted_class" in samples:
        matrix = confusion_matrix(samples["true_class"], samples["predicted_class"], labels=(0, 1, 2))
        write_csv(args.output / f"{prefix}candidate_confusion_matrix.csv", [
            {"true_class": name, "pred_harmful": int(matrix[i, 0]), "pred_indifferent": int(matrix[i, 1]), "pred_helpful": int(matrix[i, 2])}
            for i, name in enumerate(("harmful", "indifferent", "helpful"))
        ])
    curves = []
    confidence = samples["predicted_utility"] / np.maximum(samples["sigma"], 1e-3)
    order = np.argsort(-confidence)
    for coverage in (.01, .05, .10, .20, .50, 1.0):
        take_indices = order[:max(1, int(order.size * coverage))]
        curves.append({
            "coverage": coverage,
            "helpful_precision": float((samples["utility"][take_indices] > args.epsilon).mean()),
            "mean_true_utility": float(samples["utility"][take_indices].mean()),
            "mean_absolute_utility_error": float(np.abs(samples["predicted_utility"][take_indices] - samples["utility"][take_indices]).mean()),
        })
    write_csv(args.output / f"{prefix}coverage_risk_gain.csv", curves)
    diagnostic = {
        "utility_mae": float(np.abs(samples["predicted_utility"] - samples["utility"]).mean()),
        "utility_pearson": correlation(samples["predicted_utility"], samples["utility"]),
        "utility_spearman": correlation(samples["predicted_utility"], samples["utility"], True),
        "uncertainty_error_correlation": correlation(samples["sigma"], np.abs(samples["predicted_utility"] - samples["utility"])),
        "proposal_decision": decision_metrics(samples, samples["proposal_authorized"].astype(bool), args.epsilon),
        "raw_error_detector_decision": decision_metrics(samples, samples["p0_authorized"].astype(bool), args.epsilon),
    }
    save_json(args.output / f"{prefix}decision_diagnostics.json", diagnostic)
    if unseen:
        save_json(args.output / "unseen_summary.json", summary | {"decision_diagnostics": diagnostic})
        save_json(args.output / "unseen_complete.json", {
            "backbones": args.backbones, "checkpoint_sha256": sha256(checkpoint), "completed": True,
        })
        print(json.dumps(clean(summary["overall"]), indent=2)); return
    overall = summary["overall"]
    raw = overall["raw"]; proposal = overall["proposal_authorized_a2"]
    old = overall["raw_error_authorized_a2"]
    per_backbone = [row for row in summary["per_backbone"] if row["method"] == "proposal_authorized_a2"]
    improves = sum(row["epe"] < row["raw_epe"] for row in per_backbone)
    existing_gain = old["gain"]
    gates = {
        "false_update_below_5pct": proposal["false_update_rate"] < .05,
        "clean_degradation_below_3pct": proposal["clean_pixel_degradation"] < .03,
        "retains_70pct_existing_gain": proposal["gain"] >= .70 * existing_gain,
        "nonzero_coverage": proposal["intervention_coverage"] >= .002,
        "all_three_seen_backbones_improve": improves == 3,
    }
    # Proposal AP must have been better than the P0 proxy on calibration.
    calibration = json.loads((args.output / "calibration_metrics.json").read_text())
    gates["proposal_ap_exceeds_raw_error_proxy"] = (
        calibration["proposal_predictor"]["average_precision"]
        > calibration["raw_error_detector_proxy"]["average_precision"]
    )
    passed = all(gates.values())
    final = {
        "variant": args.variant, "checkpoint": str(checkpoint), "primary_grid": "cache-grid-from-cached-predictions",
        "coverage_threshold": .5, "units": "pixels at width 180", "weighting": "pixel weighted",
        "metrics": overall, "promotion_gates": gates, "seen_promotion_passed": passed,
        "decision_diagnostics": diagnostic,
        "frozen_hashes": verify_frozen_artifacts(),
    }
    save_json(args.output / "aggregate_summary.json", final)
    save_json(args.output / "safety_summary.json", {
        method: {key: values[key] for key in ("false_update_rate", "clean_pixel_degradation", "new_bad3",
            "intervention_coverage", "intervention_precision", "frames_worsened_fraction",
            "worst_frame_degradation", "p95_frame_degradation", "mean_update_magnitude_clean")}
        for method, values in overall.items()
    })
    print(json.dumps(clean(final), indent=2))


def main() -> int:
    args = arguments(); args.output.mkdir(parents=True, exist_ok=True)
    if args.mode in {"smoke", "train"}: train(args)
    elif args.mode == "calibrate": calibrate(args)
    elif args.mode == "evaluate": evaluate(args)
    else: evaluate(args, unseen=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
