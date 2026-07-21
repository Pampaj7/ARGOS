#!/usr/bin/env python3
"""Select and evaluate frozen veto-only authorization for ARGOS v2."""
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
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

V2_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(V2_ROOT), str(V2_ROOT / "scripts")]

from model_design.data.proposal_utility_dataset import ProposalUtilityDataset  # noqa: E402
from model_design.data.raw_error_dataset import CALIBRATION_SEQUENCES, TEST_SEQUENCES  # noqa: E402
from model_design.data.temporal_pair_dataset import SEEN_BACKBONES, TemporalPairDataset  # noqa: E402
from model_design.external_components.bidavideo import BiDAFlowInferenceAdapter, SEA_RAFT_CHECKPOINT  # noqa: E402
from model_design.models.abstention import authorization_mask  # noqa: E402
from model_design.models.dual_stage_authorization import (  # noqa: E402
    VetoPolicy,
    apply_cascade,
    cascade_authorization,
    p4_harmful_probability,
    veto_mask,
)
from model_design.models.proposal_applicability_detector import (  # noqa: E402
    ProposalApplicabilityDetector,
    proposal_authorization_mask,
)
from run_learned_t1_refiner import build_evidence  # noqa: E402
from run_proposal_applicability import (  # noqa: E402
    P0_CHECKPOINT,
    P0_MODES,
    aggregate_overall,
    frozen_proposal_evidence,
    load_p0,
    sha256,
)
from run_raw_error_abstention import (  # noqa: E402
    A2_CHECKPOINT,
    boundary_mask_tensor,
    detector_evidence,
    load_a2,
    map_metrics,
)


P4_CHECKPOINT = V2_ROOT / "results/proposal_applicability/P4/checkpoints/best_validation.pt"
P4_OPERATING_POINTS = V2_ROOT / "results/proposal_applicability/P4/operating_points.json"
EXPECTED_HASHES = {
    "a2": "6cd29277397001333ef3ce630b2f3bc04ec393cdc72e65aa5eb087afd3b389ea",
    "raw_error_detector": "78b1bb6cf809dc76448222e41e3bcfafb754bc9b7b6629edcdfa2e1a33444e67",
    "p4": "c4d7d732b44ede1bb831b7789d6791907412b99345105f998c49c1cecde5bd2b",
    "sea_raft": "1a21575ed6ca2c6945fb8e25c4169d241cf59ee5d12b8802c01c965206268cac",
    "bida_source": "133a13f8a4dd89065f736484f1dba1811b40e0f1272d0bbec87d74074bf5c530",
}
COVERAGE_THRESHOLDS = (0.05, 0.25, 0.50, 0.90)
EPSILON = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "select", "evaluate", "unseen"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8))
    parser.add_argument("--max-pairs", type=int, default=160)
    parser.add_argument("--sample-pixels-per-frame", type=int, default=2048)
    parser.add_argument("--coverage-threshold", type=float, default=.50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--backbones", nargs="+", default=[])
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


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
    if not rows: path.write_text(""); return
    keys: list[str] = []
    for row in rows: keys.extend(key for key in row if key not in keys)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def verify_hashes() -> dict:
    actual = {
        "a2": sha256(A2_CHECKPOINT), "raw_error_detector": sha256(P0_CHECKPOINT),
        "p4": sha256(P4_CHECKPOINT), "sea_raft": sha256(SEA_RAFT_CHECKPOINT),
        "bida_source": sha256(V2_ROOT / "model_design/external_components/bidavideo.py"),
    }
    if actual != EXPECTED_HASHES: raise RuntimeError(f"frozen artifact hash mismatch: {actual}")
    return actual


def load_p4(device: torch.device) -> ProposalApplicabilityDetector:
    state = torch.load(P4_CHECKPOINT, map_location="cpu", weights_only=False)
    model = ProposalApplicabilityDetector(state["variant"], channels=int(state["channels"]))
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval().requires_grad_(False)


def data_loader(dataset, args) -> DataLoader:
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0,
        generator=torch.Generator().manual_seed(args.seed),
    )


def to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def frozen_modules(device: torch.device):
    a2 = load_a2(device); p0, temperature, p0_mode = load_p0(device); p4 = load_p4(device)
    adapter = BiDAFlowInferenceAdapter("sea_raft", device=device)
    for model in (a2, p0, p4, adapter.model):
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("a frozen component is trainable")
    return a2, p0, temperature, p0_mode, p4, adapter


def policy_from_dict(value: dict) -> VetoPolicy:
    fields = VetoPolicy.__dataclass_fields__
    return VetoPolicy(**{key: child for key, child in value.items() if key in fields})


def p4_standalone_authorization(output, inputs) -> torch.Tensor:
    return proposal_authorization_mask(
        output, inputs, utility_margin_px=0.0, uncertainty_threshold_px=2.0,
        require_helpful_class=True,
    )


@torch.no_grad()
def collect_samples(dataset, modules, device, args) -> dict[str, np.ndarray]:
    a2, p0, temperature, p0_mode, p4, adapter = modules
    arrays = defaultdict(list)
    metadata_source = dataset if hasattr(dataset, "backbones") else dataset.base
    backbone_names = list(metadata_source.backbones)
    sequence_names = list(metadata_source.sequences)
    for cpu in data_loader(dataset, args):
        batch = to_device(cpu, device); evidence, _ = build_evidence(adapter, batch)
        inputs, proposal = frozen_proposal_evidence(a2, batch, evidence); p4_output = p4(inputs)
        p0_input, _ = detector_evidence(a2, batch, evidence); p0_output = p0(p0_input)
        raw_auth = authorization_mask(
            p0_output, mode=p0_mode, temperature=temperature,
            aligned_valid=evidence["aligned_validity"], warp_support=evidence["warp_support"],
            proposal_update=proposal.update,
        )
        p4_standalone = p4_standalone_authorization(p4_output, inputs)
        patch_mean = F.avg_pool2d(proposal.update.abs(), 5, stride=1, padding=2)
        raw_error = (batch["raw"] - batch["gt"]).abs()
        proposal_error = (proposal.disparity - batch["gt"]).abs()
        utility = raw_error - proposal_error
        harmful_probability = p4_harmful_probability(p4_output)
        for index in range(batch["raw"].shape[0]):
            common = ((batch["gt_coverage"][index:index+1] > args.coverage_threshold)
                      & batch["raw_valid"][index:index+1].bool()
                      & evidence["aligned_validity"][index:index+1].bool()
                      & evidence["warp_support"][index:index+1].bool())
            indices = common.flatten().nonzero().flatten()
            if indices.numel() > args.sample_pixels_per_frame:
                positions = torch.linspace(0, indices.numel() - 1, args.sample_pixels_per_frame,
                                           device=device).long()
                indices = indices[positions]
            take = lambda value: value[index:index+1].flatten()[indices].float().cpu().numpy()
            values = {
                "utility": utility, "raw_error": raw_error, "proposal_error": proposal_error,
                "update": proposal.update, "patch_mean_update": patch_mean,
                "raw_authorized": raw_auth, "p4_standalone": p4_standalone,
                "predicted_utility": p4_output.utility, "sigma": p4_output.sigma,
                "harmful_probability": harmful_probability,
                "p4_class": p4_output.class_logits.argmax(dim=1, keepdim=True),
            }
            for key, value in values.items(): arrays[key].append(take(value))
            count = indices.numel()
            arrays["backbone_id"].append(np.full(count, backbone_names.index(batch["backbone"][index]), np.int16))
            arrays["sequence_id"].append(np.full(count, sequence_names.index(batch["sequence"][index]), np.int16))
    result = {key: np.concatenate(value) for key, value in arrays.items()}
    result["backbone_names"] = np.asarray(backbone_names)
    result["sequence_names"] = np.asarray(sequence_names)
    return result


def numpy_veto(samples: dict, policy: VetoPolicy) -> np.ndarray:
    veto = np.zeros(samples["utility"].shape, dtype=bool)
    if policy.maximum_update_px is not None: veto |= np.abs(samples["update"]) > policy.maximum_update_px
    if policy.patch_mean_maximum_update_px is not None:
        veto |= samples["patch_mean_update"] > policy.patch_mean_maximum_update_px
    signals = []
    if policy.harmful_probability_threshold is not None:
        signals.append(samples["harmful_probability"] >= policy.harmful_probability_threshold)
    if policy.predicted_utility_ceiling_px is not None:
        signals.append(samples["predicted_utility"] <= policy.predicted_utility_ceiling_px)
    if policy.uncertainty_floor_px is not None:
        signals.append(samples["sigma"] >= policy.uncertainty_floor_px)
    if policy.require_harmful_class: signals.append(samples["p4_class"] == 0)
    if signals:
        p4_veto = signals[0].copy()
        for signal in signals[1:]:
            p4_veto = p4_veto | signal if policy.p4_logic == "any" else p4_veto & signal
        veto |= p4_veto
    return samples["raw_authorized"].astype(bool) & veto


def policy_metrics(samples: dict, policy: VetoPolicy) -> dict:
    raw_auth = samples["raw_authorized"].astype(bool)
    veto = numpy_veto(samples, policy)
    final = raw_auth & ~veto
    nontrivial = np.abs(samples["update"]) > .05
    baseline = raw_auth & nontrivial
    changed = final & nontrivial
    helpful = samples["utility"] > EPSILON; harmful = samples["utility"] < -EPSILON
    material_helpful = samples["utility"] > .02; material_harmful = samples["utility"] < -.02
    clean = samples["raw_error"] <= .5
    c0_gain = float(np.mean(samples["utility"] * raw_auth))
    gain = float(np.mean(samples["utility"] * final))
    oracle = raw_auth & ~harmful
    oracle_gain = float(np.mean(samples["utility"] * oracle))
    conditional_veto = veto & baseline
    return {
        "policy": policy.name, **policy.as_dict(),
        "intervention_coverage": float(changed.mean()),
        "intervention_precision": float((changed & material_helpful).sum() / max(changed.sum(), 1)),
        "false_update_rate": float((changed & clean).sum() / max(clean.sum(), 1)),
        "clean_degradation": float((changed & clean & material_harmful).sum() / max(clean.sum(), 1)),
        "epe_gain": gain, "gain_retained": gain / max(c0_gain, 1e-8),
        "veto_rate_conditional": float(conditional_veto.sum() / max(baseline.sum(), 1)),
        "harmful_veto_recall": float((conditional_veto & harmful).sum() / max((baseline & harmful).sum(), 1)),
        "harmful_veto_precision": float((conditional_veto & harmful).sum() / max(conditional_veto.sum(), 1)),
        "useful_proposal_retention": float((changed & helpful).sum() / max((baseline & helpful).sum(), 1)),
        "conditional_harmful_acceptance": float((changed & harmful).sum() / max((baseline & harmful).sum(), 1)),
        "conditional_helpful_rejection": float((conditional_veto & helpful).sum() / max((baseline & helpful).sum(), 1)),
        "oracle_veto_recovery": (gain - c0_gain) / max(oracle_gain - c0_gain, 1e-8),
        "conditional_count": int(baseline.sum()),
    }


def conditional_audit_rows(samples: dict) -> list[dict]:
    rows = []
    groups = [("ALL", "ALL", np.ones(samples["utility"].shape, bool))]
    for bi, backbone in enumerate(samples["backbone_names"]):
        groups.append((str(backbone), "ALL", samples["backbone_id"] == bi))
        for si, sequence in enumerate(samples["sequence_names"]):
            groups.append((str(backbone), str(sequence),
                           (samples["backbone_id"] == bi) & (samples["sequence_id"] == si)))
    for backbone, sequence, group in groups:
        selected = group & samples["raw_authorized"].astype(bool) & (np.abs(samples["update"]) > .05)
        if not selected.any(): continue
        utility = samples["utility"][selected]
        helpful = utility > EPSILON; harmful = utility < -EPSILON
        def corr(left, right):
            return float(np.corrcoef(left, right)[0, 1]) if left.size > 1 and np.std(left) and np.std(right) else None
        rows.append({
            "backbone": backbone, "sequence": sequence, "conditional_count": int(selected.sum()),
            "helpful_fraction": float(helpful.mean()), "harmful_fraction": float(harmful.mean()),
            "indifferent_fraction": float((~(helpful | harmful)).mean()),
            "utility_mean": float(utility.mean()), "utility_std": float(utility.std()),
            "harmful_probability_mean": float(samples["harmful_probability"][selected].mean()),
            "predicted_utility_mean": float(samples["predicted_utility"][selected].mean()),
            "uncertainty_mean": float(samples["sigma"][selected].mean()),
            "update_magnitude_mean": float(np.abs(samples["update"][selected]).mean()),
            "update_utility_correlation": corr(np.abs(samples["update"][selected]), utility),
        })
    return rows


def risk_gain_coverage_rows(samples: dict, split: str) -> list[dict]:
    """Diagnostic retention curve inside frozen Raw Error authorizations."""
    raw_auth = samples["raw_authorized"].astype(bool)
    nontrivial = np.abs(samples["update"]) > .05
    baseline = raw_auth & nontrivial
    indices = np.flatnonzero(baseline)
    order = indices[np.argsort(-samples["predicted_utility"][indices])]
    c0_gain = float(np.mean(samples["utility"] * raw_auth))
    helpful = samples["utility"] > EPSILON; harmful = samples["utility"] < -EPSILON
    rows = []
    for retained_fraction in (.10, .20, .50, .75, .90, 1.0):
        keep = np.zeros(raw_auth.shape, dtype=bool)
        keep[order[:max(1, int(order.size * retained_fraction))]] = True
        # Tiny updates remain authorized because they do not affect intervention coverage.
        authorization = (raw_auth & ~baseline) | keep
        changed = authorization & nontrivial
        rows.append({
            "split": split, "conditional_retained_fraction": retained_fraction,
            "intervention_coverage": float(changed.mean()),
            "intervention_precision_eps_0_1": float((changed & helpful).sum() / max(changed.sum(), 1)),
            "conditional_harmful_acceptance": float((changed & harmful).sum() / max((baseline & harmful).sum(), 1)),
            "useful_proposal_retention": float((changed & helpful).sum() / max((baseline & helpful).sum(), 1)),
            "epe_gain": float(np.mean(samples["utility"] * authorization)),
            "gain_retained": float(np.mean(samples["utility"] * authorization)) / max(c0_gain, 1e-8),
        })
    return rows


def candidate_policies() -> tuple[list[VetoPolicy], list[VetoPolicy]]:
    magnitude = []
    for threshold in (.10, .25, .50, 1.0, 2.0, 3.0):
        magnitude.append(VetoPolicy(f"C1_pixel_max_{threshold}", maximum_update_px=threshold))
        magnitude.append(VetoPolicy(f"C1_patch_max_{threshold}", patch_mean_maximum_update_px=threshold))
    p4 = [VetoPolicy("C2_harmful_class", require_harmful_class=True)]
    p4 += [VetoPolicy(f"C2_harm_p_{threshold}", harmful_probability_threshold=threshold)
           for threshold in (.10, .25, .50, .75, .90)]
    p4 += [VetoPolicy(f"C2_utility_{threshold}", predicted_utility_ceiling_px=threshold)
           for threshold in (-.25, -.10, 0.0, .05)]
    p4 += [VetoPolicy(f"C2_sigma_{threshold}", uncertainty_floor_px=threshold)
           for threshold in (.05, .10, .25, .50, 1.0)]
    p4 += [VetoPolicy(f"C2_joint_p{probability}_u{utility}",
                     harmful_probability_threshold=probability,
                     predicted_utility_ceiling_px=utility, p4_logic="all")
           for probability in (.25, .50, .75) for utility in (-.10, 0.0, .05)]
    return magnitude, p4


def choose(rows: list[dict], *, safe: bool = False) -> dict:
    eligible = [row for row in rows if row["gain_retained"] >= (.70 if safe else .80)
                and row["false_update_rate"] < (.01 if safe else .0125)
                and row["clean_degradation"] < (.005 if safe else .006)
                and row["intervention_precision"] > .80
                and row["intervention_coverage"] >= .005]
    if eligible: return max(eligible, key=lambda row: (row["epe_gain"], row["intervention_precision"]))
    # Freeze the highest-gain safety Pareto point, but mark it ineligible.
    safe_rows = [row for row in rows if row["false_update_rate"] < (.01 if safe else .0125)
                 and row["clean_degradation"] < (.005 if safe else .006)
                 and row["intervention_coverage"] >= .005]
    selected = max(safe_rows or rows, key=lambda row: (row["epe_gain"], row["intervention_precision"]))
    selected = dict(selected); selected["selection_constraints_met"] = False
    return selected


def select(args) -> None:
    seed_all(args.seed); device = torch.device(args.device); hashes = verify_hashes()
    dataset = ProposalUtilityDataset(
        SEEN_BACKBONES, CALIBRATION_SEQUENCES, coverage_threshold=args.coverage_threshold,
        max_pairs_per_sequence=args.max_pairs, random_clip_start=False, seed=args.seed,
    )
    samples = collect_samples(dataset, frozen_modules(device), device, args)
    audit = conditional_audit_rows(samples); write_csv(args.output / "conditional_error_analysis.csv", audit)
    write_csv(args.output / "validation_risk_gain_coverage.csv", risk_gain_coverage_rows(samples, "validation"))
    c0_policy = VetoPolicy("C0_raw_error")
    magnitude, p4 = candidate_policies()
    c0 = policy_metrics(samples, c0_policy)
    c1_rows = [policy_metrics(samples, policy) for policy in magnitude]
    c2_rows = [policy_metrics(samples, policy) for policy in p4]
    c1 = choose(c1_rows); c2 = choose(c2_rows)
    c1_policy = policy_from_dict(c1); c2_policy = policy_from_dict(c2)
    baseline = samples["raw_authorized"].astype(bool) & (np.abs(samples["update"]) > .05)
    harmful = samples["utility"] < -EPSILON; helpful = samples["utility"] > EPSILON
    v1, v2 = numpy_veto(samples, c1_policy) & baseline, numpy_veto(samples, c2_policy) & baseline
    unique1 = v1 & ~v2
    unique_harmful_fraction = float((unique1 & harmful).sum() / max((baseline & harmful).sum(), 1))
    c3_justified = unique_harmful_fraction >= .05 and (unique1 & harmful).sum() >= (unique1 & helpful).sum()
    c3_policy = VetoPolicy(
        "C3_combined", maximum_update_px=c1_policy.maximum_update_px,
        patch_mean_maximum_update_px=c1_policy.patch_mean_maximum_update_px,
        harmful_probability_threshold=c2_policy.harmful_probability_threshold,
        predicted_utility_ceiling_px=c2_policy.predicted_utility_ceiling_px,
        uncertainty_floor_px=c2_policy.uncertainty_floor_px,
        require_harmful_class=c2_policy.require_harmful_class,
        p4_logic=c2_policy.p4_logic,
    )
    c3 = policy_metrics(samples, c3_policy) if c3_justified else None
    comparison = [c0, c1, c2] + ([c3] if c3 else [])
    balanced = choose(comparison[1:]); safe = choose(comparison[1:], safe=True)
    selected_policy = policy_from_dict(balanced)
    selected_veto = numpy_veto(samples, selected_policy) & baseline
    random_probability = float(selected_veto.sum() / max(baseline.sum(), 1))
    overlap = {
        "selected_c1": c1_policy.as_dict(), "selected_c2": c2_policy.as_dict(),
        "correct_harmful_overlap": int((v1 & v2 & harmful).sum()),
        "c1_unique_correct_harmful": int((unique1 & harmful).sum()),
        "c2_unique_correct_harmful": int(((v2 & ~v1) & harmful).sum()),
        "c1_unique_helpful_rejected": int((unique1 & helpful).sum()),
        "veto_jaccard": float((v1 & v2).sum() / max((v1 | v2).sum(), 1)),
        "c1_unique_harmful_recall": unique_harmful_fraction,
        "c3_justified": c3_justified,
    }
    all_rows = [c0] + c1_rows + c2_rows + ([c3] if c3 else [])
    write_csv(args.output / "threshold_selection.csv", all_rows)
    write_csv(args.output / "policy_comparison_validation.csv", comparison)
    save_json(args.output / "veto_overlap.json", overlap)
    save_json(args.output / "config.json", vars(args))
    save_json(args.output / "frozen_manifest.json", {
        "frozen_hashes": hashes, "selection_sequences": list(CALIBRATION_SEQUENCES),
        "epsilon_px": EPSILON, "c0": c0_policy.as_dict(), "c1": c1_policy.as_dict(),
        "c2": c2_policy.as_dict(), "c3": c3_policy.as_dict() if c3_justified else None,
        "c3_justified": c3_justified, "balanced": selected_policy.as_dict(),
        "safe": policy_from_dict(safe).as_dict(), "balanced_validation_metrics": balanced,
        "safe_validation_metrics": safe, "random_veto_probability": random_probability,
        "standalone_update_threshold_px": json.loads(P4_OPERATING_POINTS.read_text())["matched_baselines"]["update_magnitude_threshold_px"],
        "final_seen_opened": False, "unseen_opened": False,
    })
    print(json.dumps(clean({"audit": audit[0], "comparison": comparison, "overlap": overlap,
                            "balanced": balanced}), indent=2))


@torch.no_grad()
def smoke(args) -> None:
    seed_all(args.seed); device = torch.device(args.device); verify_hashes()
    dataset = ProposalUtilityDataset(["S2M2-S"], ["dataset_3_keyframe_1"],
                                     max_pairs_per_sequence=4, random_clip_start=False)
    modules = frozen_modules(device); a2, p0, temperature, p0_mode, p4, adapter = modules
    cpu = next(iter(data_loader(dataset, args))); batch = to_device(cpu, device)
    evidence, _ = build_evidence(adapter, batch); inputs, proposal = frozen_proposal_evidence(a2, batch, evidence)
    p0_input, _ = detector_evidence(a2, batch, evidence); p0_output = p0(p0_input)
    raw_auth = authorization_mask(p0_output, mode=p0_mode, temperature=temperature,
                                  aligned_valid=evidence["aligned_validity"], warp_support=evidence["warp_support"],
                                  proposal_update=proposal.update)
    p4_output = p4(inputs); policy = VetoPolicy("smoke", harmful_probability_threshold=.5)
    final = cascade_authorization(raw_auth, proposal.update, p4_output, policy)
    prediction = apply_cascade(batch["raw"], proposal.disparity, final)
    passed = (not (final & ~raw_auth).any() and torch.equal(prediction[~final], batch["raw"][~final])
              and torch.equal(prediction[final], proposal.disparity[final])
              and all(parameter.grad is None for model in (a2, p0, p4, adapter.model) for parameter in model.parameters()))
    summary = {"pairs": len(dataset), "raw_authorized": int(raw_auth.sum()), "final_authorized": int(final.sum()),
               "finite": bool(torch.isfinite(prediction).all()), "passed": bool(passed)}
    save_json(args.output / "smoke_summary.json", summary); print(json.dumps(summary, indent=2))
    if not passed: raise RuntimeError("dual-stage smoke failed")


def aggregate_policy_samples(samples: dict, policies: dict[str, VetoPolicy]) -> list[dict]:
    return [policy_metrics(samples, policy) | {"method": name} for name, policy in policies.items()]


@torch.no_grad()
def evaluate_dataset(dataset, modules, manifest: dict, device, args):
    a2, p0, temperature, p0_mode, p4, adapter = modules
    policies = {"C0_raw_error": policy_from_dict(manifest["c0"]),
                "C1_magnitude_veto": policy_from_dict(manifest["c1"]),
                "C2_p4_veto": policy_from_dict(manifest["c2"]),
                "cascade_balanced": policy_from_dict(manifest["balanced"])}
    if manifest.get("c3_justified"): policies["C3_combined_veto"] = policy_from_dict(manifest["c3"])
    rows = []; start = time.perf_counter(); cascade_ms = p4_ms = 0.0
    generator = torch.Generator(device=device).manual_seed(args.seed)
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    for cpu in data_loader(dataset, args):
        batch = to_device(cpu, device); evidence, _ = build_evidence(adapter, batch)
        inputs, proposal = frozen_proposal_evidence(a2, batch, evidence)
        p0_input, _ = detector_evidence(a2, batch, evidence); p0_output = p0(p0_input)
        raw_auth = authorization_mask(p0_output, mode=p0_mode, temperature=temperature,
                                      aligned_valid=evidence["aligned_validity"], warp_support=evidence["warp_support"],
                                      proposal_update=proposal.update)
        if device.type == "cuda": torch.cuda.synchronize(device)
        tick = time.perf_counter(); p4_output = p4(inputs)
        if device.type == "cuda": torch.cuda.synchronize(device)
        p4_ms += (time.perf_counter() - tick) * 1000
        authorizations = {"raw": torch.zeros_like(raw_auth), "a2_unconditional": torch.ones_like(raw_auth),
                          "C0_raw_error": raw_auth,
                          "P4_standalone_prior": p4_standalone_authorization(p4_output, inputs),
                          "update_magnitude_standalone": proposal.update.abs() >= manifest["standalone_update_threshold_px"]}
        for name, policy in policies.items():
            tick = time.perf_counter(); authorizations[name] = cascade_authorization(raw_auth, proposal.update, p4_output, policy)
            cascade_ms += (time.perf_counter() - tick) * 1000
        utility = (batch["raw"] - batch["gt"]).abs() - (proposal.disparity - batch["gt"]).abs()
        authorizations["oracle_conditional_veto"] = raw_auth & ~(utility < -EPSILON)
        random_veto = (torch.rand(raw_auth.shape, generator=generator, device=device)
                       < manifest["random_veto_probability"]) & raw_auth
        authorizations["random_matched_veto"] = raw_auth & ~random_veto
        predictions = {name: apply_cascade(batch["raw"], proposal.disparity, auth)
                       for name, auth in authorizations.items()}
        boundary = boundary_mask_tensor(batch["gt"])
        for threshold in COVERAGE_THRESHOLDS:
            common = ((batch["gt_coverage"] > threshold) & batch["raw_valid"].bool()
                      & evidence["aligned_validity"].bool() & evidence["warp_support"].bool())
            for method, prediction in predictions.items():
                update = torch.where(authorizations[method], proposal.update, torch.zeros_like(proposal.update))
                for index in range(batch["raw"].shape[0]):
                    rows.append({"backbone": batch["backbone"][index], "sequence": batch["sequence"][index],
                                 "frame_id": batch["current_frame_id"][index], "coverage_threshold": threshold,
                                 "method": method,
                                 **map_metrics(prediction[index:index+1], batch["raw"][index:index+1],
                                               batch["gt"][index:index+1], common[index:index+1],
                                               boundary[index:index+1], update[index:index+1])})
    frames = len(dataset)
    wall_seconds = time.perf_counter() - start
    policy_count = max(len(policies), 1)
    sea_parameters = sum(parameter.numel() for parameter in adapter.model.parameters())
    return rows, {"frames": frames, "total_seconds": wall_seconds,
                  "wall_ms_per_frame_including_io_and_metrics": wall_seconds * 1000 / max(frames, 1),
                  "p4_latency_ms_per_frame": p4_ms / max(frames, 1),
                  "cascade_logic_ms_per_frame_all_policies": cascade_ms / max(frames, 1),
                  "cascade_logic_selected_estimated_ms_per_frame": cascade_ms / max(frames * policy_count, 1),
                  "p4_parameters": sum(parameter.numel() for parameter in p4.parameters()),
                  "raw_error_detector_parameters": sum(parameter.numel() for parameter in p0.parameters()),
                  "a2_parameters": sum(parameter.numel() for parameter in a2.parameters()),
                  "sea_raft_parameters": sea_parameters,
                  "proposal_and_authorizers_parameters": sum(parameter.numel() for parameter in p4.parameters())
                    + sum(parameter.numel() for parameter in p0.parameters())
                    + sum(parameter.numel() for parameter in a2.parameters()),
                  "trainable_parameters": 0,
                  "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0}


def evaluate(args, unseen: bool = False) -> None:
    seed_all(args.seed); device = torch.device(args.device); hashes = verify_hashes()
    manifest_path = args.output / "frozen_manifest.json"
    if not manifest_path.exists(): raise RuntimeError("threshold selection must freeze a manifest first")
    manifest = json.loads(manifest_path.read_text())
    if manifest["frozen_hashes"] != hashes: raise RuntimeError("frozen artifacts changed")
    if unseen:
        summary_path = args.output / "aggregate_summary.json"
        if not summary_path.exists() or not json.loads(summary_path.read_text()).get("promotion_passed"):
            raise RuntimeError("unseen backbones are blocked until final seen promotion")
        if not args.backbones or any(name not in {"Fast-FoundationStereo", "CREStereo"} for name in args.backbones):
            raise ValueError("unseen mode accepts only frozen Fast-FoundationStereo/CREStereo")
        if (args.output / "unseen_complete.json").exists(): raise RuntimeError("unseen evaluation already completed")
        dataset = TemporalPairDataset(args.backbones, TEST_SEQUENCES, coverage_threshold=args.coverage_threshold,
                                      max_pairs_per_sequence=args.max_pairs, random_clip_start=False, seed=args.seed)
        prefix = "unseen_"
    else:
        dataset = ProposalUtilityDataset(SEEN_BACKBONES, TEST_SEQUENCES,
                                         coverage_threshold=args.coverage_threshold,
                                         max_pairs_per_sequence=args.max_pairs,
                                         random_clip_start=False, seed=args.seed)
        prefix = ""
    modules = frozen_modules(device)
    rows, runtime = evaluate_dataset(dataset, modules, manifest, device, args)
    sequence, aggregate = aggregate_overall(rows)
    primary_frames = [row for row in rows if float(row["coverage_threshold"]) == .5]
    for method, values in aggregate["overall"].items():
        degradation = np.asarray([float(row["refined_minus_raw_epe"]) for row in primary_frames
                                  if row["method"] == method])
        values["catastrophic_tail_p99"] = float(np.quantile(degradation, .99))
    write_csv(args.output / f"{prefix}frame_metrics.csv", rows)
    write_csv(args.output / f"{prefix}sequence_metrics.csv", sequence)
    write_csv(args.output / f"{prefix}per_backbone.csv", aggregate["per_backbone"])
    save_json(args.output / f"{prefix}runtime_summary.json", runtime)
    # Reuse a compact natural sample solely for conditional decision metrics.
    samples = collect_samples(dataset, modules, device, args)
    policy_comparison = aggregate_policy_samples(samples, {
        "C0_raw_error": policy_from_dict(manifest["c0"]),
        "C1_magnitude_veto": policy_from_dict(manifest["c1"]),
        "C2_p4_veto": policy_from_dict(manifest["c2"]),
        "cascade_balanced": policy_from_dict(manifest["balanced"]),
        **({"C3_combined_veto": policy_from_dict(manifest["c3"])} if manifest.get("c3_justified") else {}),
    })
    write_csv(args.output / f"{prefix}conditional_policy_metrics.csv", policy_comparison)
    write_csv(args.output / f"{prefix}risk_gain_coverage.csv",
              risk_gain_coverage_rows(samples, "unseen" if unseen else "final_seen"))
    if unseen:
        save_json(args.output / "unseen_summary.json", aggregate)
        save_json(args.output / "unseen_complete.json", {"backbones": args.backbones, "completed": True,
                                                          "frozen_hashes": hashes})
        return
    overall = aggregate["overall"]; c0 = overall["C0_raw_error"]; selected = overall["cascade_balanced"]
    per_backbone = [row for row in aggregate["per_backbone"] if row["method"] == "cascade_balanced"]
    gates = {
        "gain_retained_at_least_80pct": selected["gain"] >= .80 * c0["gain"],
        "false_update_below_1_25pct": selected["false_update_rate"] < .0125,
        "clean_degradation_below_0_60pct": selected["clean_pixel_degradation"] < .006,
        "intervention_precision_above_80pct": selected["intervention_precision"] > .80,
        "meaningful_coverage": selected["intervention_coverage"] >= .005,
        "all_seen_backbones_improve": all(row["epe"] < row["raw_epe"] for row in per_backbone),
        "no_catastrophic_frame": selected["worst_frame_degradation"] < .10,
    }
    passed = all(gates.values())
    summary = {"grid": "cache-grid-from-cached-predictions", "coverage_threshold": .5,
               "units": "pixels at width 180", "weighting": "pixel weighted",
               "metrics": overall, "conditional_metrics": policy_comparison,
               "promotion_gates": gates, "promotion_passed": passed, "frozen_hashes": hashes}
    save_json(args.output / "aggregate_summary.json", summary)
    save_json(args.output / "safety_summary.json", {name: {key: values[key] for key in
        ("false_update_rate", "clean_pixel_degradation", "new_bad3", "intervention_coverage",
         "intervention_precision", "frames_worsened_fraction", "worst_frame_degradation",
         "p95_frame_degradation", "catastrophic_tail_p99", "mean_update_magnitude_clean")}
        for name, values in overall.items()})
    manifest["final_seen_opened"] = True; manifest["promotion_passed"] = passed
    save_json(manifest_path, manifest)
    print(json.dumps(clean(summary), indent=2))


def main() -> int:
    args = parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "smoke": smoke(args)
    elif args.mode == "select": select(args)
    elif args.mode == "evaluate": evaluate(args)
    else: evaluate(args, unseen=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
