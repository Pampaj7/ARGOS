#!/usr/bin/env python3
"""ARGOS v2 validation-only bounded-memory and hard-endpoint evaluation.

The frozen CODD-style head is unchanged.  This runner only re-anchors its
causal fused state to the raw previous-frame disparity according to a compact,
deterministic policy, or replaces its soft output by an exact endpoint choice.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_codd_style_fusion_probe import TEST, VALIDATION, seed_all, to_device  # noqa: E402
from run_codd_style_fusion_mechanism_audit import (  # noqa: E402
    aggregate, common_support, frame_metrics, grouped, metadata, save_json, sha256, write_csv,
)
from model_design.data.temporal_pair_dataset import SEEN_BACKBONES, TemporalPairDataset  # noqa: E402
from model_design.external_components.bidavideo import (  # noqa: E402
    BiDAFlowInferenceAdapter, causal_warp, temporal_disparity_evidence,
)
from model_design.models.codd_bounded_memory import (  # noqa: E402
    BoundedMemoryPolicy, ResetEvidence, advance_state_age,
)
from model_design.models.codd_style_fusion import (  # noqa: E402
    CODDFusionOutput, CODDStyleFusionHead, FrozenResNet18Layer1,
    build_codd_cues, hard_endpoint_fusion,
)


CANONICAL_CHECKPOINT = ROOT / "results/codd_style_fusion_probe/bida_memory_phase1/full_phase1/seed_0/checkpoints/best_validation.pt"
RESULT_ROOT = ROOT / "results/codd_style_bounded_memory_validation"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("evaluate", "derive", "select"), required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path, default=RESULT_ROOT / "reset_policy/fixed_horizon/h4")
    parser.add_argument("--checkpoint", type=Path, default=CANONICAL_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--preload-workers", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-name", default="fixed_h4")
    parser.add_argument("--max-age", type=int)
    parser.add_argument("--accumulated-update-max", type=float)
    parser.add_argument("--disagreement-max", type=float)
    parser.add_argument("--warp-support-min", type=float)
    parser.add_argument("--fb-confidence-min", type=float)
    parser.add_argument("--temporal-activation-max", type=float)
    parser.add_argument("--update-magnitude-max", type=float)
    parser.add_argument("--hard-threshold", type=float)
    parser.add_argument("--memory-state", choices=("recurrent", "raw_previous"), default="recurrent")
    parser.add_argument("--disable-learned-stereo-evidence", action="store_true")
    # Optional frozen-evaluation overrides.  Defaults preserve the validated
    # seen-backbone protocol; these only choose existing cache records.
    parser.add_argument("--backbones", nargs="+")
    parser.add_argument("--sequences", nargs="+")
    parser.add_argument("--frozen-policy", type=Path)
    parser.add_argument("--selection-root", type=Path, default=RESULT_ROOT / "reset_policy/validation_candidates")
    parser.add_argument("--tiny", action="store_true")
    return parser.parse_args()


def policy_from_args(config: argparse.Namespace) -> tuple[BoundedMemoryPolicy, float | None]:
    if config.frozen_policy is not None:
        record = json.loads(config.frozen_policy.read_text())
        if record.get("selection_split") != "dataset_2_validation":
            raise RuntimeError("frozen policy was not selected exclusively on dataset 2")
        return BoundedMemoryPolicy.from_dict(record["policy"]), record.get("hard_threshold")
    if config.split == "test":
        raise RuntimeError("dataset 7 requires --frozen-policy selected on dataset 2")
    return BoundedMemoryPolicy(
        name=config.policy_name,
        max_age=config.max_age,
        accumulated_update_max=config.accumulated_update_max,
        disagreement_max=config.disagreement_max,
        warp_support_min=config.warp_support_min,
        fb_confidence_min=config.fb_confidence_min,
        temporal_activation_max=config.temporal_activation_max,
        update_magnitude_max=config.update_magnitude_max,
    ), config.hard_threshold


def initial_state(item: dict) -> dict:
    return {"disparity": item["past"], "valid": item["past_valid"].bool(),
            "gt": item["past_gt"], "gt_coverage": item["past_gt_coverage"]}


def decision_stats(item: dict, evidence, output: CODDFusionOutput, age: int, accumulated: float) -> ResetEvidence:
    mask = item["raw_valid"].bool() & evidence.aligned_validity.bool() & evidence.warp_support.bool()
    if not bool(mask.any()):
        return ResetEvidence(age, accumulated, float("inf"), 0.0, 0.0, 1.0, float("inf"))
    denominator = item["raw_valid"].sum().clamp_min(1)
    return ResetEvidence(
        age=age,
        accumulated_update=accumulated,
        disagreement=float((evidence.aligned_past_disparity - item["raw"]).abs()[mask].mean()),
        warp_support=float((evidence.warp_support & evidence.aligned_validity).sum() / denominator),
        fb_confidence=float(evidence.forward_backward_confidence[mask].mean()),
        temporal_activation=float(output.temporal_weight[mask].mean()),
        update_magnitude=float((output.fused_disparity - item["raw"]).abs()[mask].mean()),
    )


def infer(model, extractor, item, state, forward, backward, *, include_learned: bool):
    evidence = temporal_disparity_evidence(
        item["raw"], state["disparity"], forward, backward,
        current_valid=item["raw_valid"], past_valid=state["valid"],
        current_rgb=item["current_rgb"], past_rgb=item["past_rgb"],
    )
    cues = build_codd_cues(
        extractor, raw=item["raw"], aligned_memory=evidence.aligned_past_disparity,
        current_rgb=item["current_rgb"], current_right_rgb=item["current_right_rgb"],
        past_rgb=item["past_rgb"], flow_current_to_past=forward,
        flow_magnitude=evidence.flow_magnitude,
        forward_backward_confidence=evidence.forward_backward_confidence,
        warp_support=evidence.warp_support, aligned_valid=evidence.aligned_validity,
        include_learned_stereo_evidence=include_learned,
    )
    return evidence, model(cues, item["raw"], evidence.aligned_past_disparity)


@torch.no_grad()
def evaluate(config: argparse.Namespace) -> None:
    policy, hard_threshold = policy_from_args(config)
    seed_all(config.seed)
    device = torch.device(config.device)
    checkpoint = torch.load(config.checkpoint, map_location="cpu", weights_only=False)
    model = CODDStyleFusionHead(checkpoint["cue_channels"]).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    include_learned = not config.disable_learned_stereo_evidence
    extractor = FrozenResNet18Layer1().to(device) if include_learned else None
    adapter = BiDAFlowInferenceAdapter("sea_raft", device=device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert extractor is None or not any(parameter.requires_grad for parameter in extractor.parameters())
    assert not any(parameter.requires_grad for parameter in adapter.model.parameters())

    sequences = tuple(config.sequences) if config.sequences else (VALIDATION if config.split == "validation" else TEST)
    backbones = tuple(config.backbones) if config.backbones else SEEN_BACKBONES
    max_pairs = None
    if config.tiny:
        sequences = (sequences[0],); backbones = (SEEN_BACKBONES[0],); max_pairs = 4
    dataset = TemporalPairDataset(backbones, sequences, coverage_threshold=.50,
        include_right_rgb=True, max_pairs_per_sequence=max_pairs)
    dataset.preload_frame_data(config.preload_workers)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.workers,
        persistent_workers=config.workers > 0, pin_memory=True,
        prefetch_factor=4 if config.workers else None)

    rows: list[dict] = []
    state = None; prior = None; age = 0; accumulated = 0.0
    for cpu in loader:
        item = to_device(cpu, device)
        sequence, backbone, _ = metadata(item)
        key = (sequence, backbone)
        sequence_reset = key != prior
        if sequence_reset:
            state = None; age = 0; accumulated = 0.0; prior = key
        forced_raw_previous = config.memory_state == "raw_previous"
        pre_reset = forced_raw_previous or state is None or policy.pre_reset(age=age, accumulated_update=accumulated)
        if pre_reset:
            state = initial_state(item); age = 0; accumulated = 0.0

        forward = adapter.current_to_past(item["current_rgb"], item["past_rgb"])
        backward = adapter.past_to_current(item["past_rgb"], item["current_rgb"])
        historical_state = initial_state(item)
        historical = temporal_disparity_evidence(
            item["raw"], item["past"], forward, backward,
            current_valid=item["raw_valid"], past_valid=item["past_valid"],
            current_rgb=item["current_rgb"], past_rgb=item["past_rgb"],
        )
        recurrent, output = infer(model, extractor, item, state, forward, backward, include_learned=include_learned)
        provisional = decision_stats(item, recurrent, output, age, accumulated)
        evidence_reset = (not pre_reset) and policy.evidence_reset(provisional)
        if evidence_reset:
            state = historical_state; age = 0; accumulated = 0.0
            recurrent, output = infer(model, extractor, item, state, forward, backward, include_learned=include_learned)
        final_stats = decision_stats(item, recurrent, output, age, accumulated)

        if hard_threshold is not None:
            hard, accepted = hard_endpoint_fusion(item["raw"], recurrent.aligned_past_disparity,
                output.temporal_weight, hard_threshold)
            output = replace(output, temporal_weight=accepted.to(output.temporal_weight.dtype), fused_disparity=hard)

        common, historical_mask, recurrent_mask = common_support(item, historical, recurrent)
        reset_now = pre_reset or evidence_reset
        next_age = advance_state_age(age, reset=reset_now)
        decision_mask = item["raw_valid"].bool() & recurrent.aligned_validity.bool() & recurrent.warp_support.bool()
        update = float((output.fused_disparity - item["raw"]).abs()[decision_mask].mean()) if bool(decision_mask.any()) else 0.0
        next_accumulated = update if reset_now else accumulated + update
        next_state = {"disparity": output.fused_disparity, "valid": item["raw_valid"].bool(),
                      "gt": item["gt"], "gt_coverage": item["gt_coverage"]}

        if bool(common.any()):
            gt_warp = causal_warp(state["gt"], forward, source_valid=state["gt_coverage"] > .50)
            row, _ = frame_metrics(
                item=item, historical_memory=historical.aligned_past_disparity,
                recurrent_memory=recurrent.aligned_past_disparity, output=output,
                common=common, historical_mask=historical_mask, recurrent_mask=recurrent_mask,
                mode=policy.name + ("_hard" if hard_threshold is not None else "_soft"),
                step_since_reset=next_age, gt_aligned=gt_warp.warped,
                gt_aligned_valid=gt_warp.valid,
            )
            row.update({
                "reset": int(reset_now), "pre_reset": int(pre_reset),
                "evidence_reset": int(evidence_reset), "state_age_before": age,
                "accumulated_update_before": accumulated,
                "disagreement_mean": final_stats.disagreement,
                "warp_support_fraction": final_stats.warp_support,
                "fb_confidence_mean": final_stats.fb_confidence,
                "decision_temporal_activation": final_stats.temporal_activation,
                "decision_update_magnitude": final_stats.update_magnitude,
            })
            rows.append(row)
        state = next_state; age = next_age; accumulated = next_accumulated

    summary = aggregate(rows)
    summary["reset_rate"] = float(np.mean([row["reset"] for row in rows]))
    summary["mean_state_age"] = float(np.mean([row["step_since_reset"] for row in rows]))
    summary["policy"] = policy.to_dict(); summary["hard_threshold"] = hard_threshold
    summary["split"] = config.split
    config.output.mkdir(parents=True, exist_ok=True)
    write_csv(config.output / "frame_metrics.csv", rows)
    write_csv(config.output / "per_backbone_metrics.csv", grouped(rows, "backbone"))
    write_csv(config.output / "per_sequence_metrics.csv", grouped(rows, "sequence"))
    age_groups = {}
    for row in rows: age_groups.setdefault(row["step_since_reset"], []).append(row)
    write_csv(config.output / "drift_by_age.csv", [{"age": age, **aggregate(group)} for age, group in sorted(age_groups.items())])
    save_json(config.output / "summary.json", summary)
    save_json(config.output / "protocol_audit.json", {
        "project": "ARGOS v2", "split": config.split,
        "sequences": list(sequences), "dataset_7_opened": config.split == "test",
        "selection_source": str(config.frozen_policy) if config.frozen_policy else "dataset 2 candidate evaluation",
        "checkpoint": str(config.checkpoint), "checkpoint_sha256": sha256(config.checkpoint),
        "no_future_access": True, "state_resets_before_final recurrent candidate": True,
        "common_support": "GT coverage>0.5 & raw valid & historical and recurrent aligned-valid/warp support",
    })


def select(config: argparse.Namespace) -> None:
    candidates = []
    for path in sorted(config.selection_root.glob("*/summary.json")):
        value = json.loads(path.read_text())
        if value.get("split") != "validation":
            continue
        candidates.append({"path": str(path.parent), **value})
    if not candidates:
        raise RuntimeError(f"no validation summaries under {config.selection_root}")
    safe = [row for row in candidates if row["fused_gain"] > 0 and row["worst_frame_degradation"] < 1.0]
    selected = max(safe or candidates, key=lambda row: row["fused_gain"])
    record = {
        "project": "ARGOS v2", "selection_split": "dataset_2_validation",
        "objective": "maximum fused EPE gain subject to positive gain and worst-frame degradation <1 EPE",
        "policy": selected["policy"], "hard_threshold": selected.get("hard_threshold"),
        "selected_validation_summary": selected, "candidate_count": len(candidates),
    }
    save_json(config.output, record)


def derive(config: argparse.Namespace) -> None:
    """Derive the compact adaptive threshold grid from dataset-2 H=4 only."""
    source = config.selection_root / "fixed_h4/frame_metrics.csv"
    if not source.exists():
        raise RuntimeError(f"dataset-2 H=4 metrics missing: {source}")
    import csv
    with source.open() as handle:
        rows = list(csv.DictReader(handle))
    def quantile(field: str, q: float) -> float:
        values = np.asarray([float(row[field]) for row in rows], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise RuntimeError(f"no finite dataset-2 values for {field}")
        return float(np.quantile(values, q))
    thresholds = {
        "source": str(source), "selection_split": "dataset_2_validation",
        "accumulated_update": {str(q): quantile("accumulated_update_before", q) for q in (.50, .75, .90)},
        "conservative_evidence": {
            "disagreement_max": quantile("disagreement_mean", .99),
            "warp_support_min": quantile("warp_support_fraction", .01),
            "fb_confidence_min": quantile("fb_confidence_mean", .01),
            "temporal_activation_max": quantile("decision_temporal_activation", .99),
            "update_magnitude_max": quantile("decision_update_magnitude", .99),
        },
    }
    save_json(config.output, thresholds)


def main() -> None:
    config = arguments()
    if config.mode == "select":
        select(config)
    elif config.mode == "derive":
        derive(config)
    else:
        evaluate(config)


if __name__ == "__main__":
    main()
