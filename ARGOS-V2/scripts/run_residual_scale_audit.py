#!/usr/bin/env python3
"""Frozen A2 residual-scale audit for ARGOS v2.

This is deliberately *not* a training path.  It evaluates
``d_raw + lambda * (d_A2 - d_raw)`` over a compact predeclared lambda grid,
selects a single scale using only the held-out SCARED-C calibration sequences,
and can then evaluate that immutable scale on the final seen split.  It exists
to distinguish an unsafe A2 proposal direction from merely over-large updates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

V2_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(V2_ROOT), str(V2_ROOT / "scripts")]

from model_design.data.proposal_utility_dataset import ProposalUtilityDataset
from model_design.data.raw_error_dataset import CALIBRATION_SEQUENCES, TEST_SEQUENCES
from model_design.external_components.bidavideo import BiDAFlowInferenceAdapter, SEA_RAFT_CHECKPOINT
from model_design.models.abstention import OperatingMode, authorization_mask
from model_design.models.raw_error_detector import RawErrorDetector, RawErrorEvidence
from run_learned_t1_refiner import build_evidence
from run_raw_error_abstention import A2_CHECKPOINT, boundary_mask_tensor, load_a2, map_metrics


SCALES = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00)
SEEN_BACKBONES = ("S2M2-S", "RAFT-Stereo", "StereoAnywhere")
EXPECTED_A2_SHA256 = "6cd29277397001333ef3ce630b2f3bc04ec393cdc72e65aa5eb087afd3b389ea"
EXPECTED_SEA_RAFT_SHA256 = "1a21575ed6ca2c6945fb8e25c4169d241cf59ee5d12b8802c01c965206268cac"
DETECTOR_CHECKPOINT = V2_ROOT / "results/raw_error_abstention/full/checkpoints/best_validation.pt"
DETECTOR_MODES = V2_ROOT / "results/raw_error_abstention/full/operating_modes.json"
EXPECTED_DETECTOR_SHA256 = "78b1bb6cf809dc76448222e41e3bcfafb754bc9b7b6629edcdfa2e1a33444e67"
EXPECTED_MODES_SHA256 = "791f27d21e3f9fa63fe267d5742c4fb85226f49e6027b285aeb90754fbe10b69"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("calibrate", "evaluate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-pairs-per-sequence", type=int, default=0,
                        help="Smoke/debug only; zero uses every eligible pair.")
    parser.add_argument("--coverage-threshold", type=float, default=0.50)
    parser.add_argument("--policy", choices=("unconditional", "raw_error"), default="unconditional",
                        help="Frozen update policy; raw_error is the validated balanced authorizer.")
    parser.add_argument("--seed", type=int, default=20260719)
    return parser.parse_args()


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        keys.extend(key for key in row if key not in keys)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def loader(dataset, args: argparse.Namespace) -> DataLoader:
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0,
        generator=torch.Generator().manual_seed(args.seed),
    )


def to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def frozen_artifacts(policy: str) -> dict:
    artifacts = {
        "a2": {"path": str(A2_CHECKPOINT), "sha256": sha256(A2_CHECKPOINT)},
        "sea_raft": {"path": str(SEA_RAFT_CHECKPOINT), "sha256": sha256(SEA_RAFT_CHECKPOINT)},
    }
    if policy == "raw_error":
        artifacts |= {
            "raw_error_detector": {"path": str(DETECTOR_CHECKPOINT), "sha256": sha256(DETECTOR_CHECKPOINT)},
            "operating_modes": {"path": str(DETECTOR_MODES), "sha256": sha256(DETECTOR_MODES)},
        }
    return artifacts


def load_raw_error_policy(device: torch.device) -> tuple[RawErrorDetector, float, OperatingMode]:
    if sha256(DETECTOR_CHECKPOINT) != EXPECTED_DETECTOR_SHA256 or sha256(DETECTOR_MODES) != EXPECTED_MODES_SHA256:
        raise RuntimeError("raw-error authorizer differs from the frozen validated artifact")
    payload = torch.load(DETECTOR_CHECKPOINT, map_location="cpu", weights_only=False)
    detector = RawErrorDetector(payload["architecture"], channels=int(payload["channels"]))
    detector.load_state_dict(payload["model"], strict=True)
    detector.to(device).eval().requires_grad_(False)
    modes = json.loads(DETECTOR_MODES.read_text())
    return detector, float(modes["temperature"]), OperatingMode(**modes["modes"]["balanced"])


def aggregate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Pixel-weight geometry plus sequence-unit diagnostics; no pixel pseudo-N."""
    metric_keys = (
        "epe", "raw_epe", "bad1", "bad3", "boundary_epe", "new_bad3",
        "false_update_rate", "clean_pixel_degradation", "intervention_coverage",
        "intervention_precision", "mean_update_magnitude_clean",
    )
    seq_groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        seq_groups[(row["scale"], row["backbone"], row["sequence"])].append(row)
    sequences: list[dict] = []
    for (scale, backbone, sequence), group in sorted(seq_groups.items()):
        valid = [row for row in group if row["valid_count"] > 0 and math.isfinite(row["epe"])]
        if not valid:
            continue
        valid_count = sum(row["valid_count"] for row in valid)
        clean_count = sum(row["clean_count"] for row in valid)
        changed_count = sum(row["changed_count"] for row in valid)
        helpful_count = sum(row["helpful_count"] for row in valid)
        false_count = sum(row["false_update_count"] for row in valid)
        degraded_count = sum(row["clean_degradation_count"] for row in valid)
        def mean(key: str) -> float:
            weight_key = "clean_count" if key in {"false_update_rate", "clean_pixel_degradation", "mean_update_magnitude_clean"} else "valid_count"
            finite = [row for row in valid if math.isfinite(row[key])]
            denominator = sum(row[weight_key] for row in finite)
            return sum(row[key] * row[weight_key] for row in finite) / max(denominator, 1) if finite else math.nan
        sequences.append({
            "scale": scale, "backbone": backbone, "sequence": sequence,
            "frames": len(valid), "valid_count": valid_count, "clean_count": clean_count,
            "changed_count": changed_count, "helpful_count": helpful_count,
            "false_update_count": false_count, "clean_degradation_count": degraded_count,
            **{key: mean(key) for key in metric_keys},
            "intervention_coverage": changed_count / max(valid_count, 1),
            "intervention_precision": helpful_count / max(changed_count, 1),
            "false_update_rate": false_count / max(clean_count, 1),
            "clean_pixel_degradation": degraded_count / max(clean_count, 1),
            "gain": mean("raw_epe") - mean("epe"),
        })
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in sequences:
        grouped[(row["scale"], row["backbone"])].append(row)
    backbone: list[dict] = []
    for (scale, name), group in sorted(grouped.items()):
        count = sum(row["valid_count"] for row in group)
        def weighted(key: str) -> float:
            weight_key = "clean_count" if key in {"false_update_rate", "clean_pixel_degradation", "mean_update_magnitude_clean"} else "valid_count"
            denominator = sum(row[weight_key] for row in group)
            return sum(row[key] * row[weight_key] for row in group) / max(denominator, 1)
        backbone.append({
            "scale": scale, "backbone": name, "valid_count": count,
            **{key: weighted(key) for key in metric_keys},
            "gain": weighted("raw_epe") - weighted("epe"),
        })
    return sequences, backbone


@torch.inference_mode()
def collect(args: argparse.Namespace, sequences: tuple[str, ...], *, scales: tuple[float, ...]) -> list[dict]:
    device = torch.device(args.device)
    if sha256(A2_CHECKPOINT) != EXPECTED_A2_SHA256:
        raise RuntimeError("A2 checkpoint hash differs from the frozen validated artifact")
    if sha256(SEA_RAFT_CHECKPOINT) != EXPECTED_SEA_RAFT_SHA256:
        raise RuntimeError("SEA-RAFT checkpoint hash differs from the frozen validated artifact")
    dataset = ProposalUtilityDataset(
        SEEN_BACKBONES, sequences, coverage_threshold=args.coverage_threshold,
        max_pairs_per_sequence=args.max_pairs_per_sequence or None,
        random_clip_start=False, seed=args.seed,
    )
    a2 = load_a2(device)
    flow = BiDAFlowInferenceAdapter("sea_raft", device=device)
    detector, temperature, operating_mode = (load_raw_error_policy(device) if args.policy == "raw_error"
                                               else (None, None, None))
    if any(parameter.requires_grad for parameter in a2.parameters()) or any(parameter.requires_grad for parameter in flow.model.parameters()):
        raise RuntimeError("frozen model unexpectedly requires gradients")
    rows: list[dict] = []
    for cpu_batch in loader(dataset, args):
        batch = to_device(cpu_batch, device)
        evidence, _ = build_evidence(flow, batch)
        proposal = a2(batch["raw"], evidence, batch["current_rgb"])
        if detector is None:
            authorization = torch.ones_like(proposal.update, dtype=torch.bool)
        else:
            detector_input = RawErrorEvidence(
                raw=batch["raw"], raw_valid=batch["raw_valid"],
                aligned=evidence["aligned_past_disparity"], aligned_valid=evidence["aligned_validity"],
                warp_support=evidence["warp_support"], forward_backward_error=evidence["forward_backward_error"],
                forward_backward_confidence=evidence["forward_backward_confidence"],
                photometric_residual=evidence["photometric_residual"], flow_magnitude=evidence["flow_magnitude"],
                a2_update=proposal.update, a2_error_gate=proposal.g_error, a2_memory_gate=proposal.c_memory,
            )
            detector_output = detector(detector_input)
            authorization = authorization_mask(
                detector_output, mode=operating_mode, temperature=temperature,
                aligned_valid=evidence["aligned_validity"], warp_support=evidence["warp_support"],
                proposal_update=proposal.update,
            )
        common = (
            (batch["gt_coverage"] > args.coverage_threshold)
            & batch["raw_valid"].bool()
            & evidence["aligned_validity"].bool()
            & evidence["warp_support"].bool()
        )
        boundary = boundary_mask_tensor(batch["gt"])
        for scale in scales:
            prediction = batch["raw"] + scale * torch.where(authorization, proposal.update, torch.zeros_like(proposal.update))
            update = prediction - batch["raw"]
            for index in range(prediction.shape[0]):
                metric = map_metrics(
                    prediction[index:index + 1], batch["raw"][index:index + 1],
                    batch["gt"][index:index + 1], common[index:index + 1],
                    boundary[index:index + 1], update[index:index + 1],
                )
                rows.append({
                    "scale": scale, "backbone": batch["backbone"][index],
                    "sequence": batch["sequence"][index], "frame_id": batch["current_frame_id"][index],
                    **metric,
                })
    return rows


def choose_scale(backbone_rows: list[dict]) -> dict:
    """Frozen SCARED-C-only robust selection; tie break toward smaller updates."""
    candidates: list[dict] = []
    for scale in SCALES:
        group = [row for row in backbone_rows if row["scale"] == scale]
        complete = {row["backbone"] for row in group} == set(SEEN_BACKBONES)
        eligible = (
            complete
            and scale > 0
            and all(row["gain"] > 0 for row in group)
            and all(row["clean_pixel_degradation"] <= .03 for row in group)
            and all(row["false_update_rate"] <= .05 for row in group)
        )
        candidates.append({
            "scale": scale,
            "minimum_backbone_gain": min((row["gain"] for row in group), default=-math.inf),
            "mean_backbone_gain": float(np.mean([row["gain"] for row in group])) if group else math.nan,
            "maximum_clean_degradation": max((row["clean_pixel_degradation"] for row in group), default=math.inf),
            "maximum_false_update_rate": max((row["false_update_rate"] for row in group), default=math.inf),
            "all_seen_backbones_present": complete,
            "eligible": eligible,
        })
    feasible = [row for row in candidates if row["eligible"]]
    selected = max(feasible or candidates, key=lambda row: (row["minimum_backbone_gain"], row["mean_backbone_gain"], -row["scale"]))
    return {"selection_rule": "maximize worst seen-backbone EPE gain subject to all-backbone clean degradation <=3% and false update <=5%; tie-break smaller scale",
            "candidates": candidates, "selected": selected, "constraints_feasible": bool(feasible)}


def calibrate(args: argparse.Namespace) -> None:
    rows = collect(args, CALIBRATION_SEQUENCES, scales=SCALES)
    sequence, backbone = aggregate(rows)
    choice = choose_scale(backbone)
    write_csv(args.output / "calibration_frame_metrics.csv", rows)
    write_csv(args.output / "calibration_sequence_metrics.csv", sequence)
    write_csv(args.output / "calibration_backbone_metrics.csv", backbone)
    save_json(args.output / "frozen_scale.json", {
        "selected_only_on": list(CALIBRATION_SEQUENCES), "coverage_threshold": args.coverage_threshold,
        "policy": args.policy,
        "scales": list(SCALES), "selection": choice,
        "frozen_artifacts": frozen_artifacts(args.policy),
        "test_opened": False,
    })


def evaluate(args: argparse.Namespace) -> None:
    manifest_path = args.output / "frozen_scale.json"
    if not manifest_path.exists():
        raise RuntimeError("calibrate before evaluating the final seen split")
    manifest = json.loads(manifest_path.read_text())
    if manifest["test_opened"]:
        raise RuntimeError("final seen scale evaluation already exists; preserve frozen protocol")
    if manifest["policy"] != args.policy or manifest["frozen_artifacts"] != frozen_artifacts(args.policy):
        raise RuntimeError("frozen policy artifacts changed after scale selection")
    scale = float(manifest["selection"]["selected"]["scale"])
    rows = collect(args, TEST_SEQUENCES, scales=(0.0, scale, 1.0))
    sequence, backbone = aggregate(rows)
    write_csv(args.output / "final_seen_frame_metrics.csv", rows)
    write_csv(args.output / "final_seen_sequence_metrics.csv", sequence)
    write_csv(args.output / "final_seen_backbone_metrics.csv", backbone)
    selected = [row for row in backbone if row["scale"] == scale]
    gate = scale > 0 and all(row["gain"] > 0 for row in selected)
    manifest["test_opened"] = True
    manifest["final_seen_selected_scale"] = scale
    manifest["final_seen_backbone_metrics"] = selected
    manifest["final_seen_all_backbones_improve"] = gate
    save_json(manifest_path, manifest)


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "calibrate":
        calibrate(args)
    else:
        evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
