#!/usr/bin/env python3
"""Tiny D2-only ARGOS v2 parity smoke; no training and no dataset 7."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
V2 = Path("/dtu/p1/leopam/ARGOS/ARGOS-V2")
sys.path[:0] = [str(ROOT / "src"), str(V2), str(V2 / "scripts")]

from argos_freezed.alignment.bida_pull_warp import temporal_disparity_evidence
from argos_freezed.alignment.sea_raft_adapter import SEARAFTFlowAdapter
from argos_freezed.constants import ANCHOR_AGES, SEA_RAFT_CHECKPOINT, SEA_RAFT_SHA256
from argos_freezed.memory_bank import RawAnchorBank
from argos_freezed.models.raw_multi_anchor_refiner import MultiAnchorEvidence, RawMultiAnchorRefiner, retrieve_and_fuse
from argos_freezed.pipeline import FrozenArgosGeometryRefiner
from argos_v2.cache_io import load_sequence_cache
from argos_v2.scared_c_data import load_frame_lr, load_sequence_info
from model_design.external_components.bidavideo import temporal_disparity_evidence as original_alignment
from model_design.models.raw_multi_anchor_refiner import MultiAnchorEvidence as OriginalEvidence
from model_design.models.raw_multi_anchor_refiner import RawMultiAnchorRefiner as OriginalModel
from model_design.models.raw_multi_anchor_refiner import retrieve_and_fuse as original_retrieve

SEQUENCE = "dataset_2_keyframe_2"
BACKBONE = "S2M2-S"
CURRENT = 8
TMP = ROOT / "experiments/00_template/results/smoke_tmp"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_tensor(info, frame_id: str, side: int, device: torch.device) -> torch.Tensor:
    pair = load_frame_lr(info, frame_id)
    resized = cv2.resize(pair[side], (180, 144), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(np.ascontiguousarray(resized)).permute(2, 0, 1).float().to(device)


class RecordedDirectFlows:
    def __init__(self, forward: torch.Tensor, backward: torch.Tensor):
        self.forward, self.backward, self.position = forward, backward, 0

    def current_to_anchor(self, current, anchor):
        return self.forward[self.position:self.position + 1]

    def anchor_to_current(self, anchor, current):
        value = self.backward[self.position:self.position + 1]; self.position += 1; return value


def maximum_difference(first: torch.Tensor, second: torch.Tensor) -> float:
    if torch.equal(first, second):
        return 0.0
    if first.dtype == torch.bool or not first.is_floating_point():
        return float("inf")
    return float((first - second).abs().max())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(); device = torch.device(args.device)
    subprocess.run([sys.executable, str(ROOT / "scripts/verify_freeze.py")], check=True)
    if sha256(SEA_RAFT_CHECKPOINT) != SEA_RAFT_SHA256:
        raise SystemExit("SEA-RAFT provenance mismatch")
    info = load_sequence_info(SEQUENCE)
    raw_cache, valid_cache, cache_ids, _ = load_sequence_cache(BACKBONE, SEQUENCE)
    if [str(value) for value in cache_ids[:CURRENT + 1]] != info.frame_ids[:CURRENT + 1]:
        raise SystemExit("D2 frame order mismatch")
    anchor_indices = [CURRENT - age for age in ANCHOR_AGES]
    needed = sorted(set(anchor_indices + [CURRENT]))
    left = {index: image_tensor(info, info.frame_ids[index], 0, device) for index in needed}
    right = image_tensor(info, info.frame_ids[CURRENT], 1, device)
    current_left = left[CURRENT][None]
    anchors_rgb = torch.stack([left[index] for index in anchor_indices])
    current_batch = current_left.expand(len(ANCHOR_AGES), -1, -1, -1)
    sea = SEARAFTFlowAdapter(device=device)
    forward = sea.infer(current_batch, anchors_rgb)
    backward = sea.infer(anchors_rgb, current_batch)
    del sea
    raw = torch.from_numpy(np.asarray(raw_cache[CURRENT], dtype=np.float32).copy())[None, None].to(device)
    valid = torch.from_numpy((np.asarray(valid_cache[CURRENT]) > 0).copy())[None, None].to(device)
    bank = RawAnchorBank()
    for index in sorted(anchor_indices):
        disparity = torch.from_numpy(np.asarray(raw_cache[index], dtype=np.float32).copy())[None, None].to(device)
        validity = torch.from_numpy((np.asarray(valid_cache[index]) > 0).copy())[None, None].to(device)
        bank.append_raw(disparity, validity, left[index][None], frame_id=info.frame_ids[index], frame_index=index)
    flow = RecordedDirectFlows(forward, backward)
    refiner = FrozenArgosGeometryRefiner(device=device, flow_adapter=flow)
    weights_before = {key: value.clone() for key, value in refiner.model.state_dict().items()}
    result = refiner.step(current_left, right[None], raw, valid, bank, frame_id=info.frame_ids[CURRENT], frame_index=CURRENT)
    if flow.position != 4 or not all(torch.equal(value, weights_before[key]) for key, value in refiner.model.state_dict().items()):
        raise SystemExit("flow count or frozen-weight contract failed")

    candidates, candidate_valid, support, confidence = [], [], [], []
    for position, index in enumerate(anchor_indices):
        anchor_raw = torch.from_numpy(np.asarray(raw_cache[index], dtype=np.float32).copy())[None, None].to(device)
        anchor_valid = torch.from_numpy((np.asarray(valid_cache[index]) > 0).copy())[None, None].to(device)
        reference = original_alignment(raw, anchor_raw, forward[position:position + 1], backward[position:position + 1],
            current_valid=valid, past_valid=anchor_valid, current_rgb=current_left, past_rgb=left[index][None])
        frozen = temporal_disparity_evidence(raw, anchor_raw, forward[position:position + 1], backward[position:position + 1],
            current_valid=valid, past_valid=anchor_valid, current_rgb=current_left, past_rgb=left[index][None])
        if not all(torch.equal(getattr(reference, name), getattr(frozen, name)) for name in frozen.__dataclass_fields__):
            raise SystemExit("extracted alignment parity failed")
        candidates.append(reference.aligned_past_disparity[:, 0]); candidate_valid.append(reference.aligned_validity[:, 0])
        support.append(reference.warp_support[:, 0]); confidence.append(reference.forward_backward_confidence[:, 0])
    values = (raw, torch.stack(candidates, 1), torch.stack(candidate_valid, 1), torch.stack(support, 1),
              torch.stack(confidence, 1), torch.tensor(ANCHOR_AGES, device=device), torch.zeros(4, device=device))
    original_evidence = OriginalEvidence(*values); frozen_evidence = MultiAnchorEvidence(*values)
    state = torch.load(ROOT / "checkpoints/raw_multi_anchor_best_validation.pt", map_location="cpu", weights_only=False)
    original_model = OriginalModel(state["channels"], state["blocks"]).to(device).eval(); original_model.load_state_dict(state["model"])
    frozen_model = RawMultiAnchorRefiner(state["channels"], state["blocks"]).to(device).eval(); frozen_model.load_state_dict(state["model"])
    original_output = original_model(original_evidence); frozen_output = frozen_model(frozen_evidence)
    reference_prediction, reference_accepted, reference_chosen, reference_effective = original_retrieve(
        raw, original_evidence, original_output, probability_threshold=.9, utility_threshold_px=.1, hard=False)
    frozen_prediction, frozen_accepted, frozen_chosen, frozen_effective = retrieve_and_fuse(
        raw, frozen_evidence, frozen_output, probability_threshold=.9, utility_threshold_px=.1, hard=False)
    selected = torch.gather(original_evidence.candidates, 1, reference_chosen)
    score = torch.gather(original_output.selection_score, 1, reference_chosen)
    fusion = torch.gather(original_output.fusion_weight, 1, reference_chosen)
    selected_support = torch.gather(original_evidence.available, 1, reference_chosen)
    reference_age = torch.tensor(ANCHOR_AGES, device=device)[reference_chosen]
    proposal = raw + fusion * (selected - raw)
    comparisons = {
        "selected_anchor_age": maximum_difference(result.selected_anchor_age, reference_age),
        "selection_score": maximum_difference(result.selection_score, score),
        "fusion_weight": maximum_difference(result.fusion_weight, reference_effective),
        "accepted_mask": maximum_difference(result.accepted_mask, reference_accepted),
        "support_mask": maximum_difference(result.support_mask, selected_support),
        "aligned_anchor": maximum_difference(result.selected_aligned_anchor, selected),
        "proposal_disparity": maximum_difference(result.proposal_disparity, proposal),
        "final_disparity": maximum_difference(result.output_disparity, reference_prediction),
        "copied_model_final": maximum_difference(frozen_prediction, reference_prediction),
        "copied_model_chosen": maximum_difference(frozen_chosen, reference_chosen),
        "copied_model_accepted": maximum_difference(frozen_accepted, reference_accepted),
        "copied_model_weight": maximum_difference(frozen_effective, reference_effective),
    }
    if any(value != 0.0 for value in comparisons.values()):
        raise SystemExit(f"tensor parity failed: {comparisons}")
    if not torch.equal(bank.anchor(CURRENT + 1, 1).disparity, raw):
        raise SystemExit("raw-only writeback failed")
    fallback_bank = RawAnchorBank()
    fallback = refiner.step(current_left, right[None], raw, valid, fallback_bank,
                            frame_id=info.frame_ids[CURRENT], frame_index=CURRENT)
    if not torch.equal(fallback.output_disparity, raw) or fallback.accepted_mask.any():
        raise SystemExit("exact raw fallback failed")
    summary = {"project": "ARGOS v2", "status": "PASS", "dataset_id": 2, "sequence": SEQUENCE,
               "backbone": BACKBONE, "frame_index": CURRENT, "anchor_ages": list(ANCHOR_AGES),
               "direct_flow_pairs": 4, "tensor_atol": 0.0, "maximum_differences": comparisons,
               "no_training": True, "no_dataset_7_access": True, "temporary_outputs_deleted": True}
    TMP.mkdir(parents=True, exist_ok=False)
    (TMP / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    shutil.rmtree(TMP)


if __name__ == "__main__":
    main()
