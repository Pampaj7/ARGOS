"""Canonical frozen ARGOS v2 geometry-v1 pipeline."""
from __future__ import annotations

import torch

from .alignment.bida_pull_warp import temporal_disparity_evidence
from .alignment.sea_raft_adapter import SEARAFTFlowAdapter
from .constants import ANCHOR_AGES, CHECKPOINT, PARAMETER_COUNT, PROBABILITY_THRESHOLD, UTILITY_THRESHOLD_PX
from .memory_bank import RawAnchorBank
from .models.raw_multi_anchor_refiner import MultiAnchorEvidence, RawMultiAnchorRefiner, retrieve_and_fuse
from .types import RefinementResult


class FrozenArgosGeometryRefiner:
    """Frozen non-recurrent refiner with raw-only memory writeback."""

    def __init__(self, *, checkpoint=CHECKPOINT, device=None, flow_adapter=None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if tuple(state["ages"]) != ANCHOR_AGES or any(state["provenance"]):
            raise RuntimeError("checkpoint is not the frozen raw CS1/CS2/CS4/CS8 model")
        self.model = RawMultiAnchorRefiner(int(state["channels"]), int(state["blocks"])).to(self.device)
        self.model.load_state_dict(state["model"], strict=True)
        self.model.eval().requires_grad_(False)
        if sum(parameter.numel() for parameter in self.model.parameters()) != PARAMETER_COUNT:
            raise RuntimeError("frozen parameter count mismatch")
        self.flow = flow_adapter or SEARAFTFlowAdapter(device=self.device)

    @torch.inference_mode()
    def step(self, current_left_rgb: torch.Tensor, current_right_rgb: torch.Tensor,
             current_raw_disparity: torch.Tensor, current_raw_validity: torch.Tensor,
             raw_anchor_bank: RawAnchorBank, *, frame_id: str, frame_index: int,
             timestamp: float | None = None) -> RefinementResult:
        raw = current_raw_disparity.to(self.device, dtype=torch.float32)
        valid = current_raw_validity.to(self.device).bool()
        left = current_left_rgb.to(self.device, dtype=torch.float32)
        right = current_right_rgb.to(self.device, dtype=torch.float32)
        if raw.ndim != 4 or raw.shape[0:2] != (1, 1) or valid.shape != raw.shape:
            raise ValueError("current raw disparity/validity must share [1,1,H,W]")
        if left.shape != (1, 3, *raw.shape[-2:]) or right.shape != left.shape:
            raise ValueError("left/right RGB must share [1,3,H,W] with disparity")
        if not torch.isfinite(raw[valid]).all() or bool((raw[valid] <= 0).any()):
            raise ValueError("valid disparity must be finite positive-left")

        candidates, candidate_valid, support, confidence, missing = [], [], [], [], []
        for age in ANCHOR_AGES:
            anchor = raw_anchor_bank.anchor(frame_index, age)
            if anchor is None:
                candidates.append(raw[:, 0])
                candidate_valid.append(torch.zeros_like(valid[:, 0]))
                support.append(torch.zeros_like(valid[:, 0]))
                confidence.append(torch.zeros_like(raw[:, 0]))
                missing.append(age)
                continue
            anchor_disparity = anchor.disparity.to(self.device, dtype=torch.float32)
            anchor_valid = anchor.validity.to(self.device).bool()
            anchor_rgb = anchor.left_rgb.to(self.device, dtype=torch.float32)
            current_to_anchor = self.flow.current_to_anchor(left, anchor_rgb)
            anchor_to_current = self.flow.anchor_to_current(anchor_rgb, left)
            aligned = temporal_disparity_evidence(raw, anchor_disparity, current_to_anchor, anchor_to_current,
                current_valid=valid, past_valid=anchor_valid, current_rgb=left, past_rgb=anchor_rgb)
            candidates.append(aligned.aligned_past_disparity[:, 0])
            candidate_valid.append(aligned.aligned_validity[:, 0])
            support.append(aligned.warp_support[:, 0])
            confidence.append(aligned.forward_backward_confidence[:, 0])

        evidence = MultiAnchorEvidence(raw, torch.stack(candidates, 1), torch.stack(candidate_valid, 1),
            torch.stack(support, 1), torch.stack(confidence, 1),
            torch.tensor(ANCHOR_AGES, device=self.device), torch.zeros(len(ANCHOR_AGES), device=self.device))
        network = self.model(evidence)
        output, accepted, chosen, effective_weight = retrieve_and_fuse(raw, evidence, network,
            probability_threshold=PROBABILITY_THRESHOLD, utility_threshold_px=UTILITY_THRESHOLD_PX, hard=False)
        selected_anchor = torch.gather(evidence.candidates, 1, chosen)
        selected_score = torch.gather(network.selection_score, 1, chosen)
        selected_weight = torch.gather(network.fusion_weight, 1, chosen)
        selected_support = torch.gather(evidence.available, 1, chosen)
        selected_confidence = torch.gather(evidence.fb_confidence, 1, chosen)
        ages = torch.tensor(ANCHOR_AGES, device=self.device)[chosen]
        proposal = raw + selected_weight * (selected_anchor - raw)
        result = RefinementResult(output, raw, proposal, ages, selected_anchor, selected_score,
            effective_weight, accepted, selected_support, valid & selected_support,
            selected_confidence, (output - raw).abs(),
            {"project": "ARGOS v2", "version": "geometry_v1", "anchor_ages": ANCHOR_AGES,
             "missing_anchor_ages": tuple(missing), "flow": "direct_current_to_anchor",
             "flow_composition": False, "recurrent_state": False, "fused_state_writeback": False})
        raw_anchor_bank.append_raw(raw, valid, left, frame_id=frame_id, frame_index=frame_index,
                                   timestamp=timestamp, provenance="independent_frozen_stereo")
        return result
