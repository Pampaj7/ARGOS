"""Gated canonical-soft endpoint over the immutable hard H4 module."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from hard_h4 import HardH4


CODE = Path(__file__).resolve()


def gated_soft_endpoint_fusion(raw: Any, fused_disparity: Any, temporal_weight: Any, threshold: float) -> tuple[Any, Any]:
    """Retain raw below threshold, else retain canonical soft fusion."""
    import torch

    accepted = temporal_weight >= threshold
    return torch.where(accepted, fused_disparity, raw), accepted


class GatedSoftH4(HardH4):
    def describe(self) -> dict[str, Any]:
        inherited = super().describe()
        return inherited | {
            "module": "gated_soft_h4",
            "endpoint": "accepted = temporal_weight >= threshold; disparity = where(accepted, output.fused_disparity, raw)",
            "hard_h4_code": inherited["code"],
            "hard_h4_code_sha256": inherited["code_sha256"],
            "code": str(CODE),
            "code_sha256": hashlib.sha256(CODE.read_bytes()).hexdigest(),
        }

    def step(self, frame: Mapping[str, Any]) -> dict[str, Any]:
        import sys
        import torch

        from model_design.comparison.canonical_h4 import ROOT as CANONICAL_ROOT

        frozen = CANONICAL_ROOT.parents[1] / "ARGOS_FREEZED/src"
        if str(frozen) not in sys.path:
            sys.path.insert(0, str(frozen))
        from argos_freezed.alignment.bida_pull_warp import temporal_disparity_evidence

        model, extractor, build_codd_cues = self._load()
        with torch.inference_mode():
            evidence = temporal_disparity_evidence(
                frame["raw"], frame["past_disparity"], frame["forward_flow"], frame["backward_flow"],
                current_valid=frame["raw_valid"], past_valid=frame["past_valid"],
                current_rgb=frame["current_rgb"], past_rgb=frame["past_rgb"],
            )
            cues = build_codd_cues(
                extractor, raw=frame["raw"], aligned_memory=evidence.aligned_past_disparity,
                current_rgb=frame["current_rgb"], current_right_rgb=frame["current_right_rgb"],
                past_rgb=frame["past_rgb"], flow_current_to_past=frame["forward_flow"],
                flow_magnitude=evidence.flow_magnitude, forward_backward_confidence=evidence.forward_backward_confidence,
                warp_support=evidence.warp_support, aligned_valid=evidence.aligned_validity,
                include_learned_stereo_evidence=True,
            )
            output = model(cues, frame["raw"], evidence.aligned_past_disparity)
            disparity, accepted = gated_soft_endpoint_fusion(frame["raw"], output.fused_disparity, output.temporal_weight, self.threshold)
            support = frame["raw_valid"].bool() & evidence.aligned_validity.bool() & evidence.warp_support.bool()
            update = float((disparity - frame["raw"]).abs()[support].mean()) if bool(support.any()) else 0.0
        return {"disparity": disparity, "support": support,
                "reset": bool(frame["reanchor"]), "state_age": int(frame["state_age"]),
                "aligned_memory": evidence.aligned_past_disparity,
                "diagnostics": {"update_magnitude": update,
                                "temporal_weight": float(accepted.float()[support].mean()) if bool(support.any()) else 0.0,
                                "fb_confidence": float(evidence.forward_backward_confidence[support].mean()) if bool(support.any()) else 0.0}}


def factory_035(**kwargs: Any) -> GatedSoftH4:
    return GatedSoftH4(threshold=.35, **kwargs)
