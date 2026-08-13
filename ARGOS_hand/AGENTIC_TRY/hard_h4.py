"""Hard raw-or-memory endpoints over the immutable canonical H4 module."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from model_design.comparison.canonical_h4 import CanonicalH4, ROOT as CANONICAL_ROOT


CODE = Path(__file__).resolve()


def hard_endpoint_fusion(raw: Any, aligned_memory: Any, temporal_weight: Any, threshold: float) -> tuple[Any, Any]:
    """The historical endpoint: retain raw below threshold, else use memory."""
    import torch

    accepted = temporal_weight >= threshold
    return torch.where(accepted, aligned_memory, raw), accepted


class HardH4(CanonicalH4):
    def __init__(self, *, threshold: float, device: str = "cuda:0") -> None:
        super().__init__(device=device)
        self.threshold = threshold

    def describe(self) -> dict[str, Any]:
        canonical = super().describe()
        return canonical | {
            "module": "hard_h4",
            "threshold": self.threshold,
            "endpoint": "accepted = temporal_weight >= threshold; disparity = where(accepted, aligned_memory, raw)",
            "canonical_h4_code": canonical["code"],
            "canonical_h4_code_sha256": canonical["code_sha256"],
            "code": str(CODE),
            "code_sha256": hashlib.sha256(CODE.read_bytes()).hexdigest(),
        }

    def step(self, frame: Mapping[str, Any]) -> dict[str, Any]:
        import sys
        import torch

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
            disparity, accepted = hard_endpoint_fusion(frame["raw"], evidence.aligned_past_disparity, output.temporal_weight, self.threshold)
            support = frame["raw_valid"].bool() & evidence.aligned_validity.bool() & evidence.warp_support.bool()
            update = float((disparity - frame["raw"]).abs()[support].mean()) if bool(support.any()) else 0.0
        return {"disparity": disparity, "support": support,
                "reset": bool(frame["reanchor"]), "state_age": int(frame["state_age"]),
                "aligned_memory": evidence.aligned_past_disparity,
                "diagnostics": {"update_magnitude": update,
                                "temporal_weight": float(accepted.float()[support].mean()) if bool(support.any()) else 0.0,
                                "fb_confidence": float(evidence.forward_backward_confidence[support].mean()) if bool(support.any()) else 0.0}}


def factory_035(**kwargs: Any) -> HardH4:
    return HardH4(threshold=.35, **kwargs)


def factory_050(**kwargs: Any) -> HardH4:
    return HardH4(threshold=.50, **kwargs)
