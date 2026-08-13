"""The frozen, dataset-2-selected bounded H=4 temporal module."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "model_design/checkpoints/codd_style_h4_best_validation.pt"
POLICY = ROOT / "model_design/checkpoints/codd_style_h4_policy.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CanonicalH4:
    """Actual frozen H4 inference; data loading and metrics live in the driver."""

    def __init__(self, *, device: str = "cuda:0") -> None:
        import sys
        sys.path.insert(0, str(ROOT / "scripts"))
        from canonical_h4_provenance import verify_canonical_inputs
        self._policy = verify_canonical_inputs(CHECKPOINT, POLICY)
        self.device = device
        self._model = self._extractor = self._build_cues = None

    def describe(self) -> dict[str, Any]:
        return {"module": "canonical_h4", "checkpoint": str(CHECKPOINT),
                "checkpoint_sha256": _sha256(CHECKPOINT), "policy": str(POLICY),
                "policy_sha256": _sha256(POLICY), "code": str(Path(__file__).resolve()),
                "code_sha256": _sha256(Path(__file__).resolve()),
                "reset_protocol": "fixed H=4; raw t-1 after re-anchor; preceding fused otherwise"}

    def _load(self):
        if self._model is not None:
            return self._model, self._extractor, self._build_cues
        import torch
        import sys
        provenance = ROOT.parents[1] / "ARGOS_FREEZED/experiments/02_massive_training/scripts"
        if str(provenance) not in sys.path:
            sys.path.insert(0, str(provenance))
        from provenance.codd_style_fusion import CODDStyleFusionHead, FrozenResNet18Layer1, build_codd_cues
        state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        self._model = CODDStyleFusionHead(state["cue_channels"]).to(self.device).eval().requires_grad_(False)
        self._model.load_state_dict(state["model"], strict=True)
        self._extractor = FrozenResNet18Layer1().to(self.device).eval().requires_grad_(False)
        self._build_cues = build_codd_cues
        return self._model, self._extractor, self._build_cues

    def start(self, frame: Mapping[str, Any]) -> dict[str, Any]:
        """The first frame of every true sequence is exact raw identity."""
        return {"disparity": frame["raw"], "support": frame["raw_valid"].bool(),
                "reset": True, "state_age": 0, "diagnostics": {"update_magnitude": 0.0}}

    def step(self, frame: Mapping[str, Any]) -> dict[str, Any]:
        """Fuse current raw with the causal memory supplied by the protocol driver."""
        import sys
        import torch
        frozen = ROOT.parents[1] / "ARGOS_FREEZED/src"
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
            support = frame["raw_valid"].bool() & evidence.aligned_validity.bool() & evidence.warp_support.bool()
            update = float((output.fused_disparity - frame["raw"]).abs()[support].mean()) if bool(support.any()) else 0.0
        return {"disparity": output.fused_disparity, "support": support,
                "reset": bool(frame["reanchor"]), "state_age": int(frame["state_age"]),
                "aligned_memory": evidence.aligned_past_disparity,
                "diagnostics": {"update_magnitude": update,
                                "temporal_weight": float(output.temporal_weight[support].mean()) if bool(support.any()) else 0.0,
                                "fb_confidence": float(evidence.forward_backward_confidence[support].mean()) if bool(support.any()) else 0.0}}


def factory(**kwargs: Any) -> CanonicalH4:
    return CanonicalH4(**kwargs)
