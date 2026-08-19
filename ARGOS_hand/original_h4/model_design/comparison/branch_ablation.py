#!/usr/bin/env python3
"""Reset-only and fusion-only, decomposing w_t = r_t * f_t.

The review asks which branch produces the gain, and the paper never separates them: it
reports the product and analyses the product. If one branch carries the result, the other
is architecture that costs parameters and explains nothing.

Both variants reuse the trained head unchanged and only change how its two published
weights are combined at inference:

    reset-only   w = r_t          (fusion weight forced to 1)
    fusion-only  w = f_t          (reset gate forced to 1)

Nothing is retrained and no threshold moves, so any difference is attributable to the
branch and not to a different model. The canonical product is the reference.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

from model_design.comparison.canonical_h4 import CanonicalH4

ROOT = Path(__file__).resolve().parents[2]


class BranchAblation(CanonicalH4):
    """Canonical head, one factor of the temporal weight suppressed at inference."""

    def __init__(self, *, branch: str, horizon: int | None = 4, **kwargs: Any) -> None:
        if branch not in {"reset_only", "fusion_only"}:
            raise ValueError("branch must be reset_only or fusion_only")
        super().__init__(**kwargs)
        self.branch, self.horizon = branch, horizon

    def describe(self) -> dict[str, Any]:
        return super().describe() | {"module": "branch_ablation", "branch": self.branch,
                                     "horizon": self.horizon,
                                     "rule": "w = r_t" if self.branch == "reset_only" else "w = f_t"}

    def step(self, frame: Mapping[str, Any]) -> dict[str, Any]:
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
                current_rgb=frame["current_rgb"], past_rgb=frame["past_rgb"])
            cues = build_codd_cues(
                extractor, raw=frame["raw"], aligned_memory=evidence.aligned_past_disparity,
                current_rgb=frame["current_rgb"], current_right_rgb=frame["current_right_rgb"],
                past_rgb=frame["past_rgb"], flow_current_to_past=frame["forward_flow"],
                flow_magnitude=evidence.flow_magnitude,
                forward_backward_confidence=evidence.forward_backward_confidence,
                warp_support=evidence.warp_support, aligned_valid=evidence.aligned_validity,
                include_learned_stereo_evidence=True)
            output = model(cues, frame["raw"], evidence.aligned_past_disparity)
            weight = output.reset_weight if self.branch == "reset_only" else output.fusion_weight
            if weight.shape != frame["raw"].shape:
                import torch.nn.functional as F
                weight = F.interpolate(weight, size=frame["raw"].shape[-2:], mode="bilinear",
                                       align_corners=True)
            fused = frame["raw"] + weight * (evidence.aligned_past_disparity - frame["raw"])
            support = (frame["raw_valid"].bool() & evidence.aligned_validity.bool()
                       & evidence.warp_support.bool())
            fused = torch.where(support, fused, frame["raw"])
            if not torch.isfinite(fused).all():
                raise ValueError("non-finite branch-ablation output")
            update = float((fused - frame["raw"]).abs()[support].mean()) if bool(support.any()) else 0.0
        return {"disparity": fused, "support": support, "reset": bool(frame["reanchor"]),
                "state_age": int(frame["state_age"]),
                "aligned_memory": evidence.aligned_past_disparity,
                "diagnostics": {"update_magnitude": update, "branch": self.branch,
                                "temporal_weight": float(weight[support].mean()) if bool(support.any()) else 0.0}}


def factory_reset_only(*, device: str = "cuda:0", horizon: int = 4, **_: Any) -> BranchAblation:
    return BranchAblation(branch="reset_only", horizon=horizon, device=device)


def factory_fusion_only(*, device: str = "cuda:0", horizon: int = 4, **_: Any) -> BranchAblation:
    return BranchAblation(branch="fusion_only", horizon=horizon, device=device)
