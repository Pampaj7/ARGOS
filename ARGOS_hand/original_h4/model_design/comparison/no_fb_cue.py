#!/usr/bin/env python3
"""The trained head with the forward-backward confidence cue replaced by a constant.

Runtime says the two SEA-RAFT passes are 76% of the module's cost and the reverse pass
exists only to produce C^FB_t. Removing it would free 13.7 ms, nearly three times what
re-canonicalising to A2 frees -- but the paper cannot propose that without knowing what the
cue is worth, and "we did not measure it" is what the review already objects to elsewhere.

This measures the accuracy half. The cue enters the evidence as one cache-grid channel
(`codd_style_fusion.py`, the concat that clamps it to [0,1]); feeding a constant removes its
information while keeping the channel count, so the trained head runs unmodified and the
difference is the cue rather than a different architecture.

What this is not: a retrained no-FB model. A head trained without the cue could learn to
use the remaining channels differently and would likely do better than this. The number
here is therefore an upper bound on the cost of dropping the cue at inference, and a lower
bound on what a retrained variant could achieve -- which is the honest direction for a
saving we would like to claim.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

from model_design.comparison.ablation_horizons import AblationHorizon
from model_design.comparison.canonical_h4 import CanonicalH4

ROOT = Path(__file__).resolve().parents[2]


class _ConstantFBCue:
    def __init__(self, *, value: float = 1.0, horizon: int | None = 4, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.value, self.horizon = value, horizon

    def describe(self) -> dict[str, Any]:
        return super().describe() | {"module": "no_fb_cue", "constant": self.value,
                                     "horizon": self.horizon,
                                     "rule": "C^FB replaced by a constant at inference"}

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
            constant = torch.full_like(evidence.forward_backward_confidence, self.value)
            cues = build_codd_cues(
                extractor, raw=frame["raw"], aligned_memory=evidence.aligned_past_disparity,
                current_rgb=frame["current_rgb"], current_right_rgb=frame["current_right_rgb"],
                past_rgb=frame["past_rgb"], flow_current_to_past=frame["forward_flow"],
                flow_magnitude=evidence.flow_magnitude,
                forward_backward_confidence=constant,
                warp_support=evidence.warp_support, aligned_valid=evidence.aligned_validity,
                # TETHER's builder forces this to False and ignores the extractor, so the
                # same call serves both heads: `include_learned_stereo_evidence` is the
                # 142-channel head's switch, not a claim about which head is running.
                include_learned_stereo_evidence=True)
            output = model(cues, frame["raw"], evidence.aligned_past_disparity)
            support = (frame["raw_valid"].bool() & evidence.aligned_validity.bool()
                       & evidence.warp_support.bool())
            update = float((output.fused_disparity - frame["raw"]).abs()[support].mean()) if bool(support.any()) else 0.0
        return {"disparity": output.fused_disparity, "support": support,
                "reset": bool(frame["reanchor"]), "state_age": int(frame["state_age"]),
                "aligned_memory": evidence.aligned_past_disparity,
                "diagnostics": {"update_magnitude": update, "fb_constant": self.value}}


class NoFBCue(_ConstantFBCue, CanonicalH4):
    """The 142-channel learned-evidence head, now the paper's ablation."""


class NoFBCueTether(_ConstantFBCue, AblationHorizon):
    """TETHER, the shipped 38-channel head.

    The cue survives recanonicalisation: it is one of the six motion channels
    (`codd_style_fusion.py`, the `motion` concat), which sits in the tail shared by both
    branches of the builder. Dropping the learned stereo evidence removed 104 correlation
    channels and left C^FB exactly where it was, so the saving has to be measured on the
    head the paper actually ships rather than inherited from the ablation's number.
    """

    def __init__(self, *, variant: str = "A2_no_learned_evidence", **kwargs: Any) -> None:
        super().__init__(variant=variant, **kwargs)


HEADS = {"tether": NoFBCueTether, "learned_evidence": NoFBCue}


def factory(*, value: float = 1.0, horizon: int = 4, device: str = "cuda:0",
            head: str = "tether", **_: Any) -> _ConstantFBCue:
    return HEADS[head](value=value, horizon=horizon, device=device)


def factory_seed1(*, value: float = 1.0, horizon: int = 4, device: str = "cuda:0",
                  **_: Any) -> _ConstantFBCue:
    return NoFBCueTether(value=value, horizon=horizon, device=device, seed=1)


def factory_seed2(*, value: float = 1.0, horizon: int = 4, device: str = "cuda:0",
                  **_: Any) -> _ConstantFBCue:
    return NoFBCueTether(value=value, horizon=horizon, device=device, seed=2)
