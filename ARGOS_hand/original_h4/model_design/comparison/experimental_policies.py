"""Frozen, non-learned temporal baselines for the H4 experimental closure."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class PolicySpec:
    name: str
    horizon: int | None
    weight: float | str
    memory: str = "recurrent"

    def __post_init__(self) -> None:
        if self.memory not in {"recurrent", "raw_previous"}:
            raise ValueError(f"unsupported memory policy: {self.memory}")
        if self.memory == "raw_previous" and self.horizon != 1:
            raise ValueError("raw_previous is defined only for H=1")
        if self.memory == "recurrent" and self.horizon is None and self.name.startswith("ema"):
            raise ValueError("EMA baselines require a finite declared horizon")


POLICIES = {
    **{f"fixed_w{weight:.1f}_h4": PolicySpec(f"fixed_w{weight:.1f}_h4", 4, weight) for weight in (.1, .2, .3, .5)},
    "fb_confidence_h4": PolicySpec("fb_confidence_h4", 4, "fb"),
    "warped_recurrent_h4": PolicySpec("warped_recurrent_h4", 4, 1.0),
    "warped_raw_previous_h1": PolicySpec("warped_raw_previous_h1", 1, 1.0, "raw_previous"),
    "ema2_h4": PolicySpec("ema2_h4", 4, .5),
    "ema3_h4": PolicySpec("ema3_h4", 4, 2 / 3),
}


class ExperimentalPolicy:
    """Uses the validated BiDA evidence; only the predeclared blend changes."""
    def __init__(self, spec: PolicySpec, *, device: str = "cuda:0") -> None:
        self.spec, self.device, self.horizon = spec, device, spec.horizon
        if spec.memory == "raw_previous" and spec.horizon != 1:
            raise ValueError("raw_previous must reset at every transition (H=1)")

    def describe(self) -> dict[str, Any]:
        return {"module": "experimental_policy", "method": self.spec.name, "horizon": self.horizon,
                "memory": self.spec.memory, "weight": self.spec.weight,
                "code": str(Path(__file__).resolve()), "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def start(self, frame: Mapping[str, Any]) -> dict[str, Any]:
        return {"disparity": frame["raw"], "support": frame["raw_valid"].bool(), "reset": True, "state_age": 0,
                "diagnostics": {"method": self.spec.name, "horizon": self.horizon, "update_magnitude": 0.0}}

    def step(self, frame: Mapping[str, Any]) -> dict[str, Any]:
        import sys
        import torch
        root = Path(__file__).resolve().parents[2]
        frozen = root.parents[1] / "ARGOS_FREEZED/src"
        if str(frozen) not in sys.path:
            sys.path.insert(0, str(frozen))
        from argos_freezed.alignment.bida_pull_warp import temporal_disparity_evidence
        with torch.inference_mode():
            for name in ("raw", "past_disparity", "forward_flow", "backward_flow"):
                if not torch.isfinite(frame[name]).all():
                    raise ValueError(f"non-finite adapter input: {name}")
            if self.spec.memory == "raw_previous" and (frame["horizon"] != 1 or not frame["reanchor"]):
                raise RuntimeError("raw_previous policy requires an H=1 raw re-anchor on every step")
            evidence = temporal_disparity_evidence(
                frame["raw"], frame["past_disparity"], frame["forward_flow"], frame["backward_flow"],
                current_valid=frame["raw_valid"], past_valid=frame["past_valid"],
                current_rgb=frame["current_rgb"], past_rgb=frame["past_rgb"])
            for name, value in evidence.as_dict().items():
                if not torch.isfinite(value).all():
                    raise ValueError(f"non-finite temporal evidence: {name}")
            support = frame["raw_valid"].bool() & evidence.aligned_validity.bool() & evidence.warp_support.bool()
            weight = evidence.forward_backward_confidence.clamp(0, 1) if self.spec.weight == "fb" else frame["raw"].new_full(frame["raw"].shape, float(self.spec.weight))
            fused = frame["raw"] + weight * (evidence.aligned_past_disparity - frame["raw"])
            fused = torch.where(support, fused, frame["raw"])
            if not torch.isfinite(weight).all() or not torch.isfinite(fused).all():
                raise ValueError("non-finite baseline output")
            update = float((fused - frame["raw"])[support].abs().mean()) if bool(support.any()) else 0.0
            mean_weight = float(weight[support].mean()) if bool(support.any()) else 0.0
        return {"disparity": fused, "support": support, "reset": bool(frame["reanchor"]), "state_age": int(frame["state_age"]),
                "aligned_memory": evidence.aligned_past_disparity,
                "diagnostics": {"method": self.spec.name, "horizon": self.horizon, "memory": self.spec.memory,
                                "update_magnitude": update, "temporal_weight": mean_weight,
                                "fb_confidence": float(evidence.forward_backward_confidence[support].mean()) if bool(support.any()) else 0.0}}


def factory(*, method: str = "fixed_w0.5_h4", device: str = "cuda:0", **_: Any) -> ExperimentalPolicy:
    try:
        return ExperimentalPolicy(POLICIES[method], device=device)
    except KeyError as error:
        raise ValueError(f"unknown experimental policy: {method}") from error
