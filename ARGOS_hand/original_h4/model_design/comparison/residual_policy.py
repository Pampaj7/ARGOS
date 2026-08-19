#!/usr/bin/env python3
"""A two-parameter, spatially varying, unlearned fusion rule.

The review's sharpest objection to the closure is that its baselines are all
spatially constant except one, and that one -- forward-backward confidence used
directly as the weight -- is deliberately badly scaled. The obvious middle ground
was never tested:

    w_t(x) = alpha * exp(-|d_raw(x) - m~(x)| / tau)

Spatially varying, two parameters, no learning, and it uses precisely the signal the
paper's own analysis says the head is responding to: the magnitude of the residual
between the current estimate and the aligned memory. If this matches the learned head,
the $177$k parameters buy a tuned constant and the contribution does not stand.

`experimental_policies.py` is pinned by the closure's freeze manifest, so this lives
beside it rather than extending it, and reuses the same evidence, support and blend so
the only difference from the fixed-weight baselines is how $w$ is computed.
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ResidualSpec:
    name: str
    horizon: int
    alpha: float
    tau: float


class ResidualPolicy:
    """Raw-versus-memory residual as the weight, with a scale and a gain."""

    def __init__(self, spec: ResidualSpec, *, device: str = "cuda:0") -> None:
        self.spec, self.device, self.horizon = spec, device, spec.horizon

    def describe(self) -> dict[str, Any]:
        return {"module": "residual_policy", "method": self.spec.name, "horizon": self.horizon,
                "alpha": self.spec.alpha, "tau": self.spec.tau,
                "rule": "w = alpha * exp(-|raw - aligned| / tau)", "trained_parameters": 0,
                "code": str(Path(__file__).resolve()),
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def start(self, frame: Mapping[str, Any]) -> dict[str, Any]:
        return {"disparity": frame["raw"], "support": frame["raw_valid"].bool(), "reset": True,
                "state_age": 0,
                "diagnostics": {"method": self.spec.name, "horizon": self.horizon,
                                "update_magnitude": 0.0}}

    def step(self, frame: Mapping[str, Any]) -> dict[str, Any]:
        import torch
        root = Path(__file__).resolve().parents[2]
        frozen = root.parents[1] / "ARGOS_FREEZED/src"
        if str(frozen) not in sys.path:
            sys.path.insert(0, str(frozen))
        from argos_freezed.alignment.bida_pull_warp import temporal_disparity_evidence
        with torch.inference_mode():
            evidence = temporal_disparity_evidence(
                frame["raw"], frame["past_disparity"], frame["forward_flow"], frame["backward_flow"],
                current_valid=frame["raw_valid"], past_valid=frame["past_valid"],
                current_rgb=frame["current_rgb"], past_rgb=frame["past_rgb"])
            support = (frame["raw_valid"].bool() & evidence.aligned_validity.bool()
                       & evidence.warp_support.bool())
            residual = (evidence.aligned_past_disparity - frame["raw"]).abs()
            weight = (self.spec.alpha * torch.exp(-residual / self.spec.tau)).clamp(0.0, 1.0)
            fused = frame["raw"] + weight * (evidence.aligned_past_disparity - frame["raw"])
            fused = torch.where(support, fused, frame["raw"])
            if not torch.isfinite(fused).all():
                raise ValueError("non-finite residual-policy output")
            update = float((fused - frame["raw"])[support].abs().mean()) if bool(support.any()) else 0.0
            mean_weight = float(weight[support].mean()) if bool(support.any()) else 0.0
        return {"disparity": fused, "support": support, "reset": bool(frame["reanchor"]),
                "state_age": int(frame["state_age"]),
                "aligned_memory": evidence.aligned_past_disparity,
                "diagnostics": {"method": self.spec.name, "horizon": self.horizon,
                                "update_magnitude": update, "temporal_weight": mean_weight}}


def factory(*, alpha: float = 0.5, tau: float = 1.0, horizon: int = 4,
            device: str = "cuda:0", **_: Any) -> ResidualPolicy:
    return ResidualPolicy(ResidualSpec(f"residual_a{alpha:g}_t{tau:g}_h{horizon}", horizon,
                                       alpha, tau), device=device)
