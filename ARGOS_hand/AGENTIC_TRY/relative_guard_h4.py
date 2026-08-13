"""Relative-consistency guard over the frozen HardH4 .35 endpoint."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from hard_h4 import HardH4


CODE = Path(__file__).resolve()
KAPPA = 2.0
TAU = .35


def relative_guard_endpoint(raw: Any, hard_disparity: Any, support: Any) -> tuple[Any, Any]:
    """Keep HardH4 only where its causal output is relatively consistent with raw."""
    import torch

    valid_raw = torch.isfinite(raw) & (raw > 0)
    valid_hard = torch.isfinite(hard_disparity) & (hard_disparity > 0)
    accepted = support.bool() & valid_raw & valid_hard & (raw / KAPPA <= hard_disparity) & (hard_disparity <= KAPPA * raw)
    return torch.where(accepted, hard_disparity, raw), accepted


class RelativeGuardH4(HardH4):
    """HardH4 .35 with a fixed, backbone-agnostic relative-value guard."""

    def __init__(self, *, device: str = "cuda:0") -> None:
        super().__init__(threshold=TAU, device=device)

    def describe(self) -> dict[str, Any]:
        hard = super().describe()
        return hard | {
            "module": "relative_guard_h4",
            "hard_h4_code": hard["code"],
            "hard_h4_code_sha256": hard["code_sha256"],
            "kappa": KAPPA,
            "endpoint": "S = raw_valid & aligned_validity & warp_support; V(x) = isfinite(x) & (x > 0); accepted = S & V(raw) & V(hard035) & (raw / 2 <= hard035) & (hard035 <= 2 * raw); disparity = where(accepted, hard035, raw)",
            "code": str(CODE),
            "code_sha256": hashlib.sha256(CODE.read_bytes()).hexdigest(),
        }

    def step(self, frame: Mapping[str, Any]) -> dict[str, Any]:
        import torch

        result = super().step(frame)
        disparity, _ = relative_guard_endpoint(frame["raw"], result["disparity"], result["support"])
        support = result["support"]
        finite_support = support & torch.isfinite(frame["raw"]) & torch.isfinite(disparity)
        update = float((disparity - frame["raw"]).abs()[finite_support].mean()) if bool(finite_support.any()) else 0.0
        return result | {"disparity": disparity,
                         "diagnostics": result["diagnostics"] | {"update_magnitude": update}}


def factory(**kwargs: Any) -> RelativeGuardH4:
    return RelativeGuardH4(**kwargs)
