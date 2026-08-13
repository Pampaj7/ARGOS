"""Lower-ratio guard over the frozen HardH4 .35 endpoint."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from hard_h4 import HardH4


CODE = Path(__file__).resolve()
TAU = .35
LOWER_RATIO = .5


def lower_guard_endpoint(raw: Any, hard_disparity: Any, support: Any) -> tuple[Any, Any]:
    """Keep hard .35 only on existing support and at least half of raw."""
    import torch

    valid_raw = torch.isfinite(raw) & (raw > 0)
    accepted = support.bool() & valid_raw & torch.isfinite(hard_disparity) & (hard_disparity >= LOWER_RATIO * raw)
    return torch.where(accepted, hard_disparity, raw), accepted


class LowerGuardH4(HardH4):
    """Frozen HardH4 .35 with the fixed lower-only ratio guard."""

    def __init__(self, *, device: str = "cuda:0") -> None:
        super().__init__(threshold=TAU, device=device)

    def describe(self) -> dict[str, Any]:
        hard = super().describe()
        return dict(hard, **{
            "module": "lower_guard_h4",
            "hard_h4_code": hard["code"],
            "hard_h4_code_sha256": hard["code_sha256"],
            "lower_ratio": LOWER_RATIO,
            "endpoint": "S = raw_valid & aligned_validity & warp_support; Vr = isfinite(raw) & (raw > 0); A = S & Vr & isfinite(hard035) & (hard035 >= 0.5 * raw); disparity = where(A, hard035, raw)",
            "code": str(CODE),
            "code_sha256": hashlib.sha256(CODE.read_bytes()).hexdigest(),
        })

    def step(self, frame: Mapping[str, Any]) -> dict[str, Any]:
        import torch

        result = super().step(frame)
        hard = result["disparity"]
        disparity, _ = lower_guard_endpoint(frame["raw"], hard, result["support"])
        finite = result["support"] & torch.isfinite(frame["raw"]) & torch.isfinite(disparity)
        update = float((disparity - frame["raw"]).abs()[finite].mean()) if bool(finite.any()) else 0.0
        updated = dict(result, disparity=disparity, hard_disparity=hard)
        updated["diagnostics"] = dict(result["diagnostics"], update_magnitude=update)
        return updated


def factory(**kwargs: Any) -> LowerGuardH4:
    return LowerGuardH4(**kwargs)
