"""Frozen canonical head with an inference-policy horizon override only."""
from __future__ import annotations

from model_design.comparison.canonical_h4 import CanonicalH4


class CanonicalHorizon(CanonicalH4):
    def __init__(self, *, horizon: int | None = 4, **kwargs) -> None:
        if horizon not in {1, 2, 4, 6, 8, None}:
            raise ValueError("canonical horizon must be one of 1,2,4,6,8,None")
        super().__init__(**kwargs); self.horizon = horizon

    def describe(self) -> dict[str, Any]:
        return super().describe() | {"module": "canonical_horizon", "horizon": self.horizon,
                                     "reset_protocol": f"frozen head; inference horizon={self.horizon}"}
