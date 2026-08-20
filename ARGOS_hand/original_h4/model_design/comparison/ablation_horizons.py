"""A pre-registered ablation head under the canonical inference-horizon override.

This lives beside `canonical_horizons.py` rather than inside it because that file's
sha256 is pinned by the closure's freeze manifest: adding a class to it made the protocol
refuse to run, which is the freeze doing its job. The frozen file stays byte-identical and
the new head arrives in its own module.
"""
from __future__ import annotations

from typing import Any

from model_design.comparison.ablation_h4 import AblationH4


class AblationHorizon(AblationH4):
    """The learned rows of the closure, run with a promoted variant instead of canonical.

    The closure's fifteen baseline policies blend raw against flow-aligned raw with a
    fixed, EMA or confidence weight and load no checkpoint at all, so they are identical
    whichever head the paper ships, and the raw-versus-memory oracle is GT-only. Re-running
    the closure for a promoted variant therefore re-runs six rows, not twenty-one.
    """

    def __init__(self, *, horizon: int | None = 4, variant: str, seed: int | None = None,
                 **kwargs: Any) -> None:
        if horizon not in {1, 2, 4, 6, 8, None}:
            raise ValueError("horizon must be one of 1,2,4,6,8,None")
        super().__init__(variant=variant, seed=seed, **kwargs)
        self.horizon = horizon

    def describe(self) -> dict[str, Any]:
        return super().describe() | {"module": "ablation_horizon", "horizon": self.horizon,
                                     "reset_protocol": f"frozen {self.variant} head; "
                                                       f"inference horizon={self.horizon}"}


def _shipped_at(horizon: int):
    """The shipped head driven at one horizon, addressable as `module:factory_a2_hN`.

    The world-frame measurement takes a `--module` and passes only `device`, so a sweep
    over the recurrence horizon needs one entry point per horizon. Nothing about the head
    changes: same checkpoint, same evidence, only the re-anchor schedule.
    """
    def _factory(*, device: str = "cuda:0", **_: Any) -> AblationHorizon:
        return AblationHorizon(horizon=horizon, variant="A2_no_learned_evidence", device=device)
    _factory.__name__ = f"factory_a2_h{horizon}"
    return _factory


for _h in (1, 2, 4, 6, 8):
    globals()[f"factory_a2_h{_h}"] = _shipped_at(_h)
