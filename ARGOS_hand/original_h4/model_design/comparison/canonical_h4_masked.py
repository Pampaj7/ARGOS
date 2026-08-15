"""Canonical H=4 with the support mask applied to its own output.

`canonical_h4.py` is pinned by 55 manifests and freeze records, so it is subclassed here
rather than edited.  The only difference is one line that every matched baseline in
`experimental_policies.py` already applies:

    fused = torch.where(support, fused, raw)

`CODDStyleFusionHead.forward` computes `(1-w)*raw + w*aligned_memory` over every pixel, and
`canonical_h4.step` returns it unmasked while reporting the support it never uses.  Where the
causal warp samples out of bounds, `padding_mode="zeros"` makes the aligned memory exactly 0.0;
if the head puts weight there the fused disparity collapses to 0, becomes an invalid
prediction, and — because `drive()` feeds `result["disparity"]` back as the next state —
propagates through the bounded recurrence until the next raw re-anchor.

Measured on DRENDS: 300 such pixels in Vid10_Liver_Med and 265 in Vid11_Liver_High, with the
aligned memory of every single one equal to 0.0.  Each is charged
`MetricConfig.invalid_penalty_mm = 10000.0`.

This is an ablation, not a redefinition of the frozen module: the checkpoint, the head, the
reset protocol and the evidence are untouched, and the two variants can be run side by side.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from model_design.comparison.canonical_h4 import CanonicalH4


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MaskedCanonicalH4(CanonicalH4):
    """Frozen canonical H4 whose output is confined to its declared support."""

    def describe(self) -> dict[str, Any]:
        parent = super().describe()
        here = Path(__file__).resolve()
        return parent | {
            "module": "canonical_h4_masked",
            "variant_of": parent.get("module", "canonical_h4"),
            "variant_code": str(here),
            "variant_code_sha256": _sha256(here),
            "variant_change": "fused = where(support, fused, raw); matches experimental_policies.py",
            "checkpoint_modified": False,
            "head_modified": False,
        }

    def step(self, frame: Mapping[str, Any]) -> dict[str, Any]:
        import torch
        result = super().step(frame)
        raw = frame["raw"]
        support = result["support"].bool()
        with torch.inference_mode():
            masked = torch.where(support, result["disparity"], raw)
            update = float((masked - raw).abs()[support].mean()) if bool(support.any()) else 0.0
        result["disparity"] = masked
        result["diagnostics"] = dict(result["diagnostics"]) | {
            "update_magnitude": update,
            "masked_to_support": True,
        }
        return result


def factory(**kwargs: Any) -> MaskedCanonicalH4:
    return MaskedCanonicalH4(**kwargs)
