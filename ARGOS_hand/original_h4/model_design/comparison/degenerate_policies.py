"""Degenerate temporal baselines for the no-reference D4D control.

A no-reference temporal-consistency score can be driven to zero by a policy that simply
copies history, so the score is only interpretable next to policies that do exactly that.
Two are provided:

``copy_previous``
    Emits the previous frame's disparity verbatim, with no motion compensation at all.
    This is the extreme of temporal smoothness and the strongest possible no-reference
    score a stale predictor can obtain.

``warped_previous``
    Emits the previous *raw* disparity after motion compensation, i.e. the aligned memory
    used at full weight. This is `warped_raw_previous_h1` from ``experimental_policies``,
    re-exported here so both controls live in one place; that pinned module is imported
    unchanged rather than edited.

Neither is a method. They exist to bound the no-reference metric from the degenerate side.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from model_design.comparison.experimental_policies import ExperimentalPolicy, POLICIES


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CopyPrevious:
    """Verbatim previous-frame disparity: no alignment, no fusion, no learning."""

    horizon = 1

    def __init__(self, *, device: str = "cuda:0") -> None:
        self.device = device

    def describe(self) -> dict[str, Any]:
        here = Path(__file__).resolve()
        return {"module": "copy_previous", "kind": "degenerate_no_reference_control",
                "code": str(here), "code_sha256": _sha256(here),
                "motion_compensation": False, "learned_parameters": 0,
                "reset_protocol": "H=1; the state is always the previous raw frame",
                "caveat": "not a method; bounds the no-reference metric from the degenerate side"}

    def start(self, frame: Mapping[str, Any]) -> dict[str, Any]:
        return {"disparity": frame["raw"], "support": frame["raw_valid"].bool(),
                "reset": True, "state_age": 0,
                "diagnostics": {"method": "copy_previous", "update_magnitude": 0.0}}

    def step(self, frame: Mapping[str, Any]) -> dict[str, Any]:
        import torch
        if frame["horizon"] != 1 or not frame["reanchor"]:
            raise RuntimeError("copy_previous requires an H=1 raw re-anchor on every step")
        with torch.inference_mode():
            past = frame["past_disparity"]
            support = frame["raw_valid"].bool() & frame["past_valid"].bool()
            fused = torch.where(support, past, frame["raw"])
            update = float((fused - frame["raw"]).abs()[support].mean()) if bool(support.any()) else 0.0
        return {"disparity": fused, "support": support, "reset": bool(frame["reanchor"]),
                "state_age": int(frame["state_age"]),
                "diagnostics": {"method": "copy_previous", "horizon": 1,
                                "update_magnitude": update, "temporal_weight": 1.0}}


def factory(*, method: str = "copy_previous", device: str = "cuda:0", **_: Any):
    if method == "copy_previous":
        return CopyPrevious(device=device)
    if method in ("warped_previous", "warped_raw_previous_h1"):
        return ExperimentalPolicy(POLICIES["warped_raw_previous_h1"], device=device)
    raise ValueError(f"unknown degenerate policy: {method}")


def factory_warped(*, device: str = "cuda:0", **_: Any):
    """Separate entry point so the driver's ``module:function`` spec can select it."""
    return ExperimentalPolicy(POLICIES["warped_raw_previous_h1"], device=device)
