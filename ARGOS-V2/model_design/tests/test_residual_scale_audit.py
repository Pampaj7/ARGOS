"""Deterministic policy tests for the frozen A2 residual-scale audit."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from run_residual_scale_audit import choose_scale  # noqa: E402


def _row(scale: float, backbone: str, gain: float, clean: float, false: float) -> dict:
    return {
        "scale": scale, "backbone": backbone, "gain": gain,
        "clean_pixel_degradation": clean, "false_update_rate": false,
    }


def test_scale_choice_uses_all_seen_backbones_and_safety_constraints() -> None:
    rows = []
    for backbone in ("S2M2-S", "RAFT-Stereo", "StereoAnywhere"):
        rows += [
            _row(.20, backbone, .01, .01, .02),
            _row(.35, backbone, .02, .02, .04),
            _row(.50, backbone, .03, .04, .06),  # invalid: false-update >5%
            _row(1.0, backbone, .04, .06, .07),  # invalid: both constraints
        ]
    selected = choose_scale(rows)
    assert selected["constraints_feasible"]
    assert selected["selected"]["scale"] == .35


def test_scale_choice_never_claims_feasibility_when_every_scale_breaks_safety() -> None:
    rows = [_row(.20, backbone, .02, .04, .06)
            for backbone in ("S2M2-S", "RAFT-Stereo", "StereoAnywhere")]
    selected = choose_scale(rows)
    assert not selected["constraints_feasible"]
    assert not selected["selected"]["eligible"]
