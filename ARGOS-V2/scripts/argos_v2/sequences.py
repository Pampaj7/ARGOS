"""Accepted SCARED-C sequence list (17, quality-gate pass) + representative-sequence picks."""
from __future__ import annotations

import csv

from argos_v2.paths import QUALITY_GATE_CSV


def load_quality_gate_rows() -> list[dict]:
    return list(csv.DictReader(QUALITY_GATE_CSV.open()))


def accepted_sequences() -> list[str]:
    rows = load_quality_gate_rows()
    return [r["sequence_id"] for r in rows if r["status"] == "pass"]


def representative_sequences() -> dict[str, str]:
    """Pick 4 sequences for final contact sheets: easy, difficult, boundary-heavy, low-coverage.
    Computed from quality_gate.csv fields, not hand-picked, so it stays correct if the gate reruns.
    """
    rows = [r for r in load_quality_gate_rows() if r["status"] == "pass"]
    easy = min(rows, key=lambda r: float(r["photometric_mae_median"]))
    difficult = max(rows, key=lambda r: float(r["photometric_mae_median"]))
    low_coverage = min(rows, key=lambda r: float(r["valid_pixel_ratio_mean"]))
    boundary_heavy = max(rows, key=lambda r: float(r["photometric_mae_max"]) - float(r["photometric_mae_median"]))
    return {
        "easy": easy["sequence_id"],
        "difficult": difficult["sequence_id"],
        "low_coverage": low_coverage["sequence_id"],
        "boundary_heavy": boundary_heavy["sequence_id"],
    }
