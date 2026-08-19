#!/usr/bin/env python3
"""The unlearned residual rule against the learned head, on complete points only.

The review's question is whether a two-parameter spatially varying rule matches the
177k-parameter head. The answer is one number per (alpha, tau): pooled D2 EPE reduction,
aggregated exactly as the closure section does -- reduction of the pooled means, so it is
directly comparable to the 4.05% the paper reports.

Incomplete points are listed and excluded rather than averaged. A sweep point with ten of
fifteen cells is missing whole sequences, and the cells finish in a fixed order, so a
partial mean is biased in a direction that is known and not small: that mistake has been
made twice in this project already.
"""
from __future__ import annotations

import csv
import glob
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[2] / "results"
SWEEP = RESULTS / "residual_policy"
CLOSURE = RESULTS / "definitive_evaluation/experimental_closure/d2/summary.csv"


def pooled(rows: list[dict]) -> float:
    raw = np.mean([float(r["raw_epe_macro_sequence"]) for r in rows])
    cand = np.mean([float(r["candidate_epe_macro_sequence"]) for r in rows])
    return 100.0 * (raw - cand) / raw


def main() -> None:
    reference = [r for r in csv.DictReader(CLOSURE.open()) if r["method"] == "canonical_h4"]
    best_fixed = max(
        (m for m in {r["method"] for r in csv.DictReader(CLOSURE.open())} if m.startswith(("fixed", "ema"))),
        key=lambda m: pooled([r for r in csv.DictReader(CLOSURE.open()) if r["method"] == m]))
    fixed_rows = [r for r in csv.DictReader(CLOSURE.open()) if r["method"] == best_fixed]
    print(f"learned head (canonical_h4)      {pooled(reference):6.2f}%   n={len(reference)}")
    print(f"best fixed/EMA ({best_fixed:<15s}) {pooled(fixed_rows):6.2f}%   n={len(fixed_rows)}\n")

    incomplete = []
    for point in sorted(SWEEP.glob("a*_t*")):
        if ".incomplete-" in point.name:
            continue
        summary = point / "summary.csv"
        if not summary.is_file():
            incomplete.append(point.name)
            continue
        rows = list(csv.DictReader(summary.open()))
        if len(rows) != len(reference):
            incomplete.append(f"{point.name} ({len(rows)}/{len(reference)})")
            continue
        print(f"{point.name:20s} {pooled(rows):6.2f}%")
    if incomplete:
        print("\nnot comparable yet: " + ", ".join(incomplete))


if __name__ == "__main__":
    main()
