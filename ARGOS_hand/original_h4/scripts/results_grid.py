#!/usr/bin/env python3
"""The backbone x arena grid: one frozen checkpoint, five estimators, three datasets.

The paper used to carry this evidence as two tables -- absolute SCARED-C numbers per
backbone, and a 2x2 of mean EPE reductions -- where the second was the first's marginals
with DRENDS bolted on. This emits the merged table instead, so the axis that matters
(unseen backbone x unseen domain) is a cell you can point at rather than an average.

Every number is read from the run reports, never transcribed. Reductions are the mean over
sequences of the per-sequence reduction, which is the aggregation the 2x2 used; taking the
reduction of the means instead moves cells by up to half a point and would silently
disagree with the text that cites them.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[2] / "results"
SCARED = RESULTS / "scared_masked/runs/{split}/model_design_comparison_canonical_h4_masked__factory/reports/{backbone}"
DRENDS = RESULTS / "drends_masked/{backbone}/reports/{backbone}"
BACKBONES = (("S2M2-S", "seen"), ("RAFT-Stereo", "seen"), ("StereoAnywhere", "seen"),
             ("CREStereo", "unseen"), ("Fast-FoundationStereo", "unseen"))
LABEL = {"S2M2-S": "S$^2$M$^2$-S", "StereoAnywhere": "Stereo Anywhere",
         "Fast-FoundationStereo": "Fast-FoundationStereo"}


def arena(pattern: str, backbone: str, **kwargs) -> dict[str, np.ndarray] | None:
    """Per-sequence raw and refined EPE/Bad1 for one backbone in one arena, or None.

    A missing directory is a run that has not happened, which the table shows as a gap
    rather than an omission; an empty one is the same thing mid-flight.
    """
    files = sorted(glob.glob(os.path.join(pattern.format(backbone=backbone, **kwargs), "*.json")))
    if not files:
        return None
    out: dict[str, list[float]] = {key: [] for key in ("raw_EPE", "ref_EPE", "raw_Bad1", "ref_Bad1")}
    for path in files:
        agg = json.load(open(path))["aggregate"]["disparity_px"]
        for source, prefix in (("raw", "raw"), ("refined", "ref")):
            for metric in ("EPE", "Bad1"):
                out[f"{prefix}_{metric}"].append(agg[source][metric]["macro_sequence"])
    return {key: np.asarray(value) for key, value in out.items()}


def cells(data: dict[str, np.ndarray] | None, metric: str, scale: float, digits: int) -> tuple[str, str]:
    if data is None:
        return "---", "---"
    raw, ref = data[f"raw_{metric}"] * scale, data[f"ref_{metric}"] * scale
    pair = f"${raw.mean():.{digits}f}\\rightarrow\\mathbf{{{ref.mean():.{digits}f}}}$"
    return pair, f"{(100 * (raw - ref) / raw).mean():.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--latex", action="store_true", help="emit the table body only")
    args = parser.parse_args()

    rows, reductions = [], {}
    for backbone, status in BACKBONES:
        arenas = [arena(str(SCARED), backbone, split="scared-d2"),
                  arena(str(SCARED), backbone, split="scared-d7"),
                  arena(str(DRENDS), backbone)]
        columns, per_arena = [], []
        for data in arenas:
            epe, epe_red = cells(data, "EPE", 1.0, 4)
            bad, bad_red = cells(data, "Bad1", 100.0, 2)
            columns += [epe.replace("0.", "."), bad]
            per_arena.append((epe_red, bad_red))
        reductions[backbone] = per_arena
        rows.append(f"{LABEL.get(backbone, backbone)} & {status} & " + " & ".join(columns) + r" \\")

    if args.latex:
        print("\n".join(rows))
        return
    for backbone, status in BACKBONES:
        red = reductions[backbone]
        print(f"{backbone:22s} {status:6s} EPE red D2/D7/DRENDS: "
              + "  ".join(f"{value[0]:>6}" for value in red)
              + "   Bad1 red: " + "  ".join(f"{value[1]:>6}" for value in red))
    for name, index in (("D2", 0), ("D7", 1), ("DRENDS", 2)):
        for status in ("seen", "unseen"):
            values = [float(reductions[b][index][0]) for b, s in BACKBONES
                      if s == status and reductions[b][index][0] != "---"]
            got = f"{np.mean(values):.2f}" if values else "---"
            print(f"  mean EPE reduction, {name:6s} {status:6s} backbone: {got}  (n={len(values)})")


if __name__ == "__main__":
    main()
