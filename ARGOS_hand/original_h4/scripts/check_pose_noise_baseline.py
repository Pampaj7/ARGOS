#!/usr/bin/env python3
"""The pose-noise runs must reproduce the published map numbers at zero noise.

`world_frame_drift.score` was restructured so the camera-frame clouds are built once and
re-transformed under each perturbed pose set. That restructuring is exactly the kind that
can silently change a number -- a support list consumed twice, a cloud ordering lost -- so
the zero-noise column of each new run is compared against the file the paper quotes.

Anything above a nanometre of disagreement means the refactor changed the measurement and
the sweep says nothing about the published result.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "results"
BASELINE = RESULTS / "world_frame_drift"
SWEEP = RESULTS / "world_frame_pose_noise"
TOLERANCE_MM = 1e-9
TOLERANCE_PCT = 1e-6


def main() -> int:
    runs = sorted(SWEEP.glob("dataset_*.json"))
    if not runs:
        print(f"no sweep runs under {SWEEP}")
        return 1
    failures = 0
    for run in runs:
        new = json.loads(run.read_text())
        old_path = BASELINE / f"{run.stem}_a2.json"
        if not old_path.is_file():
            print(f"{run.stem}: NO BASELINE at {old_path}")
            failures += 1
            continue
        old = json.loads(old_path.read_text())
        if old["frames"] != new["frames"] or old["backbone"] != new["backbone"]:
            print(f"{run.stem}: not comparable "
                  f"({old['backbone']}/{old['frames']} vs {new['backbone']}/{new['frames']})")
            failures += 1
            continue
        bad = []
        for cloud in ("gt", "raw", "refined"):
            for key in ("median_mm", "mean_mm", "p95_mm"):
                delta = abs(old["results"][cloud][key] - new["results"][cloud][key])
                if delta > TOLERANCE_MM:
                    bad.append(f"{cloud}.{key} differs by {delta:.3e}")
            if old["results"][cloud]["pixels"] != new["results"][cloud]["pixels"]:
                bad.append(f"{cloud}.pixels {old['results'][cloud]['pixels']} -> "
                           f"{new['results'][cloud]['pixels']}")
        delta = abs(old["excess_reduction_pct"] - new["excess_reduction_pct"])
        if delta > TOLERANCE_PCT:
            bad.append(f"excess_reduction_pct differs by {delta:.3e}")
        if bad:
            failures += 1
            print(f"{run.stem}: MISMATCH")
            for line in bad:
                print(f"    {line}")
        else:
            print(f"{run.stem}: reproduces baseline "
                  f"({new['excess_reduction_pct']:+.3f}%)")
    print(f"\n{len(runs) - failures}/{len(runs)} runs reproduce the published zero-noise result")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
