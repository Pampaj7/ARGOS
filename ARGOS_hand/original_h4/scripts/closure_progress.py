#!/usr/bin/env python3
"""Compare a promoted head's closure against the canonical one, on matched cells only.

A partial run must never be averaged against a complete one. Cells finish in a fixed
backbone and sequence order, so the first ones done are always the same ones and are not a
random sample: at one point tonight the four finished H=8 cells were worth 1.69 points more
to the canonical head than its own fifteen-cell average, which invented a horizon-dependent
trend that did not exist.

So this scores the reference on exactly the subset the new run has finished, prints that
subset bias next to every comparison, and refuses to call a method comparable until its
cells are complete.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[2] / "results/definitive_evaluation/experimental_closure"
CANONICAL = BASE / "d2"


def reduction(path: str) -> tuple[float, float]:
    d = json.load(open(path))["aggregate"]["disparity_px"]
    raw, ref = d["raw"]["EPE"]["macro_sequence"], d["refined"]["EPE"]["macro_sequence"]
    return raw, 100.0 * (raw - ref) / raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--head", default="A2_no_learned_evidence")
    args = parser.parse_args()

    methods = sorted({Path(f).parents[1].name
                      for f in glob.glob(str(BASE / f"d2_{args.head}*/reports/*/*/*.json"))})
    print(f"{'method':16s}{'n':>4s}{'canon(sub)':>12s}{'canon(all)':>12s}{'bias':>8s}"
          f"{'head':>9s}{'delta':>9s}  status")
    for method in methods:
        done, head = {}, {}
        for f in glob.glob(str(BASE / f"d2_{args.head}*/reports/{method}/*/*.json")):
            key = f"{Path(f).parent.name}/{Path(f).name}"
            raw, red = reduction(f)
            head[key] = red
            done[key] = raw
        sub, full = [], []
        for f in sorted(glob.glob(str(CANONICAL / f"reports/{method}/*/*.json"))):
            key = f"{Path(f).parent.name}/{Path(f).name}"
            raw, red = reduction(f)
            full.append(red)
            if key in done:
                assert abs(raw - done[key]) < 1e-9, f"raw differs on {method}/{key}"
                sub.append(red)
        if not sub:
            continue
        bias = np.mean(sub) - np.mean(full)
        complete = len(sub) == len(full)
        print(f"{method:16s}{len(sub):4d}{np.mean(sub):12.2f}{np.mean(full):12.2f}{bias:+8.2f}"
              f"{np.mean(list(head.values())):9.2f}{np.mean(list(head.values())) - np.mean(sub):+9.2f}"
              f"  {'complete' if complete else f'PARTIAL {len(sub)}/{len(full)} -- not comparable'}")


if __name__ == "__main__":
    main()
