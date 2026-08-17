#!/usr/bin/env python3
"""Sequence-level paired bootstrap of the A2-minus-canonical difference on DRENDS.

The seed-0 comparison puts A2 ahead on eleven of twelve backbone-metric pairs by margins
between 0.000 and 0.005. Eleven of twelve looks like a pattern; margins of 0.003 look like
nothing. Counting how often A2 wins cannot settle that, because the twelve comparisons
share five sequences and are not independent draws.

This resamples the SEQUENCES, which is the unit that actually varies, and keeps every
backbone of a resampled sequence together so the correlation between backbones on the
same tissue is preserved rather than averaged away. The paired difference is formed per
sequence before resampling, so the large variation in sequence difficulty -- the thing
that manufactured a threefold phantom improvement this morning -- cancels exactly.

Reports the mean difference and a percentile interval. With five sequences the interval
is wide by construction, and saying so is the point: it is the honest width, not a
failure of the method.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT.parent / "results"
CANONICAL = RESULTS / "drends_masked"
A2 = RESULTS / "drends_a2"
BACKBONES = ("CREStereo", "Fast-FoundationStereo", "RAFT-Stereo")
METRICS = ("EPE", "Bad1", "Bad3", "RMSE")


def sequences(root: Path, backbone: str) -> dict[str, dict]:
    out = {}
    for path in glob.glob(str(root / backbone / "reports/*/*.json")):
        out[Path(path).stem] = json.loads(Path(path).read_text())["aggregate"]["disparity_px"]
    return out


def paired(metric: str, seeds: list[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Per-sequence canonical and A2 values, averaged over seeds, aligned per backbone.

    Returns the sequence names and two arrays of shape [sequences, backbones]. Only
    sequences present for the canonical run and for every seed of every backbone are
    used, so no cell is filled from a different set than its counterpart.
    """
    base = {b: sequences(CANONICAL, b) for b in BACKBONES}
    variant = {(b, s): sequences(A2 / s, b) for b in BACKBONES for s in seeds}
    names = set.intersection(*(set(v) for v in base.values()),
                             *(set(v) for v in variant.values()))
    names = sorted(names)
    if not names:
        raise SystemExit("no sequence is covered by the canonical run and every requested seed")

    canonical = np.array([[base[b][n]["refined"][metric]["macro_sequence"] for b in BACKBONES]
                          for n in names])
    refined = np.array([[np.mean([variant[(b, s)][n]["refined"][metric]["macro_sequence"]
                                  for s in seeds]) for b in BACKBONES] for n in names])
    raw = np.array([[base[b][n]["raw"][metric]["macro_sequence"] for b in BACKBONES]
                    for n in names])
    for index, name in enumerate(names):
        for column, backbone in enumerate(BACKBONES):
            for seed in seeds:
                other = variant[(backbone, seed)][name]["raw"][metric]["macro_sequence"]
                if abs(other - raw[index, column]) > 1e-6:
                    raise RuntimeError(f"{backbone}/{seed}/{name}: raw predictions differ "
                                       "from canonical; the pairing is invalid")
    return names, canonical, refined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", default=None,
                        help="A2 seed directories to average; defaults to every complete one")
    args = parser.parse_args()

    available = sorted(p.name for p in A2.glob("seed*") if p.is_dir())
    complete = [s for s in available
                if all(len(sequences(A2 / s, b)) == len(sequences(CANONICAL, b))
                       for b in BACKBONES)]
    seeds = args.seeds or complete
    if not seeds:
        raise SystemExit(f"no A2 seed covers every sequence yet (present: {available})")
    print(f"A2 seeds present {available}, complete {complete}, using {seeds}\n")

    generator = np.random.default_rng(args.seed)
    print(f"{'metric':<7}{'mean diff':>11}{'95% interval':>24}{'P(A2 better)':>14}  verdict")
    for metric in METRICS:
        names, canonical, refined = paired(metric, seeds)
        difference = refined - canonical                  # negative favours A2
        per_sequence = difference.mean(axis=1)            # average over backbones, keep sequences
        index = generator.integers(0, len(names), size=(args.resamples, len(names)))
        draws = per_sequence[index].mean(axis=1)
        low, high = np.percentile(draws, [100 * args.alpha / 2, 100 * (1 - args.alpha / 2)])
        crosses = low <= 0.0 <= high
        print(f"{metric:<7}{per_sequence.mean():>11.4f}{f'[{low:+.4f}, {high:+.4f}]':>24}"
              f"{float((draws < 0).mean()):>14.3f}  "
              f"{'includes zero' if crosses else 'excludes zero'}")

    print(f"\n{len(names)} sequences x {len(BACKBONES)} backbones, {args.resamples} resamples.")
    print("Differences are A2 minus canonical, so negative favours A2. Sequences are the\n"
          "resampled unit; backbones move together within a sequence.")
    # An interval that excludes zero at a magnitude of 0.004 px is a statement about
    # detectability, not importance, and it is conditioned on the seeds actually pooled.
    print(f"\nThe interval covers variation ACROSS SEQUENCES ONLY. The canonical side is a\n"
          f"single training run and the A2 side pools {len(seeds)} "
          f"({', '.join(seeds)}), so seed variance is\nnot in these intervals. A2's own "
          "seed-to-seed spread on DRENDS is of the same order as\nthe differences above, "
          "which is why the honest reading stays 'detectable and negligible'\nrather than "
          "'better'.")


if __name__ == "__main__":
    main()
