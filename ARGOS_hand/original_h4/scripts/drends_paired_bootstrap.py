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


def bootstrap(per_sequence: np.ndarray, resamples: int, alpha: float,
              generator: np.random.Generator) -> tuple[float, float, float, float]:
    """Percentile bootstrap over sequences. Returns mean, low, high, P(difference < 0)."""
    index = generator.integers(0, len(per_sequence), size=(resamples, len(per_sequence)))
    draws = per_sequence[index].mean(axis=1)
    low, high = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(per_sequence.mean()), float(low), float(high), float((draws < 0).mean())


def self_check() -> None:
    """Synthetic check of the one property the whole conclusion rests on.

    Sequences differ enormously in difficulty here (0.5 to 50) while the method effect is
    a constant -0.01 on every one. If the pairing works, the recovered mean is the effect
    and the interval is narrow; if the difference were formed after averaging over
    different sequence sets -- the error that invented a threefold improvement on the real
    data -- the difficulty would dominate and the effect would be unrecoverable.
    """
    generator = np.random.default_rng(0)
    difficulty = np.array([0.5, 2.0, 8.0, 20.0, 50.0])[:, None]     # [sequences, 1]
    effect = -0.01
    canonical = difficulty + generator.normal(0, 0.001, size=(5, 3))
    refined = canonical + effect
    mean, low, high, probability = bootstrap((refined - canonical).mean(axis=1), 20000, 0.05, generator)
    assert abs(mean - effect) < 1e-9, mean
    assert high - low < 1e-4, (low, high)          # difficulty cancels, so the interval is tight
    assert high < 0.0 and probability == 1.0, (high, probability)

    # The converse is a CALIBRATION property, not a single-draw one. A 95% interval
    # excludes zero on 5% of null samples by definition, so asserting that one null draw
    # includes zero is a coin flip dressed as a test -- the first version of this check
    # asserted exactly that and failed on a 3-sigma draw. What must hold is the rate.
    excluded = 0
    trials = 300
    for _ in range(trials):
        null = generator.normal(0, 0.01, size=(5, 3)).mean(axis=1)
        _mean, low, high, _p = bootstrap(null, 2000, 0.05, generator)
        excluded += not (low <= 0.0 <= high)
    rate = excluded / trials
    # Wide bounds on purpose: with five sequences the percentile bootstrap is known to
    # under-cover, so this asserts "not badly broken", not "exact".
    assert 0.01 <= rate <= 0.25, rate

    # Unpaired reduction over unequal sequence sets is what the morning's bug did: compare
    # a mean over the easiest sequence against a mean over all five and a difference
    # appears from nothing. This asserts the size of that trap rather than trusting it.
    phantom = canonical[0].mean() - canonical.mean()
    assert phantom < 10 * effect, phantom
    print(f"OK: pairing recovers {effect} exactly; null exclusion rate {rate:.3f} against a "
          f"nominal 0.05 over {trials} trials;\n    unpaired subsetting would have shown "
          f"{phantom:+.2f} from no effect at all")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-check", action="store_true", help="synthetic test, touches no results")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", default=None,
                        help="A2 seed directories to average; defaults to every complete one")
    args = parser.parse_args()
    if args.self_check:
        return self_check()

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
        mean, low, high, probability = bootstrap(per_sequence, args.resamples, args.alpha, generator)
        crosses = low <= 0.0 <= high
        print(f"{metric:<7}{mean:>11.4f}{f'[{low:+.4f}, {high:+.4f}]':>24}"
              f"{probability:>14.3f}  "
              f"{'includes zero' if crosses else 'excludes zero'}")

    print(f"\n{len(names)} sequences x {len(BACKBONES)} backbones, {args.resamples} resamples.")
    print("Differences are A2 minus canonical, so negative favours A2. Sequences are the\n"
          "resampled unit; backbones move together within a sequence.")
    # An interval that excludes zero at a magnitude of 0.004 px is a statement about
    # detectability, not importance, and it is conditioned on the seeds actually pooled.
    # Measured, not assumed: --self-check runs 300 null trials at this sample size and
    # finds the interval excludes zero about 23% of the time rather than 5%. With five
    # sequences the percentile bootstrap is badly anti-conservative, so "excludes zero"
    # here is weak evidence and must not be read as a nominal 95% statement.
    print(f"\nCALIBRATION: at {len(names)} sequences this interval is anti-conservative. Run\n"
          "--self-check: under a true null it excludes zero about 23% of the time, not 5%.\n"
          "'Excludes zero' is therefore weak evidence at this sample size.")
    print(f"\nThe interval covers variation ACROSS SEQUENCES ONLY. The canonical side is a\n"
          f"single training run and the A2 side pools {len(seeds)} "
          f"({', '.join(seeds)}), so seed variance is\nnot in these intervals. A2's own "
          "seed-to-seed spread on DRENDS is of the same order as\nthe differences above, "
          "which is why the honest reading stays 'detectable and negligible'\nrather than "
          "'better'.")


if __name__ == "__main__":
    main()
