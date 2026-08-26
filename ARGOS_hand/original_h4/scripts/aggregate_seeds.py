#!/usr/bin/env python3
"""Seed variance for the canonical H=4 operating point, per `multiseed_preregister.json`.

Seed 0 is the canonical checkpoint already reported; seeds 1 and 2 were trained under the
pre-registered deviation (seed only) and evaluated through the identical frozen
support-masked inference path. This script does the reporting the pre-registration
committed to before any seed result existed:

  * mean and standard deviation of the relative Bad1 reduction across seeds 0, 1, 2;
  * a paired bootstrap confidence interval in which SEQUENCES, never pixels, are the
    resampling unit --- pixels inside one endoscopic sequence are not independent samples,
    and resampling them would manufacture significance.

The bootstrap is paired on the sequence: each resample draws the same sequences for raw
and refined, so it estimates the distribution of the *improvement*, not the difference of
two independently noisy means. Seeds are averaged inside each resample, which treats seed
choice as part of the method rather than as a free parameter --- consistent with the
pre-registered prohibition on selecting the best seed.

No threshold, policy or selection rule is touched here; this reads finished reports.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT.parent / "results"
PREREGISTER = ROOT / "model_design/multiseed_preregister.json"
OUT = RESULTS / "seed_variance"
METRICS = ("Bad1", "Bad3", "EPE")
PRIMARY = "Bad1"


def find_reports(run_root: Path) -> dict[tuple[str, str, str], Path]:
    """Map (dataset, backbone, sequence) -> report path for one evaluation run."""
    found = {}
    for report in run_root.rglob("reports/*/*.json"):
        dataset = next((p.name for p in report.parents if p.parent.name == "runs"), None)
        if dataset is None:
            continue
        found[(dataset, report.parent.name, report.stem)] = report
    return found


def read_pairs(reports: dict, metric: str) -> dict[tuple[str, str, str], tuple[float, float]]:
    """(raw, refined) macro-sequence values; each report covers exactly one sequence."""
    pairs = {}
    for key, path in reports.items():
        aggregate = json.loads(path.read_text())["aggregate"]["disparity_px"]
        try:
            pairs[key] = (aggregate["raw"][metric]["macro_sequence"],
                          aggregate["refined"][metric]["macro_sequence"])
        except KeyError:
            continue
    return pairs


def bootstrap(per_sequence: dict[str, list[tuple[float, float]]], resamples: int,
              rng: np.random.Generator) -> tuple[float, float, float, float]:
    """Paired sequence-level bootstrap of the seed-averaged relative reduction.

    `per_sequence[seq]` holds one (raw, refined) pair per seed. Each resample draws
    sequences with replacement, averages over seeds within a sequence, and pools the
    sequences with equal weight (macro over sequences, matching the primary aggregate).
    """
    names = sorted(per_sequence)
    raw = np.array([np.mean([p[0] for p in per_sequence[s]]) for s in names])
    refined = np.array([np.mean([p[1] for p in per_sequence[s]]) for s in names])
    point = 100.0 * (raw.mean() - refined.mean()) / raw.mean()
    draws = rng.integers(0, len(names), size=(resamples, len(names)))
    r, f = raw[draws].mean(axis=1), refined[draws].mean(axis=1)
    distribution = 100.0 * (r - f) / r
    low, high = np.percentile(distribution, [2.5, 97.5])
    # Fraction of resamples in which refinement did not help: a one-sided p-value.
    return point, float(low), float(high), float((distribution <= 0).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed-runs", nargs="+", required=True,
                        help="SEED=PATH, e.g. 0=/.../paper_d2_strict_all_anchors 1=/.../seed_1")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=OUT,
                        help="output directory (default: %(default)s); give a new one rather "
                             "than overwriting a finished seed study of a different model")
    args = parser.parse_args()
    out = args.out

    prohibitions = json.loads(PREREGISTER.read_text())["prohibited"]
    runs = {}
    for entry in args.seed_runs:
        seed, _, path = entry.partition("=")
        reports = find_reports(Path(path))
        if not reports:
            raise SystemExit(f"seed {seed}: no reports under {path}")
        runs[int(seed)] = reports
    seeds = sorted(runs)
    if len(seeds) < 2:
        raise SystemExit("seed variance needs at least two seeds")

    # Only keys present for EVERY seed are usable; a seed missing a sequence would
    # otherwise silently change the population being averaged.
    common = set.intersection(*(set(r) for r in runs.values()))
    dropped = {s: sorted(set(runs[s]) - common) for s in seeds}
    rng = np.random.default_rng(args.bootstrap_seed)

    rows, summary = [], {}
    for metric in METRICS:
        per_seed = {s: read_pairs({k: runs[s][k] for k in common}, metric) for s in seeds}
        usable = set.intersection(*(set(p) for p in per_seed.values()))
        groups = sorted({(d, b) for d, b, _ in usable})

        for dataset, backbone in groups + [("ALL", "ALL")]:
            selected = [k for k in usable if (dataset, backbone) == ("ALL", "ALL")
                        or (k[0], k[1]) == (dataset, backbone)]
            if not selected:
                continue
            # Per-seed macro-sequence reduction, for the mean/std the preregistration asks for.
            reductions = []
            for s in seeds:
                raw = np.mean([per_seed[s][k][0] for k in selected])
                refined = np.mean([per_seed[s][k][1] for k in selected])
                reductions.append(100.0 * (raw - refined) / raw)
            grouped: dict[str, list[tuple[float, float]]] = {}
            for k in selected:
                grouped.setdefault(f"{k[0]}/{k[1]}/{k[2]}", []).extend(per_seed[s][k] for s in seeds)
            point, low, high, p_null = bootstrap(grouped, args.resamples, rng)

            row = {"metric": metric, "dataset": dataset, "backbone": backbone,
                   "sequences": len(grouped), "seeds": len(seeds),
                   "reduction_mean_pct": float(np.mean(reductions)),
                   "reduction_std_pct": float(np.std(reductions, ddof=1)),
                   "reduction_min_pct": float(np.min(reductions)),
                   "reduction_max_pct": float(np.max(reductions)),
                   "bootstrap_point_pct": point, "ci95_low_pct": low, "ci95_high_pct": high,
                   "p_no_improvement": p_null}
            row |= {f"reduction_seed{s}_pct": reductions[i] for i, s in enumerate(seeds)}
            rows.append(row)
            if metric == PRIMARY:
                summary[f"{dataset}/{backbone}"] = row

    out.mkdir(parents=True, exist_ok=True)
    with (out / "seed_variance.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "run_manifest.json").write_text(json.dumps({
        "project": "ARGOS v2", "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "seed variance and paired sequence-level bootstrap for the canonical H=4 point",
        "preregistration": str(PREREGISTER), "prohibitions_honoured": prohibitions,
        "seed_runs": {str(s): args.seed_runs[i] for i, s in enumerate(seeds)},
        "primary_endpoint": PRIMARY,
        "resampling_unit": "sequence", "resamples": args.resamples,
        "bootstrap_rng_seed": args.bootstrap_seed,
        "sequences_common_to_all_seeds": len(common),
        "sequences_dropped_per_seed": {str(s): v for s, v in dropped.items()},
        "seed_handling": "seeds averaged inside each bootstrap resample; no seed selected",
    }, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "primary": PRIMARY,
                      "summary": {k: {m: v[m] for m in ("reduction_mean_pct", "reduction_std_pct",
                                                        "ci95_low_pct", "ci95_high_pct",
                                                        "p_no_improvement")}
                                  for k, v in summary.items()}}, indent=2))


if __name__ == "__main__":
    main()
