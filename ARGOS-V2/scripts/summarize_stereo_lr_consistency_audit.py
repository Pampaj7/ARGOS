#!/usr/bin/env python3
"""Compact sequence-unit summary for frozen ARGOS v2 LRC audit shards."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = (
    "raw_epe", "lrc_mean", "lrc_raw_error_pearson", "lrc_raw_bad1_auroc",
    "lrc_difference_utility_pearson", "lrc_memory_better_auroc",
)


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def weighted(rows: list[dict], name: str) -> float | None:
    valid = [row for row in rows if row.get(name) is not None and math.isfinite(float(row[name]))]
    if not valid:
        return None
    denom = sum(int(row["valid_pixels"]) for row in valid)
    return float(sum(float(row[name]) * int(row["valid_pixels"]) for row in valid) / max(denom, 1))


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    return value


def run(namespace: argparse.Namespace) -> None:
    paths = sorted(namespace.root.glob("*/*/aggregate_summary.json"))
    rows = []
    for path in paths:
        value = json.loads(path.read_text())
        if "lrc_memory_better_auroc" not in value:
            continue
        rows.append(value | {"path": str(path)})
    if not rows:
        raise RuntimeError("no completed --with-memory LRC shards found")
    write_csv(namespace.root / "per_combination.csv", rows)
    per_backbone = {}
    for backbone in sorted({row["backbone"] for row in rows}):
        group = [row for row in rows if row["backbone"] == backbone]
        per_backbone[backbone] = {
            "sequence_count": len(group), "valid_pixels": sum(int(row["valid_pixels"]) for row in group),
            **{name: weighted(group, name) for name in METRICS},
        }
    # Complete sequences, rather than pixels, are the independent unit.  Each
    # sequence first averages its three backbone values, then bootstrap
    # resampling creates a confidence interval over sequence means.
    per_sequence = defaultdict(list)
    for row in rows:
        per_sequence[row["sequence"]].append(row)
    rng = np.random.default_rng(20260719)
    bootstrap = {}
    for name in METRICS:
        values = np.asarray([
            np.mean([float(row[name]) for row in group if row.get(name) is not None])
            for group in per_sequence.values()
            if any(row.get(name) is not None for row in group)
        ], dtype=np.float64)
        if values.size:
            samples = np.asarray([rng.choice(values, size=values.size, replace=True).mean() for _ in range(10000)])
            bootstrap[name] = {"sequence_mean": float(values.mean()), "sequence_std": float(values.std(ddof=0)),
                               "ci95": [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))],
                               "independent_sequence_count": int(values.size)}
        else:
            bootstrap[name] = None
    overall = {"valid_pixels": sum(int(row["valid_pixels"]) for row in rows),
               "backbone_count": len(per_backbone), "sequence_count": len(per_sequence),
               **{name: weighted(rows, name) for name in METRICS}, "sequence_bootstrap": bootstrap}
    (namespace.root / "aggregate_summary.json").write_text(json.dumps(clean({"overall": overall, "per_backbone": per_backbone}), indent=2) + "\n")
    print(json.dumps(clean(overall), indent=2), flush=True)


if __name__ == "__main__":
    run(args())
