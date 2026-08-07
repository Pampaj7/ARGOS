#!/usr/bin/env python3
"""Aggregate compact ARGOS v2 per-run metric JSON files."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = (ROOT / "experiments").resolve()


def inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent)
        return True
    except ValueError:
        return False


def summarize(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in keys)].append(row)
    result = []
    for group, values in sorted(groups.items()):
        out = dict(zip(keys, group))
        numeric = sorted({key for row in values for key, value in row.items() if isinstance(value, (int, float)) and key not in keys})
        out["n"] = len(values)
        for key in numeric:
            data = [float(row[key]) for row in values if isinstance(row.get(key), (int, float))]
            out[f"{key}_mean"] = statistics.fmean(data)
            out[f"{key}_std"] = statistics.stdev(data) if len(data) > 1 else 0.0
        result.append(out)
    return result


def write_csv(path: Path, rows: list[dict], fallback_fields: list[str]) -> None:
    fields = sorted({key for row in rows for key in row}) or fallback_fields
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    args = parser.parse_args()
    experiment = args.experiment_dir.resolve()
    if not inside(experiment, EXPERIMENTS):
        raise SystemExit("experiment directory must be under ARGOS_FREEZED/experiments")
    output = experiment / "results"; output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(output.glob("seed_*/metrics.json")):
        payload = json.loads(path.read_text())
        candidate = payload if isinstance(payload, list) else payload.get("rows", [payload])
        for row in candidate:
            rows.append({"seed": path.parent.name.removeprefix("seed_"), **row})
    per_seed = summarize(rows, ("seed",))
    per_backbone = summarize(rows, ("backbone",))
    per_sequence = summarize(rows, ("backbone", "sequence"))
    aggregate = summarize(rows, ())
    (output / "aggregate_summary.json").write_text(json.dumps({"project": "ARGOS v2", "rows": len(rows), "summary": aggregate}, indent=2, sort_keys=True) + "\n")
    write_csv(output / "per_seed_metrics.csv", per_seed, ["seed", "n"])
    write_csv(output / "per_backbone_metrics.csv", per_backbone, ["backbone", "n"])
    write_csv(output / "per_sequence_metrics.csv", per_sequence, ["backbone", "sequence", "n"])
    (output / "paper_ready_tables.tex").write_text("% ARGOS v2: no completed seed metrics yet.\n" if not rows else "% Generated from aggregate_summary.json; format after metrics are frozen.\n")
    (output / "README.md").write_text(f"# ARGOS v2 aggregate\n\nAggregated {len(rows)} compact metric rows. Mean and sample standard deviation are reported across available seeds.\n")


if __name__ == "__main__":
    main()
