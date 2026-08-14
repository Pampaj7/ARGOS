#!/usr/bin/env python3
"""Compile safety scalars from completed definitive JSON reports; no inference."""
from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


METRICS = ("HUR", "HPlus", "BPlus", "BUR")
FRAME = ("P95", "P99", "Worst", "PositiveFraction")


def _leaf(value: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    for key in keys: value = value[key]
    return value


def _mean(items: Iterable[Mapping[str, Any]]) -> dict[str, float | None]:
    items = list(items)
    values = [float(item["macro_sequence"]) for item in items if item.get("macro_sequence") is not None]
    micros = [float(item["micro_pixel"]) * int(item["support_count"]) for item in items if item.get("micro_pixel") is not None]
    counts = [int(item["support_count"]) for item in items if item.get("micro_pixel") is not None]
    return {"macro_sequence": sum(values) / len(values) if values else None,
            "micro_pixel": sum(micros) / sum(counts) if counts else None,
            "support_count": sum(counts), "sequence_count": len(items)}


def extract(reports: Iterable[Path]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for path in reports:
        report = json.loads(path.read_text())
        method = str(report.get("method", "canonical_h4"))
        key = (report["dataset"], report["split"], report["backbone"], method)
        grouped.setdefault(key, []).append(report)
    rows: list[dict[str, Any]] = []
    for (dataset, split, backbone, method), values in sorted(grouped.items()):
        leaves: dict[str, list[Mapping[str, Any]]] = {"raw_EPE": [], "refined_EPE": []}
        for metric in METRICS: leaves[metric] = []
        for threshold in ("1.0", "3.0", "5.0"): leaves[f"NewBad{threshold[:-2]}"] = []
        for metric in FRAME: leaves[f"frame_degradation_{metric}"] = []
        for report in values:
            aggregate = report["aggregate"]["disparity_px"]
            leaves["raw_EPE"].append(_leaf(aggregate, "raw", "MAE")); leaves["refined_EPE"].append(_leaf(aggregate, "refined", "MAE"))
            safety = report["safety"]["disparity_px"]["aggregate"]
            for metric in METRICS: leaves[metric].append(safety[metric])
            for threshold in ("1.0", "3.0", "5.0"): leaves[f"NewBad{threshold[:-2]}"] .append(safety["thresholds"][threshold]["NewBad"])
            for metric in FRAME: leaves[f"frame_degradation_{metric}"].append(safety["FrameDegradation"][metric])
        aggregates = {name: _mean(items) for name, items in leaves.items()}
        for aggregate_name in ("macro_sequence", "micro_pixel"):
            for name, value in aggregates.items():
                rows.append({"dataset": dataset, "split": split, "backbone": backbone, "method": method, "aggregate": aggregate_name,
                             "metric": name, "value": value[aggregate_name], "support_count": value["support_count"], "sequence_count": value["sequence_count"]})
            bplus, hplus = aggregates["BPlus"][aggregate_name], aggregates["HPlus"][aggregate_name]
            rows.append({"dataset": dataset, "split": split, "backbone": backbone, "method": method, "aggregate": aggregate_name,
                         "metric": "BPlus_over_HPlus", "value": None if not hplus else bplus / hplus,
                         "support_count": aggregates["HPlus"]["support_count"], "sequence_count": aggregates["HPlus"]["sequence_count"]})
    return rows


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists(): raise FileExistsError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", newline="", delete=False, dir=path.parent, prefix=f".{path.name}.") as stream:
            temporary = Path(stream.name); writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        temporary.replace(path)
    except BaseException:
        if temporary: temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="completed definitive source root containing runs/")
    parser.add_argument("--split", choices=("d2", "d7"), required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = sorted((args.source / "runs" / f"scared-{args.split}").glob("*/reports/*/*.json"))
    if not reports:
        reports = sorted((args.source / "reports").glob("*/*/*.json"))
        reports = [path for path in reports if json.loads(path.read_text()).get("split") == args.split]
    rows = extract(reports)
    if not rows: raise RuntimeError("no completed definitive report JSON files")
    atomic_csv(args.output, rows)


if __name__ == "__main__":
    main()
