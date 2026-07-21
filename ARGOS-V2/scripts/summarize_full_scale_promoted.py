#!/usr/bin/env python3
"""Compact mean/std report for the architecture-frozen ARGOS v2 scale run."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "epe", "gain", "bad3", "boundary_epe", "false_update_rate",
    "clean_degradation", "intervention_coverage", "intervention_precision",
    "frames_worsened", "worst_frame_degradation", "runtime_ms_per_frame",
    "peak_gpu_memory_mb",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "results/full_scale_promoted")
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def aggregate_frames(path: Path, *, dataset: str, seed: int) -> dict:
    rows = [
        row for row in read_csv(path)
        if float(row["coverage_threshold"]) == 0.5
        and row["method"] == "authorized_balanced"
    ]
    if not rows:
        raise RuntimeError(f"no balanced primary rows in {path}")
    valid = sum(int(row["valid_count"]) for row in rows)
    clean = sum(int(row["clean_count"]) for row in rows)
    changed = sum(int(row["changed_count"]) for row in rows)
    weighted = lambda key: sum(float(row[key]) * int(row["valid_count"]) for row in rows) / max(valid, 1)
    degradation = np.asarray([float(row["refined_minus_raw_epe"]) for row in rows])
    runtime_path = path.parent / "runtime_summary.json"
    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
    epe = weighted("epe")
    raw_epe = weighted("raw_epe")
    return {
        "method": "full_scale", "seed": seed, "dataset": dataset,
        "frames": len(rows), "valid_count": valid,
        "raw_epe": raw_epe, "epe": epe, "gain": raw_epe - epe,
        "bad3": weighted("bad3"), "boundary_epe": weighted("boundary_epe"),
        "false_update_rate": sum(int(row["false_update_count"]) for row in rows) / max(clean, 1),
        "clean_degradation": sum(int(row["clean_degradation_count"]) for row in rows) / max(clean, 1),
        "intervention_coverage": changed / max(valid, 1),
        "intervention_precision": sum(int(row["helpful_count"]) for row in rows) / max(changed, 1),
        "frames_worsened": float((degradation > 0).mean()),
        "worst_frame_degradation": float(degradation.max()),
        "runtime_ms_per_frame": float(runtime.get("wall_ms_per_frame", math.nan)),
        "peak_gpu_memory_mb": float(runtime.get("peak_gpu_memory_mb", math.nan)),
    }


def aggregate_original_seen() -> dict:
    path = ROOT / "results/raw_error_abstention/full/frame_metrics.csv"
    row = aggregate_frames(path, dataset="seen", seed=-1)
    row["method"] = "original_promoted"
    summary = json.loads((ROOT / "results/raw_error_abstention/full/aggregate_summary.json").read_text())
    row["runtime_ms_per_frame"] = (
        summary["unseen"]["runtime"]["total_evaluation_seconds"] * 1000
        / 320
    )
    row["peak_gpu_memory_mb"] = math.nan
    return row


def aggregate_original_fast() -> dict:
    sequence_path = ROOT / "results/raw_error_abstention/full/unseen_sequence_metrics.csv"
    rows = [row for row in read_csv(sequence_path)
            if float(row["coverage_threshold"]) == .5 and row["method"] == "authorized_balanced"]
    total = sum(int(row["valid_count"]) for row in rows)
    weighted = lambda key: sum(float(row[key]) * int(row["valid_count"]) for row in rows) / max(total, 1)
    summary = json.loads((ROOT / "results/raw_error_abstention/full/unseen_fast_foundation_complete.json").read_text())
    return {
        "method": "original_promoted", "seed": -1, "dataset": "Fast-FoundationStereo",
        "frames": sum(int(row["frames"]) for row in rows), "valid_count": total,
        "raw_epe": summary["raw_epe"], "epe": summary["epe"], "gain": summary["epe_gain"],
        "bad3": weighted("bad3"), "boundary_epe": weighted("boundary_epe"),
        "false_update_rate": summary["false_update_rate"],
        "clean_degradation": summary["clean_pixel_degradation"],
        "intervention_coverage": summary["intervention_coverage"],
        "intervention_precision": weighted("intervention_precision"),
        "frames_worsened": sum(float(row["percentage_frames_worsened"]) * int(row["frames"]) for row in rows)
        / max(sum(int(row["frames"]) for row in rows), 1),
        "worst_frame_degradation": max(float(row["worst_frame_degradation"]) for row in rows),
        "runtime_ms_per_frame": summary["runtime"]["total_evaluation_seconds"] * 1000
        / max(sum(int(row["frames"]) for row in rows), 1),
        "peak_gpu_memory_mb": math.nan,
    }


def aggregate_original_cres() -> dict:
    value = json.loads((ROOT / "results/ood_generalization/crestereo/summary.json").read_text())
    return {
        "method": "original_promoted", "seed": -1, "dataset": "CREStereo",
        "frames": value["frames"], "valid_count": value["valid_count"],
        "raw_epe": value["raw_epe"], "epe": value["refined_epe"], "gain": value["epe_gain"],
        "bad3": value["refined_bad3"], "boundary_epe": value["refined_boundary_epe"],
        "false_update_rate": value["false_update_rate"],
        "clean_degradation": value["clean_degradation"],
        "intervention_coverage": value["intervention_coverage_all"],
        "intervention_precision": value["intervention_precision"],
        "frames_worsened": value["frames_worsened"],
        "worst_frame_degradation": value["worst_frame_degradation"],
        "runtime_ms_per_frame": value["runtime_ms"], "peak_gpu_memory_mb": math.nan,
    }


def mean_std(rows: list[dict]) -> dict:
    output = {}
    for key in METRICS:
        values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
        output[key] = {
            "mean": statistics.fmean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
            "count": len(values),
        }
    return output


def comparison(full: list[dict], baseline: dict) -> dict:
    result = {}
    lower_is_better = {
        "epe", "bad3", "boundary_epe", "false_update_rate", "clean_degradation",
        "frames_worsened", "worst_frame_degradation", "runtime_ms_per_frame", "peak_gpu_memory_mb",
    }
    for key in METRICS:
        base = float(baseline[key])
        if not math.isfinite(base):
            continue
        differences = []
        for row in full:
            value = float(row[key])
            differences.append(base - value if key in lower_is_better else value - base)
        mean = statistics.fmean(differences)
        std = statistics.stdev(differences) if len(differences) > 1 else 0.0
        half = 4.303 * std / math.sqrt(len(differences)) if len(differences) > 1 else 0.0
        result[key] = {
            "improvement_mean": mean, "improvement_std": std,
            "ci95_lower": mean - half, "ci95_upper": mean + half,
            "significant_at_95pct": bool(mean - half > 0 or mean + half < 0),
        }
    return result


def training_curves(root: Path, seeds: tuple[int, ...]) -> list[dict]:
    output = []
    for seed in seeds:
        for stage, relative in (("a2", "a2/training_history.csv"),
                                ("detector", "detector/training_history.csv")):
            path = root / f"seed_{seed}" / relative
            for row in read_csv(path):
                output.append({"seed": seed, "stage": stage, **row})
    return output


def diagnostic_plot(root: Path, curves: list[dict]) -> None:
    import matplotlib.pyplot as plt

    destination = root / "compact_diagnostics"
    destination.mkdir(exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for seed in sorted({int(row["seed"]) for row in curves}):
        a2 = [row for row in curves if int(row["seed"]) == seed and row["stage"] == "a2"]
        detector = [row for row in curves if int(row["seed"]) == seed and row["stage"] == "detector"]
        axes[0].plot([int(row["epoch"]) + 1 for row in a2],
                     [float(row["validation_refined_epe"]) for row in a2], label=f"seed {seed}")
        axes[1].plot([int(row["epoch"]) for row in detector],
                     [float(row["loss_total"]) for row in detector], label=f"seed {seed}")
    axes[0].set(title="A2 validation EPE", xlabel="epoch", ylabel="EPE")
    axes[1].set(title="Detector validation loss", xlabel="epoch", ylabel="loss")
    for axis in axes: axis.grid(alpha=.2); axis.legend()
    figure.tight_layout(); figure.savefig(destination / "training_curves.png", dpi=140); plt.close(figure)


def main() -> None:
    args = parse_args(); root = args.root.resolve(); seeds = tuple(args.seeds)
    rows = []
    for seed in seeds:
        seed_root = root / f"seed_{seed}" / "detector"
        rows.append(aggregate_frames(seed_root / "frame_metrics.csv", dataset="seen", seed=seed))
        for dataset, slug in (("Fast-FoundationStereo", "fast_foundation"), ("CREStereo", "crestereo")):
            path = seed_root / f"unseen_{slug}_frame_metrics.csv"
            if path.exists(): rows.append(aggregate_frames(path, dataset=dataset, seed=seed))
    baselines = {"seen": aggregate_original_seen(), "Fast-FoundationStereo": aggregate_original_fast()}
    if all(any(row["dataset"] == "CREStereo" and row["seed"] == seed for row in rows) for seed in seeds):
        baselines["CREStereo"] = aggregate_original_cres()
    write_csv(root / "per_seed.csv", [*baselines.values(), *rows])
    groups = defaultdict(list)
    for row in rows: groups[row["dataset"]].append(row)
    summary = {dataset: mean_std(group) for dataset, group in groups.items()}
    save_json(root / "seed_summary.json", summary)
    curves = training_curves(root, seeds); write_csv(root / "training_curves.csv", curves)
    diagnostic_plot(root, curves)
    aggregate = {
        "experiment": "ARGOS v2 architecture-frozen full-scale optimization",
        "seeds": list(seeds), "baseline": baselines, "full_scale": summary,
        "comparison": {dataset: comparison(groups[dataset], baseline)
                       for dataset, baseline in baselines.items() if dataset in groups},
        "architecture_changed": False, "training_only_changes": True,
    }
    save_json(root / "aggregate_summary.json", aggregate)


if __name__ == "__main__":
    main()
