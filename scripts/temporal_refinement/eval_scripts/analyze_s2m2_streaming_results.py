#!/usr/bin/env python3
"""Analyze existing streaming S2M2-S rectified temporal-GT CSV results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULT_DIR = ROOT / "results/03_temporal_refinement/evaluation/gt_temporal_rectified_streaming_s2m2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--top-frames", type=int, default=200)
    return parser.parse_args()


def f(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def quantile(values: list[float], q: float) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return math.nan
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return None
    xbar = sum(x for x, _y in pairs) / len(pairs)
    ybar = sum(y for _x, y in pairs) / len(pairs)
    num = sum((x - xbar) * (y - ybar) for x, y in pairs)
    xden = math.sqrt(sum((x - xbar) ** 2 for x, _y in pairs))
    yden = math.sqrt(sum((y - ybar) ** 2 for _x, y in pairs))
    return num / (xden * yden) if xden and yden else None


def mean(values: list[float]) -> float | None:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else None


def add_sequence_ranks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = [dict(row) for row in rows]
    for key, rank_key in [
        ("disparity_mae_mean", "rank_disparity_mae"),
        ("bad_3px_mean", "rank_bad_3px"),
        ("depth_mae_mean", "rank_depth_mae"),
    ]:
        for rank, row in enumerate(sorted(ranked, key=lambda r: f(r[key])), start=1):
            row[rank_key] = rank
    for row in ranked:
        row["composite_error_rank"] = (
            f(row["rank_disparity_mae"], 0) + f(row["rank_bad_3px"], 0) + f(row["rank_depth_mae"], 0)
        ) / 3.0
        row["skipped_ratio"] = f(row["frames_skipped"], 0.0) / max(f(row["num_frames_total"], 0.0), 1.0)
    return ranked


def add_frame_ranks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    included = [dict(row) for row in rows if str(row.get("included", "")).lower() == "true"]
    for key, rank_key in [("disparity_mae", "rank_disparity_mae_worst"), ("bad_3px", "rank_bad_3px_worst")]:
        for rank, row in enumerate(sorted(included, key=lambda r: f(r[key]), reverse=True), start=1):
            row[rank_key] = rank
    for row in included:
        row["worst_rank"] = min(f(row["rank_disparity_mae_worst"], 10**9), f(row["rank_bad_3px_worst"], 10**9))
    return sorted(included, key=lambda r: (f(r["worst_rank"]), r["sequence_id"], r["frame_id"]))


def split_sequences(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metrics = ["disparity_mae_mean", "bad_3px_mean", "depth_mae_mean"]
    thresholds: dict[str, Any] = {}
    for key in metrics:
        vals = [f(row[key]) for row in rows]
        q1 = quantile(vals, 0.25)
        q3 = quantile(vals, 0.75)
        thresholds[key] = {
            "q25": q1,
            "q50": quantile(vals, 0.50),
            "q75": q3,
            "upper_fence": q3 + 1.5 * (q3 - q1),
        }
    valid_vals = [f(row["valid_pixel_pct_mean"]) for row in rows]
    skipped_vals = [f(row["skipped_ratio"]) for row in rows]
    thresholds["valid_pixel_pct_mean"] = {
        "q10": quantile(valid_vals, 0.10),
        "q25": quantile(valid_vals, 0.25),
        "q50": quantile(valid_vals, 0.50),
    }
    thresholds["skipped_ratio"] = {"q75": quantile(skipped_vals, 0.75)}

    split_rows: list[dict[str, Any]] = []
    for row in rows:
        reasons: list[str] = []
        for key in metrics:
            if f(row[key]) >= thresholds[key]["upper_fence"]:
                reasons.append(f"{key}>={thresholds[key]['upper_fence']:.4g} upper_fence")
        if f(row["valid_pixel_pct_mean"]) <= thresholds["valid_pixel_pct_mean"]["q10"]:
            reasons.append(f"valid_pixel_pct_mean<={thresholds['valid_pixel_pct_mean']['q10']:.4g} q10")
        if reasons:
            group = "exclude_or_diagnostic"
        else:
            for key in metrics:
                if f(row[key]) >= thresholds[key]["q75"]:
                    reasons.append(f"{key}>={thresholds[key]['q75']:.4g} q75")
            if f(row["valid_pixel_pct_mean"]) <= thresholds["valid_pixel_pct_mean"]["q25"]:
                reasons.append(f"valid_pixel_pct_mean<={thresholds['valid_pixel_pct_mean']['q25']:.4g} q25")
            if f(row["skipped_ratio"]) >= thresholds["skipped_ratio"]["q75"] and f(row["skipped_ratio"]) > 0:
                reasons.append(f"skipped_ratio>={thresholds['skipped_ratio']['q75']:.4g} q75")
            group = "stress_eval" if reasons else "core_eval"
        split_rows.append(
            {
                "sequence_id": row["sequence_id"],
                "group": group,
                "reason": "; ".join(reasons) if reasons else "below q75 error thresholds and above q25 valid-pixel threshold",
                "frames_evaluated": row["frames_evaluated"],
                "frames_skipped": row["frames_skipped"],
                "valid_pixel_pct_mean": row["valid_pixel_pct_mean"],
                "disparity_mae_mean": row["disparity_mae_mean"],
                "bad_3px_mean": row["bad_3px_mean"],
                "depth_mae_mean": row["depth_mae_mean"],
            }
        )
    return split_rows, thresholds


def weighted_metric(frame_rows: list[dict[str, Any]], sequences: set[str] | None, key: str) -> float | None:
    num = 0.0
    den = 0.0
    for row in frame_rows:
        if str(row.get("included", "")).lower() != "true":
            continue
        if sequences is not None and row["sequence_id"] not in sequences:
            continue
        weight = f(row.get("valid_pixel_count"), 0.0)
        value = f(row.get(key))
        if weight > 0 and math.isfinite(value):
            num += weight * value
            den += weight
    return num / den if den else None


def contribution_summary(sequence_rows: list[dict[str, Any]], frame_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(sequence_rows, key=lambda r: f(r["disparity_mae_mean"]), reverse=True)
    all_disp = weighted_metric(frame_rows, None, "disparity_mae")
    out: dict[str, Any] = {"all_weighted_disparity_mae": all_disp}
    for n in [3, max(1, math.ceil(len(ranked) * 0.25))]:
        seqs = {row["sequence_id"] for row in ranked[:n]}
        kept = {row["sequence_id"] for row in ranked[n:]}
        out[f"without_top_{n}_weighted_disparity_mae"] = weighted_metric(frame_rows, kept, "disparity_mae")
        out[f"top_{n}_sequences"] = sorted(seqs)
    return out


def correlations(sequence_rows: list[dict[str, Any]], frame_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sequence_level": {
            key: pearson([f(r["valid_pixel_pct_mean"]) for r in sequence_rows], [f(r[key]) for r in sequence_rows])
            for key in ["disparity_mae_mean", "bad_3px_mean", "depth_mae_mean"]
        },
        "frame_level": {
            key: pearson([f(r["valid_pixel_pct"]) for r in frame_rows], [f(r[key]) for r in frame_rows])
            for key in ["disparity_mae", "bad_3px", "depth_mae"]
        },
    }


def write_plots(out_dir: Path, sequence_rows: list[dict[str, Any]], frame_rows: list[dict[str, Any]]) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    paths: list[str] = []

    def save(path: Path) -> None:
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(path.name)

    sorted_seq = sorted(sequence_rows, key=lambda r: f(r["disparity_mae_mean"]))
    plt.figure(figsize=(9, 4))
    plt.plot([f(r["disparity_mae_mean"]) for r in sorted_seq], marker="o", linewidth=1)
    plt.ylabel("sequence disparity MAE px")
    plt.xlabel("sequence rank")
    save(out_dir / "sequence_disp_mae_sorted.png")

    sorted_bad = sorted(sequence_rows, key=lambda r: f(r["bad_3px_mean"]))
    plt.figure(figsize=(9, 4))
    plt.plot([f(r["bad_3px_mean"]) for r in sorted_bad], marker="o", linewidth=1)
    plt.ylabel("sequence bad-3px %")
    plt.xlabel("sequence rank")
    save(out_dir / "sequence_bad3_sorted.png")

    included = [r for r in frame_rows if str(r.get("included", "")).lower() == "true"]
    plt.figure(figsize=(6, 4))
    plt.scatter([f(r["valid_pixel_pct"]) for r in included], [f(r["disparity_mae"]) for r in included], s=4, alpha=0.25)
    plt.xlabel("valid pixel ratio")
    plt.ylabel("frame disparity MAE px")
    save(out_dir / "error_vs_valid_ratio.png")

    plt.figure(figsize=(7, 4))
    plt.hist([f(r["disparity_mae"]) for r in included if math.isfinite(f(r["disparity_mae"]))], bins=80)
    plt.xlabel("frame disparity MAE px")
    plt.ylabel("frames")
    save(out_dir / "frame_error_histogram.png")
    return paths


def write_readme(out_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# S2M2 Streaming Result Analysis",
        "",
        summary["conclusion"],
        "",
        "Outputs:",
        "",
        "- `best_sequences.csv` and `worst_sequences.csv`: sequence rankings by disparity MAE, bad-3px, and depth MAE.",
        "- `worst_frames.csv`: highest-error included frames ranked by disparity MAE and bad-3px.",
        "- `core_stress_exclude_split.csv`: quantile-based proposed sequence split.",
        "- `analysis_summary.json`: thresholds, correlations, contribution checks, and group counts.",
        "",
        "Split rule:",
        "",
        "- `exclude_or_diagnostic`: any error metric above its Tukey upper fence, or valid-pixel mean at/below q10.",
        "- `stress_eval`: any error metric at/above q75, valid-pixel mean at/below q25, or skipped ratio at/above q75.",
        "- `core_eval`: everything else.",
    ]
    if summary.get("plots"):
        lines.extend(["", "Plots:", "", *[f"- `{name}`" for name in summary["plots"]]])
    out_dir.joinpath("README.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    result_dir = args.result_dir
    out_dir = args.out_dir or result_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    sequence_rows = add_sequence_ranks(read_csv(result_dir / "sequence_metrics.csv"))
    frame_rows = read_csv(result_dir / "frame_metrics.csv")
    worst_frames = add_frame_ranks(frame_rows)[: args.top_frames]
    split_rows, thresholds = split_sequences(sequence_rows)
    group_counts = {group: sum(1 for row in split_rows if row["group"] == group) for group in ["core_eval", "stress_eval", "exclude_or_diagnostic"]}
    contrib = contribution_summary(sequence_rows, frame_rows)
    corr = correlations(sequence_rows, frame_rows)

    core_sequences = {row["sequence_id"] for row in split_rows if row["group"] == "core_eval"}
    stress_sequences = {row["sequence_id"] for row in split_rows if row["group"] == "stress_eval"}
    exclude_sequences = {row["sequence_id"] for row in split_rows if row["group"] == "exclude_or_diagnostic"}
    group_weighted = {
        "core_eval_weighted_disparity_mae": weighted_metric(frame_rows, core_sequences, "disparity_mae"),
        "stress_eval_weighted_disparity_mae": weighted_metric(frame_rows, stress_sequences, "disparity_mae"),
        "exclude_or_diagnostic_weighted_disparity_mae": weighted_metric(frame_rows, exclude_sequences, "disparity_mae"),
    }
    median_bad3 = quantile([f(row["bad_3px_mean"]) for row in sequence_rows], 0.50)
    conclusion = (
        "Poor aggregate metrics are outlier-heavy but not isolated: removing the top three disparity-MAE "
        f"sequences lowers weighted disparity MAE from {contrib['all_weighted_disparity_mae']:.3f} px to "
        f"{contrib['without_top_3_weighted_disparity_mae']:.3f} px, while median sequence bad-3px remains "
        f"{median_bad3:.2f}%."
    )

    sequence_fields = [
        "sequence_id",
        "num_frames_total",
        "frames_evaluated",
        "frames_skipped",
        "valid_pixel_pct_mean",
        "disparity_mae_mean",
        "bad_3px_mean",
        "depth_mae_mean",
        "rank_disparity_mae",
        "rank_bad_3px",
        "rank_depth_mae",
        "composite_error_rank",
    ]
    write_csv(out_dir / "best_sequences.csv", sorted(sequence_rows, key=lambda r: f(r["composite_error_rank"])), sequence_fields)
    write_csv(out_dir / "worst_sequences.csv", sorted(sequence_rows, key=lambda r: f(r["composite_error_rank"], 0), reverse=True), sequence_fields)
    write_csv(
        out_dir / "worst_frames.csv",
        worst_frames,
        [
            "sequence_id",
            "frame_id",
            "valid_pixel_pct",
            "valid_pixel_count",
            "disparity_mae",
            "bad_3px",
            "depth_mae",
            "rank_disparity_mae_worst",
            "rank_bad_3px_worst",
        ],
    )
    write_csv(
        out_dir / "core_stress_exclude_split.csv",
        split_rows,
        [
            "sequence_id",
            "group",
            "reason",
            "frames_evaluated",
            "frames_skipped",
            "valid_pixel_pct_mean",
            "disparity_mae_mean",
            "bad_3px_mean",
            "depth_mae_mean",
        ],
    )
    plots = write_plots(out_dir, sequence_rows, frame_rows)
    summary = {
        "result_dir": str(result_dir),
        "out_dir": str(out_dir),
        "thresholds": thresholds,
        "group_counts": group_counts,
        "correlations": corr,
        "contribution_summary": contrib,
        "group_weighted_metrics": group_weighted,
        "median_sequence_bad_3px": median_bad3,
        "conclusion": conclusion,
        "plots": plots,
    }
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_readme(out_dir, summary)
    print(json.dumps({"out_dir": str(out_dir), "conclusion": conclusion, "group_counts": group_counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
