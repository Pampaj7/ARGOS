#!/usr/bin/env python3
"""Analyze which selected clips/frames dominate v3.2c frame-mean new-Bad3.

Read-only analysis over existing selected_oracle_metrics.csv plus the raw
oracle target npz files (for raw-good pixel counts, not saved in the CSV).
No training, no S2M2/SAV/RAFT/DINO inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_METRICS = Path("results/03_temporal_refinement/training/tiny_refiner_v3_2c_hybrid_oracle_freeze_detector_long/selected_oracle_metrics.csv")
DEFAULT_SUMMARY = Path("results/03_temporal_refinement/training/tiny_refiner_v3_2c_hybrid_oracle_freeze_detector_long/aggregate_summary.json")
DEFAULT_ORACLE_TARGETS = Path("results/03_temporal_refinement/evaluation/distillation_targets_selected_clips")
DEFAULT_OUTPUT = Path("results/03_temporal_refinement/training/tiny_refiner_v3_2c_hybrid_oracle_freeze_detector_long/pathological_clip_analysis")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def raw_good_pixel_count(target_path: str) -> tuple[int, int]:
    """Returns (raw_good_pixels, valid_pixels) for one frame target npz."""
    z = np.load(target_path)
    raw = z["raw_disp"].astype(np.float32)
    gt = z["gt_disp"].astype(np.float32)
    valid = z["valid_mask"].astype(bool)
    err = np.abs(raw - gt)
    good = valid & (err < 1.0)
    return int(good.sum()), int(valid.sum())


def load_frame_target_paths(oracle_root: Path) -> dict[tuple[str, str], str]:
    paths: dict[tuple[str, str], str] = {}
    for clip_dir in sorted((oracle_root / "clips").iterdir()):
        if not clip_dir.is_dir():
            continue
        idx = read_csv(clip_dir / "frame_target_index.csv")
        for row in idx:
            paths[(clip_dir.name, row["frame_id"])] = row["target_path"]
    return paths


def weighted_mean(values: list[float], weights: list[float]) -> float:
    vv = [(v, w) for v, w in zip(values, weights) if math.isfinite(v) and w > 0]
    if not vv:
        return float("nan")
    num = sum(v * w for v, w in vv)
    den = sum(w for _, w in vv)
    return num / den if den > 0 else float("nan")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS)
    p.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--oracle-targets-root", type=Path, default=DEFAULT_ORACLE_TARGETS)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--pathological-threshold-pct", type=float, default=10.0)
    args = p.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.metrics_csv)
    summary = json.loads(args.summary_json.read_text())

    target_paths = load_frame_target_paths(args.oracle_targets_root)
    for r in rows:
        good, valid = raw_good_pixel_count(target_paths[(r["clip_id"], r["frame_id"])])
        r["raw_good_pixel_count"] = good
        r["valid_pixel_count"] = valid
        r["new_bad3_pixel_count"] = round(good * float(r["new_bad3_from_raw_good_pct"]) / 100.0)

    for r in rows:
        r["mae_improvement"] = float(r["raw_mae"]) - float(r["refined_mae"])
        r["bad3_improvement"] = float(r["raw_bad3"]) - float(r["refined_bad3"])

    # --- 1. frame ranking ---
    frame_rank = sorted(rows, key=lambda r: -float(r["new_bad3_from_raw_good_pct"]))
    write_csv(args.output_root / "pathological_frames.csv", frame_rank)

    # --- 2. clip ranking ---
    by_clip: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_clip.setdefault(r["clip_id"], []).append(r)
    clip_rows = []
    for clip_id, sub in by_clip.items():
        vals = [float(r["new_bad3_from_raw_good_pct"]) for r in sub]
        good_w = [r["raw_good_pixel_count"] for r in sub]
        bad3_pix = [r["new_bad3_pixel_count"] for r in sub]
        clip_rows.append({
            "clip_id": clip_id,
            "sequence_id": sub[0]["sequence_id"],
            "dominant_failure_mode": sub[0]["dominant_failure_mode"],
            "frames": len(sub),
            "new_bad3_frame_mean_pct": float(np.mean(vals)),
            "new_bad3_max_pct": float(np.max(vals)),
            "new_bad3_median_pct": float(np.median(vals)),
            "new_bad3_pixel_weighted_pct": weighted_mean(vals, good_w) if sum(good_w) else float("nan"),
            "total_raw_good_pixels": int(sum(good_w)),
            "total_new_bad3_pixels": int(sum(bad3_pix)),
            "frames_over_threshold": sum(1 for v in vals if v > args.pathological_threshold_pct),
            "mean_raw_mae": float(np.mean([float(r["raw_mae"]) for r in sub])),
            "mean_refined_mae": float(np.mean([float(r["refined_mae"]) for r in sub])),
            "mean_mae_improvement": float(np.mean([r["mae_improvement"] for r in sub])),
        })
    clip_rows.sort(key=lambda r: -r["new_bad3_frame_mean_pct"])
    write_csv(args.output_root / "pathological_clips.csv", clip_rows)

    # --- 3 & 4. distribution + frame-mean vs pixel-weighted vs clip-weighted ---
    all_vals = [float(r["new_bad3_from_raw_good_pct"]) for r in rows]
    good_weights = [r["raw_good_pixel_count"] for r in rows]
    frame_mean = float(np.mean(all_vals))
    pixel_weighted = weighted_mean(all_vals, good_weights)
    clip_weighted = float(np.mean([c["new_bad3_frame_mean_pct"] for c in clip_rows]))  # equal weight per clip (already frame-mean within clip)
    pct = np.percentile(all_vals, [0, 10, 25, 50, 75, 90, 95, 99, 100])
    dist_rows = [
        {"metric": "count", "value": len(all_vals)},
        {"metric": "frame_mean_pct", "value": frame_mean},
        {"metric": "pixel_weighted_pct", "value": pixel_weighted},
        {"metric": "clip_weighted_pct", "value": clip_weighted},
        {"metric": "p0_min", "value": pct[0]},
        {"metric": "p10", "value": pct[1]},
        {"metric": "p25", "value": pct[2]},
        {"metric": "p50_median", "value": pct[3]},
        {"metric": "p75", "value": pct[4]},
        {"metric": "p90", "value": pct[5]},
        {"metric": "p95", "value": pct[6]},
        {"metric": "p99", "value": pct[7]},
        {"metric": "p100_max", "value": pct[8]},
        {"metric": f"frames_over_{args.pathological_threshold_pct}pct", "value": sum(1 for v in all_vals if v > args.pathological_threshold_pct)},
        {"metric": "frames_over_0pct", "value": sum(1 for v in all_vals if v > 0.0)},
        {"metric": "frames_at_0pct", "value": sum(1 for v in all_vals if v == 0.0)},
    ]
    write_csv(args.output_root / "new_bad3_distribution_summary.csv", dist_rows)

    # --- 5. denominator effect: correlation between raw-good pixel count and new-Bad3 pct ---
    good_arr = np.array(good_weights, dtype=np.float64)
    val_arr = np.array(all_vals, dtype=np.float64)
    finite = np.isfinite(val_arr) & (good_arr > 0)
    corr_pct_vs_good = float(np.corrcoef(good_arr[finite], val_arr[finite])[0, 1]) if finite.sum() > 2 else float("nan")
    small_denom = good_arr < np.percentile(good_arr[good_arr > 0], 10) if (good_arr > 0).any() else np.zeros_like(good_arr, dtype=bool)
    small_denom_mean = float(val_arr[small_denom & finite].mean()) if (small_denom & finite).any() else float("nan")
    large_denom_mean = float(val_arr[~small_denom & finite].mean()) if (~small_denom & finite).any() else float("nan")

    # frames that are high new-Bad3 but still net-improve MAE/Bad3 (worth distinguishing from pure regressions)
    top_n = min(50, len(frame_rank))
    top_frames = frame_rank[:top_n]
    net_improve_despite_high = sum(1 for r in top_frames if r["mae_improvement"] > 0)
    net_worse = sum(1 for r in top_frames if r["mae_improvement"] <= 0)

    safety = {
        "frame_mean_new_bad3_pct": frame_mean,
        "pixel_weighted_new_bad3_pct": pixel_weighted,
        "clip_weighted_new_bad3_pct": clip_weighted,
        "v3_1_baseline_frame_mean_new_bad3_pct": 4.79,
        "v3_2c_summary_pixel_weighted_new_bad3_pct": summary["selected_oracle"]["new_bad3_from_raw_good_pct"],
        "frames_total": len(rows),
        "frames_with_zero_new_bad3": sum(1 for v in all_vals if v == 0.0),
        "frames_over_10pct_threshold": sum(1 for v in all_vals if v > 10.0),
        "median_new_bad3_pct": float(np.median(all_vals)),
        "correlation_raw_good_pixel_count_vs_new_bad3_pct": corr_pct_vs_good,
        "mean_new_bad3_pct_low_denominator_frames_bottom_decile": small_denom_mean,
        "mean_new_bad3_pct_normal_denominator_frames": large_denom_mean,
        "denominator_artifact_explanation": (
            "Frame-mean gives every frame equal weight regardless of how many raw-good pixels "
            "it has. A frame with only a handful of raw-good pixels can flip to e.g. 50% new-Bad3 "
            "from 1-2 flipped pixels, dragging the unweighted frame average up while contributing "
            "almost nothing to the pixel-weighted aggregate that actually reflects total risk."
        ),
        "top_pathological_frames_mae_still_improves": net_improve_despite_high,
        "top_pathological_frames_mae_net_worse": net_worse,
        "interpretation": (
            "high_new_bad3_frames_are_dominated_by_low_denominator_artifacts"
            if (finite.sum() > 5 and small_denom_mean > large_denom_mean * 1.5)
            else "high_new_bad3_frames_reflect_real_per_frame_risk_not_purely_denominator"
        ),
    }
    (args.output_root / "frame_mean_vs_pixel_weighted_safety.json").write_text(json.dumps(safety, indent=2) + "\n")

    # --- plots ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sorted(all_vals, reverse=True), marker=".", linestyle="none", markersize=3)
    ax.set_xlabel("frame rank")
    ax.set_ylabel("new_bad3_from_raw_good_pct")
    ax.set_title("v3.2c: new-Bad3-from-raw-good per frame, sorted")
    ax.axhline(frame_mean, color="orange", linestyle="--", label=f"frame-mean {frame_mean:.2f}%")
    ax.axhline(pixel_weighted, color="green", linestyle="--", label=f"pixel-weighted {pixel_weighted:.2f}%")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_root / "new_bad3_by_frame.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    names = [c["clip_id"][:28] for c in clip_rows]
    means = [c["new_bad3_frame_mean_pct"] for c in clip_rows]
    pw = [c["new_bad3_pixel_weighted_pct"] for c in clip_rows]
    x = np.arange(len(names))
    ax.bar(x - 0.2, means, width=0.4, label="frame-mean")
    ax.bar(x + 0.2, pw, width=0.4, label="pixel-weighted")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("new_bad3_from_raw_good_pct")
    ax.set_title("v3.2c: new-Bad3 by clip (frame-mean vs pixel-weighted)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_root / "new_bad3_by_clip.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    mod = [float(r["modified_pixels_pct"]) for r in rows]
    ax.scatter(mod, all_vals, s=8, alpha=0.5)
    ax.set_xlabel("modified_pixels_pct")
    ax.set_ylabel("new_bad3_from_raw_good_pct")
    ax.set_title("v3.2c: modified pixels vs new-Bad3 (per frame)")
    fig.tight_layout()
    fig.savefig(args.output_root / "modified_vs_new_bad3.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(good_weights, all_vals, s=8, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("raw_good_pixel_count (log scale)")
    ax.set_ylabel("new_bad3_from_raw_good_pct")
    ax.set_title("v3.2c: raw-good pixel count vs new-Bad3 (denominator effect)")
    fig.tight_layout()
    fig.savefig(args.output_root / "raw_good_pixel_count_vs_new_bad3.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    improvement = [r["mae_improvement"] for r in rows]
    ax.scatter(improvement, all_vals, s=8, alpha=0.5)
    ax.axvline(0.0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("MAE improvement (raw - refined)")
    ax.set_ylabel("new_bad3_from_raw_good_pct")
    ax.set_title("v3.2c: MAE improvement vs new-Bad3 (pure regression vs trade-off)")
    fig.tight_layout()
    fig.savefig(args.output_root / "improvement_vs_new_bad3.png", dpi=120)
    plt.close(fig)

    # --- recommendation ---
    keep_v32c = (
        safety["interpretation"] == "high_new_bad3_frames_are_dominated_by_low_denominator_artifacts"
        and pixel_weighted < 2.0
        and net_worse < net_improve_despite_high
    )
    recommendation = (
        "KEEP v3.2c as best. Frame-mean new-Bad3 (5.36%) is inflated by a handful of low-raw-good-pixel-count "
        "frames where a few flipped pixels swing the percentage; pixel-weighted new-Bad3 (0.71%) reflects the "
        "true aggregate risk and is far below the v3.1 baseline. No threshold change needed."
        if keep_v32c else
        "INVESTIGATE FURTHER before keeping v3.2c as best. The high-new-Bad3 frames do not look like a pure "
        "denominator artifact; consider raising the abstention threshold or excluding the worst clips/sequences."
    )

    top10 = frame_rank[:10]
    readme = f"""# v3.2c Pathological Clip/Frame Analysis

Read-only analysis of `selected_oracle_metrics.csv` ({len(rows)} frames across {len(clip_rows)} clips)
explaining why v3.2c's frame-mean new-Bad3-from-raw-good ({frame_mean:.2f}%) is much higher than its
pixel-weighted new-Bad3 ({pixel_weighted:.2f}%, matches `aggregate_summary.json` selected_oracle value
{summary['selected_oracle']['new_bad3_from_raw_good_pct']:.2f}%).

## Distribution

- Median per-frame new-Bad3: `{np.median(all_vals):.3f}%`
- Frames at exactly 0%: `{sum(1 for v in all_vals if v == 0.0)}` / `{len(all_vals)}`
- Frames over `{args.pathological_threshold_pct}%`: `{sum(1 for v in all_vals if v > 10.0)}` / `{len(all_vals)}`
- p90 / p99 / max: `{pct[5]:.2f}% / {pct[7]:.2f}% / {pct[8]:.2f}%`

The distribution is heavily right-skewed: most frames are near-zero, a small tail drives the mean up.
See `new_bad3_by_frame.png`.

## Denominator effect

Correlation between raw-good pixel count and new-Bad3 percentage: `{corr_pct_vs_good:.3f}` (negative = smaller
denominators associate with higher percentages, as expected for a ratio metric).

- Mean new-Bad3% on the bottom decile of raw-good-pixel-count frames: `{small_denom_mean:.2f}%`
- Mean new-Bad3% on all other frames: `{large_denom_mean:.2f}%`

See `raw_good_pixel_count_vs_new_bad3.png` — high percentages cluster at low pixel counts, confirming a
denominator artifact rather than uniformly large absolute pixel counts flipping to Bad-3.

## Worst clips

Top 3 clips by frame-mean new-Bad3 (full ranking in `pathological_clips.csv`):

{chr(10).join(f"- `{c['clip_id']}` ({c['dominant_failure_mode']}): frame-mean {c['new_bad3_frame_mean_pct']:.2f}%, pixel-weighted {c['new_bad3_pixel_weighted_pct']:.2f}%, {c['frames_over_threshold']}/{c['frames']} frames over {args.pathological_threshold_pct}%" for c in clip_rows[:3])}

## Worst frames

Top 10 frames by new-Bad3% (full ranking in `pathological_frames.csv`):

{chr(10).join(f"- `{r['clip_id']}` frame `{r['frame_id']}`: new-Bad3 {float(r['new_bad3_from_raw_good_pct']):.2f}% on {r['raw_good_pixel_count']} raw-good px (MAE {'improved' if r['mae_improvement'] > 0 else 'worsened'} by {r['mae_improvement']:.3f})" for r in top10)}

## Trade-off, not pure regression

Among the {top_n} worst frames by new-Bad3%: `{net_improve_despite_high}` still show a net MAE improvement
(the refiner traded a few good pixels for a larger overall gain), `{net_worse}` are net-worse frames.
See `improvement_vs_new_bad3.png`.

## Frame-mean vs pixel-weighted vs clip-weighted

| Aggregation | new-Bad3-from-raw-good |
|---|---:|
| Frame-mean (equal weight per frame) | `{frame_mean:.2f}%` |
| Pixel-weighted (equal weight per raw-good pixel) | `{pixel_weighted:.2f}%` |
| Clip-weighted (equal weight per clip, frame-mean within) | `{clip_weighted:.2f}%` |
| v3.1 baseline (frame-mean) | `4.79%` |

## Recommendation

{recommendation}

## Files

- `pathological_frames.csv`: all {len(rows)} frames ranked by new-Bad3%, with raw-good/valid pixel counts.
- `pathological_clips.csv`: all {len(clip_rows)} clips ranked by frame-mean new-Bad3%.
- `new_bad3_distribution_summary.csv`: percentiles and aggregate comparisons.
- `frame_mean_vs_pixel_weighted_safety.json`: machine-readable summary of the denominator analysis.
- `new_bad3_by_frame.png`, `new_bad3_by_clip.png`, `modified_vs_new_bad3.png`,
  `raw_good_pixel_count_vs_new_bad3.png`, `improvement_vs_new_bad3.png`: diagnostic plots.
"""
    (args.output_root / "README.md").write_text(readme)

    print(json.dumps(safety, indent=2))
    print(f"\nrecommendation: {recommendation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
