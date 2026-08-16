#!/usr/bin/env python3
"""Per-frame behaviour over a whole sequence, from the stored evaluation reports.

The disparity maps themselves are not a useful figure: the intervention is sub-pixel
(about $0.03$ px on disparities of $8$ px), so at any honest shared colour scale the
refined map is indistinguishable from the raw one, and stretching the scale until a
difference appears would be manufacturing it.

What is genuinely visible is the per-frame behaviour, which is also what the paper argues
about: refinement as an intervention with a benefit and a cost, measured frame by frame.
This reads the `per_frame` block the evaluator already wrote, so it costs no GPU time and
cannot disagree with the tables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT.parent / "results"
DEFAULT = (RESULTS / "scared_masked/runs/scared-d2/"
           "model_design_comparison_canonical_h4_masked__factory/reports/RAFT-Stereo/"
           "dataset_2_keyframe_4.json")


def series(report: dict, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Aligned (raw, refined) per-frame values, dropping frames the evaluator left null."""
    pairs = [(p["metrics"]["disparity_px"]["raw"][metric]["value"],
              p["metrics"]["disparity_px"]["refined"][metric]["value"])
             for p in report["per_frame"]]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    return np.array([a for a, _ in pairs]), np.array([b for _, b in pairs])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, default=DEFAULT)
    parser.add_argument("--metric", default="Bad1")
    parser.add_argument("--output", type=Path, default=ROOT.parents[1] / "ARGOS_hand/paper/temporal_trace.pdf")
    parser.add_argument("--width-in", type=float, default=3.4)
    parser.add_argument("--smooth", type=int, default=31, help="rolling window, odd; raw traces stay visible")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads(args.report.read_text())
    raw, refined = series(report, args.metric)
    delta = refined - raw
    scale = 100.0 if args.metric.startswith("Bad") else 1.0
    unit = "%" if scale == 100.0 else "px"

    def roll(value: np.ndarray) -> np.ndarray:
        kernel = np.ones(args.smooth) / args.smooth
        return np.convolve(value, kernel, mode="valid")

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(args.width_in, args.width_in * 0.34),
        gridspec_kw={"width_ratios": [2.1, 1.0], "wspace": 0.42})

    frames = np.arange(len(raw))
    offset = args.smooth // 2
    left.plot(frames, raw * scale, color="0.75", linewidth=0.25)
    left.plot(frames, refined * scale, color="#1f6fb4", linewidth=0.25, alpha=0.6)
    left.plot(frames[offset:len(frames) - offset], roll(raw) * scale, color="0.35", linewidth=0.8,
              label="raw")
    left.plot(frames[offset:len(frames) - offset], roll(refined) * scale, color="#0b3d66", linewidth=0.8,
              label="TETHER")
    left.set_xlabel("frame", fontsize=5.5, labelpad=1.5)
    left.set_ylabel(f"{args.metric} ({unit})", fontsize=5.5, labelpad=1.5)
    left.legend(fontsize=5, frameon=False, loc="upper left", handlelength=1.4, borderpad=0.2)
    left.tick_params(labelsize=4.6, length=1.6, pad=1.2)

    improved = float((delta < 0).mean()) * 100
    worsened = float((delta > 0).mean()) * 100
    # A few frames move by orders of magnitude more than the rest; clipping the
    # axis to the central 98% keeps the bulk legible, and the count of clipped
    # frames is printed rather than hidden.
    limit = float(np.percentile(np.abs(delta * scale), 99)) or 1e-6
    clipped = int((np.abs(delta * scale) > limit).sum())
    right.hist(np.clip(delta * scale, -limit, limit), bins=45, color="0.55", linewidth=0)
    right.axvline(0.0, color="black", linewidth=0.5)
    right.set_xlabel(f"per-frame $\\Delta${args.metric}", fontsize=5.5, labelpad=1.5)
    right.set_ylabel("frames", fontsize=5.5, labelpad=1.5)
    right.set_yscale("log")
    right.tick_params(labelsize=4.6, length=1.6, pad=1.2)
    right.text(0.03, 0.96, f"better {improved:.0f}%\nworse {worsened:.0f}%", transform=right.transAxes,
               fontsize=4.8, va="top")
    right.text(0.97, 0.96, f"{clipped} clipped", transform=right.transAxes, fontsize=4.0,
               ha="right", va="top", color="0.45")

    for ax in (left, right):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(0.4)

    figure.savefig(args.output, bbox_inches="tight", pad_inches=0.01)
    print(json.dumps({"status": "PASS", "output": str(args.output), "metric": args.metric,
                      "frames": int(len(raw)), "improved_pct": improved, "worsened_pct": worsened,
                      "mean_raw": float(raw.mean()), "mean_refined": float(refined.mean()),
                      "pooled_reduction_pct": float(100 * (raw.mean() - refined.mean()) / raw.mean()),
                      "source_report": str(args.report)}, indent=2))


if __name__ == "__main__":
    main()
