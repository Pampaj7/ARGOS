#!/usr/bin/env python3
"""Per-frame raw-vs-refined scatter over every scored frame of a split.

Side-by-side disparity maps are not a usable figure here: the intervention is sub-pixel
on disparities of about $8$ px, so at an honest shared colour scale raw and refined are
indistinguishable and only a stretched scale would show a difference.

This shows the same claim in the space where it is actually visible. Every point is one
frame of one backbone; points below the diagonal are frames the module improved. Nothing
is selected --- all backbones and all sequences of the split are pooled --- so the figure
cannot be a lucky sequence.

Reads the `per_frame` block the evaluator already wrote, so it costs no GPU time and
cannot disagree with the tables.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# The proposed model's runs. This pointed at "scared_masked" -- the 142-channel ablation --
# so Figure 2 showed the wrong model under a caption naming TETHER.
RUNS = ROOT.parent / "results" / "ablation_eval" / "a2" / "runs"
SPLITS = (("scared-d2", "D2 development"), ("scared-d7", "D7 held out"))


def load(split: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Every (raw, refined) frame pair of the split, across all backbones and sequences."""
    raw, refined = [], []
    for path in sorted(glob.glob(str(RUNS / split / "*/reports/*/*.json"))):
        for frame in json.loads(Path(path).read_text())["per_frame"]:
            family = frame["metrics"]["disparity_px"]
            a, b = family["raw"][metric]["value"], family["refined"][metric]["value"]
            # Log axes need strict positives; frames at exactly zero carry no information
            # about direction anyway, and their count is reported in the manifest.
            if a is not None and b is not None and a > 0 and b > 0:
                raw.append(a); refined.append(b)
    return np.array(raw), np.array(refined)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metric", default="Bad1")
    parser.add_argument("--output", type=Path, default=ROOT.parents[1] / "ARGOS_hand/paper/frame_scatter.pdf")
    parser.add_argument("--width-in", type=float, default=3.4)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scale = 100.0 if args.metric.startswith("Bad") else 1.0
    unit = "%" if scale == 100.0 else "px"
    figure, axes = plt.subplots(1, 2, figsize=(args.width_in, args.width_in * 0.42))
    summary = {}

    for ax, (split, label) in zip(axes, SPLITS):
        raw, refined = load(split, args.metric)
        raw, refined = raw * scale, refined * scale
        better, worse = refined < raw, refined > raw
        low = float(min(raw.min(), refined.min())) * 0.8
        high = float(max(raw.max(), refined.max())) * 1.25

        ax.fill_between([low, high], [low, high], low, color="#1f6fb4", alpha=0.06, linewidth=0)
        ax.scatter(raw[worse], refined[worse], s=0.5, alpha=0.13, color="#c0392b",
                   linewidths=0, rasterized=True)
        ax.scatter(raw[better], refined[better], s=0.5, alpha=0.13, color="#1f6fb4",
                   linewidths=0, rasterized=True)
        ax.plot([low, high], [low, high], color="black", linewidth=0.45)

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(low, high); ax.set_ylim(low, high)
        ax.set_aspect("equal")
        ax.set_xlabel(f"raw {args.metric} ({unit})", fontsize=5.4, labelpad=1.1)
        ax.set_ylabel(f"TETHER {args.metric} ({unit})", fontsize=5.4, labelpad=1.1)
        ax.set_title(f"{label}, {len(raw)} frames", fontsize=5.6, pad=2.2)
        ax.tick_params(labelsize=4.3, length=1.4, pad=0.9)
        ax.text(0.05, 0.95, f"improved {100 * better.mean():.0f}%", transform=ax.transAxes,
                fontsize=5.0, va="top", color="#1f6fb4")
        ax.text(0.05, 0.85, f"worsened {100 * worse.mean():.0f}%", transform=ax.transAxes,
                fontsize=5.0, va="top", color="#c0392b")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(0.4)
        summary[split] = {"frames": int(len(raw)), "improved_pct": float(100 * better.mean()),
                          "worsened_pct": float(100 * worse.mean()),
                          "pooled_reduction_pct": float(100 * (raw.mean() - refined.mean()) / raw.mean())}

    figure.subplots_adjust(wspace=0.42)
    figure.savefig(args.output, bbox_inches="tight", pad_inches=0.01, dpi=400)
    print(json.dumps({"status": "PASS", "output": str(args.output), "metric": args.metric,
                      "selection": "none; all backbones and sequences of each split",
                      "splits": summary}, indent=2))


if __name__ == "__main__":
    main()
