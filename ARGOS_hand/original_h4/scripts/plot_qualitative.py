#!/usr/bin/env python3
"""Render the qualitative figure from the dumped panels.

Reads what `dump_qualitative.py` wrote and produces a vector PDF sized for one IEEE
column. The row order is the selection order, so the harmful frame is last and visible
rather than omitted: a refiner paper that shows only its successes is not evidence.

Design constraints, in order of priority:
  * readable in greyscale, since disparity and error are both magnitudes;
  * one shared colour scale per column, so panels in a column are comparable;
  * the fusion weight on its own scale, because it is a weight and not a disparity;
  * no interpolation, so per-pixel structure is not smoothed away by the renderer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PANELS = ROOT.parent / "results" / "qualitative_panels"
ROWS = ("best", "median", "upper_quartile", "worst")
ROW_LABEL = {"best": "best", "median": "median", "upper_quartile": "upper quartile", "worst": "worst"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence", default="dataset_2_keyframe_4")
    parser.add_argument("--output", type=Path, default=ROOT.parents[1] / "ARGOS_hand/paper/qualitative.pdf")
    parser.add_argument("--width-in", type=float, default=3.4, help="IEEE single column")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = json.loads((PANELS / "run_manifest.json").read_text())
    data = {r: np.load(PANELS / f"{args.sequence}_{r}.npz", allow_pickle=False) for r in ROWS}

    # One disparity scale across every panel, and one error scale, so columns compare.
    disparity = np.concatenate([np.stack([d["raw"], d["fused"]])[np.isfinite(np.stack([d["raw"], d["fused"]]))]
                                for d in data.values()])
    dlo, dhi = np.percentile(disparity, [2, 98])
    # Diverging panels get their own symmetric scales, from the data, so a subtle
    # intervention is visible instead of washed out by an outlier-driven range.
    updates = np.concatenate([(d["fused"] - d["raw"])[np.isfinite(d["fused"] - d["raw"])].ravel()
                              for d in data.values()])
    uhi = float(np.percentile(np.abs(updates), 99)) or 1e-3
    changes = np.concatenate([
        np.where(d["support"], d["error_fused"] - d["error_raw"], np.nan)[
            np.isfinite(np.where(d["support"], d["error_fused"] - d["error_raw"], np.nan))].ravel()
        for d in data.values()])
    ehi = float(np.percentile(np.abs(changes), 99)) or 1e-3

    columns = ["rgb_left", "raw", "aligned_memory", "weight", "update", "error"]
    titles = ["left image", "raw $d^{\\mathrm{raw}}$", "memory $\\widetilde m$",
              "weight $w$", "update", "error change"]
    height = args.width_in / len(columns) * len(ROWS) * 1.28
    figure, axes = plt.subplots(len(ROWS), len(columns), figsize=(args.width_in, height),
                                gridspec_kw={"wspace": 0.04, "hspace": 0.06})

    for row, name in enumerate(ROWS):
        d = data[name]
        for column, key in enumerate(columns):
            ax = axes[row, column]
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.3)
            if key == "rgb_left":
                image = np.transpose(d["rgb_left"], (1, 2, 0))
                ax.imshow(np.clip(image / 255.0 if image.max() > 1.5 else image, 0, 1), interpolation="nearest")
            elif key == "weight":
                ax.imshow(np.ma.masked_invalid(d["weight"]), cmap="magma", vmin=0, vmax=1, interpolation="nearest")
            elif key == "update":
                # What the module actually did: fused minus raw, signed.
                ax.imshow(np.ma.masked_invalid(d["fused"] - d["raw"]), cmap="PuOr",
                          vmin=-uhi, vmax=uhi, interpolation="nearest")
            elif key == "error":
                # Did it help? coolwarm maps vmin to blue, so negative (error fell) is blue
                # and positive (error rose) is red.
                change = np.where(d["support"], d["error_fused"] - d["error_raw"], np.nan)
                ax.imshow(np.ma.masked_invalid(change), cmap="coolwarm", vmin=-ehi, vmax=ehi,
                          interpolation="nearest")
            else:
                ax.imshow(np.ma.masked_invalid(d[key]), cmap="viridis", vmin=dlo, vmax=dhi, interpolation="nearest")
            if row == 0:
                ax.set_title(titles[column], fontsize=4.2, pad=1.6)
        panel = manifest["panels"][name]
        axes[row, 0].set_ylabel(f"{ROW_LABEL[name]}\n$\\Delta$EPE {panel['delta_epe_px']:+.3f}", fontsize=4.0,
                                labelpad=1.5)

    figure.savefig(args.output, bbox_inches="tight", pad_inches=0.01)
    print(json.dumps({"status": "PASS", "output": str(args.output),
                      "disparity_scale_px": [float(dlo), float(dhi)],
                      "update_scale_px": uhi, "error_change_scale_px": ehi,
                      "rows": {r: manifest["panels"][r]["delta_epe_px"] for r in ROWS}}, indent=2))


if __name__ == "__main__":
    main()
