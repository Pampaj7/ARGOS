#!/usr/bin/env python3
"""Render the overview figure's disparity panels, and report what the caption quotes.

The panels were originally produced by a snippet that lived only in a commit message, so
when the proposed model changed there was no way to regenerate them and the paper's opening
figure kept showing the ablation. This is that snippet, as a script.

Two things are load-bearing. The colour scale is shared between the raw and fused rows: the
honest fact is that the two rows look nearly identical, and normalising them separately
would manufacture a visible difference that is not in the data. And a single plane is
removed from all ten panels.

That plane needs justifying, because it means the panels are not absolute disparity. The
liver surface at this cache resolution is genuinely smooth -- a global depth ramp of about
seven pixels with roughly one pixel of anatomy on top -- so an absolute rendering is a
featureless gradient in which none of the scene is visible. Choosing different frames does
not help; the whole of D2 looks like this, which was checked rather than assumed. The plane
is fitted once, on the first raw frame shown, and subtracted from every panel including the
fused ones, so the rows stay comparable and nothing that differs between them is an artefact
of a refitted trend. It also leaves the caption's mean |fused - raw| untouched, since the
same constant surface leaves both sides of a difference.

`--verify` re-renders from whatever `fused_window.npz` currently holds and compares against
the PNGs on disk, which is how this script was checked against the panels it replaces.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARGOS = ROOT.parents[1]
BOUNDARY = (ARGOS / "ARGOS_hand/external_comparison/results/bidastabilizer_raftstereo_robust"
            / "d2_full/dataset_2_keyframe_4")
ASSETS = ARGOS / "ARGOS_hand/paper/figure_assets"
SHOWN = range(1100, 1105)          # the five consecutive frames in the figure


def render(values: np.ndarray, limit: float) -> np.ndarray:
    from matplotlib import cm
    scaled = np.clip((values + limit) / (2.0 * limit), 0.0, 1.0)
    return (cm.RdBu_r(scaled) * 255.0 + 0.5).astype(np.uint8)


def fitted_plane(reference: np.ndarray) -> np.ndarray:
    """The global depth ramp, fitted once and subtracted from every panel."""
    rows, columns = np.mgrid[0:reference.shape[0], 0:reference.shape[1]]
    design = np.c_[columns.ravel(), rows.ravel(), np.ones(reference.size)]
    coefficients, _residuals, _rank, _sv = np.linalg.lstsq(design, reference.ravel(), rcond=None)
    return (design @ coefficients).reshape(reference.shape)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true",
                        help="compare against the PNGs on disk instead of overwriting them")
    args = parser.parse_args()
    from PIL import Image

    window = np.load(ASSETS / "fused_window.npz")
    fused_all, start = window["fused"], int(window["start"])
    raw_all = np.load(BOUNDARY / "raw.npz", allow_pickle=False)["raw_disparity"][:, 0]

    fused = np.stack([fused_all[i - start] for i in SHOWN]).astype(np.float64)
    raw = np.stack([raw_all[i] for i in SHOWN]).astype(np.float64)

    plane = fitted_plane(raw[0])
    raw, fused = raw - plane, fused - plane

    limit = float(np.percentile(np.abs(np.concatenate([raw.ravel(), fused.ravel()])), 98))
    # The window, not the five shown frames: the caption speaks of the driven window. Taken
    # before detrending, though it is identical after it.
    delta = float(np.abs(fused_all - raw_all[start:start + len(fused_all)]).mean())

    print(f"shared symmetric colour scale +/-{limit:.2f} px after removing one fitted plane")
    print(f"mean |fused - raw| over the {len(fused_all)}-frame window: {delta:.4f} px")
    print(f"                   over the five shown frames:            {np.abs(fused - raw).mean():.4f} px")

    worst = 0.0
    for k, index in enumerate(SHOWN):
        for name, values in (("disp", raw[k]), ("fused", fused[k])):
            image = render(values, limit)
            path = ASSETS / f"{name}_{k}.png"
            if args.verify:
                existing = np.array(Image.open(path))
                diff = np.abs(existing.astype(int) - image.astype(int)).max()
                worst = max(worst, diff)
                print(f"  {path.name}: max channel difference {diff}")
            else:
                Image.fromarray(image, mode="RGBA").save(path)
    if args.verify:
        print(f"PASS: renderer reproduces the panels (worst channel difference {worst})"
              if worst <= 1 else
              f"MISMATCH: worst channel difference {worst}; the renderer is not the original")
    else:
        print(f"wrote {2 * len(list(SHOWN))} panels to {ASSETS}")


if __name__ == "__main__":
    main()
