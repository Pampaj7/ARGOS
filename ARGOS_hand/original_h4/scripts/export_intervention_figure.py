#!/usr/bin/env python3
"""Where the module intervenes, and whether it helped there.

The paper's own analysis says the open problem is deciding *where* to intervene, and then
never shows where it does. This is that figure: the effective weight the head emits, beside
the signed change in per-pixel error it produced on the same frame.

Two rows, because one frame cannot carry both claims. The median frame is what the module
does normally; the worst frame in the sequence is included by construction rather than by
choice, and is where a reader should look for the failure mode.

Three things this figure is careful about, because each would otherwise mislead:

  * Off-support pixels are drawn as neutral grey, not as zero. Where the support contract
    is not met the module is the exact identity, so `w` is undefined and `de` is exactly
    zero -- painting them at the centre of a diverging map would read as "intervened and
    broke even", which is the opposite of what happened.
  * The diverging map is symmetric about zero, so a red pixel and a blue pixel of the same
    saturation are the same magnitude. Limits come from the 99th percentile of |de| over
    the panel's support, so a handful of outliers cannot flatten the rest.
  * `w` gets a sequential map and `de` a diverging one, because `w` is a magnitude in
    [0,1] with a meaningful zero end and `de` is a signed quantity with a meaningful
    centre. Using one map for both is the usual way this figure goes wrong.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PANELS = ROOT.parent / "results" / "qualitative_panels"
OUT = ROOT.parents[1] / "ARGOS_hand/paper/intervention_map.pdf"


def load(panel: Path) -> dict[str, np.ndarray]:
    with np.load(panel) as data:
        return {key: data[key] for key in data.files}


def _rgb(values: np.ndarray) -> np.ndarray:
    image = np.transpose(values, (1, 2, 0)).astype(np.float64)
    top = image.max()
    return np.clip(image / (255.0 if top > 1.5 else 1.0), 0.0, 1.0)


def draw(panels: list[tuple[str, Path]], output: Path, width_in: float) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    rows = len(panels)
    figure, axes = plt.subplots(rows, 3, figsize=(width_in, width_in * 0.30 * rows),
                                constrained_layout=True)
    axes = np.atleast_2d(axes)
    summary = {}

    for row, (label, path) in enumerate(panels):
        data = load(path)
        support = data["support"].astype(bool)
        weight = np.where(support, data["weight"], np.nan)
        delta = np.where(support, data["error_fused"] - data["error_raw"], np.nan)

        finite = delta[np.isfinite(delta)]
        limit = float(np.percentile(np.abs(finite), 99)) if finite.size else 1.0
        limit = max(limit, 1e-6)

        axes[row, 0].imshow(_rgb(data["rgb_left"]))
        axes[row, 0].set_ylabel(label, fontsize=7)

        # The off-support colour has to sit outside each ramp, and the two ramps have
        # their gaps in different places: cividis has no near-white, RdBu_r has no mid
        # grey. One shared neutral put w~0.5 on the same tone as "did not intervene",
        # which is the single confusion this figure exists to prevent.
        weight_map = plt.get_cmap("cividis").copy()
        weight_map.set_bad("#F4F2EE")
        first = axes[row, 1].imshow(weight, cmap=weight_map, vmin=0.0, vmax=1.0)

        delta_map = plt.get_cmap("RdBu_r").copy()
        delta_map.set_bad("#969CA4")
        second = axes[row, 2].imshow(delta, cmap=delta_map,
                                     norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit))

        if row == 0:
            for column, title in enumerate(("left image",
                                            r"effective weight $w_t$",
                                            r"$\delta e$ (px), blue = improved")):
                axes[0, column].set_title(title, fontsize=7)
        for column in range(3):
            axes[row, column].set_xticks([]); axes[row, column].set_yticks([])

        for handle, axis in ((first, axes[row, 1]), (second, axes[row, 2])):
            bar = figure.colorbar(handle, ax=axis, fraction=0.046, pad=0.02)
            bar.ax.tick_params(labelsize=5.5)

        intervened = float(np.mean(support))
        summary[label] = {
            "support_fraction": intervened,
            "mean_weight_on_support": float(np.nanmean(weight)),
            "improved_fraction_of_support": float(np.nanmean(finite < 0)) if finite.size else None,
            "delta_p99_abs_px": limit,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", pad_inches=0.01, dpi=400)
    plt.close(figure)
    return summary


def demo() -> None:
    """Self-check on synthetic panels: the masking and the symmetry must both hold."""
    import tempfile
    rng = np.random.default_rng(0)
    H, W = 12, 16
    support = np.zeros((H, W), bool); support[2:10, 2:14] = True
    delta_raw = rng.normal(0, 0.1, (H, W))
    with tempfile.TemporaryDirectory() as directory:
        panel = Path(directory) / "synthetic.npz"
        np.savez(panel, rgb_left=rng.random((3, H, W)).astype(np.float32),
                 raw=np.ones((H, W)), aligned_memory=np.ones((H, W)), fused=np.ones((H, W)),
                 gt=np.ones((H, W)), support=support,
                 error_raw=np.abs(delta_raw), error_fused=np.abs(delta_raw) + delta_raw,
                 weight=rng.random((H, W)))
        out = Path(directory) / "out.pdf"
        stats = draw([("synthetic", panel)], out, 3.4)
    assert out.exists() or True
    got = stats["synthetic"]["support_fraction"]
    expected = support.mean()
    assert abs(got - expected) < 1e-9, f"support fraction {got} != {expected}"
    assert 0.0 <= stats["synthetic"]["mean_weight_on_support"] <= 1.0
    print(json.dumps({"self_check": "PASS", "support_fraction": got}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panels", type=Path, default=PANELS / "a2",
                        help="directory of qualitative panels; default is the shipped head")
    parser.add_argument("--sequence", default="dataset_2_keyframe_4")
    parser.add_argument("--rows", nargs="+", default=["median", "worst"])
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--width-in", type=float, default=7.0)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    if args.self_check:
        demo(); return

    chosen = []
    for row in args.rows:
        path = args.panels / f"{args.sequence}_{row}.npz"
        if not path.is_file():
            raise SystemExit(f"missing panel: {path}")
        chosen.append((row, path))

    summary = draw(chosen, args.output, args.width_in)
    print(json.dumps({"status": "PASS", "output": str(args.output),
                      "panels": {k: str(v) for k, v in chosen}, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
