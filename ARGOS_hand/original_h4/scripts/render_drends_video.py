#!/usr/bin/env python3
"""Render the DRENDS attachment video from dumped frames.

Four acts, about ninety seconds, well inside the call's 180 s and 20 MB.

  1. RGB, raw disparity, TETHER disparity. The point is motion: a frame-wise
     estimator flickers between frames and the refined stream does not.
  2. |raw - GT| against |TETHER - GT| on one colour scale, with the running EPE.
     This act exists because act 1 alone argues for smoothness, and Sec. V-E of
     the paper shows a degenerate replay wins any smoothness metric outright. The
     video must show error falling, not just jitter falling, or it makes the
     argument we spent a section refuting.
  3. Signed change in absolute error, diverging about zero: blue where refinement
     helped, red where it hurt. The red is not cropped out. The paper reports
     introduced error beside recovered error and the video does the same.
  4. All five recordings, raw against TETHER. All five, not the best four --
     excluding one would exclude Vid13, which Sec. V-I already names as the
     recording where our intervention is a net harm on BiDA's support.

Colour scales are computed once per recording over every frame and then held
fixed, so nothing in the picture is autoscaling between frames. The hero
recording is chosen by a written rule -- the median of the five by EPE
reduction -- and the rule is printed on screen.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Every act draws on the same canvas. ffmpeg's image2 demuxer needs a constant
# frame size, and per-act figsizes gave four different heights.
FIG = (10, 3.6)
FRAMES = ROOT.parent / "results" / "drends_video_frames"
FPS = 15


def load(path: Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _percentiles(values: np.ndarray, mask: np.ndarray, lo: float, hi: float) -> tuple[float, float]:
    sample = values[mask]
    if sample.size == 0:
        return 0.0, 1.0
    return float(np.percentile(sample, lo)), float(np.percentile(sample, hi))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=Path, default=FRAMES)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "paper" / "tether_attachment.mp4")
    parser.add_argument("--seconds", type=float, nargs=4, default=[20, 30, 20, 20],
                        help="duration of each act")
    parser.add_argument("--dpi", type=int, default=110)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    index = json.loads((args.frames / "index.json").read_text())
    per = index["recordings"]
    order = sorted(per, key=lambda r: per[r]["reduction_pct"])
    hero = order[len(order) // 2]
    print(f"hero recording (median of {len(order)} by EPE reduction): {hero}")

    backbone = index["backbone"]
    data = {r: load(args.frames / f"{backbone}__{r}.npz") for r in order}

    work = args.output.parent / "_video_frames"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    plt.rcParams.update({"font.size": 8, "text.color": "0.9",
                         "axes.labelcolor": "0.9", "figure.facecolor": "0.08"})

    def cm(name):
        # Invalid reference is masked, and white made the holes louder than the data.
        return matplotlib.colormaps[name].with_extremes(bad="0.16")

    def panel(ax, image, cmap, vmin, vmax, title):
        ax.imshow(image, cmap=cm(cmap) if isinstance(cmap, str) else cmap,
                  vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("0.35")
        ax.set_title(title, fontsize=8, pad=3)

    d = data[hero]
    valid = d["gt_valid"] & np.isfinite(d["gt"].astype(np.float32)) & (d["gt"].astype(np.float32) > 0)
    raw = d["raw"].astype(np.float32); ref = d["refined"].astype(np.float32)
    gt = d["gt"].astype(np.float32)
    dlo, dhi = _percentiles(np.concatenate([raw[valid], ref[valid]]),
                            np.ones(valid.sum() * 2, bool), 2, 98)
    err_raw = np.abs(raw - gt); err_ref = np.abs(ref - gt)
    _, ehi = _percentiles(err_raw, valid, 0, 97)
    cmax = float(np.percentile(np.abs(ref - raw)[valid], 99))
    delta = err_ref - err_raw
    dmax = float(np.percentile(np.abs(delta[valid]), 99))

    n = len(raw)
    counter = [0]

    def save(fig):
        fig.savefig(work / f"{counter[0]:06d}.png", dpi=args.dpi,
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        counter[0] += 1

    def indices(seconds):
        want = int(round(seconds * FPS))
        return np.linspace(0, n - 1, want).astype(int)

    # --- Title: an attachment that does not name the model is a slideshow ---
    for _ in range(int(round(2.5 * FPS))):
        fig = plt.figure(figsize=FIG)
        fig.text(0.5, 0.62, "TETHER", ha="center", size=26, weight="bold", color="#e8a765")
        fig.text(0.5, 0.47, "a causal temporal refiner for frozen stereo",
                 ha="center", size=12, color="0.88")
        fig.text(0.5, 0.34, "one $154{,}874$-parameter checkpoint, trained once, attached "
                            "unchanged to a frozen estimator", ha="center", size=9, color="0.72")
        fig.text(0.5, 0.24, f"shown here on {backbone} over DRENDS -- zero-shot: no part of "
                            "the module has seen this domain", ha="center", size=9, color="0.72")
        save(fig)

    # --- Act 1: what a frame-wise estimator does between frames -------------
    for t in indices(args.seconds[0]):
        fig = plt.figure(figsize=FIG)
        gs = gridspec.GridSpec(1, 4, wspace=0.04, left=0.015, right=0.985, top=0.84, bottom=0.03)
        panel(fig.add_subplot(gs[0]), d["rgb"][t].transpose(1, 2, 0), None, None, None,
              f"{hero}  ({backbone})")
        panel(fig.add_subplot(gs[1]), np.where(valid[t], raw[t], np.nan), "turbo", dlo, dhi,
              "frozen estimator")
        panel(fig.add_subplot(gs[2]), np.where(valid[t], ref[t], np.nan), "turbo", dlo, dhi,
              "+ TETHER")
        panel(fig.add_subplot(gs[3]), np.where(valid[t], np.abs(ref[t] - raw[t]), np.nan),
              "viridis", 0, cmax, "what TETHER changed")
        fig.suptitle("the two depth maps differ by about five percent: the change itself "
                     "is the fourth panel", fontsize=9, y=0.975)
        save(fig)

    # --- Act 2: error against the reference, not smoothness -----------------
    run_raw, run_ref = [], []
    for t in indices(args.seconds[1]):
        m = valid[t]
        run_raw.append(float(err_raw[t][m].mean()) if m.any() else np.nan)
        run_ref.append(float(err_ref[t][m].mean()) if m.any() else np.nan)
        fig = plt.figure(figsize=FIG)
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.20, left=0.02, right=0.97, top=0.86, bottom=0.13)
        panel(fig.add_subplot(gs[0]), np.where(m, err_raw[t], np.nan), "inferno", 0, ehi,
              "|frozen - reference|")
        panel(fig.add_subplot(gs[1]), np.where(m, err_ref[t], np.nan), "inferno", 0, ehi,
              "|TETHER - reference|")
        ax = fig.add_subplot(gs[2]); ax.set_facecolor("0.12")
        ax.plot(run_raw, color="#f08c3a", lw=1.1, label="frozen")
        ax.plot(run_ref, color="#4fb3d9", lw=1.1, label="+ TETHER")
        ax.set_xlim(0, max(60, len(run_raw))); ax.set_ylabel("frame EPE (px)", fontsize=7, labelpad=1)
        ax.legend(loc="upper right", frameon=False, fontsize=7)
        ax.tick_params(colors="0.7", labelsize=7)
        for s in ax.spines.values():
            s.set_color("0.35")
        fig.suptitle("smoothness is not correctness: the error against the reference falls",
                     fontsize=9, y=0.985)
        save(fig)

    # --- Act 3: where it helped, and where it did not -----------------------
    for t in indices(args.seconds[2]):
        m = valid[t]
        fig = plt.figure(figsize=FIG)
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.1], wspace=0.06, left=0.06, right=0.94, top=0.86, bottom=0.03)
        panel(fig.add_subplot(gs[0]), d["rgb"][t].transpose(1, 2, 0), None, None, None, hero)
        ax = fig.add_subplot(gs[1])
        im = ax.imshow(np.where(m, -delta[t], np.nan), cmap=cm("RdBu"), vmin=-dmax, vmax=dmax,
                       interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("0.35")
        ax.set_title("blue: error recovered      red: error introduced", fontsize=8, pad=3)
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
        cb.ax.tick_params(colors="0.7", labelsize=6)
        fig.suptitle("the intervention is selective, and it is not free everywhere",
                     fontsize=9, y=0.985)
        save(fig)

    # --- Act 4: every recording, not the flattering ones --------------------
    scales = {}
    for r in order:
        e = data[r]
        v = e["gt_valid"] & (e["gt"].astype(np.float32) > 0)
        a = np.abs(e["raw"].astype(np.float32) - e["gt"].astype(np.float32))
        scales[r] = (0.0, float(np.percentile(a[v], 97)) if v.any() else 1.0)
    shortest = min(len(data[r]["raw"]) for r in order)
    for t in np.linspace(0, shortest - 1, int(round(args.seconds[3] * FPS))).astype(int):
        fig = plt.figure(figsize=FIG)
        gs = gridspec.GridSpec(2, len(order), hspace=0.14, wspace=0.04, left=0.045, right=0.99, top=0.87, bottom=0.02)
        for c, r in enumerate(order):
            e = data[r]
            v = e["gt_valid"][t] & (e["gt"][t].astype(np.float32) > 0)
            g = e["gt"][t].astype(np.float32)
            lo, hi = scales[r]
            panel(fig.add_subplot(gs[0, c]),
                  np.where(v, np.abs(e["raw"][t].astype(np.float32) - g), np.nan),
                  "inferno", lo, hi, r.split("_")[0])
            panel(fig.add_subplot(gs[1, c]),
                  np.where(v, np.abs(e["refined"][t].astype(np.float32) - g), np.nan),
                  "inferno", lo, hi, None)
        fig.text(0.012, 0.72, "frozen", rotation=90, va="center", fontsize=8)
        fig.text(0.012, 0.28, "+ TETHER", rotation=90, va="center", fontsize=8)
        fig.suptitle("all five recordings, absolute error against the reference "
                     "(no recording excluded)", fontsize=9, y=0.98)
        save(fig)

    print(f"{counter[0]} frames rendered")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # This site's ffmpeg has no libx264; libopenh264 is the available H.264 encoder
    # and takes a bitrate rather than a CRF. 1400k over ninety seconds lands near
    # 15 MB, inside the call's 20 MB.
    cmd = ["ffmpeg", "-y", "-framerate", str(FPS), "-i", str(work / "%06d.png"),
           "-c:v", "libopenh264", "-b:v", "1400k", "-pix_fmt", "yuv420p",
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(args.output)]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode:
        print(done.stderr[-1500:])
        raise SystemExit("ffmpeg failed")
    size = args.output.stat().st_size / 1e6
    seconds = counter[0] / FPS
    print(f"{args.output}: {size:.1f} MB, {seconds:.0f} s "
          f"({'OK' if size <= 20 and seconds <= 180 else 'OVER THE CALL LIMITS'})")


if __name__ == "__main__":
    main()
