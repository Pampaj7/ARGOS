#!/usr/bin/env python3
"""Cache-resolution metrics (144x180) for every built (backbone, sequence) pair — reads
already-built caches only, no model inference.

Uses ALL frames per sequence, not a subsample: the strict coverage_threshold=0.9 GT mask
(see metrics.py) already discards most pixels per frame (~97/25920 measured on a real
frame), so a 30-frame subsample was leaving some sequences with as few as 1 usable frame
after filtering — a leaderboard built on n=1 per sequence is not trustworthy. Full-frame
sweep is I/O-bound (cache reads + native GT reads + resize), not compute-bound, so it's
affordable without a GPU."""
from __future__ import annotations

import csv
import multiprocessing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from argos_v2.backbones import BACKBONE_NAMES
from argos_v2.cache_io import is_complete
from argos_v2.metrics import compute_cache_metrics
from argos_v2.paths import CACHE_DIR, V2_ROOT
from argos_v2.scared_c_data import load_frame_gt, load_sequence_info
from argos_v2.sequences import accepted_sequences

OUT_CSV = V2_ROOT / "cache_scaredc_backbones/reports_full/backbone_metric_summary_cache.csv"


def _process_one(args: tuple[str, str]) -> dict | None:
    backbone, seq = args
    if not is_complete(backbone, seq):
        print(f"SKIP (not complete): {backbone}/{seq}", flush=True)
        return None
    d = CACHE_DIR / backbone / seq
    disp_cache = np.load(d / "disparity.npy", mmap_mode="r")
    valid_cache = np.load(d / "valid_mask.npy", mmap_mode="r")
    frame_ids = list(np.load(d / "frame_ids.npy"))
    info = load_sequence_info(seq)

    frame_metrics = []
    for k, fid in enumerate(frame_ids):
        gt_disp, gt_valid = load_frame_gt(info, fid)
        m = compute_cache_metrics(disp_cache[k], valid_cache[k], gt_disp, gt_valid, native_w=gt_disp.shape[1])
        if m["epe_cache_px"] is not None:
            frame_metrics.append(m)

    n_available = len(frame_ids)
    n_usable = len(frame_metrics)
    if not frame_metrics:
        print(f"{backbone}/{seq}: 0/{n_available} frames usable at coverage_threshold=0.9 — SKIPPED", flush=True)
        return None
    agg = {k: float(np.mean([m[k] for m in frame_metrics if m[k] is not None])) for k in frame_metrics[0]}
    agg.update({"backbone": backbone, "sequence": seq, "n_frames_available": n_available, "n_frames_usable": n_usable})
    print(f"{backbone}/{seq}: epe_cache_px={agg['epe_cache_px']:.3f} (n={n_usable}/{n_available})", flush=True)
    return agg


def main() -> int:
    jobs = [(b, s) for b in BACKBONE_NAMES for s in accepted_sequences()]
    with multiprocessing.Pool(min(32, len(jobs))) as pool:
        results = pool.map(_process_one, jobs)
    rows = [r for r in results if r is not None]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = ["backbone", "sequence", "n_frames_available", "n_frames_usable", "epe_cache_px", "bad1_cache", "bad3_cache",
            "absrel_cache", "valid_ratio_cache", "disparity_min_cache", "disparity_median_cache", "disparity_max_cache"]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_CSV}: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
