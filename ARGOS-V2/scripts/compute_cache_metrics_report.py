#!/usr/bin/env python3
"""Cache-resolution metrics (144x180) for every built (backbone, sequence) pair — reads
already-built caches only, no model inference. Samples ~30 frames/sequence for speed
(cache-res GT comparison is I/O-bound, not compute-bound, but full-density isn't needed
for a sanity/relative-comparison report)."""
from __future__ import annotations

import csv
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
SAMPLE_STRIDE_TARGET = 30  # ~ frames sampled per sequence


def main() -> int:
    rows = []
    for backbone in BACKBONE_NAMES:
        for seq in accepted_sequences():
            if not is_complete(backbone, seq):
                print(f"SKIP (not complete): {backbone}/{seq}", flush=True)
                continue
            d = CACHE_DIR / backbone / seq
            disp_cache = np.load(d / "disparity.npy", mmap_mode="r")
            valid_cache = np.load(d / "valid_mask.npy", mmap_mode="r")
            frame_ids = list(np.load(d / "frame_ids.npy"))
            info = load_sequence_info(seq)

            stride = max(1, len(frame_ids) // SAMPLE_STRIDE_TARGET)
            frame_metrics = []
            for k in range(0, len(frame_ids), stride):
                fid = frame_ids[k]
                gt_disp, gt_valid = load_frame_gt(info, fid)
                m = compute_cache_metrics(disp_cache[k], valid_cache[k], gt_disp, gt_valid, native_w=gt_disp.shape[1])
                if m["epe_cache_px"] is not None:
                    frame_metrics.append(m)

            if not frame_metrics:
                continue
            agg = {k: float(np.mean([m[k] for m in frame_metrics if m[k] is not None])) for k in frame_metrics[0]}
            agg.update({"backbone": backbone, "sequence": seq, "n_frames_sampled": len(frame_metrics)})
            rows.append(agg)
            print(f"{backbone}/{seq}: epe_cache_px={agg['epe_cache_px']:.3f} (n={len(frame_metrics)})", flush=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = ["backbone", "sequence", "n_frames_sampled", "epe_cache_px", "bad1_cache", "bad3_cache",
            "absrel_cache", "valid_ratio_cache", "disparity_min_cache", "disparity_median_cache", "disparity_max_cache"]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_CSV}: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
