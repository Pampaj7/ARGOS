#!/usr/bin/env python3
"""Native-resolution sanity metrics for all 5 backbones on the FIXED, deterministic
native_validation_subset.json (51 frames, 3/sequence x 17 sequences) — computed with NO
cache resizing at all, direct pixel-for-pixel comparison at source resolution. This is the
only place native-resolution numbers come from; never derive them from the cache.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from argos_v2.backbones import BACKBONE_NAMES, build_predictor
from argos_v2.metrics import compute_native_metrics
from argos_v2.paths import RESULTS_DIR, V2_ROOT
from argos_v2.scared_c_data import load_frame_gt, load_frame_lr, load_sequence_info

SUBSET_PATH = V2_ROOT / "cache_scaredc_backbones/reports_full/native_validation_subset.json"
OUT_CSV = V2_ROOT / "cache_scaredc_backbones/reports_full/native_resolution_sanity.csv"


def main() -> int:
    subset = json.loads(SUBSET_PATH.read_text())["sequences"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for backbone in BACKBONE_NAMES:
        print(f"=== {backbone} ===", flush=True)
        _method, _ckpt, predict = build_predictor(backbone, device)
        for seq, frame_ids in subset.items():
            info = load_sequence_info(seq)
            for frame_id in frame_ids:
                left, right = load_frame_lr(info, frame_id)
                gt_disp, gt_valid = load_frame_gt(info, frame_id)
                t0 = time.perf_counter()
                pred, _ms = predict(left, right)
                runtime_s = time.perf_counter() - t0
                m = compute_native_metrics(pred, gt_disp, gt_valid)
                m.update({"backbone": backbone, "sequence": seq, "frame_id": frame_id, "runtime_s": runtime_s})
                rows.append(m)
                print(f"  {seq}/{frame_id}: epe_native_px={m['epe_native_px']}", flush=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = ["backbone", "sequence", "frame_id", "epe_native_px", "bad1_native", "bad3_native",
            "valid_ratio_native", "native_height", "native_width", "runtime_s"]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # per-backbone summary print (sanity reference, e.g. S2M2-S ~1.07px on the pilot sequence)
    import numpy as np
    for backbone in BACKBONE_NAMES:
        vals = [r["epe_native_px"] for r in rows if r["backbone"] == backbone and r["epe_native_px"] is not None]
        print(f"{backbone}: mean epe_native_px over {len(vals)} frames = {np.mean(vals):.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
