#!/usr/bin/env python3
"""Model-free quality gate for SCARED-C corrected_temporal_gt: for every dataset_N/keyframe_M
video sequence, sample ~15 evenly-spaced co-registered frames, build their reprojected GT
disparity, then check photometric warp consistency (warp right image onto left using ONLY
the GT disparity — no stereo model involved) as an independent geometry-correctness signal.

A sequence whose corrected pose / scale-recovery went wrong will show elevated photometric
error even though pixel coverage (valid_pixel_ratio) can look perfectly healthy — coverage
only means points project somewhere in frame, not that they land in the RIGHT place.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.argos_paths import DATASET_DIR
from scripts.scared_c import build_corrected_temporal_gt as builder
from scripts.scared_c.build_corrected_temporal_gt import RAW

import cv2
import numpy as np

OUT_MANIFEST = DATASET_DIR / "SCARED-C/curated/manifests/quality_gate.csv"
SAMPLE_ROOT = DATASET_DIR / "SCARED-C/curated/geometric_gt/corrected_temporal_gt_quality_samples"
SAMPLE_N = 15
PASS_THRESHOLD = 15.0  # photometric MAE (0-255); dataset_3/keyframe_1 (known-good) ~5-8, dataset_1/keyframe_1 (known-broken) ~24-25


def photometric_consistency(seq: str, stem: str) -> tuple[float, float] | None:
    d = SAMPLE_ROOT / seq
    left = cv2.imread(str(d / "left" / f"{stem}.png"))
    right = cv2.imread(str(d / "right" / f"{stem}.png"))
    if left is None or right is None:
        return None
    left = left.astype(np.float32)
    right = right.astype(np.float32)
    disp = np.load(d / "gt" / f"{stem}_disp.npy")
    valid = cv2.imread(str(d / "gt" / f"{stem}_valid.png"), cv2.IMREAD_GRAYSCALE) > 0
    h, w = disp.shape
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (xs - disp).astype(np.float32)
    map_y = ys.astype(np.float32)
    warped_right = cv2.remap(right, map_x, map_y, cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    in_bounds = (map_x >= 0) & (map_x < w)
    m = valid & in_bounds
    if m.sum() == 0:
        return None
    err = np.abs(left - warped_right).mean(axis=2)
    return float(err[m].mean()), float(m.mean())


def main() -> int:
    sequences = []
    for dataset_dir in sorted(RAW.glob("dataset_*"), key=lambda p: int(p.name.split("_")[1])):
        for kf_dir in sorted(dataset_dir.glob("keyframe_*"), key=lambda p: int(p.name.split("_")[1])):
            if (kf_dir / "data" / "frame_data.tar.gz").exists():
                sequences.append((dataset_dir.name, kf_dir.name))

    rows = []
    for dataset_id, keyframe_id in sequences:
        seq = f"{dataset_id}_{keyframe_id}"
        print(f"=== {seq} ===", flush=True)
        frame_log = json.loads((RAW / dataset_id / keyframe_id / "frame_log.json").read_text())
        included_count_full = frame_log["included_count"]  # true co-registered count, before sampling
        builder.OUT = SAMPLE_ROOT
        summary = builder.build_sequence(dataset_id, keyframe_id, sample_n=SAMPLE_N)
        manifest = list(csv.DictReader(open(SAMPLE_ROOT / seq / "manifest.csv")))
        photo_errs, valid_ratios = [], []
        for r in manifest:
            res = photometric_consistency(seq, r["frame_id"])
            if res:
                photo_errs.append(res[0])
                valid_ratios.append(float(r["valid_pixel_ratio"]))
        if not photo_errs:
            rows.append({"sequence_id": seq, "n_sampled": summary["n_frames"], "included_count_full": included_count_full,
                         "photometric_mae_median": None, "valid_pixel_ratio_mean": None, "status": "no_valid_pixels"})
            continue
        mae_med = float(np.median(photo_errs))
        status = "pass" if mae_med < PASS_THRESHOLD else "fail"
        rows.append({
            "sequence_id": seq, "n_sampled": summary["n_frames"], "included_count_full": included_count_full,
            "photometric_mae_median": mae_med, "photometric_mae_max": float(np.max(photo_errs)),
            "valid_pixel_ratio_mean": float(np.mean(valid_ratios)), "status": status,
        })
        print(f"{seq}: photometric_mae_median={mae_med:.2f} status={status}", flush=True)

    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with OUT_MANIFEST.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
