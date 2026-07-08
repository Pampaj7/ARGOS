#!/usr/bin/env python3
"""S2M2-S zero-shot eval on the 45 (fixed-rectification) SCARED strong_keyframes_rectified anchors."""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.argos_paths import DATASET_DIR, RESULTS_DIR

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "temporal_refinement/data_prep"))
from predict_s2m2_long_sequences import build_model, infer  # noqa: E402

SRC = DATASET_DIR / "SCARED/curated/geometric_gt/strong_keyframes_rectified"
OUT = RESULTS_DIR / "01_frame_stereo/SCARED/s2m2_s_strong_keyframes_rectified"
WIDTH = 512


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device, "S")

    rows = []
    t0 = time.time()
    for dataset_dir in sorted(SRC.glob("dataset_*"), key=lambda p: int(p.name.split("_")[1])):
        for kf_dir in sorted(dataset_dir.glob("keyframe_*"), key=lambda p: int(p.name.split("_")[1])):
            left = read_rgb(kf_dir / "left_rectified.png")
            right = read_rgb(kf_dir / "right_rectified.png")
            gt_disp = np.load(kf_dir / "gt_disparity.npy").astype(np.float32)
            gt_depth = np.load(kf_dir / "gt_depth.npy").astype(np.float32)
            valid = cv2.imread(str(kf_dir / "valid_mask.png"), cv2.IMREAD_GRAYSCALE) > 0
            calib = json.loads((kf_dir / "calibration.json").read_text())
            fx, baseline_mm = calib["fx"], calib["baseline_mm"]

            pred, runtime_ms, _scale = infer(model, left, right, device, WIDTH)

            m = valid & np.isfinite(pred) & (gt_disp > 0) & np.isfinite(gt_disp)
            e = np.abs(pred[m] - gt_disp[m])
            pred_depth = fx * baseline_mm / np.maximum(pred, 1e-6)
            depth_e = np.abs(pred_depth[m] - gt_depth[m])
            rel = depth_e / np.maximum(gt_depth[m], 1e-6)
            row = {
                "dataset_id": dataset_dir.name,
                "keyframe_id": kf_dir.name,
                "valid_px": int(m.sum()),
                "epe_px": float(e.mean()),
                "rmse_px": float(np.sqrt((e ** 2).mean())),
                "bad3_pct": float((e > 3).mean() * 100),
                "abs_rel": float(rel.mean()),
                "depth_mae_mm": float(depth_e.mean()),
                "delta1_pct": float((np.maximum(pred_depth[m] / gt_depth[m], gt_depth[m] / pred_depth[m]) < 1.25).mean() * 100),
                "runtime_ms": runtime_ms,
            }
            rows.append(row)
            print(f"{row['dataset_id']}/{row['keyframe_id']}: EPE={row['epe_px']:.3f} AbsRel={row['abs_rel']:.4f} d1={row['delta1_pct']:.2f}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with (OUT / "per_keyframe.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    mean = {k: float(np.mean([r[k] for r in rows])) for k in cols if k not in ("dataset_id", "keyframe_id", "valid_px")}
    (OUT / "summary.json").write_text(json.dumps(mean, indent=2))
    print(json.dumps({"n": len(rows), "elapsed_s": time.time() - t0, **mean}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
