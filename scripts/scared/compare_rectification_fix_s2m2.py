#!/usr/bin/env python3
"""Compare S2M2-S eval metrics on the 45 SCARED keyframes using the OLD buggy
cv2.remap-based GT rectification vs the FIXED scatter_min_depth (R1-rotation + z-buffer)
rectification, to quantify the real impact of the bug found and fixed in this session.

Same rectified left/right images and same S2M2 prediction either way (only the GT
computation differed) so this reruns inference once and scores it against both GTs.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.argos_paths import DATASET_DIR, RESULTS_DIR
from scripts.scared.convert_scared_keyframes import scatter_min_depth

import cv2
import numpy as np
import tifffile
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "temporal_refinement/data_prep"))
from predict_s2m2_long_sequences import build_model, infer  # noqa: E402

RAW = DATASET_DIR / "SCARED/curated/geometric_gt/strong_keyframes"
OUT = RESULTS_DIR / "01_frame_stereo/SCARED/s2m2_s_rectification_bug_comparison"
WIDTH = 512


def load_calib(path: Path):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    out = {name: fs.getNode(name).mat() for name in ["M1", "D1", "M2", "D2", "R", "T"]}
    fs.release()
    return out


def rectify_both(left, right, xyz, calib_path: Path):
    h, w = left.shape[:2]
    calib = load_calib(calib_path)
    r1, r2, p1, p2, _q, _roi1, _roi2 = cv2.stereoRectify(
        calib["M1"], calib["D1"], calib["M2"], calib["D2"], (w, h),
        calib["R"], calib["T"].reshape(3, 1), flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )
    map1x, map1y = cv2.initUndistortRectifyMap(calib["M1"], calib["D1"], r1, p1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(calib["M2"], calib["D2"], r2, p2, (w, h), cv2.CV_32FC1)
    left_r = cv2.remap(left, map1x, map1y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    right_r = cv2.remap(right, map2x, map2y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    fx = float(p1[0, 0])
    baseline_mm = float(abs(p2[0, 3] / p2[0, 0]))

    # OLD (buggy): cv2.remap the Z-channel like an image
    z = xyz[..., 2].astype(np.float32)
    valid = (np.isfinite(xyz).all(axis=-1) & (z > 0)).astype(np.uint8)
    z_clean = np.where(valid > 0, z, 0).astype(np.float32)
    z_old = cv2.remap(z_clean, map1x, map1y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    valid_old = cv2.remap(valid, map1x, map1y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT).astype(bool)
    valid_old &= z_old > 0
    disp_old = fx * baseline_mm / np.maximum(z_old, 1e-6)

    # NEW (fixed): rotate points by R1, project via P1/P2 with z-buffer scatter
    pts_rot = (xyz.reshape(-1, 3) @ r1.T).reshape(h, w, 3)
    depth_new, disp_new = scatter_min_depth(pts_rot, p1, p2, (h, w))
    valid_new = depth_new > 0

    return {
        "left_r": left_r, "right_r": right_r, "fx": fx, "baseline_mm": baseline_mm,
        "old": {"disp": disp_old, "depth": z_old, "valid": valid_old},
        "new": {"disp": disp_new, "depth": depth_new, "valid": valid_new},
    }


def score(pred, gt_disp, gt_depth, valid, fx, baseline_mm):
    m = valid & np.isfinite(pred) & (gt_disp > 0) & np.isfinite(gt_disp)
    e = np.abs(pred[m] - gt_disp[m])
    pred_depth = fx * baseline_mm / np.maximum(pred, 1e-6)
    depth_e = np.abs(pred_depth[m] - gt_depth[m])
    rel = depth_e / np.maximum(gt_depth[m], 1e-6)
    return {
        "valid_px": int(m.sum()),
        "valid_ratio": float(valid.mean()),
        "epe_px": float(e.mean()),
        "bad3_pct": float((e > 3).mean() * 100),
        "abs_rel": float(rel.mean()),
        "depth_mae_mm": float(depth_e.mean()),
        "delta1_pct": float((np.maximum(pred_depth[m] / gt_depth[m], gt_depth[m] / pred_depth[m]) < 1.25).mean() * 100),
    }


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device, "S")

    old_rows, new_rows = [], []
    t0 = time.time()
    for dataset_dir in sorted(RAW.glob("dataset_*"), key=lambda p: int(p.name.split("_")[1])):
        for kf_dir in sorted(dataset_dir.glob("keyframe_*"), key=lambda p: int(p.name.split("_")[1])):
            left = cv2.imread(str(kf_dir / "Left_Image.png"), cv2.IMREAD_COLOR)
            right = cv2.imread(str(kf_dir / "Right_Image.png"), cv2.IMREAD_COLOR)
            xyz = tifffile.imread(kf_dir / "left_depth_map.tiff").astype(np.float32)
            r = rectify_both(left, right, xyz, kf_dir / "endoscope_calibration.yaml")

            left_rgb = cv2.cvtColor(r["left_r"], cv2.COLOR_BGR2RGB)
            right_rgb = cv2.cvtColor(r["right_r"], cv2.COLOR_BGR2RGB)
            pred, _ms, _scale = infer(model, left_rgb, right_rgb, device, WIDTH)

            key = {"dataset_id": dataset_dir.name, "keyframe_id": kf_dir.name}
            old_rows.append({**key, **score(pred, r["old"]["disp"], r["old"]["depth"], r["old"]["valid"], r["fx"], r["baseline_mm"])})
            new_rows.append({**key, **score(pred, r["new"]["disp"], r["new"]["depth"], r["new"]["valid"], r["fx"], r["baseline_mm"])})
            print(f"{dataset_dir.name}/{kf_dir.name}: OLD EPE={old_rows[-1]['epe_px']:.3f} valid={old_rows[-1]['valid_ratio']:.3f} | "
                  f"NEW EPE={new_rows[-1]['epe_px']:.3f} valid={new_rows[-1]['valid_ratio']:.3f}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    cols = list(old_rows[0].keys())
    for name, rows in [("old_buggy_remap", old_rows), ("new_fixed_scatter", new_rows)]:
        with (OUT / f"{name}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    def agg(rows):
        return {k: float(np.mean([r[k] for r in rows])) for k in cols if k not in ("dataset_id", "keyframe_id")}

    ds45 = {"old": agg(old_rows), "new": agg(new_rows)}
    excl = {"dataset_4", "dataset_5"}
    ds35 = {
        "old": agg([r for r in old_rows if r["dataset_id"] not in excl]),
        "new": agg([r for r in new_rows if r["dataset_id"] not in excl]),
    }
    summary = {"n45": ds45, "n35_excl_ds4_ds5": ds35, "elapsed_s": time.time() - t0}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
