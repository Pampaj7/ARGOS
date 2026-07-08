#!/usr/bin/env python3
"""Build a pre-rectified, disparity-ready cache from curated/geometric_gt/strong_keyframes/.

Consolidates the rectification math that used to be copy-pasted (and buggy) across three
separate eval/prep scripts into one canonical generator: rotate the raw structured-light
3D points by R1 and re-project through P1/P2 with a z-buffer scatter (official scared_toolkit
convention), instead of cv2.remap-ing the depth channel like an image.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.argos_paths import DATASET_DIR
from scripts.scared.convert_scared_keyframes import scatter_min_depth

import cv2
import numpy as np
import tifffile

SRC = DATASET_DIR / "SCARED/curated/geometric_gt/strong_keyframes"
OUT = DATASET_DIR / "SCARED/curated/geometric_gt/strong_keyframes_rectified"


def load_calib(path: Path):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Cannot open calibration: {path}")
    out = {name: fs.getNode(name).mat() for name in ["M1", "D1", "M2", "D2", "R", "T"]}
    fs.release()
    return out


def build_one(kf_dir: Path, out_dir: Path) -> dict:
    left = cv2.imread(str(kf_dir / "Left_Image.png"), cv2.IMREAD_COLOR)
    right = cv2.imread(str(kf_dir / "Right_Image.png"), cv2.IMREAD_COLOR)
    xyz = tifffile.imread(kf_dir / "left_depth_map.tiff").astype(np.float32)
    calib = load_calib(kf_dir / "endoscope_calibration.yaml")
    h, w = left.shape[:2]

    r1, r2, p1, p2, _q, _roi1, _roi2 = cv2.stereoRectify(
        calib["M1"], calib["D1"], calib["M2"], calib["D2"], (w, h),
        calib["R"], calib["T"].reshape(3, 1), flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )
    map1x, map1y = cv2.initUndistortRectifyMap(calib["M1"], calib["D1"], r1, p1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(calib["M2"], calib["D2"], r2, p2, (w, h), cv2.CV_32FC1)
    left_r = cv2.remap(left, map1x, map1y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    right_r = cv2.remap(right, map2x, map2y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    pts_rot = (xyz.reshape(-1, 3) @ r1.T).reshape(h, w, 3)
    depth_r, disp_r = scatter_min_depth(pts_rot, p1, p2, (h, w))
    valid_r = depth_r > 0

    fx = float(p1[0, 0])
    baseline_mm = float(abs(p2[0, 3] / p2[0, 0]))

    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "left_rectified.png"), left_r)
    cv2.imwrite(str(out_dir / "right_rectified.png"), right_r)
    np.save(out_dir / "gt_depth.npy", depth_r.astype(np.float32))
    np.save(out_dir / "gt_disparity.npy", disp_r.astype(np.float32))
    cv2.imwrite(str(out_dir / "valid_mask.png"), (valid_r.astype(np.uint8) * 255))
    calib_json = {
        "P1": p1.tolist(), "P2": p2.tolist(), "R1": r1.tolist(), "R2": r2.tolist(),
        "fx": fx, "fy": float(p1[1, 1]), "cx": float(p1[0, 2]), "cy": float(p1[1, 2]),
        "baseline_mm": baseline_mm, "width": w, "height": h,
        "gt_projection": "raw structured-light points rotated by R1, projected via P1/P2 with z-buffer scatter (scared_toolkit convention)",
    }
    (out_dir / "calibration.json").write_text(json.dumps(calib_json, indent=2))

    valid_ratio = float(valid_r.mean())
    disp_valid = disp_r[valid_r]
    return {
        "dataset_id": kf_dir.parent.name,
        "keyframe_id": kf_dir.name,
        "height": h,
        "width": w,
        "fx": fx,
        "baseline_mm": baseline_mm,
        "valid_pixel_ratio": valid_ratio,
        "disp_min": float(disp_valid.min()) if disp_valid.size else "",
        "disp_max": float(disp_valid.max()) if disp_valid.size else "",
        "disp_mean": float(disp_valid.mean()) if disp_valid.size else "",
        "left_rectified_path": str((out_dir / "left_rectified.png").relative_to(DATASET_DIR.parent)),
        "right_rectified_path": str((out_dir / "right_rectified.png").relative_to(DATASET_DIR.parent)),
        "gt_depth_path": str((out_dir / "gt_depth.npy").relative_to(DATASET_DIR.parent)),
        "gt_disparity_path": str((out_dir / "gt_disparity.npy").relative_to(DATASET_DIR.parent)),
        "valid_mask_path": str((out_dir / "valid_mask.png").relative_to(DATASET_DIR.parent)),
        "calibration_path": str((out_dir / "calibration.json").relative_to(DATASET_DIR.parent)),
        "gt_source": "official structured-light (Gray-code) keyframe scan, rectified",
        "has_strong_geometric_gt": True,
    }


def main() -> int:
    rows = []
    for dataset_dir in sorted(SRC.glob("dataset_*"), key=lambda p: int(p.name.split("_")[1])):
        for kf_dir in sorted(dataset_dir.glob("keyframe_*"), key=lambda p: int(p.name.split("_")[1])):
            out_dir = OUT / dataset_dir.name / kf_dir.name
            row = build_one(kf_dir, out_dir)
            rows.append(row)
            print(f"{dataset_dir.name}/{kf_dir.name}: valid={row['valid_pixel_ratio']:.3f}")

    manifest_dir = DATASET_DIR / "SCARED/curated/manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with (manifest_dir / "strong_keyframes_rectified_manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({"total_keyframes": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
