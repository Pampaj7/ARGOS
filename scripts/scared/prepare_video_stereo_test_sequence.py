#!/usr/bin/env python3
import json
from pathlib import Path

import cv2
import numpy as np
import tifffile


SRC = Path("/dtu/p1/leopam/ARGOS/dataset/SCARED/curated/keyframes_gt_dataset8/dataset_8")
OUT = Path("/dtu/p1/leopam/ARGOS/results/video_stereo_repos/test_sequence")


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_calib(path: Path):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Cannot open calibration {path}")
    out = {name: fs.getNode(name).mat() for name in ["M1", "D1", "M2", "D2", "R", "T"]}
    fs.release()
    return out


def rectify(left, right, xyz, calib_path: Path):
    h, w = left.shape[:2]
    calib = load_calib(calib_path)
    r1, r2, p1, p2, _q, _roi1, _roi2 = cv2.stereoRectify(
        calib["M1"],
        calib["D1"],
        calib["M2"],
        calib["D2"],
        (w, h),
        calib["R"],
        calib["T"].reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )
    map1x, map1y = cv2.initUndistortRectifyMap(calib["M1"], calib["D1"], r1, p1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(calib["M2"], calib["D2"], r2, p2, (w, h), cv2.CV_32FC1)
    left_r = cv2.remap(left, map1x, map1y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    right_r = cv2.remap(right, map2x, map2y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    z = xyz[..., 2].astype(np.float32)
    valid = (np.isfinite(xyz).all(axis=-1) & (z > 0)).astype(np.uint8)
    z_clean = np.where(valid > 0, z, 0).astype(np.float32)
    z_r = cv2.remap(z_clean, map1x, map1y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    valid_r = cv2.remap(valid, map1x, map1y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT).astype(bool)
    valid_r &= z_r > 0
    fx = float(p1[0, 0])
    baseline_mm = float(abs(p2[0, 3] / p2[0, 0]))
    disp = fx * baseline_mm / np.maximum(z_r, 1e-6)
    return left_r, right_r, disp.astype(np.float32), z_r.astype(np.float32), valid_r, fx, baseline_mm


def main():
    for name in ["left", "right", "gt_disparity", "gt_depth", "valid_mask"]:
        (OUT / name).mkdir(parents=True, exist_ok=True)

    frames = []
    for idx, keyframe in enumerate(sorted(SRC.glob("keyframe_*"))[:5]):
        left = read_rgb(keyframe / "Left_Image.png")
        right = read_rgb(keyframe / "Right_Image.png")
        xyz = tifffile.imread(keyframe / "left_depth_map.tiff").astype(np.float32)
        left_r, right_r, disp, depth, valid, fx, baseline_mm = rectify(left, right, xyz, keyframe / "endoscope_calibration.yaml")
        name = f"{idx:06d}"
        cv2.imwrite(str(OUT / "left" / f"{name}.png"), cv2.cvtColor(left_r, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(OUT / "right" / f"{name}.png"), cv2.cvtColor(right_r, cv2.COLOR_RGB2BGR))
        np.save(OUT / "gt_disparity" / f"{name}.npy", disp)
        np.save(OUT / "gt_depth" / f"{name}.npy", depth)
        cv2.imwrite(str(OUT / "valid_mask" / f"{name}.png"), (valid.astype(np.uint8) * 255))
        frames.append(
            {
                "index": idx,
                "source_keyframe": keyframe.name,
                "left": f"left/{name}.png",
                "right": f"right/{name}.png",
                "gt_disparity": f"gt_disparity/{name}.npy",
                "gt_depth": f"gt_depth/{name}.npy",
                "valid_mask": f"valid_mask/{name}.png",
                "fx": fx,
                "baseline_mm": baseline_mm,
                "height": int(left_r.shape[0]),
                "width": int(left_r.shape[1]),
            }
        )
    metadata = {
        "source": str(SRC),
        "note": "Rectified SCARED dataset_8 keyframes. These keyframes are clean ARGOS test frames, not guaranteed temporally consecutive video frames.",
        "frames": frames,
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
