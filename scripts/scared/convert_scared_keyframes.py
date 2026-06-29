import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import tifffile


def read_cv_image(zf, name, flags=cv2.IMREAD_COLOR):
    data = np.frombuffer(zf.read(name), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise ValueError(f"Cannot decode image from {name}")
    return image


def read_tiff(zf, name):
    with zf.open(name) as f:
        return tifffile.imread(f)


def read_calibration(zf, name):
    with tempfile.NamedTemporaryFile(suffix=".yaml") as tmp:
        tmp.write(zf.read(name))
        tmp.flush()
        fs = cv2.FileStorage(tmp.name, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise ValueError(f"Cannot open calibration from {name}")
        calib = {
            key: fs.getNode(key).mat().astype(np.float64)
            for key in ["M1", "D1", "M2", "D2", "R", "T"]
        }
        fs.release()
    calib["T"] = calib["T"].reshape(3, 1)
    return calib


def list_keyframes(zf, dataset_name):
    keyframes = set()
    for name in zf.namelist():
        parts = name.split("/")
        if len(parts) >= 3 and parts[0] == dataset_name and parts[1].startswith("keyframe_"):
            keyframes.add(parts[1])
    return sorted(keyframes)


def scatter_min_depth(points_rect, p1, p2, image_shape):
    h, w = image_shape
    pts = points_rect.reshape(-1, 3)
    valid = np.isfinite(pts).all(axis=1) & (pts[:, 2] > 1e-6)
    pts = pts[valid]
    if pts.size == 0:
        return np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)

    fx = float(p1[0, 0])
    fy = float(p1[1, 1])
    cx = float(p1[0, 2])
    cy = float(p1[1, 2])
    baseline = abs(float(p2[0, 3] / p2[0, 0]))

    z = pts[:, 2]
    u = np.rint((fx * pts[:, 0] / z) + cx).astype(np.int32)
    v = np.rint((fy * pts[:, 1] / z) + cy).astype(np.int32)
    in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u = u[in_bounds]
    v = v[in_bounds]
    z = z[in_bounds]
    if z.size == 0:
        return np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)

    flat_idx = v * w + u
    order = np.lexsort((z, flat_idx))
    flat_sorted = flat_idx[order]
    z_sorted = z[order]
    first = np.r_[True, flat_sorted[1:] != flat_sorted[:-1]]
    flat_keep = flat_sorted[first]
    z_keep = z_sorted[first]

    depth = np.zeros(h * w, dtype=np.float32)
    depth[flat_keep] = z_keep.astype(np.float32)
    depth = depth.reshape(h, w)

    disp = np.zeros_like(depth)
    mask = depth > 0
    disp[mask] = fx * baseline / depth[mask]
    return depth, disp


def write_scaled_png(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    scaled = np.clip(values * 256.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    cv2.imwrite(str(path), scaled)


def write_float_gt(ref_dir, stem, depth, disp):
    valid_mask = (depth > 0) & (disp > 0) & np.isfinite(depth) & np.isfinite(disp)
    outputs = [
        (ref_dir / "DepthL_float32" / f"{stem}.npy", depth.astype(np.float32)),
        (ref_dir / "Disparity_float32" / f"{stem}.npy", disp.astype(np.float32)),
        (ref_dir / "ValidMask" / f"{stem}.npy", valid_mask),
    ]
    for path, array in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array)


def convert_keyframe(zf, dataset_name, keyframe, out_root):
    prefix = f"{dataset_name}/{keyframe}"
    required = [
        f"{prefix}/Left_Image.png",
        f"{prefix}/Right_Image.png",
        f"{prefix}/left_depth_map.tiff",
        f"{prefix}/endoscope_calibration.yaml",
    ]
    if any(name not in zf.namelist() for name in required):
        return None

    left = read_cv_image(zf, f"{prefix}/Left_Image.png")
    right = read_cv_image(zf, f"{prefix}/Right_Image.png")
    points = read_tiff(zf, f"{prefix}/left_depth_map.tiff").astype(np.float32)
    calib = read_calibration(zf, f"{prefix}/endoscope_calibration.yaml")

    h, w = left.shape[:2]
    r1, r2, p1, p2, _q, _roi1, _roi2 = cv2.stereoRectify(
        calib["M1"],
        calib["D1"],
        calib["M2"],
        calib["D2"],
        (w, h),
        calib["R"],
        calib["T"],
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )
    map1x, map1y = cv2.initUndistortRectifyMap(calib["M1"], calib["D1"], r1, p1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(calib["M2"], calib["D2"], r2, p2, (w, h), cv2.CV_32FC1)
    left_rect = cv2.remap(left, map1x, map1y, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right, map2x, map2y, cv2.INTER_LINEAR)

    points_rect = points @ r1.T
    depth, disp = scatter_min_depth(points_rect, p1, p2, (h, w))

    exp = dataset_name
    stem = keyframe
    exp_dir = out_root / exp
    (exp_dir / "Left_rectified").mkdir(parents=True, exist_ok=True)
    (exp_dir / "Right_rectified").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(exp_dir / "Left_rectified" / f"{stem}.png"), left_rect)
    cv2.imwrite(str(exp_dir / "Right_rectified" / f"{stem}.png"), right_rect)
    ref_dir = exp_dir / "Reference_SCARED"
    write_scaled_png(ref_dir / "DepthL" / f"{stem}.png", depth)
    write_scaled_png(ref_dir / "Disparity" / f"{stem}.png", disp)
    write_float_gt(ref_dir, stem, depth, disp)
    (exp_dir / "Rectified_calibration").mkdir(parents=True, exist_ok=True)
    calib_json = {
        "P1": {"rows": 3, "cols": 4, "data": p1.astype(float).reshape(-1).tolist()},
        "P2": {"rows": 3, "cols": 4, "data": p2.astype(float).reshape(-1).tolist()},
    }
    (exp_dir / "Rectified_calibration" / f"{stem}.json").write_text(json.dumps(calib_json, indent=2))

    valid = int((depth > 0).sum())
    return {
        "dataset": dataset_name,
        "keyframe": keyframe,
        "valid_px": valid,
        "valid_pct": valid / float(h * w) * 100.0,
        "baseline": abs(float(p2[0, 3] / p2[0, 0])),
        "fx": float(p1[0, 0]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scared_dir", default="stereo/Fast-FoundationStereo/data/surgical_stereo/scared")
    parser.add_argument("--out_root", default="stereo/Fast-FoundationStereo/data/surgical_stereo/scared_keyframes")
    parser.add_argument("--datasets", nargs="*", default=[f"dataset_{i}" for i in range(1, 10)])
    args = parser.parse_args()

    scared_dir = Path(args.scared_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset_name in args.datasets:
        zip_path = scared_dir / f"{dataset_name}.zip"
        if not zip_path.exists():
            print(f"missing {zip_path}")
            continue
        print(f"converting {zip_path}")
        with zipfile.ZipFile(zip_path) as zf:
            for keyframe in list_keyframes(zf, dataset_name):
                row = convert_keyframe(zf, dataset_name, keyframe, out_root)
                if row is None:
                    print(f"  skipped {dataset_name}/{keyframe}")
                    continue
                rows.append(row)
                print(
                    f"  {dataset_name}/{keyframe}: valid={row['valid_px']} "
                    f"({row['valid_pct']:.1f}%)"
                )

    (out_root / "conversion_summary.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps({"converted": len(rows), "out_root": str(out_root)}, indent=2))


if __name__ == "__main__":
    main()
