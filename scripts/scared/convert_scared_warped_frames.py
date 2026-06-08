import argparse
import csv
import io
import json
import math
import random
import tarfile
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import tifffile

from convert_scared_keyframes import scatter_min_depth


DEFAULT_KEYFRAMES = {
    "dataset_1": ["keyframe_1", "keyframe_2", "keyframe_3"],
    "dataset_5": ["keyframe_1", "keyframe_2", "keyframe_3"],
    "dataset_8": ["keyframe_0", "keyframe_1", "keyframe_2"],
}


def list_keyframes_with_warped(zf, dataset_id):
    keyframes = set()
    names = set(zf.namelist())
    for name in names:
        parts = name.split("/")
        if len(parts) >= 3 and parts[0] == dataset_id and parts[1].startswith("keyframe_"):
            keyframe_id = parts[1]
            prefix = f"{dataset_id}/{keyframe_id}/data"
            if all(f"{prefix}/{rel}" in names for rel in ["rgb.mp4", "frame_data.tar.gz", "scene_points.tar.gz"]):
                keyframes.add(keyframe_id)
    return sorted(keyframes)


def parse_frame_id(name, prefix, suffix):
    base = Path(name).name
    if not (base.startswith(prefix) and base.endswith(suffix)):
        return None
    return int(base[len(prefix) : -len(suffix)])


def read_frame_data(zf, path, selected_ids):
    out = {}
    with zf.open(path) as raw:
        with tarfile.open(fileobj=raw, mode="r|gz") as tf:
            for member in tf:
                fid = parse_frame_id(member.name, "frame_data", ".json")
                if fid is None or fid not in selected_ids:
                    continue
                f = tf.extractfile(member)
                if f is not None:
                    out[fid] = json.load(f)
                if len(out) == len(selected_ids):
                    break
    return out


def iter_scene_points(zf, path, selected_ids):
    found = set()
    with zf.open(path) as raw:
        with tarfile.open(fileobj=raw, mode="r|gz") as tf:
            for member in tf:
                fid = parse_frame_id(member.name, "scene_points", ".tiff")
                if fid is None or fid not in selected_ids:
                    continue
                f = tf.extractfile(member)
                if f is not None:
                    yield fid, tifffile.imread(io.BytesIO(f.read())).astype(np.float32)
                    found.add(fid)
                if len(found) == len(selected_ids):
                    break


def read_video_frames(zf, path, selected_ids):
    frames = {}
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(zf.read(path))
        tmp.flush()
        cap = cv2.VideoCapture(tmp.name)
        max_id = max(selected_ids)
        fid = 0
        while fid <= max_id:
            ok, frame = cap.read()
            if not ok:
                break
            if fid in selected_ids:
                h = frame.shape[0] // 2
                frames[fid] = (frame[:h].copy(), frame[h : h * 2].copy())
            fid += 1
        cap.release()
    return frames


def calib_from_frame_data(frame_data, image_size):
    c = frame_data["camera-calibration"]
    m1 = np.array(c["KL"], dtype=np.float64)
    m2 = np.array(c["KR"], dtype=np.float64)
    d1 = np.array(c["DL"], dtype=np.float64).reshape(-1, 1)
    d2 = np.array(c["DR"], dtype=np.float64).reshape(-1, 1)
    r = np.array(c["R"], dtype=np.float64)
    t = np.array(c["T"], dtype=np.float64).reshape(3, 1)
    r1, r2, p1, p2, _q, _roi1, _roi2 = cv2.stereoRectify(
        m1, d1, m2, d2, image_size, r, t, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )
    map1x, map1y = cv2.initUndistortRectifyMap(m1, d1, r1, p1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(m2, d2, r2, p2, image_size, cv2.CV_32FC1)
    return r1, p1, p2, (map1x, map1y, map2x, map2y)


def save_float_gt(ref_dir, stem, depth, disp):
    valid = (depth > 0) & (disp > 0) & np.isfinite(depth) & np.isfinite(disp)
    paths = {
        "depth": ref_dir / "DepthL_float32" / f"{stem}.npy",
        "disp": ref_dir / "Disparity_float32" / f"{stem}.npy",
        "mask": ref_dir / "ValidMask" / f"{stem}.npy",
    }
    for path, array in [
        (paths["depth"], depth.astype(np.float32)),
        (paths["disp"], disp.astype(np.float32)),
        (paths["mask"], valid),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array)
    return valid, paths


def image_vis(x, mask=None, p=99):
    valid = np.isfinite(x)
    if mask is not None:
        valid &= mask
    max_val = np.percentile(x[valid], p) if valid.any() else 1.0
    y = np.zeros_like(x, dtype=np.float32)
    y[valid] = x[valid]
    y = np.clip(y, 0, max(max_val, 1e-6))
    y = (y / max(max_val, 1e-6) * 255).astype(np.uint8)
    return cv2.applyColorMap(y, cv2.COLORMAP_TURBO)


def mask_vis(mask):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask] = 255
    return out


def make_montage(samples, out_path):
    rows = []
    for sample in samples:
        panels = [
            sample["left"],
            sample["right"],
            image_vis(sample["depth"], sample["valid"]),
            image_vis(sample["disp"], sample["valid"]),
            mask_vis(sample["valid"]),
        ]
        panels = [cv2.resize(p, (256, 205), interpolation=cv2.INTER_AREA) for p in panels]
        row = np.concatenate(panels, axis=1)
        canvas = np.full((229, row.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(canvas, sample["label"], (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
        canvas[24:] = row
        rows.append(canvas)
    cv2.imwrite(str(out_path), np.concatenate(rows, axis=0))


def make_hist(values, out_path, title, bins=30):
    values = np.asarray(values, dtype=np.float32)
    hist, edges = np.histogram(values, bins=bins, range=(0, max(1.0, float(values.max()))))
    h, w = 420, 720
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.putText(img, title, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    left, top, right, bottom = 60, 60, w - 20, h - 45
    cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 0), 1)
    max_count = max(int(hist.max()), 1)
    bw = (right - left) / bins
    for i, count in enumerate(hist):
        x0 = int(left + i * bw)
        x1 = int(left + (i + 1) * bw - 1)
        y1 = bottom
        y0 = int(bottom - (bottom - top) * (count / max_count))
        cv2.rectangle(img, (x0, y0), (x1, y1), (80, 140, 220), -1)
    cv2.putText(img, f"min={values.min():.3f} mean={values.mean():.3f} max={values.max():.3f}", (20, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), img)


def clean_keyframe_stats(root="stereo/Fast-FoundationStereo/data/surgical_stereo/scared_keyframes"):
    root = Path(root)
    rows = []
    for depth_path in sorted(root.glob("dataset_*/Reference_SCARED/DepthL_float32/*.npy")):
        ref_dir = depth_path.parents[1]
        stem = depth_path.stem
        disp_path = ref_dir / "Disparity_float32" / f"{stem}.npy"
        mask_path = ref_dir / "ValidMask" / f"{stem}.npy"
        if not disp_path.exists():
            continue
        depth = np.load(depth_path)
        disp = np.load(disp_path)
        mask = np.load(mask_path).astype(bool) if mask_path.exists() else (depth > 0) & (disp > 0)
        if not mask.any():
            continue
        rows.append(
            {
                "valid_pixel_ratio": float(mask.mean()),
                "depth_median_mm": float(np.median(depth[mask])),
                "depth_p95_mm": float(np.percentile(depth[mask], 95)),
                "disp_median_px": float(np.median(disp[mask])),
                "disp_p95_px": float(np.percentile(disp[mask], 95)),
            }
        )
    out = {}
    for col in ["valid_pixel_ratio", "depth_median_mm", "depth_p95_mm", "disp_median_px", "disp_p95_px"]:
        vals = [r[col] for r in rows]
        if vals:
            out[col] = {"mean": float(np.mean(vals)), "median": float(np.median(vals))}
    out["frames"] = len(rows)
    return out


def choose_audit_samples(limit):
    keyframes = [(ds, kf) for ds, kfs in DEFAULT_KEYFRAMES.items() for kf in kfs]
    base = limit // len(keyframes)
    extra = limit % len(keyframes)
    plan = {}
    for i, item in enumerate(keyframes):
        n = base + (1 if i < extra else 0)
        plan[item] = list(range(n))
    return plan


def count_frame_data(zf, path):
    count = 0
    with zf.open(path) as raw:
        with tarfile.open(fileobj=raw, mode="r|gz") as tf:
            for member in tf:
                if parse_frame_id(member.name, "frame_data", ".json") is not None:
                    count += 1
    return count


def choose_stride_samples(scared_dir, datasets, stride, cap):
    plan = {}
    total = 0
    for dataset_id in datasets:
        zip_path = Path(scared_dir) / f"{dataset_id}.zip"
        with zipfile.ZipFile(zip_path) as zf:
            for keyframe_id in list_keyframes_with_warped(zf, dataset_id):
                frame_data_path = f"{dataset_id}/{keyframe_id}/data/frame_data.tar.gz"
                n = count_frame_data(zf, frame_data_path)
                ids = list(range(0, n, stride))
                if cap and total + len(ids) > cap:
                    ids = ids[: max(0, cap - total)]
                if ids:
                    plan[(dataset_id, keyframe_id)] = ids
                    total += len(ids)
                if cap and total >= cap:
                    return plan
    return plan


def suspicious_notes(valid_ratio, depth_median, disp_p99, unreadable=False):
    notes = []
    if unreadable:
        notes.append("unreadable frame/GT")
    if valid_ratio < 0.20:
        notes.append("valid coverage < 20%")
    if depth_median < 10 or depth_median > 200:
        notes.append("depth median outside expected range")
    if disp_p99 > 400:
        notes.append("disparity p99 extremely high")
    return "; ".join(notes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scared_dir", default="stereo/Fast-FoundationStereo/data/surgical_stereo/scared")
    parser.add_argument("--out_root", default="stereo/Fast-FoundationStereo/data/surgical_stereo/scared_warped")
    parser.add_argument("--metadata_csv", default="results/scared_warped_metadata.csv")
    parser.add_argument("--audit_dir", default="results/scared_warped_audit")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--mode", choices=["audit", "stride"], default="audit")
    parser.add_argument("--datasets", nargs="*", default=[f"dataset_{i}" for i in range(1, 7)])
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--cap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    scared_dir = Path(args.scared_dir)
    out_root = Path(args.out_root)
    audit_dir = Path(args.audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(args.metadata_csv)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "audit":
        plan = choose_audit_samples(args.limit)
    else:
        plan = choose_stride_samples(args.scared_dir, args.datasets, args.stride, args.cap)
    rows = []
    montage_candidates = []

    for dataset_id in sorted({ds for ds, _kf in plan}):
        zip_path = scared_dir / f"{dataset_id}.zip"
        with zipfile.ZipFile(zip_path) as zf:
            for keyframe_id in [kf for ds, kf in plan if ds == dataset_id]:
                selected_ids = set(plan[(dataset_id, keyframe_id)])
                prefix = f"{dataset_id}/{keyframe_id}/data"
                rgb_path = f"{prefix}/rgb.mp4"
                frame_data_path = f"{prefix}/frame_data.tar.gz"
                scene_points_path = f"{prefix}/scene_points.tar.gz"
                required = [rgb_path, frame_data_path, scene_points_path]
                if any(p not in zf.namelist() for p in required):
                    for fid in sorted(selected_ids):
                        rows.append(
                            {
                                "dataset_id": dataset_id,
                                "keyframe_id": keyframe_id,
                                "frame_id": fid,
                                "left_path": "",
                                "right_path": "",
                                "depth_float32_path": "",
                                "disparity_float32_path": "",
                                "valid_mask_path": "",
                                "calibration_path": "",
                                "valid_pixel_ratio": 0.0,
                                "depth_median_mm": math.nan,
                                "depth_p95_mm": math.nan,
                                "disp_median_px": math.nan,
                                "disp_p95_px": math.nan,
                                "notes": "unreadable frame/GT",
                            }
                        )
                    continue

                frame_data = read_frame_data(zf, frame_data_path, selected_ids)
                video_frames = read_video_frames(zf, rgb_path, selected_ids)
                for fid, points in iter_scene_points(zf, scene_points_path, selected_ids):
                    if fid not in frame_data or fid not in video_frames:
                        continue
                    left, right = video_frames[fid]
                    h, w = left.shape[:2]
                    r1, p1, p2, maps = calib_from_frame_data(frame_data[fid], (w, h))
                    map1x, map1y, map2x, map2y = maps
                    left_rect = cv2.remap(left, map1x, map1y, cv2.INTER_LINEAR)
                    right_rect = cv2.remap(right, map2x, map2y, cv2.INTER_LINEAR)

                    invalid_xyz = np.isclose(points, 0.0).all(axis=2)
                    points = points.copy()
                    points[invalid_xyz] = 0.0
                    points_rect = points @ r1.T
                    points_rect[invalid_xyz] = 0.0
                    depth, disp = scatter_min_depth(points_rect, p1, p2, (h, w))

                    stem = f"frame_{fid:06d}"
                    sample_dir = out_root / dataset_id / keyframe_id
                    left_path = sample_dir / "Left_rectified" / f"{stem}.png"
                    right_path = sample_dir / "Right_rectified" / f"{stem}.png"
                    left_path.parent.mkdir(parents=True, exist_ok=True)
                    right_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(left_path), left_rect)
                    cv2.imwrite(str(right_path), right_rect)
                    ref_dir = sample_dir / "Reference_SCARED_Warped"
                    valid, gt_paths = save_float_gt(ref_dir, stem, depth, disp)

                    calib_path = sample_dir / "Rectified_calibration" / f"{stem}.json"
                    calib_path.parent.mkdir(parents=True, exist_ok=True)
                    calib_json = {
                        "P1": {"rows": 3, "cols": 4, "data": p1.astype(float).reshape(-1).tolist()},
                        "P2": {"rows": 3, "cols": 4, "data": p2.astype(float).reshape(-1).tolist()},
                    }
                    calib_path.write_text(json.dumps(calib_json, indent=2))

                    if valid.any():
                        depth_vals = depth[valid]
                        disp_vals = disp[valid]
                        valid_ratio = float(valid.mean())
                        depth_median = float(np.median(depth_vals))
                        depth_p95 = float(np.percentile(depth_vals, 95))
                        disp_median = float(np.median(disp_vals))
                        disp_p95 = float(np.percentile(disp_vals, 95))
                        disp_p99 = float(np.percentile(disp_vals, 99))
                    else:
                        valid_ratio = 0.0
                        depth_median = depth_p95 = disp_median = disp_p95 = disp_p99 = math.nan
                    notes = suspicious_notes(valid_ratio, depth_median, disp_p99)

                    row = {
                        "dataset_id": dataset_id,
                        "keyframe_id": keyframe_id,
                        "frame_id": fid,
                        "left_path": str(left_path),
                        "right_path": str(right_path),
                        "depth_float32_path": str(gt_paths["depth"]),
                        "disparity_float32_path": str(gt_paths["disp"]),
                        "valid_mask_path": str(gt_paths["mask"]),
                        "calibration_path": str(calib_path),
                        "valid_pixel_ratio": valid_ratio,
                        "depth_median_mm": depth_median,
                        "depth_p95_mm": depth_p95,
                        "disp_median_px": disp_median,
                        "disp_p95_px": disp_p95,
                        "notes": notes,
                    }
                    rows.append(row)
                    montage_candidates.append(
                        {
                            "label": f"{dataset_id}/{keyframe_id}/{stem}",
                            "left": left_rect,
                            "right": right_rect,
                            "depth": depth,
                            "disp": disp,
                            "valid": valid,
                        }
                    )
                    print(f"{dataset_id}/{keyframe_id}/{stem} valid={valid_ratio:.3f} {notes}", flush=True)

    keys = [
        "dataset_id",
        "keyframe_id",
        "frame_id",
        "left_path",
        "right_path",
        "depth_float32_path",
        "disparity_float32_path",
        "valid_mask_path",
        "calibration_path",
        "valid_pixel_ratio",
        "depth_median_mm",
        "depth_p95_mm",
        "disp_median_px",
        "disp_p95_px",
        "notes",
    ]
    rows = sorted(rows, key=lambda r: (r["dataset_id"], r["keyframe_id"], int(r["frame_id"])))
    with open(metadata_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    good_rows = [r for r in rows if r["left_path"]]
    if montage_candidates:
        random.shuffle(montage_candidates)
        make_montage(montage_candidates[:50], audit_dir / "montage_50_random_samples.png")
        make_hist([float(r["valid_pixel_ratio"]) for r in good_rows], audit_dir / "valid_pixel_ratio_histogram.png", "Warped valid pixel ratio")

    clean_stats = clean_keyframe_stats()
    suspicious = [r for r in rows if r["notes"]]
    report = {
        "converted_frames": len(good_rows),
        "requested_limit": args.limit if args.mode == "audit" else args.cap,
        "mode": args.mode,
        "stride": args.stride if args.mode == "stride" else None,
        "datasets": sorted({r["dataset_id"] for r in good_rows}),
        "keyframes": sorted({f"{r['dataset_id']}/{r['keyframe_id']}" for r in good_rows}),
        "valid_pixel_ratio_mean": float(np.mean([float(r["valid_pixel_ratio"]) for r in good_rows])) if good_rows else math.nan,
        "valid_pixel_ratio_median": float(np.median([float(r["valid_pixel_ratio"]) for r in good_rows])) if good_rows else math.nan,
        "depth_median_mm_mean": float(np.mean([float(r["depth_median_mm"]) for r in good_rows])) if good_rows else math.nan,
        "disp_median_px_mean": float(np.mean([float(r["disp_median_px"]) for r in good_rows])) if good_rows else math.nan,
        "suspicious_frames": len(suspicious),
        "clean_keyframe_stats": clean_stats,
    }
    (audit_dir / "audit_summary.json").write_text(json.dumps(report, indent=2))
    md = [
        "# SCARED Warped Audit",
        "",
        f"Converted frames: {report['converted_frames']} / requested {args.limit}",
        f"Datasets: {', '.join(report['datasets'])}",
        f"Keyframes: {', '.join(report['keyframes'])}",
        f"Mean valid pixel ratio: {report['valid_pixel_ratio_mean']:.3f}",
        f"Median valid pixel ratio: {report['valid_pixel_ratio_median']:.3f}",
        f"Mean depth median: {report['depth_median_mm_mean']:.3f} mm",
        f"Mean disparity median: {report['disp_median_px_mean']:.3f} px",
        f"Suspicious frames: {report['suspicious_frames']}",
        "",
        "Outputs:",
        f"- Metadata: `{metadata_path}`",
        f"- Montage: `{audit_dir / 'montage_50_random_samples.png'}`",
        f"- Valid-ratio histogram: `{audit_dir / 'valid_pixel_ratio_histogram.png'}`",
        "",
        "Clean keyframe comparison:",
        json.dumps(clean_stats, indent=2),
    ]
    if suspicious:
        md.extend(["", "Suspicious frame examples:"])
        for r in suspicious[:20]:
            md.append(f"- {r['dataset_id']}/{r['keyframe_id']}/frame_{int(r['frame_id']):06d}: {r['notes']}")
    (audit_dir / "audit_report.md").write_text("\n".join(md) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
