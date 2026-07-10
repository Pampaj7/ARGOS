#!/usr/bin/env python3
"""Build corrected_temporal_gt for one SCARED-C sequence: reproject the keyframe's own
structured-light depth through each COLMAP-corrected, scale-recovered per-frame pose
(data/frame_data.tar.gz), for the frames COLMAP successfully co-registered (frame_log.json
included_frames only — never the excluded/uncoregistered ones).

Same geometry as vanilla SCARED's convert_scared_keyframe_to_temporal_gt_rectified.py:
rotate points by R1, project via P1/P2 with a z-buffer scatter (scatter_min_depth) — just
fed the SCARED-C corrected pose instead of a raw kinematics tf chain, and the keyframe's own
depth map instead of a per-frame scene_points.tar.gz scan.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.argos_paths import DATASET_DIR
from scripts.scared.convert_scared_keyframes import scatter_min_depth

import cv2
import numpy as np
import tifffile

RAW = DATASET_DIR / "SCARED-C/raw"
OUT = DATASET_DIR / "SCARED-C/curated/geometric_gt/corrected_temporal_gt"


def load_calib(path: Path):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    out = {name: fs.getNode(name).mat() for name in ["M1", "D1", "M2", "D2", "R", "T"]}
    fs.release()
    return out


def load_poses(frame_data_tar: Path, frame_ids: set[int]) -> dict[int, np.ndarray]:
    poses = {}
    with tarfile.open(frame_data_tar, "r:gz") as tf:
        for member in tf.getmembers():
            fid = int("".join(c for c in Path(member.name).stem if c.isdigit()))
            if fid not in frame_ids:
                continue
            f = tf.extractfile(member)
            data = json.loads(f.read())
            poses[fid] = np.array(data["camera-pose"], dtype=np.float64).reshape(4, 4)
    return poses


def build_sequence(dataset_id: str, keyframe_id: str, max_frames: int = 0, sample_n: int = 0) -> dict:
    kf_dir = RAW / dataset_id / keyframe_id
    frame_log = json.loads((kf_dir / "frame_log.json").read_text())
    included = frame_log["included_frames"]
    if sample_n and sample_n < len(included):
        idx = np.linspace(0, len(included) - 1, sample_n).astype(int)
        included = [included[i] for i in sorted(set(idx.tolist()))]
    elif max_frames:
        included = included[:max_frames]
    frame_ids = set(included)

    xyz_kf = tifffile.imread(kf_dir / "left_depth_map.tiff").astype(np.float64)
    left_kf = cv2.imread(str(kf_dir / "Left_Image.png"))
    h, w = left_kf.shape[:2]
    calib = load_calib(kf_dir / "endoscope_calibration.yaml")
    r1, r2, p1, p2, _q, _roi1, _roi2 = cv2.stereoRectify(
        calib["M1"], calib["D1"], calib["M2"], calib["D2"], (w, h),
        calib["R"], calib["T"].reshape(3, 1), flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )
    map1x, map1y = cv2.initUndistortRectifyMap(calib["M1"], calib["D1"], r1, p1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(calib["M2"], calib["D2"], r2, p2, (w, h), cv2.CV_32FC1)
    fx = float(p1[0, 0])
    baseline_mm = float(abs(p2[0, 3] / p2[0, 0]))

    poses = load_poses(kf_dir / "data" / "frame_data.tar.gz", frame_ids)

    out_dir = OUT / f"{dataset_id}_{keyframe_id}"
    for sub in ["left", "right", "gt"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(kf_dir / "data" / "rgb.mp4"))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {kf_dir / 'data' / 'rgb.mp4'}")

    rows = []
    frame_idx = 0  # 0-based video frame counter; frame_log ids are 1-based
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx not in poses:
            continue
        pose = poses[frame_idx]
        fh, fw = frame_bgr.shape[:2]
        left_raw, right_raw = frame_bgr[: fh // 2], frame_bgr[fh // 2 :]

        left_r = cv2.remap(left_raw, map1x, map1y, cv2.INTER_LINEAR)
        right_r = cv2.remap(right_raw, map2x, map2y, cv2.INTER_LINEAR)

        pts_h = np.concatenate([xyz_kf.reshape(-1, 3), np.ones((xyz_kf.size // 3, 1))], axis=1)
        pts_frame = (pose @ pts_h.T).T
        pts_frame = (pts_frame[:, :3] / pts_frame[:, 3:4]).reshape(h, w, 3)
        pts_rot = (pts_frame.reshape(-1, 3) @ r1.T).reshape(h, w, 3)
        depth, disp = scatter_min_depth(pts_rot, p1, p2, (h, w))
        valid = depth > 0

        stem = f"{frame_idx:06d}"
        for sub in ["left", "right", "gt"]:
            (out_dir / sub).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "left" / f"{stem}.png"), left_r)
        cv2.imwrite(str(out_dir / "right" / f"{stem}.png"), right_r)
        np.save(out_dir / "gt" / f"{stem}_depth.npy", depth.astype(np.float32))
        np.save(out_dir / "gt" / f"{stem}_disp.npy", disp.astype(np.float32))
        cv2.imwrite(str(out_dir / "gt" / f"{stem}_valid.png"), (valid.astype(np.uint8) * 255))

        rows.append({
            "sequence_id": f"{dataset_id}_{keyframe_id}", "frame_id": stem,
            "valid_pixel_ratio": float(valid.mean()), "fx": fx, "baseline_mm": baseline_mm,
        })
        print(f"{dataset_id}/{keyframe_id}/{stem}: valid={rows[-1]['valid_pixel_ratio']:.3f}", flush=True)

    cap.release()
    manifest_path = out_dir / "manifest.csv"
    if rows:
        with manifest_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    calib_json = {"P1": p1.tolist(), "P2": p2.tolist(), "fx": fx, "baseline_mm": baseline_mm,
                  "gt_source": "corrected COLMAP+scale-recovery pose, keyframe depth reprojected via R1-rotation + z-buffer scatter"}
    (out_dir / "calibration.json").write_text(json.dumps(calib_json, indent=2))
    return {"sequence_id": f"{dataset_id}_{keyframe_id}", "n_frames": len(rows), "n_expected": len(included)}


def _build_one(args_tuple):
    dataset_id, keyframe_id, max_frames, sample_n = args_tuple
    return build_sequence(dataset_id, keyframe_id, max_frames, sample_n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sequences", nargs="+", required=True, help="dataset_N/keyframe_M entries")
    ap.add_argument("--max-frames", type=int, default=0, help="cap frames per sequence to the first N (0 = all included)")
    ap.add_argument("--sample-n", type=int, default=0, help="evenly-spaced N frames across the whole sequence (overrides --max-frames)")
    ap.add_argument("--workers", type=int, default=1, help="parallelize across sequences (one process per sequence)")
    args = ap.parse_args()

    jobs = [(*seq.split("/"), args.max_frames, args.sample_n) for seq in args.sequences]
    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(min(args.workers, len(jobs))) as pool:
            summaries = pool.map(_build_one, jobs)
    else:
        summaries = [_build_one(j) for j in jobs]
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
