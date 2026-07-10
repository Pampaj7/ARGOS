#!/usr/bin/env python3
"""Convert StereoMIS raw sequences into ARGOS curated format.

StereoMIS ships vertically-stacked stereo video (top=left, bottom=right), a per-side
distorted pinhole calibration (StereoCalibration.ini), dense per-frame kinematics pose
(groundtruth.txt: `frame tx ty tz qx qy qz qw`), sparse-to-dense instrument masks, and a
split CSV giving the usable frame range(s) (StereoMIS_0_0_1's original recording includes
setup/teardown footage outside these ranges).

There is no dense depth/disparity ground truth (confirmed against the StereoMIS paper,
Hayoz et al. 2023: poses come from da Vinci forward kinematics, not structured light).
This converter only rectifies frames and carries forward pose + mask, following the same
stereoRectify convention used for SCARED (see scripts/scared/build_strong_keyframes_rectified.py).
"""
from __future__ import annotations

import argparse
import configparser
import csv
import json
import multiprocessing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.argos_paths import DATASET_DIR

import cv2
import numpy as np

RAW = DATASET_DIR / "StereoMIS/raw/extracted/StereoMIS_0_0_1/StereoMIS_0_0_1"
OUT = DATASET_DIR / "StereoMIS/curated/geometric_gt/temporal_sequences"
PILOT_SEQUENCES = ["P1", "P2_8", "P3"]


def parse_calib(ini_path: Path):
    cfg = configparser.ConfigParser()
    cfg.read(ini_path)

    def side(name):
        s = cfg[name]
        K = np.array([[float(s["fc_x"]), 0, float(s["cc_x"])],
                      [0, float(s["fc_y"]), float(s["cc_y"])],
                      [0, 0, 1]], dtype=np.float64)
        D = np.array([float(s[f"kc_{i}"]) for i in range(5)], dtype=np.float64)
        w, h = int(float(s["res_x"])), int(float(s["res_y"]))
        return K, D, (w, h)

    K1, D1, size = side("StereoLeft")
    K2, D2, _ = side("StereoRight")
    right = cfg["StereoRight"]
    R = np.array([float(right[f"R_{i}"]) for i in range(9)], dtype=np.float64).reshape(3, 3)
    T = np.array([float(right[f"T_{i}"]) for i in range(3)], dtype=np.float64)
    return K1, D1, K2, D2, R, T, size


def build_rectify_maps(K1, D1, K2, D2, R, T, size):
    R1, R2, P1, P2, _Q, _roi1, _roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, size, cv2.CV_32FC1)
    return map1x, map1y, map2x, map2y, P1, P2


def load_split_ranges(split_csv: Path) -> list[tuple[int, int]]:
    rows = list(csv.DictReader(split_csv.open()))
    return [(int(r["start"]), int(r["end"])) for r in rows]


def load_poses(groundtruth_txt: Path) -> dict[int, tuple]:
    poses = {}
    for line in groundtruth_txt.open():
        parts = line.split()
        if len(parts) != 8:
            continue
        frame = int(parts[0])
        poses[frame] = tuple(float(x) for x in parts[1:])
    return poses


def convert_sequence(seq: str, max_frames: int = 0) -> dict:
    seq_dir = RAW / seq
    videos = list(seq_dir.glob("*.mp4"))
    assert len(videos) == 1, f"expected 1 video in {seq_dir}, found {videos}"
    video_path = videos[0]

    split_files = list(seq_dir.glob("*_split.csv"))
    assert len(split_files) == 1
    split_name = split_files[0].stem.replace("_split", "")  # "train" or "test"
    ranges = load_split_ranges(split_files[0])
    wanted = set()
    for start, end in ranges:
        wanted.update(range(start, end + 1))
    if max_frames:
        wanted = set(sorted(wanted)[:max_frames])

    poses = load_poses(seq_dir / "groundtruth.txt")
    K1, D1, K2, D2, R, T, size = parse_calib(seq_dir / "StereoCalibration.ini")
    map1x, map1y, map2x, map2y, P1, P2 = build_rectify_maps(K1, D1, K2, D2, R, T, size)
    w, h = size

    out_dir = OUT / seq
    (out_dir / "left").mkdir(parents=True, exist_ok=True)
    (out_dir / "right").mkdir(parents=True, exist_ok=True)
    (out_dir / "mask").mkdir(parents=True, exist_ok=True)

    fx = float(P1[0, 0])
    baseline_mm = float(abs(P2[0, 3] / P2[0, 0]))
    calib_json = {
        "P1": P1.tolist(), "P2": P2.tolist(),
        "fx": fx, "fy": float(P1[1, 1]), "cx_left": float(P1[0, 2]), "cy_left": float(P1[1, 2]),
        "cx_right": float(P2[0, 2]), "cy_right": float(P2[1, 2]),
        "baseline_mm": baseline_mm, "width": w, "height": h,
    }
    (out_dir / "calibration.json").write_text(json.dumps(calib_json, indent=2))

    cap = cv2.VideoCapture(str(video_path))
    rows = []
    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx not in wanted:
            continue
        top, bottom = frame[:h], frame[h:2 * h]
        left_r = cv2.remap(top, map1x, map1y, interpolation=cv2.INTER_LINEAR)
        right_r = cv2.remap(bottom, map2x, map2y, interpolation=cv2.INTER_LINEAR)
        stem = f"{frame_idx:06d}"
        jpg_opts = [cv2.IMWRITE_JPEG_QUALITY, 95]
        cv2.imwrite(str(out_dir / "left" / f"{stem}.jpg"), left_r, jpg_opts)
        cv2.imwrite(str(out_dir / "right" / f"{stem}.jpg"), right_r, jpg_opts)

        mask_src = seq_dir / "masks" / f"{stem}l.png"
        has_mask = mask_src.exists()
        if has_mask:
            mask = cv2.imread(str(mask_src), cv2.IMREAD_GRAYSCALE)
            if mask.shape[:2] != (h, w):
                # masks ship at inconsistent resolution across sequences (some half-res) —
                # upsample/downsample to native frame size before remap, or map1x/map1y
                # (built for (w,h)) index outside the mask array and produce a bogus artifact.
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_r = cv2.remap(mask, map1x, map1y, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(out_dir / "mask" / f"{stem}.png"), mask_r)

        pose = poses.get(frame_idx)
        rows.append({
            "sequence": seq, "frame_id": stem, "split": split_name, "has_mask": has_mask,
            "tx": pose[0] if pose else "", "ty": pose[1] if pose else "", "tz": pose[2] if pose else "",
            "qx": pose[3] if pose else "", "qy": pose[4] if pose else "", "qz": pose[5] if pose else "",
            "qw": pose[6] if pose else "",
        })
    cap.release()
    return {"sequence": seq, "n_frames": len(rows), "rows": rows}


def _convert_one(args_tuple):
    seq, max_frames = args_tuple
    print(f"=== {seq} start ===", flush=True)
    summary = convert_sequence(seq, max_frames=max_frames)
    print(f"=== {seq} done: {summary['n_frames']} frames ===", flush=True)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", nargs="+", default=PILOT_SEQUENCES)
    ap.add_argument("--max-frames", type=int, default=0, help="cap frames per sequence (0 = no cap, for dry runs)")
    ap.add_argument("--workers", type=int, default=0, help="parallel sequences (0 = one worker per sequence)")
    args = ap.parse_args()

    workers = args.workers or len(args.sequences)
    jobs = [(seq, args.max_frames) for seq in args.sequences]
    with multiprocessing.Pool(min(workers, len(jobs))) as pool:
        summaries = pool.map(_convert_one, jobs)

    all_rows = []
    for summary in summaries:
        all_rows.extend(summary["rows"])

    manifest_dir = DATASET_DIR / "StereoMIS/curated/manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    cols = ["sequence", "frame_id", "split", "has_mask", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]
    with (manifest_dir / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)
    print(json.dumps({"total_frames": len(all_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
