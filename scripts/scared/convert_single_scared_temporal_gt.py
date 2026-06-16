#!/usr/bin/env python3
"""Convert one SCARED video block to rectified stereo frames with GT metadata."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

from convert_scared_warped_frames import (
    calib_from_frame_data,
    iter_scene_points,
    read_frame_data,
    read_video_frames,
    save_float_gt,
)
from convert_scared_keyframes import scatter_min_depth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--keyframe-id", required=True)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-count", type=int, default=130)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()

    selected_ids = set(range(args.frame_start, args.frame_start + args.frame_count))
    seq_id = f"test_{args.dataset_id}_{args.keyframe_id}"
    out_dir = args.out_root / seq_id
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    with zipfile.ZipFile(args.zip_path) as zf:
        prefix = f"{args.dataset_id}/{args.keyframe_id}/data"
        frame_data_path = f"{prefix}/frame_data.tar.gz"
        scene_points_path = f"{prefix}/scene_points.tar.gz"
        rgb_path = f"{prefix}/rgb.mp4"
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

            stem = f"{fid:06d}"
            left_path = out_dir / "left" / f"{stem}.png"
            right_path = out_dir / "right" / f"{stem}.png"
            left_path.parent.mkdir(parents=True, exist_ok=True)
            right_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(left_path), left_rect)
            cv2.imwrite(str(right_path), right_rect)

            ref_dir = out_dir / "gt"
            valid, gt_paths = save_float_gt(ref_dir, stem, depth, disp)
            calib_path = out_dir / "calibration" / f"{stem}.json"
            calib_path.parent.mkdir(parents=True, exist_ok=True)
            calib_path.write_text(
                json.dumps(
                    {
                        "P1": {"rows": 3, "cols": 4, "data": p1.astype(float).reshape(-1).tolist()},
                        "P2": {"rows": 3, "cols": 4, "data": p2.astype(float).reshape(-1).tolist()},
                    },
                    indent=2,
                )
                + "\n"
            )
            rows.append(
                {
                    "sequence_id": seq_id,
                    "frame_id": stem,
                    "left_path": str(left_path),
                    "right_path": str(right_path),
                    "depth_float32_path": str(gt_paths["depth"]),
                    "disparity_float32_path": str(gt_paths["disp"]),
                    "valid_mask_path": str(gt_paths["mask"]),
                    "calibration_path": str(calib_path),
                    "valid_pixel_ratio": float(valid.mean()),
                }
            )
            print(f"{seq_id}/{stem} valid={valid.mean():.3f}", flush=True)

    metadata_path = out_dir / "metadata.csv"
    with metadata_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "sequence_id": seq_id,
                "frames": len(rows),
                "frame_start": args.frame_start,
                "frame_count_requested": args.frame_count,
                "valid_pixel_ratio_mean": float(np.mean([r["valid_pixel_ratio"] for r in rows])),
                "metadata_csv": str(metadata_path),
            },
            indent=2,
        )
        + "\n"
    )
    print(metadata_path)


if __name__ == "__main__":
    main()
