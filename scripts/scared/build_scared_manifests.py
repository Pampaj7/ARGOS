#!/usr/bin/env python3
"""Rebuild dataset/SCARED/curated/manifests/ (strong_keyframes + temporal_sequences).

Run after populate_strong_keyframes.py / extract_scared_full.py, or whenever the two
curated collections change. Produces `has_strong_geometric_gt` true/false manifests so
downstream code can never accidentally treat GT-less temporal video as geometric ground
truth.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.argos_paths import SCARED_DIR

STRONG = SCARED_DIR / "curated/geometric_gt/strong_keyframes"
TEMPORAL = SCARED_DIR / "curated/temporal_sequences"
MANIFESTS = SCARED_DIR / "curated/manifests"


def build_strong_keyframes_manifest() -> int:
    rows = []
    for dataset_dir in sorted(STRONG.glob("dataset_*"), key=lambda p: int(p.name.split("_")[1])):
        dataset_id = dataset_dir.name
        for kf_dir in sorted(dataset_dir.glob("keyframe_*"), key=lambda p: int(p.name.split("_")[1])):
            keyframe_id = kf_dir.name
            ptcloud = "point_cloud.obj" if (kf_dir / "point_cloud.obj").exists() else "left_point_cloud.obj;right_point_cloud.obj"
            rel = kf_dir.relative_to(SCARED_DIR)
            rows.append({
                "dataset_id": dataset_id,
                "keyframe_id": keyframe_id,
                "left_image_path": str(rel / "Left_Image.png"),
                "right_image_path": str(rel / "Right_Image.png"),
                "left_depth_map_path": str(rel / "left_depth_map.tiff"),
                "right_depth_map_path": str(rel / "right_depth_map.tiff"),
                "pointcloud_path": ";".join(str(rel / p) for p in ptcloud.split(";")),
                "calibration_path": str(rel / "endoscope_calibration.yaml"),
                "gt_source": "official structured-light (Gray-code) keyframe scan",
                "has_strong_geometric_gt": True,
            })

    cols = ["dataset_id", "keyframe_id", "left_image_path", "right_image_path", "left_depth_map_path",
            "right_depth_map_path", "pointcloud_path", "calibration_path", "gt_source", "has_strong_geometric_gt"]
    with (MANIFESTS / "strong_keyframes_manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"strong_keyframes_manifest.csv: {len(rows)} rows")
    return len(rows)


def build_temporal_sequences_manifest() -> int:
    meta = json.loads((TEMPORAL / "metadata.json").read_text())
    rows = []
    for seq in meta["sequences"]:
        seq_id = seq["sequence_id"]
        parts = seq_id.split("_keyframe_")
        dataset_id = parts[0]
        anchor_keyframe = f"keyframe_{parts[1]}" if len(parts) > 1 else ""
        rows.append({
            "sequence_id": seq_id,
            "dataset_id": dataset_id,
            "anchor_keyframe": anchor_keyframe,
            "left_dir": f"curated/temporal_sequences/{seq_id}/left",
            "right_dir": f"curated/temporal_sequences/{seq_id}/right",
            "frames_written": seq["frames_written"],
            "source_frames": seq["source_frames"],
            "fps": seq["fps"],
            "height": seq["image_shape"][0] if seq.get("image_shape") else "",
            "width": seq["image_shape"][1] if seq.get("image_shape") else "",
            "stereo_layout": seq["stereo_layout"],
            "gt_source": "none (real stereo video; not official dense ground truth)",
            "has_strong_geometric_gt": False,
        })

    cols = ["sequence_id", "dataset_id", "anchor_keyframe", "left_dir", "right_dir", "frames_written",
            "source_frames", "fps", "height", "width", "stereo_layout", "gt_source", "has_strong_geometric_gt"]
    with (MANIFESTS / "temporal_sequences_manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"temporal_sequences_manifest.csv: {len(rows)} rows")
    return len(rows)


def main() -> int:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    build_strong_keyframes_manifest()
    build_temporal_sequences_manifest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
