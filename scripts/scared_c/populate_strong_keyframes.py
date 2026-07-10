#!/usr/bin/env python3
"""Populate dataset/SCARED-C/curated/geometric_gt/strong_keyframes/ (25 real structured-light
keyframes: dataset_1,2,3,6,7 x keyframe_1..5). Identical convention to vanilla SCARED's
populate_strong_keyframes.py, since SCARED-C keeps the original keyframe GT unchanged and
only replaces the non-keyframe frame poses.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.argos_paths import DATASET_DIR

RAW = DATASET_DIR / "SCARED-C/raw"
OUT = DATASET_DIR / "SCARED-C/curated/geometric_gt/strong_keyframes"

REQUIRED = ["Left_Image.png", "Right_Image.png", "left_depth_map.tiff", "right_depth_map.tiff", "endoscope_calibration.yaml"]


def main() -> int:
    rows = []
    for dataset_dir in sorted(RAW.glob("dataset_*"), key=lambda p: int(p.name.split("_")[1])):
        dataset_id = dataset_dir.name
        for kf_dir in sorted(dataset_dir.glob("keyframe_*"), key=lambda p: int(p.name.split("_")[1])):
            if not all((kf_dir / f).exists() for f in REQUIRED):
                continue  # keyframe_5-style single-image keyframes with no video sequence still qualify; only skip if truly incomplete
            keyframe_id = kf_dir.name
            out_dir = OUT / dataset_id / keyframe_id
            out_dir.mkdir(parents=True, exist_ok=True)
            copied = list(REQUIRED)
            for fname in REQUIRED:
                shutil.copy2(kf_dir / fname, out_dir / fname)
            if (kf_dir / "point_cloud.obj").exists():
                shutil.copy2(kf_dir / "point_cloud.obj", out_dir / "point_cloud.obj")
                copied.append("point_cloud.obj")
            elif (kf_dir / "left_point_cloud.obj").exists():
                for fname in ["left_point_cloud.obj", "right_point_cloud.obj"]:
                    shutil.copy2(kf_dir / fname, out_dir / fname)
                    copied.append(fname)
            if (kf_dir / "intrinsics_colmap.yaml").exists():
                shutil.copy2(kf_dir / "intrinsics_colmap.yaml", out_dir / "intrinsics_colmap.yaml")
                copied.append("intrinsics_colmap.yaml")
            rows.append({"dataset_id": dataset_id, "keyframe_id": keyframe_id, "files_copied": ";".join(copied)})
            print(f"{dataset_id}/{keyframe_id} -> {len(copied)} files")

    (OUT.parent / "_strong_keyframes_build_manifest.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps({"total_keyframes": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
