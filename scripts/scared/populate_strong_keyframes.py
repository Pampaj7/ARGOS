#!/usr/bin/env python3
"""Populate curated/geometric_gt/strong_keyframes/ from raw/extracted (official SCARED keyframe GT only).

Copies, per keyframe, only the files that carry real structured-light ground truth:
Left/Right_Image.png, left/right_depth_map.tiff, point_cloud.obj (or the
left_point_cloud.obj/right_point_cloud.obj naming variant used by dataset_4-7),
endoscope_calibration.yaml. Does not touch data/ (unrectified video + kinematics-propagated
scene_points.tar.gz), which is never used as ground truth in ARGOS.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.argos_paths import SCARED_DIR

RAW = SCARED_DIR / "raw/extracted"
OUT = SCARED_DIR / "curated/geometric_gt/strong_keyframes"

REQUIRED = ["Left_Image.png", "Right_Image.png", "left_depth_map.tiff", "right_depth_map.tiff", "endoscope_calibration.yaml"]
PTCLOUD_VARIANTS = [["point_cloud.obj"], ["left_point_cloud.obj", "right_point_cloud.obj"]]


def main() -> int:
    rows = []
    for dataset_dir in sorted(RAW.glob("dataset_*"), key=lambda p: int(p.name.split("_")[1])):
        dataset_id = dataset_dir.name
        inner = dataset_dir / dataset_id
        if not inner.is_dir():
            raise RuntimeError(f"unexpected layout: {inner}")
        for kf_dir in sorted(inner.glob("keyframe_*"), key=lambda p: int(p.name.split("_")[1])):
            keyframe_id = kf_dir.name
            out_dir = OUT / dataset_id / keyframe_id
            out_dir.mkdir(parents=True, exist_ok=True)
            copied = []
            for fname in REQUIRED:
                src = kf_dir / fname
                if not src.exists():
                    raise RuntimeError(f"missing required file {src}")
                shutil.copy2(src, out_dir / fname)
                copied.append(fname)
            ptcloud_used = None
            for variant in PTCLOUD_VARIANTS:
                if all((kf_dir / f).exists() for f in variant):
                    for fname in variant:
                        shutil.copy2(kf_dir / fname, out_dir / fname)
                        copied.append(fname)
                    ptcloud_used = "+".join(variant)
                    break
            if ptcloud_used is None:
                raise RuntimeError(f"no point cloud variant found in {kf_dir}")
            rows.append({
                "dataset_id": dataset_id,
                "keyframe_id": keyframe_id,
                "source_path": str(kf_dir),
                "output_path": str(out_dir),
                "files_copied": ";".join(copied),
                "pointcloud_variant": ptcloud_used,
                "has_strong_geometric_gt": True,
            })
            print(f"{dataset_id}/{keyframe_id} -> {len(copied)} files ({ptcloud_used})")

    (OUT.parent / "_strong_keyframes_build_manifest.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps({"total_keyframes": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
