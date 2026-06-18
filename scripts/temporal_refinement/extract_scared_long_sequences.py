#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import zipfile
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.argos_paths import DATASET_DIR, RESULTS_DIR, EXTERNAL_DIR, FRAME_STEREO_REPOS_DIR, VIDEO_STEREO_REPOS_DIR

import cv2


ROOT = Path("/dtu/p1/leopam/ARGOS")
RAW = ROOT / "dataset/SCARED/raw/source"
OUT = ROOT / "results/04_dataset_derivatives/SCARED/scared_long_sequences"
SOURCE = OUT / "_source_videos"


def extract_member(zip_path: Path, member: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as src, dst.open("wb") as f:
        shutil.copyfileobj(src, f)


def split_video(video_path: Path, seq_id: str, max_frames: int):
    out = OUT / seq_id
    left_dir = out / "left"
    right_dir = out / "right"
    if (out / "metadata.json").exists() and len(list(left_dir.glob("*.png"))) >= max_frames:
        return json.loads((out / "metadata.json").read_text())
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    written = 0
    while written < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if h % 2 != 0:
            raise RuntimeError(f"Expected vertical stereo stack, got odd height {h} in {video_path}")
        left = frame[: h // 2]
        right = frame[h // 2 :]
        name = f"{written:06d}.png"
        cv2.imwrite(str(left_dir / name), left)
        cv2.imwrite(str(right_dir / name), right)
        written += 1
    cap.release()
    meta = {
        "sequence_id": seq_id,
        "source_video": str(video_path),
        "frames_written": written,
        "source_frames": total,
        "fps": fps,
        "image_shape": [h // 2, w, 3] if written else None,
        "stereo_layout": "vertical_stack_top_left_bottom_right",
        "rectification": "uses SCARED provided test rgb.mp4 stereo stream; no additional rectification applied",
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def parse_args():
    parser = argparse.ArgumentParser(description="Extract long consecutive SCARED stereo streams.")
    parser.add_argument("--max-per-sequence", type=int, default=130)
    return parser.parse_args()


def main():
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    zips = [RAW / "test_dataset_8.zip", RAW / "test_dataset_9.zip"]
    max_per_sequence = args.max_per_sequence
    metas = []
    for zip_path in zips:
        with zipfile.ZipFile(zip_path) as zf:
            rgb_members = sorted(n for n in zf.namelist() if n.endswith("rgb.mp4"))
            calib_members = sorted(n for n in zf.namelist() if n.endswith("endoscope_calibration.yaml"))
        for member in rgb_members:
            parts = Path(member).parts
            dataset = parts[0]
            keyframe = parts[1]
            seq_id = f"{dataset}_{keyframe}"
            video_dst = SOURCE / seq_id / "rgb.mp4"
            extract_member(zip_path, member, video_dst)
            for calib in calib_members:
                if f"/{keyframe}/" in f"/{calib}":
                    extract_member(zip_path, calib, SOURCE / seq_id / "endoscope_calibration.yaml")
            metas.append(split_video(video_dst, seq_id, max_per_sequence))
    summary = {
        "target_valid_5frame_windows": 1000,
        "max_per_sequence": max_per_sequence,
        "sequences": metas,
        "total_frames": sum(m["frames_written"] for m in metas),
        "estimated_valid_5frame_windows": sum(max(0, m["frames_written"] - 4) for m in metas),
    }
    (OUT / "metadata.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
