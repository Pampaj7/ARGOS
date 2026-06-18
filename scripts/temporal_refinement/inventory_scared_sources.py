#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path


ROOT = Path("/dtu/p1/leopam/ARGOS")
RAW = ROOT / "dataset/SCARED/raw/source"
OUT = ROOT / "results/03_temporal_refinement/cache/large_v2_source_inventory.md"


def ffprobe(path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames,duration,r_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    try:
        return json.loads(subprocess.check_output(cmd, text=True))
    except Exception:
        return {}


def zip_video_entries(zip_path: Path):
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        for name in names:
            if name.endswith("rgb.mp4"):
                rows.append(name)
    return rows


def main():
    lines = ["# SCARED Temporal Source Inventory", ""]
    lines.append("## Raw Archives")
    lines.append("")
    lines.append("| Archive | Size | rgb.mp4 entries | GT depth maps | frame_data | scene_points |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for zip_path in sorted(RAW.glob("*.zip")):
        size_gb = zip_path.stat().st_size / (1024**3)
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            n_rgb = sum(n.endswith("rgb.mp4") for n in names)
            n_depth = sum("depth_map.tiff" in n for n in names)
            n_frame = sum("frame_data" in n for n in names)
            n_scene = sum("scene_points" in n for n in names)
        lines.append(f"| `{zip_path.name}` | {size_gb:.2f} GB | {n_rgb} | {n_depth} | {n_frame} | {n_scene} |")

    lines += ["", "## Extracted Sources", ""]
    lines.append("| Source | Frames | Size | Left/right extracted | Calibration | Notes |")
    lines.append("|---|---:|---|---|---|---|")
    for mp4 in sorted(ROOT.glob("dataset/**/*.mp4")):
        meta = ffprobe(mp4)
        stream = meta.get("streams", [{}])[0] if meta.get("streams") else {}
        frames = stream.get("nb_frames", "?")
        wh = f"{stream.get('width', '?')}x{stream.get('height', '?')}"
        seq_root = mp4.parent.parent if mp4.parent.name == "source" else mp4.parent
        left_right = (seq_root / "left").exists() and (seq_root / "right").exists()
        calib = any(mp4.parent.glob("*calib*")) or any(mp4.parent.glob("*calibration*"))
        lines.append(f"| `{mp4.relative_to(ROOT)}` | {frames} | {wh} | {left_right} | {calib} | stereo video likely vertical stack |")

    lines += [
        "",
        "## Recommendation",
        "",
        "- Use `test_dataset_8.zip` and `test_dataset_9.zip` first: they are small and contain `rgb.mp4` sequences plus calibration/keyframe images.",
        "- Split each `1280x2048` video frame vertically into left/right `1280x1024` images.",
        "- Use 8 test sequences initially, capped at about 125 frames each, to target about 1000 frames total.",
        "- Full `dataset_*.zip` archives include GT depth maps and scene points, but extracting all scene points is too large for this first pass.",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
