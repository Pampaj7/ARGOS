#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import random
import shutil
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.argos_paths import DATASET_DIR, RESULTS_DIR, EXTERNAL_DIR, FRAME_STEREO_REPOS_DIR, VIDEO_STEREO_REPOS_DIR

import cv2
import numpy as np

from scripts.temporal_refinement.build_debug_cache import colorize


def copy_available_cache(src: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    (out / "samples").mkdir(exist_ok=True)
    (out / "sanity_montages").mkdir(exist_ok=True)
    rows = list(csv.DictReader((src / "index.csv").open()))
    new_rows = []
    for row in rows:
        sample_id = int(row["sample_id"])
        src_sample = src / row["sample_path"]
        dst_name = f"sample_{sample_id:06d}.npz"
        dst_sample = out / "samples" / dst_name
        if dst_sample.exists():
            dst_sample.unlink()
        os.link(src_sample, dst_sample)
        row = dict(row)
        row["sample_path"] = f"samples/{dst_name}"
        row["large_cache_source"] = "debug_v1_available_predictions"
        new_rows.append(row)
    return new_rows


def write_index(out: Path, rows: list[dict]):
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with (out / "index.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def make_montage(out: Path, row: dict):
    sample = np.load(out / row["sample_path"])
    rgb = sample["center_rgb"]
    s2m2 = sample["s2m2_l736_disp_window"][2]
    teacher = sample["stereoanyvideo_disp_center"]
    diff = np.abs(s2m2 - teacher)
    valid = np.isfinite(s2m2) & np.isfinite(teacher)
    vmax = float(np.nanpercentile(np.concatenate([s2m2[valid], teacher[valid]]), 99)) if valid.any() else None
    tiles = [
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        colorize(s2m2, vmax),
        colorize(teacher, vmax),
        colorize(diff, 8.0, cv2.COLORMAP_MAGMA),
    ]
    labels = ["RGB", "S2M2-L@736", "StereoAnyVideo", "abs diff"]
    if bool(sample["has_gt"]):
        tiles.append(colorize(sample["gt_disparity"], vmax))
        labels.append("GT disp")
    small = []
    for tile, label in zip(tiles, labels):
        tile = cv2.resize(tile, (220, 176), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
        small.append(tile)
    cv2.imwrite(str(out / "sanity_montages" / f"sample_{int(row['sample_id']):06d}.png"), np.concatenate(small, axis=1))


def write_metadata(out: Path, rows: list[dict], src: Path):
    diffs = np.array([float(r["mean_abs_s2m2_teacher_diff"]) for r in rows], dtype=np.float32)
    seq_counts = {}
    for row in rows:
        seq_counts[row["source_sequence"]] = seq_counts.get(row["source_sequence"], 0) + 1
    gt_ratios = [float(r["valid_mask_ratio"]) for r in rows if r.get("valid_mask_ratio")]
    metadata = {
        "cache_name": "large_v1",
        "status": "seed cache from all currently available complete S2M2-L@736 and StereoAnyVideo predictions",
        "target_sample_count": {"minimum": 1000, "eventual": 5000},
        "actual_sample_count": len(rows),
        "limitation": "ARGOS currently has complete S2M2/StereoAnyVideo prediction pairs for consecutive32 and gt5 only. Raw SCARED video archives exist, but long-sequence stereo predictions still need to be generated before this cache can reach 1000+ samples.",
        "source_cache": str(src),
        "output_root": str(out),
        "samples_by_sequence": seq_counts,
        "coordinate_system": "original image disparity coordinates",
        "window_size": 5,
        "models": {"backbone": "S2M2-L@736", "teacher": "StereoAnyVideo@384x640"},
        "quick_statistics": {
            "mean_abs_s2m2_teacher_diff_mean": float(diffs.mean()) if len(diffs) else None,
            "mean_abs_s2m2_teacher_diff_min": float(diffs.min()) if len(diffs) else None,
            "mean_abs_s2m2_teacher_diff_max": float(diffs.max()) if len(diffs) else None,
            "valid_gt_ratio_mean_where_available": float(np.mean(gt_ratios)) if gt_ratios else None,
        },
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def write_readme(out: Path, rows: list[dict]):
    text = f"""# Temporal Refinement Cache Large V1

This is the first larger-cache scaffold for ARGOS temporal refinement.

Actual samples: `{len(rows)}`.

Important limitation: the requested target is 1,000-5,000 samples, but only the existing `consecutive32` and `gt5` sequences currently have complete S2M2-L@736 and StereoAnyVideo predictions in ARGOS. This cache therefore seeds `large_v1` with all complete samples and keeps the schema ready for expansion once long SCARED video predictions are generated.

Payloads:

- `samples/*.npz`: ignored by Git;
- `index.csv`: tracked summary/index;
- `metadata.json`: cache metadata and statistics;
- `sanity_montages/`: random sample checks.

All disparity maps are stored in original image coordinates.
"""
    (out / "README.md").write_text(text)


def main():
    root = Path("/dtu/p1/leopam/ARGOS")
    src = root / "results/03_temporal_refinement/cache/debug_v1"
    out = root / "results/03_temporal_refinement/cache/large_v1"
    if out.exists():
        shutil.rmtree(out)
    rows = copy_available_cache(src, out)
    write_index(out, rows)
    random.seed(7)
    for row in random.sample(rows, min(20, len(rows))):
        make_montage(out, row)
    write_metadata(out, rows, src)
    write_readme(out, rows)
    print(f"Wrote {len(rows)} samples to {out}")


if __name__ == "__main__":
    main()
