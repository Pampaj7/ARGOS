#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import random
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.argos_paths import RESULTS_DIR

import cv2
import numpy as np

from scripts.temporal_refinement.build_debug_cache import colorize


SEQ_ROOT = RESULTS_DIR / "04_dataset_derivatives/SCARED/scared_long_sequences"
S2M2_ROOT = RESULTS_DIR / "04_dataset_derivatives/SCARED/scared_long_predictions/s2m2_l736"
SAV_ROOT = RESULTS_DIR / "04_dataset_derivatives/SCARED/scared_long_predictions/stereoanyvideo_384x640"
OUT = RESULTS_DIR / "03_temporal_refinement/cache/large_v2"


def read_rgb(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def stats(prefix, x, row):
    finite = np.isfinite(x)
    row[f"{prefix}_min"] = float(np.min(x[finite]))
    row[f"{prefix}_max"] = float(np.max(x[finite]))
    row[f"{prefix}_mean"] = float(np.mean(x[finite]))


def write_sample(sample_id: int, seq: Path, center_idx: int, frame_ids: list[str]):
    center = frame_ids[center_idx]
    window_ids = frame_ids[center_idx - 2 : center_idx + 3]
    rgb = read_rgb(seq / "left" / f"{center}.png")
    h, w = rgb.shape[:2]
    s2m2_window = np.stack([np.load(S2M2_ROOT / seq.name / "disp" / f"{fid}.npy").astype(np.float32) for fid in window_ids])
    teacher_window = np.stack([np.load(SAV_ROOT / seq.name / "disp" / f"{fid}.npy").astype(np.float32) for fid in window_ids])
    if s2m2_window.shape != (5, h, w) or teacher_window.shape != (5, h, w):
        raise RuntimeError(f"Shape mismatch {seq.name} {center}: {s2m2_window.shape} {teacher_window.shape} {(h,w)}")
    sample_name = f"sample_{sample_id:06d}.npz"
    np.savez_compressed(
        OUT / "samples" / sample_name,
        center_rgb=rgb.astype(np.uint8),
        s2m2_l736_disp_window=s2m2_window,
        s2m2_l736_disp_center=s2m2_window[2],
        stereoanyvideo_disp_center=teacher_window[2],
        stereoanyvideo_disp_window=teacher_window,
        frame_ids=np.array(window_ids),
        center_frame_id=np.array(center),
        source_sequence=np.array(seq.name),
        has_gt=np.array(False),
        scale_x=np.array(1.0, dtype=np.float32),
        scale_y=np.array(1.0, dtype=np.float32),
    )
    diff = np.abs(s2m2_window[2] - teacher_window[2])
    row = {
        "sample_id": sample_id,
        "sample_path": f"samples/{sample_name}",
        "source_sequence": seq.name,
        "center_index": center_idx,
        "center_frame_id": center,
        "window_frame_ids": " ".join(window_ids),
        "height": h,
        "width": w,
        "has_gt": False,
        "scale_x": 1.0,
        "scale_y": 1.0,
        "mean_abs_s2m2_teacher_diff": float(np.mean(diff[np.isfinite(diff)])),
    }
    stats("s2m2_center_disp", s2m2_window[2], row)
    stats("teacher_center_disp", teacher_window[2], row)
    return row


def montage(row):
    sample = np.load(OUT / row["sample_path"])
    rgb = sample["center_rgb"]
    s2m2 = sample["s2m2_l736_disp_center"]
    teacher = sample["stereoanyvideo_disp_center"]
    diff = np.abs(s2m2 - teacher)
    vmax = float(np.nanpercentile(np.concatenate([s2m2.ravel(), teacher.ravel()]), 99))
    tiles = [cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), colorize(s2m2, vmax), colorize(teacher, vmax), colorize(diff, 8.0, cv2.COLORMAP_MAGMA)]
    labels = ["RGB", "S2M2-L@736", "StereoAnyVideo", "abs diff"]
    small = []
    for tile, label in zip(tiles, labels):
        tile = cv2.resize(tile, (220, 176), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
        small.append(tile)
    cv2.imwrite(str(OUT / "sanity_montages" / f"sample_{int(row['sample_id']):06d}.png"), np.concatenate(small, axis=1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "samples").mkdir(exist_ok=True)
    (OUT / "sanity_montages").mkdir(exist_ok=True)
    rows = []
    sample_id = 0
    for seq in sorted(d for d in SEQ_ROOT.iterdir() if d.is_dir() and not d.name.startswith("_")):
        left_ids = [p.stem for p in sorted((seq / "left").glob("*.png"))]
        ids = [fid for fid in left_ids if (S2M2_ROOT / seq.name / "disp" / f"{fid}.npy").exists() and (SAV_ROOT / seq.name / "disp" / f"{fid}.npy").exists()]
        for center_idx in range(2, len(ids) - 2):
            rows.append(write_sample(sample_id, seq, center_idx, ids))
            sample_id += 1
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with (OUT / "index.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    random.seed(7)
    for row in random.sample(rows, min(20, len(rows))):
        montage(row)
    diffs = np.array([r["mean_abs_s2m2_teacher_diff"] for r in rows], dtype=np.float32)
    by_seq = {}
    for row in rows:
        by_seq[row["source_sequence"]] = by_seq.get(row["source_sequence"], 0) + 1
    meta = {
        "cache_name": "large_v2",
        "sample_count": len(rows),
        "samples_by_sequence": by_seq,
        "coordinate_system": "original image disparity coordinates",
        "window_size": 5,
        "quick_statistics": {
            "mean_abs_s2m2_teacher_diff_mean": float(diffs.mean()) if len(diffs) else None,
            "mean_abs_s2m2_teacher_diff_min": float(diffs.min()) if len(diffs) else None,
            "mean_abs_s2m2_teacher_diff_max": float(diffs.max()) if len(diffs) else None,
        },
    }
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    (OUT / "README.md").write_text(f"# Temporal Refinement Cache Large V2\n\nSamples: `{len(rows)}`.\n\nGenerated from `results/04_dataset_derivatives/SCARED/scared_long_sequences/`, S2M2-L@736 predictions, and StereoAnyVideo@384x640 predictions.\n\nPayload `.npz` files are ignored by Git.\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
