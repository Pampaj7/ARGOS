#!/usr/bin/env python3
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.argos_paths import DATASET_DIR, RESULTS_DIR

import cv2
import numpy as np


EVAL_ROOT = RESULTS_DIR / "02_video_stereo/stereoanyvideo_temporal_eval"
GT5_SOURCE = RESULTS_DIR / "02_video_stereo/test_sequence"
CONSEC32_SOURCE = DATASET_DIR / "SCARED/curated/consecutive32"
OUT = RESULTS_DIR / "03_temporal_refinement/cache/debug_v1"

S2M2_MODEL = "S2M2-L@736"
S2M2_DIRNAME = "S2M2-L_736"
TEACHER_MODEL = "StereoAnyVideo@384x640"
TEACHER_DIRNAME = "StereoAnyVideo_384x640"
WINDOW_RADIUS = 2
WINDOW_SIZE = 2 * WINDOW_RADIUS + 1


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Cannot read mask: {path}")
    return mask > 0


def colorize(x: np.ndarray, vmax=None, cmap=cv2.COLORMAP_TURBO) -> np.ndarray:
    x = x.astype(np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros((*x.shape, 3), dtype=np.uint8)
    if vmax is None:
        lo = float(np.nanpercentile(x[finite], 1))
        hi = float(np.nanpercentile(x[finite], 99))
    else:
        lo, hi = 0.0, float(vmax)
    if hi <= lo:
        return np.zeros((*x.shape, 3), dtype=np.uint8)
    y = (np.clip((x - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(y, cmap)


def collect_sequence(sequence: str):
    eval_seq = EVAL_ROOT / sequence
    s2m2_paths = sorted((eval_seq / "predictions" / S2M2_DIRNAME).glob("*.npy"))
    teacher_paths = sorted((eval_seq / "predictions" / TEACHER_DIRNAME).glob("*.npy"))
    if len(s2m2_paths) != len(teacher_paths) or not s2m2_paths:
        raise RuntimeError(f"Missing matching predictions for {sequence}")

    frames = []
    if sequence == "gt5":
        meta = json.loads((GT5_SOURCE / "metadata.json").read_text())
        for idx, frame in enumerate(meta["frames"]):
            frames.append(
                {
                    "frame_id": f"{idx:06d}",
                    "left_path": GT5_SOURCE / frame["left"],
                    "gt_disp_path": GT5_SOURCE / frame["gt_disparity"],
                    "gt_depth_path": GT5_SOURCE / frame["gt_depth"],
                    "valid_mask_path": GT5_SOURCE / frame["valid_mask"],
                    "fx": float(frame["fx"]),
                    "baseline_mm": float(frame["baseline_mm"]),
                }
            )
    else:
        left_paths = sorted((CONSEC32_SOURCE / "left").glob("*.png"))
        for idx, path in enumerate(left_paths):
            frames.append({"frame_id": path.stem, "left_path": path})

    if len(frames) != len(s2m2_paths):
        raise RuntimeError(f"Frame/prediction count mismatch for {sequence}: {len(frames)} vs {len(s2m2_paths)}")

    for idx, frame in enumerate(frames):
        frame["s2m2_path"] = s2m2_paths[idx]
        frame["teacher_path"] = teacher_paths[idx]
    return frames


def stats(prefix: str, x: np.ndarray, row: dict, mask: np.ndarray | None = None):
    finite = np.isfinite(x)
    if mask is not None:
        finite &= mask
    if not finite.any():
        row[f"{prefix}_min"] = np.nan
        row[f"{prefix}_max"] = np.nan
        row[f"{prefix}_mean"] = np.nan
        return
    row[f"{prefix}_min"] = float(np.min(x[finite]))
    row[f"{prefix}_max"] = float(np.max(x[finite]))
    row[f"{prefix}_mean"] = float(np.mean(x[finite]))


def validate_shape(name: str, arr: np.ndarray, expected):
    if arr.shape != expected:
        raise RuntimeError(f"{name} shape mismatch: got {arr.shape}, expected {expected}")


def write_sample(sample_idx: int, sequence: str, center_idx: int, frames: list[dict]):
    center = frames[center_idx]
    window = frames[center_idx - WINDOW_RADIUS : center_idx + WINDOW_RADIUS + 1]
    frame_ids = [f["frame_id"] for f in window]

    rgb = read_rgb(center["left_path"])
    h, w = rgb.shape[:2]
    s2m2_window = np.stack([np.load(f["s2m2_path"]).astype(np.float32) for f in window], axis=0)
    teacher_window = np.stack([np.load(f["teacher_path"]).astype(np.float32) for f in window], axis=0)
    teacher_center = teacher_window[WINDOW_RADIUS]
    s2m2_center = s2m2_window[WINDOW_RADIUS]

    validate_shape("s2m2_window", s2m2_window, (WINDOW_SIZE, h, w))
    validate_shape("teacher_window", teacher_window, (WINDOW_SIZE, h, w))

    payload = {
        "center_rgb": rgb.astype(np.uint8),
        "s2m2_l736_disp_window": s2m2_window,
        "stereoanyvideo_disp_center": teacher_center,
        "stereoanyvideo_disp_window": teacher_window,
        "frame_ids": np.array(frame_ids),
        "center_frame_id": np.array(center["frame_id"]),
        "has_gt": np.array(False),
        "scale_x": np.array(1.0, dtype=np.float32),
        "scale_y": np.array(1.0, dtype=np.float32),
    }

    valid_mask_ratio = ""
    if "gt_disp_path" in center:
        gt_disp = np.load(center["gt_disp_path"]).astype(np.float32)
        gt_depth = np.load(center["gt_depth_path"]).astype(np.float32)
        valid = read_mask(center["valid_mask_path"])
        validate_shape("gt_disp", gt_disp, (h, w))
        validate_shape("gt_depth", gt_depth, (h, w))
        validate_shape("valid_mask", valid, (h, w))
        payload["gt_disparity"] = gt_disp
        payload["gt_depth"] = gt_depth
        payload["valid_mask"] = valid.astype(np.uint8)
        payload["has_gt"] = np.array(True)
        payload["fx"] = np.array(center["fx"], dtype=np.float32)
        payload["baseline_mm"] = np.array(center["baseline_mm"], dtype=np.float32)
        valid_mask_ratio = float(valid.mean())

    sample_name = f"sample_{sample_idx:06d}.npz"
    sample_path = OUT / "samples" / sample_name
    if not sample_path.exists():
        np.savez_compressed(sample_path, **payload)

    diff = np.abs(s2m2_center - teacher_center)
    row = {
        "sample_id": sample_idx,
        "sample_path": f"samples/{sample_name}",
        "source_sequence": sequence,
        "center_index": center_idx,
        "center_frame_id": center["frame_id"],
        "window_frame_ids": " ".join(frame_ids),
        "height": h,
        "width": w,
        "window_size": WINDOW_SIZE,
        "has_gt": bool("gt_disp_path" in center),
        "valid_mask_ratio": valid_mask_ratio,
        "scale_x": 1.0,
        "scale_y": 1.0,
        "s2m2_model": S2M2_MODEL,
        "teacher_model": TEACHER_MODEL,
        "mean_abs_s2m2_teacher_diff": float(np.mean(diff[np.isfinite(diff)])),
    }
    stats("s2m2_center_disp", s2m2_center, row)
    stats("teacher_center_disp", teacher_center, row)
    if "gt_disparity" in payload:
        stats("gt_disp_valid", payload["gt_disparity"], row, payload["valid_mask"].astype(bool))
    return row


def make_montage(row: dict):
    sample = np.load(OUT / row["sample_path"])
    rgb = sample["center_rgb"]
    s2m2 = sample["s2m2_l736_disp_window"][WINDOW_RADIUS]
    teacher = sample["stereoanyvideo_disp_center"]
    diff = np.abs(s2m2 - teacher)
    valid = np.isfinite(s2m2) & np.isfinite(teacher)
    vmax = float(np.nanpercentile(np.concatenate([s2m2[valid], teacher[valid]]), 99)) if valid.any() else None
    tiles = [cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), colorize(s2m2, vmax), colorize(teacher, vmax), colorize(diff, 20.0, cv2.COLORMAP_MAGMA)]
    labels = ["RGB center", "S2M2-L@736", "StereoAnyVideo", "abs diff"]
    if bool(sample["has_gt"]):
        gt = sample["gt_disparity"]
        tiles.append(colorize(gt, vmax))
        labels.append("GT disp")
    small = []
    for tile, label in zip(tiles, labels):
        tile = cv2.resize(tile, (220, 176), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
        small.append(tile)
    out = np.concatenate(small, axis=1)
    cv2.imwrite(str(OUT / "sanity_montages" / f"sample_{int(row['sample_id']):06d}.png"), out)


def write_csv(rows):
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with (OUT / "index.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(rows):
    by_seq = {}
    for row in rows:
        seq = row["source_sequence"]
        by_seq.setdefault(seq, 0)
        by_seq[seq] += 1
    diffs = np.array([r["mean_abs_s2m2_teacher_diff"] for r in rows], dtype=np.float32)
    gt_ratios = [r["valid_mask_ratio"] for r in rows if r["valid_mask_ratio"] != ""]
    metadata = {
        "cache_name": "debug_v1",
        "output_root": str(OUT),
        "created_from": {
            "temporal_eval_root": str(EVAL_ROOT),
            "gt5_source": str(GT5_SOURCE),
            "consecutive32_source": str(CONSEC32_SOURCE),
        },
        "models": {
            "backbone": S2M2_MODEL,
            "teacher": TEACHER_MODEL,
        },
        "coordinate_system": "original image disparity coordinates",
        "scale_x": 1.0,
        "scale_y": 1.0,
        "window_radius": WINDOW_RADIUS,
        "window_size": WINDOW_SIZE,
        "sample_count": len(rows),
        "samples_by_sequence": by_seq,
        "quick_statistics": {
            "mean_abs_s2m2_teacher_diff_mean": float(diffs.mean()),
            "mean_abs_s2m2_teacher_diff_min": float(diffs.min()),
            "mean_abs_s2m2_teacher_diff_max": float(diffs.max()),
            "valid_mask_ratio_mean_where_gt_exists": float(np.mean(gt_ratios)) if gt_ratios else None,
        },
        "tensor_schema": {
            "center_rgb": "uint8 [H,W,3]",
            "s2m2_l736_disp_window": "float32 [5,H,W], original disparity coordinates",
            "stereoanyvideo_disp_center": "float32 [H,W], original disparity coordinates",
            "stereoanyvideo_disp_window": "float32 [5,H,W], original disparity coordinates",
            "gt_disparity": "optional float32 [H,W]",
            "gt_depth": "optional float32 [H,W] in mm",
            "valid_mask": "optional uint8 [H,W]",
        },
        "gt_note": "Raw GT disparity/depth arrays may contain invalid large or zero values outside valid_mask. Index GT statistics are computed only inside valid_mask.",
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def write_readme(rows):
    gt_count = sum(1 for r in rows if r["has_gt"])
    text = f"""# Temporal Refinement Cache Debug V1

First ARGOS cache for the lightweight temporal stereo residual refiner.

No training was performed.

## Contents

- `samples/*.npz`: compressed per-center-frame samples.
- `index.csv`: one row per sample.
- `metadata.json`: cache-level metadata and quick statistics.
- `sanity_montages/`: visual checks for representative samples.

## Sample Schema

Each sample stores:

- `center_rgb`: uint8 `[H,W,3]`;
- `s2m2_l736_disp_window`: float32 `[5,H,W]`, frames `t-2:t+2`;
- `stereoanyvideo_disp_center`: float32 `[H,W]`;
- `stereoanyvideo_disp_window`: float32 `[5,H,W]`;
- optional `gt_disparity`, `gt_depth`, `valid_mask` when GT exists;
- metadata arrays for frame ids and scale factors.

All disparity maps are stored in original image coordinates. `scale_x = 1.0`, `scale_y = 1.0` for this cache because the source evaluation already rescaled predictions back to original resolution.

## Counts

- total samples: `{len(rows)}`;
- GT samples: `{gt_count}`;
- non-GT temporal samples: `{len(rows) - gt_count}`.

## Sources

- `consecutive32`: 32-frame SCARED consecutive sequence from `results/stereoanyvideo_temporal_eval/consecutive32`.
- `gt5`: 5-frame GT subset from `results/stereoanyvideo_temporal_eval/gt5`.

This cache is intended for the first Tiny 2D U-Net residual refiner prototype, using frozen S2M2-L@736 predictions and StereoAnyVideo@384x640 teacher disparity.
"""
    (OUT / "README.md").write_text(text)


def main():
    (OUT / "samples").mkdir(parents=True, exist_ok=True)
    (OUT / "sanity_montages").mkdir(parents=True, exist_ok=True)
    rows = []
    sample_idx = 0
    for sequence in ["consecutive32", "gt5"]:
        frames = collect_sequence(sequence)
        for center_idx in range(WINDOW_RADIUS, len(frames) - WINDOW_RADIUS):
            rows.append(write_sample(sample_idx, sequence, center_idx, frames))
            sample_idx += 1
    if not rows:
        raise RuntimeError("No samples generated")
    write_csv(rows)
    write_metadata(rows)
    write_readme(rows)
    montage_indices = sorted(set([0, len(rows) // 4, len(rows) // 2, 3 * len(rows) // 4, len(rows) - 1]))
    for idx in montage_indices:
        make_montage(rows[idx])
    print(json.dumps({"samples": len(rows), "gt_samples": sum(1 for r in rows if r["has_gt"])}, indent=2))


if __name__ == "__main__":
    main()
