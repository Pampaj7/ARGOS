#!/usr/bin/env python3
"""Audit rectified temporal-GT SCARED sequences.

The script is read-only with respect to the dataset root. It writes audit
artifacts only under the requested output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency.
    tqdm = None


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
MATRIX_KEYS = ("P1", "P2", "R1", "R2", "Q")
DEFAULT_OUT = Path("dataset/SCARED/curated/audit/temporal_gt_rectified_integrity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("dataset/SCARED/curated/temporal_gt_rectified"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--valid-pixel-threshold", type=float, default=0.05)
    parser.add_argument("--extreme-depth-p99", type=float, default=1000.0)
    parser.add_argument("--extreme-disparity-p99", type=float, default=1000.0)
    return parser.parse_args()


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def as_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def mean(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return float(sum(finite) / len(finite)) if finite else math.nan


def min_f(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return float(min(finite)) if finite else math.nan


def max_f(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return float(max(finite)) if finite else math.nan


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def shape_str(shape: tuple[int, ...] | list[int] | None) -> str:
    return "x".join(map(str, shape)) if shape else ""


def frame_id(path: Path) -> str:
    return path.stem


def index_files(root: Path, exts: set[str]) -> dict[str, Path]:
    if not root.exists():
        return {}
    return {frame_id(p): p for p in sorted(root.iterdir()) if p.is_file() and p.suffix.lower() in exts}


def read_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def read_calibration(path: Path | None) -> tuple[dict[str, Any], list[str]]:
    if path is None or not path.exists():
        return {}, ["missing_calibration"]
    try:
        with path.open() as f:
            data = json.load(f)
    except Exception as exc:  # noqa: BLE001 - audit records bad files.
        return {}, [f"unreadable_calibration:{exc!r}"]

    flags: list[str] = []
    missing = [key for key in MATRIX_KEYS if key not in data]
    if missing:
        flags.append("missing_" + "_".join(missing))
    if "P1" not in data or "P2" not in data:
        flags.append("missing_P1_P2")
    return data, flags


def read_image_shape(path: Path | None) -> tuple[str, list[str]]:
    if path is None or not path.exists():
        return "", []
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return "", [f"unreadable_image:{path.name}"]
    return shape_str(img.shape), []


def load_array(path: Path | None) -> tuple[np.ndarray | None, str, list[str]]:
    if path is None or not path.exists():
        return None, "", []
    try:
        arr = np.load(path)
    except Exception as exc:  # noqa: BLE001 - audit records bad files.
        return None, "", [f"unreadable_array:{path.name}:{exc!r}"]
    return arr, shape_str(arr.shape), []


def valid_stats(depth: np.ndarray | None, disp: np.ndarray | None, mask: np.ndarray | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "valid_pixel_pct": math.nan,
        "depth_valid_min": math.nan,
        "depth_valid_median": math.nan,
        "depth_valid_max": math.nan,
        "depth_valid_p99": math.nan,
        "disparity_valid_min": math.nan,
        "disparity_valid_median": math.nan,
        "disparity_valid_max": math.nan,
        "disparity_valid_p99": math.nan,
        "has_non_finite": False,
    }
    arrays = [a for a in (depth, disp, mask) if a is not None]
    out["has_non_finite"] = any(not np.all(np.isfinite(a)) for a in arrays)
    if mask is None:
        return out

    valid = mask.astype(bool)
    out["valid_pixel_pct"] = float(np.mean(valid)) if valid.size else math.nan

    for name, arr in (("depth", depth), ("disparity", disp)):
        if arr is None:
            continue
        finite_valid = valid & np.isfinite(arr)
        vals = arr[finite_valid]
        if vals.size == 0:
            continue
        vals = vals.astype(np.float64, copy=False)
        out[f"{name}_valid_min"] = float(np.min(vals))
        out[f"{name}_valid_median"] = float(np.median(vals))
        out[f"{name}_valid_max"] = float(np.max(vals))
        out[f"{name}_valid_p99"] = float(np.percentile(vals, 99))
    return out


def audit_frame(
    sequence_id: str,
    fid: str,
    paths: dict[str, Path | None],
    args: argparse.Namespace,
) -> dict[str, Any]:
    flags: list[str] = []
    left_shape, left_flags = read_image_shape(paths["left"])
    right_shape, right_flags = read_image_shape(paths["right"])
    depth, depth_shape, depth_flags = load_array(paths["depth"])
    disp, disp_shape, disp_flags = load_array(paths["disparity"])
    mask, mask_shape, mask_flags = load_array(paths["valid"])
    calib, calib_flags = read_calibration(paths["calibration"])
    flags.extend(left_flags + right_flags + depth_flags + disp_flags + mask_flags + calib_flags)

    missing = [name for name, path in paths.items() if path is None or not path.exists()]
    flags.extend([f"missing_{name}" for name in missing])

    shapes = [s for s in (left_shape, right_shape, depth_shape, disp_shape, mask_shape) if s]
    spatial_shapes = [s.split("x")[:2] for s in shapes]
    if spatial_shapes and any(s != spatial_shapes[0] for s in spatial_shapes):
        flags.append("shape_mismatch")

    stats = valid_stats(depth, disp, mask)
    valid_pct = as_float(stats["valid_pixel_pct"])
    depth_med = as_float(stats["depth_valid_median"])
    disp_med = as_float(stats["disparity_valid_median"])
    if math.isfinite(valid_pct) and valid_pct < args.valid_pixel_threshold:
        flags.append("low_valid_pixel_pct")
    if math.isfinite(disp_med) and disp_med <= 0:
        flags.append("non_positive_disparity_median")
    if math.isfinite(depth_med) and depth_med <= 0:
        flags.append("non_positive_depth_median")
    if stats["has_non_finite"]:
        flags.append("non_finite_values")
    if as_float(stats["depth_valid_p99"]) > args.extreme_depth_p99:
        flags.append("extreme_depth_outlier")
    if as_float(stats["disparity_valid_p99"]) > args.extreme_disparity_p99:
        flags.append("extreme_disparity_outlier")

    return {
        "sequence_id": sequence_id,
        "frame_id": fid,
        "left_exists": paths["left"] is not None and paths["left"].exists(),
        "right_exists": paths["right"] is not None and paths["right"].exists(),
        "depth_exists": paths["depth"] is not None and paths["depth"].exists(),
        "disparity_exists": paths["disparity"] is not None and paths["disparity"].exists(),
        "valid_exists": paths["valid"] is not None and paths["valid"].exists(),
        "calibration_exists": paths["calibration"] is not None and paths["calibration"].exists(),
        "left_shape": left_shape,
        "right_shape": right_shape,
        "depth_shape": depth_shape,
        "disparity_shape": disp_shape,
        "valid_shape": mask_shape,
        "valid_pixel_pct": valid_pct,
        "depth_valid_min": stats["depth_valid_min"],
        "depth_valid_median": depth_med,
        "depth_valid_max": stats["depth_valid_max"],
        "disparity_valid_min": stats["disparity_valid_min"],
        "disparity_valid_median": disp_med,
        "disparity_valid_max": stats["disparity_valid_max"],
        "fx": as_float(calib.get("fx")),
        "baseline": as_float(calib.get("baseline", calib.get("baseline_mm"))),
        "has_P1_P2_R1_R2_Q": all(key in calib for key in MATRIX_KEYS),
        "flags": ";".join(dict.fromkeys(flags)),
        "passes_basic_quality": not flags,
    }


def audit_sequence(seq: Path, args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sequence_id = seq.name
    summary = read_summary(seq / "summary.json")
    left = index_files(seq / "left", IMAGE_EXTS)
    right = index_files(seq / "right", IMAGE_EXTS)
    depth = index_files(seq / "gt" / "DepthL_float32", {".npy"})
    disp = index_files(seq / "gt" / "Disparity_float32", {".npy"})
    valid = index_files(seq / "gt" / "ValidMask", {".npy"})
    calibration = index_files(seq / "calibration", {".json"})

    ids = sorted(set(left) | set(right) | set(depth) | set(disp) | set(valid) | set(calibration))
    if tqdm is not None:
        iterator = tqdm(ids, desc=sequence_id, unit="frame", leave=False)
    else:
        print(f"[{now()}] {sequence_id}: auditing {len(ids)} frames", file=sys.stderr, flush=True)
        iterator = ids

    frame_rows = []
    for idx, fid in enumerate(iterator, start=1):
        frame_rows.append(
            audit_frame(
                sequence_id,
                fid,
                {
                    "left": left.get(fid),
                    "right": right.get(fid),
                    "depth": depth.get(fid),
                    "disparity": disp.get(fid),
                    "valid": valid.get(fid),
                    "calibration": calibration.get(fid),
                },
                args,
            )
        )
        if tqdm is None and (idx == len(ids) or idx % 100 == 0):
            print(f"[{now()}] {sequence_id}: {idx}/{len(ids)} frames", file=sys.stderr, flush=True)

    counts = {
        "left": len(left),
        "right": len(right),
        "DepthL_float32": len(depth),
        "Disparity_float32": len(disp),
        "ValidMask": len(valid),
        "calibration": len(calibration),
    }
    expected = int(summary.get("num_frames") or summary.get("num_processed_frames") or summary.get("validation", {}).get("expected_count") or len(ids))
    complete_counts = all(count == expected for count in counts.values())
    complete_frames = all(
        row["left_exists"]
        and row["right_exists"]
        and row["depth_exists"]
        and row["disparity_exists"]
        and row["valid_exists"]
        and row["calibration_exists"]
        for row in frame_rows
    )
    warnings = list(summary.get("warnings") or [])
    bad_frames = [row for row in frame_rows if row["flags"]]
    if bad_frames:
        warnings.append(f"{len(bad_frames)} flagged frames")
    if not complete_counts:
        warnings.append("file counts do not match expected frame count")

    seq_row = {
        "sequence_id": sequence_id,
        "num_frames": expected,
        "count_left": counts["left"],
        "count_right": counts["right"],
        "count_DepthL_float32": counts["DepthL_float32"],
        "count_Disparity_float32": counts["Disparity_float32"],
        "count_ValidMask": counts["ValidMask"],
        "count_calibration": counts["calibration"],
        "is_complete": complete_counts and complete_frames and bool(frame_rows),
        "valid_pixel_pct_mean": mean([as_float(r["valid_pixel_pct"]) for r in frame_rows]),
        "valid_pixel_pct_min": min_f([as_float(r["valid_pixel_pct"]) for r in frame_rows]),
        "valid_pixel_pct_max": max_f([as_float(r["valid_pixel_pct"]) for r in frame_rows]),
        "depth_median_mean": mean([as_float(r["depth_valid_median"]) for r in frame_rows]),
        "disparity_median_mean": mean([as_float(r["disparity_valid_median"]) for r in frame_rows]),
        "fx_mean": mean([as_float(r["fx"]) for r in frame_rows]),
        "baseline_mean": mean([as_float(r["baseline"]) for r in frame_rows]),
        "has_P1_P2_R1_R2_Q": all(r["has_P1_P2_R1_R2_Q"] for r in frame_rows) if frame_rows else False,
        "num_flagged_frames": len(bad_frames),
        "num_quality_pass_frames": sum(bool(r["passes_basic_quality"]) for r in frame_rows),
        "warnings": ";".join(dict.fromkeys(map(str, warnings))),
    }
    return seq_row, frame_rows


def make_readme(summary: dict[str, Any], sequence_rows: list[dict[str, Any]], frame_rows: list[dict[str, Any]]) -> str:
    complete = [r for r in sequence_rows if r["is_complete"]]
    safe = [
        r for r in sequence_rows
        if r["is_complete"] and r["num_flagged_frames"] == 0 and r["has_P1_P2_R1_R2_Q"]
    ]
    cautious = [r for r in sequence_rows if r["num_flagged_frames"] or r["warnings"] or not r["is_complete"]]
    flagged_frames = [r for r in frame_rows if r["flags"]]

    safe_lines = "\n".join(
        f"- {r['sequence_id']}: {r['num_frames']} frames, valid pixel mean {r['valid_pixel_pct_mean']:.4f}"
        for r in safe
    ) or "- None"
    cautious_lines = "\n".join(
        f"- {r['sequence_id']}: {r['num_flagged_frames']} flagged frames; {r['warnings'] or 'no sequence warning'}"
        for r in cautious
    ) or "- None"
    frame_lines = "\n".join(
        f"- {r['sequence_id']}/{r['frame_id']}: {r['flags']}"
        for r in flagged_frames[:50]
    ) or "- None"
    if len(flagged_frames) > 50:
        frame_lines += f"\n- ... {len(flagged_frames) - 50} more flagged frames in frame_integrity.csv"

    return f"""# Rectified Temporal-GT Integrity Audit

Generated: {summary['generated_at']}

## Summary

- Complete sequences: {len(complete)} / {len(sequence_rows)}
- Available frames: {summary['total_frames']}
- Frames passing basic quality filter: {summary['quality_pass_frames']} / {summary['total_frames']}
- Flagged frames: {summary['flagged_frames']}

## Safest Sequences For Evaluation

{safe_lines}

## Exclude Or Treat Cautiously

{cautious_lines}

## Flagged Keyframes

{frame_lines}

## Files

- `sequence_integrity.csv`: one row per sequence.
- `frame_integrity.csv`: one row per frame.
- `audit_summary.json`: machine-readable aggregate summary.
- `run.log`: command and progress log.
"""


def main() -> int:
    args = parse_args()
    root = args.root
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    log_lines = [
        f"[{now()}] audit_scared_temporal_gt_rectified.py",
        f"root={root}",
        f"out={out}",
    ]
    if not root.exists():
        log_lines.append(f"[{now()}] ERROR root does not exist")
        (out / "run.log").write_text("\n".join(log_lines) + "\n")
        return 2

    sequence_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    sequences = sorted(p for p in root.iterdir() if p.is_dir() and (p / "summary.json").exists())
    log_lines.append(f"[{now()}] found {len(sequences)} sequences")
    if tqdm is not None:
        sequence_iter = tqdm(sequences, desc="sequences", unit="seq")
    else:
        sequence_iter = sequences

    for seq in sequence_iter:
        log_lines.append(f"[{now()}] auditing {seq.name}")
        seq_row, rows = audit_sequence(seq, args)
        sequence_rows.append(seq_row)
        frame_rows.extend(rows)

    frame_fields = [
        "sequence_id", "frame_id",
        "left_exists", "right_exists", "depth_exists", "disparity_exists", "valid_exists", "calibration_exists",
        "left_shape", "right_shape", "depth_shape", "disparity_shape", "valid_shape",
        "valid_pixel_pct",
        "depth_valid_min", "depth_valid_median", "depth_valid_max",
        "disparity_valid_min", "disparity_valid_median", "disparity_valid_max",
        "fx", "baseline", "has_P1_P2_R1_R2_Q", "flags", "passes_basic_quality",
    ]
    seq_fields = [
        "sequence_id", "num_frames",
        "count_left", "count_right", "count_DepthL_float32", "count_Disparity_float32", "count_ValidMask", "count_calibration",
        "is_complete",
        "valid_pixel_pct_mean", "valid_pixel_pct_min", "valid_pixel_pct_max",
        "depth_median_mean", "disparity_median_mean", "fx_mean", "baseline_mean",
        "has_P1_P2_R1_R2_Q", "num_flagged_frames", "num_quality_pass_frames", "warnings",
    ]
    write_csv(out / "sequence_integrity.csv", sequence_rows, seq_fields)
    write_csv(out / "frame_integrity.csv", frame_rows, frame_fields)

    summary = {
        "generated_at": now(),
        "root": str(root),
        "output": str(out),
        "num_sequences": len(sequence_rows),
        "complete_sequences": sum(bool(r["is_complete"]) for r in sequence_rows),
        "total_frames": len(frame_rows),
        "quality_pass_frames": sum(bool(r["passes_basic_quality"]) for r in frame_rows),
        "flagged_frames": sum(bool(r["flags"]) for r in frame_rows),
        "safe_sequences_for_evaluation": [
            r["sequence_id"]
            for r in sequence_rows
            if r["is_complete"] and r["num_flagged_frames"] == 0 and r["has_P1_P2_R1_R2_Q"]
        ],
        "cautious_sequences": [
            r["sequence_id"]
            for r in sequence_rows
            if r["num_flagged_frames"] or r["warnings"] or not r["is_complete"]
        ],
        "thresholds": {
            "valid_pixel_pct_min": args.valid_pixel_threshold,
            "extreme_depth_p99": args.extreme_depth_p99,
            "extreme_disparity_p99": args.extreme_disparity_p99,
        },
    }
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=2, default=json_default) + "\n")
    (out / "README.md").write_text(make_readme(summary, sequence_rows, frame_rows))
    log_lines.append(f"[{now()}] wrote sequence_integrity.csv")
    log_lines.append(f"[{now()}] wrote frame_integrity.csv")
    log_lines.append(f"[{now()}] wrote audit_summary.json")
    log_lines.append(f"[{now()}] wrote README.md")
    log_lines.append(f"[{now()}] complete")
    (out / "run.log").write_text("\n".join(log_lines) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
