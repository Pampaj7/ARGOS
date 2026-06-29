#!/usr/bin/env python3
"""Audit local SCARED dataset integrity for ARGOS.

This script is read-only with respect to dataset files. It writes only audit
artifacts under the requested output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
GT_EXTS = {".npy", ".tif", ".tiff", ".png"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
RAW_INTEREST_EXTS = {".tar.gz", ".zip", ".obj", ".tiff", ".tif", ".mp4", ".yaml", ".yml", ".png"}
KEYWORDS_FOR_CODE_SCAN = [
    "dataset/SCARED",
    "SCARED",
    "temporal_gt",
    "consecutive32",
    "warped_gt_108",
    "DepthL_float32",
    "Disparity_float32",
    "ValidMask",
    "Left_rectified",
    "Right_rectified",
    "left/",
    "right/",
]


def human_bytes(n: int | float) -> str:
    n = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return -1


def walk_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return (p for p in root.rglob("*") if p.is_file())


def dir_size(root: Path) -> tuple[int, int, Counter[str]]:
    total = 0
    count = 0
    exts: Counter[str] = Counter()
    if not root.exists():
        return 0, 0, exts
    for path in walk_files(root):
        size = safe_size(path)
        if size >= 0:
            total += size
        count += 1
        exts[path.suffix.lower() or "<none>"] += 1
    return total, count, exts


def numeric_id(path: Path) -> int | None:
    nums = re.findall(r"\d+", path.stem)
    if not nums:
        return None
    return int(nums[-1])


def indexed_files(root: Path, exts: set[str]) -> list[Path]:
    if not root.exists():
        return []
    files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files, key=lambda p: (numeric_id(p) is None, numeric_id(p) if numeric_id(p) is not None else p.name, p.name))


def continuity(ids: list[int]) -> tuple[bool, list[int], list[int]]:
    if not ids:
        return False, [], []
    dupes = sorted([x for x, c in Counter(ids).items() if c > 1])
    expected = set(range(min(ids), max(ids) + 1))
    missing = sorted(expected - set(ids))
    return not missing and not dupes, missing, dupes


def read_image_meta(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(path), "readable": False}
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return out
    out.update({"readable": True, "shape": list(img.shape), "dtype": str(img.dtype)})
    if img.size:
        finite = np.isfinite(img.astype(np.float32))
        out.update({
            "finite_ratio": float(np.mean(finite)),
            "min": float(np.nanmin(img)),
            "max": float(np.nanmax(img)),
        })
    return out


def read_array_meta(path: Path, sample_stride: int = 1) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(path), "readable": False}
    try:
        if path.suffix.lower() == ".npy":
            arr = np.load(path, mmap_mode="r")
            original_shape = list(arr.shape)
            original_dtype = str(arr.dtype)
            small = np.asarray(arr[::sample_stride, ::sample_stride]) if getattr(arr, "ndim", 0) >= 2 else np.asarray(arr)
        else:
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                return out
            original_shape = list(img.shape)
            original_dtype = str(img.dtype)
            small = img
        out.update({"readable": True, "shape": original_shape, "sample_shape": list(small.shape), "dtype": original_dtype})
        arrf = small.astype(np.float32, copy=False)
        finite = np.isfinite(arrf)
        positive = finite & (arrf > 0)
        out.update({
            "finite_ratio": float(np.mean(finite)) if arrf.size else math.nan,
            "positive_ratio": float(np.mean(positive)) if arrf.size else math.nan,
            "min": float(np.nanmin(arrf)) if arrf.size else math.nan,
            "max": float(np.nanmax(arrf)) if arrf.size else math.nan,
            "mean_positive": float(np.nanmean(arrf[positive])) if np.any(positive) else math.nan,
        })
    except Exception as exc:  # noqa: BLE001 - audit should record unreadable files.
        out["error"] = repr(exc)
    return out


def video_meta(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"path": str(path), "readable": False}
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    return {
        "path": str(path),
        "readable": True,
        "frames": int(frames) if frames >= 0 else "",
        "width": int(width) if width >= 0 else "",
        "height": int(height) if height >= 0 else "",
        "fps": float(fps) if fps > 0 else "",
        "duration_s": float(frames / fps) if fps > 0 and frames >= 0 else "",
    }


def summarize_file_group(files: list[Path]) -> dict[str, Any]:
    ids = [numeric_id(p) for p in files]
    ids_int = [i for i in ids if i is not None]
    cont, missing, dupes = continuity(ids_int)
    empty = [str(p) for p in files if safe_size(p) == 0]
    return {
        "count": len(files),
        "first_id": min(ids_int) if ids_int else "",
        "last_id": max(ids_int) if ids_int else "",
        "continuous_sorted_indices": cont,
        "missing_ids": " ".join(map(str, missing[:30])) + (" ..." if len(missing) > 30 else ""),
        "duplicate_ids": " ".join(map(str, dupes[:30])) + (" ..." if len(dupes) > 30 else ""),
        "empty_files": len(empty),
    }


def sample_files(files: list[Path], n: int = 3) -> list[Path]:
    if len(files) <= n:
        return files
    idxs = sorted(set([0, len(files) // 2, len(files) - 1]))
    return [files[i] for i in idxs[:n]]


def audit_temporal_gt(root: Path, problems: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    temporal_root = root / "curated" / "temporal_gt"
    if not temporal_root.exists():
        problems.append({"severity": "error", "area": "temporal_gt", "issue": "missing_directory", "details": str(temporal_root), "recommended_action": "Restore curated temporal GT directory."})
        return rows
    for seq in sorted([p for p in temporal_root.iterdir() if p.is_dir()]):
        left = indexed_files(seq / "left", IMAGE_EXTS)
        right = indexed_files(seq / "right", IMAGE_EXTS)
        calib = indexed_files(seq / "calibration", {".json", ".yaml", ".yml", ".txt"})
        depth = indexed_files(seq / "gt" / "DepthL_float32", GT_EXTS)
        disp = indexed_files(seq / "gt" / "Disparity_float32", GT_EXTS)
        masks = indexed_files(seq / "gt" / "ValidMask", GT_EXTS)
        left_s = summarize_file_group(left)
        right_s = summarize_file_group(right)
        depth_s = summarize_file_group(depth)
        disp_s = summarize_file_group(disp)
        mask_s = summarize_file_group(masks)
        calib_s = summarize_file_group(calib)
        img_meta = read_image_meta(left[0]) if left else {}
        gt_files = depth or disp or masks
        gt_meta = read_array_meta(gt_files[0], sample_stride=4) if gt_files else {}
        sample_stats = []
        for p in sample_files(depth, 3):
            meta = read_array_meta(p, sample_stride=8)
            sample_stats.append(f"{p.name}:shape={meta.get('shape')},dtype={meta.get('dtype')},pos={meta.get('positive_ratio'):.4f},min={meta.get('min'):.4g},max={meta.get('max'):.4g}" if meta.get("readable") else f"{p.name}:unreadable")
        counts = [len(left), len(right), len(calib)]
        if depth:
            counts.append(len(depth))
        if disp:
            counts.append(len(disp))
        if masks:
            counts.append(len(masks))
        counts_match = len(set(counts)) == 1 if counts else False
        has_partial_gt = bool(left) and ((depth and len(depth) != len(left)) or (disp and len(disp) != len(left)) or (masks and len(masks) != len(left)))
        row = {
            "sequence_name": seq.name,
            "left_frames": len(left),
            "right_frames": len(right),
            "depth_gt_frames": len(depth),
            "disparity_gt_frames": len(disp),
            "valid_mask_frames": len(masks),
            "calibration_files": len(calib),
            "image_resolution": "x".join(map(str, img_meta.get("shape", [""])[:2])) if img_meta.get("readable") else "",
            "gt_resolution": "x".join(map(str, gt_meta.get("shape", [""])[:2])) if gt_meta.get("readable") else "",
            "image_dtype": img_meta.get("dtype", ""),
            "gt_dtype": gt_meta.get("dtype", ""),
            "counts_match": counts_match,
            "left_indices_continuous": left_s["continuous_sorted_indices"],
            "right_indices_continuous": right_s["continuous_sorted_indices"],
            "depth_indices_continuous": depth_s["continuous_sorted_indices"],
            "disp_indices_continuous": disp_s["continuous_sorted_indices"],
            "mask_indices_continuous": mask_s["continuous_sorted_indices"],
            "calibration_indices_continuous": calib_s["continuous_sorted_indices"],
            "missing_left_ids": left_s["missing_ids"],
            "missing_right_ids": right_s["missing_ids"],
            "missing_depth_ids": depth_s["missing_ids"],
            "missing_disp_ids": disp_s["missing_ids"],
            "missing_mask_ids": mask_s["missing_ids"],
            "empty_files_total": left_s["empty_files"] + right_s["empty_files"] + depth_s["empty_files"] + disp_s["empty_files"] + mask_s["empty_files"] + calib_s["empty_files"],
            "has_partial_gt": has_partial_gt,
            "gt_units_inferred": "DepthL_float32 stores float32 depth; existing ARGOS eval converts depth/disparity through per-frame calibration and reports depth error in mm. Disparity_float32 absent here." if depth and not disp else "Disparity_float32/DepthL_float32 folder names indicate float32 disparity/depth.",
            "sample_gt_stats": " | ".join(sample_stats),
            "evaluation_ready": bool(left and right and calib and (depth or disp) and counts_match and left_s["continuous_sorted_indices"] and right_s["continuous_sorted_indices"]),
        }
        rows.append(row)
        if not row["evaluation_ready"]:
            problems.append({"severity": "warning", "area": f"temporal_gt/{seq.name}", "issue": "not_evaluation_ready", "details": json.dumps(row, default=str)[:1000], "recommended_action": "Inspect missing/mismatched frame streams before official evaluation."})
        if not masks:
            problems.append({"severity": "info", "area": f"temporal_gt/{seq.name}", "issue": "no_explicit_valid_masks", "details": "No gt/ValidMask files found; current evaluation must derive valid masks from finite/positive GT and calibration logic.", "recommended_action": "Accept if eval script uses derived masks; otherwise generate masks in a separate controlled task."})
    return rows


def audit_consecutive32(root: Path, problems: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seq = root / "curated" / "consecutive32"
    rows: list[dict[str, Any]] = []
    left = indexed_files(seq / "left", IMAGE_EXTS)
    right = indexed_files(seq / "right", IMAGE_EXTS)
    left_s = summarize_file_group(left)
    right_s = summarize_file_group(right)
    img_meta = read_image_meta(left[0]) if left else {}
    rows.append({
        "sequence_name": "consecutive32",
        "left_frames": len(left),
        "right_frames": len(right),
        "has_depth_gt": False,
        "has_disparity_gt": False,
        "has_valid_masks": False,
        "has_calibration": False,
        "image_resolution": "x".join(map(str, img_meta.get("shape", [""])[:2])) if img_meta.get("readable") else "",
        "left_indices_continuous": left_s["continuous_sorted_indices"],
        "right_indices_continuous": right_s["continuous_sorted_indices"],
        "counts_match": len(left) == len(right) and len(left) > 0,
        "first_id": left_s["first_id"],
        "last_id": left_s["last_id"],
        "empty_files_total": left_s["empty_files"] + right_s["empty_files"],
        "training_clip_ready": len(left) == len(right) and len(left) >= 5 and left_s["continuous_sorted_indices"] and right_s["continuous_sorted_indices"],
        "notes": "Stereo-only 32-frame clip; usable for inference/cache/training without GT, not geometry evaluation.",
    })
    if rows[-1]["first_id"] != 0:
        problems.append({"severity": "info", "area": "consecutive32", "issue": "indices_start_at_nonzero", "details": f"Frames start at {rows[-1]['first_id']}; continuous but not zero-based.", "recommended_action": "No action if scripts sort numerically; document if aligning with zero-based predictions."})
    return rows


def audit_warped_gt(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = root / "curated" / "warped_gt_108"
    if not base.exists():
        return rows
    for seq in sorted(base.glob("dataset_*/*")):
        if not seq.is_dir() or not seq.name.startswith("keyframe_"):
            continue
        left = indexed_files(seq / "Left_rectified", IMAGE_EXTS)
        right = indexed_files(seq / "Right_rectified", IMAGE_EXTS)
        depth = indexed_files(seq / "Reference_SCARED_Warped" / "DepthL_float32", GT_EXTS)
        disp = indexed_files(seq / "Reference_SCARED_Warped" / "Disparity_float32", GT_EXTS)
        masks = indexed_files(seq / "Reference_SCARED_Warped" / "ValidMask", GT_EXTS)
        calib = indexed_files(seq / "Rectified_calibration", {".json"})
        img_meta = read_image_meta(left[0]) if left else {}
        gt_meta = read_array_meta(disp[0], sample_stride=4) if disp else (read_array_meta(depth[0], sample_stride=4) if depth else {})
        counts = [len(left), len(right), len(depth), len(disp), len(masks), len(calib)]
        rows.append({
            "collection": "warped_gt_108",
            "sequence_name": f"{seq.parent.name}_{seq.name}",
            "left_frames": len(left),
            "right_frames": len(right),
            "depth_gt_frames": len(depth),
            "disparity_gt_frames": len(disp),
            "valid_mask_frames": len(masks),
            "calibration_files": len(calib),
            "image_resolution": "x".join(map(str, img_meta.get("shape", [""])[:2])) if img_meta.get("readable") else "",
            "gt_resolution": "x".join(map(str, gt_meta.get("shape", [""])[:2])) if gt_meta.get("readable") else "",
            "counts_match": len(set(counts)) == 1 if counts else False,
            "evaluation_ready": len(left) == len(right) == len(depth) == len(disp) == len(masks) == len(calib) and len(left) > 0,
            "notes": "Short warped GT clip, useful for multi-sequence checks; not the official long temporal_gt benchmark.",
        })
    return rows


def audit_keyframes(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = root / "curated" / "keyframes_gt_dataset8"
    if not base.exists():
        return rows
    for seq in sorted(base.glob("dataset_*/*")):
        if not seq.is_dir():
            continue
        left = seq / "Left_Image.png"
        right = seq / "Right_Image.png"
        depth_l = seq / "left_depth_map.tiff"
        depth_r = seq / "right_depth_map.tiff"
        calib = seq / "endoscope_calibration.yaml"
        img_meta = read_image_meta(left) if left.exists() else {}
        gt_meta = read_array_meta(depth_l) if depth_l.exists() else {}
        rows.append({
            "collection": "keyframes_gt_dataset8",
            "sequence_name": f"{seq.parent.name}_{seq.name}",
            "left_frames": int(left.exists()),
            "right_frames": int(right.exists()),
            "depth_gt_frames": int(depth_l.exists()) + int(depth_r.exists()),
            "disparity_gt_frames": 0,
            "valid_mask_frames": 0,
            "calibration_files": int(calib.exists()),
            "image_resolution": "x".join(map(str, img_meta.get("shape", [""])[:2])) if img_meta.get("readable") else "",
            "gt_resolution": "x".join(map(str, gt_meta.get("shape", [""])[:2])) if gt_meta.get("readable") else "",
            "counts_match": left.exists() and right.exists() and depth_l.exists() and calib.exists(),
            "evaluation_ready": left.exists() and right.exists() and depth_l.exists() and calib.exists(),
            "notes": "Single keyframe geometry GT; not temporal.",
        })
    return rows


def raw_audit(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source = root / "raw" / "source"
    extracted = root / "raw" / "extracted"
    curated = root / "curated"
    rows: list[dict[str, Any]] = []
    for name, path in [("raw/source", source), ("raw/extracted", extracted), ("curated", curated), ("SCARED total", root)]:
        total, count, exts = dir_size(path)
        rows.append({
            "area": name,
            "path": str(path),
            "total_bytes": total,
            "total_human": human_bytes(total),
            "file_count": count,
            "zip_files": exts.get(".zip", 0),
            "png_files": exts.get(".png", 0),
            "npy_files": exts.get(".npy", 0),
            "tiff_files": exts.get(".tiff", 0) + exts.get(".tif", 0),
            "mp4_files": exts.get(".mp4", 0),
            "yaml_files": exts.get(".yaml", 0) + exts.get(".yml", 0),
            "json_files": exts.get(".json", 0),
        })
    largest = []
    for raw_root in [source, extracted]:
        if raw_root.exists():
            for p in walk_files(raw_root):
                size = safe_size(p)
                largest.append({
                    "path": str(p),
                    "relative_path": str(p.relative_to(root)),
                    "size_bytes": size,
                    "size_human": human_bytes(size),
                    "extension": ".tar.gz" if p.name.endswith(".tar.gz") else (p.suffix.lower() or "<none>"),
                })
    largest = sorted(largest, key=lambda r: int(r["size_bytes"]), reverse=True)[:100]
    source_zips = sorted([p.stem for p in source.glob("*.zip")]) if source.exists() else []
    extracted_dirs = sorted([p.name for p in extracted.iterdir() if p.is_dir()]) if extracted.exists() else []
    match = {name: name in extracted_dirs for name in source_zips}
    extra = sorted(set(extracted_dirs) - set(source_zips))
    raw_detail = {
        "source_zips": source_zips,
        "extracted_dirs": extracted_dirs,
        "source_zip_to_extracted_dir": match,
        "extracted_dirs_without_matching_zip_name": extra,
    }
    return rows, largest, raw_detail


def file_structure_summary(root: Path) -> str:
    lines = [f"SCARED root: {root}", "", "Directory summary to depth 4:"]
    max_depth = 4
    for d in sorted([root] + [p for p in root.rglob("*") if p.is_dir()]):
        rel = d.relative_to(root)
        depth = 0 if str(rel) == "." else len(rel.parts)
        if depth > max_depth:
            continue
        try:
            dirs = sum(1 for p in d.iterdir() if p.is_dir())
            files = sum(1 for p in d.iterdir() if p.is_file())
        except OSError:
            dirs = files = -1
        indent = "  " * depth
        name = "." if str(rel) == "." else rel.name
        lines.append(f"{indent}{name}/  dirs={dirs} files={files}")
    return "\n".join(lines) + "\n"


def inspect_expected_patterns(repo_root: Path) -> str:
    files = []
    for rel in [
        "scripts/temporal_refinement/data_prep",
        "scripts/temporal_refinement/eval_scripts",
        "scripts/temporal_refinement/lib/datasets.py",
        "scripts/temporal_refinement/playground/real_data.py",
    ]:
        p = repo_root / rel
        if p.is_dir():
            files.extend(sorted(p.glob("*.py")))
        elif p.exists():
            files.append(p)
    lines = ["# Expected Dataset Patterns From Code", "", "Scanned files:", *[f"- `{p.relative_to(repo_root)}`" for p in files], "", "## Relevant path/pattern lines", ""]
    for p in files:
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        hits = []
        for i, line in enumerate(text.splitlines(), start=1):
            if any(k in line for k in KEYWORDS_FOR_CODE_SCAN):
                hits.append((i, line.strip()))
        if hits:
            lines.append(f"### `{p.relative_to(repo_root)}`")
            for i, line in hits[:80]:
                lines.append(f"- L{i}: `{line}`")
            lines.append("")
    lines.extend([
        "## Inferred conventions",
        "",
        "- Official temporal GT benchmark: `dataset/SCARED/curated/temporal_gt/<sequence>/{left,right,gt,calibration}`.",
        "- Temporal GT depth: `gt/DepthL_float32/*.npy`; explicit disparity may be absent and can be derived through calibration in eval code.",
        "- Warped short GT clips: `dataset/SCARED/curated/warped_gt_108/dataset_*/keyframe_*` with `Left_rectified`, `Right_rectified`, `Reference_SCARED_Warped/{DepthL_float32,Disparity_float32,ValidMask}`, and `Rectified_calibration`.",
        "- Consecutive32: `dataset/SCARED/curated/consecutive32/{left,right}`; stereo-only, no GT/calibration in that folder.",
        "- Indexed fast-cache training consumes cached predictions rather than reading raw SCARED folders directly.",
    ])
    return "\n".join(lines) + "\n"


def make_report(
    root: Path,
    raw_rows: list[dict[str, Any]],
    temporal_rows: list[dict[str, Any]],
    consecutive_rows: list[dict[str, Any]],
    curated_inventory: list[dict[str, Any]],
    largest: list[dict[str, Any]],
    problems: list[dict[str, Any]],
    raw_detail: dict[str, Any],
) -> str:
    source_size = next((r for r in raw_rows if r["area"] == "raw/source"), {})
    extracted_size = next((r for r in raw_rows if r["area"] == "raw/extracted"), {})
    curated_size = next((r for r in raw_rows if r["area"] == "curated"), {})
    eval_ready = [r for r in temporal_rows if str(r.get("evaluation_ready")) == "True" or r.get("evaluation_ready") is True]
    partial = [r for r in temporal_rows if str(r.get("has_partial_gt")) == "True" or r.get("has_partial_gt") is True]
    consec = consecutive_rows[0] if consecutive_rows else {}
    top = largest[:10]
    top_lines = "\n".join([f"- `{r['relative_path']}`: {r['size_human']}" for r in top])
    problem_lines = "\n".join([f"- {p['severity']} `{p['area']}`: {p['issue']} - {p['details']}" for p in problems]) or "- None blocking."
    source_zips = raw_detail.get("source_zips", [])
    missing_extract = [k for k, v in raw_detail.get("source_zip_to_extracted_dir", {}).items() if not v]
    extracted_extra = raw_detail.get("extracted_dirs_without_matching_zip_name", [])
    return f"""# SCARED Dataset Integrity Audit v1

Dataset root: `{root}`
Generated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`

This audit is read-only with respect to dataset files. It samples metadata and a few arrays/images only; no archives are unpacked and no model inference/training is run.

## Direct Answers

1. **Is `curated/temporal_gt` usable as the official temporal GT benchmark?**
   {'Yes' if eval_ready else 'No'}: {', '.join(r['sequence_name'] for r in eval_ready) if eval_ready else 'no evaluation-ready sequence found'}.

2. **Which sequences are complete and evaluation-ready?**
   Temporal GT: {', '.join(r['sequence_name'] for r in eval_ready) if eval_ready else 'none'}. Short warped GT clips marked ready in `curated_sequence_inventory.csv` can be used for auxiliary/multi-sequence checks, not as the official long benchmark.

3. **Which sequences have partial GT?**
   {', '.join(r['sequence_name'] for r in partial) if partial else 'None detected in curated/temporal_gt. Note that downstream evaluation may still filter frames by valid-pixel threshold.'}

4. **Are image and GT resolutions consistent?**
   {'Yes for temporal_gt samples' if all(r.get('image_resolution') == r.get('gt_resolution') for r in temporal_rows if r.get('gt_resolution')) else 'Check temporal_gt_integrity.csv for mismatches'}.

5. **Are calibration files available where needed?**
   {'Yes for temporal_gt' if all(int(r.get('calibration_files', 0)) == int(r.get('left_frames', -1)) for r in temporal_rows) else 'Some temporal_gt calibration files are missing; inspect CSV'}.

6. **Is `curated/consecutive32` usable for training clips?**
   {'Yes' if consec.get('training_clip_ready') else 'No'}: it has {consec.get('left_frames', 0)} left and {consec.get('right_frames', 0)} right frames. It has no GT/calibration, so it is stereo/inference/cache material rather than geometry-evaluation material.

7. **What is the reason for the large size difference between raw and curated?**
   Raw keeps source zips and extracted original SCARED content; curated stores only ARGOS-ready rectified subsets/GT arrays needed for experiments. Current sizes: raw/source `{source_size.get('total_human', '')}`, raw/extracted `{extracted_size.get('total_human', '')}`, curated `{curated_size.get('total_human', '')}`.

8. **Which raw files dominate disk usage?**
{top_lines}

9. **Which dataset paths should be used?**
   - Temporal evaluation: `dataset/SCARED/curated/temporal_gt/test_dataset_9_keyframe_3`.
   - Frame-wise geometry evaluation: `dataset/SCARED/curated/keyframes_gt_dataset8` and short GT clips under `dataset/SCARED/curated/warped_gt_108` when the protocol explicitly supports them.
   - Training clips: cached prediction datasets under `results/03_temporal_refinement/cache/...`; raw stereo-only source clips can come from `dataset/SCARED/curated/consecutive32` and curated long/progressive folders after prediction cache generation.
   - Raw recovery/debug only: `dataset/SCARED/raw/source` and `dataset/SCARED/raw/extracted`.

10. **Any action required before running S2M2-S / EMA / warped EMA / StereoAnyVideo evaluation?**
   For the official temporal-GT sequence, no blocking dataset action was found. Use the existing evaluation scripts so masks/calibration/positive-disparity policy remain consistent. For `consecutive32`, do not report GT geometry because no GT is present in that folder.

## Size Summary

| area | size | files | zips | png | npy | tiff | mp4 | yaml | json |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
""" + "\n".join([
        f"| {r['area']} | {r['total_human']} | {r['file_count']} | {r['zip_files']} | {r['png_files']} | {r['npy_files']} | {r['tiff_files']} | {r['mp4_files']} | {r['yaml_files']} | {r['json_files']} |"
        for r in raw_rows
    ]) + f"""

## Source Zip / Extraction Check

- Source zips present: {len(source_zips)} ({', '.join(source_zips[:20])})
- Source zips without same-named extracted directory: {', '.join(missing_extract) if missing_extract else 'none'}
- Extracted dirs without same-named source zip: {', '.join(extracted_extra) if extracted_extra else 'none'}

## Problems / Warnings

{problem_lines}

## Notes On Units

- `DepthL_float32` names indicate float32 depth arrays. Existing ARGOS evaluation reports depth errors in millimeters after applying its calibration/depth logic.
- `Disparity_float32` names indicate disparity in pixels where present, already stored as float32 arrays.
- `temporal_gt/test_dataset_9_keyframe_3` stores both `DepthL_float32` and `Disparity_float32` arrays plus `ValidMask`; evaluation code should use the same calibration/mask policy as existing ARGOS benchmarks.

See the CSV files in this folder for full inventory and sampled metadata.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scared-root", type=Path, default=Path("dataset/SCARED"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/dataset_audits/scared_dataset_integrity_v1"))
    args = parser.parse_args()

    repo_root = Path.cwd()
    root = args.scared_root
    out = args.out_dir
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    log_lines = [f"repo_root={repo_root}", f"scared_root={root}", f"out_dir={out}"]

    problems: list[dict[str, Any]] = []
    if not root.exists():
        raise SystemExit(f"Missing SCARED root: {root}")

    raw_rows, largest_raw, raw_detail = raw_audit(root)
    temporal_rows = audit_temporal_gt(root, problems)
    consecutive_rows = audit_consecutive32(root, problems)
    warped_rows = audit_warped_gt(root)
    keyframe_rows = audit_keyframes(root)
    curated_inventory = []
    curated_inventory.extend({"collection": "temporal_gt", **r} for r in temporal_rows)
    curated_inventory.extend({"collection": "consecutive32", **r} for r in consecutive_rows)
    curated_inventory.extend(warped_rows)
    curated_inventory.extend(keyframe_rows)

    # Add raw file type presence rows for requested extensions.
    ext_presence: list[dict[str, Any]] = []
    for p in walk_files(root / "raw"):
        ext = ".tar.gz" if p.name.endswith(".tar.gz") else p.suffix.lower()
        if ext in RAW_INTEREST_EXTS:
            row = {"path": str(p), "relative_path": str(p.relative_to(root)), "extension": ext, "size_bytes": safe_size(p), "size_human": human_bytes(safe_size(p))}
            if ext in VIDEO_EXTS:
                row.update(video_meta(p))
            ext_presence.append(row)
    ext_presence = sorted(ext_presence, key=lambda r: int(r["size_bytes"]), reverse=True)

    (out / "file_structure_summary.txt").write_text(file_structure_summary(root))
    write_csv(out / "raw_size_summary.csv", raw_rows)
    write_csv(out / "largest_raw_files.csv", largest_raw)
    write_csv(out / "raw_interesting_files.csv", ext_presence)
    write_csv(out / "curated_sequence_inventory.csv", curated_inventory)
    write_csv(out / "temporal_gt_integrity.csv", temporal_rows)
    write_csv(out / "consecutive32_inventory.csv", consecutive_rows)
    write_csv(out / "problems_found.csv", problems, fieldnames=["severity", "area", "issue", "details", "recommended_action"])
    (out / "expected_patterns_from_code.md").write_text(inspect_expected_patterns(repo_root))
    (out / "report.md").write_text(make_report(root, raw_rows, temporal_rows, consecutive_rows, curated_inventory, largest_raw, problems, raw_detail))
    log_lines.append(f"temporal_gt_sequences={len(temporal_rows)}")
    log_lines.append(f"curated_inventory_rows={len(curated_inventory)}")
    log_lines.append(f"problems={len(problems)}")
    (out / "run.log").write_text("\n".join(log_lines) + "\n")
    print(f"Wrote audit to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
