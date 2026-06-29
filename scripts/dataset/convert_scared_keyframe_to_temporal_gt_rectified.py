#!/usr/bin/env python3
"""Convert one raw SCARED keyframe video block into rectified temporal GT.

This converter follows the rectification path used by the older ARGOS SCARED
scripts: split the vertical stereo video into left/right views, use OpenCV
stereoRectify/initUndistortRectifyMap for images, rotate reference-view scene
points by R1, and reproject them through P1/P2 with scatter_min_depth.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import shutil
import tarfile
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.scared.convert_scared_keyframes import scatter_min_depth


DEFAULT_KEYFRAME_PATH = Path("dataset/SCARED/raw/extracted/dataset_1/dataset_1/keyframe_1")
DEFAULT_OUTPUT_DIR = Path("dataset/SCARED/curated/temporal_gt_rectified/dataset_1_keyframe_1")


@dataclass(frozen=True)
class SourceCounts:
    rgb: int
    frame_data: int
    scene_points: int


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert one raw SCARED keyframe video block into rectified temporal GT.")
    parser.add_argument("--keyframe-path", type=Path, default=DEFAULT_KEYFRAME_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--view-layout", choices=["vertical_stereo_stack"], default="vertical_stereo_stack")
    parser.add_argument("--reference-view", choices=["top"], default="top")
    parser.add_argument("--max-frames", type=int, default=0, help="0 processes all frames.")
    parser.add_argument("--overwrite", nargs="?", const=True, default=False, type=parse_bool)
    parser.add_argument("--allow-partial", nargs="?", const=True, default=False, type=parse_bool)
    parser.add_argument("--save-debug", nargs="?", const=True, default=True, type=parse_bool)
    parser.add_argument(
        "--defer-scene-count",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool,
        help="Skip the expensive pre-count of scene_points.tar.gz; intended for trusted batch conversion.",
    )
    return parser.parse_args()


def basename(name: str) -> str:
    return Path(name).name


def parse_frame_id(name: str, prefix: str, suffix: str) -> int | None:
    base = basename(name)
    if not (base.startswith(prefix) and base.endswith(suffix)):
        return None
    return int(base[len(prefix) : -len(suffix)])


def count_tar_members(path: Path, suffix: str) -> int:
    count = 0
    with tarfile.open(path, "r|gz") as tf:
        for member in tf:
            if member.isfile() and basename(member.name).endswith(suffix):
                count += 1
    return count


def video_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def read_frame_data(frame_tar: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    with tarfile.open(frame_tar, "r:gz") as tf:
        for member in tf.getmembers():
            fid = parse_frame_id(member.name, "frame_data", ".json")
            if fid is None or not member.isfile():
                continue
            f = tf.extractfile(member)
            if f is not None:
                out[fid] = json.loads(f.read().decode("utf-8"))
    return out


def iter_scene_points(scene_tar: Path, max_frames: int) -> Iterable[tuple[int, np.ndarray]]:
    try:
        import tifffile
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("tifffile is required to read float32 scene_points TIFF files") from exc
    with tarfile.open(scene_tar, "r|gz") as tf:
        yielded = 0
        for member in tf:
            fid = parse_frame_id(member.name, "scene_points", ".tiff")
            if fid is None or not member.isfile():
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            yield fid, np.asarray(tifffile.imread(io.BytesIO(f.read())), dtype=np.float32)
            yielded += 1
            if max_frames > 0 and yielded >= max_frames:
                break


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output directory already exists: {path}. Pass --overwrite true to replace it.")
    if path.exists() and overwrite:
        shutil.rmtree(path)
    for rel in ["left", "right", "gt/DepthL_float32", "gt/Disparity_float32", "gt/ValidMask", "calibration", "debug"]:
        (path / rel).mkdir(parents=True, exist_ok=True)


def split_vertical_stereo(rgb_bgr: np.ndarray, scene_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rgb_bgr.shape[0] % 2 != 0 or scene_points.shape[0] % 2 != 0:
        raise RuntimeError(f"Expected even vertical stack heights, got rgb={rgb_bgr.shape}, scene={scene_points.shape}")
    rgb_mid = rgb_bgr.shape[0] // 2
    scene_mid = scene_points.shape[0] // 2
    left = rgb_bgr[:rgb_mid].copy()
    right = rgb_bgr[rgb_mid : rgb_mid * 2].copy()
    scene_ref = scene_points[:scene_mid].copy()
    if left.shape[:2] != scene_ref.shape[:2]:
        raise RuntimeError(f"Left/image and reference scene_points shapes differ: left={left.shape}, scene_ref={scene_ref.shape}")
    return left, right, scene_ref


def calibration_from_frame_data(frame_data: dict[str, Any], image_size: tuple[int, int]) -> dict[str, Any]:
    raw = frame_data["camera-calibration"]
    kl = np.asarray(raw["KL"], dtype=np.float64)
    kr = np.asarray(raw["KR"], dtype=np.float64)
    dl = np.asarray(raw["DL"], dtype=np.float64).reshape(-1, 1)
    dr = np.asarray(raw["DR"], dtype=np.float64).reshape(-1, 1)
    r = np.asarray(raw["R"], dtype=np.float64)
    t = np.asarray(raw["T"], dtype=np.float64).reshape(3, 1)
    r1, r2, p1, p2, q, roi1, roi2 = cv2.stereoRectify(
        kl,
        dl,
        kr,
        dr,
        image_size,
        r,
        t,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )
    map1x, map1y = cv2.initUndistortRectifyMap(kl, dl, r1, p1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(kr, dr, r2, p2, image_size, cv2.CV_32FC1)
    baseline = abs(float(p2[0, 3] / p2[0, 0]))
    return {
        "raw": raw,
        "R1": r1,
        "R2": r2,
        "P1": p1,
        "P2": p2,
        "Q": q,
        "roi1": roi1,
        "roi2": roi2,
        "maps": (map1x, map1y, map2x, map2y),
        "fx": float(p1[0, 0]),
        "fy": float(p1[1, 1]),
        "cx": float(p1[0, 2]),
        "cy": float(p1[1, 2]),
        "baseline": baseline,
        "camera_pose": frame_data.get("camera-pose"),
        "timestamp": frame_data.get("timestamp"),
    }


def mat_payload(mat: np.ndarray) -> dict[str, Any]:
    return {"rows": int(mat.shape[0]), "cols": int(mat.shape[1]), "data": mat.astype(float).reshape(-1).tolist()}


def write_calibration(path: Path, cal: dict[str, Any], frame_idx: int, view_layout: str, reference_view: str) -> None:
    raw = cal["raw"]
    payload = {
        "P1": mat_payload(cal["P1"]),
        "P2": mat_payload(cal["P2"]),
        "R1": mat_payload(cal["R1"]),
        "R2": mat_payload(cal["R2"]),
        "Q": mat_payload(cal["Q"]),
        "fx": cal["fx"],
        "fy": cal["fy"],
        "cx": cal["cx"],
        "cy": cal["cy"],
        "baseline": cal["baseline"],
        "baseline_mm": cal["baseline"],
        "KL": raw["KL"],
        "KR": raw["KR"],
        "DL": raw["DL"],
        "DR": raw["DR"],
        "R": raw["R"],
        "T": raw["T"],
        "camera_pose": cal["camera_pose"],
        "timestamp": cal["timestamp"],
        "source_frame_index": frame_idx,
        "view_layout": view_layout,
        "reference_view": reference_view,
        "rectification": "cv2.stereoRectify(flags=CALIB_ZERO_DISPARITY, alpha=0)",
        "gt_projection": "scene_points_ref @ R1.T then scatter_min_depth(P1, P2)",
        "depth_unit_assumption": "mm",
        "disparity_formula": "fx * baseline / rectified_depth",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def save_float_gt(ref_dir: Path, stem: str, depth: np.ndarray, disp: np.ndarray) -> tuple[np.ndarray, dict[str, Path]]:
    valid = (depth > 0) & (disp > 0) & np.isfinite(depth) & np.isfinite(disp)
    paths = {
        "depth": ref_dir / "DepthL_float32" / f"{stem}.npy",
        "disp": ref_dir / "Disparity_float32" / f"{stem}.npy",
        "mask": ref_dir / "ValidMask" / f"{stem}.npy",
    }
    for key, array in [("depth", depth.astype(np.float32)), ("disp", disp.astype(np.float32)), ("mask", valid)]:
        np.save(paths[key], array)
    return valid, paths


def finite_stats(values: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    vals = values[valid & np.isfinite(values) & (values > 0)]
    if vals.size == 0:
        return {"min": math.nan, "median": math.nan, "max": math.nan}
    return {"min": float(vals.min()), "median": float(np.median(vals)), "max": float(vals.max())}


def scalar_preview(values: np.ndarray, valid: np.ndarray | None = None, vmax: float | None = None, cmap: int = cv2.COLORMAP_TURBO) -> np.ndarray:
    arr = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mask = np.isfinite(values)
    if valid is not None:
        mask &= valid
    if vmax is None:
        vmax = float(np.percentile(arr[mask], 99)) if mask.any() else 1.0
    norm = np.clip(arr / max(vmax, 1e-6), 0.0, 1.0)
    out = cv2.applyColorMap((norm * 255).astype(np.uint8), cmap)
    if valid is not None:
        out[~valid] = 0
    return out


def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = image_bgr.copy()
    green = np.zeros_like(out)
    green[..., 1] = 255
    out[mask] = cv2.addWeighted(out[mask], 0.55, green[mask], 0.45, 0)
    return out


def draw_epipolar_guides(left: np.ndarray, right: np.ndarray, label: str) -> np.ndarray:
    left_vis = left.copy()
    right_vis = right.copy()
    h, w = left.shape[:2]
    for y in np.linspace(80, h - 80, 8).astype(int):
        color = (0, 255, 255)
        cv2.line(left_vis, (0, y), (w - 1, y), color, 1, cv2.LINE_AA)
        cv2.line(right_vis, (0, y), (w - 1, y), color, 1, cv2.LINE_AA)
    combo = np.concatenate([left_vis, right_vis], axis=1)
    cv2.rectangle(combo, (0, 0), (combo.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(combo, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return combo


def label_tile(tile: np.ndarray, label: str, size: tuple[int, int] = (256, 205)) -> np.ndarray:
    resized = cv2.resize(tile, size, interpolation=cv2.INTER_AREA)
    cv2.rectangle(resized, (0, 0), (resized.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(resized, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return resized


def make_contact_sheet(samples: list[dict[str, Any]], out_path: Path) -> None:
    rows = []
    for sample in samples:
        panels = [
            label_tile(sample["raw_left"], f"{sample['stem']} raw left"),
            label_tile(sample["raw_right"], "raw right"),
            label_tile(sample["rect_left"], "rect left"),
            label_tile(sample["rect_right"], "rect right"),
            label_tile(overlay_mask(sample["rect_left"], sample["valid"]), "valid overlay"),
            label_tile(scalar_preview(sample["disp"], sample["valid"]), "rect disparity"),
        ]
        row1 = np.concatenate(panels, axis=1)
        before = cv2.resize(draw_epipolar_guides(sample["raw_left"], sample["raw_right"], "epipolar guides before rectification"), (row1.shape[1], 205), interpolation=cv2.INTER_AREA)
        after = cv2.resize(draw_epipolar_guides(sample["rect_left"], sample["rect_right"], "epipolar guides after rectification"), (row1.shape[1], 205), interpolation=cv2.INTER_AREA)
        rows.extend([row1, before, after])
    cv2.imwrite(str(out_path), np.concatenate(rows, axis=0))


def validate_output(output_dir: Path, rows: list[dict[str, Any]], image_hw: tuple[int, int]) -> dict[str, Any]:
    expected = len(rows)
    counts = {
        "left": len(list((output_dir / "left").glob("*.png"))),
        "right": len(list((output_dir / "right").glob("*.png"))),
        "depth": len(list((output_dir / "gt" / "DepthL_float32").glob("*.npy"))),
        "disparity": len(list((output_dir / "gt" / "Disparity_float32").glob("*.npy"))),
        "valid_mask": len(list((output_dir / "gt" / "ValidMask").glob("*.npy"))),
        "calibration": len(list((output_dir / "calibration").glob("*.json"))),
    }
    errors: list[str] = []
    if any(count != expected for count in counts.values()):
        errors.append(f"file_count_mismatch:{counts}, expected={expected}")
    h, w = image_hw
    for row in rows:
        left = cv2.imread(row["left_path"], cv2.IMREAD_COLOR)
        right = cv2.imread(row["right_path"], cv2.IMREAD_COLOR)
        depth = np.load(row["depth_float32_path"])
        disp = np.load(row["disparity_float32_path"])
        valid = np.load(row["valid_mask_path"]).astype(bool)
        if left is None or right is None:
            errors.append(f"missing_image:{row['frame_id']}")
            continue
        if left.shape[:2] != (h, w) or right.shape[:2] != (h, w):
            errors.append(f"image_shape_mismatch:{row['frame_id']}")
        if depth.shape != (h, w) or disp.shape != (h, w) or valid.shape != (h, w):
            errors.append(f"gt_shape_mismatch:{row['frame_id']}")
        if np.any(depth[valid] <= 0):
            errors.append(f"nonpositive_valid_depth:{row['frame_id']}")
        if not np.isfinite(disp[valid]).all():
            errors.append(f"nonfinite_valid_disparity:{row['frame_id']}")
    return {"counts": counts, "expected_count": expected, "is_valid": not errors, "errors": errors}


def write_metadata(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "sequence_id",
        "frame_id",
        "left_path",
        "right_path",
        "depth_float32_path",
        "disparity_float32_path",
        "valid_mask_path",
        "calibration_path",
        "valid_pixel_ratio",
        "timestamp",
        "baseline",
        "fx",
        "depth_median",
        "disparity_median",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in columns} for row in rows])


def write_summary_readme(output_dir: Path, summary: dict[str, Any]) -> None:
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Rectified SCARED Temporal-GT Sequence",
        "",
        f"Source keyframe: `{summary['source_keyframe_path']}`",
        f"Frames processed: `{summary['num_processed_frames']}`",
        f"Output size: `{summary['output_image_width']}x{summary['output_image_height']}`",
        f"Rectification: `{summary['rectification']}`",
        f"GT projection: `{summary['gt_projection']}`",
        f"Mean valid pixels: `{summary['valid_pixel_ratio_mean']:.6f}`",
        f"Mean median depth: `{summary['depth_median_mean']:.6f}`",
        f"Mean median disparity: `{summary['disparity_median_mean']:.6f}`",
        f"Complete: `{summary['is_complete']}`",
        f"Validation passed: `{summary['validation']['is_valid']}`",
        "",
        "This converter does not run S2M2 or any temporal-refinement benchmark.",
        "",
        "## Validation",
        "",
        f"Counts: `{summary['validation']['counts']}`",
        "",
        f"Errors: `{summary['validation']['errors']}`",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if args.max_frames < 0:
        raise ValueError("--max-frames must be >= 0")
    keyframe = args.keyframe_path
    frame_tar = keyframe / "data" / "frame_data.tar.gz"
    scene_tar = keyframe / "data" / "scene_points.tar.gz"
    video_path = keyframe / "data" / "rgb.mp4"
    for path in [frame_tar, scene_tar, video_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    frame_data = read_frame_data(frame_tar)
    scene_count = -1 if args.defer_scene_count else count_tar_members(scene_tar, ".tiff")
    counts = SourceCounts(rgb=video_frame_count(video_path), frame_data=len(frame_data), scene_points=scene_count)
    count_values = {counts.rgb, counts.frame_data} if args.defer_scene_count else {counts.rgb, counts.frame_data, counts.scene_points}
    if len(count_values) != 1 and not args.allow_partial:
        raise RuntimeError(f"Source frame counts mismatch: {counts}. Pass --allow-partial true to convert common prefix.")
    process_count = min(counts.rgb, counts.frame_data) if args.defer_scene_count else min(counts.rgb, counts.frame_data, counts.scene_points)
    if args.max_frames > 0:
        process_count = min(process_count, args.max_frames)
    ensure_output_dir(args.output_dir, bool(args.overwrite))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    sequence_id = args.output_dir.name
    rows: list[dict[str, Any]] = []
    valid_ratios: list[float] = []
    depth_medians: list[float] = []
    disp_medians: list[float] = []
    baselines: list[float] = []
    fxs: list[float] = []
    debug_samples: list[dict[str, Any]] = []
    debug_ids = {0, process_count // 2, max(process_count - 1, 0)}
    image_hw: tuple[int, int] | None = None
    next_video_idx = 0

    for scene_idx, scene_points in iter_scene_points(scene_tar, process_count):
        if scene_idx >= process_count:
            break
        if scene_idx not in frame_data:
            if args.allow_partial:
                continue
            raise RuntimeError(f"Missing frame_data for frame {scene_idx}")
        # Scene-point members are normally contiguous and sorted. Reading the
        # video stream forward avoids an expensive random seek for every frame.
        while next_video_idx <= scene_idx:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            next_video_idx += 1
        if not ok:
            if args.allow_partial:
                break
            raise RuntimeError(f"Could not read RGB frame {scene_idx}")
        raw_left, raw_right, scene_ref = split_vertical_stereo(frame_bgr, scene_points)
        h, w = raw_left.shape[:2]
        if image_hw is None:
            image_hw = (h, w)
        cal = calibration_from_frame_data(frame_data[scene_idx], (w, h))
        map1x, map1y, map2x, map2y = cal["maps"]
        rect_left = cv2.remap(raw_left, map1x, map1y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        rect_right = cv2.remap(raw_right, map2x, map2y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        invalid_xyz = np.isclose(scene_ref, 0.0).all(axis=2) | ~np.isfinite(scene_ref).all(axis=2) | (scene_ref[..., 2] <= 0)
        points_ref = scene_ref.copy()
        points_ref[invalid_xyz] = 0.0
        points_rect = points_ref @ cal["R1"].T
        points_rect[invalid_xyz] = 0.0
        depth, disp = scatter_min_depth(points_rect, cal["P1"], cal["P2"], (h, w))

        stem = f"{scene_idx:06d}"
        left_path = args.output_dir / "left" / f"{stem}.png"
        right_path = args.output_dir / "right" / f"{stem}.png"
        cv2.imwrite(str(left_path), rect_left)
        cv2.imwrite(str(right_path), rect_right)
        valid, gt_paths = save_float_gt(args.output_dir / "gt", stem, depth, disp)
        calib_path = args.output_dir / "calibration" / f"{stem}.json"
        write_calibration(calib_path, cal, scene_idx, args.view_layout, args.reference_view)

        depth_stats = finite_stats(depth, valid)
        disp_stats = finite_stats(disp, valid)
        valid_ratio = float(valid.mean())
        valid_ratios.append(valid_ratio)
        depth_medians.append(depth_stats["median"])
        disp_medians.append(disp_stats["median"])
        baselines.append(cal["baseline"])
        fxs.append(cal["fx"])
        rows.append(
            {
                "sequence_id": sequence_id,
                "frame_id": stem,
                "left_path": str(left_path),
                "right_path": str(right_path),
                "depth_float32_path": str(gt_paths["depth"]),
                "disparity_float32_path": str(gt_paths["disp"]),
                "valid_mask_path": str(gt_paths["mask"]),
                "calibration_path": str(calib_path),
                "valid_pixel_ratio": valid_ratio,
                "timestamp": cal["timestamp"],
                "baseline": cal["baseline"],
                "fx": cal["fx"],
                "depth_median": depth_stats["median"],
                "disparity_median": disp_stats["median"],
            }
        )
        if args.save_debug and scene_idx in debug_ids:
            debug_samples.append(
                {
                    "stem": stem,
                    "raw_left": raw_left,
                    "raw_right": raw_right,
                    "rect_left": rect_left,
                    "rect_right": rect_right,
                    "valid": valid,
                    "disp": disp,
                }
            )
        print(f"{sequence_id}/{stem} valid={valid_ratio:.3f}", flush=True)

    cap.release()
    if not rows or image_hw is None:
        raise RuntimeError("No frames were converted")
    if args.defer_scene_count and len(rows) != process_count and not args.allow_partial:
        raise RuntimeError(f"Converted {len(rows)} scene-point frames but expected {process_count}.")
    write_metadata(args.output_dir / "metadata.csv", rows)
    if args.save_debug and debug_samples:
        make_contact_sheet(debug_samples, args.output_dir / "debug" / "rectification_contact_sheet.png")
    validation = validate_output(args.output_dir, rows, image_hw)
    warnings: list[str] = []
    if counts.rgb != counts.frame_data or counts.rgb != counts.scene_points:
        warnings.append(f"source_count_mismatch:{counts}")
    if args.defer_scene_count:
        warnings = [w for w in warnings if not w.startswith("source_count_mismatch")]
        warnings.append("scene_points_count_deferred")
    summary = {
        "sequence_id": sequence_id,
        "source_keyframe_path": str(keyframe),
        "num_rgb_frames": counts.rgb,
        "num_frame_data_json": counts.frame_data,
        "num_scene_points_tiff": len(rows) if args.defer_scene_count else counts.scene_points,
        "num_processed_frames": len(rows),
        "output_image_height": image_hw[0],
        "output_image_width": image_hw[1],
        "view_layout": args.view_layout,
        "reference_view": args.reference_view,
        "rectification": "cv2.stereoRectify + cv2.initUndistortRectifyMap, CALIB_ZERO_DISPARITY, alpha=0",
        "gt_projection": "top/reference scene_points rotated by R1 and projected through P1/P2 with scatter_min_depth",
        "metadata_convention": "matches test_dataset_9_keyframe_3: frame_id, DepthL_float32, Disparity_float32, ValidMask",
        "baseline_mean": float(np.mean(baselines)),
        "baseline_std": float(np.std(baselines)),
        "fx_mean": float(np.mean(fxs)),
        "fx_std": float(np.std(fxs)),
        "valid_pixel_ratio_mean": float(np.mean(valid_ratios)),
        "depth_median_mean": float(np.nanmean(depth_medians)),
        "disparity_median_mean": float(np.nanmean(disp_medians)),
        "is_complete": bool(len(rows) == counts.rgb == counts.frame_data == counts.scene_points),
        "warnings": warnings,
        "validation": validation,
    }
    write_summary_readme(args.output_dir, summary)
    run_log = [
        "SCARED raw keyframe to rectified temporal-GT conversion",
        f"keyframe_path={keyframe}",
        f"output_dir={args.output_dir}",
        f"source_counts={counts}",
        f"processed_frames={len(rows)}",
        f"validation_is_valid={validation['is_valid']}",
        f"warnings={warnings}",
    ]
    (args.output_dir / "run.log").write_text("\n".join(run_log) + "\n")
    print(json.dumps({"output_dir": str(args.output_dir), "processed_frames": len(rows), "validation": validation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
