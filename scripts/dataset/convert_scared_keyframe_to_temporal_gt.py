#!/usr/bin/env python3
"""Convert one raw SCARED keyframe into a curated temporal-GT sequence.

The converter is intentionally conservative: it handles one keyframe folder,
assumes the probed vertical stereo stack layout by default, writes a clean
curated sequence, and validates the result. It never modifies raw inputs.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


DEFAULT_KEYFRAME_PATH = Path("dataset/SCARED/raw/extracted/dataset_1/dataset_1/keyframe_1")
DEFAULT_OUTPUT_DIR = Path("dataset/SCARED/curated/temporal_gt/dataset_1_keyframe_1")


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
    parser = argparse.ArgumentParser(description="Convert one raw SCARED keyframe into temporal-GT format.")
    parser.add_argument("--keyframe-path", type=Path, default=DEFAULT_KEYFRAME_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--view-layout", choices=["vertical_stereo_stack"], default="vertical_stereo_stack")
    parser.add_argument("--reference-view", choices=["top"], default="top")
    parser.add_argument("--max-frames", type=int, default=0, help="0 processes all frames.")
    parser.add_argument("--overwrite", nargs="?", const=True, default=False, type=parse_bool)
    parser.add_argument("--allow-partial", nargs="?", const=True, default=False, type=parse_bool)
    parser.add_argument("--save-scene-points", nargs="?", const=True, default=False, type=parse_bool)
    parser.add_argument("--save-debug", nargs="?", const=True, default=True, type=parse_bool)
    return parser.parse_args()


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output directory already exists: {path}. Pass --overwrite true to replace it.")
    if path.exists() and overwrite:
        shutil.rmtree(path)
    for rel in [
        "left",
        "right",
        "gt/depth_npy",
        "gt/disparity_npy",
        "gt/valid_mask",
        "calibration",
        "debug",
    ]:
        (path / rel).mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_run_log(path: Path, lines: Iterable[str]) -> None:
    path.write_text("\n".join(lines) + "\n")


def basename(name: str) -> str:
    return Path(name).name


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
            if not member.isfile() or not basename(member.name).endswith(".json"):
                continue
            stem = basename(member.name).replace("frame_data", "").replace(".json", "")
            idx = int(stem)
            f = tf.extractfile(member)
            if f is None:
                continue
            out[idx] = json.loads(f.read().decode("utf-8"))
    return out


def iter_scene_points(scene_tar: Path, max_frames: int) -> Iterable[tuple[int, np.ndarray]]:
    try:
        import tifffile
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("tifffile is required to read float32 scene_points TIFF files") from exc
    with tarfile.open(scene_tar, "r|gz") as tf:
        yielded = 0
        for member in tf:
            if not member.isfile() or not basename(member.name).endswith((".tiff", ".tif")):
                continue
            stem = basename(member.name).replace("scene_points", "").replace(".tiff", "").replace(".tif", "")
            idx = int(stem)
            f = tf.extractfile(member)
            if f is None:
                continue
            arr = np.asarray(tifffile.imread(io.BytesIO(f.read())), dtype=np.float32)
            yield idx, arr
            yielded += 1
            if max_frames > 0 and yielded >= max_frames:
                break


def find_nested_key(data: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        if name in data:
            return data[name]
    for value in data.values():
        if isinstance(value, dict):
            found = find_nested_key(value, names)
            if found is not None:
                return found
    return None


def numeric_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def calibration_from_frame_data(frame_data: dict[str, Any]) -> dict[str, Any]:
    cal = {key: find_nested_key(frame_data, [key, key.lower(), key.upper()]) for key in ["KL", "KR", "DL", "DR", "R", "T"]}
    missing = [key for key, value in cal.items() if value is None]
    if missing:
        raise RuntimeError(f"Missing calibration fields in frame_data: {missing}")
    kl = numeric_array(cal["KL"])
    t = numeric_array(cal["T"]).reshape(-1)
    baseline = float(np.linalg.norm(t))
    return {
        **cal,
        "baseline": baseline,
        "fx": float(kl[0, 0]),
        "fy": float(kl[1, 1]),
        "cx": float(kl[0, 2]),
        "cy": float(kl[1, 2]),
        "camera_pose": find_nested_key(frame_data, ["camera_pose", "camera-pose", "pose", "M"]),
        "timestamp": find_nested_key(frame_data, ["timestamp", "time", "time_stamp"]),
    }


def split_vertical_stereo(rgb: np.ndarray, scene_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rgb.shape[0] % 2 != 0 or scene_points.shape[0] % 2 != 0:
        raise RuntimeError(f"Expected even height for vertical stack, got rgb={rgb.shape}, scene={scene_points.shape}")
    rgb_mid = rgb.shape[0] // 2
    scene_mid = scene_points.shape[0] // 2
    left = rgb[:rgb_mid]
    right = rgb[rgb_mid:]
    scene_ref = scene_points[:scene_mid]
    if left.shape[:2] != scene_ref.shape[:2] or right.shape[:2] != scene_ref.shape[:2]:
        raise RuntimeError(f"Split dimensions mismatch: left={left.shape}, right={right.shape}, scene_ref={scene_ref.shape}")
    return left, right, scene_ref


def compute_depth_disparity(scene_ref: np.ndarray, fx: float, baseline: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = scene_ref[..., 2].astype(np.float32, copy=True)
    norm = np.linalg.norm(np.nan_to_num(scene_ref, nan=0.0), axis=2)
    valid = np.isfinite(depth) & (depth > 0.0) & np.isfinite(norm) & (norm > 0.0)
    disp = np.zeros_like(depth, dtype=np.float32)
    depth_out = np.where(valid, depth, 0.0).astype(np.float32)
    disp[valid] = (float(fx) * float(baseline) / np.maximum(depth[valid], 1e-6)).astype(np.float32)
    return depth_out, disp, valid


def save_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def save_mask_png(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask.astype(np.uint8) * 255))


def scalar_preview(value: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    arr = value.astype(np.float32, copy=False)
    norm = np.clip((arr - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    rgb = np.stack([norm, 1.0 - np.abs(norm - 0.5) * 2.0, 1.0 - norm], axis=-1)
    rgb[~np.isfinite(arr)] = 0.0
    return (rgb * 255.0).astype(np.uint8)


def finite_stats(values: np.ndarray) -> dict[str, float]:
    vals = values[np.isfinite(values) & (values > 0)]
    if vals.size == 0:
        return {"min": math.nan, "median": math.nan, "max": math.nan}
    return {"min": float(np.min(vals)), "median": float(np.median(vals)), "max": float(np.max(vals))}


def write_calibration(path: Path, cal: dict[str, Any], frame_idx: int, view_layout: str, reference_view: str) -> None:
    payload = {
        "KL": cal["KL"],
        "KR": cal["KR"],
        "DL": cal["DL"],
        "DR": cal["DR"],
        "R": cal["R"],
        "T": cal["T"],
        "baseline": cal["baseline"],
        "fx": cal["fx"],
        "fy": cal["fy"],
        "cx": cal["cx"],
        "cy": cal["cy"],
        "camera_pose": cal["camera_pose"],
        "timestamp": cal["timestamp"],
        "source_frame_index": frame_idx,
        "view_layout": view_layout,
        "reference_view": reference_view,
        "depth_unit_assumption": "mm",
        "disparity_formula": "fx * baseline / depth",
    }
    write_json(path, payload)


def make_contact_sheet(path: Path, debug_items: list[tuple[str, np.ndarray]]) -> None:
    from PIL import Image, ImageDraw

    panel_w, panel_h = 240, 192
    label_h = 24
    cols = 5
    rows = int(math.ceil(len(debug_items) / cols))
    canvas = np.full((rows * (panel_h + label_h), cols * panel_w, 3), 255, dtype=np.uint8)
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    for idx, (label, item) in enumerate(debug_items):
        r, c = divmod(idx, cols)
        x = c * panel_w
        y = r * (panel_h + label_h)
        if item.ndim == 2:
            tile = np.repeat(item[..., None], 3, axis=2)
        else:
            tile = item[..., :3]
        resized = Image.fromarray(tile.astype(np.uint8)).resize((panel_w, panel_h), Image.Resampling.BILINEAR)
        image.paste(resized, (x, y + label_h))
        draw.text((x + 4, y + 5), label[:36], fill=(0, 0, 0))
    image.save(path)


def validate_output(output_dir: Path, rows: list[dict[str, Any]], image_hw: tuple[int, int]) -> dict[str, Any]:
    expected = len(rows)
    counts = {
        "left": len(list((output_dir / "left").glob("*.png"))),
        "right": len(list((output_dir / "right").glob("*.png"))),
        "depth": len(list((output_dir / "gt" / "depth_npy").glob("*.npy"))),
        "disparity": len(list((output_dir / "gt" / "disparity_npy").glob("*.npy"))),
        "valid_mask": len(list((output_dir / "gt" / "valid_mask").glob("*.png"))),
        "calibration": len(list((output_dir / "calibration").glob("*.json"))),
    }
    errors: list[str] = []
    if any(count != expected for count in counts.values()):
        errors.append(f"file_count_mismatch:{counts}, expected={expected}")
    h, w = image_hw
    for row in rows:
        left = cv2.imread(row["left_path"], cv2.IMREAD_COLOR)
        right = cv2.imread(row["right_path"], cv2.IMREAD_COLOR)
        depth = np.load(row["depth_path"])
        disp = np.load(row["disparity_path"])
        valid = cv2.imread(row["valid_mask_path"], cv2.IMREAD_GRAYSCALE) > 0
        if left is None or right is None:
            errors.append(f"missing_image:{row['frame_index']}")
            continue
        if left.shape[:2] != (h, w) or right.shape[:2] != (h, w):
            errors.append(f"image_shape_mismatch:{row['frame_index']}")
        if depth.shape != (h, w) or disp.shape != (h, w) or valid.shape != (h, w):
            errors.append(f"gt_shape_mismatch:{row['frame_index']}")
        if not np.isfinite(disp[valid]).all():
            errors.append(f"nonfinite_valid_disparity:{row['frame_index']}")
        if np.any(depth[valid] < 0):
            errors.append(f"negative_valid_depth:{row['frame_index']}")
    return {"counts": counts, "expected_count": expected, "is_valid": not errors, "errors": errors}


def write_metadata(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "frame_index",
        "left_path",
        "right_path",
        "depth_path",
        "disparity_path",
        "valid_mask_path",
        "calibration_path",
        "timestamp",
        "baseline",
        "fx",
        "valid_pixel_pct",
        "depth_min",
        "depth_median",
        "depth_max",
        "disp_min",
        "disp_median",
        "disp_max",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{col: row.get(col, "") for col in columns} for row in rows])


def write_readme(path: Path, summary: dict[str, Any]) -> None:
    validation = summary["validation"]
    readme = f"""# SCARED Curated Temporal-GT Sequence

Converted from `{summary['source_keyframe_path']}` using the conservative raw-keyframe converter.

- Frames processed: `{summary['num_processed_frames']}`
- Output size: `{summary['output_image_width']}x{summary['output_image_height']}`
- View layout: `{summary['view_layout']}`
- Reference view: `{summary['reference_view']}`
- Depth source: `{summary['depth_source']}`
- Valid mask: `{summary['valid_mask_definition']}`
- Baseline mean/std: `{summary['baseline_mean']:.6f}` / `{summary['baseline_std']:.6f}`
- fx mean/std: `{summary['fx_mean']:.6f}` / `{summary['fx_std']:.6f}`
- Mean valid pixels: `{summary['valid_pixel_pct_mean']:.3f}%`
- Mean median depth: `{summary['depth_median_mean']:.6f}`
- Mean median disparity: `{summary['disparity_median_mean']:.6f}`
- Complete: `{summary['is_complete']}`
- Validation passed: `{validation['is_valid']}`

This sequence is ready for later stereo inference/evaluation. No S2M2 inference has been run by this converter.

## Validation

Counts: `{validation['counts']}`

Errors: `{validation['errors']}`

## Notes

The converter splits each `rgb.mp4` frame as a vertical stereo stack: top half as left/reference and bottom half as right. It applies the same top-half split to `scene_points` and derives reference-view depth from `scene_points[..., 2]`. Disparity is derived as `fx * baseline / depth` on valid pixels and set to zero elsewhere.
"""
    path.write_text(readme)


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
    counts = SourceCounts(
        rgb=video_frame_count(video_path),
        frame_data=len(frame_data),
        scene_points=count_tar_members(scene_tar, ".tiff"),
    )
    if len({counts.rgb, counts.frame_data, counts.scene_points}) != 1 and not args.allow_partial:
        raise RuntimeError(f"Source frame counts mismatch: {counts}. Pass --allow-partial true to convert common prefix.")
    process_count = min(counts.rgb, counts.frame_data, counts.scene_points)
    if args.max_frames > 0:
        process_count = min(process_count, args.max_frames)
    ensure_output_dir(args.output_dir, bool(args.overwrite))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    rows: list[dict[str, Any]] = []
    baselines: list[float] = []
    fxs: list[float] = []
    valid_pcts: list[float] = []
    depth_medians: list[float] = []
    disp_medians: list[float] = []
    debug_items: list[tuple[str, np.ndarray]] = []
    debug_indices = {0, process_count // 2, max(process_count - 1, 0)}
    image_hw: tuple[int, int] | None = None

    for scene_idx, scene_points in iter_scene_points(scene_tar, process_count):
        if scene_idx >= process_count:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, scene_idx)
        ok, bgr = cap.read()
        if not ok:
            if args.allow_partial:
                break
            raise RuntimeError(f"Could not read RGB frame {scene_idx}")
        if scene_idx not in frame_data:
            if args.allow_partial:
                continue
            raise RuntimeError(f"Missing frame_data for frame {scene_idx}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        left, right, scene_ref = split_vertical_stereo(rgb, scene_points)
        if image_hw is None:
            image_hw = left.shape[:2]
        cal = calibration_from_frame_data(frame_data[scene_idx])
        depth, disp, valid = compute_depth_disparity(scene_ref, cal["fx"], cal["baseline"])
        stem = f"{scene_idx:06d}"
        left_path = args.output_dir / "left" / f"{stem}.png"
        right_path = args.output_dir / "right" / f"{stem}.png"
        depth_path = args.output_dir / "gt" / "depth_npy" / f"{stem}.npy"
        disp_path = args.output_dir / "gt" / "disparity_npy" / f"{stem}.npy"
        mask_path = args.output_dir / "gt" / "valid_mask" / f"{stem}.png"
        calib_path = args.output_dir / "calibration" / f"{stem}.json"
        save_png(left_path, left)
        save_png(right_path, right)
        np.save(depth_path, depth.astype(np.float32))
        np.save(disp_path, disp.astype(np.float32))
        save_mask_png(mask_path, valid)
        write_calibration(calib_path, cal, scene_idx, args.view_layout, args.reference_view)
        if args.save_scene_points:
            scene_dir = args.output_dir / "gt" / "scene_points_ref_npy"
            scene_dir.mkdir(parents=True, exist_ok=True)
            np.save(scene_dir / f"{stem}.npy", scene_ref.astype(np.float32))

        depth_stats = finite_stats(depth)
        disp_stats = finite_stats(disp)
        valid_pct = float(valid.mean() * 100.0)
        baselines.append(cal["baseline"])
        fxs.append(cal["fx"])
        valid_pcts.append(valid_pct)
        depth_medians.append(depth_stats["median"])
        disp_medians.append(disp_stats["median"])
        row = {
            "frame_index": stem,
            "left_path": str(left_path),
            "right_path": str(right_path),
            "depth_path": str(depth_path),
            "disparity_path": str(disp_path),
            "valid_mask_path": str(mask_path),
            "calibration_path": str(calib_path),
            "timestamp": cal["timestamp"],
            "baseline": cal["baseline"],
            "fx": cal["fx"],
            "valid_pixel_pct": valid_pct,
            "depth_min": depth_stats["min"],
            "depth_median": depth_stats["median"],
            "depth_max": depth_stats["max"],
            "disp_min": disp_stats["min"],
            "disp_median": disp_stats["median"],
            "disp_max": disp_stats["max"],
        }
        rows.append(row)

        if args.save_debug and scene_idx in debug_indices:
            depth_prev = scalar_preview(depth, 0.0, 120.0)
            disp_prev = scalar_preview(disp, 0.0, 120.0)
            mask_prev = np.repeat((valid.astype(np.uint8) * 255)[..., None], 3, axis=2)
            debug_items.extend(
                [
                    (f"{stem} left", left),
                    (f"{stem} right", right),
                    (f"{stem} depth", depth_prev),
                    (f"{stem} disparity", disp_prev),
                    (f"{stem} valid", mask_prev),
                ]
            )
            if scene_idx == 0:
                save_png(args.output_dir / "debug" / "depth_preview_000000.png", depth_prev)
                save_png(args.output_dir / "debug" / "disparity_preview_000000.png", disp_prev)
                save_png(args.output_dir / "debug" / "valid_mask_preview_000000.png", mask_prev)
        print(f"converted {stem} valid={valid_pct:.2f}%", flush=True)

    cap.release()
    if not rows or image_hw is None:
        raise RuntimeError("No frames were converted")
    write_metadata(args.output_dir / "metadata.csv", rows)
    if args.save_debug and debug_items:
        make_contact_sheet(args.output_dir / "debug" / "contact_sheet.png", debug_items)
    validation = validate_output(args.output_dir, rows, image_hw)
    warnings: list[str] = []
    if counts.rgb != counts.frame_data or counts.rgb != counts.scene_points:
        warnings.append(f"source_count_mismatch:{counts}")
    summary = {
        "sequence_name": args.output_dir.name,
        "source_keyframe_path": str(keyframe),
        "num_rgb_frames": counts.rgb,
        "num_frame_data_json": counts.frame_data,
        "num_scene_points_tiff": counts.scene_points,
        "num_processed_frames": len(rows),
        "output_image_height": image_hw[0],
        "output_image_width": image_hw[1],
        "view_layout": args.view_layout,
        "reference_view": args.reference_view,
        "depth_source": "scene_points_channel_2",
        "valid_mask_definition": "finite(depth) & (depth > 0) & (norm(scene_points_ref) > 0)",
        "baseline_mean": float(np.mean(baselines)),
        "baseline_std": float(np.std(baselines)),
        "fx_mean": float(np.mean(fxs)),
        "fx_std": float(np.std(fxs)),
        "valid_pixel_pct_mean": float(np.mean(valid_pcts)),
        "depth_median_mean": float(np.mean(depth_medians)),
        "disparity_median_mean": float(np.mean(disp_medians)),
        "is_complete": bool(len(rows) == counts.rgb == counts.frame_data == counts.scene_points),
        "warnings": warnings,
        "validation": validation,
    }
    write_json(args.output_dir / "summary.json", summary)
    write_readme(args.output_dir / "README.md", summary)
    write_run_log(
        args.output_dir / "run.log",
        [
            "SCARED raw keyframe to temporal-GT conversion",
            f"keyframe_path={keyframe}",
            f"output_dir={args.output_dir}",
            f"view_layout={args.view_layout}",
            f"reference_view={args.reference_view}",
            f"source_counts={counts}",
            f"processed_frames={len(rows)}",
            f"validation_is_valid={validation['is_valid']}",
            f"warnings={warnings}",
        ],
    )
    print(json.dumps({"output_dir": str(args.output_dir), "processed_frames": len(rows), "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
