#!/usr/bin/env python3
"""Probe a raw SCARED keyframe folder without extracting full archives."""

from __future__ import annotations

import argparse
import io
import json
import math
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_KEYFRAME_PATH = Path("dataset/SCARED/raw/extracted/dataset_1/dataset_1/keyframe_1")
DEFAULT_OUTPUT_DIR = Path("dataset/SCARED/curated/audit/probe_scared_keyframe_structure/dataset_1_keyframe_1")
DEFAULT_SAMPLE_INDICES = [0, 1, 50, 100, 196]


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
    parser = argparse.ArgumentParser(description="Probe SCARED raw keyframe temporal structure.")
    parser.add_argument("--keyframe-path", type=Path, default=DEFAULT_KEYFRAME_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-indices", nargs="+", type=int, default=DEFAULT_SAMPLE_INDICES)
    parser.add_argument("--overwrite", nargs="?", const=True, default=False, type=parse_bool)
    parser.add_argument("--extract-samples-only", nargs="?", const=True, default=True, type=parse_bool)
    return parser.parse_args()


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output directory already exists: {path}. Pass --overwrite true to replace it.")
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_run_log(path: Path, lines: Iterable[str]) -> None:
    path.write_text("\n".join(lines) + "\n")


def basename(name: str) -> str:
    return Path(name).name


def tar_member_count(path: Path, suffix: str) -> int:
    count = 0
    with tarfile.open(path, "r|gz") as tf:
        for member in tf:
            if member.isfile() and basename(member.name).endswith(suffix):
                count += 1
    return count


def load_frame_data_samples(tar_path: Path, sample_indices: list[int]) -> tuple[dict[int, dict[str, Any]], int]:
    targets = {f"frame_data{idx:06d}.json": idx for idx in sample_indices}
    out: dict[int, dict[str, Any]] = {}
    count = 0
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile() or not basename(member.name).endswith(".json"):
                continue
            count += 1
            idx = targets.get(basename(member.name))
            if idx is None:
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            out[idx] = json.loads(f.read().decode("utf-8"))
    return out, count


def load_tiff_from_bytes(data: bytes) -> np.ndarray:
    try:
        import tifffile

        return np.asarray(tifffile.imread(io.BytesIO(data)))
    except Exception:
        from PIL import Image

        return np.asarray(Image.open(io.BytesIO(data)))


def load_scene_point_samples(tar_path: Path, sample_indices: list[int]) -> tuple[dict[int, np.ndarray], int]:
    targets = {f"scene_points{idx:06d}.tiff": idx for idx in sample_indices}
    out: dict[int, np.ndarray] = {}
    count = 0
    with tarfile.open(tar_path, "r|gz") as tf:
        for member in tf:
            if not member.isfile() or not basename(member.name).endswith((".tiff", ".tif")):
                continue
            count += 1
            idx = targets.get(basename(member.name))
            if idx is None:
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            out[idx] = load_tiff_from_bytes(f.read())
    return out, count


def matrix_shape(value: Any) -> list[int] | None:
    arr = np.asarray(value)
    return list(arr.shape) if arr.size else None


def find_first_key(data: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        if name in data:
            return data[name]
    return None


def find_nested_key(data: dict[str, Any], names: list[str]) -> Any:
    found = find_first_key(data, names)
    if found is not None:
        return found
    for value in data.values():
        if isinstance(value, dict):
            found = find_nested_key(value, names)
            if found is not None:
                return found
    return None


def flatten_numeric(value: Any) -> np.ndarray:
    try:
        return np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return np.asarray([], dtype=np.float64)


def frame_data_summary(samples: dict[int, dict[str, Any]], frame_count: int) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    klt_ref = None
    pose_ref = None
    timestamps: list[tuple[int, float]] = []
    calibration_changes = False
    pose_changes = False
    for idx, data in sorted(samples.items()):
        keys = sorted(data.keys())
        cal = {key: find_nested_key(data, [key, key.lower(), key.upper()]) for key in ["KL", "KR", "DL", "DR", "R", "T"]}
        pose = find_nested_key(data, ["camera_pose", "camera-pose", "pose", "M"])
        timestamp = find_nested_key(data, ["timestamp", "time", "time_stamp"])
        if timestamp is not None:
            try:
                timestamps.append((idx, float(timestamp)))
            except (TypeError, ValueError):
                pass
        cal_parts = [flatten_numeric(v) for v in cal.values() if v is not None and flatten_numeric(v).size]
        cal_numeric = np.concatenate(cal_parts) if cal_parts else np.asarray([], dtype=np.float64)
        pose_numeric = flatten_numeric(pose)
        if klt_ref is None:
            klt_ref = cal_numeric
        elif cal_numeric.size and klt_ref.size and not np.allclose(cal_numeric, klt_ref, equal_nan=True):
            calibration_changes = True
        if pose_ref is None:
            pose_ref = pose_numeric
        elif pose_numeric.size and pose_ref.size and not np.allclose(pose_numeric, pose_ref, equal_nan=True):
            pose_changes = True
        t_vec = flatten_numeric(cal.get("T"))
        rows[f"{idx:06d}"] = {
            "top_level_keys": keys,
            "KL": cal.get("KL"),
            "KR": cal.get("KR"),
            "DL": cal.get("DL"),
            "DR": cal.get("DR"),
            "R": cal.get("R"),
            "T": cal.get("T"),
            "baseline_vector_T": t_vec.tolist(),
            "baseline_magnitude": float(np.linalg.norm(t_vec)) if t_vec.size else math.nan,
            "camera_pose_matrix_shape": matrix_shape(pose),
            "timestamp": timestamp,
        }
    timestamps = sorted(timestamps)
    deltas = [timestamps[i][1] - timestamps[i - 1][1] for i in range(1, len(timestamps))]
    frame_deltas = [timestamps[i][0] - timestamps[i - 1][0] for i in range(1, len(timestamps))]
    positive_deltas = [d for d in deltas if d > 0]
    median_delta = float(np.median(positive_deltas)) if positive_deltas else math.nan
    if np.isfinite(median_delta) and median_delta > 1e8:
        timestamp_unit = "nanoseconds"
        seconds_scale = 1e-9
    elif np.isfinite(median_delta) and median_delta > 1e4:
        timestamp_unit = "microseconds"
        seconds_scale = 1e-6
    elif np.isfinite(median_delta) and median_delta > 10:
        timestamp_unit = "milliseconds"
        seconds_scale = 1e-3
    else:
        timestamp_unit = "seconds"
        seconds_scale = 1.0
    fps_values = [
        float(df) / (float(dt) * seconds_scale)
        for df, dt in zip(frame_deltas, deltas)
        if df > 0 and dt > 0
    ]
    return {
        "frame_data_count": frame_count,
        "sampled_frames": rows,
        "calibration_changes_over_samples": calibration_changes,
        "pose_changes_over_samples": pose_changes,
        "timestamp_samples": [{"frame_index": idx, "timestamp": ts} for idx, ts in timestamps],
        "timestamp_deltas": deltas,
        "timestamp_frame_index_deltas": frame_deltas,
        "inferred_timestamp_unit": timestamp_unit,
        "approx_fps_from_timestamps": float(np.mean(fps_values)) if fps_values else math.nan,
        "approx_fps_from_first_adjacent_timestamp_pair": fps_values[0] if fps_values else math.nan,
    }


def channel_stats(channel: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(channel)
    nonzero = finite & (channel != 0)
    positive = finite & (channel > 0)
    values = channel[nonzero].astype(np.float64)
    stats: dict[str, Any] = {
        "finite_pct": float(np.mean(finite) * 100.0),
        "nonzero_pct": float(np.mean(nonzero) * 100.0),
        "positive_pct": float(np.mean(positive) * 100.0),
    }
    if values.size:
        stats.update(
            {
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
                "p1": float(np.percentile(values, 1)),
                "p50": float(np.percentile(values, 50)),
                "p99": float(np.percentile(values, 99)),
            }
        )
    return stats


def scene_points_summary(samples: dict[int, np.ndarray], scene_count: int) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    z_medians: list[float] = []
    for idx, arr in sorted(samples.items()):
        arr_f = arr.astype(np.float32, copy=False)
        finite = np.isfinite(arr_f)
        norm = np.linalg.norm(np.nan_to_num(arr_f, nan=0.0), axis=2) if arr_f.ndim == 3 and arr_f.shape[2] >= 3 else np.abs(arr_f)
        valid = np.isfinite(norm) & (norm > 0)
        channels = {}
        if arr_f.ndim == 3:
            for c in range(arr_f.shape[2]):
                channels[f"C{c}"] = channel_stats(arr_f[..., c])
            if arr_f.shape[2] >= 3 and "p50" in channels["C2"]:
                z_medians.append(float(channels["C2"]["p50"]))
        rows[f"{idx:06d}"] = {
            "shape": list(arr_f.shape),
            "dtype": str(arr.dtype),
            "finite_percentage": float(np.mean(finite) * 100.0),
            "valid_percentage_norm_xyz_gt_0": float(np.mean(valid) * 100.0),
            "per_channel": channels,
        }
    looks_mm = bool(z_medians and 10.0 <= float(np.nanmedian(z_medians)) <= 1000.0)
    return {
        "scene_points_count": scene_count,
        "sampled_frames": rows,
        "inferred_semantic_meaning": {"C0": "likely X", "C1": "likely Y", "C2": "likely Z/depth"},
        "recommended_valid_mask": "np.isfinite(scene_points).all(axis=2) & (np.linalg.norm(scene_points, axis=2) > 0)",
        "recommended_depth_extraction_formula": "depth = scene_points[..., 2] if C2 remains positive Z/depth after visual alignment checks",
        "values_look_like_millimeters": looks_mm,
    }


def ffprobe_video(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        data = json.loads(result.stdout)
        return data.get("streams", [{}])[0]
    except Exception as exc:
        return {"error": str(exc)}


def read_video_samples(path: Path, sample_indices: list[int]) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    meta = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration": float(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
        if cap.get(cv2.CAP_PROP_FPS)
        else math.nan,
    }
    frames: dict[int, np.ndarray] = {}
    for idx in sample_indices:
        if idx < 0 or idx >= meta["frame_count"]:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if ok:
            frames[idx] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cap.release()
    return frames, meta


def save_rgb(path: Path, image: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image.astype(np.uint8)).save(path)


def colorize_scalar(value: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    arr = value.astype(np.float32, copy=False)
    finite = np.isfinite(arr)
    if vmin is None:
        vmin = float(np.nanpercentile(arr[finite], 1)) if finite.any() else 0.0
    if vmax is None:
        vmax = float(np.nanpercentile(arr[finite], 99)) if finite.any() else 1.0
    norm = np.clip((arr - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    rgb = np.stack([norm, 1.0 - np.abs(norm - 0.5) * 2.0, 1.0 - norm], axis=-1)
    rgb[~finite] = 0.0
    return (rgb * 255.0).astype(np.uint8)


def make_board(tiles: list[tuple[str, np.ndarray]], cols: int = 3, panel_size: tuple[int, int] = (320, 256)) -> np.ndarray:
    from PIL import Image, ImageDraw

    panel_w, panel_h = panel_size
    label_h = 24
    rows = int(math.ceil(len(tiles) / cols))
    canvas = np.full((rows * (panel_h + label_h), cols * panel_w, 3), 255, dtype=np.uint8)
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    for idx, (label, tile) in enumerate(tiles):
        r, c = divmod(idx, cols)
        x = c * panel_w
        y = r * (panel_h + label_h)
        if tile.ndim == 2:
            tile_rgb = np.repeat(tile[..., None], 3, axis=2)
        else:
            tile_rgb = tile[..., :3]
        resized = Image.fromarray(tile_rgb.astype(np.uint8)).resize((panel_w, panel_h), Image.Resampling.BILINEAR)
        image.paste(resized, (x, y + label_h))
        draw.text((x + 4, y + 5), label[:40], fill=(0, 0, 0))
    return np.asarray(image)


def resize_like_rgb(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    h, w = target_shape
    if image.ndim == 2:
        pil = Image.fromarray(colorize_scalar(image))
    else:
        pil = Image.fromarray(image.astype(np.uint8))
    return np.asarray(pil.resize((w, h), Image.Resampling.BILINEAR))


def corr_gray(a: np.ndarray, b: np.ndarray) -> float:
    def gray(x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:
            return x.astype(np.float32)
        xf = x[..., :3].astype(np.float32)
        return 0.299 * xf[..., 0] + 0.587 * xf[..., 1] + 0.114 * xf[..., 2]

    ag = gray(a)
    bg = gray(b)
    if ag.shape != bg.shape:
        bg = resize_like_rgb(bg, ag.shape[:2])
        if bg.ndim == 3:
            bg = gray(bg)
    av = ag.reshape(-1).astype(np.float64)
    bv = bg.reshape(-1).astype(np.float64)
    av -= av.mean()
    bv -= bv.mean()
    denom = float(np.sqrt(np.sum(av * av) * np.sum(bv * bv)))
    return float(np.sum(av * bv) / denom) if denom > 1e-12 else math.nan


def infer_layout(rgb: np.ndarray, scene_shape: tuple[int, int]) -> dict[str, Any]:
    h, w = rgb.shape[:2]
    top_bottom_corr = corr_gray(rgb[: h // 2], rgb[h // 2 :]) if h % 2 == 0 else math.nan
    left_right_corr = corr_gray(rgb[:, : w // 2], rgb[:, w // 2 :]) if w % 2 == 0 else math.nan
    direct_match = (h, w) == scene_shape
    swapped_match = (w, h) == scene_shape
    if direct_match and top_bottom_corr > 0.25:
        layout = "vertical_stereo_stack"
    elif direct_match and left_right_corr > 0.25:
        layout = "horizontal_stereo_stack"
    elif direct_match:
        layout = "single_view"
    elif swapped_match:
        layout = "rotated_single_view"
    else:
        layout = "unknown"
    return {
        "inferred_layout": layout,
        "rgb_shape": [h, w, int(rgb.shape[2])],
        "scene_points_hw": list(scene_shape),
        "direct_dimension_match": direct_match,
        "swapped_dimension_match": swapped_match,
        "top_bottom_correlation": top_bottom_corr,
        "left_right_correlation": left_right_corr,
    }


def save_visual_diagnostics(output_dir: Path, video_samples: dict[int, np.ndarray], scene_samples: dict[int, np.ndarray]) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for idx in [0, 50, 100]:
        if idx in video_samples:
            save_rgb(output_dir / f"rgb_frame_{idx:06d}.png", video_samples[idx])
    if 0 not in scene_samples or 0 not in video_samples:
        return diagnostics
    rgb = video_samples[0]
    scene = scene_samples[0].astype(np.float32, copy=False)
    z = scene[..., 2] if scene.ndim == 3 and scene.shape[2] >= 3 else scene.astype(np.float32)
    norm = np.linalg.norm(np.nan_to_num(scene, nan=0.0), axis=2) if scene.ndim == 3 else np.abs(scene)
    valid = np.isfinite(z) & (norm > 0)
    save_rgb(output_dir / "scene_depth_000000.png", colorize_scalar(z))
    save_rgb(output_dir / "scene_valid_mask_000000.png", np.repeat((valid.astype(np.uint8) * 255)[..., None], 3, axis=2))
    save_rgb(output_dir / "scene_z_colormap_000000.png", colorize_scalar(z))
    save_rgb(output_dir / "scene_xyz_norm_000000.png", colorize_scalar(norm))

    z_rgb = colorize_scalar(z)
    z_resized = resize_like_rgb(z_rgb, rgb.shape[:2])
    overlay = (0.55 * rgb.astype(np.float32) + 0.45 * z_resized.astype(np.float32)).clip(0, 255).astype(np.uint8)
    save_rgb(output_dir / "overlay_rgb_depth_candidates_000000.png", overlay)

    hypotheses = [
        ("A RGB/depth as-is", rgb),
        ("B RGB rot90 cw", np.rot90(rgb, k=3)),
        ("C RGB rot90 ccw", np.rot90(rgb, k=1)),
        ("D RGB flip vertical", np.flipud(rgb)),
    ]
    if rgb.shape[0] % 2 == 0:
        hypotheses.extend([("E top half", rgb[: rgb.shape[0] // 2]), ("E bottom half", rgb[rgb.shape[0] // 2 :])])
        save_rgb(output_dir / "rgb_top_half_000000.png", rgb[: rgb.shape[0] // 2])
        save_rgb(output_dir / "rgb_bottom_half_000000.png", rgb[rgb.shape[0] // 2 :])
    if rgb.shape[1] % 2 == 0:
        hypotheses.extend([("F left half", rgb[:, : rgb.shape[1] // 2]), ("F right half", rgb[:, rgb.shape[1] // 2 :])])
        save_rgb(output_dir / "rgb_left_half_000000.png", rgb[:, : rgb.shape[1] // 2])
        save_rgb(output_dir / "rgb_right_half_000000.png", rgb[:, rgb.shape[1] // 2 :])
    tiles = []
    for label, candidate in hypotheses:
        depth_resized = resize_like_rgb(z_rgb, candidate.shape[:2])
        cand_overlay = (0.55 * candidate[..., :3].astype(np.float32) + 0.45 * depth_resized.astype(np.float32)).clip(0, 255).astype(np.uint8)
        tiles.append((label, cand_overlay))
    save_rgb(output_dir / "alignment_hypotheses_000000.png", make_board(tiles, cols=2))
    split_tiles = [("RGB as-is", rgb)]
    if rgb.shape[0] % 2 == 0:
        split_tiles += [("top half", rgb[: rgb.shape[0] // 2]), ("bottom half", rgb[rgb.shape[0] // 2 :])]
    if rgb.shape[1] % 2 == 0:
        split_tiles += [("left half", rgb[:, : rgb.shape[1] // 2]), ("right half", rgb[:, rgb.shape[1] // 2 :])]
    save_rgb(output_dir / "possible_stereo_layout_000000.png", make_board(split_tiles, cols=2))
    diagnostics["saved_visual_diagnostics"] = True
    return diagnostics


def video_summary(
    video_path: Path,
    sample_indices: list[int],
    scene_count: int,
    first_scene_shape: tuple[int, int] | None,
) -> tuple[dict[str, Any], dict[int, np.ndarray]]:
    samples, cv_meta = read_video_samples(video_path, sample_indices)
    ff_meta = ffprobe_video(video_path)
    sample_shapes = {f"{idx:06d}": list(frame.shape) for idx, frame in sorted(samples.items())}
    first_rgb = samples[min(samples)] if samples else None
    orientation = {}
    if first_rgb is not None and first_scene_shape is not None:
        orientation = {
            "dimensions_match_scene_points_directly": tuple(first_rgb.shape[:2]) == tuple(first_scene_shape),
            "dimensions_match_scene_points_swapped": tuple(first_rgb.shape[:2]) == tuple(reversed(first_scene_shape)),
        }
    return (
        {
            "ffprobe": ff_meta,
            **cv_meta,
            "frame_count_matches_scene_points_count": cv_meta["frame_count"] == scene_count,
            "extracted_sample_frame_shapes": sample_shapes,
            **orientation,
        },
        samples,
    )


def conversion_feasibility(
    *,
    video_meta: dict[str, Any],
    frame_data_count: int,
    scene_count: int,
    layout: str,
) -> dict[str, Any]:
    has_rgb = int(video_meta.get("frame_count", 0)) > 0
    has_frame_data = frame_data_count > 0
    has_scene = scene_count > 0
    can_stereo = layout in {"vertical_stereo_stack", "horizontal_stereo_stack", "rotated_vertical_stereo_stack"}
    blocking = "" if can_stereo else "rgb.mp4 does not look like an unambiguous temporal stereo pair; left/right temporal extraction is unresolved."
    return {
        "has_temporal_rgb_video": has_rgb,
        "has_temporal_frame_data": has_frame_data,
        "has_temporal_scene_points": has_scene,
        "frame_count_rgb": video_meta.get("frame_count"),
        "frame_count_frame_data": frame_data_count,
        "frame_count_scene_points": scene_count,
        "can_extract_temporal_depth_gt": bool(has_rgb and has_scene and video_meta.get("frame_count") == scene_count),
        "can_extract_temporal_stereo_left_right": bool(can_stereo),
        "blocking_issue_for_stereo_benchmark": blocking,
        "recommended_next_step": (
            "Implement a converter using scene_points[...,2] as depth GT and split rgb.mp4 into left/right temporal frames."
            if can_stereo
            else "Inspect alignment_hypotheses_000000.png and possible_stereo_layout_000000.png; locate temporal right/left video source before S2M2 conversion."
        ),
    }


def write_readme(path: Path, frame_summary: dict[str, Any], scene_summary: dict[str, Any], video_meta: dict[str, Any], layout: dict[str, Any], feasibility: dict[str, Any]) -> None:
    readme = f"""# SCARED Raw Keyframe Structure Probe

This report probes a raw SCARED keyframe folder without extracting the full raw archives.

## frame_data.tar.gz

`frame_data.tar.gz` contains `{frame_summary['frame_data_count']}` per-frame JSON files. The sampled JSON files expose camera calibration fields such as `KL`, `KR`, `DL`, `DR`, `R`, `T`, camera pose information when present, and timestamps. Calibration changes over sampled frames: `{frame_summary['calibration_changes_over_samples']}`. Pose changes over sampled frames: `{frame_summary['pose_changes_over_samples']}`. Approximate FPS from sampled timestamps: `{frame_summary['approx_fps_from_timestamps']}`.

## scene_points.tar.gz

`scene_points.tar.gz` contains `{scene_summary['scene_points_count']}` sampled/listed TIFF frames. The sampled arrays are treated as XYZ scene points. Channel interpretation is inferred as C0≈X, C1≈Y, C2≈Z/depth. The recommended depth candidate is `scene_points[..., 2]`, guarded by the recommended valid mask in `scene_points_summary.json`. Values look like millimeters: `{scene_summary['values_look_like_millimeters']}`.

## rgb.mp4

`rgb.mp4` reports width `{video_meta.get('width')}`, height `{video_meta.get('height')}`, FPS `{video_meta.get('fps')}`, and frame count `{video_meta.get('frame_count')}`. It matches scene-point count: `{video_meta.get('frame_count_matches_scene_points_count')}`. Inferred layout: `{layout['inferred_layout']}`. Top/bottom correlation: `{layout['top_bottom_correlation']}`. Left/right correlation: `{layout['left_right_correlation']}`.

## Conversion Feasibility

Scene points can likely provide temporal depth GT: `{feasibility['can_extract_temporal_depth_gt']}`. Temporal stereo left/right extraction is currently feasible: `{feasibility['can_extract_temporal_stereo_left_right']}`.

Blocking stereo issue: {feasibility['blocking_issue_for_stereo_benchmark'] or 'none'}

Recommended next step: {feasibility['recommended_next_step']}

## Suggested Curated Output Structure

If conversion is feasible, use a sequence folder with:

- `left/000000.png`, `right/000000.png` for temporal stereo frames
- `gt/DepthL_float32/000000.npy` from scene-point Z/depth
- `gt/ValidMask/000000.npy` from finite nonzero XYZ norm
- `calibration/000000.json` from frame-data calibration
- `metadata.csv` linking frame id, image paths, GT paths, calibration, timestamp, and validity ratio

Do not assume `rgb.mp4` is stereo until the saved visual diagnostics confirm the layout.
"""
    path.write_text(readme)


def main() -> None:
    args = parse_args()
    if not args.extract_samples_only:
        raise RuntimeError("This probe intentionally supports sample-only extraction; full extraction is out of scope.")
    sample_indices = sorted(set(args.sample_indices))
    ensure_output_dir(args.output_dir, bool(args.overwrite))
    keyframe = args.keyframe_path
    frame_tar = keyframe / "data" / "frame_data.tar.gz"
    scene_tar = keyframe / "data" / "scene_points.tar.gz"
    video_path = keyframe / "data" / "rgb.mp4"
    for path in [frame_tar, scene_tar, video_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    frame_samples, frame_count = load_frame_data_samples(frame_tar, sample_indices)
    scene_samples, scene_count = load_scene_point_samples(scene_tar, sample_indices)
    frame_summary = frame_data_summary(frame_samples, frame_count)
    scene_summary = scene_points_summary(scene_samples, scene_count)
    first_scene_shape = None
    if scene_samples:
        first_scene = scene_samples[min(scene_samples)]
        first_scene_shape = tuple(first_scene.shape[:2])
    vid_summary, video_samples = video_summary(video_path, sample_indices, scene_count, first_scene_shape)
    layout = infer_layout(video_samples[min(video_samples)], first_scene_shape) if video_samples and first_scene_shape else {"inferred_layout": "unknown"}
    diagnostics = save_visual_diagnostics(args.output_dir, video_samples, scene_samples)
    feasibility = conversion_feasibility(
        video_meta=vid_summary,
        frame_data_count=frame_count,
        scene_count=scene_count,
        layout=str(layout.get("inferred_layout", "unknown")),
    )

    write_json(args.output_dir / "frame_data_summary.json", frame_summary)
    write_json(args.output_dir / "scene_points_summary.json", scene_summary)
    write_json(args.output_dir / "video_summary.json", {**vid_summary, "stereo_layout_inference": layout})
    write_json(args.output_dir / "conversion_feasibility.json", feasibility)
    write_readme(args.output_dir / "README.md", frame_summary, scene_summary, vid_summary, layout, feasibility)
    write_run_log(
        args.output_dir / "run.log",
        [
            "SCARED raw keyframe structure probe",
            f"keyframe_path={keyframe}",
            f"output_dir={args.output_dir}",
            f"sample_indices={sample_indices}",
            f"frame_data_count={frame_count}",
            f"scene_points_count={scene_count}",
            f"video_frame_count={vid_summary.get('frame_count')}",
            f"inferred_layout={layout.get('inferred_layout')}",
            f"can_extract_temporal_depth_gt={feasibility['can_extract_temporal_depth_gt']}",
            f"can_extract_temporal_stereo_left_right={feasibility['can_extract_temporal_stereo_left_right']}",
            f"visual_diagnostics={diagnostics}",
        ],
    )
    print(json.dumps({"output_dir": str(args.output_dir), "layout": layout.get("inferred_layout"), "scene_points_count": scene_count}, indent=2))


if __name__ == "__main__":
    main()
