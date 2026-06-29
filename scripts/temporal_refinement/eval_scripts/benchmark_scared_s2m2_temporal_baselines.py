#!/usr/bin/env python3
"""Cache-only SCARED temporal-GT benchmark for S2M2-S temporal baselines."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
LIB_DIR = ROOT / "scripts" / "temporal_refinement" / "lib"
sys.path.insert(0, str(LIB_DIR))

from temporal_baselines import (  # noqa: E402
    BaselineResult,
    adaptive_no_raft_diff_grad_sequence,
    adaptive_no_raft_diff_sequence,
    confidence_reset_warped_ema_sequence,
    conservative_adaptive_ema_sequence,
    fixed_ema_sequence,
    raft_warped_ema_sequence,
    warp_disparity_numpy,
)
from video_qualitative import colorize_scalar, make_board, write_mp4  # noqa: E402


DEFAULT_SEQUENCE_DIR = Path("dataset/SCARED/curated/temporal_gt/test_dataset_9_keyframe_3")
DEFAULT_S2M2_CACHE = Path(
    "results/03_temporal_refinement/evaluation/gt_temporal_test_dataset_9_keyframe_3/predictions/S2M2-S_512"
)
DEFAULT_SAV_CACHE = Path(
    "results/03_temporal_refinement/evaluation/gt_temporal_test_dataset_9_keyframe_3/predictions/StereoAnyVideo_384x640"
)
DEFAULT_FLOW_CACHE = Path("results/04_dataset_derivatives/SCARED/temporal_gt_flow_cache/test_dataset_9_keyframe_3/raft")
DEFAULT_OUTPUT_DIR = Path("results/03_temporal_refinement/scared_s2m2_temporal_baselines_v2_no_raft_adaptive")

SUMMARY_COLUMNS = [
    "method_id",
    "method_name",
    "method_type",
    "alpha",
    "valid_frame_count",
    "valid_pixel_coverage",
    "depth_mae_mm",
    "disp_mae_px",
    "disp_rmse_px",
    "bad_1px_pct",
    "bad_2px_pct",
    "bad_3px_pct",
    "bad_1mm_pct",
    "bad_2mm_pct",
    "bad_5mm_pct",
    "raw_temporal_disp_diff_px",
    "motion_compensated_temporal_mae_px",
    "temporal_pair_count",
    "runtime_postprocess_ms",
    "inherited_model_runtime_ms",
    "peak_inherited_vram_mb",
    "flow_forward_runtime_ms",
    "flow_backward_runtime_ms",
    "flow_runtime_used_ms",
    "online_runtime_estimated_ms",
    "online_runtime_formula",
    "online_runtime_notes",
    "online_peak_vram_estimated_mb",
    "runtime_fairness_category",
    "role",
    "notes",
]

PER_FRAME_COLUMNS = [
    "method_id",
    "method_name",
    "frame_id",
    "valid_pixel_count",
    "gt_valid_pixel_count",
    "valid_pixel_coverage",
    "depth_mae_mm",
    "disp_mae_px",
    "disp_rmse_px",
    "bad_1px_pct",
    "bad_2px_pct",
    "bad_3px_pct",
    "bad_1mm_pct",
    "bad_2mm_pct",
    "bad_5mm_pct",
]

PER_PAIR_COLUMNS = [
    "method_id",
    "method_name",
    "prev_frame_id",
    "cur_frame_id",
    "valid_pixel_count",
    "raw_temporal_disp_diff_px",
    "motion_compensated_temporal_mae_px",
    "metric_flow_source",
]

ADAPTIVE_SWEEP_COLUMNS = [
    "method_id",
    "method_name",
    "method_type",
    "alpha_min",
    "alpha_max",
    "diff_scale_px",
    "grad_scale_px",
    "w_diff",
    "w_grad",
    "depth_mae_mm",
    "disp_mae_px",
    "motion_compensated_temporal_mae_px",
    "raw_temporal_disp_diff_px",
    "online_runtime_estimated_ms",
]


@dataclass(frozen=True)
class FrameRecord:
    frame_id: str
    left_path: Path
    gt_disp_path: Path
    gt_depth_path: Path
    valid_mask_path: Path
    calibration_path: Path
    valid_ratio: float
    fx: float
    baseline_mm: float


@dataclass
class MethodRecord:
    method_id: str
    method_name: str
    method_type: str
    predictions: list[np.ndarray]
    alpha: float | None = None
    postprocess_ms: float = 0.0
    inherited_runtime_ms: float = math.nan
    inherited_vram_mb: float = math.nan
    role: str = "benchmark"
    notes: str = ""
    params: dict[str, float] | None = None


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_alphas(value: str) -> list[float]:
    out = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise argparse.ArgumentTypeError("--ema-alphas must contain at least one alpha")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache-only SCARED S2M2-S temporal baseline benchmark.")
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE_DIR)
    parser.add_argument("--s2m2-cache-dir", type=Path, default=DEFAULT_S2M2_CACHE)
    parser.add_argument("--sav-cache-dir", type=Path, default=DEFAULT_SAV_CACHE)
    parser.add_argument("--flow-cache-dir", type=Path, default=DEFAULT_FLOW_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-valid-ratio", type=float, default=0.2)
    parser.add_argument("--ema-alphas", type=parse_alphas, default=parse_alphas("0.50"))
    parser.add_argument("--overwrite", nargs="?", const=True, default=False, type=parse_bool)
    parser.add_argument("--warp-device", default="auto", help="Device used for torch warp_disp calls: auto, cpu, cuda, ...")
    return parser.parse_args()


def resolve_path(sequence_dir: Path, value: str | None, fallback: Path) -> Path:
    if not value:
        return fallback
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    rooted = ROOT / path
    if rooted.exists():
        return rooted
    return sequence_dir / path


def calibration_fx_baseline(path: Path) -> tuple[float, float]:
    calib = json.loads(path.read_text())
    if "fx" in calib and "baseline_mm" in calib:
        return float(calib["fx"]), float(calib["baseline_mm"])
    p1 = np.array(calib["P1"]["data"], dtype=np.float64).reshape(calib["P1"]["rows"], calib["P1"]["cols"])
    p2 = np.array(calib["P2"]["data"], dtype=np.float64).reshape(calib["P2"]["rows"], calib["P2"]["cols"])
    return float(p1[0, 0]), float(abs(p2[0, 3] / p2[0, 0]))


def load_frames(sequence_dir: Path) -> list[FrameRecord]:
    metadata_csv = sequence_dir / "metadata.csv"
    with metadata_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    frames: list[FrameRecord] = []
    ids: list[int] = []
    for row in rows:
        frame_id = row.get("frame_id") or row.get("id")
        if frame_id is None:
            raise RuntimeError("metadata.csv must contain frame_id or id")
        ids.append(int(frame_id))
        calib_path = resolve_path(sequence_dir, row.get("calibration_path"), sequence_dir / "calibration" / f"{frame_id}.json")
        fx, baseline_mm = calibration_fx_baseline(calib_path)
        frames.append(
            FrameRecord(
                frame_id=frame_id,
                left_path=resolve_path(sequence_dir, row.get("left_path"), sequence_dir / "left" / f"{frame_id}.png"),
                gt_disp_path=resolve_path(
                    sequence_dir,
                    row.get("disparity_float32_path"),
                    sequence_dir / "gt" / "Disparity_float32" / f"{frame_id}.npy",
                ),
                gt_depth_path=resolve_path(
                    sequence_dir,
                    row.get("depth_float32_path"),
                    sequence_dir / "gt" / "DepthL_float32" / f"{frame_id}.npy",
                ),
                valid_mask_path=resolve_path(
                    sequence_dir,
                    row.get("valid_mask_path"),
                    sequence_dir / "gt" / "ValidMask" / f"{frame_id}.npy",
                ),
                calibration_path=calib_path,
                valid_ratio=float(row.get("valid_pixel_ratio", "0") or 0.0),
                fx=fx,
                baseline_mm=baseline_mm,
            )
        )
    if not frames:
        raise RuntimeError(f"No frames found in {metadata_csv}")
    if ids != sorted(ids):
        raise RuntimeError("Frame ids in metadata.csv are not sorted")
    if ids != list(range(ids[0], ids[0] + len(ids))):
        raise RuntimeError("Frame ids in metadata.csv are not continuous")
    return frames


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    dataset_root = (ROOT / "dataset").resolve()
    resolved = path.resolve()
    if resolved == dataset_root or resolved.is_relative_to(dataset_root):
        raise RuntimeError(f"Refusing to write benchmark outputs under dataset/: {path}")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output directory already exists: {path}. Pass --overwrite true to replace benchmark outputs.")
    path.mkdir(parents=True, exist_ok=True)
    (path / "qualitative").mkdir(exist_ok=True)


def load_prediction(cache_dir: Path, frame_id: str) -> np.ndarray:
    candidates = [cache_dir / f"{frame_id}.npy", cache_dir / f"test_dataset_9_keyframe_3_frame_{frame_id}_disp.npy"]
    for candidate in candidates:
        if candidate.exists():
            return np.load(candidate, allow_pickle=False).astype(np.float32)
    matches = sorted(cache_dir.glob(f"*{frame_id}*disp.npy"))
    if matches:
        return np.load(matches[0], allow_pickle=False).astype(np.float32)
    raise FileNotFoundError(f"Missing cached prediction for frame {frame_id} in {cache_dir}")


def load_prediction_sequence(cache_dir: Path, frames: Sequence[FrameRecord]) -> list[np.ndarray]:
    return [load_prediction(cache_dir, frame.frame_id) for frame in frames]


def flow_path(flow_cache_dir: Path, subdir: str, prev_id: str, cur_id: str) -> Path:
    return flow_cache_dir / subdir / f"{prev_id}_to_{cur_id}.npy"


def load_forward_flow(flow_cache_dir: Path, prev_id: str, cur_id: str) -> np.ndarray:
    return np.load(flow_path(flow_cache_dir, "forward_flow", prev_id, cur_id), allow_pickle=False).astype(np.float32)


def load_forward_confidence(flow_cache_dir: Path, prev_id: str, cur_id: str) -> np.ndarray:
    return np.load(flow_path(flow_cache_dir, "forward_confidence", prev_id, cur_id), allow_pickle=False).astype(np.float32)


def load_occlusion(flow_cache_dir: Path, prev_id: str, cur_id: str) -> np.ndarray:
    return np.load(flow_path(flow_cache_dir, "occlusion", prev_id, cur_id), allow_pickle=False)


def load_metadata_runtime(cache_dir: Path) -> tuple[float, float]:
    for name in ["metadata.json", "summary.json"]:
        path = cache_dir / name
        if path.exists():
            data = json.loads(path.read_text())
            runtime = data.get("avg_runtime_ms", data.get("runtime_ms", math.nan))
            vram = data.get("peak_vram_mb", data.get("peak_gpu_memory_mb", math.nan))
            return float(runtime), float(vram)
    return math.nan, math.nan


def _mean_csv_float(rows: Sequence[dict[str, str]], column: str) -> float:
    values: list[float] = []
    for row in rows:
        raw = row.get(column, "")
        if raw == "":
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if math.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else math.nan


def load_flow_runtime_metadata(flow_cache_dir: Path) -> dict[str, float | str]:
    summary_path = flow_cache_dir / "flow_cache_summary.json"
    manifest_path = flow_cache_dir / "flow_cache_manifest.csv"
    source = "missing"
    forward = math.nan
    backward = math.nan
    peak = math.nan
    if summary_path.exists():
        data = json.loads(summary_path.read_text())
        forward = float(data.get("average_forward_runtime_ms", math.nan))
        backward = float(data.get("average_backward_runtime_ms", math.nan))
        peak = float(data.get("peak_vram_mb", math.nan))
        source = str(summary_path)
    if (not math.isfinite(forward) or not math.isfinite(backward)) and manifest_path.exists():
        with manifest_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if not math.isfinite(forward):
            forward = _mean_csv_float(rows, "forward_runtime_ms")
        if not math.isfinite(backward):
            backward = _mean_csv_float(rows, "backward_runtime_ms")
        source = str(manifest_path)
    return {
        "average_forward_runtime_ms": forward,
        "average_backward_runtime_ms": backward,
        "peak_vram_mb": peak,
        "source": source,
    }


def finite_sum(*values: float) -> float:
    if any(not math.isfinite(float(value)) for value in values):
        return math.nan
    return float(sum(values))


def finite_max(*values: float) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return max(finite) if finite else math.nan


def online_runtime_fields(method: MethodRecord, flow_runtime: dict[str, float | str]) -> dict[str, Any]:
    fwd = float(flow_runtime["average_forward_runtime_ms"])
    bwd = float(flow_runtime["average_backward_runtime_ms"])
    raft_peak = float(flow_runtime["peak_vram_mb"])
    inherited = float(method.inherited_runtime_ms)
    post = float(method.postprocess_ms)
    inherited_vram = float(method.inherited_vram_mb)
    base_note = "VRAM estimate uses max(model_peak, RAFT_peak) for RAFT-based methods; it does not sum VRAM because S2M2 and RAFT are not explicitly run concurrently."

    if method.method_id == "stereoanyvideo":
        return {
            "flow_forward_runtime_ms": fwd,
            "flow_backward_runtime_ms": bwd,
            "flow_runtime_used_ms": 0.0,
            "online_runtime_estimated_ms": inherited,
            "online_runtime_formula": "inherited_model_runtime_ms",
            "online_runtime_notes": "Cached SAV runtime from metadata; no RAFT flow is added.",
            "online_peak_vram_estimated_mb": inherited_vram,
            "runtime_fairness_category": "video_model_cached_runtime",
        }
    if method.method_type == "raft_warped_ema":
        flow_used = fwd
        return {
            "flow_forward_runtime_ms": fwd,
            "flow_backward_runtime_ms": bwd,
            "flow_runtime_used_ms": flow_used,
            "online_runtime_estimated_ms": finite_sum(inherited, flow_used, post),
            "online_runtime_formula": "inherited_model_runtime_ms + average_forward_runtime_ms + runtime_postprocess_ms",
            "online_runtime_notes": base_note,
            "online_peak_vram_estimated_mb": finite_max(inherited_vram, raft_peak),
            "runtime_fairness_category": "requires_online_forward_flow",
        }
    if method.method_type in {"confidence_occlusion_reset", "conservative_adaptive_ema"}:
        flow_used = finite_sum(fwd, bwd)
        return {
            "flow_forward_runtime_ms": fwd,
            "flow_backward_runtime_ms": bwd,
            "flow_runtime_used_ms": flow_used,
            "online_runtime_estimated_ms": finite_sum(inherited, flow_used, post),
            "online_runtime_formula": "inherited_model_runtime_ms + average_forward_runtime_ms + average_backward_runtime_ms + runtime_postprocess_ms",
            "online_runtime_notes": base_note,
            "online_peak_vram_estimated_mb": finite_max(inherited_vram, raft_peak),
            "runtime_fairness_category": "requires_online_forward_backward_flow",
        }
    if method.method_type in {"adaptive_no_raft_diff", "adaptive_no_raft_diff_grad"}:
        return {
            "flow_forward_runtime_ms": fwd,
            "flow_backward_runtime_ms": bwd,
            "flow_runtime_used_ms": 0.0,
            "online_runtime_estimated_ms": finite_sum(inherited, post),
            "online_runtime_formula": "inherited_model_runtime_ms + runtime_postprocess_ms",
            "online_runtime_notes": "No optical flow used in the prediction formula; RAFT flow is metric-only.",
            "online_peak_vram_estimated_mb": inherited_vram,
            "runtime_fairness_category": "no_flow_adaptive",
        }
    if method.method_type == "fixed_ema":
        return {
            "flow_forward_runtime_ms": fwd,
            "flow_backward_runtime_ms": bwd,
            "flow_runtime_used_ms": 0.0,
            "online_runtime_estimated_ms": finite_sum(inherited, post),
            "online_runtime_formula": "inherited_model_runtime_ms + runtime_postprocess_ms",
            "online_runtime_notes": "No optical flow required.",
            "online_peak_vram_estimated_mb": inherited_vram,
            "runtime_fairness_category": "no_flow",
        }
    return {
        "flow_forward_runtime_ms": fwd,
        "flow_backward_runtime_ms": bwd,
        "flow_runtime_used_ms": 0.0,
        "online_runtime_estimated_ms": inherited,
        "online_runtime_formula": "inherited_model_runtime_ms",
        "online_runtime_notes": "No optical flow required.",
        "online_peak_vram_estimated_mb": inherited_vram,
        "runtime_fairness_category": "no_flow",
    }


def read_mask(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False).astype(bool)
    try:
        from PIL import Image

        return np.array(Image.open(path)) > 0
    except Exception as exc:  # pragma: no cover - non-npy masks are not expected here
        raise RuntimeError(f"Could not read mask {path}: {exc}") from exc


def audit_array(path: Path, expected_shape: tuple[int, ...], dtype_kind: set[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(path), "exists": path.exists(), "shape": "", "dtype": "", "finite": False, "valid": False}
    if not path.exists():
        out["error"] = "missing"
        return out
    arr = np.load(path, allow_pickle=False)
    out["shape"] = list(arr.shape)
    out["dtype"] = str(arr.dtype)
    out["finite"] = bool(np.isfinite(arr).all())
    out["valid"] = tuple(arr.shape) == expected_shape and arr.dtype.kind in dtype_kind and out["finite"]
    if not out["valid"]:
        out["error"] = f"expected_shape={expected_shape}, expected_dtype_kind={sorted(dtype_kind)}"
    return out


def cache_audit(
    frames: Sequence[FrameRecord],
    s2m2_cache_dir: Path,
    sav_cache_dir: Path,
    flow_cache_dir: Path,
    gt_shape: tuple[int, int],
) -> dict[str, Any]:
    expected_frames = len(frames)
    expected_pairs = expected_frames - 1
    audit: dict[str, Any] = {
        "expected_frames": expected_frames,
        "expected_pairs": expected_pairs,
        "s2m2_cache_dir": str(s2m2_cache_dir),
        "sav_cache_dir": str(sav_cache_dir),
        "flow_cache_dir": str(flow_cache_dir),
        "s2m2_cache_exists": s2m2_cache_dir.exists(),
        "sav_cache_exists": sav_cache_dir.exists(),
        "flow_cache_exists": flow_cache_dir.exists(),
        "s2m2_prediction_count": len(list(s2m2_cache_dir.glob("*.npy"))) if s2m2_cache_dir.exists() else 0,
        "sav_prediction_count": len(list(sav_cache_dir.glob("*.npy"))) if sav_cache_dir.exists() else 0,
        "forward_flow_count": len(list((flow_cache_dir / "forward_flow").glob("*.npy"))) if flow_cache_dir.exists() else 0,
        "forward_confidence_count": len(list((flow_cache_dir / "forward_confidence").glob("*.npy"))) if flow_cache_dir.exists() else 0,
        "occlusion_count": len(list((flow_cache_dir / "occlusion").glob("*.npy"))) if flow_cache_dir.exists() else 0,
        "prediction_shape_errors": [],
        "flow_shape_errors": [],
    }
    for frame in frames:
        for label, cache_dir in [("s2m2", s2m2_cache_dir), ("sav", sav_cache_dir)]:
            try:
                pred = load_prediction(cache_dir, frame.frame_id)
                if pred.shape != gt_shape or not np.isfinite(pred).all():
                    audit["prediction_shape_errors"].append({"cache": label, "frame_id": frame.frame_id, "shape": list(pred.shape)})
            except Exception as exc:
                audit["prediction_shape_errors"].append({"cache": label, "frame_id": frame.frame_id, "error": str(exc)})
    for prev, cur in zip(frames[:-1], frames[1:]):
        specs = [
            (flow_path(flow_cache_dir, "forward_flow", prev.frame_id, cur.frame_id), (gt_shape[0], gt_shape[1], 2), {"f"}),
            (flow_path(flow_cache_dir, "forward_confidence", prev.frame_id, cur.frame_id), gt_shape, {"f"}),
            (flow_path(flow_cache_dir, "occlusion", prev.frame_id, cur.frame_id), gt_shape, {"b", "f"}),
        ]
        for path, shape, kind in specs:
            result = audit_array(path, shape, kind)
            if not result["valid"]:
                audit["flow_shape_errors"].append(result)
    audit["complete"] = bool(
        audit["s2m2_cache_exists"]
        and audit["sav_cache_exists"]
        and audit["flow_cache_exists"]
        and audit["s2m2_prediction_count"] == expected_frames
        and audit["sav_prediction_count"] == expected_frames
        and audit["forward_flow_count"] == expected_pairs
        and audit["forward_confidence_count"] == expected_pairs
        and audit["occlusion_count"] == expected_pairs
        and not audit["prediction_shape_errors"]
        and not audit["flow_shape_errors"]
    )
    return audit


def mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else math.nan


def frame_metrics(frame: FrameRecord, pred: np.ndarray) -> dict[str, Any]:
    gt_disp = np.load(frame.gt_disp_path, allow_pickle=False).astype(np.float32)
    gt_depth = np.load(frame.gt_depth_path, allow_pickle=False).astype(np.float32)
    valid_mask = read_mask(frame.valid_mask_path)
    gt_valid = valid_mask & np.isfinite(gt_disp) & np.isfinite(gt_depth) & (gt_disp > 0) & (gt_depth > 0)
    valid = gt_valid & np.isfinite(pred) & (pred > 0.1)
    gt_count = int(gt_valid.sum())
    valid_count = int(valid.sum())
    row: dict[str, Any] = {
        "frame_id": frame.frame_id,
        "valid_pixel_count": valid_count,
        "gt_valid_pixel_count": gt_count,
        "valid_pixel_coverage": float(valid_count / gt_count) if gt_count else math.nan,
    }
    if valid_count == 0:
        for key in [
            "depth_mae_mm",
            "disp_mae_px",
            "disp_rmse_px",
            "bad_1px_pct",
            "bad_2px_pct",
            "bad_3px_pct",
            "bad_1mm_pct",
            "bad_2mm_pct",
            "bad_5mm_pct",
        ]:
            row[key] = math.nan
        return row
    disp_err = np.abs(pred[valid] - gt_disp[valid])
    pred_depth = frame.fx * frame.baseline_mm / np.maximum(pred.astype(np.float32), 1e-6)
    depth_err = np.abs(pred_depth[valid] - gt_depth[valid])
    row.update(
        {
            "depth_mae_mm": float(np.mean(depth_err)),
            "disp_mae_px": float(np.mean(disp_err)),
            "disp_rmse_px": float(np.sqrt(np.mean(disp_err**2))),
            "bad_1px_pct": float(np.mean(disp_err > 1.0) * 100.0),
            "bad_2px_pct": float(np.mean(disp_err > 2.0) * 100.0),
            "bad_3px_pct": float(np.mean(disp_err > 3.0) * 100.0),
            "bad_1mm_pct": float(np.mean(depth_err > 1.0) * 100.0),
            "bad_2mm_pct": float(np.mean(depth_err > 2.0) * 100.0),
            "bad_5mm_pct": float(np.mean(depth_err > 5.0) * 100.0),
        }
    )
    return row


def temporal_pair_metrics(
    prev_frame: FrameRecord,
    cur_frame: FrameRecord,
    prev_pred: np.ndarray,
    cur_pred: np.ndarray,
    flow_cache_dir: Path,
    warp_device: str,
) -> dict[str, Any]:
    prev_mask = read_mask(prev_frame.valid_mask_path)
    cur_mask = read_mask(cur_frame.valid_mask_path)
    raw_valid = prev_mask & cur_mask & np.isfinite(prev_pred) & np.isfinite(cur_pred) & (prev_pred > 0.1) & (cur_pred > 0.1)
    raw_diff = float(np.mean(np.abs(cur_pred[raw_valid] - prev_pred[raw_valid]))) if raw_valid.any() else math.nan
    flow = load_forward_flow(flow_cache_dir, prev_frame.frame_id, cur_frame.frame_id)
    warped_prev = warp_disparity_numpy(prev_pred, flow, device=warp_device)
    mc_valid = raw_valid & np.isfinite(warped_prev) & (warped_prev > 0.1)
    mc = float(np.mean(np.abs(cur_pred[mc_valid] - warped_prev[mc_valid]))) if mc_valid.any() else math.nan
    return {
        "prev_frame_id": prev_frame.frame_id,
        "cur_frame_id": cur_frame.frame_id,
        "valid_pixel_count": int(mc_valid.sum()),
        "raw_temporal_disp_diff_px": raw_diff,
        "motion_compensated_temporal_mae_px": mc,
        "metric_flow_source": str(flow_cache_dir),
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_methods(
    frames: Sequence[FrameRecord],
    s2m2_raw: list[np.ndarray],
    sav: list[np.ndarray],
    flow_cache_dir: Path,
    ema_alphas: Sequence[float],
    s2m2_runtime: float,
    s2m2_vram: float,
    sav_runtime: float,
    sav_vram: float,
    warp_device: str,
) -> list[MethodRecord]:
    frame_ids = [frame.frame_id for frame in frames]
    flow_loader = lambda prev_id, cur_id: load_forward_flow(flow_cache_dir, prev_id, cur_id)
    conf_loader = lambda prev_id, cur_id: load_forward_confidence(flow_cache_dir, prev_id, cur_id)
    occ_loader = lambda prev_id, cur_id: load_occlusion(flow_cache_dir, prev_id, cur_id)
    methods = [
        MethodRecord(
            "s2m2_s_raw",
            "S2M2-S raw",
            "cached_prediction",
            s2m2_raw,
            postprocess_ms=0.0,
            inherited_runtime_ms=s2m2_runtime,
            inherited_vram_mb=s2m2_vram,
        )
    ]
    for alpha in ema_alphas:
        result = fixed_ema_sequence(s2m2_raw, alpha)
        methods.append(
            MethodRecord(
                f"s2m2_s_fixed_ema_a{alpha:.2f}",
                f"S2M2-S fixed EMA alpha={alpha:.2f}",
                "fixed_ema",
                result.predictions,
                alpha=alpha,
                postprocess_ms=result.postprocess_ms_per_frame,
                inherited_runtime_ms=s2m2_runtime,
                inherited_vram_mb=s2m2_vram,
            )
        )
    alpha = 0.50
    warped = raft_warped_ema_sequence(s2m2_raw, frame_ids, flow_loader, alpha, warp_device=warp_device)
    methods.append(
        MethodRecord(
            "s2m2_s_raft_warped_ema_a0.50",
            "S2M2-S RAFT-warped EMA alpha=0.50",
            "raft_warped_ema",
            warped.predictions,
            alpha=alpha,
            postprocess_ms=warped.postprocess_ms_per_frame,
            inherited_runtime_ms=s2m2_runtime,
            inherited_vram_mb=s2m2_vram,
        )
    )
    reset = confidence_reset_warped_ema_sequence(
        s2m2_raw, frame_ids, flow_loader, conf_loader, occ_loader, alpha, warp_device=warp_device
    )
    methods.append(
        MethodRecord(
            "s2m2_s_confidence_occlusion_reset_a0.50",
            "S2M2-S warped EMA + confidence/occlusion reset alpha=0.50",
            "confidence_occlusion_reset",
            reset.predictions,
            alpha=alpha,
            postprocess_ms=reset.postprocess_ms_per_frame,
            inherited_runtime_ms=s2m2_runtime,
            inherited_vram_mb=s2m2_vram,
        )
    )
    adaptive = conservative_adaptive_ema_sequence(
        s2m2_raw, frame_ids, flow_loader, conf_loader, occ_loader, warp_device=warp_device
    )
    methods.append(
        MethodRecord(
            "s2m2_s_conservative_adaptive_ema",
            "S2M2-S conservative adaptive EMA",
            "conservative_adaptive_ema",
            adaptive.predictions,
            postprocess_ms=adaptive.postprocess_ms_per_frame,
            inherited_runtime_ms=s2m2_runtime,
            inherited_vram_mb=s2m2_vram,
            notes="alpha_min=0.40 alpha_max=0.80 diff_scale_px=3.0 equal risk weights",
        )
    )
    methods.append(
        MethodRecord(
            "stereoanyvideo",
            "StereoAnyVideo",
            "cached_prediction",
            sav,
            postprocess_ms=0.0,
            inherited_runtime_ms=sav_runtime,
            inherited_vram_mb=sav_vram,
            role="teacher_or_comparison_not_gt",
            notes="Cached comparison/teacher; not ground truth.",
        )
    )
    return methods


def aggregate_summary(
    method: MethodRecord,
    frame_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    flow_runtime: dict[str, float | str],
) -> dict[str, Any]:
    total_valid = sum(int(row["valid_pixel_count"]) for row in frame_rows)
    total_gt = sum(int(row["gt_valid_pixel_count"]) for row in frame_rows)
    runtime_fields = online_runtime_fields(method, flow_runtime)
    return {
        "method_id": method.method_id,
        "method_name": method.method_name,
        "method_type": method.method_type,
        "alpha": "" if method.alpha is None else f"{method.alpha:.2f}",
        "valid_frame_count": sum(1 for row in frame_rows if int(row["valid_pixel_count"]) > 0),
        "valid_pixel_coverage": float(total_valid / total_gt) if total_gt else math.nan,
        "depth_mae_mm": mean([row["depth_mae_mm"] for row in frame_rows]),
        "disp_mae_px": mean([row["disp_mae_px"] for row in frame_rows]),
        "disp_rmse_px": mean([row["disp_rmse_px"] for row in frame_rows]),
        "bad_1px_pct": mean([row["bad_1px_pct"] for row in frame_rows]),
        "bad_2px_pct": mean([row["bad_2px_pct"] for row in frame_rows]),
        "bad_3px_pct": mean([row["bad_3px_pct"] for row in frame_rows]),
        "bad_1mm_pct": mean([row["bad_1mm_pct"] for row in frame_rows]),
        "bad_2mm_pct": mean([row["bad_2mm_pct"] for row in frame_rows]),
        "bad_5mm_pct": mean([row["bad_5mm_pct"] for row in frame_rows]),
        "raw_temporal_disp_diff_px": mean([row["raw_temporal_disp_diff_px"] for row in pair_rows]),
        "motion_compensated_temporal_mae_px": mean([row["motion_compensated_temporal_mae_px"] for row in pair_rows]),
        "temporal_pair_count": len(pair_rows),
        "runtime_postprocess_ms": method.postprocess_ms,
        "inherited_model_runtime_ms": method.inherited_runtime_ms,
        "peak_inherited_vram_mb": method.inherited_vram_mb,
        **runtime_fields,
        "role": method.role,
        "notes": method.notes,
    }


def no_raft_adaptive_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for alpha_min in [0.25, 0.30, 0.35, 0.40]:
        for alpha_max in [0.70, 0.80, 0.90]:
            for diff_scale in [2.0, 3.0, 5.0, 8.0]:
                is_default = alpha_min == 0.30 and alpha_max == 0.80 and diff_scale == 3.0
                method_id = "s2m2_s_adaptive_no_raft_diff" if is_default else (
                    f"s2m2_s_adaptive_no_raft_diff_amin{alpha_min:.2f}_amax{alpha_max:.2f}_diff{diff_scale:.1f}"
                )
                configs.append(
                    {
                        "method_id": method_id,
                        "method_name": (
                            f"S2M2-S adaptive no-RAFT diff amin={alpha_min:.2f} "
                            f"amax={alpha_max:.2f} diff={diff_scale:.1f}"
                        ),
                        "method_type": "adaptive_no_raft_diff",
                        "alpha_min": alpha_min,
                        "alpha_max": alpha_max,
                        "diff_scale_px": diff_scale,
                        "grad_scale_px": math.nan,
                        "w_diff": 1.0,
                        "w_grad": math.nan,
                    }
                )
    for alpha_min in [0.25, 0.30, 0.35]:
        for alpha_max in [0.75, 0.85, 0.90]:
            for diff_scale in [3.0, 5.0]:
                for grad_scale in [3.0, 5.0, 8.0]:
                    for w_grad in [0.25, 0.50, 1.00]:
                        is_default = (
                            alpha_min == 0.30
                            and alpha_max == 0.85
                            and diff_scale == 3.0
                            and grad_scale == 5.0
                            and w_grad == 0.50
                        )
                        method_id = "s2m2_s_adaptive_no_raft_diff_grad" if is_default else (
                            f"s2m2_s_adaptive_no_raft_diff_grad_amin{alpha_min:.2f}_amax{alpha_max:.2f}"
                            f"_diff{diff_scale:.1f}_grad{grad_scale:.1f}_wg{w_grad:.2f}"
                        )
                        configs.append(
                            {
                                "method_id": method_id,
                                "method_name": (
                                    f"S2M2-S adaptive no-RAFT diff+grad amin={alpha_min:.2f} "
                                    f"amax={alpha_max:.2f} diff={diff_scale:.1f} grad={grad_scale:.1f} wg={w_grad:.2f}"
                                ),
                                "method_type": "adaptive_no_raft_diff_grad",
                                "alpha_min": alpha_min,
                                "alpha_max": alpha_max,
                                "diff_scale_px": diff_scale,
                                "grad_scale_px": grad_scale,
                                "w_diff": 1.0,
                                "w_grad": w_grad,
                            }
                        )
    return configs


def build_no_raft_adaptive_method(
    config: dict[str, Any],
    s2m2_raw: list[np.ndarray],
    inherited_runtime_ms: float,
    inherited_vram_mb: float,
) -> MethodRecord:
    if config["method_type"] == "adaptive_no_raft_diff":
        result = adaptive_no_raft_diff_sequence(
            s2m2_raw,
            alpha_min=float(config["alpha_min"]),
            alpha_max=float(config["alpha_max"]),
            diff_scale_px=float(config["diff_scale_px"]),
        )
    elif config["method_type"] == "adaptive_no_raft_diff_grad":
        result = adaptive_no_raft_diff_grad_sequence(
            s2m2_raw,
            alpha_min=float(config["alpha_min"]),
            alpha_max=float(config["alpha_max"]),
            diff_scale_px=float(config["diff_scale_px"]),
            grad_scale_px=float(config["grad_scale_px"]),
            w_diff=float(config["w_diff"]),
            w_grad=float(config["w_grad"]),
        )
    else:
        raise ValueError(str(config["method_type"]))
    params = {
        key: float(config[key])
        for key in ["alpha_min", "alpha_max", "diff_scale_px", "grad_scale_px", "w_diff", "w_grad"]
        if math.isfinite(float(config[key]))
    }
    return MethodRecord(
        method_id=str(config["method_id"]),
        method_name=str(config["method_name"]),
        method_type=str(config["method_type"]),
        predictions=result.predictions,
        postprocess_ms=result.postprocess_ms_per_frame,
        inherited_runtime_ms=inherited_runtime_ms,
        inherited_vram_mb=inherited_vram_mb,
        notes="No optical flow in prediction formula; RAFT flow is metric-only.",
        params=params,
    )


def evaluate_method(
    method: MethodRecord,
    frames: Sequence[FrameRecord],
    metric_indices: Sequence[int],
    metric_pair_indices: Sequence[int],
    flow_cache_dir: Path,
    warp_device: str,
    flow_runtime: dict[str, float | str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    method_frame_rows: list[dict[str, Any]] = []
    method_pair_rows: list[dict[str, Any]] = []
    for idx in metric_indices:
        row = frame_metrics(frames[idx], method.predictions[idx])
        row.update({"method_id": method.method_id, "method_name": method.method_name})
        method_frame_rows.append(row)
    for idx in metric_pair_indices:
        row = temporal_pair_metrics(
            frames[idx - 1],
            frames[idx],
            method.predictions[idx - 1],
            method.predictions[idx],
            flow_cache_dir,
            warp_device,
        )
        row.update({"method_id": method.method_id, "method_name": method.method_name})
        method_pair_rows.append(row)
    return aggregate_summary(method, method_frame_rows, method_pair_rows, flow_runtime), method_frame_rows, method_pair_rows


def warp_stack_with_flow(prev_stack: np.ndarray, flow: np.ndarray) -> np.ndarray:
    if prev_stack.ndim != 3:
        raise ValueError(f"Expected BxHxW stack, got {prev_stack.shape}")
    b, h, w = prev_stack.shape
    if flow.shape != (h, w, 2):
        raise ValueError(f"Flow shape {flow.shape} does not match stack {(b, h, w)}")
    try:
        import torch
        import torch.nn.functional as F

        if torch.cuda.is_available():
            device = torch.device("cuda")
            pred = torch.from_numpy(prev_stack.astype(np.float32, copy=False))[:, None].to(device)
            flow_t = torch.from_numpy(flow.astype(np.float32, copy=False)).permute(2, 0, 1)[None].to(device)
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, dtype=torch.float32, device=device),
                torch.arange(w, dtype=torch.float32, device=device),
                indexing="ij",
            )
            grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
            coords = grid + flow_t
            norm_x = 2.0 * coords[:, 0] / (w - 1) - 1.0
            norm_y = 2.0 * coords[:, 1] / (h - 1) - 1.0
            grid_norm = torch.stack([norm_x, norm_y], dim=-1).expand(b, -1, -1, -1)
            with torch.no_grad():
                warped = F.grid_sample(pred, grid_norm, mode="bilinear", padding_mode="border", align_corners=True)
            return warped[:, 0].detach().float().cpu().numpy().astype(np.float32)
    except ModuleNotFoundError:
        pass

    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    sample_x = np.clip(xx + flow[..., 0].astype(np.float32, copy=False), 0.0, float(w - 1))
    sample_y = np.clip(yy + flow[..., 1].astype(np.float32, copy=False), 0.0, float(h - 1))
    try:
        from scipy import ndimage

        method_axis = np.broadcast_to(np.arange(b, dtype=np.float32)[:, None, None], (b, h, w))
        coords = np.stack(
            [
                method_axis,
                np.broadcast_to(sample_y, (b, h, w)),
                np.broadcast_to(sample_x, (b, h, w)),
            ],
            axis=0,
        )
        return ndimage.map_coordinates(
            prev_stack.astype(np.float32, copy=False),
            coords,
            order=1,
            mode="nearest",
            prefilter=False,
        ).astype(np.float32)
    except ModuleNotFoundError:
        return np.stack([warp_disparity_numpy(prev_stack[i], flow) for i in range(b)], axis=0)


def evaluate_methods_batch(
    methods: Sequence[MethodRecord],
    frames: Sequence[FrameRecord],
    metric_indices: Sequence[int],
    metric_pair_indices: Sequence[int],
    flow_cache_dir: Path,
    flow_runtime: dict[str, float | str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    frame_rows_by_id: dict[str, list[dict[str, Any]]] = {method.method_id: [] for method in methods}
    pair_rows_by_id: dict[str, list[dict[str, Any]]] = {method.method_id: [] for method in methods}
    for method in methods:
        rows = frame_rows_by_id[method.method_id]
        for idx in metric_indices:
            row = frame_metrics(frames[idx], method.predictions[idx])
            row.update({"method_id": method.method_id, "method_name": method.method_name})
            rows.append(row)
    for idx in metric_pair_indices:
        prev_frame = frames[idx - 1]
        cur_frame = frames[idx]
        prev_mask = read_mask(prev_frame.valid_mask_path)
        cur_mask = read_mask(cur_frame.valid_mask_path)
        common_mask = prev_mask & cur_mask
        flow = load_forward_flow(flow_cache_dir, prev_frame.frame_id, cur_frame.frame_id)
        prev_stack = np.stack([method.predictions[idx - 1] for method in methods], axis=0).astype(np.float32, copy=False)
        cur_stack = np.stack([method.predictions[idx] for method in methods], axis=0).astype(np.float32, copy=False)
        warped_stack = warp_stack_with_flow(prev_stack, flow)
        for method_idx, method in enumerate(methods):
            prev_pred = prev_stack[method_idx]
            cur_pred = cur_stack[method_idx]
            warped_prev = warped_stack[method_idx]
            raw_valid = common_mask & np.isfinite(prev_pred) & np.isfinite(cur_pred) & (prev_pred > 0.1) & (cur_pred > 0.1)
            raw_diff = float(np.mean(np.abs(cur_pred[raw_valid] - prev_pred[raw_valid]))) if raw_valid.any() else math.nan
            mc_valid = raw_valid & np.isfinite(warped_prev) & (warped_prev > 0.1)
            mc = float(np.mean(np.abs(cur_pred[mc_valid] - warped_prev[mc_valid]))) if mc_valid.any() else math.nan
            pair_rows_by_id[method.method_id].append(
                {
                    "method_id": method.method_id,
                    "method_name": method.method_name,
                    "prev_frame_id": prev_frame.frame_id,
                    "cur_frame_id": cur_frame.frame_id,
                    "valid_pixel_count": int(mc_valid.sum()),
                    "raw_temporal_disp_diff_px": raw_diff,
                    "motion_compensated_temporal_mae_px": mc,
                    "metric_flow_source": str(flow_cache_dir),
                }
            )
    summary_rows: list[dict[str, Any]] = []
    all_frame_rows: list[dict[str, Any]] = []
    all_pair_rows: list[dict[str, Any]] = []
    for method in methods:
        frame_rows = frame_rows_by_id[method.method_id]
        pair_rows = pair_rows_by_id[method.method_id]
        summary_rows.append(aggregate_summary(method, frame_rows, pair_rows, flow_runtime))
        all_frame_rows.extend(frame_rows)
        all_pair_rows.extend(pair_rows)
    return summary_rows, all_frame_rows, all_pair_rows


def adaptive_sweep_row(config: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "method_id": config["method_id"],
        "method_name": config["method_name"],
        "method_type": config["method_type"],
        "alpha_min": config["alpha_min"],
        "alpha_max": config["alpha_max"],
        "diff_scale_px": config["diff_scale_px"],
        "grad_scale_px": "" if not math.isfinite(float(config["grad_scale_px"])) else config["grad_scale_px"],
        "w_diff": config["w_diff"],
        "w_grad": "" if not math.isfinite(float(config["w_grad"])) else config["w_grad"],
        "depth_mae_mm": summary["depth_mae_mm"],
        "disp_mae_px": summary["disp_mae_px"],
        "motion_compensated_temporal_mae_px": summary["motion_compensated_temporal_mae_px"],
        "raw_temporal_disp_diff_px": summary["raw_temporal_disp_diff_px"],
        "online_runtime_estimated_ms": summary["online_runtime_estimated_ms"],
    }


def best_summary(rows: Sequence[dict[str, Any]], key: str, max_runtime_ms: float | None = None) -> dict[str, Any] | None:
    candidates = []
    for row in rows:
        value = float(row.get(key, math.nan))
        runtime = float(row.get("online_runtime_estimated_ms", math.nan))
        if not math.isfinite(value):
            continue
        if max_runtime_ms is not None and (not math.isfinite(runtime) or runtime > max_runtime_ms):
            continue
        candidates.append(row)
    return min(candidates, key=lambda r: float(r[key])) if candidates else None


def read_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    return np.array(Image.open(path).convert("RGB"))


def colorize(value: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
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


def resize_nearest(img: np.ndarray, height: int = 180) -> np.ndarray:
    from PIL import Image

    h, w = img.shape[:2]
    new_w = max(1, int(round(w * height / max(h, 1))))
    return np.array(Image.fromarray(img).resize((new_w, height), Image.Resampling.BILINEAR))


def save_montage(path: Path, tiles: list[tuple[str, np.ndarray]]) -> None:
    from PIL import Image, ImageDraw

    resized = [(title, resize_nearest(tile)) for title, tile in tiles]
    tile_h = max(tile.shape[0] for _, tile in resized) + 24
    tile_w = max(tile.shape[1] for _, tile in resized)
    cols = 4
    rows = int(math.ceil(len(resized) / cols))
    canvas = np.full((rows * tile_h, cols * tile_w, 3), 255, dtype=np.uint8)
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    for idx, (title, tile) in enumerate(resized):
        r, c = divmod(idx, cols)
        y = r * tile_h
        x = c * tile_w
        image.paste(Image.fromarray(tile), (x, y + 24))
        draw.text((x + 4, y + 4), title[:34], fill=(0, 0, 0))
    image.save(path)


def write_qualitative(output_dir: Path, frames: Sequence[FrameRecord], methods: Sequence[MethodRecord]) -> None:
    method_ids = [
        "s2m2_s_raw",
        "s2m2_s_fixed_ema_a0.50",
        "s2m2_s_raft_warped_ema_a0.50",
        "s2m2_s_confidence_occlusion_reset_a0.50",
        "s2m2_s_conservative_adaptive_ema",
        "stereoanyvideo",
    ]
    by_id = {method.method_id: method for method in methods}
    selected_indices = sorted({0, len(frames) // 2, min(100, len(frames) - 1), len(frames) - 1})
    for idx in selected_indices:
        frame = frames[idx]
        gt = np.load(frame.gt_disp_path, allow_pickle=False).astype(np.float32)
        valid = read_mask(frame.valid_mask_path)
        tiles: list[tuple[str, np.ndarray]] = [("RGB", read_rgb(frame.left_path)), ("GT disparity", colorize(gt))]
        for method_id in method_ids:
            method = by_id.get(method_id)
            if method is None:
                continue
            pred = method.predictions[idx]
            err = np.abs(pred - gt)
            err[~valid] = np.nan
            tiles.append((method.method_name, colorize(pred)))
            tiles.append((f"abs error {method.method_name}", colorize(err, vmin=0.0, vmax=5.0)))
        save_montage(output_dir / "qualitative" / f"frame_{frame.frame_id}.png", tiles)


def sampled_disparity_vmax(frames: Sequence[FrameRecord], methods: Sequence[MethodRecord]) -> float:
    values: list[np.ndarray] = []
    for idx in range(0, len(frames), max(len(frames) // 12, 1)):
        gt = np.load(frames[idx].gt_disp_path, allow_pickle=False).astype(np.float32)
        values.append(gt[::8, ::8].ravel())
        for method in methods:
            values.append(method.predictions[idx][::8, ::8].ravel())
    merged = np.concatenate([v[np.isfinite(v)] for v in values if v.size])
    return float(np.nanpercentile(merged, 99)) if merged.size else 1.0


def generate_qualitative_videos(
    output_dir: Path,
    frames: Sequence[FrameRecord],
    method_by_id: dict[str, MethodRecord],
    best_depth_id: str,
    best_mc_id: str,
    flow_cache_dir: Path,
    warp_device: str,
) -> list[Path]:
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    fixed_id = "s2m2_s_fixed_ema_a0.35"
    raw_id = "s2m2_s_raw"
    raft_id = "s2m2_s_raft_warped_ema_a0.50"
    sav_id = "stereoanyvideo"
    disparity_ids = [raw_id, fixed_id, best_depth_id, best_mc_id, raft_id, sav_id]
    disparity_ids = list(dict.fromkeys([mid for mid in disparity_ids if mid in method_by_id]))
    disparity_methods = [method_by_id[mid] for mid in disparity_ids]
    disp_vmax = sampled_disparity_vmax(frames, disparity_methods)
    paths: list[Path] = []

    def rgb_at(idx: int) -> np.ndarray:
        return read_rgb(frames[idx].left_path)

    def gt_at(idx: int) -> np.ndarray:
        return np.load(frames[idx].gt_disp_path, allow_pickle=False).astype(np.float32)

    def disparity_frames():
        for idx, frame in enumerate(frames):
            tiles = [("RGB", rgb_at(idx)), ("GT disparity", colorize_scalar(gt_at(idx), 0.0, disp_vmax))]
            for method in disparity_methods:
                tiles.append((method.method_name, colorize_scalar(method.predictions[idx], 0.0, disp_vmax)))
            yield make_board(tiles, panel_size=(320, 256), cols=4)

    path = videos_dir / "disparity_comparison_board.mp4"
    write_mp4(path, disparity_frames(), fps=10)
    paths.append(path)

    error_ids = [raw_id, fixed_id, best_depth_id, raft_id, sav_id]
    error_ids = list(dict.fromkeys([mid for mid in error_ids if mid in method_by_id]))
    error_methods = [method_by_id[mid] for mid in error_ids]

    def error_frames():
        for idx, frame in enumerate(frames):
            gt = gt_at(idx)
            valid = read_mask(frame.valid_mask_path)
            tiles = [("RGB", rgb_at(idx))]
            for method in error_methods:
                err = np.abs(method.predictions[idx] - gt)
                err[~valid] = np.nan
                tiles.append((f"{method.method_name} abs err", colorize_scalar(err, 0.0, 10.0)))
            yield make_board(tiles, panel_size=(320, 256), cols=3)

    path = videos_dir / "error_comparison_board.mp4"
    write_mp4(path, error_frames(), fps=10)
    paths.append(path)

    temporal_ids = [raw_id, fixed_id, best_mc_id, raft_id, sav_id]
    temporal_ids = list(dict.fromkeys([mid for mid in temporal_ids if mid in method_by_id]))
    temporal_methods = [method_by_id[mid] for mid in temporal_ids]

    def temporal_frames():
        for idx in range(1, len(frames)):
            tiles = [("RGB", rgb_at(idx))]
            for method in temporal_methods:
                diff = np.abs(method.predictions[idx] - method.predictions[idx - 1])
                tiles.append((f"{method.method_name} |dt|", colorize_scalar(diff, 0.0, 5.0)))
            yield make_board(tiles, panel_size=(320, 256), cols=3)

    path = videos_dir / "temporal_difference_board.mp4"
    write_mp4(path, temporal_frames(), fps=10)
    paths.append(path)

    def mc_temporal_frames():
        for idx in range(1, len(frames)):
            prev_id = frames[idx - 1].frame_id
            cur_id = frames[idx].frame_id
            flow = load_forward_flow(flow_cache_dir, prev_id, cur_id)
            tiles = [("RGB", rgb_at(idx))]
            for method in temporal_methods:
                warped_prev = warp_disparity_numpy(method.predictions[idx - 1], flow, device=warp_device)
                diff = np.abs(method.predictions[idx] - warped_prev)
                tiles.append((f"{method.method_name} |mc dt|", colorize_scalar(diff, 0.0, 5.0)))
            yield make_board(tiles, panel_size=(320, 256), cols=3)

    path = videos_dir / "motion_compensated_difference_board.mp4"
    write_mp4(path, mc_temporal_frames(), fps=10)
    paths.append(path)
    return paths


def format_best(row: dict[str, Any] | None) -> str:
    if row is None:
        return "none"
    return (
        f"{row['method_id']} | depth={float(row['depth_mae_mm']):.4f} mm | "
        f"disp={float(row['disp_mae_px']):.4f} px | "
        f"mc={float(row['motion_compensated_temporal_mae_px']):.4f} px | "
        f"online={float(row['online_runtime_estimated_ms']):.4f} ms"
    )


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    ensure_output_dir(args.output_dir, bool(args.overwrite))
    frames = load_frames(args.sequence_dir)
    metric_indices = [idx for idx, frame in enumerate(frames) if frame.valid_ratio >= args.min_valid_ratio]
    if not metric_indices:
        raise RuntimeError("No frames remain after min_valid_ratio filtering")
    gt_shape = np.load(frames[0].gt_disp_path, allow_pickle=False).shape
    audit = cache_audit(frames, args.s2m2_cache_dir, args.sav_cache_dir, args.flow_cache_dir, gt_shape)
    write_json(args.output_dir / "cache_audit.json", audit)
    if not audit["complete"]:
        raise RuntimeError(f"Cache audit failed; see {args.output_dir / 'cache_audit.json'}")

    s2m2_runtime, s2m2_vram = load_metadata_runtime(args.s2m2_cache_dir)
    sav_runtime, sav_vram = load_metadata_runtime(args.sav_cache_dir)
    flow_runtime = load_flow_runtime_metadata(args.flow_cache_dir)
    s2m2_raw = load_prediction_sequence(args.s2m2_cache_dir, frames)
    sav = load_prediction_sequence(args.sav_cache_dir, frames)
    core_methods = build_methods(
        frames,
        s2m2_raw,
        sav,
        args.flow_cache_dir,
        args.ema_alphas,
        s2m2_runtime,
        s2m2_vram,
        sav_runtime,
        sav_vram,
        args.warp_device,
    )

    summary_rows: list[dict[str, Any]] = []
    all_frame_rows: list[dict[str, Any]] = []
    all_pair_rows: list[dict[str, Any]] = []
    adaptive_sweep_rows: list[dict[str, Any]] = []
    adaptive_summary_rows: list[dict[str, Any]] = []
    adaptive_configs = no_raft_adaptive_configs()
    adaptive_config_by_id = {str(config["method_id"]): config for config in adaptive_configs}
    metric_pair_indices = [idx for idx in metric_indices if idx > 0 and idx - 1 in metric_indices]

    core_summaries, core_frame_rows, core_pair_rows = evaluate_methods_batch(
        core_methods,
        frames,
        metric_indices,
        metric_pair_indices,
        args.flow_cache_dir,
        flow_runtime,
    )
    summary_rows.extend(core_summaries)
    all_frame_rows.extend(core_frame_rows)
    all_pair_rows.extend(core_pair_rows)

    adaptive_batch_size = 8
    for batch_start in range(0, len(adaptive_configs), adaptive_batch_size):
        batch_configs = adaptive_configs[batch_start : batch_start + adaptive_batch_size]
        batch_methods = [
            build_no_raft_adaptive_method(config, s2m2_raw, s2m2_runtime, s2m2_vram)
            for config in batch_configs
        ]
        batch_summaries, batch_frame_rows, batch_pair_rows = evaluate_methods_batch(
            batch_methods,
            frames,
            metric_indices,
            metric_pair_indices,
            args.flow_cache_dir,
            flow_runtime,
        )
        summary_rows.extend(batch_summaries)
        adaptive_summary_rows.extend(batch_summaries)
        adaptive_sweep_rows.extend(
            adaptive_sweep_row(config, summary)
            for config, summary in zip(batch_configs, batch_summaries)
        )
        all_frame_rows.extend(batch_frame_rows)
        all_pair_rows.extend(batch_pair_rows)
        done = min(batch_start + len(batch_configs), len(adaptive_configs))
        print(f"evaluated_no_raft_adaptive_variants={done}/{len(adaptive_configs)}", flush=True)
        del batch_methods

    best_depth = best_summary(adaptive_summary_rows, "depth_mae_mm")
    best_disp = best_summary(adaptive_summary_rows, "disp_mae_px")
    best_mc = best_summary(adaptive_summary_rows, "motion_compensated_temporal_mae_px")
    best_under_70 = best_summary(adaptive_summary_rows, "depth_mae_mm", max_runtime_ms=70.0)
    selected_adaptive_ids = {
        row["method_id"]
        for row in [best_depth, best_disp, best_mc, best_under_70]
        if row is not None and row.get("method_id") in adaptive_config_by_id
    }
    selected_adaptive_methods = [
        build_no_raft_adaptive_method(adaptive_config_by_id[method_id], s2m2_raw, s2m2_runtime, s2m2_vram)
        for method_id in sorted(selected_adaptive_ids)
    ]
    method_by_id = {method.method_id: method for method in [*core_methods, *selected_adaptive_methods]}
    best_depth_id = str(best_depth["method_id"]) if best_depth is not None else "s2m2_s_fixed_ema_a0.35"
    best_mc_id = str(best_mc["method_id"]) if best_mc is not None else best_depth_id
    video_paths = generate_qualitative_videos(
        args.output_dir,
        frames,
        method_by_id,
        best_depth_id,
        best_mc_id,
        args.flow_cache_dir,
        args.warp_device,
    )

    write_csv(args.output_dir / "summary.csv", summary_rows, SUMMARY_COLUMNS)
    write_csv(args.output_dir / "per_frame_metrics.csv", all_frame_rows, PER_FRAME_COLUMNS)
    write_csv(args.output_dir / "per_pair_temporal_metrics.csv", all_pair_rows, PER_PAIR_COLUMNS)
    write_csv(args.output_dir / "adaptive_sweep_summary.csv", adaptive_sweep_rows, ADAPTIVE_SWEEP_COLUMNS)
    write_json(
        args.output_dir / "method_config.json",
        {
            "sequence_dir": str(args.sequence_dir),
            "s2m2_cache_dir": str(args.s2m2_cache_dir),
            "sav_cache_dir": str(args.sav_cache_dir),
            "flow_cache_dir": str(args.flow_cache_dir),
            "min_valid_ratio": args.min_valid_ratio,
            "ema_alphas": args.ema_alphas,
            "warp_device": args.warp_device,
            "metric_flow_source": str(args.flow_cache_dir),
            "flow_runtime_source": flow_runtime["source"],
            "average_forward_runtime_ms": flow_runtime["average_forward_runtime_ms"],
            "average_backward_runtime_ms": flow_runtime["average_backward_runtime_ms"],
            "raft_peak_vram_mb": flow_runtime["peak_vram_mb"],
            "no_raft_adaptive_variant_count": len(adaptive_configs),
            "best_no_raft_adaptive_by_depth_mae": best_depth,
            "best_no_raft_adaptive_by_disp_mae": best_disp,
            "best_no_raft_adaptive_by_motion_compensated_temporal_mae": best_mc,
            "best_no_raft_adaptive_under_70ms": best_under_70,
            "core_methods": [
                {
                    "method_id": method.method_id,
                    "method_name": method.method_name,
                    "method_type": method.method_type,
                    "alpha": method.alpha,
                    "role": method.role,
                    "notes": method.notes,
                    "params": method.params or {},
                }
                for method in core_methods
            ],
            "no_raft_adaptive_sweep_configs": adaptive_configs,
        },
    )
    write_qualitative(args.output_dir, frames, core_methods)
    elapsed = time.perf_counter() - start

    summary_by_id = {row["method_id"]: row for row in summary_rows}
    refs = {
        "S2M2-S raw": summary_by_id.get("s2m2_s_raw"),
        "fixed EMA alpha=0.35": summary_by_id.get("s2m2_s_fixed_ema_a0.35"),
        "RAFT-warped EMA alpha=0.50": summary_by_id.get("s2m2_s_raft_warped_ema_a0.50"),
        "StereoAnyVideo": summary_by_id.get("stereoanyvideo"),
    }
    ref_lines = "\n".join(f"- {name}: {format_best(row)}" for name, row in refs.items())
    readme = f"""# SCARED S2M2 Temporal Baselines v2 No-RAFT Adaptive

Cache-only benchmark on `{args.sequence_dir}`.

- S2M2-S cache: `{args.s2m2_cache_dir}`
- StereoAnyVideo cache: `{args.sav_cache_dir}`
- Metric and RAFT-comparison flow source: RAFT flow cache from `{args.flow_cache_dir}`
- Flow runtime source: `{flow_runtime['source']}`
- Average RAFT forward runtime: `{float(flow_runtime['average_forward_runtime_ms']):.4f} ms`
- Average RAFT backward runtime: `{float(flow_runtime['average_backward_runtime_ms']):.4f} ms`
- Minimum valid frame ratio: `{args.min_valid_ratio}`
- EMA alphas: `{','.join(f'{a:.2f}' for a in args.ema_alphas)}`
- No-RAFT adaptive variants evaluated: `{len(adaptive_configs)}`

No S2M2, StereoAnyVideo, RAFT, or training inference is launched by this benchmark. The new no-RAFT adaptive prediction formulas do not read optical flow; RAFT flow is used only for the motion-compensated metric and for the existing RAFT-based comparison methods.

## Best No-RAFT Adaptive Selections

- best_no_raft_adaptive_by_depth_mae: {format_best(best_depth)}
- best_no_raft_adaptive_by_disp_mae: {format_best(best_disp)}
- best_no_raft_adaptive_by_motion_compensated_temporal_mae: {format_best(best_mc)}
- best_no_raft_adaptive_under_70ms: {format_best(best_under_70)}

## Reference Comparisons

{ref_lines}

## Runtime Interpretation

Quality metrics remain cache-based. `runtime_postprocess_ms` measures deterministic postprocessing in this benchmark. For deployment comparison, use `online_runtime_estimated_ms`. No-RAFT adaptive methods use `runtime_fairness_category=no_flow_adaptive`, with `flow_runtime_used_ms=0`, so they are directly comparable to fixed EMA.

## Videos

- `videos/disparity_comparison_board.mp4`
- `videos/error_comparison_board.mp4`
- `videos/temporal_difference_board.mp4`
- `videos/motion_compensated_difference_board.mp4`
"""
    (args.output_dir / "README.md").write_text(readme)
    (args.output_dir / "run.log").write_text(
        "\n".join(
            [
                "Cache-only SCARED S2M2 no-RAFT adaptive temporal baseline benchmark",
                f"output_dir={args.output_dir}",
                f"num_frames={len(frames)}",
                f"metric_frame_count={len(metric_indices)}",
                f"metric_pair_count={len(metric_pair_indices)}",
                f"method_count={len(summary_rows)}",
                f"no_raft_adaptive_variant_count={len(adaptive_configs)}",
                f"cache_audit_complete={audit['complete']}",
                f"best_no_raft_adaptive_by_depth_mae={format_best(best_depth)}",
                f"best_no_raft_adaptive_by_disp_mae={format_best(best_disp)}",
                f"best_no_raft_adaptive_by_motion_compensated_temporal_mae={format_best(best_mc)}",
                f"best_no_raft_adaptive_under_70ms={format_best(best_under_70)}",
                f"videos={','.join(str(path) for path in video_paths)}",
                f"elapsed_seconds={elapsed:.3f}",
            ]
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "summary_csv": str(args.output_dir / "summary.csv"),
                "adaptive_sweep_summary_csv": str(args.output_dir / "adaptive_sweep_summary.csv"),
                "cache_audit_complete": audit["complete"],
                "method_count": len(summary_rows),
                "no_raft_adaptive_variant_count": len(adaptive_configs),
                "metric_frame_count": len(metric_indices),
                "metric_pair_count": len(metric_pair_indices),
                "best_no_raft_adaptive_by_depth_mae": best_depth,
                "best_no_raft_adaptive_by_disp_mae": best_disp,
                "best_no_raft_adaptive_by_motion_compensated_temporal_mae": best_mc,
                "best_no_raft_adaptive_under_70ms": best_under_70,
                "video_files": [str(path) for path in video_paths],
                "elapsed_seconds": elapsed,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
