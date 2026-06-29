#!/usr/bin/env python3
"""Streaming S2M2-S evaluation on rectified SCARED temporal-GT sequences.

This reuses the validated S2M2-S loading and inference path from
run_s2m2_temporal_gt_compat.py, but evaluates frame-by-frame without writing
prediction caches unless explicitly requested.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
S2M2_REPO = ROOT / "external/frame_stereo_repos/s2m2"
S2M2_SRC = S2M2_REPO / "src"
S2M2_WEIGHTS = S2M2_REPO / "weights/pretrain_weights"

DEFAULT_INPUT_ROOT = ROOT / "dataset/SCARED/curated/temporal_gt_rectified"
DEFAULT_AUDIT_FRAME_CSV = ROOT / "dataset/SCARED/curated/audit/temporal_gt_rectified_integrity/frame_integrity.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "results/03_temporal_refinement/evaluation/gt_temporal_rectified_streaming_s2m2"

FRAME_FIELDS = [
    "sequence_id",
    "frame_id",
    "valid_pixel_pct",
    "included",
    "exclusion_reason",
    "valid_pixel_count",
    "disparity_mae",
    "disparity_rmse",
    "bad_1px",
    "bad_2px",
    "bad_3px",
    "depth_mae",
    "runtime_ms",
    "warning_flags",
]

SEQUENCE_FIELDS = [
    "sequence_id",
    "num_frames_total",
    "frames_evaluated",
    "frames_skipped",
    "valid_pixel_pct_mean",
    "valid_pixel_pct_min",
    "valid_pixel_pct_max",
    "disparity_mae_mean",
    "disparity_mae_median",
    "disparity_rmse_mean",
    "bad_1px_mean",
    "bad_2px_mean",
    "bad_3px_mean",
    "depth_mae_mean",
    "depth_mae_median",
    "runtime_ms_mean",
    "runtime_ms_median",
    "warnings",
]


@dataclass(frozen=True)
class FrameRecord:
    sequence_id: str
    frame_id: str
    left_path: Path
    right_path: Path
    gt_depth_path: Path
    gt_disp_path: Path
    valid_mask_path: Path
    calibration_path: Path
    fx: float
    baseline: float
    manifest_valid_ratio: float


@dataclass
class EvalFrame:
    record: FrameRecord
    left_rgb: np.ndarray
    right_rgb: np.ndarray
    gt_disp: np.ndarray
    pred_disp: np.ndarray
    valid_mask: np.ndarray
    abs_disp_error: np.ndarray
    row: dict[str, Any]


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--audit-frame-csv", type=Path, default=DEFAULT_AUDIT_FRAME_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-gpus", type=int, choices=[1, 2], default=1, help="Number of GPU worker processes to use.")
    parser.add_argument("--gpu-ids", default="", help="Comma-separated physical GPU ids for multi-GPU mode. Defaults to 0,1.")
    parser.add_argument("--variant", choices=["S"], default="S")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--min-valid-ratio", type=float, default=0.05)
    parser.add_argument("--skip-suspicious", nargs="?", const=True, default=True, type=parse_bool)
    parser.add_argument("--safe-sequences-only", nargs="?", const=True, default=False, type=parse_bool)
    parser.add_argument("--limit-sequences", type=int, default=0)
    parser.add_argument("--limit-frames-per-sequence", type=int, default=0)
    parser.add_argument("--save-predictions", nargs="?", const=True, default=False, type=parse_bool)
    parser.add_argument("--diagnostics", nargs="?", const=True, default=True, type=parse_bool)
    parser.add_argument("--diagnostic-count", type=int, default=3)
    parser.add_argument("--overwrite", nargs="?", const=True, default=False, type=parse_bool)
    parser.add_argument("--worker-id", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--worker-count", type=int, default=1, help=argparse.SUPPRESS)
    return parser.parse_args()


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def mean(values: list[float]) -> float | None:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(finite)) if finite else None


def median(values: list[float]) -> float | None:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.median(finite)) if finite else None


def min_f(values: list[float]) -> float | None:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.min(finite)) if finite else None


def max_f(values: list[float]) -> float | None:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.max(finite)) if finite else None


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def resolve_manifest_path(sequence_root: Path, value: str | None, fallback: Path) -> Path:
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
    return sequence_root / path


def calibration_fx_baseline(calibration_path: Path) -> tuple[float, float]:
    calibration = json.loads(calibration_path.read_text())
    if "fx" in calibration and "baseline" in calibration:
        return float(calibration["fx"]), float(calibration["baseline"])
    if "fx" in calibration and "baseline_mm" in calibration:
        return float(calibration["fx"]), float(calibration["baseline_mm"])
    if "KL" in calibration and "T" in calibration:
        kl = np.asarray(calibration["KL"], dtype=np.float64)
        t = np.asarray(calibration["T"], dtype=np.float64).reshape(-1)
        return float(kl[0, 0]), float(np.linalg.norm(t))
    if "P1" in calibration and "P2" in calibration:
        p1 = np.asarray(calibration["P1"]["data"], dtype=np.float64).reshape(calibration["P1"]["rows"], calibration["P1"]["cols"])
        p2 = np.asarray(calibration["P2"]["data"], dtype=np.float64).reshape(calibration["P2"]["rows"], calibration["P2"]["cols"])
        return float(p1[0, 0]), float(abs(p2[0, 3] / p2[0, 0]))
    raise RuntimeError(f"Could not infer fx/baseline from {calibration_path}")


def discover_sequences(input_root: Path, safe_sequences: set[str] | None, limit_sequences: int) -> list[Path]:
    sequences = sorted(path for path in input_root.iterdir() if path.is_dir() and (path / "metadata.csv").exists())
    if safe_sequences is not None:
        sequences = [path for path in sequences if path.name in safe_sequences]
    if limit_sequences > 0:
        sequences = sequences[:limit_sequences]
    return sequences


def shard_sequences(sequences: list[Path], worker_id: int, worker_count: int) -> list[Path]:
    if worker_count <= 1:
        return sequences
    if worker_id < 0 or worker_id >= worker_count:
        raise ValueError(f"Invalid worker shard {worker_id}/{worker_count}")
    return [path for index, path in enumerate(sequences) if index % worker_count == worker_id]


def read_frames(sequence_root: Path, limit_frames: int = 0) -> list[FrameRecord]:
    metadata_path = sequence_root / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    with metadata_path.open() as f:
        rows = list(csv.DictReader(f))
    if limit_frames > 0:
        rows = rows[:limit_frames]

    frames: list[FrameRecord] = []
    for row in rows:
        frame_id = row.get("frame_index") or row.get("frame_id") or row.get("id")
        if not frame_id:
            raise RuntimeError(f"Missing frame id in metadata row: {row}")

        fallback_valid = sequence_root / "gt" / "ValidMask" / f"{frame_id}.npy"
        if not fallback_valid.exists():
            fallback_valid = sequence_root / "gt" / "ValidMask" / f"{frame_id}.png"
        left_path = resolve_manifest_path(sequence_root, row.get("left_path"), sequence_root / "left" / f"{frame_id}.png")
        right_path = resolve_manifest_path(sequence_root, row.get("right_path"), sequence_root / "right" / f"{frame_id}.png")
        gt_depth_path = resolve_manifest_path(
            sequence_root,
            row.get("depth_path") or row.get("depth_float32_path"),
            sequence_root / "gt" / "DepthL_float32" / f"{frame_id}.npy",
        )
        gt_disp_path = resolve_manifest_path(
            sequence_root,
            row.get("disparity_path") or row.get("disparity_float32_path"),
            sequence_root / "gt" / "Disparity_float32" / f"{frame_id}.npy",
        )
        valid_mask_path = resolve_manifest_path(sequence_root, row.get("valid_mask_path"), fallback_valid)
        calibration_path = resolve_manifest_path(sequence_root, row.get("calibration_path"), sequence_root / "calibration" / f"{frame_id}.json")
        for path in [left_path, right_path, gt_depth_path, gt_disp_path, valid_mask_path, calibration_path]:
            if not path.exists():
                raise FileNotFoundError(path)
        fx, baseline = calibration_fx_baseline(calibration_path)
        frames.append(
            FrameRecord(
                sequence_id=row.get("sequence_id") or sequence_root.name,
                frame_id=frame_id,
                left_path=left_path,
                right_path=right_path,
                gt_depth_path=gt_depth_path,
                gt_disp_path=gt_disp_path,
                valid_mask_path=valid_mask_path,
                calibration_path=calibration_path,
                fx=fx,
                baseline=baseline,
                manifest_valid_ratio=finite_float(row.get("valid_pixel_ratio")),
            )
        )
    return frames


def read_audit_frames(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {(row["sequence_id"], row["frame_id"]): row for row in csv.DictReader(f)}


def read_safe_sequences(audit_frame_csv: Path) -> set[str] | None:
    summary_path = audit_frame_csv.parent / "audit_summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    safe = summary.get("safe_sequences_for_evaluation")
    return set(map(str, safe)) if safe else None


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path).astype(bool)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Cannot read mask: {path}")
    return mask > 0


def build_s2m2_s(device: torch.device):
    sys.path.insert(0, str(S2M2_REPO.resolve()))
    sys.path.insert(0, str(S2M2_SRC.resolve()))
    from s2m2.core.model.s2m2 import S2M2

    model = S2M2(feature_channels=128, dim_expansion=1, num_transformer=1, use_positivity=True, refine_iter=3)
    ckpt = torch.load(S2M2_WEIGHTS / "CH128NTR1.pth", map_location="cpu", weights_only=True)
    model.my_load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def infer_frame(model, left: np.ndarray, right: np.ndarray, width: int, device: torch.device) -> tuple[np.ndarray, float, float]:
    sys.path.insert(0, str(S2M2_REPO.resolve()))
    sys.path.insert(0, str(S2M2_SRC.resolve()))
    from s2m2.core.utils.image_utils import image_crop, image_pad

    orig_h, orig_w = left.shape[:2]
    scale_x = width / float(orig_w)
    new_h = int(round(orig_h * scale_x))
    left_in = cv2.resize(left, (width, new_h), interpolation=cv2.INTER_LINEAR)
    right_in = cv2.resize(right, (width, new_h), interpolation=cv2.INTER_LINEAR)
    left_t = torch.from_numpy(left_in).permute(2, 0, 1).unsqueeze(0).float().to(device)
    right_t = torch.from_numpy(right_in).permute(2, 0, 1).unsqueeze(0).float().to(device)
    h, w = left_t.shape[-2:]
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        pred, _occ, _conf = model(image_pad(left_t, 32), image_pad(right_t, 32))
    if device.type == "cuda":
        torch.cuda.synchronize()
    runtime_ms = (time.perf_counter() - t0) * 1000.0
    pred_np = image_crop(pred, (h, w)).squeeze().float().cpu().numpy()
    pred_np = cv2.resize(pred_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) / scale_x
    return np.clip(pred_np.astype(np.float32), 0.0, None), runtime_ms, scale_x


def metric_row(
    frame: FrameRecord,
    gt_disp: np.ndarray,
    gt_depth: np.ndarray,
    valid_mask: np.ndarray,
    pred_disp: np.ndarray,
    runtime_ms: float,
    warning_flags: str,
) -> tuple[dict[str, Any], np.ndarray]:
    if pred_disp.shape != gt_disp.shape:
        pred_disp = cv2.resize(pred_disp, (gt_disp.shape[1], gt_disp.shape[0]), interpolation=cv2.INTER_LINEAR)
        warning_flags = ";".join(filter(None, [warning_flags, "prediction_shape_resized_for_metrics"]))

    valid = valid_mask & np.isfinite(gt_disp) & np.isfinite(gt_depth) & np.isfinite(pred_disp) & (gt_disp > 0) & (gt_depth > 0) & (pred_disp > 0.1)
    valid_pixel_pct = float(np.mean(valid_mask)) if valid_mask.size else math.nan
    abs_disp_error = np.zeros_like(gt_disp, dtype=np.float32)
    if valid.any():
        disp_err = np.abs(pred_disp[valid] - gt_disp[valid]).astype(np.float64)
        pred_depth = frame.fx * frame.baseline / np.maximum(pred_disp, 1e-6)
        depth_err = np.abs(pred_depth[valid] - gt_depth[valid]).astype(np.float64)
        abs_disp_error[valid] = disp_err.astype(np.float32)
        row = {
            "sequence_id": frame.sequence_id,
            "frame_id": frame.frame_id,
            "valid_pixel_pct": valid_pixel_pct,
            "included": True,
            "exclusion_reason": "",
            "valid_pixel_count": int(valid.sum()),
            "disparity_mae": float(np.mean(disp_err)),
            "disparity_rmse": float(np.sqrt(np.mean(disp_err**2))),
            "bad_1px": float(np.mean(disp_err > 1.0) * 100.0),
            "bad_2px": float(np.mean(disp_err > 2.0) * 100.0),
            "bad_3px": float(np.mean(disp_err > 3.0) * 100.0),
            "depth_mae": float(np.mean(depth_err)),
            "runtime_ms": runtime_ms,
            "warning_flags": warning_flags,
        }
    else:
        row = skipped_frame_row(frame, valid_pixel_pct, "no_metric_valid_pixels", warning_flags)
        row["runtime_ms"] = runtime_ms
    return row, abs_disp_error


def skipped_frame_row(frame: FrameRecord, valid_pixel_pct: float, reason: str, warning_flags: str) -> dict[str, Any]:
    return {
        "sequence_id": frame.sequence_id,
        "frame_id": frame.frame_id,
        "valid_pixel_pct": valid_pixel_pct,
        "included": False,
        "exclusion_reason": reason,
        "valid_pixel_count": 0,
        "disparity_mae": None,
        "disparity_rmse": None,
        "bad_1px": None,
        "bad_2px": None,
        "bad_3px": None,
        "depth_mae": None,
        "runtime_ms": None,
        "warning_flags": warning_flags,
    }


def scalar_preview(values: np.ndarray, valid: np.ndarray | None, vmax: float, cmap: int = cv2.COLORMAP_TURBO) -> np.ndarray:
    arr = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    normalized = np.clip(arr / max(vmax, 1e-6), 0.0, 1.0)
    preview = cv2.applyColorMap((normalized * 255).astype(np.uint8), cmap)
    if valid is not None:
        preview[~valid] = 0
    return preview


def label_tile(tile_bgr: np.ndarray, label: str, size: tuple[int, int] = (260, 208)) -> np.ndarray:
    tile = cv2.resize(tile_bgr, size, interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (tile.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(tile, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def write_contact_sheet(eval_frame: EvalFrame, path: Path) -> None:
    frame = eval_frame.record
    valid = eval_frame.valid_mask & np.isfinite(eval_frame.gt_disp) & (eval_frame.gt_disp > 0)
    pred = eval_frame.pred_disp
    gt = eval_frame.gt_disp
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
    if valid.any():
        vmax = float(np.nanpercentile(np.concatenate([gt[valid].ravel(), pred[valid].ravel()]), 99))
    else:
        vmax = 120.0
    mask_bgr = cv2.cvtColor((valid.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)
    tiles = [
        label_tile(cv2.cvtColor(eval_frame.left_rgb, cv2.COLOR_RGB2BGR), f"{frame.frame_id} left RGB"),
        label_tile(cv2.cvtColor(eval_frame.right_rgb, cv2.COLOR_RGB2BGR), "right RGB"),
        label_tile(scalar_preview(gt, valid, vmax), "GT disparity"),
        label_tile(scalar_preview(pred, valid, vmax), "S2M2 pred disparity"),
        label_tile(scalar_preview(eval_frame.abs_disp_error, valid, 20.0, cv2.COLORMAP_MAGMA), "abs disp error"),
        label_tile(mask_bgr, "valid mask"),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.concatenate(tiles, axis=1))


def select_diagnostics(evaluated: list[EvalFrame], count: int) -> list[tuple[str, EvalFrame]]:
    if not evaluated or count <= 0:
        return []
    if count == 1 or len(evaluated) == 1:
        return [("first", evaluated[0])]
    if count == 2:
        indices = [0, len(evaluated) - 1]
        labels = ["first", "last"]
    else:
        indices = np.linspace(0, len(evaluated) - 1, min(count, len(evaluated)), dtype=int).tolist()
        labels = ["first"] + [f"middle{i}" for i in range(1, len(indices) - 1)] + ["last"]
        if len(indices) == 3:
            labels = ["first", "middle", "last"]
    seen: set[int] = set()
    out: list[tuple[str, EvalFrame]] = []
    for label, idx in zip(labels, indices):
        if idx in seen:
            continue
        seen.add(idx)
        out.append((label, evaluated[idx]))
    return out


def frame_should_skip(
    frame: FrameRecord,
    audit: dict[tuple[str, str], dict[str, str]],
    skip_suspicious: bool,
    min_valid_ratio: float,
) -> tuple[bool, str, float, str]:
    audit_row = audit.get((frame.sequence_id, frame.frame_id), {})
    audit_flags = str(audit_row.get("flags") or "")
    audit_valid = finite_float(audit_row.get("valid_pixel_pct"), frame.manifest_valid_ratio)
    valid_pixel_pct = audit_valid if math.isfinite(audit_valid) else frame.manifest_valid_ratio
    reasons: list[str] = []
    if skip_suspicious and audit_flags:
        reasons.append("audit_flags")
    if math.isfinite(valid_pixel_pct) and valid_pixel_pct < min_valid_ratio:
        reasons.append("low_valid_ratio")
    return bool(reasons), ";".join(reasons), valid_pixel_pct, audit_flags


def evaluate_sequence(
    sequence_root: Path,
    args: argparse.Namespace,
    audit: dict[tuple[str, str], dict[str, str]],
    model: Any,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frames = read_frames(sequence_root, args.limit_frames_per_sequence)
    frame_rows: list[dict[str, Any]] = []
    evaluated_for_diag: list[EvalFrame] = []
    prediction_dir = args.output_root / "predictions" / sequence_root.name if args.save_predictions else None
    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{now()}] evaluating {sequence_root.name}: {len(frames)} candidate frames", flush=True)
    for idx, frame in enumerate(frames, start=1):
        skip, reason, valid_pixel_pct, audit_flags = frame_should_skip(
            frame,
            audit,
            bool(args.skip_suspicious),
            args.min_valid_ratio,
        )
        if skip:
            frame_rows.append(skipped_frame_row(frame, valid_pixel_pct, reason, audit_flags))
            continue

        left_rgb = read_rgb(frame.left_path)
        right_rgb = read_rgb(frame.right_path)
        gt_disp = np.load(frame.gt_disp_path).astype(np.float32)
        gt_depth = np.load(frame.gt_depth_path).astype(np.float32)
        valid_mask = read_mask(frame.valid_mask_path)
        pred_disp, runtime_ms, _scale_x = infer_frame(model, left_rgb, right_rgb, args.width, device)
        if prediction_dir is not None:
            np.save(prediction_dir / f"{frame.frame_id}.npy", pred_disp.astype(np.float32))
        row, abs_disp_error = metric_row(frame, gt_disp, gt_depth, valid_mask, pred_disp, runtime_ms, audit_flags)
        if not row["included"]:
            row["exclusion_reason"] = ";".join(filter(None, [row["exclusion_reason"], "post_inference_metric_filter"]))
        frame_rows.append(row)
        if bool(args.diagnostics):
            evaluated_for_diag.append(
                EvalFrame(
                    record=frame,
                    left_rgb=left_rgb,
                    right_rgb=right_rgb,
                    gt_disp=gt_disp,
                    pred_disp=pred_disp,
                    valid_mask=valid_mask,
                    abs_disp_error=abs_disp_error,
                    row=row,
                )
            )
        if idx == len(frames) or idx % 50 == 0:
            evaluated = sum(bool(row["included"]) for row in frame_rows)
            print(f"[{now()}] {sequence_root.name}: {idx}/{len(frames)} scanned, {evaluated} evaluated", flush=True)

    if bool(args.diagnostics):
        diag_dir = args.output_root / "diagnostics" / sequence_root.name
        for label, eval_frame in select_diagnostics(evaluated_for_diag, args.diagnostic_count):
            write_contact_sheet(eval_frame, diag_dir / f"{label}_{eval_frame.record.frame_id}_contact_sheet.png")

    seq_row = summarize_sequence(sequence_root.name, frame_rows, frames)
    return frame_rows, seq_row


def summarize_sequence(sequence_id: str, frame_rows: list[dict[str, Any]], frames: list[FrameRecord]) -> dict[str, Any]:
    included = [row for row in frame_rows if row["included"]]
    warnings = sorted({str(row["warning_flags"]) for row in frame_rows if row.get("warning_flags")})
    return {
        "sequence_id": sequence_id,
        "num_frames_total": len(frames),
        "frames_evaluated": len(included),
        "frames_skipped": len(frame_rows) - len(included),
        "valid_pixel_pct_mean": mean([finite_float(row["valid_pixel_pct"]) for row in frame_rows]),
        "valid_pixel_pct_min": min_f([finite_float(row["valid_pixel_pct"]) for row in frame_rows]),
        "valid_pixel_pct_max": max_f([finite_float(row["valid_pixel_pct"]) for row in frame_rows]),
        "disparity_mae_mean": mean([finite_float(row["disparity_mae"]) for row in included]),
        "disparity_mae_median": median([finite_float(row["disparity_mae"]) for row in included]),
        "disparity_rmse_mean": mean([finite_float(row["disparity_rmse"]) for row in included]),
        "bad_1px_mean": mean([finite_float(row["bad_1px"]) for row in included]),
        "bad_2px_mean": mean([finite_float(row["bad_2px"]) for row in included]),
        "bad_3px_mean": mean([finite_float(row["bad_3px"]) for row in included]),
        "depth_mae_mean": mean([finite_float(row["depth_mae"]) for row in included]),
        "depth_mae_median": median([finite_float(row["depth_mae"]) for row in included]),
        "runtime_ms_mean": mean([finite_float(row["runtime_ms"]) for row in included]),
        "runtime_ms_median": median([finite_float(row["runtime_ms"]) for row in included]),
        "warnings": ";".join(warnings),
    }


def weighted_frame_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = []
    weights = []
    for row in rows:
        if not truthy(row["included"]):
            continue
        value = finite_float(row.get(key))
        weight = finite_float(row.get("valid_pixel_count"), 0.0)
        if math.isfinite(value) and weight > 0:
            values.append(value)
            weights.append(weight)
    return float(np.average(values, weights=weights)) if values else None


def aggregate_summary(
    args: argparse.Namespace,
    frame_rows: list[dict[str, Any]],
    sequence_rows: list[dict[str, Any]],
    start_time: float,
    peak_vram_mb: float,
) -> dict[str, Any]:
    included_frames = [row for row in frame_rows if truthy(row["included"])]
    evaluated_sequences = [row for row in sequence_rows if int(row["frames_evaluated"]) > 0]
    total_frames = len(frame_rows)
    total_frames_evaluated = len(included_frames)
    estimated_cache_bytes = int(sum(1024 * 1280 * 4 for _ in included_frames)) if not bool(args.save_predictions) else 0
    return {
        "generated_at": now(),
        "input_root": str(args.input_root),
        "audit_frame_csv": str(args.audit_frame_csv),
        "output_root": str(args.output_root),
        "model": "S2M2-S",
        "variant": args.variant,
        "resize_width": args.width,
        "device_requested": args.device,
        "skip_suspicious": bool(args.skip_suspicious),
        "min_valid_ratio": args.min_valid_ratio,
        "safe_sequences_only": bool(args.safe_sequences_only),
        "save_predictions": bool(args.save_predictions),
        "total_sequences": len(sequence_rows),
        "total_frames": total_frames,
        "total_frames_evaluated": total_frames_evaluated,
        "total_frames_skipped": total_frames - total_frames_evaluated,
        "weighted_metrics_over_frames": {
            "disparity_mae": weighted_frame_metric(frame_rows, "disparity_mae"),
            "disparity_rmse": weighted_frame_metric(frame_rows, "disparity_rmse"),
            "bad_1px": weighted_frame_metric(frame_rows, "bad_1px"),
            "bad_2px": weighted_frame_metric(frame_rows, "bad_2px"),
            "bad_3px": weighted_frame_metric(frame_rows, "bad_3px"),
            "depth_mae": weighted_frame_metric(frame_rows, "depth_mae"),
        },
        "unweighted_mean_over_sequences": {
            "disparity_mae": mean([finite_float(row["disparity_mae_mean"]) for row in evaluated_sequences]),
            "disparity_rmse": mean([finite_float(row["disparity_rmse_mean"]) for row in evaluated_sequences]),
            "bad_1px": mean([finite_float(row["bad_1px_mean"]) for row in evaluated_sequences]),
            "bad_2px": mean([finite_float(row["bad_2px_mean"]) for row in evaluated_sequences]),
            "bad_3px": mean([finite_float(row["bad_3px_mean"]) for row in evaluated_sequences]),
            "depth_mae": mean([finite_float(row["depth_mae_mean"]) for row in evaluated_sequences]),
        },
        "total_runtime_sec": float(time.perf_counter() - start_time),
        "median_runtime_ms_per_frame": median([finite_float(row["runtime_ms"]) for row in included_frames]),
        "peak_vram_mb": peak_vram_mb,
        "estimated_prediction_cache_storage_saved_bytes": estimated_cache_bytes,
        "estimated_prediction_cache_storage_saved_gib": float(estimated_cache_bytes / (1024**3)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_readme(path: Path, summary: dict[str, Any]) -> None:
    weighted = summary["weighted_metrics_over_frames"]
    lines = [
        "# Streaming S2M2-S Rectified Temporal-GT Evaluation",
        "",
        "This run evaluates S2M2-S frame-by-frame on rectified SCARED temporal-GT and discards predictions by default.",
        "No RAFT, StereoAnyVideo, ConvGRU, temporal smoothing, oracle selection, or optical flow is run by this script.",
        "",
        f"- Input root: `{summary['input_root']}`",
        f"- Audit frame CSV: `{summary['audit_frame_csv']}`",
        f"- Sequences: `{summary['total_sequences']}`",
        f"- Frames: `{summary['total_frames']}`",
        f"- Evaluated frames: `{summary['total_frames_evaluated']}`",
        f"- Skipped frames: `{summary['total_frames_skipped']}`",
        f"- Resize width: `{summary['resize_width']}`",
        f"- Skip suspicious: `{summary['skip_suspicious']}`",
        f"- Minimum valid ratio: `{summary['min_valid_ratio']}`",
        f"- Saved predictions: `{summary['save_predictions']}`",
        f"- Disparity MAE weighted: `{weighted['disparity_mae']}`",
        f"- Disparity RMSE weighted: `{weighted['disparity_rmse']}`",
        f"- Bad-1px weighted pct: `{weighted['bad_1px']}`",
        f"- Bad-2px weighted pct: `{weighted['bad_2px']}`",
        f"- Bad-3px weighted pct: `{weighted['bad_3px']}`",
        f"- Depth MAE weighted: `{weighted['depth_mae']}`",
        f"- Median runtime per evaluated frame ms: `{summary['median_runtime_ms_per_frame']}`",
        f"- Peak VRAM MB: `{summary['peak_vram_mb']}`",
        f"- Estimated cache storage saved GiB: `{summary['estimated_prediction_cache_storage_saved_gib']}`",
        "",
        "Outputs:",
        "",
        "- `frame_metrics.csv`: per-frame include/skip status and metrics.",
        "- `sequence_metrics.csv`: per-sequence aggregate metrics.",
        "- `aggregate_summary.json`: machine-readable aggregate summary.",
        "- `diagnostics/<sequence_id>/`: compact contact sheets for selected evaluated frames.",
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_gpu_ids(value: str, num_gpus: int) -> list[str]:
    if value.strip():
        gpu_ids = [part.strip() for part in value.split(",") if part.strip()]
    else:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        visible_ids = [part.strip() for part in visible.split(",") if part.strip()]
        gpu_ids = visible_ids if len(visible_ids) >= num_gpus else [str(index) for index in range(num_gpus)]
    if len(gpu_ids) < num_gpus:
        raise ValueError(f"--num-gpus {num_gpus} requires at least {num_gpus} ids in --gpu-ids or CUDA_VISIBLE_DEVICES")
    return gpu_ids[:num_gpus]


def build_worker_command(args: argparse.Namespace, worker_id: int, worker_count: int, worker_output: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--input-root", str(args.input_root),
        "--audit-frame-csv", str(args.audit_frame_csv),
        "--output-root", str(worker_output),
        "--device", args.device,
        "--num-gpus", "1",
        "--variant", args.variant,
        "--width", str(args.width),
        "--min-valid-ratio", str(args.min_valid_ratio),
        "--skip-suspicious", str(bool(args.skip_suspicious)).lower(),
        "--safe-sequences-only", str(bool(args.safe_sequences_only)).lower(),
        "--limit-sequences", str(args.limit_sequences),
        "--limit-frames-per-sequence", str(args.limit_frames_per_sequence),
        "--save-predictions", str(bool(args.save_predictions)).lower(),
        "--diagnostics", str(bool(args.diagnostics)).lower(),
        "--diagnostic-count", str(args.diagnostic_count),
        "--overwrite", "true",
        "--worker-id", str(worker_id),
        "--worker-count", str(worker_count),
    ]


def run_multi_gpu(args: argparse.Namespace) -> int:
    prepare_output(args.output_root, bool(args.overwrite))
    start_time = time.perf_counter()
    gpu_ids = parse_gpu_ids(args.gpu_ids, args.num_gpus)
    worker_root = args.output_root / "workers"
    worker_root.mkdir(parents=True, exist_ok=True)
    run_log = [
        f"[{now()}] evaluate_s2m2_streaming_temporal_gt_rectified.py multi-gpu coordinator",
        f"output_root={args.output_root}",
        f"num_gpus={args.num_gpus}",
        f"gpu_ids={','.join(gpu_ids)}",
    ]
    (args.output_root / "run.log").write_text("\n".join(run_log) + "\n")

    processes: list[tuple[int, Path, Path, subprocess.Popen[str]]] = []
    for worker_id, gpu_id in enumerate(gpu_ids):
        worker_output = worker_root / f"worker_{worker_id}"
        log_path = args.output_root / f"worker_{worker_id}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        command = build_worker_command(args, worker_id, args.num_gpus, worker_output)
        log_file = log_path.open("w")
        process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, text=True, env=env, cwd=ROOT)
        log_file.close()
        processes.append((worker_id, worker_output, log_path, process))
        run_log.append(f"[{now()}] worker_start id={worker_id} gpu={gpu_id} pid={process.pid} log={log_path}")
        (args.output_root / "run.log").write_text("\n".join(run_log) + "\n")

    failed: list[tuple[int, int, Path]] = []
    for worker_id, _worker_output, log_path, process in processes:
        return_code = process.wait()
        run_log.append(f"[{now()}] worker_done id={worker_id} return_code={return_code}")
        (args.output_root / "run.log").write_text("\n".join(run_log) + "\n")
        if return_code != 0:
            failed.append((worker_id, return_code, log_path))
    if failed:
        details = ", ".join(f"worker {wid} rc={rc} log={log}" for wid, rc, log in failed)
        raise RuntimeError(f"One or more GPU workers failed: {details}")

    frame_rows: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    peak_vram_values: list[float] = []
    for _worker_id, worker_output, _log_path, _process in processes:
        frame_rows.extend(read_csv_rows(worker_output / "frame_metrics.csv"))
        sequence_rows.extend(read_csv_rows(worker_output / "sequence_metrics.csv"))
        summary_path = worker_output / "aggregate_summary.json"
        if summary_path.exists():
            worker_summary = json.loads(summary_path.read_text())
            peak_vram_values.append(finite_float(worker_summary.get("peak_vram_mb"), 0.0))
        diagnostics_dir = worker_output / "diagnostics"
        if diagnostics_dir.exists():
            shutil.copytree(diagnostics_dir, args.output_root / "diagnostics", dirs_exist_ok=True)
        predictions_dir = worker_output / "predictions"
        if predictions_dir.exists():
            shutil.copytree(predictions_dir, args.output_root / "predictions", dirs_exist_ok=True)

    frame_rows.sort(key=lambda row: (row.get("sequence_id", ""), row.get("frame_id", "")))
    sequence_rows.sort(key=lambda row: row.get("sequence_id", ""))
    summary = aggregate_summary(args, frame_rows, sequence_rows, start_time, max(peak_vram_values) if peak_vram_values else 0.0)
    summary["num_gpus"] = args.num_gpus
    summary["gpu_ids"] = gpu_ids
    summary["worker_outputs"] = [str(worker_output) for _worker_id, worker_output, _log_path, _process in processes]

    write_csv(args.output_root / "frame_metrics.csv", frame_rows, FRAME_FIELDS)
    write_csv(args.output_root / "sequence_metrics.csv", sequence_rows, SEQUENCE_FIELDS)
    (args.output_root / "aggregate_summary.json").write_text(json.dumps(summary, indent=2, default=json_default) + "\n")
    write_readme(args.output_root / "README.md", summary)
    run_log.extend(
        [
            f"[{now()}] merged_workers={len(processes)}",
            f"[{now()}] wrote frame_metrics.csv",
            f"[{now()}] wrote sequence_metrics.csv",
            f"[{now()}] wrote aggregate_summary.json",
            f"[{now()}] wrote README.md",
            f"[{now()}] complete",
        ]
    )
    (args.output_root / "run.log").write_text("\n".join(run_log) + "\n")
    print(json.dumps(summary, indent=2, default=json_default), flush=True)
    return 0


def prepare_output(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {output_root}. Pass --overwrite true to replace it.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def main() -> int:
    args = parse_args()
    if args.limit_sequences < 0 or args.limit_frames_per_sequence < 0:
        raise ValueError("limits must be >= 0")
    if args.diagnostic_count < 0:
        raise ValueError("--diagnostic-count must be >= 0")
    if args.worker_count < 1:
        raise ValueError("--worker-count must be >= 1")
    if args.num_gpus > 1 and args.worker_id < 0:
        return run_multi_gpu(args)
    prepare_output(args.output_root, bool(args.overwrite))

    run_log: list[str] = [
        f"[{now()}] evaluate_s2m2_streaming_temporal_gt_rectified.py",
        f"input_root={args.input_root}",
        f"audit_frame_csv={args.audit_frame_csv}",
        f"output_root={args.output_root}",
        f"variant={args.variant}",
        f"width={args.width}",
        f"skip_suspicious={bool(args.skip_suspicious)}",
        f"min_valid_ratio={args.min_valid_ratio}",
        f"save_predictions={bool(args.save_predictions)}",
        f"diagnostics={bool(args.diagnostics)}",
        f"num_gpus={args.num_gpus}",
        f"worker_id={args.worker_id}",
        f"worker_count={args.worker_count}",
    ]
    (args.output_root / "run.log").write_text("\n".join(run_log) + "\n")

    audit = read_audit_frames(args.audit_frame_csv)
    safe_sequences = read_safe_sequences(args.audit_frame_csv) if bool(args.safe_sequences_only) else None
    sequences = discover_sequences(args.input_root, safe_sequences, args.limit_sequences)
    if args.worker_id >= 0:
        sequences = shard_sequences(sequences, args.worker_id, args.worker_count)
    run_log.append(f"[{now()}] discovered_sequences={len(sequences)}")
    (args.output_root / "run.log").write_text("\n".join(run_log) + "\n")
    if not sequences:
        raise RuntimeError("No sequences selected for evaluation.")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    run_log.append(f"[{now()}] resolved_device={device}")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = build_s2m2_s(device)

    start_time = time.perf_counter()
    all_frame_rows: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    for sequence_root in sequences:
        run_log.append(f"[{now()}] sequence_start={sequence_root.name}")
        (args.output_root / "run.log").write_text("\n".join(run_log) + "\n")
        frame_rows, sequence_row = evaluate_sequence(sequence_root, args, audit, model, device)
        all_frame_rows.extend(frame_rows)
        sequence_rows.append(sequence_row)
        run_log.append(
            f"[{now()}] sequence_done={sequence_root.name} evaluated={sequence_row['frames_evaluated']} skipped={sequence_row['frames_skipped']}"
        )
        (args.output_root / "run.log").write_text("\n".join(run_log) + "\n")

    peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024**2)) if device.type == "cuda" else 0.0
    summary = aggregate_summary(args, all_frame_rows, sequence_rows, start_time, peak_vram_mb)

    write_csv(args.output_root / "frame_metrics.csv", all_frame_rows, FRAME_FIELDS)
    write_csv(args.output_root / "sequence_metrics.csv", sequence_rows, SEQUENCE_FIELDS)
    (args.output_root / "aggregate_summary.json").write_text(json.dumps(summary, indent=2, default=json_default) + "\n")
    write_readme(args.output_root / "README.md", summary)
    run_log.extend(
        [
            f"[{now()}] wrote frame_metrics.csv",
            f"[{now()}] wrote sequence_metrics.csv",
            f"[{now()}] wrote aggregate_summary.json",
            f"[{now()}] wrote README.md",
            f"[{now()}] complete",
        ]
    )
    (args.output_root / "run.log").write_text("\n".join(run_log) + "\n")
    print(json.dumps(summary, indent=2, default=json_default), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
