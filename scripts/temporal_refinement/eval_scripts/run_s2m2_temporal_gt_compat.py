#!/usr/bin/env python3
"""Run a focused S2M2-S compatibility check on one curated temporal-GT sequence.

This script is intentionally narrow: it runs only S2M2-S at a resize width of
512, writes one disparity prediction per frame, then checks that the predictions
line up with the curated temporal-GT depth/disparity/mask files.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
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
DEFAULT_SEQUENCE_ROOT = ROOT / "dataset/SCARED/curated/temporal_gt/dataset_1_keyframe_1"
DEFAULT_OUT_DIR = ROOT / "results/03_temporal_refinement/evaluation/gt_temporal_dataset_1_keyframe_1"


@dataclass(frozen=True)
class FrameRecord:
    frame_id: str
    left_path: Path
    right_path: Path
    gt_depth_path: Path
    gt_disp_path: Path
    valid_mask_path: Path
    calibration_path: Path
    fx: float
    baseline: float


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
    parser = argparse.ArgumentParser(description="Run S2M2-S@512 on a temporal-GT sequence and check compatibility.")
    parser.add_argument("--sequence-root", type=Path, default=DEFAULT_SEQUENCE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--variant", choices=["S"], default="S")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", nargs="?", const=True, default=False, type=parse_bool)
    parser.add_argument("--max-frames", type=int, default=0, help="0 processes all frames.")
    return parser.parse_args()


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


def read_frames(sequence_root: Path, max_frames: int = 0) -> list[FrameRecord]:
    metadata_path = sequence_root / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    with metadata_path.open() as f:
        rows = list(csv.DictReader(f))
    if max_frames > 0:
        rows = rows[:max_frames]
    frames: list[FrameRecord] = []
    for row in rows:
        frame_id = row.get("frame_index") or row.get("frame_id") or row.get("id")
        if not frame_id:
            raise RuntimeError(f"Missing frame id in metadata row: {row}")
        left_path = resolve_manifest_path(sequence_root, row.get("left_path"), sequence_root / "left" / f"{frame_id}.png")
        right_path = resolve_manifest_path(sequence_root, row.get("right_path"), sequence_root / "right" / f"{frame_id}.png")
        gt_depth_path = resolve_manifest_path(
            sequence_root,
            row.get("depth_path") or row.get("depth_float32_path"),
            sequence_root / "gt" / "depth_npy" / f"{frame_id}.npy",
        )
        gt_disp_path = resolve_manifest_path(
            sequence_root,
            row.get("disparity_path") or row.get("disparity_float32_path"),
            sequence_root / "gt" / "disparity_npy" / f"{frame_id}.npy",
        )
        valid_mask_path = resolve_manifest_path(sequence_root, row.get("valid_mask_path"), sequence_root / "gt" / "valid_mask" / f"{frame_id}.png")
        calibration_path = resolve_manifest_path(sequence_root, row.get("calibration_path"), sequence_root / "calibration" / f"{frame_id}.json")
        for path in [left_path, right_path, gt_depth_path, gt_disp_path, valid_mask_path, calibration_path]:
            if not path.exists():
                raise FileNotFoundError(path)
        fx, baseline = calibration_fx_baseline(calibration_path)
        frames.append(
            FrameRecord(
                frame_id=frame_id,
                left_path=left_path,
                right_path=right_path,
                gt_depth_path=gt_depth_path,
                gt_disp_path=gt_disp_path,
                valid_mask_path=valid_mask_path,
                calibration_path=calibration_path,
                fx=fx,
                baseline=baseline,
            )
        )
    return frames


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).astype(bool)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Cannot read mask: {path}")
    return mask > 0


def prepare_output(pred_dir: Path, diag_dir: Path, overwrite: bool) -> None:
    existing = [path for path in [pred_dir, diag_dir] if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Output exists: {existing}. Pass --overwrite true to replace it.")
    for path in existing:
        shutil.rmtree(path)
    pred_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)


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


def make_contact_sheet(frames: list[FrameRecord], pred_dir: Path, diag_dir: Path, selected_ids: list[str]) -> None:
    rows = []
    for frame_id in selected_ids:
        matches = [frame for frame in frames if frame.frame_id == frame_id]
        if not matches:
            continue
        frame = matches[0]
        left = cv2.cvtColor(read_rgb(frame.left_path), cv2.COLOR_RGB2BGR)
        right = cv2.cvtColor(read_rgb(frame.right_path), cv2.COLOR_RGB2BGR)
        gt = np.load(frame.gt_disp_path).astype(np.float32)
        pred = np.load(pred_dir / f"{frame.frame_id}.npy").astype(np.float32)
        valid = read_mask(frame.valid_mask_path) & (gt > 0)
        err = np.zeros_like(gt, dtype=np.float32)
        err[valid] = np.abs(pred[valid] - gt[valid])
        mask_bgr = cv2.cvtColor((valid.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)
        vmax = float(np.nanpercentile(np.concatenate([gt[valid].ravel(), pred[valid].ravel()]), 99)) if valid.any() else 120.0
        tiles = [
            label_tile(left, f"{frame_id} left RGB"),
            label_tile(right, "right RGB"),
            label_tile(scalar_preview(gt, valid, vmax), "GT disparity"),
            label_tile(scalar_preview(pred, valid, vmax), "S2M2 pred disparity"),
            label_tile(scalar_preview(err, valid, 20.0, cv2.COLORMAP_MAGMA), "abs disp error"),
            label_tile(mask_bgr, "valid mask"),
        ]
        rows.append(np.concatenate(tiles, axis=1))
    if rows:
        cv2.imwrite(str(diag_dir / "contact_sheet.png"), np.concatenate(rows, axis=0))


def compute_metrics(frames: list[FrameRecord], pred_dir: Path) -> dict[str, Any]:
    warnings: list[str] = []
    all_disp_err: list[np.ndarray] = []
    all_depth_err: list[np.ndarray] = []
    shape_rows = []
    resized_count = 0
    for frame in frames:
        pred = np.load(pred_dir / f"{frame.frame_id}.npy").astype(np.float32)
        gt_disp = np.load(frame.gt_disp_path).astype(np.float32)
        gt_depth = np.load(frame.gt_depth_path).astype(np.float32)
        valid = read_mask(frame.valid_mask_path) & np.isfinite(gt_disp) & np.isfinite(gt_depth) & (gt_disp > 0) & (gt_depth > 0)
        original_pred_shape = list(pred.shape)
        if pred.shape != gt_disp.shape:
            warnings.append(f"prediction_shape_mismatch:{frame.frame_id}:pred={pred.shape}:gt={gt_disp.shape}; resized for metrics")
            pred = cv2.resize(pred, (gt_disp.shape[1], gt_disp.shape[0]), interpolation=cv2.INTER_LINEAR)
            resized_count += 1
        metric_valid = valid & np.isfinite(pred) & (pred > 0.1)
        if metric_valid.any():
            disp_err = np.abs(pred[metric_valid] - gt_disp[metric_valid])
            pred_depth = frame.fx * frame.baseline / np.maximum(pred, 1e-6)
            depth_err = np.abs(pred_depth[metric_valid] - gt_depth[metric_valid])
            all_disp_err.append(disp_err.astype(np.float64))
            all_depth_err.append(depth_err.astype(np.float64))
        shape_rows.append(
            {
                "frame_id": frame.frame_id,
                "prediction_shape": original_pred_shape,
                "gt_disparity_shape": list(gt_disp.shape),
                "valid_mask_shape": list(valid.shape),
                "valid_pixel_count": int(metric_valid.sum()),
            }
        )
    disp_err_all = np.concatenate(all_disp_err) if all_disp_err else np.asarray([], dtype=np.float64)
    depth_err_all = np.concatenate(all_depth_err) if all_depth_err else np.asarray([], dtype=np.float64)
    first_pred = np.load(pred_dir / f"{frames[0].frame_id}.npy").astype(np.float32)
    first_gt = np.load(frames[0].gt_disp_path).astype(np.float32)
    first_mask = read_mask(frames[0].valid_mask_path)
    return {
        "number_of_predictions": len(list(pred_dir.glob("*.npy"))),
        "expected_number_of_frames": len(frames),
        "prediction_shape": list(first_pred.shape),
        "gt_disparity_shape": list(first_gt.shape),
        "valid_mask_shape": list(first_mask.shape),
        "disparity_mae_valid_px": float(np.mean(disp_err_all)) if disp_err_all.size else None,
        "disparity_rmse_valid_px": float(np.sqrt(np.mean(disp_err_all**2))) if disp_err_all.size else None,
        "bad_1px_pct": float(np.mean(disp_err_all > 1.0) * 100.0) if disp_err_all.size else None,
        "bad_2px_pct": float(np.mean(disp_err_all > 2.0) * 100.0) if disp_err_all.size else None,
        "bad_3px_pct": float(np.mean(disp_err_all > 3.0) * 100.0) if disp_err_all.size else None,
        "depth_mae_valid_mm": float(np.mean(depth_err_all)) if depth_err_all.size else None,
        "resized_prediction_count_for_metrics": resized_count,
        "warnings": warnings,
        "per_frame_shapes": shape_rows,
    }


def write_readme(path: Path, metrics: dict[str, Any], pred_dir: Path) -> None:
    lines = [
        "# S2M2-S Compatibility Check",
        "",
        "This is a focused end-to-end compatibility test for the newly converted SCARED temporal-GT sequence.",
        "Only S2M2-S at resize width 512 was run; RAFT, RAFT-Small, StereoAnyVideo, and temporal benchmarks were not launched.",
        "",
        f"- Predictions: `{pred_dir}`",
        f"- Predictions written: `{metrics['number_of_predictions']}` / `{metrics['expected_number_of_frames']}`",
        f"- Prediction shape: `{metrics['prediction_shape']}`",
        f"- GT disparity shape: `{metrics['gt_disparity_shape']}`",
        f"- Valid mask shape: `{metrics['valid_mask_shape']}`",
        f"- Disparity MAE valid px: `{metrics['disparity_mae_valid_px']}`",
        f"- Disparity RMSE valid px: `{metrics['disparity_rmse_valid_px']}`",
        f"- Bad-1px pct: `{metrics['bad_1px_pct']}`",
        f"- Bad-2px pct: `{metrics['bad_2px_pct']}`",
        f"- Bad-3px pct: `{metrics['bad_3px_pct']}`",
        f"- Depth MAE valid mm: `{metrics['depth_mae_valid_mm']}`",
        f"- Resized prediction count for metrics: `{metrics['resized_prediction_count_for_metrics']}`",
        f"- Warnings: `{metrics['warnings']}`",
        "",
        "Diagnostics include `contact_sheet.png` for frames 000000, 000098, and 000196.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if args.max_frames < 0:
        raise ValueError("--max-frames must be >= 0")
    method_name = "S2M2-S_512"
    pred_dir = args.out_dir / "predictions" / method_name
    diag_dir = args.out_dir / "diagnostics" / method_name
    prepare_output(pred_dir, diag_dir, bool(args.overwrite))
    frames = read_frames(args.sequence_root, args.max_frames)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    run_log: list[str] = [
        "S2M2-S temporal-GT compatibility run",
        f"sequence_root={args.sequence_root}",
        f"out_dir={args.out_dir}",
        f"frames={len(frames)}",
        f"device={device}",
        f"width={args.width}",
        f"overwrite={bool(args.overwrite)}",
    ]
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = build_s2m2_s(device)
    runtimes: list[float] = []
    scale_x_values: list[float] = []
    for frame in frames:
        pred_path = pred_dir / f"{frame.frame_id}.npy"
        pred, runtime_ms, scale_x = infer_frame(model, read_rgb(frame.left_path), read_rgb(frame.right_path), args.width, device)
        np.save(pred_path, pred.astype(np.float32))
        runtimes.append(runtime_ms)
        scale_x_values.append(scale_x)
        print(f"predicted {frame.frame_id} runtime_ms={runtime_ms:.2f}", flush=True)
    peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024**2)) if device.type == "cuda" else 0.0
    prediction_summary = {
        "method": method_name,
        "model": "S2M2-S",
        "resize_width": args.width,
        "sequence_root": str(args.sequence_root),
        "prediction_dir": str(pred_dir),
        "frames": len(frames),
        "device": str(device),
        "coordinate_system": "original image disparity coordinates",
        "runtime_ms_values": runtimes,
        "avg_runtime_ms": float(np.mean(runtimes)) if runtimes else None,
        "median_runtime_ms": float(np.median(runtimes)) if runtimes else None,
        "peak_vram_mb": peak_vram_mb,
        "scale_x_values": scale_x_values,
    }
    (pred_dir / "prediction_summary.json").write_text(json.dumps(prediction_summary, indent=2) + "\n")
    metrics = compute_metrics(frames, pred_dir)
    metrics["prediction_summary"] = prediction_summary
    (diag_dir / "s2m2_compatibility_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    make_contact_sheet(frames, pred_dir, diag_dir, ["000000", "000098", "000196"])
    write_readme(diag_dir / "README.md", metrics, pred_dir)
    run_log.extend(
        [
            f"predictions={metrics['number_of_predictions']}",
            f"expected_frames={metrics['expected_number_of_frames']}",
            f"avg_runtime_ms={prediction_summary['avg_runtime_ms']}",
            f"peak_vram_mb={peak_vram_mb}",
            f"metrics={diag_dir / 's2m2_compatibility_metrics.json'}",
            f"warnings={metrics['warnings']}",
        ]
    )
    (pred_dir / "run.log").write_text("\n".join(run_log) + "\n")
    console_metrics = {key: metrics[key] for key in [
        "number_of_predictions",
        "expected_number_of_frames",
        "prediction_shape",
        "gt_disparity_shape",
        "valid_mask_shape",
        "disparity_mae_valid_px",
        "disparity_rmse_valid_px",
        "bad_1px_pct",
        "bad_2px_pct",
        "bad_3px_pct",
        "depth_mae_valid_mm",
        "resized_prediction_count_for_metrics",
        "warnings",
    ]}
    print(json.dumps({"prediction_dir": str(pred_dir), "diagnostics_dir": str(diag_dir), "metrics": console_metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
