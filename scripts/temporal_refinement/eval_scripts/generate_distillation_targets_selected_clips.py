#!/usr/bin/env python3
"""Generate compact oracle/distillation targets for selected SCARED clips."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = ROOT / "scripts" / "temporal_refinement" / "eval_scripts"
LIB_DIR = ROOT / "scripts" / "temporal_refinement" / "lib"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(LIB_DIR))

import evaluate_s2m2_streaming_temporal_gt_rectified as streaming  # noqa: E402
from artifact_metrics import gradient_magnitude, percentile_mask  # noqa: E402
from temporal_baselines import (  # noqa: E402
    adaptive_no_raft_diff_grad_sequence,
    fixed_ema_sequence,
    raft_warped_ema_sequence,
)


DEFAULT_PLANNING_CSV = ROOT / "results/03_temporal_refinement/evaluation/distillation_planning/candidate_clips_for_distillation.csv"
DEFAULT_DATASET_ROOT = ROOT / "dataset/SCARED/curated/temporal_gt_rectified"
DEFAULT_OUTPUT_ROOT = ROOT / "results/03_temporal_refinement/evaluation/distillation_targets_selected_clips"
RAFT_SMALL_CKPT = ROOT / "external/frame_stereo_repos/RAFT/checkpoints/raft-small.pth"

CANDIDATE_IDS = {
    "raw_s2m2": 0,
    "fixed_ema_alpha_0.35": 1,
    "adaptive_no_raft": 2,
    "raft_small_6_warped_ema_alpha_0.50": 3,
    "stereoanyvideo": 4,
}
INVALID_LABEL = 255


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def finite_mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def pct(value: np.ndarray, valid: np.ndarray) -> float:
    den = int(valid.sum())
    return float(value[valid].mean() * 100.0) if den else float("nan")


def target_hw(shape: tuple[int, int], scale: float) -> tuple[int, int]:
    h, w = shape
    return max(1, int(round(h * scale))), max(1, int(round(w * scale)))


def valid_masked_downsample_disparity(
    disp: np.ndarray,
    valid_mask: np.ndarray,
    out_h: int,
    out_w: int,
    min_valid_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    valid = valid_mask.astype(bool) & np.isfinite(disp)
    if disp.shape == (out_h, out_w):
        out = disp.astype(np.float32, copy=True)
        out_valid = valid.copy()
        out[~out_valid] = 0.0
        return out, out_valid
    valid_f = valid.astype(np.float32)
    coverage = cv2.resize(valid_f, (out_w, out_h), interpolation=cv2.INTER_AREA)
    weighted = cv2.resize(disp.astype(np.float32) * valid_f, (out_w, out_h), interpolation=cv2.INTER_AREA)
    out_valid = coverage >= float(min_valid_ratio)
    out = np.zeros((out_h, out_w), dtype=np.float32)
    out[out_valid] = weighted[out_valid] / np.maximum(coverage[out_valid], 1e-6)
    return out, out_valid


def downsample_label(array: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    if array.shape == (out_h, out_w):
        return array.astype(np.uint8, copy=False)
    return cv2.resize(array.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)


def resize_like(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if pred.shape == ref.shape:
        return pred.astype(np.float32, copy=False)
    sx = pred.shape[1] / float(ref.shape[1])
    return (cv2.resize(pred.astype(np.float32), (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_LINEAR) / sx).astype(np.float32)


def load_clips(path: Path, max_clips: int) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[:max_clips] if max_clips > 0 else rows


def clip_id(row: dict[str, str]) -> str:
    return f"{row['sequence_id']}_{row['start_frame']}_{row['end_frame']}"


def selected_frames(
    sequence_root: Path,
    row: dict[str, str],
    audit: dict[tuple[str, str], dict[str, str]],
    max_frames: int,
) -> list[streaming.FrameRecord]:
    frames = []
    for frame in streaming.read_frames(sequence_root):
        if not (row["start_frame"] <= frame.frame_id <= row["end_frame"]):
            continue
        skip, _reason, _valid_pct, _flags = streaming.frame_should_skip(frame, audit, True, 0.05)
        if skip:
            continue
        frames.append(frame)
        if max_frames > 0 and len(frames) >= max_frames:
            break
    return frames


def frame_as_sav_dict(frame: streaming.FrameRecord) -> dict[str, Any]:
    return {
        "id": frame.frame_id,
        "left_path": frame.left_path,
        "right_path": frame.right_path,
        "gt_disp_path": frame.gt_disp_path,
        "gt_depth_path": frame.gt_depth_path,
        "valid_mask_path": frame.valid_mask_path,
        "calib_path": frame.calibration_path,
        "fx": frame.fx,
        "baseline_mm": frame.baseline,
    }


def infer_raw_s2m2(frames: list[streaming.FrameRecord], device: torch.device) -> tuple[list[np.ndarray], list[float], float]:
    model = streaming.build_s2m2_s(device)
    preds, runtimes = [], []
    peak = 0.0
    for frame in frames:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        pred, runtime_ms, _scale = streaming.infer_frame(
            model,
            streaming.read_rgb(frame.left_path),
            streaming.read_rgb(frame.right_path),
            512,
            device,
        )
        preds.append(pred)
        runtimes.append(runtime_ms)
        if device.type == "cuda":
            peak = max(peak, torch.cuda.max_memory_allocated() / (1024**2))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return preds, runtimes, peak


def infer_sav(
    frames: list[streaming.FrameRecord],
    device: torch.device,
    chunk_size: int,
    pad_last_chunk: bool,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    import evaluate_scared_temporal_gt as old_eval

    if chunk_size <= 0:
        raise ValueError("--sav-chunk-size must be positive")
    model = old_eval.build_sav(device)
    preds: list[np.ndarray] = []
    padded = 0
    chunks = 0
    try:
        frame_dicts = [frame_as_sav_dict(frame) for frame in frames]
        for start in range(0, len(frame_dicts), chunk_size):
            chunk = frame_dicts[start : start + chunk_size]
            requested = len(chunk)
            if requested < chunk_size and pad_last_chunk:
                pad_count = chunk_size - requested
                chunk = chunk + [chunk[-1]] * pad_count
                padded += pad_count
            chunk_preds, _runtime, _peak = old_eval.infer_sav_chunk(model, chunk, (384, 640), 6, device)
            preds.extend(chunk_preds[:requested])
            chunks += 1
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if len(preds) != len(frames):
        raise RuntimeError(f"SAV produced {len(preds)} predictions for {len(frames)} frames")
    meta = {
        "sav_chunk_size": chunk_size,
        "sav_pad_last_chunk": bool(pad_last_chunk),
        "sav_chunks_used": chunks,
        "sav_padded_frames": padded,
        "sav_available": True,
    }
    return preds, meta


def rgb_tensor(frame: streaming.FrameRecord, device: torch.device) -> torch.Tensor:
    img = streaming.read_rgb(frame.left_path)
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(device)


def infer_raftsmall_warped_ema(
    frames: list[streaming.FrameRecord],
    raw_preds: list[np.ndarray],
    iters: int,
    device: torch.device,
) -> list[np.ndarray]:
    from flow import FrozenRAFT

    if not RAFT_SMALL_CKPT.exists():
        raise FileNotFoundError(RAFT_SMALL_CKPT)
    model = FrozenRAFT(checkpoint=RAFT_SMALL_CKPT, iters=iters, small=True).to(device).eval()
    flows: dict[tuple[str, str], np.ndarray] = {}
    try:
        with torch.no_grad():
            for prev, cur in zip(frames[:-1], frames[1:]):
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                flow = model(rgb_tensor(prev, device), rgb_tensor(cur, device))
                flows[(prev.frame_id, cur.frame_id)] = flow[0].permute(1, 2, 0).detach().float().cpu().numpy().astype(np.float32)
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def load_flow(prev_id: str, cur_id: str) -> np.ndarray:
        return flows[(prev_id, cur_id)]

    result = raft_warped_ema_sequence(raw_preds, [f.frame_id for f in frames], load_flow, 0.50, warp_device="auto")
    return result.predictions


def candidate_errors(candidates: dict[str, list[np.ndarray]], gt: np.ndarray, valid: np.ndarray, idx: int) -> dict[str, np.ndarray]:
    return {
        name: np.where(valid, np.abs(resize_like(preds[idx], gt) - gt), np.inf).astype(np.float32)
        for name, preds in candidates.items()
    }


def oracle_for_maps(names: list[str], maps: dict[str, np.ndarray], gt: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not names:
        return np.full(gt.shape, INVALID_LABEL, dtype=np.uint8), np.zeros(gt.shape, dtype=np.float32)
    stacks = [np.where(valid, np.abs(maps[name] - gt), np.inf) for name in names]
    best = np.argmin(np.stack(stacks, axis=0), axis=0)
    labels = np.full(gt.shape, INVALID_LABEL, dtype=np.uint8)
    disp = np.zeros(gt.shape, dtype=np.float32)
    for local_idx, name in enumerate(names):
        mask = valid & (best == local_idx)
        labels[mask] = CANDIDATE_IDS[name]
        disp[mask] = maps[name][mask]
    return labels, disp


def oracle_for(names: list[str], candidates: dict[str, list[np.ndarray]], gt: np.ndarray, valid: np.ndarray, idx: int) -> tuple[np.ndarray, np.ndarray]:
    if not names:
        labels = np.full(gt.shape, INVALID_LABEL, dtype=np.uint8)
        return labels, np.zeros(gt.shape, dtype=np.float32)
    stacks = [np.where(valid, np.abs(resize_like(candidates[name][idx], gt) - gt), np.inf) for name in names]
    best = np.argmin(np.stack(stacks, axis=0), axis=0)
    labels = np.full(gt.shape, INVALID_LABEL, dtype=np.uint8)
    disp = np.zeros(gt.shape, dtype=np.float32)
    for local_idx, name in enumerate(names):
        mask = valid & (best == local_idx)
        labels[mask] = CANDIDATE_IDS[name]
        disp[mask] = resize_like(candidates[name][idx], gt)[mask]
    return labels, disp


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_diag(path: Path, raw: np.ndarray, gt: np.ndarray, valid: np.ndarray, high_error: np.ndarray) -> None:
    err = np.zeros_like(raw, dtype=np.float32)
    err[valid] = np.abs(raw[valid] - gt[valid])
    vis = np.concatenate(
        [
            cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            cv2.normalize(gt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            np.clip(err * 20.0, 0, 255).astype(np.uint8),
            (high_error.astype(np.uint8) * 255),
        ],
        axis=1,
    )
    cv2.imwrite(str(path), vis)


def process_clip(
    row: dict[str, str],
    args: argparse.Namespace,
    audit: dict[tuple[str, str], dict[str, str]],
    device: torch.device,
    availability: dict[str, Any],
) -> dict[str, Any]:
    cid = clip_id(row)
    sequence_root = args.dataset_root / row["sequence_id"]
    frames = selected_frames(sequence_root, row, audit, args.max_frames_per_clip)
    out_dir = args.output_root / "clips" / cid
    targets_dir = out_dir / "targets"
    diag_dir = out_dir / "diagnostics"
    targets_dir.mkdir(parents=True, exist_ok=True)
    if args.save_diagnostics:
        diag_dir.mkdir(parents=True, exist_ok=True)

    raw_preds, raw_runtime, raw_peak = infer_raw_s2m2(frames, device) if frames else ([], [], 0.0)
    fixed = fixed_ema_sequence(raw_preds, 0.35).predictions
    adaptive = adaptive_no_raft_diff_grad_sequence(
        raw_preds,
        alpha_min=0.25,
        alpha_max=0.75,
        diff_scale_px=3.0,
        grad_scale_px=8.0,
        w_diff=1.0,
        w_grad=1.0,
    ).predictions
    candidates: dict[str, list[np.ndarray]] = {
        "raw_s2m2": raw_preds,
        "fixed_ema_alpha_0.35": fixed,
        "adaptive_no_raft": adaptive,
    }

    if args.include_raft_small and len(frames) >= 2:
        try:
            candidates["raft_small_6_warped_ema_alpha_0.50"] = infer_raftsmall_warped_ema(
                frames, raw_preds, args.raft_small_iters, device
            )
            availability["raft_small"]["available_clips"].append(cid)
        except Exception as exc:  # keep no-flow targets if offline teacher fails
            availability["raft_small"]["failed_clips"][cid] = f"{type(exc).__name__}: {exc}"
    sav_meta = {
        "sav_chunk_size": args.sav_chunk_size,
        "sav_pad_last_chunk": bool(args.sav_pad_last_chunk),
        "sav_chunks_used": 0,
        "sav_padded_frames": 0,
        "sav_available": False,
    }
    if args.include_sav and frames:
        try:
            sav_preds, sav_meta = infer_sav(frames, device, args.sav_chunk_size, bool(args.sav_pad_last_chunk))
            candidates["stereoanyvideo"] = [resize_like(p, raw_preds[0]) for p in sav_preds]
            availability["stereoanyvideo"]["available_clips"].append(cid)
            availability["stereoanyvideo"].setdefault("chunk_metadata", {})[cid] = sav_meta
        except Exception as exc:
            availability["stereoanyvideo"]["failed_clips"][cid] = f"{type(exc).__name__}: {exc}"

    no_flow_names = ["raw_s2m2", "fixed_ema_alpha_0.35", "adaptive_no_raft"]
    raw_fixed_raft_names = [name for name in ["raw_s2m2", "fixed_ema_alpha_0.35", "raft_small_6_warped_ema_alpha_0.50"] if name in candidates]
    all_names = list(candidates)

    frame_rows: list[dict[str, Any]] = []
    per_candidate_mae = {name: [] for name in CANDIDATE_IDS}
    oracle_mae = {"no_flow": [], "raw_fixed_raftsmall": [], "all_available": []}
    selected_counts = {name: [] for name in CANDIDATE_IDS}
    edge_raw_pcts, stable_raft_pcts = [], []
    prev_raw: np.ndarray | None = None
    prev_gt: np.ndarray | None = None
    prev_valid: np.ndarray | None = None

    for idx, frame in enumerate(frames):
        gt_full = np.load(frame.gt_disp_path).astype(np.float32)
        valid_full = streaming.read_mask(frame.valid_mask_path) & np.isfinite(gt_full) & (gt_full > 0)
        out_h, out_w = target_hw(gt_full.shape, args.target_scale)
        gt, valid = valid_masked_downsample_disparity(
            gt_full,
            valid_full,
            out_h,
            out_w,
            args.downsample_min_valid_ratio,
        )
        lowres_candidates: dict[str, np.ndarray] = {}
        for name, preds in candidates.items():
            pred_full = resize_like(preds[idx], gt_full)
            pred_low, pred_valid = valid_masked_downsample_disparity(
                pred_full,
                valid_full,
                out_h,
                out_w,
                args.downsample_min_valid_ratio,
            )
            lowres_candidates[name] = pred_low

        gt_save = gt.astype(np.float16)
        gt = gt_save.astype(np.float32)
        lowres_saved = {name: pred.astype(np.float16) for name, pred in lowres_candidates.items()}
        lowres_eval = {name: pred.astype(np.float32) for name, pred in lowres_saved.items()}
        for name, pred in lowres_eval.items():
            per_candidate_mae[name].append(float(np.mean(np.abs(pred[valid] - gt[valid]))) if np.any(valid) else float("nan"))

        raw = lowres_eval["raw_s2m2"]
        lab_no_flow, disp_no_flow = oracle_for_maps(no_flow_names, lowres_eval, gt, valid)
        lab_rfr, disp_rfr = oracle_for_maps(raw_fixed_raft_names, lowres_eval, gt, valid)
        lab_all, disp_all = oracle_for_maps(all_names, lowres_eval, gt, valid)
        oracle_mae["no_flow"].append(float(np.mean(np.abs(disp_no_flow[valid] - gt[valid]))) if np.any(valid) else float("nan"))
        oracle_mae["raw_fixed_raftsmall"].append(float(np.mean(np.abs(disp_rfr[valid] - gt[valid]))) if np.any(valid) else float("nan"))
        oracle_mae["all_available"].append(float(np.mean(np.abs(disp_all[valid] - gt[valid]))) if np.any(valid) else float("nan"))

        raw_err = np.abs(raw - gt)
        gt_edge = percentile_mask(gradient_magnitude(gt), valid, 80.0)
        high_error = valid & (raw_err > 3.0)
        high_boundary = gt_edge & high_error
        if prev_raw is None:
            temporal_mismatch = np.zeros_like(valid, dtype=bool)
        else:
            tv = valid & prev_valid & np.isfinite(prev_raw) & np.isfinite(prev_gt)
            mismatch = np.abs(np.abs(raw - prev_raw) - np.abs(gt - prev_gt))
            threshold = float(np.nanpercentile(mismatch[tv], 75.0)) if np.any(tv) else math.nan
            temporal_mismatch = tv & np.isfinite(mismatch) & (mismatch >= threshold)

        for name, label in CANDIDATE_IDS.items():
            selected_counts[name].append(pct(lab_all == label, valid))
        edge_raw_pcts.append(pct(lab_all == CANDIDATE_IDS["raw_s2m2"], gt_edge))
        stable = valid & ~temporal_mismatch
        stable_raft_pcts.append(pct(lab_all == CANDIDATE_IDS["raft_small_6_warped_ema_alpha_0.50"], stable))

        raw_save = lowres_saved["raw_s2m2"]
        disp_no_flow_save = disp_no_flow.astype(np.float16)
        disp_rfr_save = disp_rfr.astype(np.float16)
        disp_all_save = disp_all.astype(np.float16)
        target_payload = {
            "raw_disp": raw_save,
            "gt_disp": gt_save,
            "valid_mask": valid.astype(np.uint8),
            "oracle_selected_candidate_id_no_flow": lab_no_flow.astype(np.uint8),
            "oracle_selected_candidate_id_raw_fixed_raftsmall": lab_rfr.astype(np.uint8),
            "oracle_selected_candidate_id_all_available": lab_all.astype(np.uint8),
            "oracle_no_flow_disp": disp_no_flow_save,
            "oracle_raw_fixed_raftsmall_disp": disp_rfr_save,
            "oracle_all_available_disp": disp_all_save,
            # Backward-compatible aliases used by the v0 trainer/evaluator.
            "oracle_disp_no_flow": disp_no_flow_save,
            "oracle_disp_all_available": disp_all_save,
            "delta_disp_oracle_no_flow_minus_raw": disp_no_flow_save.astype(np.float32) - raw_save.astype(np.float32),
            "delta_disp_oracle_all_available_minus_raw": disp_all_save.astype(np.float32) - raw_save.astype(np.float32),
            "raw_confidence_binary": (valid & (raw_err <= 1.0)).astype(np.uint8),
            "high_error_mask": high_error.astype(np.uint8),
            "high_boundary_error_mask": high_boundary.astype(np.uint8),
            "high_temporal_mismatch_mask": temporal_mismatch.astype(np.uint8),
        }
        candidate_output_names = {
            "fixed_ema_alpha_0.35": "fixed_ema_disp",
            "adaptive_no_raft": "adaptive_no_raft_disp",
            "raft_small_6_warped_ema_alpha_0.50": "raftsmall_disp",
            "stereoanyvideo": "sav_disp",
        }
        for name, out_name in candidate_output_names.items():
            if name in lowres_saved:
                target_payload[out_name] = lowres_saved[name]
        if args.save_full_resolution_targets:
            target_payload.update({f"fullres_{k}": v for k, v in {"valid_mask": valid_full.astype(np.uint8)}.items()})
        target_path = targets_dir / f"{frame.frame_id}.npz"
        np.savez_compressed(target_path, **target_payload)
        if args.save_diagnostics:
            save_diag(diag_dir / f"{frame.frame_id}.png", raw, gt, valid, high_error)

        frame_rows.append(
            {
                "frame_id": frame.frame_id,
                "target_path": str(target_path),
                "valid_pixel_pct": float(valid.mean() * 100.0),
                "raw_disp_mae": per_candidate_mae["raw_s2m2"][-1],
                "oracle_no_flow_disp_mae": oracle_mae["no_flow"][-1],
                "oracle_raw_fixed_raftsmall_disp_mae": oracle_mae["raw_fixed_raftsmall"][-1],
                "oracle_all_available_disp_mae": oracle_mae["all_available"][-1],
                "candidates_available": "|".join(all_names),
            }
        )
        prev_raw, prev_gt, prev_valid = raw, gt, valid

    write_csv(out_dir / "frame_target_index.csv", frame_rows)
    metadata = {
        "clip_id": cid,
        "sequence_id": row["sequence_id"],
        "frame_start": row["start_frame"],
        "frame_end": row["end_frame"],
        "frames_processed": len(frames),
        "target_scale": args.target_scale,
        "downsample_min_valid_ratio": args.downsample_min_valid_ratio,
        "candidate_ids": CANDIDATE_IDS,
        "candidates_available": all_names,
        "raw_s2m2_runtime_ms_mean": finite_mean(raw_runtime),
        "raw_s2m2_peak_vram_mb": raw_peak,
        **sav_meta,
    }
    (out_dir / "clip_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    size_mb = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file()) / (1024**2)
    raw_mae = finite_mean(per_candidate_mae["raw_s2m2"])
    return {
        "clip_id": cid,
        "sequence_id": row["sequence_id"],
        "frame_start": row["start_frame"],
        "frame_end": row["end_frame"],
        "frames_requested": int(row.get("num_frames") or 0),
        "frames_processed": len(frames),
        "dominant_failure_mode": row.get("dominant_failure_mode", ""),
        "candidates_available": "|".join(all_names),
        "raw_disp_mae_mean": raw_mae,
        "fixed_ema_disp_mae_mean": finite_mean(per_candidate_mae["fixed_ema_alpha_0.35"]),
        "adaptive_no_raft_disp_mae_mean": finite_mean(per_candidate_mae["adaptive_no_raft"]),
        "raftsmall_disp_mae_mean": finite_mean(per_candidate_mae["raft_small_6_warped_ema_alpha_0.50"]),
        "sav_disp_mae_mean": finite_mean(per_candidate_mae["stereoanyvideo"]),
        "oracle_no_flow_disp_mae_mean": finite_mean(oracle_mae["no_flow"]),
        "oracle_raw_fixed_raftsmall_disp_mae_mean": finite_mean(oracle_mae["raw_fixed_raftsmall"]),
        "oracle_all_available_disp_mae_mean": finite_mean(oracle_mae["all_available"]),
        "oracle_gain_no_flow_vs_raw_px": raw_mae - finite_mean(oracle_mae["no_flow"]),
        "oracle_gain_raw_fixed_raftsmall_vs_raw_px": raw_mae - finite_mean(oracle_mae["raw_fixed_raftsmall"]),
        "oracle_gain_all_available_vs_raw_px": raw_mae - finite_mean(oracle_mae["all_available"]),
        "mean_selected_raw_pct": finite_mean(selected_counts["raw_s2m2"]),
        "mean_selected_fixed_pct": finite_mean(selected_counts["fixed_ema_alpha_0.35"]),
        "mean_selected_adaptive_pct": finite_mean(selected_counts["adaptive_no_raft"]),
        "mean_selected_raftsmall_pct": finite_mean(selected_counts["raft_small_6_warped_ema_alpha_0.50"]),
        "mean_selected_sav_pct": finite_mean(selected_counts["stereoanyvideo"]),
        "edge_region_raw_selection_pct": finite_mean(edge_raw_pcts),
        "occlusion_region_raw_selection_pct": float("nan"),
        "stable_region_raft_selection_pct": finite_mean(stable_raft_pcts),
        "output_size_mb": size_mb,
    }


def write_readme(path: Path, args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        f"""# Selected-Clip Distillation Targets

This run generated compact teacher/oracle targets for selected planning clips only.
Oracle labels use ground truth, so they are training targets and are not deployable online.

- Planning CSV: `{args.planning_csv}`
- Dataset root: `{args.dataset_root}`
- Clips processed: `{len(rows)}`
- Target scale: `{args.target_scale}`
- Downsample min valid ratio: `{args.downsample_min_valid_ratio}`
- Full-resolution targets: `{args.save_full_resolution_targets}`
- Candidate caches: not written. Dense candidate predictions live only in memory.
- Targets include raw S2M2, fixed EMA, adaptive no-RAFT, RAFT-Small, and SAV when available.
- SAV chunking: chunk size `{args.sav_chunk_size}`, pad last chunk `{args.sav_pad_last_chunk}`.
- `oracle_all_available` includes SAV when `teacher_availability_report.json` marks SAV available.
- RAFT-Small and StereoAnyVideo: offline teacher candidates only; see `teacher_availability_report.json`.
- These outputs are selected-clip targets for future lightweight distillation.
- Saved `.npz` metrics are low-resolution target-space metrics, computed after valid-mask-aware downsampling and low-resolution oracle selection.

Targets are stored as compressed `.npz` files under `clips/*/targets/`. Full-resolution target export requires explicit `--save-full-resolution-targets true`.
"""
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--planning-csv", type=Path, default=DEFAULT_PLANNING_CSV)
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--max-clips", type=int, default=0)
    p.add_argument("--max-frames-per-clip", type=int, default=0)
    p.add_argument("--target-scale", type=float, default=0.25)
    p.add_argument("--downsample-min-valid-ratio", type=float, default=0.25)
    p.add_argument("--save-full-resolution-targets", nargs="?", const=True, default=False, type=parse_bool)
    p.add_argument("--include-sav", nargs="?", const=True, default=True, type=parse_bool)
    p.add_argument("--sav-chunk-size", type=int, default=32)
    p.add_argument("--sav-pad-last-chunk", nargs="?", const=True, default=True, type=parse_bool)
    p.add_argument("--include-raft-small", nargs="?", const=True, default=True, type=parse_bool)
    p.add_argument("--raft-small-iters", type=int, default=6)
    p.add_argument("--include-raft-full", nargs="?", const=True, default=False, type=parse_bool)
    p.add_argument("--save-diagnostics", nargs="?", const=True, default=True, type=parse_bool)
    p.add_argument("--overwrite", nargs="?", const=True, default=False, type=parse_bool)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.planning_csv = resolve_path(args.planning_csv)
    args.dataset_root = resolve_path(args.dataset_root)
    args.output_root = resolve_path(args.output_root)
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output root exists: {args.output_root}")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    audit = streaming.read_audit_frames(streaming.DEFAULT_AUDIT_FRAME_CSV)
    availability = {
        "raft_small": {"requested": bool(args.include_raft_small), "checkpoint": str(RAFT_SMALL_CKPT), "available_clips": [], "failed_clips": {}},
        "stereoanyvideo": {
            "requested": bool(args.include_sav),
            "sav_chunk_size": args.sav_chunk_size,
            "sav_pad_last_chunk": bool(args.sav_pad_last_chunk),
            "available_clips": [],
            "failed_clips": {},
        },
        "raft_full": {"requested": bool(args.include_raft_full), "available_clips": [], "failed_clips": {}, "note": "not implemented for this selected-clip smoke script"},
    }

    log_path = args.output_root / "run.log"
    start = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with log_path.open("w") as log:
        log.write(f"start={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"device={device}\n")
        for row in load_clips(args.planning_csv, args.max_clips):
            cid = clip_id(row)
            log.write(f"processing {cid}\n")
            log.flush()
            rows.append(process_clip(row, args, audit, device, availability))
        log.write(f"elapsed_seconds={time.perf_counter() - start:.3f}\n")

    write_csv(args.output_root / "clip_targets_index.csv", rows)
    summary = {
        "output_root": str(args.output_root),
        "clips_processed": len(rows),
        "frames_processed": int(sum(int(r["frames_processed"]) for r in rows)),
        "target_scale": args.target_scale,
        "save_full_resolution_targets": bool(args.save_full_resolution_targets),
        "mean_raw_disp_mae": finite_mean([r["raw_disp_mae_mean"] for r in rows]),
        "mean_oracle_no_flow_disp_mae": finite_mean([r["oracle_no_flow_disp_mae_mean"] for r in rows]),
        "mean_oracle_all_available_disp_mae": finite_mean([r["oracle_all_available_disp_mae_mean"] for r in rows]),
        "elapsed_seconds": time.perf_counter() - start,
    }
    (args.output_root / "aggregate_target_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_root / "teacher_availability_report.json").write_text(json.dumps(availability, indent=2) + "\n")
    write_readme(args.output_root / "README.md", args, rows)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
