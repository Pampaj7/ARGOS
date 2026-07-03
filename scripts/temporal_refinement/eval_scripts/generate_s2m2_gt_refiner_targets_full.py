#!/usr/bin/env python3
"""Generate compact low-res S2M2->GT refiner targets for all valid frames."""

from __future__ import annotations

import argparse
import csv
import json
import math
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
sys.path.insert(0, str(EVAL_DIR))

import evaluate_s2m2_streaming_temporal_gt_rectified as streaming  # noqa: E402
from generate_distillation_targets_selected_clips import target_hw, valid_masked_downsample_disparity  # noqa: E402


DEFAULT_DATASET_ROOT = ROOT / "dataset/SCARED/curated/temporal_gt_rectified"
DEFAULT_FRAME_METRICS = (
    ROOT / "results/03_temporal_refinement/evaluation/gt_temporal_rectified_streaming_s2m2_v2_artifact_temporal/frame_metrics.csv"
)
DEFAULT_OUTPUT_ROOT = ROOT / "results/03_temporal_refinement/evaluation/s2m2_gt_refiner_targets_full"


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def finite_mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def finite_median(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.median(vals)) if vals else float("nan")


def read_eval_included(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    with path.open(newline="") as f:
        return {
            (row["sequence_id"], row["frame_id"])
            for row in csv.DictReader(f)
            if str(row.get("included", "")).strip().lower() == "true"
        }


def selected_frames(args: argparse.Namespace) -> list[streaming.FrameRecord]:
    included = read_eval_included(args.frame_metrics_csv)
    audit = streaming.read_audit_frames(streaming.DEFAULT_AUDIT_FRAME_CSV)
    sequences = streaming.discover_sequences(args.dataset_root, None, args.limit_sequences)
    frames: list[streaming.FrameRecord] = []
    for seq in sequences:
        seq_frames: list[streaming.FrameRecord] = []
        for frame in streaming.read_frames(seq):
            if included:
                keep = (frame.sequence_id, frame.frame_id) in included
            else:
                skip, _reason, _valid, _flags = streaming.frame_should_skip(frame, audit, True, 0.05)
                keep = not skip
            if keep:
                seq_frames.append(frame)
                if args.limit_frames_per_sequence > 0 and len(seq_frames) >= args.limit_frames_per_sequence:
                    break
        frames.extend(seq_frames)
    return frames


def mae(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    return float(np.mean(np.abs(pred[valid] - gt[valid]))) if np.any(valid) else float("nan")


def bad(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, tau: float) -> float:
    return float(np.mean(np.abs(pred[valid] - gt[valid]) > tau) * 100.0) if np.any(valid) else float("nan")


def save_diag(path: Path, raw: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> None:
    err = np.zeros_like(raw, dtype=np.float32)
    err[valid] = np.abs(raw[valid] - gt[valid])
    vmax = float(np.percentile(gt[valid], 98)) if np.any(valid) else 64.0
    tiles = [
        cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.normalize(gt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) if vmax > 0 else gt.astype(np.uint8),
        np.clip(err * 20.0, 0, 255).astype(np.uint8),
        valid.astype(np.uint8) * 255,
    ]
    cv2.imwrite(str(path), np.concatenate(tiles, axis=1))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sequence_summary(sequence_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    shard_bytes = {r["target_path"]: float(r.get("shard_size_bytes", 0)) for r in rows}
    return {
        "sequence_id": sequence_id,
        "frames": len(rows),
        "valid_ratio_lowres_mean": finite_mean([r["valid_ratio_lowres"] for r in rows]),
        "raw_disp_mae_lowres_mean": finite_mean([r["raw_disp_mae_lowres"] for r in rows]),
        "raw_disp_mae_lowres_median": finite_median([r["raw_disp_mae_lowres"] for r in rows]),
        "raw_bad_1px_lowres_mean": finite_mean([r["raw_bad_1px_lowres"] for r in rows]),
        "raw_bad_3px_lowres_mean": finite_mean([r["raw_bad_3px_lowres"] for r in rows]),
        "delta_abs_mean": finite_mean([r["delta_abs_mean"] for r in rows]),
        "runtime_ms_mean": finite_mean([r["runtime_ms"] for r in rows]),
        "output_size_mb": sum(shard_bytes.values()) / (1024**2),
    }


def process(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output root exists: {args.output_root}")
        shutil.rmtree(args.output_root)
    targets_dir = args.output_root / "targets"
    diag_dir = args.output_root / "diagnostics"
    targets_dir.mkdir(parents=True, exist_ok=True)
    if args.save_diagnostics:
        diag_dir.mkdir(parents=True, exist_ok=True)

    run_log = [
        f"[{now()}] generate_s2m2_gt_refiner_targets_full.py",
        f"dataset_root={args.dataset_root}",
        f"frame_metrics_csv={args.frame_metrics_csv}",
        f"output_root={args.output_root}",
        f"target_scale={args.target_scale}",
        f"min_valid_ratio={args.min_valid_ratio}",
        f"save_diagnostics={bool(args.save_diagnostics)}",
    ]
    (args.output_root / "run.log").write_text("\n".join(run_log) + "\n")

    frames = selected_frames(args)
    if not frames:
        raise RuntimeError("No frames selected.")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = streaming.build_s2m2_s(device)
    start = time.perf_counter()
    frame_rows: list[dict[str, Any]] = []
    previous_by_sequence: dict[str, str] = {}
    next_by_key = {(cur.sequence_id, cur.frame_id): nxt.frame_id for cur, nxt in zip(frames[:-1], frames[1:]) if cur.sequence_id == nxt.sequence_id}
    current_sequence = ""
    shard: dict[str, list[Any]] = {}

    def reset_shard(sequence_id: str) -> None:
        nonlocal current_sequence, shard
        current_sequence = sequence_id
        shard = {
            "raw_disp": [],
            "gt_disp": [],
            "valid_mask": [],
            "delta_disp_gt_minus_raw": [],
            "frame_id": [],
            "frame_index": [],
            "previous_frame_id": [],
            "next_frame_id": [],
            "row_indices": [],
        }

    def flush_shard() -> None:
        if not current_sequence or not shard.get("raw_disp"):
            return
        path = targets_dir / f"{current_sequence}.npz"
        np.savez_compressed(
            path,
            raw_disp=np.stack(shard["raw_disp"], axis=0),
            gt_disp=np.stack(shard["gt_disp"], axis=0),
            valid_mask=np.stack(shard["valid_mask"], axis=0),
            delta_disp_gt_minus_raw=np.stack(shard["delta_disp_gt_minus_raw"], axis=0),
            sequence_id=np.asarray(current_sequence),
            frame_id=np.asarray(shard["frame_id"]),
            frame_index=np.asarray(shard["frame_index"], dtype=np.int32),
            previous_frame_id=np.asarray(shard["previous_frame_id"]),
            next_frame_id=np.asarray(shard["next_frame_id"]),
        )
        size = path.stat().st_size
        for offset, row_index in enumerate(shard["row_indices"]):
            frame_rows[row_index]["target_path"] = str(path)
            frame_rows[row_index]["frame_offset"] = offset
            frame_rows[row_index]["shard_size_bytes"] = size
        run_log.append(f"[{now()}] sequence_done={current_sequence} frames={len(shard['raw_disp'])} shard_size_mb={size / (1024**2):.2f}")
        (args.output_root / "run.log").write_text("\n".join(run_log) + "\n")

    for index, frame in enumerate(frames):
        if index == 0 or frame.sequence_id != frames[index - 1].sequence_id:
            flush_shard()
            reset_shard(frame.sequence_id)
            run_log.append(f"[{now()}] sequence_start={frame.sequence_id}")
            (args.output_root / "run.log").write_text("\n".join(run_log) + "\n")

        gt_full = np.load(frame.gt_disp_path).astype(np.float32)
        valid_full = streaming.read_mask(frame.valid_mask_path) & np.isfinite(gt_full) & (gt_full > 0)
        pred_full, runtime_ms, _scale = streaming.infer_frame(
            model,
            streaming.read_rgb(frame.left_path),
            streaming.read_rgb(frame.right_path),
            512,
            device,
        )
        out_h, out_w = target_hw(gt_full.shape, args.target_scale)
        gt, valid = valid_masked_downsample_disparity(gt_full, valid_full, out_h, out_w, args.min_valid_ratio)
        raw, _raw_valid = valid_masked_downsample_disparity(pred_full, valid_full, out_h, out_w, args.min_valid_ratio)
        gt16 = gt.astype(np.float16)
        raw16 = raw.astype(np.float16)
        gt_eval = gt16.astype(np.float32)
        raw_eval = raw16.astype(np.float32)
        valid = valid & np.isfinite(gt_eval) & np.isfinite(raw_eval) & (gt_eval > 0)
        delta = gt_eval - raw_eval

        if args.save_diagnostics and len(frame_rows) < args.diagnostic_count:
            save_diag(diag_dir / f"{frame.sequence_id}_{frame.frame_id}.png", raw_eval, gt_eval, valid)

        row = {
            "sequence_id": frame.sequence_id,
            "frame_id": frame.frame_id,
            "frame_index": index,
            "previous_frame_id": previous_by_sequence.get(frame.sequence_id, ""),
            "next_frame_id": next_by_key.get((frame.sequence_id, frame.frame_id), ""),
            "target_path": "",
            "frame_offset": -1,
            "target_h": out_h,
            "target_w": out_w,
            "valid_ratio_lowres": float(valid.mean()) if valid.size else math.nan,
            "valid_pixels_lowres": int(valid.sum()),
            "raw_disp_mae_lowres": mae(raw_eval, gt_eval, valid),
            "raw_bad_1px_lowres": bad(raw_eval, gt_eval, valid, 1.0),
            "raw_bad_3px_lowres": bad(raw_eval, gt_eval, valid, 3.0),
            "delta_mean": float(np.mean(delta[valid])) if np.any(valid) else math.nan,
            "delta_abs_mean": float(np.mean(np.abs(delta[valid]))) if np.any(valid) else math.nan,
            "delta_abs_p95": float(np.percentile(np.abs(delta[valid]), 95)) if np.any(valid) else math.nan,
            "runtime_ms": runtime_ms,
            "shard_size_bytes": 0,
        }
        frame_rows.append(row)
        shard["raw_disp"].append(raw16)
        shard["gt_disp"].append(gt16)
        shard["valid_mask"].append(valid.astype(np.uint8))
        shard["delta_disp_gt_minus_raw"].append(delta.astype(np.float16))
        shard["frame_id"].append(frame.frame_id)
        shard["frame_index"].append(index)
        shard["previous_frame_id"].append(row["previous_frame_id"])
        shard["next_frame_id"].append(row["next_frame_id"])
        shard["row_indices"].append(len(frame_rows) - 1)
        previous_by_sequence[frame.sequence_id] = frame.frame_id

        if (index + 1) % 250 == 0:
            run_log.append(f"[{now()}] processed_frames={index + 1}/{len(frames)}")
            (args.output_root / "run.log").write_text("\n".join(run_log) + "\n")

    flush_shard()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    sequence_rows = [sequence_summary(seq, [r for r in frame_rows if r["sequence_id"] == seq]) for seq in sorted({r["sequence_id"] for r in frame_rows})]
    elapsed = time.perf_counter() - start
    output_size = sum(p.stat().st_size for p in args.output_root.rglob("*") if p.is_file())
    summary = {
        "dataset_root": str(args.dataset_root),
        "frame_metrics_csv": str(args.frame_metrics_csv),
        "output_root": str(args.output_root),
        "target_scale": args.target_scale,
        "min_valid_ratio": args.min_valid_ratio,
        "sequences": len(sequence_rows),
        "frames": len(frame_rows),
        "output_size_mb": output_size / (1024**2),
        "raw_disp_mae_lowres": finite_mean([r["raw_disp_mae_lowres"] for r in frame_rows]),
        "raw_bad_1px_lowres": finite_mean([r["raw_bad_1px_lowres"] for r in frame_rows]),
        "raw_bad_3px_lowres": finite_mean([r["raw_bad_3px_lowres"] for r in frame_rows]),
        "valid_ratio_lowres_mean": finite_mean([r["valid_ratio_lowres"] for r in frame_rows]),
        "delta_abs_mean": finite_mean([r["delta_abs_mean"] for r in frame_rows]),
        "runtime_sec": elapsed,
        "runtime_ms_per_frame_mean": finite_mean([r["runtime_ms"] for r in frame_rows]),
        "estimated_size_for_20621_frames_mb": (output_size / max(1, len(frame_rows)) * 20621) / (1024**2),
        "no_full_resolution_prediction_cache": True,
        "teachers_not_run": ["SAV", "RAFT", "DINO"],
    }
    write_csv(args.output_root / "frame_targets_index.csv", frame_rows)
    write_csv(args.output_root / "sequence_targets_summary.csv", sequence_rows)
    (args.output_root / "aggregate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_root / "README.md").write_text(
        f"""# S2M2 GT Refiner Targets Full

Compact low-resolution supervised targets for tiny temporal refinement.

- Dataset root: `{args.dataset_root}`
- Valid frame source: `{args.frame_metrics_csv}`
- Frames: `{summary['frames']}`
- Sequences: `{summary['sequences']}`
- Target scale: `{args.target_scale}`
- Valid-mask-aware downsample min valid ratio: `{args.min_valid_ratio}`
- Saved arrays per sequence shard: `raw_disp`, `gt_disp`, `valid_mask`, `delta_disp_gt_minus_raw`
- Full-resolution prediction caches: not written
- SAV/RAFT/DINO: not run

Each `.npz` shard is under `targets/<sequence_id>.npz`; `frame_targets_index.csv` maps every frame to `target_path` and `frame_offset`.
This avoids one-file-per-frame allocation waste on filesystems with large block sizes while preserving per-frame metadata and temporal links.
"""
    )
    run_log.extend([f"[{now()}] wrote outputs", f"[{now()}] complete"])
    (args.output_root / "run.log").write_text("\n".join(run_log) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--frame-metrics-csv", type=Path, default=DEFAULT_FRAME_METRICS)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--target-scale", type=float, default=0.25)
    p.add_argument("--min-valid-ratio", type=float, default=0.25)
    p.add_argument("--limit-sequences", type=int, default=0)
    p.add_argument("--limit-frames-per-sequence", type=int, default=0)
    p.add_argument("--save-diagnostics", nargs="?", const=True, default=True, type=parse_bool)
    p.add_argument("--diagnostic-count", type=int, default=6)
    p.add_argument("--overwrite", nargs="?", const=True, default=False, type=parse_bool)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.dataset_root = resolve(args.dataset_root)
    args.frame_metrics_csv = resolve(args.frame_metrics_csv)
    args.output_root = resolve(args.output_root)
    summary = process(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
