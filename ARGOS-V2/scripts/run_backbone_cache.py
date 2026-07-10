#!/usr/bin/env python3
"""Single (backbone, sequence) cache-building job. Meant to run as its own OS subprocess
(see run_full.py orchestrator) — avoids cross-repo Python import collisions and lets each
job pin its own CUDA_VISIBLE_DEVICES.

Writes the cache, self-validates it (16-point check), and only sets the completion flag
if validation passes — a failed validation leaves the cache dir present but incomplete,
so the next run (resume) will detect and retry it rather than silently trusting bad output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from argos_v2.backbones import build_predictor, inference_resolution
from argos_v2.cache_io import (
    mark_complete, now_iso, resize_pred_to_cache, resume_ok, validate_written_cache, write_sequence_cache,
)
from argos_v2.paths import ARGOS_ROOT, CACHE_HEIGHT, CACHE_WIDTH
from argos_v2.scared_c_data import load_frame_lr, load_sequence_info

CHECKPOINT_ROOTS = {
    "RAFT-Stereo": ARGOS_ROOT / "external/frame_stereo_repos/RAFT-Stereo/models/raftstereo-middlebury.pth",
    "StereoAnywhere": ARGOS_ROOT / "external/frame_stereo_repos/stereoanywhere/weights/stereoanywhere_sceneflow.pth",
    "S2M2-S": ARGOS_ROOT / "external/frame_stereo_repos/s2m2/weights/pretrain_weights/CH128NTR1.pth",
    "Fast-FoundationStereo": ARGOS_ROOT / "external/frame_stereo_repos/Fast-FoundationStereo/weights/onnx/20_30_48/320x736/20_30_48_iters_4_res_320x736.onnx",
}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ARGOS_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def checkpoint_sha256_short(path: Path, n_bytes: int = 8_000_000) -> str:
    """Hash the first n_bytes only — full-file hashing of multi-GB checkpoints is wasted
    I/O for a metadata identifier; a partial hash still catches the common case of someone
    swapping a checkpoint file at the same path."""
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            h.update(f.read(n_bytes))
        return h.hexdigest()[:16]
    except Exception:
        return "unavailable"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0, help="smoke-test cap; writes to a separate _smoke backbone namespace, never touches the real cache")
    args = ap.parse_args()

    backbone_key = f"_smoke_{args.backbone}" if args.max_frames else args.backbone
    info = load_sequence_info(args.sequence)
    if args.max_frames:
        info.frame_ids = info.frame_ids[: args.max_frames]

    if not args.force and resume_ok(backbone_key, args.sequence, info.frame_ids):
        print(f"SKIP (already complete + revalidated): {backbone_key}/{args.sequence}", flush=True)
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    method, checkpoint, predict = build_predictor(args.backbone, device)

    n = len(info.frame_ids)
    disp_stack = np.empty((n, CACHE_HEIGHT, CACHE_WIDTH), dtype=np.float16)
    valid_stack = np.empty((n, CACHE_HEIGHT, CACHE_WIDTH), dtype=np.uint8)
    runtimes_s = []
    native_shape = None
    FP32_SAMPLE_N = 5
    fp32_samples = []

    start_time = now_iso()
    t_start = time.perf_counter()
    for i, frame_id in enumerate(info.frame_ids):
        left, right = load_frame_lr(info, frame_id)
        if native_shape is None:
            native_shape = left.shape[:2]  # (H, W)
        pred, ms = predict(left, right)
        runtimes_s.append(ms / 1000.0)
        disp_c, valid_c, disp_fp32 = resize_pred_to_cache(pred, native_w=left.shape[1])
        disp_stack[i] = disp_c
        valid_stack[i] = valid_c
        if i < FP32_SAMPLE_N:
            fp32_samples.append(disp_fp32)
        if (i + 1) % 200 == 0 or i == n - 1:
            print(f"{args.backbone}/{args.sequence}: {i + 1}/{n} frames", flush=True)
    total_s = time.perf_counter() - t_start
    end_time = now_iso()

    native_h, native_w = native_shape
    inf_h, inf_w, inf_note = inference_resolution(args.backbone, native_h, native_w)
    ckpt_path = CHECKPOINT_ROOTS.get(args.backbone)
    runtimes_arr = np.array(runtimes_s)

    metadata = {
        "project": "ARGOS v2",
        "backbone": args.backbone,
        "method": method,
        "checkpoint": checkpoint,
        "checkpoint_path": str(ckpt_path) if ckpt_path else "n/a",
        "checkpoint_sha256_partial": checkpoint_sha256_short(ckpt_path) if ckpt_path and ckpt_path.exists() else "n/a",
        "sequence_id": args.sequence,
        "frame_count": n,
        "frame_ids_source": str(info.seq_dir / "manifest.csv"),
        "source_height": native_h,
        "source_width": native_w,
        "model_inference_height": inf_h,
        "model_inference_width": inf_w,
        "model_inference_note": inf_note,
        "cache_height": CACHE_HEIGHT,
        "cache_width": CACHE_WIDTH,
        "disparity_dtype": "float16",
        "mask_dtype": "uint8",
        "disparity_units": "pixels_at_cache_resolution",
        "disparity_convention": "positive_left_disparity",
        "resize_interpolation": "INTER_AREA (disparity), INTER_NEAREST (validity)",
        "disparity_scale_formula": "d_cache = resize(d_source, (144,180)) * (180.0 / source_width)",
        "invalid_value_policy": "invalid native pixels (non-finite or <=0) are zeroed before resize to avoid blending into valid neighbors",
        "prediction_valid_policy": "prediction validity only (isfinite & >0 at native res, nearest-resized); independent of SCARED-C GT validity",
        "git_commit": git_commit(),
        "script_path": str(Path(__file__).resolve()),
        "command_line_args": sys.argv[1:],
        "start_time": start_time,
        "end_time": end_time,
        "total_runtime_s": total_s,
        "avg_runtime_per_frame_s": float(runtimes_arr.mean()),
        "median_runtime_per_frame_s": float(np.median(runtimes_arr)),
        "p95_runtime_per_frame_s": float(np.percentile(runtimes_arr, 95)),
        "device": str(device),
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated() / 1024**2) if device.type == "cuda" else None,
    }

    # resume_ok() already ran above and gated whether we got this far at all — if we're here,
    # any existing cache dir at this path was already determined untrustworthy (or --force was
    # passed), so the write must be allowed to replace it. write_sequence_cache()'s own guard
    # only checks flag-file *existence*, not validity, so it must not re-litigate that decision.
    write_sequence_cache(
        backbone_key, args.sequence, disp_stack, valid_stack, info.frame_ids, metadata, force=True,
        fp32_sample=np.stack(fp32_samples, axis=0) if fp32_samples else None,
    )

    checks = validate_written_cache(backbone_key, args.sequence, info.frame_ids)
    if checks.get("passed"):
        mark_complete(backbone_key, args.sequence, checks)
        print(f"DONE {backbone_key}/{args.sequence} ({total_s:.1f}s, {n} frames) — validated OK", flush=True)
        return 0
    else:
        print(f"FAILED VALIDATION {backbone_key}/{args.sequence}: {json.dumps(checks, default=str)}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
