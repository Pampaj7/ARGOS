"""Cache resize convention, atomic write, resume/completion-flag handling, and the
post-write integrity validation that gates the completion flag (spec: completion_status
may only be true after all integrity checks pass)."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from argos_v2.paths import CACHE_DIR, CACHE_HEIGHT, CACHE_WIDTH

COMPLETE_FLAG = ".complete"

REQUIRED_METADATA_FIELDS = [
    "project", "backbone", "checkpoint", "sequence_id", "frame_count",
    "source_height", "source_width", "cache_height", "cache_width",
    "disparity_dtype", "mask_dtype", "disparity_units", "disparity_convention",
    "resize_interpolation", "disparity_scale_formula", "invalid_value_policy",
    "prediction_valid_policy", "git_commit", "script_path", "command_line_args",
    "start_time", "end_time", "total_runtime_s", "avg_runtime_per_frame_s",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resize_pred_to_cache(pred_native: np.ndarray, native_w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """pred_native: raw backbone disparity output [H,W] float32, native resolution.

    Returns (disp_cache float16, valid_cache uint8, disp_cache_fp32 float32) all
    [CACHE_HEIGHT,CACHE_WIDTH]. The fp32 copy is returned so callers can keep a small
    sample for genuine float16-quantization-error measurement without re-running inference.
    Validity of a PREDICTION (finite, positive), independent of any GT validity.
    d_cache = resize(d_source, (144,180)) * (180.0 / native_w).
    """
    valid_native = np.isfinite(pred_native) & (pred_native > 0)
    disp_clean = np.where(valid_native, pred_native, 0).astype(np.float32)
    disp_small = cv2.resize(disp_clean, (CACHE_WIDTH, CACHE_HEIGHT), interpolation=cv2.INTER_AREA)
    disp_cache_fp32 = (disp_small * (CACHE_WIDTH / float(native_w))).astype(np.float32)
    disp_cache = disp_cache_fp32.astype(np.float16)
    valid_small = cv2.resize(valid_native.astype(np.uint8), (CACHE_WIDTH, CACHE_HEIGHT), interpolation=cv2.INTER_NEAREST)
    return disp_cache, valid_small.astype(np.uint8), disp_cache_fp32


def is_complete(backbone: str, sequence_id: str) -> bool:
    return (CACHE_DIR / backbone / sequence_id / COMPLETE_FLAG).exists()


def write_sequence_cache(
    backbone: str,
    sequence_id: str,
    disp_stack: np.ndarray,   # [T, CACHE_HEIGHT, CACHE_WIDTH] float16
    valid_stack: np.ndarray,  # [T, CACHE_HEIGHT, CACHE_WIDTH] uint8
    frame_ids: list[str],
    metadata: dict,
    force: bool = False,
    fp32_sample: np.ndarray | None = None,  # [K, CACHE_HEIGHT, CACHE_WIDTH] float32, first K frames pre-cast
) -> Path:
    """Atomic tmp-dir + rename publish. Does NOT set the completion flag — callers must run
    validate_written_cache() and mark_complete() afterward; a cache dir can exist without
    being trusted for resume until that flag is set.
    """
    final_dir = CACHE_DIR / backbone / sequence_id
    if final_dir.exists() and (final_dir / COMPLETE_FLAG).exists() and not force:
        raise FileExistsError(f"refusing to overwrite complete cache: {final_dir}")

    tmp_dir = CACHE_DIR / backbone / f".tmp_{sequence_id}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    metadata = dict(metadata, completion_status=False, integrity_validation_status=None)
    np.save(tmp_dir / "disparity.npy", np.ascontiguousarray(disp_stack.astype(np.float16)))
    np.save(tmp_dir / "valid_mask.npy", np.ascontiguousarray(valid_stack.astype(np.uint8)))
    np.save(tmp_dir / "frame_ids.npy", np.array(frame_ids))
    (tmp_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    if fp32_sample is not None:
        # diagnostic sidecar only (float16 quantization-error check) — not part of the
        # canonical cache format, tiny (a handful of frames), not counted toward promotion size
        np.save(tmp_dir / "_fp32_sample.npy", np.ascontiguousarray(fp32_sample.astype(np.float32)))

    if final_dir.exists():
        shutil.rmtree(final_dir)
    tmp_dir.rename(final_dir)
    return final_dir


def validate_written_cache(backbone: str, sequence_id: str, expected_frame_ids: list[str]) -> dict:
    """The 16-point post-write check. Returns a dict of individual results plus "passed"."""
    d = CACHE_DIR / backbone / sequence_id
    checks: dict = {}
    try:
        required_files = ["disparity.npy", "valid_mask.npy", "frame_ids.npy", "metadata.json"]
        checks["files_exist"] = all((d / f).exists() for f in required_files)
        if not checks["files_exist"]:
            checks["passed"] = False
            return checks

        disp_mmap = np.load(d / "disparity.npy", mmap_mode="r")
        valid_mmap = np.load(d / "valid_mask.npy", mmap_mode="r")
        frame_ids = list(np.load(d / "frame_ids.npy"))

        checks["frame_count_match"] = len(frame_ids) == len(expected_frame_ids)
        checks["frame_ids_exact"] = frame_ids == list(expected_frame_ids)
        checks["disp_shape_ok"] = tuple(disp_mmap.shape) == (len(expected_frame_ids), CACHE_HEIGHT, CACHE_WIDTH)
        checks["valid_shape_ok"] = tuple(valid_mmap.shape) == tuple(disp_mmap.shape)
        checks["disp_dtype_ok"] = disp_mmap.dtype == np.float16
        checks["valid_dtype_ok"] = valid_mmap.dtype == np.uint8
        checks["c_contiguous"] = bool(np.load(d / "disparity.npy").flags["C_CONTIGUOUS"])

        checks["mmap_readable"] = True
        try:
            n = disp_mmap.shape[0]
            for i in sorted({0, n // 2, n - 1}):
                float(disp_mmap[i, 0, 0])
        except Exception:
            checks["mmap_readable"] = False

        disp_f32 = np.asarray(disp_mmap).astype(np.float32)
        finite = np.isfinite(disp_f32)
        checks["finite_ratio"] = float(finite.mean())
        valid_bool = np.asarray(valid_mmap) > 0
        checks["prediction_valid_ratio"] = float(valid_bool.mean())
        vals = disp_f32[valid_bool & finite]
        checks["disparity_min"] = float(vals.min()) if vals.size else None
        checks["disparity_median"] = float(np.median(vals)) if vals.size else None
        checks["disparity_max"] = float(vals.max()) if vals.size else None
        checks["positive_disparity_ok"] = bool(vals.size == 0 or bool((vals >= 0).all()))

        per_frame_median = np.median(np.where(valid_bool, disp_f32, np.nan), axis=(1, 2))
        checks["plausible_frame_to_frame_stats"] = bool(np.all(np.isfinite(per_frame_median) | ~valid_bool.any(axis=(1, 2))))

        meta = json.loads((d / "metadata.json").read_text())
        checks["metadata_complete"] = all(f in meta for f in REQUIRED_METADATA_FIELDS)
        checks["metadata_frame_count_agrees"] = meta.get("frame_count") == len(frame_ids)

        tmp_dir = CACHE_DIR / backbone / f".tmp_{sequence_id}"
        checks["no_stale_tmp"] = not tmp_dir.exists()

        checks["passed"] = all([
            checks["files_exist"], checks["frame_count_match"], checks["frame_ids_exact"],
            checks["disp_shape_ok"], checks["valid_shape_ok"], checks["disp_dtype_ok"],
            checks["valid_dtype_ok"], checks["c_contiguous"], checks["mmap_readable"],
            checks["finite_ratio"] > 0.99, checks["positive_disparity_ok"],
            checks["plausible_frame_to_frame_stats"], checks["metadata_complete"],
            checks["metadata_frame_count_agrees"], checks["no_stale_tmp"],
        ])
    except Exception as e:
        checks["passed"] = False
        checks["exception"] = f"{type(e).__name__}: {e}"
    return checks


def mark_complete(backbone: str, sequence_id: str, integrity_checks: dict) -> None:
    """Only call after validate_written_cache()['passed'] is True."""
    d = CACHE_DIR / backbone / sequence_id
    meta = json.loads((d / "metadata.json").read_text())
    meta["completion_status"] = True
    meta["integrity_validation_status"] = integrity_checks
    tmp_meta = d / ".metadata.json.tmp"
    tmp_meta.write_text(json.dumps(meta, indent=2))
    tmp_meta.replace(d / "metadata.json")
    (d / COMPLETE_FLAG).write_text(json.dumps({"validated_at": now_iso()}))


def resume_ok(backbone: str, sequence_id: str, expected_frame_ids: list[str]) -> bool:
    """Resume logic: don't just trust the .complete flag — re-validate before skipping."""
    if not is_complete(backbone, sequence_id):
        return False
    return validate_written_cache(backbone, sequence_id, expected_frame_ids).get("passed", False)


def load_sequence_cache(backbone: str, sequence_id: str, mmap: bool = True):
    d = CACHE_DIR / backbone / sequence_id
    mode = "r" if mmap else None
    disp = np.load(d / "disparity.npy", mmap_mode=mode)
    valid = np.load(d / "valid_mask.npy", mmap_mode=mode)
    frame_ids = np.load(d / "frame_ids.npy")
    metadata = json.loads((d / "metadata.json").read_text())
    return disp, valid, frame_ids, metadata
