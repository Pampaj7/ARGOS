"""Two strictly separate metric namespaces — cache-resolution (144x180) and native-resolution
— plus the corrected (and, for the regression test only, the old broken) GT resize logic.

Cache-resolution and native-resolution metrics are NOT linearly convertible: aggressive
downsampling (native width -> 180, ~7.1x for SCARED-C) changes depth-boundary sharpness and
GT sparsity, so a cache-resolution EPE must never be rescaled and reported as a native-EPE
estimate. Keep the two namespaces separate everywhere (field names, CSV columns, reports).
"""
from __future__ import annotations

import cv2
import numpy as np

from argos_v2.paths import CACHE_HEIGHT, CACHE_WIDTH


def resize_gt_to_cache_naive(gt_disp: np.ndarray, gt_valid: np.ndarray, native_w: int) -> tuple[np.ndarray, np.ndarray]:
    """BROKEN (kept only for the regression test): resizes disparity values and the validity
    mask independently, using nearest-neighbor for the mask. INTER_AREA on the raw disparity
    blends invalid (0-filled) neighbors into any cache cell near a valid/invalid boundary,
    but the nearest-neighbor mask only checks a single source pixel — so contaminated cells
    get marked "valid" anyway. This is the bug found while validating the ARGOS-V2 pilot.
    """
    gt_small = cv2.resize(gt_disp, (CACHE_WIDTH, CACHE_HEIGHT), interpolation=cv2.INTER_AREA) * (CACHE_WIDTH / native_w)
    gt_valid_small = cv2.resize(gt_valid.astype(np.uint8), (CACHE_WIDTH, CACHE_HEIGHT), interpolation=cv2.INTER_NEAREST) > 0
    return gt_small, gt_valid_small


def resize_gt_to_cache_corrected(
    gt_disp: np.ndarray, gt_valid: np.ndarray, native_w: int, coverage_threshold: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Corrected: a cache cell is only marked valid if the FRACTION of valid native pixels
    inside its downsampling box exceeds coverage_threshold — filters out boundary-blended
    cells instead of trusting a single nearest-neighbor sample.
    """
    gt_small = cv2.resize(gt_disp, (CACHE_WIDTH, CACHE_HEIGHT), interpolation=cv2.INTER_AREA) * (CACHE_WIDTH / native_w)
    coverage = cv2.resize(gt_valid.astype(np.float32), (CACHE_WIDTH, CACHE_HEIGHT), interpolation=cv2.INTER_AREA)
    gt_valid_small = coverage > coverage_threshold
    return gt_small, gt_valid_small


def _epe_bad_absrel(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict:
    if mask.sum() == 0:
        return {"epe_px": None, "bad1": None, "bad3": None, "absrel": None, "valid_ratio": 0.0}
    err = np.abs(pred[mask] - gt[mask])
    return {
        "epe_px": float(err.mean()),
        "bad1": float((err > 1).mean()),
        "bad3": float((err > 3).mean()),
        "absrel": float((err / np.maximum(gt[mask], 1e-6)).mean()),
        "valid_ratio": float(mask.mean()),
    }


def compute_cache_metrics(pred_disp_cache: np.ndarray, pred_valid_cache: np.ndarray, gt_disp_native: np.ndarray,
                           gt_valid_native: np.ndarray, native_w: int, coverage_threshold: float = 0.9) -> dict:
    """Cache-resolution (144x180) metrics only. Field names carry an explicit _cache suffix —
    never rename these to imply native-resolution equivalence."""
    gt_small, gt_valid_small = resize_gt_to_cache_corrected(gt_disp_native, gt_valid_native, native_w, coverage_threshold)
    mask = gt_valid_small & (pred_valid_cache > 0)
    r = _epe_bad_absrel(pred_disp_cache.astype(np.float32), gt_small.astype(np.float32), mask)
    valid_pixels = pred_disp_cache[pred_valid_cache > 0]
    return {
        "epe_cache_px": r["epe_px"], "bad1_cache": r["bad1"], "bad3_cache": r["bad3"],
        "absrel_cache": r["absrel"], "valid_ratio_cache": r["valid_ratio"],
        "disparity_min_cache": float(valid_pixels.min()) if valid_pixels.size else None,
        "disparity_median_cache": float(np.median(valid_pixels)) if valid_pixels.size else None,
        "disparity_max_cache": float(valid_pixels.max()) if valid_pixels.size else None,
    }


def compute_native_metrics(pred_disp_native: np.ndarray, gt_disp_native: np.ndarray, gt_valid_native: np.ndarray) -> dict:
    """Native-resolution metrics only, computed with NO cache resizing at all — direct
    pixel-for-pixel comparison at the source resolution. Field names carry an explicit
    _native suffix."""
    mask = gt_valid_native & np.isfinite(pred_disp_native) & (pred_disp_native > 0) & (gt_disp_native > 0)
    r = _epe_bad_absrel(pred_disp_native.astype(np.float32), gt_disp_native.astype(np.float32), mask)
    h, w = gt_disp_native.shape
    return {
        "epe_native_px": r["epe_px"], "bad1_native": r["bad1"], "bad3_native": r["bad3"],
        "valid_ratio_native": r["valid_ratio"], "native_height": h, "native_width": w,
    }
