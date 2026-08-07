"""Small NumPy reference implementation of audit metrics; never model inputs."""
from __future__ import annotations

import numpy as np

EPS = 1e-6


def _flow(flow: np.ndarray) -> np.ndarray:
    value = np.asarray(flow, dtype=np.float32)
    # Prefer HWC: a two-pixel-high image is otherwise ambiguous with CHW.
    if value.ndim == 3 and value.shape[-1] != 2 and value.shape[0] == 2:
        value = np.moveaxis(value, 0, -1)
    if value.ndim != 3 or value.shape[-1] != 2:
        raise ValueError("flow must be [H,W,2] or [2,H,W]")
    return value


def pull_warp(source: np.ndarray, flow_target_to_source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear target-to-source pull warp matching geometry_v1 coordinates."""
    source = np.asarray(source, dtype=np.float32)
    if source.ndim not in (2, 3):
        raise ValueError("source must be [H,W] or [H,W,C]")
    flow = _flow(flow_target_to_source)
    height, width = source.shape[:2]
    if flow.shape[:2] != (height, width):
        raise ValueError("source/flow shape mismatch")
    yy, xx = np.indices((height, width), dtype=np.float32)
    x, y = xx + flow[..., 0], yy + flow[..., 1]
    inbounds = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1, y1 = np.clip(x0 + 1, 0, width - 1), np.clip(y0 + 1, 0, height - 1)
    x0, y0 = np.clip(x0, 0, width - 1), np.clip(y0, 0, height - 1)
    wx, wy = x - np.floor(x), y - np.floor(y)
    def sample(array: np.ndarray) -> np.ndarray:
        return ((1 - wx)[..., None] * (1 - wy)[..., None] * array[y0, x0] +
                wx[..., None] * (1 - wy)[..., None] * array[y0, x1] +
                (1 - wx)[..., None] * wy[..., None] * array[y1, x0] +
                wx[..., None] * wy[..., None] * array[y1, x1])
    if source.ndim == 2:
        result = sample(source[..., None])[..., 0]
    else:
        result = sample(source)
    result = result.astype(np.float32, copy=False)
    result[~inbounds] = np.nan
    return result, inbounds


def _mask(*values: np.ndarray | None) -> np.ndarray:
    first = next(value for value in values if value is not None)
    result = np.ones(np.asarray(first).shape[:2], dtype=bool)
    for value in values:
        if value is not None:
            result &= np.asarray(value, dtype=bool)
    return result


def _finite_positive(*values: np.ndarray) -> np.ndarray:
    result = np.ones(np.asarray(values[0]).shape, dtype=bool)
    for value in values:
        result &= np.isfinite(value) & (value > 0)
    return result


def gt_tce(current_prediction: np.ndarray, past_prediction: np.ndarray, current_gt: np.ndarray, past_gt: np.ndarray,
           flow_current_to_past: np.ndarray, *, support: np.ndarray | None = None, current_gt_valid: np.ndarray | None = None,
           past_gt_valid: np.ndarray | None = None, fb_support: np.ndarray | None = None, epsilon: float = EPS) -> dict[str, float | int]:
    """GT temporal-change error and its reference-disparity normalization."""
    predicted_past, inbounds = pull_warp(past_prediction, flow_current_to_past)
    gt_past, _ = pull_warp(past_gt, flow_current_to_past)
    source_valid_float, _ = pull_warp(np.asarray(past_gt_valid if past_gt_valid is not None else np.ones_like(past_gt), dtype=np.float32), flow_current_to_past)
    valid = _mask(inbounds, support, current_gt_valid, fb_support)
    valid &= source_valid_float >= 0.999
    valid &= _finite_positive(current_prediction, predicted_past, current_gt, gt_past)
    residual = np.abs((current_prediction - predicted_past) - (current_gt - gt_past))
    count = int(valid.sum())
    value = float(residual[valid].mean()) if count else float("nan")
    reference = float(np.abs(current_gt[valid]).mean()) if count else float("nan")
    return {"gt_tce": value, "rgt_tce": value / (reference + epsilon) if count else float("nan"), "count": count,
            "error_sum": float(residual[valid].sum()) if count else 0.0, "reference_mean": reference}


def nr_tce(current_prediction: np.ndarray, past_prediction: np.ndarray, flow_current_to_past: np.ndarray, *,
           support: np.ndarray | None = None, current_valid: np.ndarray | None = None, past_valid: np.ndarray | None = None,
           fb_support: np.ndarray | None = None, trim_fraction: float = 0.1, epsilon: float = EPS) -> dict[str, float | int]:
    """Robust 10%-trimmed motion-compensated normalized disparity change."""
    warped, inbounds = pull_warp(past_prediction, flow_current_to_past)
    past_valid_float, _ = pull_warp(np.asarray(past_valid if past_valid is not None else np.ones_like(past_prediction), dtype=np.float32), flow_current_to_past)
    valid = _mask(inbounds, support, current_valid, fb_support)
    valid &= past_valid_float >= 0.999
    valid &= _finite_positive(current_prediction, warped)
    values = np.abs(current_prediction[valid] - warped[valid]) / (0.5 * (np.abs(current_prediction[valid]) + np.abs(warped[valid])) + epsilon)
    if not len(values):
        return {"nr_tce": float("nan"), "count": 0, "trimmed_count": 0, "error_sum": 0.0}
    values.sort(); cut = int(np.floor(len(values) * trim_fraction)); kept = values[cut:len(values) - cut] if 2 * cut < len(values) else values
    return {"nr_tce": float(kept.mean()), "count": int(len(values)), "trimmed_count": int(len(kept)), "error_sum": float(kept.sum())}


def stereo_photo(left_rgb: np.ndarray, right_rgb: np.ndarray, disparity: np.ndarray, *, support: np.ndarray | None = None,
                 epsilon: float = 1e-3) -> dict[str, float | int]:
    """Current-frame Charbonnier residual; positive-left disparity samples right at x-d."""
    flow = np.stack((-np.asarray(disparity, dtype=np.float32), np.zeros_like(disparity, dtype=np.float32)), axis=-1)
    warped, inbounds = pull_warp(right_rgb, flow)
    left = np.asarray(left_rgb, dtype=np.float32); right = np.asarray(right_rgb, dtype=np.float32)
    if max(float(np.nanmax(left)), float(np.nanmax(right))) > 1.5:
        left, warped = left / 255.0, warped / 255.0
    valid = _mask(inbounds, support) & np.isfinite(disparity) & (disparity > 0) & np.isfinite(warped).all(axis=-1)
    residual = np.sqrt(np.square(left - warped).mean(axis=-1) + epsilon * epsilon)
    return {"stereo_photo": float(residual[valid].mean()) if valid.any() else float("nan"), "count": int(valid.sum())}


def temporal_photo(current_rgb: np.ndarray, past_rgb: np.ndarray, flow_current_to_past: np.ndarray, *, support: np.ndarray | None = None,
                   epsilon: float = 1e-3) -> dict[str, float | int]:
    """Motion-quality stratifier, not a disparity-accuracy metric."""
    warped, inbounds = pull_warp(past_rgb, flow_current_to_past)
    current = np.asarray(current_rgb, dtype=np.float32)
    if max(float(np.nanmax(current)), float(np.nanmax(warped))) > 1.5:
        current, warped = current / 255.0, warped / 255.0
    valid = _mask(inbounds, support) & np.isfinite(warped).all(axis=-1)
    residual = np.sqrt(np.square(current - warped).mean(axis=-1) + epsilon * epsilon)
    return {"temporal_photo": float(residual[valid].mean()) if valid.any() else float("nan"), "count": int(valid.sum())}
