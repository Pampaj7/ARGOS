from __future__ import annotations

from typing import Any

import numpy as np


def finite_mean(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size else float("nan")


def finite_ratio(num: np.ndarray, den: np.ndarray) -> float:
    den_mean = finite_mean(den)
    if not np.isfinite(den_mean) or abs(den_mean) < 1e-8:
        return float("nan")
    return float(finite_mean(num) / den_mean)


def gradient_magnitude(value: np.ndarray) -> np.ndarray:
    """Simple finite-difference gradient magnitude with HxW output."""
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected HxW array, got {arr.shape}")
    gx = np.zeros_like(arr, dtype=np.float32)
    gy = np.zeros_like(arr, dtype=np.float32)
    gx[:, :-1] = arr[:, 1:] - arr[:, :-1]
    if arr.shape[1] > 1:
        gx[:, -1] = gx[:, -2]
    gy[:-1, :] = arr[1:, :] - arr[:-1, :]
    if arr.shape[0] > 1:
        gy[-1, :] = gy[-2, :]
    grad = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    grad[~np.isfinite(arr)] = np.nan
    return grad


def grayscale_gradient(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"Expected RGB HxWx3 array, got {rgb.shape}")
    rgb_f = rgb[..., :3].astype(np.float32)
    gray = 0.299 * rgb_f[..., 0] + 0.587 * rgb_f[..., 1] + 0.114 * rgb_f[..., 2]
    return gradient_magnitude(gray)


def percentile_mask(values: np.ndarray, valid: np.ndarray, percentile: float) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32)
    mask = valid & np.isfinite(vals)
    if not np.any(mask):
        return np.zeros_like(valid, dtype=bool)
    threshold = float(np.nanpercentile(vals[mask], percentile))
    return mask & (vals > threshold)


def pearson_corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & np.isfinite(a) & np.isfinite(b)
    if int(valid.sum()) < 2:
        return float("nan")
    av = a[valid].astype(np.float64)
    bv = b[valid].astype(np.float64)
    av -= av.mean()
    bv -= bv.mean()
    denom = float(np.sqrt(np.sum(av * av) * np.sum(bv * bv)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(av * bv) / denom)


def warp_array(value: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warp HxW or HxWxC arrays with flow semantics matching grid + flow."""
    arr = np.asarray(value, dtype=np.float32)
    flow_f = np.asarray(flow, dtype=np.float32)
    if flow_f.ndim != 3 or flow_f.shape[2] != 2:
        raise ValueError(f"Expected flow HxWx2, got {flow_f.shape}")
    h, w = flow_f.shape[:2]
    if arr.shape[:2] != (h, w):
        raise ValueError(f"Array shape {arr.shape[:2]} does not match flow shape {(h, w)}")
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    sample_y = np.clip(yy + flow_f[..., 1], 0.0, float(h - 1))
    sample_x = np.clip(xx + flow_f[..., 0], 0.0, float(w - 1))
    try:
        from scipy import ndimage

        if arr.ndim == 2:
            return ndimage.map_coordinates(
                arr, [sample_y, sample_x], order=1, mode="nearest", prefilter=False
            ).astype(np.float32)
        channels = [
            ndimage.map_coordinates(
                arr[..., c], [sample_y, sample_x], order=1, mode="nearest", prefilter=False
            )
            for c in range(arr.shape[2])
        ]
        return np.stack(channels, axis=-1).astype(np.float32)
    except ModuleNotFoundError:
        x0 = np.floor(sample_x).astype(np.int64)
        y0 = np.floor(sample_y).astype(np.int64)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)
        wx = sample_x - x0.astype(np.float32)
        wy = sample_y - y0.astype(np.float32)

        def interp(channel: np.ndarray) -> np.ndarray:
            top = (1.0 - wx) * channel[y0, x0] + wx * channel[y0, x1]
            bottom = (1.0 - wx) * channel[y1, x0] + wx * channel[y1, x1]
            return ((1.0 - wy) * top + wy * bottom).astype(np.float32)

        if arr.ndim == 2:
            return interp(arr)
        return np.stack([interp(arr[..., c]) for c in range(arr.shape[2])], axis=-1).astype(np.float32)


def forward_backward_occlusion_mask(flow_fwd: np.ndarray, flow_bwd: np.ndarray, tau_fb: float = 1.5) -> np.ndarray:
    warped_bwd = warp_array(flow_bwd, flow_fwd)
    fb = np.asarray(flow_fwd, dtype=np.float32) + warped_bwd.astype(np.float32)
    err = np.sqrt(np.sum(fb * fb, axis=2))
    return np.isfinite(err) & (err > float(tau_fb))


def frame_artifact_metrics(
    *,
    pred: np.ndarray,
    raw: np.ndarray,
    gt: np.ndarray,
    valid_mask: np.ndarray,
    rgb: np.ndarray,
) -> dict[str, float]:
    pred_g = gradient_magnitude(pred)
    raw_g = gradient_magnitude(raw)
    gt_g = gradient_magnitude(gt)
    rgb_g = grayscale_gradient(rgb)
    valid = valid_mask & np.isfinite(pred) & np.isfinite(raw) & np.isfinite(gt)
    raw_edge_mask = percentile_mask(raw_g, valid, 90.0)
    gt_edge_p90 = percentile_mask(gt_g, valid, 90.0)
    gt_edge_p80 = percentile_mask(gt_g, valid, 80.0)
    rgb_edge_p80 = percentile_mask(rgb_g, valid, 80.0)
    err = np.abs(pred.astype(np.float32) - gt.astype(np.float32))
    return {
        "edge_sharpness_ratio_raw": finite_ratio(pred_g[valid], raw_g[valid]),
        "edge_sharpness_ratio_raw_edges": finite_ratio(pred_g[raw_edge_mask], raw_g[raw_edge_mask]),
        "boundary_disp_mae_px": finite_mean(err[gt_edge_p90]),
        "boundary_disp_mae_px_p80": finite_mean(err[gt_edge_p80]),
        "rgb_disp_edge_corr": pearson_corr(rgb_g, pred_g, valid),
        "rgb_disp_edge_corr_rgb_edges": pearson_corr(rgb_g, pred_g, rgb_edge_p80),
    }


def pair_artifact_metrics(
    *,
    prev_pred: np.ndarray,
    cur_pred: np.ndarray,
    cur_raw: np.ndarray,
    prev_gt: np.ndarray,
    cur_gt: np.ndarray,
    prev_valid_mask: np.ndarray,
    cur_valid_mask: np.ndarray,
    flow_fwd: np.ndarray,
    flow_bwd: np.ndarray,
    tau_fb: float = 1.5,
) -> dict[str, float | int]:
    warped_prev = warp_array(prev_pred, flow_fwd)
    valid_cur = cur_valid_mask & np.isfinite(cur_pred) & np.isfinite(cur_raw) & np.isfinite(cur_gt)
    abs_pred_raw = np.abs(cur_pred.astype(np.float32) - cur_raw.astype(np.float32))
    abs_pred_gt = np.abs(cur_pred.astype(np.float32) - cur_gt.astype(np.float32))

    out: dict[str, float | int] = {}
    for tau in [2.0, 5.0]:
        motion = valid_cur & np.isfinite(warped_prev) & (np.abs(cur_raw - warped_prev) > tau)
        suffix = "tau2" if tau == 2.0 else "tau5"
        out[f"ghosting_score_px_{suffix}"] = finite_mean(abs_pred_raw[motion])
        out[f"ghosting_gt_error_px_{suffix}"] = finite_mean(abs_pred_gt[motion])

    occ_mask = forward_backward_occlusion_mask(flow_fwd, flow_bwd, tau_fb=tau_fb)
    out["occlusion_disp_mae_px"] = finite_mean(abs_pred_gt[valid_cur & occ_mask])

    lag_mask = prev_valid_mask & cur_valid_mask & np.isfinite(prev_gt) & np.isfinite(cur_gt) & np.isfinite(cur_pred)
    err_current = finite_mean(np.abs(cur_pred.astype(np.float32) - cur_gt.astype(np.float32))[lag_mask])
    err_prev_gt = finite_mean(np.abs(cur_pred.astype(np.float32) - prev_gt.astype(np.float32))[lag_mask])
    lagged = int(np.isfinite(err_current) and np.isfinite(err_prev_gt) and err_prev_gt < err_current)
    out["lagged_frame"] = lagged
    out["lag_error_margin_px"] = float(err_current - err_prev_gt) if lagged else float("nan")
    return out


def nanmean_rows(rows: list[dict[str, Any]], key: str) -> float:
    vals = []
    for row in rows:
        try:
            value = float(row.get(key, float("nan")))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            vals.append(value)
    return float(np.mean(vals)) if vals else float("nan")


def lag_rate(rows: list[dict[str, Any]]) -> float:
    vals = []
    for row in rows:
        value = row.get("lagged_frame", "")
        if value == "":
            continue
        vals.append(int(value))
    return float(np.mean(vals)) if vals else float("nan")
