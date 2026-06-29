from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


Array = np.ndarray
FlowLoader = Callable[[str, str], Array]
MapLoader = Callable[[str, str], Array]


@dataclass(frozen=True)
class BaselineResult:
    predictions: list[Array]
    postprocess_ms_per_frame: float


def fixed_ema_sequence(raw: Sequence[Array], alpha: float) -> BaselineResult:
    start = time.perf_counter()
    fused: list[Array] = []
    prev: Array | None = None
    for cur in raw:
        cur_f = cur.astype(np.float32, copy=False)
        if prev is None:
            out = cur_f.copy()
        else:
            out = alpha * cur_f + (1.0 - alpha) * prev
        out = out.astype(np.float32, copy=False)
        fused.append(out)
        prev = out
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return BaselineResult(fused, elapsed_ms / max(len(fused), 1))


def _warp_disparity_numpy_fallback(disp: Array, flow: Array) -> Array:
    # Matches scripts.temporal_refinement.lib.flow.warp_disp semantics:
    # sample the disparity at grid + flow with border padding.
    h, w = disp.shape
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    sample_x = np.clip(xx + flow[..., 0].astype(np.float32, copy=False), 0.0, float(w - 1))
    sample_y = np.clip(yy + flow[..., 1].astype(np.float32, copy=False), 0.0, float(h - 1))
    disp_f = disp.astype(np.float32, copy=False)
    try:
        from scipy import ndimage

        return ndimage.map_coordinates(
            disp_f,
            [sample_y, sample_x],
            order=1,
            mode="nearest",
        ).astype(np.float32)
    except ModuleNotFoundError:
        x0 = np.floor(sample_x).astype(np.int64)
        y0 = np.floor(sample_y).astype(np.int64)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)
        wx = sample_x - x0.astype(np.float32)
        wy = sample_y - y0.astype(np.float32)
        top = (1.0 - wx) * disp_f[y0, x0] + wx * disp_f[y0, x1]
        bottom = (1.0 - wx) * disp_f[y1, x0] + wx * disp_f[y1, x1]
        return ((1.0 - wy) * top + wy * bottom).astype(np.float32)


def warp_disparity_numpy(disp: Array, flow: Array, device: str = "auto") -> Array:
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(f"Expected flow shape HxWx2, got {flow.shape}")
    if disp.shape != flow.shape[:2]:
        raise ValueError(f"Disparity shape {disp.shape} does not match flow shape {flow.shape[:2]}")
    if device in {"numpy", "scipy"}:
        return _warp_disparity_numpy_fallback(disp, flow)
    try:
        import torch
        from flow import warp_disp
    except ModuleNotFoundError:
        return _warp_disparity_numpy_fallback(disp, flow)

    if device == "auto":
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)
        if device_obj.type == "cuda" and not torch.cuda.is_available():
            device_obj = torch.device("cpu")
    disp_t = torch.from_numpy(disp.astype(np.float32, copy=False))[None, None].to(device_obj)
    flow_t = torch.from_numpy(flow.astype(np.float32, copy=False)).permute(2, 0, 1)[None].to(device_obj)
    with torch.no_grad():
        warped = warp_disp(disp_t, flow_t)
    return warped[0, 0].detach().float().cpu().numpy().astype(np.float32)


def raft_warped_ema_sequence(
    raw: Sequence[Array],
    frame_ids: Sequence[str],
    flow_loader: FlowLoader,
    alpha: float,
    warp_device: str = "auto",
) -> BaselineResult:
    start = time.perf_counter()
    fused: list[Array] = []
    prev: Array | None = None
    for idx, cur in enumerate(raw):
        cur_f = cur.astype(np.float32, copy=False)
        if prev is None:
            out = cur_f.copy()
        else:
            flow = flow_loader(frame_ids[idx - 1], frame_ids[idx])
            warped_prev = warp_disparity_numpy(prev, flow, device=warp_device)
            out = alpha * cur_f + (1.0 - alpha) * warped_prev
        out = out.astype(np.float32, copy=False)
        fused.append(out)
        prev = out
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return BaselineResult(fused, elapsed_ms / max(len(fused), 1))


def confidence_reset_warped_ema_sequence(
    raw: Sequence[Array],
    frame_ids: Sequence[str],
    flow_loader: FlowLoader,
    confidence_loader: MapLoader,
    occlusion_loader: MapLoader,
    alpha: float,
    warp_device: str = "auto",
) -> BaselineResult:
    start = time.perf_counter()
    fused: list[Array] = []
    prev: Array | None = None
    for idx, cur in enumerate(raw):
        cur_f = cur.astype(np.float32, copy=False)
        if prev is None:
            out = cur_f.copy()
        else:
            prev_id, cur_id = frame_ids[idx - 1], frame_ids[idx]
            flow = flow_loader(prev_id, cur_id)
            warped_prev = warp_disparity_numpy(prev, flow, device=warp_device)
            confidence = np.clip(confidence_loader(prev_id, cur_id).astype(np.float32, copy=False), 0.0, 1.0)
            occlusion = occlusion_loader(prev_id, cur_id).astype(np.float32, copy=False)
            occlusion = np.clip(occlusion, 0.0, 1.0)
            memory_weight = (1.0 - alpha) * confidence * (1.0 - occlusion)
            raw_weight = 1.0 - memory_weight
            out = raw_weight * cur_f + memory_weight * warped_prev
        out = out.astype(np.float32, copy=False)
        fused.append(out)
        prev = out
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return BaselineResult(fused, elapsed_ms / max(len(fused), 1))


def conservative_adaptive_ema_sequence(
    raw: Sequence[Array],
    frame_ids: Sequence[str],
    flow_loader: FlowLoader,
    confidence_loader: MapLoader,
    occlusion_loader: MapLoader,
    alpha_min: float = 0.40,
    alpha_max: float = 0.80,
    diff_scale_px: float = 3.0,
    w_conf: float = 1.0,
    w_occ: float = 1.0,
    w_diff: float = 1.0,
    warp_device: str = "auto",
) -> BaselineResult:
    start = time.perf_counter()
    fused: list[Array] = []
    prev: Array | None = None
    denom = max(w_conf + w_occ + w_diff, 1e-6)
    for idx, cur in enumerate(raw):
        cur_f = cur.astype(np.float32, copy=False)
        if prev is None:
            out = cur_f.copy()
        else:
            prev_id, cur_id = frame_ids[idx - 1], frame_ids[idx]
            flow = flow_loader(prev_id, cur_id)
            warped_prev = warp_disparity_numpy(prev, flow, device=warp_device)
            confidence = np.clip(confidence_loader(prev_id, cur_id).astype(np.float32, copy=False), 0.0, 1.0)
            occlusion = np.clip(occlusion_loader(prev_id, cur_id).astype(np.float32, copy=False), 0.0, 1.0)
            conf_risk = 1.0 - confidence
            occ_risk = occlusion
            diff_risk = np.clip(np.abs(cur_f - warped_prev) / max(diff_scale_px, 1e-6), 0.0, 1.0)
            risk = w_conf * conf_risk + w_occ * occ_risk + w_diff * diff_risk
            risk = np.clip(risk / denom, 0.0, 1.0)
            alpha_map = alpha_min + (alpha_max - alpha_min) * risk
            out = alpha_map * cur_f + (1.0 - alpha_map) * warped_prev
        out = out.astype(np.float32, copy=False)
        fused.append(out)
        prev = out
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return BaselineResult(fused, elapsed_ms / max(len(fused), 1))


def disparity_gradient_magnitude(disp: Array) -> Array:
    disp_f = disp.astype(np.float32, copy=False)
    grad = np.zeros_like(disp_f, dtype=np.float32)
    gx = np.abs(disp_f[:, 1:] - disp_f[:, :-1])
    gy = np.abs(disp_f[1:, :] - disp_f[:-1, :])
    grad[:, 1:] = np.maximum(grad[:, 1:], gx)
    grad[:, :-1] = np.maximum(grad[:, :-1], gx)
    grad[1:, :] = np.maximum(grad[1:, :], gy)
    grad[:-1, :] = np.maximum(grad[:-1, :], gy)
    return grad


def adaptive_no_raft_diff_sequence(
    raw: Sequence[Array],
    alpha_min: float = 0.30,
    alpha_max: float = 0.80,
    diff_scale_px: float = 3.0,
) -> BaselineResult:
    start = time.perf_counter()
    fused: list[Array] = []
    prev: Array | None = None
    for cur in raw:
        cur_f = cur.astype(np.float32, copy=False)
        if prev is None:
            out = cur_f.copy()
        else:
            diff_risk = np.clip(np.abs(cur_f - prev) / max(diff_scale_px, 1e-6), 0.0, 1.0)
            alpha_map = alpha_min + (alpha_max - alpha_min) * diff_risk
            out = alpha_map * cur_f + (1.0 - alpha_map) * prev
        out = out.astype(np.float32, copy=False)
        fused.append(out)
        prev = out
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return BaselineResult(fused, elapsed_ms / max(len(fused), 1))


def adaptive_no_raft_diff_grad_sequence(
    raw: Sequence[Array],
    alpha_min: float = 0.30,
    alpha_max: float = 0.85,
    diff_scale_px: float = 3.0,
    grad_scale_px: float = 5.0,
    w_diff: float = 1.0,
    w_grad: float = 0.5,
) -> BaselineResult:
    start = time.perf_counter()
    fused: list[Array] = []
    prev: Array | None = None
    denom = max(w_diff + w_grad, 1e-6)
    for cur in raw:
        cur_f = cur.astype(np.float32, copy=False)
        if prev is None:
            out = cur_f.copy()
        else:
            diff_risk = np.clip(np.abs(cur_f - prev) / max(diff_scale_px, 1e-6), 0.0, 1.0)
            grad_risk = np.clip(disparity_gradient_magnitude(cur_f) / max(grad_scale_px, 1e-6), 0.0, 1.0)
            risk = np.clip((w_diff * diff_risk + w_grad * grad_risk) / denom, 0.0, 1.0)
            alpha_map = alpha_min + (alpha_max - alpha_min) * risk
            out = alpha_map * cur_f + (1.0 - alpha_map) * prev
        out = out.astype(np.float32, copy=False)
        fused.append(out)
        prev = out
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return BaselineResult(fused, elapsed_ms / max(len(fused), 1))
