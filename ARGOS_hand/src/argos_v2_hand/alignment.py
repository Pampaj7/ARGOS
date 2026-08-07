"""Causal BiDA-style target-to-source tensor alignment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class WarpResult:
    warped: torch.Tensor
    support: torch.Tensor
    sampled_source_valid: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class ForwardBackwardResult:
    error: torch.Tensor
    support: torch.Tensor
    valid: torch.Tensor
    confidence: torch.Tensor
    threshold: torch.Tensor


@dataclass(frozen=True)
class PhotometricResult:
    aligned_source_rgb: torch.Tensor
    l1_residual: torch.Tensor
    robust_normalized_residual: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class TemporalDisparityEvidence:
    aligned_past_disparity: torch.Tensor
    warp_support: torch.Tensor
    aligned_validity: torch.Tensor
    forward_backward_error: torch.Tensor
    forward_backward_confidence: torch.Tensor
    photometric_residual: torch.Tensor
    absolute_disparity_disagreement: torch.Tensor
    signed_disparity_disagreement: torch.Tensor
    flow_magnitude: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _flow_bchw(flow: torch.Tensor) -> torch.Tensor:
    if flow.ndim != 4:
        raise ValueError(f"flow must be rank 4, got {tuple(flow.shape)}")
    if flow.shape[1] == 2:
        return flow
    if flow.shape[-1] == 2:
        return flow.permute(0, 3, 1, 2)
    raise ValueError(f"flow must be [B,2,H,W] or [B,H,W,2], got {tuple(flow.shape)}")


def resize_flow(flow: torch.Tensor, size: tuple[int, int], *, mode: Literal["bilinear", "nearest"] = "bilinear") -> torch.Tensor:
    flow = _flow_bchw(flow)
    source_h, source_w = flow.shape[-2:]
    target_h, target_w = size
    if (source_h, source_w) == (target_h, target_w):
        return flow
    kwargs = {"mode": mode}
    if mode == "bilinear":
        kwargs["align_corners"] = True
    resized = F.interpolate(flow, size=size, **kwargs)
    scale = resized.new_tensor([target_w / source_w, target_h / source_h]).view(1, 2, 1, 1)
    return resized * scale


def causal_warp(source: torch.Tensor, flow_target_to_source: torch.Tensor, *, source_valid: torch.Tensor | None = None, mode: Literal["bilinear", "nearest"] = "bilinear", valid_threshold: float = 0.999) -> WarpResult:
    if source.ndim != 4:
        raise ValueError(f"source must be [B,C,H,W], got {tuple(source.shape)}")
    flow = _flow_bchw(flow_target_to_source).to(device=source.device, dtype=source.dtype)
    b, _c, h, w = source.shape
    if flow.shape != (b, 2, h, w):
        raise ValueError(f"flow {tuple(flow.shape)} incompatible with source {tuple(source.shape)}")
    ys, xs = torch.meshgrid(torch.arange(h, device=source.device, dtype=source.dtype), torch.arange(w, device=source.device, dtype=source.dtype), indexing="ij")
    sample_x, sample_y = xs[None] + flow[:, 0], ys[None] + flow[:, 1]
    support = ((sample_x >= 0) & (sample_x <= max(w - 1, 0)) & (sample_y >= 0) & (sample_y <= max(h - 1, 0)))[:, None]
    grid = torch.stack((2.0 * sample_x / max(w - 1, 1) - 1.0, 2.0 * sample_y / max(h - 1, 1) - 1.0), dim=-1)
    warped = F.grid_sample(source, grid, mode=mode, padding_mode="zeros", align_corners=True)
    if source_valid is None:
        sampled_valid = torch.ones((b, 1, h, w), dtype=torch.bool, device=source.device)
    else:
        if source_valid.ndim == 3:
            source_valid = source_valid[:, None]
        if source_valid.shape != (b, 1, h, w):
            raise ValueError(f"source_valid must be [B,1,H,W], got {tuple(source_valid.shape)}")
        sampled_valid = F.grid_sample(source_valid.to(source.dtype), grid, mode=mode, padding_mode="zeros", align_corners=True) >= valid_threshold
    return WarpResult(warped, support, sampled_valid, support & sampled_valid)


def forward_backward_consistency(flow_target_to_source: torch.Tensor, flow_source_to_target: torch.Tensor, *, absolute_threshold: float = 0.5, relative_threshold: float = 0.01) -> ForwardBackwardResult:
    forward = _flow_bchw(flow_target_to_source)
    backward = _flow_bchw(flow_source_to_target).to(forward)
    if forward.shape != backward.shape:
        raise ValueError(f"flow shapes differ: {tuple(forward.shape)} vs {tuple(backward.shape)}")
    aligned_backward = causal_warp(backward, forward)
    cycle = forward + aligned_backward.warped
    error = torch.linalg.vector_norm(cycle, dim=1, keepdim=True)
    magnitude = torch.linalg.vector_norm(forward, dim=1, keepdim=True) + torch.linalg.vector_norm(aligned_backward.warped, dim=1, keepdim=True)
    threshold = absolute_threshold + relative_threshold * magnitude
    support = aligned_backward.support
    valid = support & (error <= threshold)
    confidence = torch.exp(-error / threshold.clamp_min(1e-6)) * support.to(error.dtype)
    return ForwardBackwardResult(error, support, valid, confidence, threshold)


def photometric_consistency(target_rgb: torch.Tensor, source_rgb: torch.Tensor, flow_target_to_source: torch.Tensor, *, target_valid: torch.Tensor | None = None, source_valid: torch.Tensor | None = None, robust_scale: float = 0.1) -> PhotometricResult:
    if target_rgb.shape != source_rgb.shape or target_rgb.ndim != 4:
        raise ValueError("target_rgb and source_rgb must share [B,C,H,W]")
    target, source = target_rgb.float(), source_rgb.float()
    if max(float(target.detach().max()), float(source.detach().max())) > 1.5:
        target, source = target / 255.0, source / 255.0
    aligned = causal_warp(source, flow_target_to_source, source_valid=source_valid)
    valid = aligned.valid
    if target_valid is not None:
        valid = valid & (target_valid[:, None] if target_valid.ndim == 3 else target_valid).bool()
    l1 = (target - aligned.warped).abs().mean(dim=1, keepdim=True)
    robust = l1 / (l1 + robust_scale)
    return PhotometricResult(aligned.warped, l1 * valid.to(l1.dtype), robust * valid.to(l1.dtype), valid)


def temporal_disparity_evidence(current_disparity: torch.Tensor, past_disparity: torch.Tensor, flow_current_to_past: torch.Tensor, flow_past_to_current: torch.Tensor, *, current_valid: torch.Tensor, past_valid: torch.Tensor, current_rgb: torch.Tensor, past_rgb: torch.Tensor, fb_absolute_threshold: float = 0.5, fb_relative_threshold: float = 0.01, photometric_robust_scale: float = 0.1) -> TemporalDisparityEvidence:
    aligned = causal_warp(past_disparity, flow_current_to_past, source_valid=past_valid)
    current_valid_b = current_valid[:, None] if current_valid.ndim == 3 else current_valid
    aligned_validity = aligned.valid & current_valid_b.bool()
    fb = forward_backward_consistency(flow_current_to_past, flow_past_to_current, absolute_threshold=fb_absolute_threshold, relative_threshold=fb_relative_threshold)
    photo = photometric_consistency(current_rgb, past_rgb, flow_current_to_past, target_valid=current_valid_b, source_valid=past_valid, robust_scale=photometric_robust_scale)
    signed = aligned.warped - current_disparity
    valid_f = aligned_validity.to(signed.dtype)
    flow_magnitude = torch.linalg.vector_norm(_flow_bchw(flow_current_to_past), dim=1, keepdim=True)
    return TemporalDisparityEvidence(aligned.warped, aligned.support, aligned_validity, fb.error, fb.confidence, photo.robust_normalized_residual, signed.abs() * valid_f, signed * valid_f, flow_magnitude)
