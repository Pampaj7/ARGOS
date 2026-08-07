"""Backbone-agnostic stereo reprojection evidence for ARGOS v2."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class StereoPhotometricEvidence:
    reconstructed_right: torch.Tensor
    right_support: torch.Tensor
    rgb_l1: torch.Tensor
    local_rgb_l1: torch.Tensor
    zncc_cost: torch.Tensor
    ternary_census_cost: torch.Tensor
    census_support: torch.Tensor


def _as_unit_rgb(image: torch.Tensor) -> torch.Tensor:
    image = image.float()
    return image / 255.0 if image.detach().amax() > 1.5 else image


def warp_right_to_left(right_rgb: torch.Tensor, positive_left_disparity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if right_rgb.ndim != 4 or right_rgb.shape[1] < 1:
        raise ValueError("right_rgb/features must have shape [B,C,H,W] with C >= 1")
    if positive_left_disparity.ndim != 4 or positive_left_disparity.shape[1] != 1:
        raise ValueError("positive_left_disparity must have shape [B,1,H,W]")
    if right_rgb.shape[0] != positive_left_disparity.shape[0] or right_rgb.shape[-2:] != positive_left_disparity.shape[-2:]:
        raise ValueError("right RGB and disparity must share batch/spatial dimensions")
    batch, _, height, width = right_rgb.shape
    y, x = torch.meshgrid(torch.arange(height, device=right_rgb.device, dtype=positive_left_disparity.dtype), torch.arange(width, device=right_rgb.device, dtype=positive_left_disparity.dtype), indexing="ij")
    source_x, source_y = x.unsqueeze(0) - positive_left_disparity[:, 0], y.unsqueeze(0).expand(batch, -1, -1)
    support = (torch.isfinite(source_x) & (source_x >= 0) & (source_x <= width - 1) & (source_y >= 0) & (source_y <= height - 1)).unsqueeze(1)
    grid = torch.stack((2.0 * source_x / max(width - 1, 1) - 1.0, 2.0 * source_y / max(height - 1, 1) - 1.0), dim=-1)
    reconstructed = F.grid_sample(_as_unit_rgb(right_rgb), grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return reconstructed, support


def stereo_photometric_evidence(left_rgb: torch.Tensor, right_rgb: torch.Tensor, positive_left_disparity: torch.Tensor, *, local_kernel: int = 31, census_kernel: int = 7, census_threshold: float = .02, eps: float = 1e-6) -> StereoPhotometricEvidence:
    if local_kernel < 1 or local_kernel % 2 == 0:
        raise ValueError("local_kernel must be a positive odd integer")
    if census_kernel < 3 or census_kernel % 2 == 0:
        raise ValueError("census_kernel must be an odd integer >=3")
    if census_threshold < 0:
        raise ValueError("census_threshold must be non-negative")
    left = _as_unit_rgb(left_rgb)
    reconstructed, support = warp_right_to_left(right_rgb, positive_left_disparity)
    rgb_l1 = (left - reconstructed).abs().mean(dim=1, keepdim=True)
    local_l1 = F.avg_pool2d(rgb_l1, local_kernel, stride=1, padding=local_kernel // 2)
    lum_left = 0.299 * left[:, 0:1] + 0.587 * left[:, 1:2] + 0.114 * left[:, 2:3]
    lum_right = 0.299 * reconstructed[:, 0:1] + 0.587 * reconstructed[:, 1:2] + 0.114 * reconstructed[:, 2:3]
    pad = local_kernel // 2
    mean_left, mean_right = F.avg_pool2d(lum_left, local_kernel, stride=1, padding=pad), F.avg_pool2d(lum_right, local_kernel, stride=1, padding=pad)
    variance_left = F.avg_pool2d(lum_left.square(), local_kernel, stride=1, padding=pad) - mean_left.square()
    variance_right = F.avg_pool2d(lum_right.square(), local_kernel, stride=1, padding=pad) - mean_right.square()
    covariance = F.avg_pool2d(lum_left * lum_right, local_kernel, stride=1, padding=pad) - mean_left * mean_right
    ternary_cost, census_support = ternary_census_cost(lum_left, lum_right, support, kernel=census_kernel, threshold=census_threshold)
    return StereoPhotometricEvidence(reconstructed, support, rgb_l1, local_l1, (1.0 - covariance / (variance_left.clamp_min(eps).sqrt() * variance_right.clamp_min(eps).sqrt())).clamp(0.0, 2.0), ternary_cost, census_support)


def ternary_census_cost(left_luminance: torch.Tensor, reconstructed_right_luminance: torch.Tensor, right_support: torch.Tensor, *, kernel: int = 7, threshold: float = .02) -> tuple[torch.Tensor, torch.Tensor]:
    if left_luminance.ndim != 4 or left_luminance.shape[1] != 1:
        raise ValueError("left_luminance must be [B,1,H,W]")
    if reconstructed_right_luminance.shape != left_luminance.shape:
        raise ValueError("reconstructed right luminance must match left")
    if right_support.shape != left_luminance.shape:
        raise ValueError("right_support must match luminance")
    if kernel < 3 or kernel % 2 == 0:
        raise ValueError("kernel must be odd and >=3")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    batch, _, height, width = left_luminance.shape
    radius = kernel // 2
    left_patch = F.unfold(left_luminance, kernel, padding=radius).view(batch, kernel * kernel, height, width)
    right_patch = F.unfold(reconstructed_right_luminance, kernel, padding=radius).view(batch, kernel * kernel, height, width)
    center_index = (kernel * kernel) // 2
    center_left, center_right = left_patch[:, center_index:center_index + 1], right_patch[:, center_index:center_index + 1]

    def code(patch: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        difference = patch - center
        return (difference > threshold).float() - (difference < -threshold).float()

    cost = (code(left_patch, center_left) - code(right_patch, center_right)).abs().mean(dim=1, keepdim=True) * .5
    patch_invalid = F.max_pool2d((~right_support.bool()).float(), kernel, stride=1, padding=radius) > 0
    return cost.clamp(0.0, 1.0), ~patch_invalid


def select_lower_stereo_cost(raw_cost: torch.Tensor, memory_cost: torch.Tensor, valid: torch.Tensor, *, minimum_improvement: float = 0.0) -> torch.Tensor:
    if minimum_improvement < 0:
        raise ValueError("minimum_improvement must be non-negative")
    return valid.bool() & torch.isfinite(raw_cost) & torch.isfinite(memory_cost) & (memory_cost < raw_cost - minimum_improvement)
