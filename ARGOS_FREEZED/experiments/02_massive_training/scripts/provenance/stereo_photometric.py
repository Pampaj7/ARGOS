"""Backbone-agnostic stereo reprojection evidence for ARGOS v2.

This module does *not* estimate disparity.  Given a positive left-disparity
candidate on the left-image grid, it samples the current right image at
``x_right = x_left - disparity`` and returns deterministic matching costs.
It is useful as a universal quality cue for comparing a raw stereo prediction
with a causally BiDA-aligned temporal candidate.  It has no trainable state,
no future-frame access, and no dependency on the producing stereo backbone.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class StereoPhotometricEvidence:
    """Current-frame stereo matching evidence, all maps ``[B,1,H,W]``.

    ``right_support`` is true where the candidate samples inside the right
    image.  ``rgb_l1`` is local-mean RGB L1 in [0, 1], while ``zncc_cost`` is
    ``1 - ZNCC`` of luminance patches (nominally [0, 2]).  ``ternary_census``
    is a local ordinal matching cost in [0, 1], invariant to an affine local
    brightness offset and less sensitive than L1/ZNCC to endoscopic exposure
    changes.  ``census_support`` explicitly rejects a census window touching
    an out-of-bounds reconstructed-right sample.  The caller must still
    intersect the selected support with prediction and causal-warp validity.
    """

    reconstructed_right: torch.Tensor
    right_support: torch.Tensor
    rgb_l1: torch.Tensor
    local_rgb_l1: torch.Tensor
    zncc_cost: torch.Tensor
    ternary_census_cost: torch.Tensor
    census_support: torch.Tensor


def _as_unit_rgb(image: torch.Tensor) -> torch.Tensor:
    """Accept uint8-like [0,255] or already-normalized RGB tensors."""
    image = image.float()
    return image / 255.0 if image.detach().amax() > 1.5 else image


def warp_right_to_left(right_rgb: torch.Tensor, positive_left_disparity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample right RGB or a right feature map into left coordinates.

    Tensor contract: ``right_rgb`` is ``[B,3,H,W]`` and disparity is
    ``[B,1,H,W]`` in pixels at exactly that grid.  The grid uses the same
    ``align_corners=True`` / ``(W-1,H-1)`` convention as canonical BiDA, but
    stereo correspondence is horizontal and therefore samples ``x - d``.
    No component scaling occurs here because images and disparity share a
    resolution.
    """
    if right_rgb.ndim != 4 or right_rgb.shape[1] < 1:
        raise ValueError("right_rgb/features must have shape [B,C,H,W] with C >= 1")
    if positive_left_disparity.ndim != 4 or positive_left_disparity.shape[1] != 1:
        raise ValueError("positive_left_disparity must have shape [B,1,H,W]")
    if right_rgb.shape[0] != positive_left_disparity.shape[0] or right_rgb.shape[-2:] != positive_left_disparity.shape[-2:]:
        raise ValueError("right RGB and disparity must share batch/spatial dimensions")

    batch, _, height, width = right_rgb.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=right_rgb.device, dtype=positive_left_disparity.dtype),
        torch.arange(width, device=right_rgb.device, dtype=positive_left_disparity.dtype),
        indexing="ij",
    )
    source_x = x.unsqueeze(0) - positive_left_disparity[:, 0]
    source_y = y.unsqueeze(0).expand(batch, -1, -1)
    support = (
        torch.isfinite(source_x)
        & (source_x >= 0)
        & (source_x <= width - 1)
        & (source_y >= 0)
        & (source_y <= height - 1)
    ).unsqueeze(1)
    grid = torch.stack(
        (2.0 * source_x / max(width - 1, 1) - 1.0, 2.0 * source_y / max(height - 1, 1) - 1.0),
        dim=-1,
    )
    reconstructed = F.grid_sample(
        _as_unit_rgb(right_rgb), grid, mode="bilinear", padding_mode="zeros", align_corners=True,
    )
    return reconstructed, support


def stereo_photometric_evidence(
    left_rgb: torch.Tensor,
    right_rgb: torch.Tensor,
    positive_left_disparity: torch.Tensor,
    *,
    local_kernel: int = 31,
    census_kernel: int = 7,
    census_threshold: float = .02,
    eps: float = 1e-6,
) -> StereoPhotometricEvidence:
    """Compute deterministic local photometric evidence for one candidate.

    The local window is odd and is intentionally an explicit parameter: it
    may be selected on held-out validation only.  ZNCC removes local affine
    brightness changes; RGB L1 remains useful where ZNCC is uninformative.
    """
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
    # ITU-R BT.601 luminance.  The comparison is deterministic and has no
    # learned appearance representation.
    lum_left = 0.299 * left[:, 0:1] + 0.587 * left[:, 1:2] + 0.114 * left[:, 2:3]
    lum_right = 0.299 * reconstructed[:, 0:1] + 0.587 * reconstructed[:, 1:2] + 0.114 * reconstructed[:, 2:3]
    pad = local_kernel // 2
    mean_left = F.avg_pool2d(lum_left, local_kernel, stride=1, padding=pad)
    mean_right = F.avg_pool2d(lum_right, local_kernel, stride=1, padding=pad)
    variance_left = F.avg_pool2d(lum_left.square(), local_kernel, stride=1, padding=pad) - mean_left.square()
    variance_right = F.avg_pool2d(lum_right.square(), local_kernel, stride=1, padding=pad) - mean_right.square()
    covariance = F.avg_pool2d(lum_left * lum_right, local_kernel, stride=1, padding=pad) - mean_left * mean_right
    zncc_cost = 1.0 - covariance / (variance_left.clamp_min(eps).sqrt() * variance_right.clamp_min(eps).sqrt())
    ternary_cost, census_support = ternary_census_cost(
        lum_left, lum_right, support, kernel=census_kernel, threshold=census_threshold,
    )
    return StereoPhotometricEvidence(
        reconstructed_right=reconstructed,
        right_support=support,
        rgb_l1=rgb_l1,
        local_rgb_l1=local_l1,
        zncc_cost=zncc_cost.clamp(0.0, 2.0),
        ternary_census_cost=ternary_cost,
        census_support=census_support,
    )


def ternary_census_cost(
    left_luminance: torch.Tensor,
    reconstructed_right_luminance: torch.Tensor,
    right_support: torch.Tensor,
    *,
    kernel: int = 7,
    threshold: float = .02,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a robust local ternary-census cost and its exact patch support.

    This is a deterministic *candidate quality* cue, not a learned feature or
    a replacement stereo matcher.  At every pixel it compares the ordinal
    relation of each luminance neighbour to its centre in the left image and
    in the candidate-reconstructed right image.  The ternary dead zone makes
    the cost robust to small sensor noise; ranking relations make it tolerant
    to local exposure offsets.  A cost involving zero padding must never look
    reliable, hence the erosion of ``right_support`` over the full window.

    Tensor contract: luminance and support are ``[B,1,H,W]`` at the disparity
    grid.  Returned cost is ``[B,1,H,W]`` in ``[0,1]`` and support has the
    same shape.  The operation is intentionally no-grad in typical use, but
    remains differentiable almost everywhere with respect to neither binary
    code; selector gradients flow only into the downstream CNN.
    """
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
    # ``unfold`` returns fixed-grid patches, including padding.  The support
    # below explicitly removes every location whose reconstructed patch touches
    # a sample outside the right image; padded values cannot lower the cost.
    left_patch = F.unfold(left_luminance, kernel, padding=radius).view(batch, kernel * kernel, height, width)
    right_patch = F.unfold(reconstructed_right_luminance, kernel, padding=radius).view(batch, kernel * kernel, height, width)
    center_index = (kernel * kernel) // 2
    center_left = left_patch[:, center_index:center_index + 1]
    center_right = right_patch[:, center_index:center_index + 1]

    def code(patch: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        difference = patch - center
        return (difference > threshold).float() - (difference < -threshold).float()

    cost = (code(left_patch, center_left) - code(right_patch, center_right)).abs().mean(dim=1, keepdim=True) * .5
    invalid = (~right_support.bool()).float()
    patch_invalid = F.max_pool2d(invalid, kernel, stride=1, padding=radius) > 0
    support = ~patch_invalid
    return cost.clamp(0.0, 1.0), support


def select_lower_stereo_cost(
    raw_cost: torch.Tensor,
    memory_cost: torch.Tensor,
    valid: torch.Tensor,
    *,
    minimum_improvement: float = 0.0,
) -> torch.Tensor:
    """Authorize memory only when its photometric cost beats raw by a margin."""
    if minimum_improvement < 0:
        raise ValueError("minimum_improvement must be non-negative")
    return valid.bool() & torch.isfinite(raw_cost) & torch.isfinite(memory_cost) & (
        memory_cost < raw_cost - minimum_improvement
    )


__all__ = [
    "StereoPhotometricEvidence", "stereo_photometric_evidence", "ternary_census_cost", "warp_right_to_left", "select_lower_stereo_cost",
]
