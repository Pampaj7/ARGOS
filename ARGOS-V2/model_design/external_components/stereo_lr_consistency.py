"""Backbone-agnostic left--right disparity consistency primitives.

The canonical ARGOS cache stores a positive *left* disparity ``d_L``: a left
pixel ``x_L`` matches the right pixel ``x_R = x_L - d_L``.  A right-reference
prediction can be obtained from a black-box positive-disparity stereo model by
running the model on ``flip(right), flip(left)`` and horizontally unflipping
its output.  Its value is denoted ``d_R`` on the original right grid and has
the same positive magnitude convention.  The ordinary LRC residual is then

``abs(d_L(x_L) - d_R(x_L - d_L(x_L)))``.

This module deliberately contains no model loader, cache writer, or learned
score.  It establishes the one geometrically valid reusable operation needed
by a subsequent frozen audit.  All callers must keep the returned support and
sampled-valid masks in their common comparison mask.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LeftRightConsistencyEvidence:
    """LRC values on the left grid.

    Every tensor is ``[B,1,H,W]``.  ``right_disparity_sampled`` is valid only
    under ``right_support & sampled_right_valid``.  ``valid`` additionally
    requires the current left prediction mask.  No invalid value is silently
    interpreted as a zero residual.
    """

    right_disparity_sampled: torch.Tensor
    right_support: torch.Tensor
    sampled_right_valid: torch.Tensor
    valid: torch.Tensor
    residual: torch.Tensor


def flip_swap_stereo_pair(left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the positive-disparity-compatible pair for right-reference inference.

    With rectified original views, ``(flip(right), flip(left))`` has positive
    horizontal disparity equal to the original magnitude.  The returned pair
    retains shape ``[B,C,H,W]`` and does not resize or normalize RGB values.
    """
    if left.shape != right.shape or left.ndim != 4:
        raise ValueError("left and right must have the identical [B,C,H,W] shape")
    return right.flip(-1), left.flip(-1)


def unflip_right_reference_disparity(disparity: torch.Tensor) -> torch.Tensor:
    """Map a disparity predicted on ``flip(right)`` back to original right x."""
    if disparity.ndim != 4 or disparity.shape[1] != 1:
        raise ValueError("disparity must be [B,1,H,W]")
    return disparity.flip(-1)


def _sample_right_on_left_grid(
    right_value: torch.Tensor,
    positive_left_disparity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a right-grid map at ``x_R=x_L-d_L`` with BiDA-compatible grid rules."""
    if right_value.ndim != 4 or positive_left_disparity.ndim != 4:
        raise ValueError("right value and left disparity must be rank-4 tensors")
    if right_value.shape[0] != positive_left_disparity.shape[0] or right_value.shape[-2:] != positive_left_disparity.shape[-2:]:
        raise ValueError("right value and left disparity must share batch/spatial shape")
    if positive_left_disparity.shape[1] != 1:
        raise ValueError("left disparity must have one channel")
    batch, _channels, height, width = right_value.shape
    dtype = positive_left_disparity.dtype
    yy, xx = torch.meshgrid(
        torch.arange(height, device=right_value.device, dtype=dtype),
        torch.arange(width, device=right_value.device, dtype=dtype),
        indexing="ij",
    )
    x_right = xx.unsqueeze(0) - positive_left_disparity[:, 0]
    y_right = yy.unsqueeze(0).expand(batch, -1, -1)
    support = ((x_right >= 0) & (x_right <= width - 1) & (y_right >= 0) & (y_right <= height - 1)).unsqueeze(1)
    # align_corners=True: pixel 0 -> -1, pixel W-1 -> +1.
    grid_x = 2.0 * x_right / max(width - 1, 1) - 1.0
    grid_y = 2.0 * y_right / max(height - 1, 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1)
    sampled = F.grid_sample(right_value, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return sampled, support


def left_right_consistency(
    left_disparity: torch.Tensor,
    right_reference_disparity: torch.Tensor,
    *,
    left_valid: torch.Tensor | None = None,
    right_valid: torch.Tensor | None = None,
) -> LeftRightConsistencyEvidence:
    """Compute occlusion-aware LRC residual on the left frame/grid.

    ``right_reference_disparity`` must already be unflipped to original right
    coordinates.  Its units must equal ``left_disparity`` units (for ARGOS
    cache-grid use, both are pixels at width 180).
    """
    sampled, support = _sample_right_on_left_grid(right_reference_disparity, left_disparity)
    if left_valid is None:
        left_valid = torch.ones_like(left_disparity, dtype=torch.bool)
    if right_valid is None:
        right_valid = torch.ones_like(right_reference_disparity, dtype=torch.bool)
    if left_valid.shape != left_disparity.shape or right_valid.shape != right_reference_disparity.shape:
        raise ValueError("validity maps must match their disparity maps")
    sampled_valid, _ = _sample_right_on_left_grid(right_valid.float(), left_disparity)
    sampled_valid = sampled_valid >= (1.0 - 1e-6)
    valid = left_valid.bool() & support & sampled_valid
    return LeftRightConsistencyEvidence(
        right_disparity_sampled=sampled,
        right_support=support,
        sampled_right_valid=sampled_valid,
        valid=valid,
        residual=(left_disparity - sampled).abs(),
    )


__all__ = [
    "LeftRightConsistencyEvidence",
    "flip_swap_stereo_pair",
    "unflip_right_reference_disparity",
    "left_right_consistency",
]
