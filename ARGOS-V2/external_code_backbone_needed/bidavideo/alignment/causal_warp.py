"""Clean-room ARGOS v2 causal warping utility.

Original repository reference: https://github.com/MatchLab-Imperial/bidavideo
Original paths inspected:
- models/core/bidastabilizer.py
- train_utils/losses.py
Source commit inspected: dae817df1ceaafcb865ebd9c7aa44b16c535e856

Reason for export: preserve the target-to-source grid-sample convention needed by
ARGOS v2 without importing the non-causal BiDAStabilizer module.
Copied unchanged: no. This is a minimal clean-room implementation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def warp_to_current(source: torch.Tensor, flow_t_to_source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Warp a source tensor into the current/target frame.

    Args:
        source: `[B, C, H, W]` tensor sampled at source-frame coordinates.
        flow_t_to_source: `[B, 2, H, W]` or `[B, H, W, 2]` target-to-source flow in pixels.

    Returns:
        `(warped, valid)` where `warped` is `[B, C, H, W]` and `valid` is
        `[B, 1, H, W]`. Positive `flow[..., 0]` samples from the right.
    """
    if source.ndim != 4:
        raise ValueError(f"source must be [B,C,H,W], got {tuple(source.shape)}")
    if flow_t_to_source.ndim != 4:
        raise ValueError(f"flow must be rank 4, got {tuple(flow_t_to_source.shape)}")
    if flow_t_to_source.shape[1] == 2:
        flow = flow_t_to_source.permute(0, 2, 3, 1)
    else:
        flow = flow_t_to_source
    b, _c, h, w = source.shape
    if flow.shape != (b, h, w, 2):
        raise ValueError(f"flow shape {tuple(flow.shape)} incompatible with source {tuple(source.shape)}")

    ys, xs = torch.meshgrid(
        torch.arange(h, device=source.device, dtype=source.dtype),
        torch.arange(w, device=source.device, dtype=source.dtype),
        indexing="ij",
    )
    grid = torch.stack((xs, ys), dim=-1).unsqueeze(0) + flow.to(dtype=source.dtype)
    valid = (grid[..., 0] >= 0) & (grid[..., 0] <= w - 1) & (grid[..., 1] >= 0) & (grid[..., 1] <= h - 1)
    norm_x = 2.0 * grid[..., 0] / max(w - 1, 1) - 1.0
    norm_y = 2.0 * grid[..., 1] / max(h - 1, 1) - 1.0
    norm_grid = torch.stack((norm_x, norm_y), dim=-1)
    warped = F.grid_sample(source, norm_grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return warped, valid.unsqueeze(1).to(source.dtype)


def forward_backward_consistency(
    flow_t_to_prev: torch.Tensor,
    flow_prev_to_t: torch.Tensor,
    threshold_px: float = 1.0,
) -> torch.Tensor:
    """Return a simple forward-backward support mask for target-to-source flow."""
    warped_back, valid = warp_to_current(flow_prev_to_t, flow_t_to_prev)
    err = torch.linalg.vector_norm(flow_t_to_prev + warped_back, dim=1, keepdim=True)
    return ((err <= threshold_px).to(flow_t_to_prev.dtype) * valid).to(flow_t_to_prev.dtype)
