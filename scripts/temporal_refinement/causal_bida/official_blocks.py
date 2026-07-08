"""Small BiDAStabilizer blocks adapted for ARGOS v2.

Upstream: https://github.com/TomTomTommi/bidavideo.git
Commit: dae817df1ceaafcb865ebd9c7aa44b16c535e856
Source file: models/core/bidastabilizer.py
License: MIT

Modifications:
- removed embedded SEA-RAFT dependency; ARGOS supplies target-to-source flow;
- kept official channel counts, kernel sizes, activations, and residual block structure;
- optional zero init for the final residual head for ARGOS identity-safe smoke tests.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def make_layer(block: type[nn.Module], num_blocks: int, **kwargs) -> nn.Sequential:
    return nn.Sequential(*[block(**kwargs) for _ in range(num_blocks)])


class ResidualBlockNoBN(nn.Module):
    """Official two-conv residual block, no normalisation, ReLU."""

    def __init__(self, mid_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.relu(self.conv1(x)))


class ResidualBlocksWithInputConv(nn.Module):
    """Official input conv + LeakyReLU + N residual blocks."""

    def __init__(self, in_channels: int, out_channels: int = 64, num_blocks: int = 30):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_blocks, mid_channels=out_channels),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.main(feat)


def flow_warp(x: torch.Tensor, flow: torch.Tensor, padding_mode: str = "zeros") -> torch.Tensor:
    """Official warp convention: output(p) = x(p + flow(p))."""

    if flow.size(3) != 2:
        flow = flow.permute(0, 2, 3, 1)
    if x.shape[-2:] != flow.shape[1:3]:
        raise ValueError(f"input {x.shape[-2:]} and flow {flow.shape[1:3]} sizes differ")
    _, _, h, w = x.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=x.device, dtype=x.dtype),
        torch.arange(w, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    grid = torch.stack((xx, yy), -1).unsqueeze(0) + flow
    grid_x = 2.0 * grid[..., 0] / max(w - 1, 1) - 1.0
    grid_y = 2.0 * grid[..., 1] / max(h - 1, 1) - 1.0
    return F.grid_sample(
        x,
        torch.stack((grid_x, grid_y), -1),
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )
