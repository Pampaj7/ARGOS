"""Modern fast residual refiner v4 prototype.

Lightweight encoder-decoder over the existing v3 input format (16 channels:
4-frame disparity context + valid masks + temporal/spatial features). Inverted-residual
depthwise-separable blocks, 3 scales with FPN-style skips, and three 1x1 heads:

  1. bad-pixel/confidence head (logit)
  2. residual head (bounded by tanh * residual_scale)
  3. damping head — per-pixel correction aggressiveness in [0, 1]

Final prediction (computed by the caller, same convention as v3):
  refined = raw + gate(p_bad) * damping * residual_scale * tanh(residual)

forward() returns (bad_logit, p_bad, damped_residual, damping); the first three
positions match the v3 AbstentionCropRefiner interface so existing eval code can use
`bad_logit, p_bad, residual = model(x, scale)[:3]`.

Zero-initialized heads make the model an exact identity at initialization
(residual = 0), matching v3 behavior. No RGB input required.
"""

from __future__ import annotations

import torch
from torch import nn


def _norm(ch: int) -> nn.Module:
    return nn.GroupNorm(min(8, ch), ch)


class InvertedResidual(nn.Module):
    """MobileNetV2-style inverted residual with depthwise 3x3, GN + SiLU."""

    def __init__(self, ch: int, expansion: float = 2.0):
        super().__init__()
        hidden = int(ch * expansion)
        self.block = nn.Sequential(
            nn.Conv2d(ch, hidden, 1, bias=False),
            _norm(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            _norm(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, ch, 1, bias=False),
            _norm(ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


def _stage(ch: int, depth: int, expansion: float) -> nn.Sequential:
    return nn.Sequential(*[InvertedResidual(ch, expansion) for _ in range(depth)])


class ModernRefinerV4(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        base: int = 48,
        depths: tuple[int, int, int] = (2, 2, 2),
        expansion: float = 2.0,
        residual_scale: float = 3.0,
    ):
        super().__init__()
        self.residual_scale = residual_scale
        c1, c2, c3 = base, base * 2, base * 4
        self.stem = nn.Sequential(nn.Conv2d(in_channels, c1, 3, padding=1, bias=False), _norm(c1), nn.SiLU(inplace=True))
        self.stage1 = _stage(c1, depths[0], expansion)
        self.down1 = nn.Sequential(nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False), _norm(c2), nn.SiLU(inplace=True))
        self.stage2 = _stage(c2, depths[1], expansion)
        self.down2 = nn.Sequential(nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False), _norm(c3), nn.SiLU(inplace=True))
        self.stage3 = _stage(c3, depths[2], expansion)
        self.up2 = nn.Sequential(nn.Conv2d(c3 + c2, c2, 3, padding=1, bias=False), _norm(c2), nn.SiLU(inplace=True))
        self.up1 = nn.Sequential(nn.Conv2d(c2 + c1, c1, 3, padding=1, bias=False), _norm(c1), nn.SiLU(inplace=True))
        self.refine = InvertedResidual(c1, expansion)
        self.bad_head = nn.Conv2d(c1, 1, 1)
        self.residual_head = nn.Conv2d(c1, 1, 1)
        self.damping_head = nn.Conv2d(c1, 1, 1)
        for head in (self.bad_head, self.residual_head, self.damping_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, x: torch.Tensor, residual_scale: float | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = self.residual_scale if residual_scale is None else residual_scale
        f1 = self.stage1(self.stem(x))
        f2 = self.stage2(self.down1(f1))
        f3 = self.stage3(self.down2(f2))
        u2 = self.up2(torch.cat([nn.functional.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False), f2], dim=1))
        u1 = self.up1(torch.cat([nn.functional.interpolate(u2, size=f1.shape[-2:], mode="bilinear", align_corners=False), f1], dim=1))
        feat = self.refine(u1)
        bad_logit = self.bad_head(feat)
        p_bad = torch.sigmoid(bad_logit)
        damping = torch.sigmoid(self.damping_head(feat))
        residual = damping * scale * torch.tanh(self.residual_head(feat))
        return bad_logit, p_bad, residual, damping


def v4_tiny(in_channels: int = 16, residual_scale: float = 3.0) -> ModernRefinerV4:
    return ModernRefinerV4(in_channels, base=48, depths=(2, 2, 2), expansion=2.0, residual_scale=residual_scale)


def v4_small(in_channels: int = 16, residual_scale: float = 3.0) -> ModernRefinerV4:
    return ModernRefinerV4(in_channels, base=72, depths=(2, 3, 3), expansion=2.0, residual_scale=residual_scale)
