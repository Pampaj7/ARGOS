from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        num_groups = min(groups, out_channels)
        while out_channels % num_groups != 0:
            num_groups -= 1
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyUNetRefiner(nn.Module):
    """Small residual U-Net for disparity refinement.

    The model predicts a bounded residual disparity. The caller adds it to the
    frozen S2M2 center disparity.
    """

    def __init__(
        self,
        in_channels: int = 8,
        base_channels: int = 24,
        residual_clamp_px: float = 4.0,
    ):
        super().__init__()
        self.residual_clamp_px = float(residual_clamp_px)
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 4)
        self.dec2 = ConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = ConvBlock(base_channels * 2 + base_channels, base_channels)
        self.head = nn.Conv2d(base_channels, 1, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        b = self.bottleneck(e3)
        d2 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.residual_clamp_px * torch.tanh(self.head(d1))


class ConvGRUCell(nn.Module):
    """Convolutional GRU cell for spatial feature maps."""

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        gate_channels = input_channels + hidden_channels
        self.hidden_channels = hidden_channels
        self.reset_gate = nn.Conv2d(gate_channels, hidden_channels, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(gate_channels, hidden_channels, kernel_size, padding=padding)
        self.candidate = nn.Conv2d(gate_channels, hidden_channels, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None) -> torch.Tensor:
        if h is None:
            h = x.new_zeros(x.shape[0], self.hidden_channels, x.shape[-2], x.shape[-1])
        combined = torch.cat([x, h], dim=1)
        reset = torch.sigmoid(self.reset_gate(combined))
        update = torch.sigmoid(self.update_gate(combined))
        candidate = torch.tanh(self.candidate(torch.cat([x, reset * h], dim=1)))
        return (1.0 - update) * h + update * candidate


class ConvGRURefiner(nn.Module):
    """Causal online temporal disparity refiner.

    The model consumes one frame at a time: RGB plus one normalized backbone
    disparity channel. The caller owns the recurrent hidden state and passes it
    from frame to frame within a clip or video stream.
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 16,
        hidden_channels: int = 64,
        residual_clamp_px: float = 2.0,
    ):
        super().__init__()
        self.residual_clamp_px = float(residual_clamp_px)
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, hidden_channels)
        self.gru = ConvGRUCell(hidden_channels, hidden_channels)
        self.dec2 = ConvBlock(hidden_channels + base_channels * 2, base_channels * 2)
        self.dec1 = ConvBlock(base_channels * 2 + base_channels, base_channels)
        self.head = nn.Conv2d(base_channels, 1, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        hidden = self.gru(e3, hidden)
        d2 = F.interpolate(hidden, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        delta = self.residual_clamp_px * torch.tanh(self.head(d1))
        return delta, hidden


class AdaptiveMotionFusionRefiner(nn.Module):
    """Causal adaptive temporal fusion model for online disparity filtering.

    The network predicts a per-pixel raw/previous fusion weight, a reset mask,
    and a bounded residual. The caller provides the previous filtered disparity
    already aligned to the current frame. When no learned flow is available, the
    same API can receive an identity-warped previous disparity plus motion proxy
    channels.
    """

    def __init__(
        self,
        in_channels: int = 8,
        base_channels: int = 48,
        hidden_channels: int = 96,
        residual_clamp_px: float = 1.5,
    ):
        super().__init__()
        self.residual_clamp_px = float(residual_clamp_px)
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, hidden_channels)
        self.gru = ConvGRUCell(hidden_channels, hidden_channels)
        self.dec2 = ConvBlock(hidden_channels + base_channels * 2, base_channels * 2)
        self.dec1 = ConvBlock(base_channels * 2 + base_channels, base_channels)
        self.head = nn.Conv2d(base_channels, 3, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        # Head channels are alpha, reset, residual. Zero logits initialize
        # alpha/reset at 0.5 and residual at 0, which matches fixed EMA before
        # learning a spatially adaptive policy.
        with torch.no_grad():
            self.head.bias[0].fill_(0.0)
            self.head.bias[1].fill_(0.0)
            self.head.bias[2].fill_(0.0)

    def forward(
        self,
        x: torch.Tensor,
        raw_disp: torch.Tensor,
        warped_prev_disp: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        hidden = self.gru(e3, hidden)
        d2 = F.interpolate(hidden, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        head = self.head(d1)
        alpha = torch.sigmoid(head[:, 0:1])
        reset = torch.sigmoid(head[:, 1:2])
        residual = self.residual_clamp_px * torch.tanh(head[:, 2:3])
        previous_weight = (1.0 - alpha) * (1.0 - reset)
        fused = alpha * raw_disp + previous_weight * warped_prev_disp + residual
        fused = torch.clamp(fused, min=0.0)
        return fused, alpha, reset, residual, hidden
