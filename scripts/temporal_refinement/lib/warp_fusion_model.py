"""
lib/warp_fusion_model.py – CausalWarpedFusionRefiner

Motion-compensated causal online disparity refiner with learned RAFT optical
flow. The refiner predicts per-pixel alpha (blend weight), reset (scene-change
mask), and a bounded residual.  It operates one frame at a time and carries a
ConvGRU hidden state at 1/8 resolution.

Architecture:
    RGB_(t-1), RGB_t  →  FrozenRAFT  →  flow_fwd, flow_bwd
    flow_fwd, flow_bwd  →  flow_confidence, occlusion_mask
    warp(fused_{t-1}, flow_fwd)  →  warped_prev

    input features (8 channels at full resolution):
        RGB_t           3 ch
        raw_t / d_norm  1 ch
        warped_prev / d_norm  1 ch
        |raw_t - warped_prev| / d_norm  1 ch
        flow_magnitude  1 ch  (normalised)
        flow_confidence 1 ch
        occlusion_mask  1 ch  (← replaces binary flow_valid proxy)

    encoder: 3 × ConvBlock with stride-2 average pooling
        enc1: (8→base)         full res
        enc2: (base→base*2)    1/2 res
        enc3: (base*2→hidden)  1/4 res
        pool  → 1/8 res input to GRU

    ConvGRU hidden state: hidden channels @ 1/8 resolution

    decoder:
        upsample 1/8 → 1/4, concat enc3, ConvBlock
        upsample 1/4 → 1/2, concat enc2, ConvBlock
        upsample 1/2 → 1/1, concat enc1, ConvBlock
        head: 1×1 conv → 3 channels (alpha logit, reset logit, residual)
        alpha + reset upsampled from 1/8 via bilinear (memory saving)

    Fusion:
        fused_t = alpha_t * raw_t
                + (1 − alpha_t) * (1 − reset_t) * warped_prev_t
                + residual_t
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """2 × (Conv3x3 + GroupNorm + SiLU) residual-free block."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        groups = 1
        for g in [8, 4, 2, 1]:
            if out_ch % g == 0:
                groups = g
                break
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvGRUCell(nn.Module):
    """Convolutional GRU cell operating on 2-D feature maps."""

    def __init__(self, input_ch: int, hidden_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        gate_ch = input_ch + hidden_ch
        self.hidden_ch = hidden_ch
        self.reset_gate  = nn.Conv2d(gate_ch, hidden_ch, kernel_size, padding=pad)
        self.update_gate = nn.Conv2d(gate_ch, hidden_ch, kernel_size, padding=pad)
        self.candidate   = nn.Conv2d(gate_ch, hidden_ch, kernel_size, padding=pad)

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if h is None:
            h = x.new_zeros(x.shape[0], self.hidden_ch, x.shape[-2], x.shape[-1])
        combined = torch.cat([x, h], dim=1)
        r = torch.sigmoid(self.reset_gate(combined))
        u = torch.sigmoid(self.update_gate(combined))
        c = torch.tanh(self.candidate(torch.cat([x, r * h], dim=1)))
        return (1.0 - u) * h + u * c


# ──────────────────────────────────────────────────────────────────────────────
# Main model
# ──────────────────────────────────────────────────────────────────────────────

class CausalWarpedFusionRefiner(nn.Module):
    """Causal motion-warped fusion refiner.

    Processes a single frame per forward call. The caller manages the hidden
    state and the previous fused disparity across frames. The optical-flow
    model (FrozenRAFT) is **not** a submodule here; it lives in the training
    loop so that its parameters are never passed to the optimiser.

    Args:
        in_channels:       Number of input channels (default 8, see module doc).
        base_channels:     Width of the encoder / decoder (default 32).
        hidden_channels:   ConvGRU hidden size (default 64).
        residual_clamp_px: Hard clamp for the residual head output.
    """

    IN_CHANNELS = 9  # documented above

    def __init__(
        self,
        in_channels: int = 9,
        base_channels: int = 32,
        hidden_channels: int = 64,
        residual_clamp_px: float = 1.5,
    ) -> None:
        super().__init__()
        self.residual_clamp_px = float(residual_clamp_px)
        b = base_channels
        h = hidden_channels

        # Encoder (full → 1/2 → 1/4)
        self.enc1 = ConvBlock(in_channels, b)        # full res
        self.enc2 = ConvBlock(b,           b * 2)    # 1/2
        self.enc3 = ConvBlock(b * 2,       b * 4)    # 1/4

        # GRU lives at 1/8
        self.gru_proj = nn.Conv2d(b * 4, h, 1)       # 1/4 → h channels for GRU input
        self.gru = ConvGRUCell(h, h)                  # hidden at 1/8

        # Decoder (1/8 → 1/4 → 1/2 → full)
        self.dec3 = ConvBlock(h + b * 4,  b * 2)     # 1/4
        self.dec2 = ConvBlock(b * 2 + b * 2, b)      # 1/2
        self.dec1 = ConvBlock(b + b,       b)         # full

        # Prediction head: alpha logit, reset logit, residual
        self.head = nn.Conv2d(b, 3, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        x: torch.Tensor,
        raw_disp: torch.Tensor,
        warped_prev_disp: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-frame forward pass.

        Args:
            x:                 (B, 8, H, W) stacked input features.
            raw_disp:          (B, 1, H, W) raw disparity from S2M2-S.
            warped_prev_disp:  (B, 1, H, W) motion-warped fused disparity.
            hidden:            ConvGRU hidden state (B, hidden_ch, H/8, W/8) or None.

        Returns:
            fused:      (B, 1, H, W) filtered disparity.
            alpha:      (B, 1, H, W) blend weight ∈ (0, 1).
            reset:      (B, 1, H, W) scene-change probability ∈ (0, 1).
            residual:   (B, 1, H, W) bounded additive correction.
            new_hidden: (B, hidden_ch, H/8, W/8) updated GRU state.
        """
        # ── Encoder ──────────────────────────────────────────────────────────
        e1 = self.enc1(x)                                    # (B, b, H, W)
        e2 = self.enc2(F.avg_pool2d(e1, 2))                  # (B, b*2, H/2, W/2)
        e3 = self.enc3(F.avg_pool2d(e2, 2))                  # (B, b*4, H/4, W/4)

        # ── ConvGRU at 1/8 ───────────────────────────────────────────────────
        gru_in = F.avg_pool2d(self.gru_proj(e3), 2)          # (B, h, H/8, W/8)
        new_hidden = self.gru(gru_in, hidden)                 # (B, h, H/8, W/8)

        # ── Decoder ──────────────────────────────────────────────────────────
        d3 = F.interpolate(new_hidden, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))           # (B, b*2, H/4, W/4)

        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))           # (B, b, H/2, W/2)

        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))           # (B, b, H, W)

        # ── Heads ─────────────────────────────────────────────────────────────
        head = self.head(d1)                                  # (B, 3, H, W)
        alpha    = torch.sigmoid(head[:, 0:1])
        reset    = torch.sigmoid(head[:, 1:2])
        residual = self.residual_clamp_px * torch.tanh(head[:, 2:3])

        # ── Fusion ───────────────────────────────────────────────────────────
        prev_weight = (1.0 - alpha) * (1.0 - reset)
        fused = alpha * raw_disp + prev_weight * warped_prev_disp + residual
        fused = torch.clamp(fused, min=0.0)

        return fused, alpha, reset, residual, new_hidden
