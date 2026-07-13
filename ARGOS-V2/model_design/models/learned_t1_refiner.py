"""Small causal t-1 selector/refiner for universal BiDA evidence.

Tensor contract: every map is ``[B,C,H,W]`` at the cache grid. Disparities are
positive-left pixels at width 180. The model never receives a backbone identity,
stereo feature, cost volume, or matcher confidence.

Backbone-independent normalization is fixed rather than fitted:

* disparity: clamp to [0, 64] / 64;
* signed current-minus-memory disagreement: clamp to [-16, 16] / 16;
* absolute disagreement: clamp to [0, 16] / 16;
* FB error: clamp to [0, 8] / 8;
* flow magnitude: clamp to [0, 32] / 32;
* masks/confidences/robust photometric residual: [0, 1];
* RGB: [0, 255] -> [-1, 1].

The output is identity preserving by construction:
``raw + g_error * c_memory * tau * tanh(delta)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn


VARIANT_GROUPS = {
    "A2": ("disparity", "validity"),
    "A3": ("disparity", "validity", "flow"),
    "A4": ("disparity", "validity", "flow", "photo"),
    "A5": ("disparity", "validity", "flow", "photo", "rgb"),
    "A6": ("disparity", "validity", "flow", "photo", "rgb"),
    "A7": ("disparity", "validity", "flow", "photo", "rgb"),
}
GROUP_CHANNELS = {"disparity": 4, "validity": 3, "flow": 3, "photo": 1, "rgb": 3}


@dataclass
class RefinerOutput:
    disparity: torch.Tensor
    update: torch.Tensor
    g_error: torch.Tensor
    c_memory: torch.Tensor
    delta: torch.Tensor
    tau: torch.Tensor
    error_logits: torch.Tensor
    memory_logits: torch.Tensor


class ConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class LearnedT1Refiner(nn.Module):
    """A roughly 40k-parameter, fully convolutional learned selector/refiner."""

    def __init__(self, variant: str = "A7", channels: int = 32, tau_px: float = 3.0) -> None:
        super().__init__()
        if variant not in VARIANT_GROUPS:
            raise ValueError(f"variant must be one of {sorted(VARIANT_GROUPS)}, got {variant!r}")
        if channels % 8:
            raise ValueError("channels must be divisible by 8 for GroupNorm")
        self.variant = variant
        self.groups = VARIANT_GROUPS[variant]
        in_channels = sum(GROUP_CHANNELS[group] for group in self.groups)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
            ConvBlock(channels),
            ConvBlock(channels),
        )
        self.head_error = nn.Conv2d(channels, 1, 1)
        self.head_memory = nn.Conv2d(channels, 1, 1)
        self.head_delta = nn.Conv2d(channels, 1, 1)
        self.register_buffer("tau_px", torch.tensor(float(tau_px)), persistent=True)
        self.reset_identity_heads()

    def reset_identity_heads(self) -> None:
        nn.init.zeros_(self.head_error.weight)
        nn.init.constant_(self.head_error.bias, -4.0)
        nn.init.zeros_(self.head_memory.weight)
        nn.init.zeros_(self.head_memory.bias)
        nn.init.zeros_(self.head_delta.weight)
        nn.init.zeros_(self.head_delta.bias)

    def normalized_inputs(
        self,
        raw: torch.Tensor,
        evidence: Mapping[str, torch.Tensor],
        current_rgb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        aligned = evidence["aligned_past_disparity"]
        current_minus_memory = raw - aligned
        tensors: list[torch.Tensor] = []
        if "disparity" in self.groups:
            tensors.extend(
                (
                    raw.clamp(0, 64) / 64,
                    aligned.clamp(0, 64) / 64,
                    current_minus_memory.clamp(-16, 16) / 16,
                    current_minus_memory.abs().clamp(0, 16) / 16,
                )
            )
        if "validity" in self.groups:
            tensors.extend(
                (
                    evidence["current_valid"].float(),
                    evidence["aligned_validity"].float(),
                    evidence["warp_support"].float(),
                )
            )
        if "flow" in self.groups:
            tensors.extend(
                (
                    evidence["forward_backward_error"].clamp(0, 8) / 8,
                    evidence["forward_backward_confidence"].clamp(0, 1),
                    evidence["flow_magnitude"].clamp(0, 32) / 32,
                )
            )
        if "photo" in self.groups:
            tensors.append(evidence["photometric_residual"].clamp(0, 1))
        if "rgb" in self.groups:
            if current_rgb is None:
                raise ValueError(f"{self.variant} requires current RGB")
            rgb = current_rgb.float()
            if float(rgb.detach().max()) <= 1.5:
                rgb = rgb * 255.0
            tensors.append(rgb / 127.5 - 1.0)
        return torch.cat(tensors, dim=1)

    def forward(
        self,
        raw: torch.Tensor,
        evidence: Mapping[str, torch.Tensor],
        current_rgb: torch.Tensor | None = None,
    ) -> RefinerOutput:
        features = self.encoder(self.normalized_inputs(raw, evidence, current_rgb))
        error_logits = self.head_error(features)
        memory_logits = self.head_memory(features)
        delta = self.head_delta(features)
        g_error = torch.sigmoid(error_logits)
        c_memory = torch.sigmoid(memory_logits)
        tau = self.tau_px.to(dtype=raw.dtype, device=raw.device)
        update = g_error * c_memory * tau * torch.tanh(delta)
        return RefinerOutput(
            disparity=raw + update,
            update=update,
            g_error=g_error,
            c_memory=c_memory,
            delta=delta,
            tau=tau,
            error_logits=error_logits,
            memory_logits=memory_logits,
        )
