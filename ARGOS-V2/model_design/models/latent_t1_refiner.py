"""Small identity-preserving ARGOS v2 latent-state refiners E2-E5."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from model_design.external_components.endostreamdepth import (
    CausalState,
    ExplicitMultiScaleState,
    state_statistics,
)


VARIANT_SCALES = {
    "E2": ("s8",),
    "E3": ("s8",),
    "E4": ("s4", "s8", "s16"),
    "E5": ("s4", "s8", "s16"),
}
SCALE_FACTORS = {"s4": 4, "s8": 8, "s16": 16}


@dataclass
class LatentRefinerOutput:
    disparity: torch.Tensor
    update: torch.Tensor
    g_error: torch.Tensor
    c_temporal: torch.Tensor
    delta: torch.Tensor
    tau: torch.Tensor
    error_logits: torch.Tensor
    memory_logits: torch.Tensor
    state: CausalState
    state_statistics: dict[str, torch.Tensor]
    forget_maps: dict[str, torch.Tensor]


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

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(value + self.block(value))


class LatentT1Refiner(nn.Module):
    """Universal causal state plus the validated bounded residual formulation.

    E2 uses only current raw disparity, validity and disparity gradients. E3-E5
    additionally use canonical BiDA t-1 evidence. No RGB/backbone identity,
    cost volume, stereo feature, matcher confidence or future frame is accepted.
    """

    def __init__(
        self,
        variant: str,
        *,
        feature_channels: int = 32,
        state_channels: int = 16,
        tau_px: float = 3.0,
    ) -> None:
        super().__init__()
        if variant not in VARIANT_SCALES:
            raise ValueError(f"variant must be one of {tuple(VARIANT_SCALES)}")
        self.variant = variant
        self.uses_bida = variant != "E2"
        self.scales = VARIANT_SCALES[variant]
        in_channels = 13 if self.uses_bida else 4
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, feature_channels),
            nn.SiLU(inplace=True),
            ConvBlock(feature_channels),
        )
        self.scale_projections = nn.ModuleDict({
            scale: nn.Sequential(
                nn.Conv2d(feature_channels, state_channels, 3, padding=1, bias=False),
                # A single group remains well-defined for streaming B=1 even
                # when the coarsest feature map is 1x1.  Per-channel groups do
                # not: PyTorch then sees one value in every normalization group.
                nn.GroupNorm(1, state_channels),
                nn.SiLU(inplace=True),
            )
            for scale in self.scales
        })
        self.state_operator = ExplicitMultiScaleState(self.scales, state_channels)
        self.temporal_fusion = nn.Conv2d(state_channels * len(self.scales), feature_channels, 1, bias=False)
        self.state_injection = nn.Parameter(torch.zeros(()))
        self.forget_heads = nn.ModuleDict()
        if variant == "E5":
            for scale in self.scales:
                head = nn.Conv2d(5, 1, 3, padding=1)
                nn.init.zeros_(head.weight)
                nn.init.constant_(head.bias, -2.0)
                self.forget_heads[scale] = head
        self.head_error = nn.Conv2d(feature_channels, 1, 1)
        self.head_temporal = nn.Conv2d(feature_channels, 1, 1)
        self.head_delta = nn.Conv2d(feature_channels, 1, 1)
        self.register_buffer("tau_px", torch.tensor(float(tau_px)), persistent=True)
        self.reset_identity_heads()

    def reset_identity_heads(self) -> None:
        nn.init.zeros_(self.head_error.weight)
        nn.init.constant_(self.head_error.bias, -4.0)
        nn.init.zeros_(self.head_temporal.weight)
        nn.init.zeros_(self.head_temporal.bias)
        nn.init.zeros_(self.head_delta.weight)
        nn.init.zeros_(self.head_delta.bias)

    @staticmethod
    def _gradients(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dx = F.pad(raw[..., 1:] - raw[..., :-1], (0, 1, 0, 0))
        dy = F.pad(raw[..., 1:, :] - raw[..., :-1, :], (0, 0, 0, 1))
        return dx.clamp(-4, 4) / 4, dy.clamp(-4, 4) / 4

    def normalized_inputs(self, raw: torch.Tensor, evidence: Mapping[str, torch.Tensor]) -> torch.Tensor:
        dx, dy = self._gradients(raw)
        if not self.uses_bida:
            return torch.cat((raw.clamp(0, 64) / 64, evidence["current_valid"].float(), dx, dy), dim=1)
        aligned = evidence["aligned_past_disparity"]
        disagreement = raw - aligned
        return torch.cat((
            raw.clamp(0, 64) / 64,
            aligned.clamp(0, 64) / 64,
            disagreement.clamp(-16, 16) / 16,
            disagreement.abs().clamp(0, 16) / 16,
            evidence["current_valid"].float(),
            evidence["aligned_validity"].float(),
            evidence["warp_support"].float(),
            evidence["forward_backward_error"].clamp(0, 8) / 8,
            evidence["forward_backward_confidence"].clamp(0, 1),
            evidence["flow_magnitude"].clamp(0, 32) / 32,
            evidence["photometric_residual"].clamp(0, 1),
            dx,
            dy,
        ), dim=1)

    @staticmethod
    def _scale_size(height: int, width: int, factor: int) -> tuple[int, int]:
        return max(1, height // factor), max(1, width // factor)

    def _state_features(self, encoded: torch.Tensor) -> dict[str, torch.Tensor]:
        height, width = encoded.shape[-2:]
        return {
            scale: self.scale_projections[scale](
                F.adaptive_avg_pool2d(encoded, self._scale_size(height, width, SCALE_FACTORS[scale]))
            )
            for scale in self.scales
        }

    def _forget_maps(
        self,
        evidence: Mapping[str, torch.Tensor],
        state_features: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if self.variant != "E5":
            return {}
        reliability = torch.cat((
            1.0 - evidence["warp_support"].float(),
            1.0 - evidence["forward_backward_confidence"].clamp(0, 1),
            evidence["photometric_residual"].clamp(0, 1),
            evidence["flow_magnitude"].clamp(0, 32) / 32,
            evidence["absolute_disparity_disagreement"].clamp(0, 16) / 16,
        ), dim=1)
        maps = {}
        for scale, feature in state_features.items():
            local = F.adaptive_avg_pool2d(reliability, feature.shape[-2:])
            rule = local.amax(dim=1, keepdim=True)
            maps[scale] = torch.sigmoid(self.forget_heads[scale](local)) * rule
        return maps

    def forward(
        self,
        raw: torch.Tensor,
        evidence: Mapping[str, torch.Tensor],
        state: CausalState | None,
        *,
        sequence_ids: Sequence[str],
        frame_indices: torch.Tensor,
        reset_mask: torch.Tensor | None = None,
    ) -> LatentRefinerOutput:
        encoded = self.encoder(self.normalized_inputs(raw, evidence))
        per_scale = self._state_features(encoded)
        forget = self._forget_maps(evidence, per_scale)
        state_outputs, new_state = self.state_operator(
            per_scale,
            state,
            sequence_ids=sequence_ids,
            frame_indices=frame_indices,
            reset_mask=reset_mask,
            forget=forget,
        )
        temporal = torch.cat([
            F.interpolate(state_outputs[scale], raw.shape[-2:], mode="bilinear", align_corners=True)
            for scale in self.scales
        ], dim=1)
        fused = encoded + torch.tanh(self.state_injection) * self.temporal_fusion(temporal)
        error_logits = self.head_error(fused)
        temporal_logits = self.head_temporal(fused)
        delta = self.head_delta(fused)
        g_error = torch.sigmoid(error_logits)
        c_temporal = torch.sigmoid(temporal_logits)
        tau = self.tau_px.to(raw)
        update = g_error * c_temporal * tau * torch.tanh(delta)
        return LatentRefinerOutput(
            disparity=raw + update,
            update=update,
            g_error=g_error,
            c_temporal=c_temporal,
            delta=delta,
            tau=tau,
            error_logits=error_logits,
            memory_logits=temporal_logits,
            state=new_state,
            state_statistics=state_statistics(new_state),
            forget_maps=forget,
        )


__all__ = ["LatentRefinerOutput", "LatentT1Refiner", "VARIANT_SCALES"]
