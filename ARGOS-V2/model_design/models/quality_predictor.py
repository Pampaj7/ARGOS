"""Small shared-candidate Q0 expected-error and uncertainty predictors."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from model_design.data.quality_prediction_dataset import QualityCandidateBatch


FEATURE_CHANNELS = 19
ARCHITECTURES = ("q0_1", "q0_2", "q0_3", "q0_4", "q0_5")


@dataclass
class QualityPredictionOutput:
    """Candidate maps, each ``[B,K,H,W]``."""

    mu: torch.Tensor
    sigma: torch.Tensor
    advantage: torch.Tensor
    raw_mu: torch.Tensor
    raw_sigma: torch.Tensor


def _normalization(channels: int) -> nn.GroupNorm:
    groups = 4 if channels % 4 == 0 else 1
    return nn.GroupNorm(groups, channels)


class PixelEncoder(nn.Module):
    def __init__(self, input_channels: int, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, channels, 1, bias=False),
            _normalization(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.SiLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class LocalEncoder(nn.Module):
    def __init__(self, input_channels: int, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, channels, 3, padding=1, bias=False),
            _normalization(channels), nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            _normalization(channels), nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.SiLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class MiniUNetEncoder(nn.Module):
    """Two-resolution regional encoder; no attention or temporal state."""

    def __init__(self, input_channels: int, channels: int) -> None:
        super().__init__()
        self.local = LocalEncoder(input_channels, channels)
        self.down = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False),
            _normalization(channels), nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            _normalization(channels), nn.SiLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 3, padding=1, bias=False),
            _normalization(channels), nn.SiLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        local = self.local(value)
        coarse = self.down(local)
        coarse = F.interpolate(coarse, size=local.shape[-2:], mode="bilinear", align_corners=True)
        return self.fuse(torch.cat((local, coarse), dim=1))


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class QualityPredictor(nn.Module):
    """Shared candidate encoder with optional lightweight per-candidate heads.

    The model never receives metadata such as backbone or sequence. Candidate
    age is a normalized input map. Q0-1 is pixel-wise, Q0-2 local, Q0-3 a
    shared Mini U-Net, Q0-4 adds candidate-specific 1x1 heads, and Q0-5 makes
    those uncertainty heads trainable. ``predict_uncertainty`` can enable the
    same heteroscedastic head on a smaller encoder for a capacity-controlled
    representation pilot; it does not change candidate/error features.
    """

    def __init__(
        self,
        architecture: str = "q0_3",
        *,
        channels: int = 24,
        candidates: int = 5,
        sigma_epsilon: float = 1e-3,
        predict_uncertainty: bool | None = None,
    ) -> None:
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {ARCHITECTURES}")
        self.architecture = architecture
        self.candidates = int(candidates)
        self.sigma_epsilon = float(sigma_epsilon)
        self.candidate_specific_heads = architecture in {"q0_4", "q0_5"}
        self.predicts_uncertainty = (
            architecture == "q0_5" if predict_uncertainty is None else bool(predict_uncertainty)
        )
        encoder_type = PixelEncoder if architecture == "q0_1" else LocalEncoder
        if architecture in {"q0_3", "q0_4", "q0_5"}:
            encoder_type = MiniUNetEncoder
        self.shared_encoder = encoder_type(FEATURE_CHANNELS, channels)
        head_count = self.candidates if self.candidate_specific_heads else 1
        self.mu_heads = nn.ModuleList(nn.Conv2d(channels, 1, 1) for _ in range(head_count))
        self.advantage_heads = nn.ModuleList(nn.Conv2d(channels, 1, 1) for _ in range(head_count))
        self.sigma_heads = nn.ModuleList(nn.Conv2d(channels, 1, 1) for _ in range(head_count))
        for head in self.mu_heads:
            nn.init.zeros_(head.weight); nn.init.constant_(head.bias, _inverse_softplus(0.5))
        for head in self.advantage_heads:
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
        for head in self.sigma_heads:
            nn.init.zeros_(head.weight); nn.init.constant_(head.bias, _inverse_softplus(0.5))
            if not self.predicts_uncertainty:
                head.requires_grad_(False)

    @staticmethod
    def _gradients(disparity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dx = F.pad(disparity[..., 1:] - disparity[..., :-1], (0, 1, 0, 0))
        dy = F.pad(disparity[..., 1:, :] - disparity[..., :-1, :], (0, 0, 0, 1))
        return dx, dy

    @staticmethod
    def _local_variance(disparity: torch.Tensor) -> torch.Tensor:
        mean = F.avg_pool2d(disparity, 5, stride=1, padding=2)
        square = F.avg_pool2d(disparity.square(), 5, stride=1, padding=2)
        return (square - mean.square()).clamp_min(0)

    def normalized_inputs(self, candidates: QualityCandidateBatch) -> torch.Tensor:
        disparity = candidates.disparity
        b, k, _c, h, w = disparity.shape
        raw = disparity[:, :1].expand(-1, k, -1, -1, -1)
        signed = raw - disparity
        flat = disparity.reshape(b * k, 1, h, w)
        dx, dy = self._gradients(flat)
        variance = self._local_variance(flat)
        dx = dx.reshape(b, k, 1, h, w); dy = dy.reshape(b, k, 1, h, w)
        variance = variance.reshape(b, k, 1, h, w)
        age = (candidates.ages.to(disparity).view(1, k, 1, 1, 1) / 8.0).expand(b, k, 1, h, w)
        median = torch.nan_to_num(candidates.consensus_median).unsqueeze(1).expand(-1, k, -1, -1, -1)
        mad = torch.nan_to_num(candidates.consensus_mad).unsqueeze(1).expand_as(median)
        count = (candidates.witness_count / 4.0).unsqueeze(1).expand_as(median)
        deviation = (disparity - median).abs()
        raw_valid = candidates.candidate_valid[:, :1].expand_as(candidates.candidate_valid)
        tensors = (
            raw.clamp(0, 64) / 64,
            disparity.clamp(0, 64) / 64,
            signed.clamp(-16, 16) / 16,
            signed.abs().clamp(0, 16) / 16,
            dx.clamp(-4, 4) / 4,
            dy.clamp(-4, 4) / 4,
            variance.clamp(0, 16) / 16,
            raw_valid.float(),
            candidates.candidate_valid.float(),
            candidates.warp_support.float(),
            candidates.forward_backward_error.clamp(0, 8) / 8,
            candidates.forward_backward_confidence.clamp(0, 1),
            candidates.photometric_residual.clamp(0, 1),
            candidates.flow_magnitude.clamp(0, 32) / 32,
            age,
            median.clamp(0, 64) / 64,
            mad.clamp(0, 8) / 8,
            count,
            deviation.clamp(0, 16) / 16,
        )
        return torch.cat(tensors, dim=2)

    def _heads(self, features: torch.Tensor, heads: nn.ModuleList, b: int, k: int) -> torch.Tensor:
        if not self.candidate_specific_heads:
            return heads[0](features).reshape(b, k, *features.shape[-2:])
        shaped = features.reshape(b, k, features.shape[1], *features.shape[-2:])
        return torch.cat([heads[index](shaped[:, index]) for index in range(k)], dim=1)

    def forward(self, candidates: QualityCandidateBatch) -> QualityPredictionOutput:
        inputs = self.normalized_inputs(candidates)
        b, k, f, h, w = inputs.shape
        if k != self.candidates or f != FEATURE_CHANNELS:
            raise ValueError(f"expected [B,{self.candidates},{FEATURE_CHANNELS},H,W]")
        features = self.shared_encoder(inputs.reshape(b * k, f, h, w))
        raw_mu = self._heads(features, self.mu_heads, b, k)
        raw_sigma = self._heads(features, self.sigma_heads, b, k)
        advantage = self._heads(features, self.advantage_heads, b, k)
        mu = F.softplus(raw_mu)
        sigma = F.softplus(raw_sigma) + self.sigma_epsilon
        return QualityPredictionOutput(mu, sigma, advantage, raw_mu, raw_sigma)


__all__ = [
    "ARCHITECTURES",
    "FEATURE_CHANNELS",
    "QualityPredictionOutput",
    "QualityPredictor",
]
