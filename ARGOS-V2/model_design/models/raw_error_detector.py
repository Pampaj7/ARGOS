"""Small universal raw-disparity error detectors for ARGOS v2."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


FEATURE_CHANNELS = 17
ARCHITECTURES = ("s1", "s2", "s3", "s4")
RECEPTIVE_FIELDS = {"s1": 1, "s2": 7, "s3": 19, "s4": 7}


@dataclass(frozen=True)
class RawErrorEvidence:
    raw: torch.Tensor
    raw_valid: torch.Tensor
    aligned: torch.Tensor
    aligned_valid: torch.Tensor
    warp_support: torch.Tensor
    forward_backward_error: torch.Tensor
    forward_backward_confidence: torch.Tensor
    photometric_residual: torch.Tensor
    flow_magnitude: torch.Tensor
    a2_update: torch.Tensor
    a2_error_gate: torch.Tensor
    a2_memory_gate: torch.Tensor


@dataclass
class RawErrorOutput:
    probability: torch.Tensor
    mu: torch.Tensor
    sigma: torch.Tensor
    logits: torch.Tensor
    raw_mu: torch.Tensor
    raw_sigma: torch.Tensor


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def _norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(4 if channels % 4 == 0 else 1, channels)


class PixelEncoder(nn.Module):
    def __init__(self, inputs: int, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(inputs, channels, 1, bias=False), _norm(channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 1, bias=False), nn.SiLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class LocalEncoder(nn.Module):
    def __init__(self, inputs: int, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(inputs, channels, 3, padding=1, bias=False), _norm(channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False), _norm(channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.SiLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class MiniUNetEncoder(nn.Module):
    def __init__(self, inputs: int, channels: int) -> None:
        super().__init__()
        self.local = LocalEncoder(inputs, channels)
        self.down = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False),
            _norm(channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False), _norm(channels), nn.SiLU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 3, padding=1, bias=False),
            _norm(channels), nn.SiLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        local = self.local(value)
        coarse = self.down(local)
        coarse = F.interpolate(coarse, local.shape[-2:], mode="bilinear", align_corners=True)
        return self.fuse(torch.cat((local, coarse), dim=1))


class RawErrorDetector(nn.Module):
    """Predict raw-wrong probability, expected error and uncertainty."""

    def __init__(self, architecture: str = "s2", *, channels: int = 24) -> None:
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {ARCHITECTURES}")
        self.architecture = architecture
        encoder = PixelEncoder if architecture == "s1" else LocalEncoder
        if architecture == "s3":
            encoder = MiniUNetEncoder
        self.encoder = encoder(FEATURE_CHANNELS, channels)
        self.head_error = nn.Conv2d(channels, 1, 1)
        self.head_mu = nn.Conv2d(channels, 1, 1)
        self.head_sigma = nn.Conv2d(channels, 1, 1)
        nn.init.zeros_(self.head_error.weight); nn.init.constant_(self.head_error.bias, -2.0)
        nn.init.zeros_(self.head_mu.weight); nn.init.constant_(self.head_mu.bias, _inverse_softplus(0.5))
        nn.init.zeros_(self.head_sigma.weight); nn.init.constant_(self.head_sigma.bias, _inverse_softplus(0.5))

    @staticmethod
    def _gradients(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dx = F.pad(value[..., 1:] - value[..., :-1], (0, 1, 0, 0))
        dy = F.pad(value[..., 1:, :] - value[..., :-1, :], (0, 0, 0, 1))
        return dx, dy

    @staticmethod
    def _variance(value: torch.Tensor) -> torch.Tensor:
        mean = F.avg_pool2d(value, 5, stride=1, padding=2)
        square = F.avg_pool2d(value.square(), 5, stride=1, padding=2)
        return (square - mean.square()).clamp_min(0)

    def normalized_inputs(self, evidence: RawErrorEvidence) -> torch.Tensor:
        raw = evidence.raw
        dx, dy = self._gradients(raw)
        disagreement = raw - evidence.aligned
        return torch.cat((
            raw.clamp(0, 64) / 64,
            dx.clamp(-4, 4) / 4,
            dy.clamp(-4, 4) / 4,
            self._variance(raw).clamp(0, 16) / 16,
            evidence.raw_valid.float(),
            evidence.aligned.clamp(0, 64) / 64,
            disagreement.clamp(-16, 16) / 16,
            disagreement.abs().clamp(0, 16) / 16,
            evidence.aligned_valid.float(),
            evidence.warp_support.float(),
            evidence.forward_backward_error.clamp(0, 8) / 8,
            evidence.forward_backward_confidence.clamp(0, 1),
            evidence.photometric_residual.clamp(0, 1),
            evidence.flow_magnitude.clamp(0, 32) / 32,
            evidence.a2_update.abs().clamp(0, 3) / 3,
            evidence.a2_error_gate.clamp(0, 1),
            evidence.a2_memory_gate.clamp(0, 1),
        ), dim=1)

    def forward(self, evidence: RawErrorEvidence) -> RawErrorOutput:
        features = self.encoder(self.normalized_inputs(evidence))
        logits = self.head_error(features)
        raw_mu = self.head_mu(features)
        raw_sigma = self.head_sigma(features)
        return RawErrorOutput(
            probability=torch.sigmoid(logits),
            mu=F.softplus(raw_mu),
            sigma=F.softplus(raw_sigma) + 1e-3,
            logits=logits,
            raw_mu=raw_mu,
            raw_sigma=raw_sigma,
        )


__all__ = [
    "ARCHITECTURES", "FEATURE_CHANNELS", "RECEPTIVE_FIELDS",
    "RawErrorDetector", "RawErrorEvidence", "RawErrorOutput",
]
