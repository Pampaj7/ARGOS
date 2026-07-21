"""Small causal utility-aware raw-versus-t-1-memory selector for ARGOS v2."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


INPUT_CHANNELS = 13
STEREO_PHOTOMETRIC_INPUT_CHANNELS = 18


@dataclass(frozen=True)
class UtilitySelectorEvidence:
    raw: torch.Tensor
    aligned_memory: torch.Tensor
    flow: torch.Tensor
    flow_magnitude: torch.Tensor
    forward_backward_confidence: torch.Tensor
    warp_support: torch.Tensor
    aligned_valid: torch.Tensor
    raw_valid: torch.Tensor
    raw_stereo_l1: torch.Tensor | None = None
    memory_stereo_l1: torch.Tensor | None = None
    raw_stereo_zncc: torch.Tensor | None = None
    memory_stereo_zncc: torch.Tensor | None = None
    stereo_common_support: torch.Tensor | None = None
    # Optional candidate-conditioned current-frame stereo correspondence maps.
    # They are input evidence only: unlike stereo_common_support they must not
    # change target, calibration, authorization or metric masks.
    stereo_matching_features: torch.Tensor | None = None


@dataclass(frozen=True)
class UtilitySelectorOutput:
    memory_better_logit: torch.Tensor
    memory_better_probability: torch.Tensor
    expected_positive_gain: torch.Tensor
    expected_harmful_magnitude: torch.Tensor

    @property
    def expected_utility(self) -> torch.Tensor:
        return self.expected_positive_gain - self.expected_harmful_magnitude

    @property
    def conditional_expected_utility(self) -> torch.Tensor:
        """Decision-theoretic utility under the predicted helpful probability.

        The magnitude heads are interpreted conditionally in the utility-risk
        objective: gain given that memory helps and loss given that it harms.
        """
        probability = self.memory_better_probability
        return probability * self.expected_positive_gain - (1.0 - probability) * self.expected_harmful_magnitude

    @property
    def conditional_harm_risk(self) -> torch.Tensor:
        return (1.0 - self.memory_better_probability) * self.expected_harmful_magnitude


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(value + self.net(value))


class UtilityMemorySelector(nn.Module):
    """CNN that chooses only between raw and causally warped t-1 disparity.

    Input maps are all ``[B,1,144,180]`` except flow ``[B,2,144,180]``.
    Values are normalized with fixed backbone-independent constants.  The model
    has no state and cannot access future frames.
    """

    def __init__(
        self,
        *,
        channels: int = 64,
        blocks: int = 4,
        include_stereo_photometric: bool = False,
        stereo_matching_feature_channels: int = 0,
    ) -> None:
        super().__init__()
        if channels % 8:
            raise ValueError("channels must be divisible by eight for GroupNorm")
        self.channels, self.blocks = int(channels), int(blocks)
        self.include_stereo_photometric = bool(include_stereo_photometric)
        self.stereo_matching_feature_channels = int(stereo_matching_feature_channels)
        if self.stereo_matching_feature_channels < 0:
            raise ValueError("stereo_matching_feature_channels must be non-negative")
        input_channels = (STEREO_PHOTOMETRIC_INPUT_CHANNELS if self.include_stereo_photometric else INPUT_CHANNELS) + self.stereo_matching_feature_channels
        layers: list[nn.Module] = [
            nn.Conv2d(input_channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels), nn.SiLU(inplace=True),
        ]
        layers.extend(ResidualBlock(channels) for _ in range(blocks))
        self.encoder = nn.Sequential(*layers)
        self.head_probability = nn.Conv2d(channels, 1, 1)
        self.head_gain = nn.Conv2d(channels, 1, 1)
        self.head_harm = nn.Conv2d(channels, 1, 1)
        # Conservative neutral initialization; the learned decision starts close
        # to abstention, but training remains fully differentiable.
        nn.init.zeros_(self.head_probability.weight); nn.init.constant_(self.head_probability.bias, -1.0)
        nn.init.zeros_(self.head_gain.weight); nn.init.constant_(self.head_gain.bias, -3.0)
        nn.init.zeros_(self.head_harm.weight); nn.init.constant_(self.head_harm.bias, -3.0)

    @staticmethod
    def _gradient_magnitude(value: torch.Tensor) -> torch.Tensor:
        dx = F.pad(value[..., 1:] - value[..., :-1], (0, 1, 0, 0))
        dy = F.pad(value[..., 1:, :] - value[..., :-1, :], (0, 0, 0, 1))
        return torch.sqrt(dx.square() + dy.square() + 1e-12)

    def normalized_inputs(self, evidence: UtilitySelectorEvidence) -> torch.Tensor:
        residual = evidence.aligned_memory - evidence.raw
        inputs = [
            evidence.raw.clamp(0, 64) / 64,
            evidence.aligned_memory.clamp(0, 64) / 64,
            residual.clamp(-16, 16) / 16,
            residual.abs().clamp(0, 16) / 16,
            evidence.flow[:, 0:1].clamp(-32, 32) / 32,
            evidence.flow[:, 1:2].clamp(-32, 32) / 32,
            evidence.flow_magnitude.clamp(0, 32) / 32,
            evidence.forward_backward_confidence.clamp(0, 1),
            evidence.warp_support.float(),
            evidence.aligned_valid.float(),
            evidence.raw_valid.float(),
            self._gradient_magnitude(evidence.raw).clamp(0, 4) / 4,
            self._gradient_magnitude(evidence.aligned_memory).clamp(0, 4) / 4,
        ]
        if self.include_stereo_photometric:
            required = (evidence.raw_stereo_l1, evidence.memory_stereo_l1,
                        evidence.raw_stereo_zncc, evidence.memory_stereo_zncc)
            if any(value is None for value in required):
                raise ValueError("stereo-photometric selector requires raw/memory L1 and ZNCC evidence")
            raw_l1, memory_l1, raw_zncc, memory_zncc = required
            inputs.extend((
                raw_l1.clamp(0, 1), memory_l1.clamp(0, 1),
                (memory_l1 - raw_l1).clamp(-1, 1),
                raw_zncc.clamp(0, 2) / 2, memory_zncc.clamp(0, 2) / 2,
            ))
        if self.stereo_matching_feature_channels:
            features = evidence.stereo_matching_features
            if features is None:
                raise ValueError("selector requires candidate-conditioned stereo matching evidence")
            expected = (evidence.raw.shape[0], self.stereo_matching_feature_channels, *evidence.raw.shape[-2:])
            if tuple(features.shape) != expected:
                raise ValueError(f"stereo matching evidence must have shape {expected}, got {tuple(features.shape)}")
            if not torch.isfinite(features).all():
                raise ValueError("stereo matching evidence contains non-finite values")
            inputs.append(features.clamp(-1, 1))
        return torch.cat(inputs, dim=1)

    def forward(self, evidence: UtilitySelectorEvidence) -> UtilitySelectorOutput:
        features = self.encoder(self.normalized_inputs(evidence))
        logits = self.head_probability(features)
        return UtilitySelectorOutput(
            memory_better_logit=logits,
            memory_better_probability=torch.sigmoid(logits),
            expected_positive_gain=F.softplus(self.head_gain(features)),
            expected_harmful_magnitude=F.softplus(self.head_harm(features)),
        )


def memory_authorization(
    output: UtilitySelectorOutput,
    evidence: UtilitySelectorEvidence,
    *,
    probability_threshold: float,
    utility_threshold_px: float,
    harm_threshold_px: float,
) -> torch.Tensor:
    """Safe hard selection: rejected pixels remain raw bit-exactly."""
    return (
        (output.memory_better_probability >= probability_threshold)
        & (output.expected_utility >= utility_threshold_px)
        & (output.expected_harmful_magnitude <= harm_threshold_px)
        & evidence.warp_support.bool() & evidence.aligned_valid.bool() & evidence.raw_valid.bool()
        & (evidence.stereo_common_support.bool() if evidence.stereo_common_support is not None else torch.ones_like(evidence.raw_valid, dtype=torch.bool))
    )


def utility_risk_authorization(
    output: UtilitySelectorOutput,
    evidence: UtilitySelectorEvidence,
    *,
    utility_threshold_px: float,
) -> torch.Tensor:
    """Single-score policy aligned with the utility-risk training objective."""
    return (
        (output.conditional_expected_utility >= utility_threshold_px)
        & evidence.warp_support.bool() & evidence.aligned_valid.bool() & evidence.raw_valid.bool()
        & (evidence.stereo_common_support.bool() if evidence.stereo_common_support is not None else torch.ones_like(evidence.raw_valid, dtype=torch.bool))
    )


def select_raw_or_memory(raw: torch.Tensor, memory: torch.Tensor, authorized: torch.Tensor) -> torch.Tensor:
    return torch.where(authorized.bool(), memory, raw)


__all__ = [
    "INPUT_CHANNELS", "STEREO_PHOTOMETRIC_INPUT_CHANNELS", "UtilityMemorySelector", "UtilitySelectorEvidence", "UtilitySelectorOutput",
    "memory_authorization", "utility_risk_authorization", "select_raw_or_memory",
]
