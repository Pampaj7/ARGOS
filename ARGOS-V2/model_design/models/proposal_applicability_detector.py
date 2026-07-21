"""Small proposal-conditioned applicability detectors for ARGOS v2."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


FEATURE_CHANNELS = 23
VARIANTS = ("P1", "P2", "P3", "P4")
# The learned encoder RF is 1 (P1) or 7 (P2-P4). Precomputed forward
# disparity-gradient channels add one neighbouring sample to the end-to-end RF.
RECEPTIVE_FIELDS = {"P1": 2, "P2": 8, "P3": 8, "P4": 8}


@dataclass(frozen=True)
class ProposalEvidence:
    raw: torch.Tensor
    aligned: torch.Tensor
    proposal: torch.Tensor
    update: torch.Tensor
    a2_error_gate: torch.Tensor
    a2_memory_gate: torch.Tensor
    a2_delta: torch.Tensor
    raw_valid: torch.Tensor
    aligned_valid: torch.Tensor
    warp_support: torch.Tensor
    flow_magnitude: torch.Tensor
    photometric_residual: torch.Tensor
    forward_backward_error: torch.Tensor
    forward_backward_confidence: torch.Tensor


@dataclass
class ProposalApplicabilityOutput:
    utility: torch.Tensor
    sigma: torch.Tensor
    class_logits: torch.Tensor | None
    raw_utility: torch.Tensor
    raw_sigma: torch.Tensor | None

    @property
    def class_probability(self) -> torch.Tensor | None:
        return None if self.class_logits is None else torch.softmax(self.class_logits, dim=1)


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class PixelEncoder(nn.Module):
    def __init__(self, inputs: int, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(inputs, channels, 1), nn.SiLU(),
            nn.Conv2d(channels, channels, 1), nn.SiLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class LocalEncoder(nn.Module):
    def __init__(self, inputs: int, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(inputs, channels, 3, padding=1), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class ProposalApplicabilityDetector(nn.Module):
    """Predict frozen-A2 utility, uncertainty and optional three-way class."""

    def __init__(self, variant: str = "P4", *, channels: int = 24) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if variant != "P1" and channels % 4:
            raise ValueError("local variants require channels divisible by four")
        self.variant = variant
        self.predicts_uncertainty = variant in {"P3", "P4"}
        self.predicts_classes = variant == "P4"
        self.encoder = (PixelEncoder if variant == "P1" else LocalEncoder)(FEATURE_CHANNELS, channels)
        self.head_utility = nn.Conv2d(channels, 1, 1)
        self.head_sigma = nn.Conv2d(channels, 1, 1) if self.predicts_uncertainty else None
        self.head_classes = nn.Conv2d(channels, 3, 1) if self.predicts_classes else None
        nn.init.zeros_(self.head_utility.weight)
        nn.init.zeros_(self.head_utility.bias)
        if self.head_sigma is not None:
            nn.init.zeros_(self.head_sigma.weight)
            nn.init.constant_(self.head_sigma.bias, _inverse_softplus(0.25))
        if self.head_classes is not None:
            nn.init.zeros_(self.head_classes.weight)
            nn.init.zeros_(self.head_classes.bias)

    @staticmethod
    def _gradients(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            F.pad(value[..., 1:] - value[..., :-1], (0, 1, 0, 0)),
            F.pad(value[..., 1:, :] - value[..., :-1, :], (0, 0, 0, 1)),
        )

    def normalized_inputs(self, evidence: ProposalEvidence) -> torch.Tensor:
        raw_dx, raw_dy = self._gradients(evidence.raw)
        proposal_dx, proposal_dy = self._gradients(evidence.proposal)
        update_dx, update_dy = self._gradients(evidence.update)
        disagreement = evidence.raw - evidence.aligned
        return torch.cat((
            evidence.raw.clamp(0, 64) / 64,
            evidence.aligned.clamp(0, 64) / 64,
            evidence.proposal.clamp(0, 64) / 64,
            evidence.update.clamp(-3, 3) / 3,
            evidence.update.abs().clamp(0, 3) / 3,
            disagreement.clamp(-16, 16) / 16,
            disagreement.abs().clamp(0, 16) / 16,
            evidence.a2_error_gate.clamp(0, 1),
            evidence.a2_memory_gate.clamp(0, 1),
            evidence.a2_delta.clamp(-4, 4) / 4,
            evidence.raw_valid.float(),
            evidence.aligned_valid.float(),
            evidence.warp_support.float(),
            raw_dx.clamp(-4, 4) / 4,
            raw_dy.clamp(-4, 4) / 4,
            proposal_dx.clamp(-4, 4) / 4,
            proposal_dy.clamp(-4, 4) / 4,
            update_dx.clamp(-3, 3) / 3,
            update_dy.clamp(-3, 3) / 3,
            evidence.flow_magnitude.clamp(0, 32) / 32,
            evidence.photometric_residual.clamp(0, 1),
            evidence.forward_backward_error.clamp(0, 8) / 8,
            evidence.forward_backward_confidence.clamp(0, 1),
        ), dim=1)

    def forward(self, evidence: ProposalEvidence) -> ProposalApplicabilityOutput:
        features = self.encoder(self.normalized_inputs(evidence))
        raw_utility = self.head_utility(features)
        utility = 3.0 * torch.tanh(raw_utility)
        raw_sigma = None if self.head_sigma is None else self.head_sigma(features)
        sigma = torch.ones_like(utility) if raw_sigma is None else F.softplus(raw_sigma) + 1e-3
        class_logits = None if self.head_classes is None else self.head_classes(features)
        return ProposalApplicabilityOutput(utility, sigma, class_logits, raw_utility, raw_sigma)


def proposal_authorization_mask(
    output: ProposalApplicabilityOutput,
    evidence: ProposalEvidence,
    *,
    utility_margin_px: float,
    uncertainty_threshold_px: float = math.inf,
    require_helpful_class: bool = False,
) -> torch.Tensor:
    bounded = torch.isfinite(evidence.update) & (evidence.update.abs() <= 3.0 + 1e-6)
    authorized = (
        (output.utility > utility_margin_px)
        & (output.sigma < uncertainty_threshold_px)
        & evidence.aligned_valid.bool()
        & evidence.warp_support.bool()
        & bounded
    )
    if require_helpful_class:
        if output.class_logits is None:
            raise ValueError("helpful-class authorization requires P4 class logits")
        authorized &= output.class_logits.argmax(dim=1, keepdim=True) == 2
    return authorized


def apply_frozen_proposal(
    raw: torch.Tensor,
    proposal: torch.Tensor,
    authorization: torch.Tensor,
) -> torch.Tensor:
    """Return raw bit-exactly when rejected and frozen A2 exactly when accepted."""
    return torch.where(authorization.bool(), proposal, raw)


__all__ = [
    "FEATURE_CHANNELS", "RECEPTIVE_FIELDS", "VARIANTS",
    "ProposalApplicabilityDetector", "ProposalApplicabilityOutput", "ProposalEvidence",
    "apply_frozen_proposal", "proposal_authorization_mask",
]
