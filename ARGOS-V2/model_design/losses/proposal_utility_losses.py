"""Controlled losses for frozen-proposal utility prediction."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from model_design.data.proposal_utility_dataset import ProposalUtilityTargets
from model_design.models.proposal_applicability_detector import ProposalApplicabilityOutput


@dataclass(frozen=True)
class ProposalUtilityLossConfig:
    utility_weight: float = 1.0
    heteroscedastic_weight: float = 0.0
    classification_weight: float = 0.0
    harmful_as_helpful_weight: float = 0.0
    huber_delta_px: float = 0.10


def _masked_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    selected = value[valid.bool()]
    return selected.mean() if selected.numel() else value.sum() * 0.0


def proposal_utility_losses(
    output: ProposalApplicabilityOutput,
    target: ProposalUtilityTargets,
    config: ProposalUtilityLossConfig,
) -> dict[str, torch.Tensor]:
    valid = target.regression_valid
    utility = _masked_mean(
        F.huber_loss(output.utility, target.utility, reduction="none", delta=config.huber_delta_px),
        valid,
    )
    residual = (target.utility - output.utility).abs()
    sigma = output.sigma.clamp(1e-3, 3.0)
    heteroscedastic = _masked_mean(residual / sigma + sigma.log(), valid)
    zero = output.utility.sum() * 0.0
    classification = zero
    harmful_as_helpful = zero
    if output.class_logits is not None:
        per_pixel = F.cross_entropy(
            output.class_logits,
            target.classes.squeeze(1),
            reduction="none",
            ignore_index=-100,
            weight=torch.tensor((3.0, 0.5, 1.0), device=output.utility.device),
        ).unsqueeze(1)
        classification = _masked_mean(per_pixel, target.classification_valid)
        harmful_as_helpful = _masked_mean(
            torch.softmax(output.class_logits, dim=1)[:, 2:3], target.harmful
        )
    total = (
        config.utility_weight * utility
        + config.heteroscedastic_weight * heteroscedastic
        + config.classification_weight * classification
        + config.harmful_as_helpful_weight * harmful_as_helpful
    )
    return {
        "total": total,
        "utility": utility,
        "heteroscedastic": heteroscedastic,
        "classification": classification,
        "harmful_as_helpful": harmful_as_helpful,
        "predicted_utility_mean": _masked_mean(output.utility, valid),
        "sigma_mean": _masked_mean(output.sigma, valid),
    }


__all__ = ["ProposalUtilityLossConfig", "proposal_utility_losses"]
