"""Controlled raw-error classification, regression and uncertainty losses."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from model_design.data.raw_error_dataset import RawErrorTargets
from model_design.models.raw_error_detector import RawErrorOutput


@dataclass(frozen=True)
class RawErrorLossConfig:
    mode: str = "a4"
    false_positive_cost: float = 3.0
    classification_weight: float = 1.0
    regression_weight: float = 1.0
    uncertainty_weight: float = 0.25
    clean_weight: float = 0.25
    sigma_max: float = 10.0

    def __post_init__(self) -> None:
        if self.mode not in {"a0", "a1", "a2", "a3", "a4"}:
            raise ValueError("mode must be a0..a4")
        if self.false_positive_cost <= 0:
            raise ValueError("false_positive_cost must be positive")


def masked_mean(value: torch.Tensor, mask: torch.Tensor, weight=None) -> torch.Tensor:
    selected = mask.to(value.dtype)
    if weight is not None:
        selected = selected * weight.to(value.dtype)
    return (value * selected).sum() / selected.sum().clamp_min(1)


def raw_error_losses(
    output: RawErrorOutput,
    targets: RawErrorTargets,
    config: RawErrorLossConfig,
) -> dict[str, torch.Tensor]:
    class_valid = targets.classification_valid.bool()
    positive = class_valid & (targets.label > 0.5)
    negative = class_valid & ~positive
    positive_weight = negative.sum().to(output.logits.dtype) / positive.sum().clamp_min(1)
    weights = torch.where(
        targets.label > 0.5,
        positive_weight.clamp(1, 20),
        torch.as_tensor(config.false_positive_cost, device=output.logits.device),
    )
    classification = masked_mean(
        F.binary_cross_entropy_with_logits(
            output.logits, targets.label, reduction="none"
        ),
        class_valid,
        weights,
    )
    regression = masked_mean(
        F.smooth_l1_loss(output.mu, targets.error, reduction="none", beta=0.25),
        targets.regression_valid,
    )
    sigma = output.sigma.clamp(1e-3, config.sigma_max)
    nll_map = (targets.error - output.mu).abs() / sigma + torch.log(2 * sigma)
    uncertainty = masked_mean(nll_map, targets.regression_valid)
    sigma_penalty = masked_mean(
        F.relu(output.sigma - 5).square(), targets.regression_valid
    )
    clean = masked_mean(output.probability, targets.clean)

    zero = classification * 0
    total = zero
    if config.mode in {"a0", "a2", "a3", "a4"}:
        total = total + config.classification_weight * classification
    if config.mode in {"a1", "a2", "a3", "a4"}:
        total = total + config.regression_weight * regression
    if config.mode in {"a3", "a4"}:
        total = total + config.uncertainty_weight * (uncertainty + 0.01 * sigma_penalty)
    if config.mode == "a4":
        total = total + config.clean_weight * clean
    return {
        "total": total,
        "classification": classification,
        "regression": regression,
        "uncertainty": uncertainty,
        "sigma_penalty": sigma_penalty,
        "clean_authorization": clean,
        "probability_mean": masked_mean(output.probability, targets.regression_valid),
        "mu_mean": masked_mean(output.mu, targets.regression_valid),
        "sigma_mean": masked_mean(output.sigma, targets.regression_valid),
        "positive_prevalence": masked_mean(targets.label, class_valid),
        "classification_valid_count": class_valid.sum().to(output.mu.dtype),
        "regression_valid_count": targets.regression_valid.sum().to(output.mu.dtype),
    }


__all__ = ["RawErrorLossConfig", "masked_mean", "raw_error_losses"]
