"""Interpretable Q0 error, uncertainty, advantage and ranking losses."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from model_design.data.quality_prediction_dataset import QualityCandidateBatch
from model_design.models.quality_predictor import QualityPredictionOutput


@dataclass(frozen=True)
class QualityLossConfig:
    target_mode: str = "uncertainty"
    patch_size: int = 1
    indifference_margin_px: float = 0.10
    rank_margin_px: float = 0.05
    error_weight: float = 1.0
    ranking_weight: float = 0.0
    uncertainty_weight: float = 1.0
    advantage_weight: float = 0.0
    sigma_penalty_weight: float = 0.01
    sigma_max_likelihood: float = 10.0
    hard_negative_weight: float = 0.0
    epsilon: float = 1e-3

    def __post_init__(self) -> None:
        if self.target_mode not in {"absolute", "log", "advantage", "joint", "uncertainty"}:
            raise ValueError("unknown Q0 target mode")
        if self.patch_size not in {1, 8, 16}:
            raise ValueError("patch_size must be 1, 8 or 16")


def masked_mean(
    value: torch.Tensor,
    mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    weight = mask.to(value.dtype)
    if sample_weight is not None:
        weight = weight * sample_weight.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def hard_negative_weights(
    candidates: QualityCandidateBatch,
    valid: torch.Tensor,
    *,
    indifference_margin_px: float,
    boost: float,
) -> torch.Tensor:
    """Balanced real-label pixel weights, never synthetic supervision.

    Raw-clean/middle/wrong and memory-better/raw-better/indifferent strata are
    inverse-frequency balanced within a batch. Predeclared observed failure
    masks then receive a bounded multiplicative boost. Output is ``[B,K,H,W]``.
    """
    if boost <= 0:
        return torch.ones_like(valid, dtype=candidates.target_error.dtype)
    error = candidates.target_error[:, :, 0]
    raw = error[:, 0]
    memory = error[:, 1:].masked_fill(~valid[:, 1:], torch.inf).min(dim=1).values
    pixel_valid = valid[:, 0] & torch.isfinite(memory)
    memory_better = pixel_valid & (memory + indifference_margin_px < raw)
    raw_better = pixel_valid & (raw + indifference_margin_px < memory)
    indifferent = pixel_valid & ~(memory_better | raw_better)
    raw_clean = pixel_valid & (raw <= 0.50)
    raw_wrong = pixel_valid & (raw > 1.0)
    raw_middle = pixel_valid & ~(raw_clean | raw_wrong)

    def balanced(groups: tuple[torch.Tensor, ...]) -> torch.Tensor:
        result = torch.ones_like(raw)
        nonempty = [group for group in groups if bool(group.any())]
        total = pixel_valid.sum().to(raw.dtype).clamp_min(1)
        for group in nonempty:
            class_weight = total / (len(nonempty) * group.sum().to(raw.dtype).clamp_min(1))
            result = torch.where(group, class_weight.clamp(0.25, 4.0), result)
        return result

    weight = torch.sqrt(
        balanced((memory_better, raw_better, indifferent))
        * balanced((raw_clean, raw_middle, raw_wrong))
    )
    hard = torch.zeros_like(pixel_valid)
    for failure in candidates.failure_masks.values():
        hard |= failure[:, 0].bool()
    weight = weight * (1.0 + float(boost) * hard.to(weight.dtype))
    return weight.clamp(0.25, 6.0)[:, None].expand_as(valid)


def masked_patch_mean(
    value: torch.Tensor,
    valid: torch.Tensor,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked non-overlapping patch means for ``[B,K,H,W]`` maps."""
    if patch_size == 1:
        return value, valid.bool()
    b, k, h, w = value.shape
    flat_value = value.reshape(b * k, 1, h, w)
    flat_valid = valid.reshape(b * k, 1, h, w).to(value.dtype)
    numerator = F.avg_pool2d(flat_value * flat_valid, patch_size, stride=patch_size)
    coverage = F.avg_pool2d(flat_valid, patch_size, stride=patch_size)
    pooled = numerator / coverage.clamp_min(1e-8)
    shape = (b, k, pooled.shape[-2], pooled.shape[-1])
    return pooled.reshape(shape), (coverage > 0).reshape(shape)


def pairwise_ranking_loss(
    predicted_error: torch.Tensor,
    target_error: torch.Tensor,
    valid: torch.Tensor,
    *,
    indifference_margin_px: float,
    rank_margin_px: float,
    sample_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hinge ranking over valid non-tied candidate pairs."""
    losses = []
    masks = []
    weights = []
    candidates = predicted_error.shape[1]
    for left in range(candidates):
        for right in range(left + 1, candidates):
            true_difference = target_error[:, right] - target_error[:, left]
            predicted_difference = predicted_error[:, right] - predicted_error[:, left]
            pair_valid = valid[:, left] & valid[:, right] & (
                true_difference.abs() > indifference_margin_px
            )
            sign = true_difference.sign()
            losses.append(F.relu(rank_margin_px - sign * predicted_difference))
            masks.append(pair_valid)
            if sample_weight is not None:
                weights.append(0.5 * (sample_weight[:, left] + sample_weight[:, right]))
    stacked_loss = torch.stack(losses, dim=1)
    stacked_valid = torch.stack(masks, dim=1)
    stacked_weight = torch.stack(weights, dim=1) if weights else None
    return masked_mean(stacked_loss, stacked_valid, stacked_weight), stacked_valid.sum()


def laplace_uncertainty_loss(
    predicted_error: torch.Tensor,
    target_error: torch.Tensor,
    sigma: torch.Tensor,
    valid: torch.Tensor,
    *,
    sigma_max: float = 10.0,
    sigma_penalty_weight: float = 0.01,
    sample_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stable Laplace NLL plus an explicit anti-inflation penalty."""
    safe_sigma = sigma.clamp(1e-3, sigma_max)
    nll_map = (target_error - predicted_error).abs() / safe_sigma + torch.log(2 * safe_sigma)
    nll = masked_mean(nll_map, valid, sample_weight)
    penalty = masked_mean(F.relu(sigma - 5.0).square(), valid, sample_weight)
    return nll + sigma_penalty_weight * penalty, nll, penalty


def quality_prediction_losses(
    output: QualityPredictionOutput,
    candidates: QualityCandidateBatch,
    config: QualityLossConfig,
) -> dict[str, torch.Tensor]:
    """Compute the predeclared Q0 loss composition and log every term."""
    target_error = candidates.target_error[:, :, 0]
    target_advantage = candidates.target_advantage[:, :, 0]
    valid = candidates.target_valid[:, :, 0].bool()
    mu, sigma, advantage = output.mu, output.sigma, output.advantage
    sample_weight = hard_negative_weights(
        candidates, valid,
        indifference_margin_px=config.indifference_margin_px,
        boost=config.hard_negative_weight,
    )
    if config.patch_size > 1:
        target_error, patch_valid = masked_patch_mean(target_error, valid, config.patch_size)
        target_advantage, _ = masked_patch_mean(target_advantage, valid, config.patch_size)
        mu, _ = masked_patch_mean(mu, valid, config.patch_size)
        sigma, _ = masked_patch_mean(sigma, valid, config.patch_size)
        advantage, _ = masked_patch_mean(advantage, valid, config.patch_size)
        sample_weight, _ = masked_patch_mean(sample_weight, valid, config.patch_size)
        valid = patch_valid

    error = masked_mean(
        F.smooth_l1_loss(mu, target_error, reduction="none", beta=0.25), valid,
        sample_weight,
    )
    log_error = masked_mean(
        F.smooth_l1_loss(
            torch.log(mu + config.epsilon),
            torch.log(target_error + config.epsilon),
            reduction="none",
            beta=0.25,
        ),
        valid, sample_weight,
    )
    advantage_loss = masked_mean(
        F.smooth_l1_loss(advantage, target_advantage, reduction="none", beta=0.25),
        valid, sample_weight,
    )
    uncertainty, nll, sigma_penalty = laplace_uncertainty_loss(
        mu, target_error, sigma, valid,
        sigma_max=config.sigma_max_likelihood,
        sigma_penalty_weight=config.sigma_penalty_weight,
        sample_weight=sample_weight,
    )
    ranking, ranked_pair_count = pairwise_ranking_loss(
        mu, target_error, valid,
        indifference_margin_px=config.indifference_margin_px,
        rank_margin_px=config.rank_margin_px,
        sample_weight=sample_weight,
    )

    zero = error * 0
    if config.target_mode == "absolute":
        primary_error, uncertainty_term, advantage_term = error, zero, zero
    elif config.target_mode == "log":
        primary_error, uncertainty_term, advantage_term = log_error, zero, zero
    elif config.target_mode == "advantage":
        primary_error, uncertainty_term, advantage_term = zero, zero, advantage_loss
    elif config.target_mode == "joint":
        primary_error, uncertainty_term, advantage_term = error, zero, advantage_loss
    else:
        primary_error, uncertainty_term, advantage_term = error, uncertainty, zero
    total = (
        config.error_weight * primary_error
        + config.ranking_weight * ranking
        + config.uncertainty_weight * uncertainty_term
        + config.advantage_weight * advantage_term
    )
    # Advantage-only and joint modes must train their advantage head even when
    # callers leave the generic advantage weight at zero.
    if config.target_mode in {"advantage", "joint"} and config.advantage_weight == 0:
        total = total + advantage_loss
    return {
        "total": total,
        "absolute_error": error,
        "log_error": log_error,
        "advantage": advantage_loss,
        "ranking": ranking,
        "uncertainty": uncertainty,
        "laplace_nll": nll,
        "sigma_penalty": sigma_penalty,
        "ranked_pair_count": ranked_pair_count.to(mu.dtype),
        "valid_count": valid.sum().to(mu.dtype),
        "mu_mean": masked_mean(mu, valid, sample_weight),
        "sigma_mean": masked_mean(sigma, valid, sample_weight),
        "target_error_mean": masked_mean(target_error, valid, sample_weight),
        "hard_weight_mean": masked_mean(sample_weight, valid),
    }


__all__ = [
    "QualityLossConfig",
    "laplace_uncertainty_loss",
    "hard_negative_weights",
    "masked_mean",
    "masked_patch_mean",
    "pairwise_ranking_loss",
    "quality_prediction_losses",
]
