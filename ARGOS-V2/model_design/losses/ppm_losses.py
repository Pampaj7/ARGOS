"""Targeted selection losses for the universal learned long-memory adapter."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from model_design.losses.safety_losses import SafetyLossConfig, learned_t1_losses
from model_design.models.learned_ppm_selector import LearnedPPMOutput


@dataclass(frozen=True)
class PPMLossConfig:
    safety: SafetyLossConfig = field(default_factory=SafetyLossConfig)
    listwise_weight: float = 0.20
    regret_weight: float = 0.20
    entropy_weight: float = 0.001


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def learned_ppm_losses(
    output: LearnedPPMOutput,
    *,
    raw: torch.Tensor,
    candidates: torch.Tensor,
    candidate_valid: torch.Tensor,
    gt: torch.Tensor,
    valid: torch.Tensor,
    config: PPMLossConfig,
) -> dict[str, torch.Tensor]:
    """Geometry/safety plus listwise best-memory and selected-regret losses."""
    base = learned_t1_losses(
        output,  # compatible bounded-refiner output contract
        raw=raw,
        aligned_memory=output.aggregated_memory,
        gt=gt,
        valid=valid,
        safety_valid=valid,
        config=config.safety,
    )
    raw_error = (raw - gt).abs()
    candidate_error = (candidates - gt[:, None]).abs().masked_fill(~candidate_valid, torch.inf)
    all_error = torch.cat((raw_error[:, None], candidate_error), dim=1)
    best_target = all_error[:, :, 0].argmin(dim=1)
    raw_logits = torch.zeros_like(output.candidate_logits[:, :1])
    all_logits = torch.cat((raw_logits, output.candidate_logits), dim=1)[:, :, 0]
    listwise_map = F.cross_entropy(all_logits, best_target, reduction="none")[:, None]
    listwise = _masked_mean(listwise_map, valid)

    expected_error = (
        output.raw_abstain_weight * raw_error
        + (output.play_weights * candidate_error.nan_to_num(posinf=0.0)).sum(dim=1)
    )
    best_error = all_error.min(dim=1).values
    regret = _masked_mean((expected_error - best_error).clamp_min(0), valid)
    all_weights = torch.cat((output.raw_abstain_weight[:, None], output.play_weights), dim=1)
    entropy_map = -(all_weights * all_weights.clamp_min(1e-12).log()).sum(dim=1)
    entropy = _masked_mean(entropy_map, valid)
    total = (
        base["total"]
        + config.listwise_weight * listwise
        + config.regret_weight * regret
        - config.entropy_weight * entropy
    )
    return base | {
        "total": total,
        "listwise_selection": listwise,
        "selected_regret": regret,
        "play_entropy": entropy,
        "raw_abstain_mean": _masked_mean(output.raw_abstain_weight, valid),
    }
