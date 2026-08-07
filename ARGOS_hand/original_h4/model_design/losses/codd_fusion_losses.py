"""CODD Eq. (4) supervision for the fixed-BiDA fusion diagnostic."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from model_design.models.codd_style_fusion import CODDFusionOutput


@dataclass(frozen=True)
class CODDFusionLossConfig:
    # CODD reports 5px and 1px at its native disparity grid.  The ARGOS runner
    # converts these to cache-grid units before instantiating this config.
    tau_reset_px: float
    tau_fusion_px: float
    alpha_reg: float = 0.2
    alpha_disp: float = 1.0
    alpha_reset: float = 1.0
    alpha_fusion: float = 1.0
    huber_delta_px: float = 1.0


def codd_targets(raw: torch.Tensor, memory: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
    raw_error = (raw - gt).abs().detach()
    memory_error = (memory - gt).abs().detach()
    return {
        "raw_error": raw_error,
        "memory_error": memory_error,
        "valid": valid.bool().detach(),
        "error_difference": (memory_error - raw_error).detach(),
    }


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (value * mask.to(value.dtype)).sum() / mask.sum().clamp_min(1).to(value.dtype)


def codd_fusion_losses_with_gt(
    output: CODDFusionOutput,
    *,
    raw: torch.Tensor,
    memory: torch.Tensor,
    gt: torch.Tensor,
    valid: torch.Tensor,
    config: CODDFusionLossConfig,
) -> dict[str, torch.Tensor]:
    targets = codd_targets(raw, memory, gt, valid)
    mask = targets["valid"]
    diff = targets["error_difference"]
    reset_worse = mask & (diff > config.tau_reset_px)
    reset_better = mask & (diff < -config.tau_reset_px)
    reset = _masked_mean(output.reset_weight, reset_worse) + _masked_mean(1.0 - output.reset_weight, reset_better)
    fusion_worse = mask & (diff > config.tau_fusion_px)
    fusion_better = mask & (diff < -config.tau_fusion_px)
    fusion_tie = mask & ~(fusion_worse | fusion_better)
    fusion = (
        _masked_mean(output.fusion_weight, fusion_worse)
        + _masked_mean(1.0 - output.fusion_weight, fusion_better)
        + config.alpha_reg * _masked_mean((output.fusion_weight - 0.5).abs(), fusion_tie)
    )
    disparity = _masked_mean(F.huber_loss(output.fused_disparity, gt, delta=config.huber_delta_px, reduction="none"), mask)
    total = config.alpha_disp * disparity + config.alpha_reset * reset + config.alpha_fusion * fusion
    return {
        "total": total,
        "disparity": disparity,
        "reset": reset,
        "fusion": fusion,
        "reset_worse_fraction": reset_worse.float().mean(),
        "reset_better_fraction": reset_better.float().mean(),
        "fusion_tie_fraction": fusion_tie.float().mean(),
    }


__all__ = ["CODDFusionLossConfig", "codd_targets", "codd_fusion_losses_with_gt"]
