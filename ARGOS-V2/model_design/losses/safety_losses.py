"""Interpretable geometry, selector, and safety losses for the t-1 refiner."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from model_design.models.learned_t1_refiner import RefinerOutput


@dataclass(frozen=True)
class SafetyLossConfig:
    sigma_error_px: float = 3.0
    memory_margin_px: float = 0.05
    memory_temperature_px: float = 0.10
    clean_pixel_threshold_px: float = 0.50
    clean_frame_epe_threshold_px: float = 1.00
    ranking_tolerance_px: float = 0.02
    geometry_weight: float = 1.0
    correction_weight: float = 0.25
    error_gate_weight: float = 0.10
    memory_gate_weight: float = 0.10
    clean_weight: float = 0.20
    ranking_weight: float = 0.20
    update_weight: float = 0.01


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _frame_masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(value.dtype)
    numerator = (value * weight).flatten(1).sum(1)
    denominator = weight.flatten(1).sum(1).clamp_min(1.0)
    return numerator / denominator


def learned_t1_losses(
    output: RefinerOutput,
    *,
    raw: torch.Tensor,
    aligned_memory: torch.Tensor,
    gt: torch.Tensor,
    valid: torch.Tensor,
    safety_valid: torch.Tensor | None = None,
    config: SafetyLossConfig,
) -> dict[str, torch.Tensor]:
    """Return every unweighted term plus ``total`` and supervised targets.

    The memory target ignores sub-margin wins and transitions softly across the
    margin. Clean preservation covers both clean pixels and all pixels in a clean
    frame, using the explicit thresholds stored in ``SafetyLossConfig``.
    """
    valid = valid.bool()
    safety_valid = valid if safety_valid is None else safety_valid.bool()
    raw_error = (raw - gt).abs()
    memory_error = (aligned_memory - gt).abs()
    refined_error = (output.disparity - gt).abs()
    advantage = raw_error - memory_error

    target_error = (raw_error / config.sigma_error_px).clamp(0, 1)
    target_memory = torch.sigmoid(
        (advantage - config.memory_margin_px) / config.memory_temperature_px
    )
    target_correction = (gt - raw).clamp(-float(output.tau), float(output.tau))

    geometry = _masked_mean(torch.sqrt(refined_error.square() + 1e-4) - 1e-2, valid)
    correction = _masked_mean(
        F.smooth_l1_loss(output.update, target_correction, reduction="none", beta=0.25),
        valid,
    )
    error_gate = _masked_mean(
        F.binary_cross_entropy_with_logits(output.error_logits, target_error, reduction="none"),
        valid,
    )
    memory_gate = _masked_mean(
        F.binary_cross_entropy_with_logits(output.memory_logits, target_memory, reduction="none"),
        valid,
    )

    clean_pixel = safety_valid & (raw_error <= config.clean_pixel_threshold_px)
    raw_frame_epe = _frame_masked_mean(raw_error, safety_valid)
    frame_valid_count = safety_valid.flatten(1).sum(1)
    clean_frame = (frame_valid_count > 0) & (
        raw_frame_epe <= config.clean_frame_epe_threshold_px
    )
    clean_frame_pixels = safety_valid & clean_frame[:, None, None, None]
    clean_pixel_update = _masked_mean(output.update.abs(), clean_pixel)
    clean_frame_update = _masked_mean(output.update.abs(), clean_frame_pixels)
    clean_preservation = 0.5 * (clean_pixel_update + clean_frame_update)

    safety_ranking = _masked_mean(
        F.relu(refined_error - raw_error - config.ranking_tolerance_px), safety_valid
    )
    update_magnitude = _masked_mean(output.update.abs(), valid)

    total = (
        config.geometry_weight * geometry
        + config.correction_weight * correction
        + config.error_gate_weight * error_gate
        + config.memory_gate_weight * memory_gate
        + config.clean_weight * clean_preservation
        + config.ranking_weight * safety_ranking
        + config.update_weight * update_magnitude
    )
    return {
        "total": total,
        "geometry": geometry,
        "correction": correction,
        "error_gate": error_gate,
        "memory_gate": memory_gate,
        "clean_preservation": clean_preservation,
        "clean_pixel_update": clean_pixel_update,
        "clean_frame_update": clean_frame_update,
        "safety_ranking": safety_ranking,
        "update_magnitude": update_magnitude,
        "target_error_mean": _masked_mean(target_error, valid),
        "target_memory_mean": _masked_mean(target_memory, valid),
        "clean_pixel_ratio": clean_pixel.float().mean(),
        "clean_frame_ratio": clean_frame.float().mean(),
    }
