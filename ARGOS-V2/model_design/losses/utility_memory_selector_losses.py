"""Minimal, asymmetric utility losses for the ARGOS v2 causal selector."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from model_design.data.utility_memory_selector_dataset import UtilityTargets
from model_design.models.utility_memory_selector import UtilitySelectorOutput


@dataclass(frozen=True)
class UtilitySelectorLossConfig:
    objective: str = "legacy"
    classification_weight: float = 1.0
    gain_weight: float = 1.0
    harm_weight: float = 1.5
    harmful_selection_weight: float = 2.0
    harmful_class_weight: float = 4.0
    huber_delta_px: float = 0.10
    signed_utility_weight: float = 1.0
    policy_risk_weight: float = 1.0
    policy_temperature_px: float = 0.10
    utility_clip_px: float = 5.0
    selective_target_coverage: float = 0.02
    selective_coverage_weight: float = 32.0
    selective_risk_weight: float = 10.0


def _mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    items = value[valid.bool()]
    return items.mean() if items.numel() else value.sum() * 0.0


def _category_mean(value: torch.Tensor, categories: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Equal category weighting without fabricating or duplicating pixels."""
    means = [value[mask.bool()].mean() for mask in categories if mask.any()]
    return torch.stack(means).mean() if means else value.sum() * 0.0


def _raw_error_strata(target: UtilityTargets, *classes: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Cross decision classes with preregistered raw-error difficulty bins."""
    bins = (
        target.raw_error <= 1.0,
        (target.raw_error > 1.0) & (target.raw_error <= 3.0),
        target.raw_error > 3.0,
    )
    return tuple(cls & raw_bin for cls in classes for raw_bin in bins)


def utility_selector_losses(
    output: UtilitySelectorOutput,
    target: UtilityTargets,
    config: UtilitySelectorLossConfig,
) -> dict[str, torch.Tensor]:
    valid = target.valid
    if config.objective not in {"legacy", "utility_risk", "utility_calibrated", "utility_weighted", "selective_utility"}:
        raise ValueError(f"unknown selector objective: {config.objective}")
    if config.objective in {"utility_risk", "utility_calibrated", "utility_weighted", "selective_utility"}:
        helpful = valid & (target.helpful_gain > 0)
        harmful = valid & (target.harmful_magnitude > 0)
        indifferent = valid & ~helpful & ~harmful
        decisive = helpful | harmful
        label = helpful.float()
        decision_strata = _raw_error_strata(target, helpful, harmful)
        all_strata = _raw_error_strata(target, helpful, harmful, indifferent)
        if config.objective in {"utility_calibrated", "utility_weighted", "selective_utility"}:
            # The operational label remains \"memory beats raw by epsilon\"
            # over *all* valid pixels.  Treating indifference as unlabelled
            # made the probability head arbitrary on the dominant abstention
            # region and destroyed global ranking.  Conditionality belongs to
            # the cost heads below, not to the abstention label.
            if config.objective in {"utility_weighted", "selective_utility"}:
                # Selective action errors are asymmetric: missing a small
                # improvement is preferable to accepting a large harmful
                # replacement.  Weights use only the supervised utility
                # target and remain bounded, so this is a cost-sensitive BCE,
                # not a learned policy or a new architecture.
                class_weight = torch.where(
                    helpful,
                    1.0 + target.helpful_gain.clamp(max=config.utility_clip_px),
                    torch.where(
                        harmful,
                        1.0 + config.harmful_class_weight * target.harmful_magnitude.clamp(max=config.utility_clip_px),
                        torch.full_like(target.utility, 0.25),
                    ),
                )
            else:
                class_weight = torch.where(harmful, config.harmful_class_weight, 1.0)
            classification = _mean(
                F.binary_cross_entropy_with_logits(output.memory_better_logit, label, reduction="none") * class_weight,
                valid,
            )
            signed_utility_mask = valid
        else:
            classification = _category_mean(
                F.binary_cross_entropy_with_logits(output.memory_better_logit, label, reduction="none"),
                decision_strata,
            )
            signed_utility_mask = all_strata
        # Magnitudes are conditional: zero-valued pixels from other classes no
        # longer dominate either regression head.
        gain = _mean(
            F.huber_loss(output.expected_positive_gain, target.helpful_gain, reduction="none", delta=config.huber_delta_px),
            helpful,
        )
        harm = _mean(
            F.huber_loss(output.expected_harmful_magnitude, target.harmful_magnitude, reduction="none", delta=config.huber_delta_px),
            harmful,
        )
        predicted_utility = output.conditional_expected_utility
        signed_error = F.huber_loss(
            predicted_utility, target.supervision_utility.clamp(-config.utility_clip_px, config.utility_clip_px),
            reduction="none", delta=config.huber_delta_px,
        )
        signed_utility = (_mean(signed_error, signed_utility_mask)
                          if config.objective == "utility_calibrated"
                          else _category_mean(signed_error, all_strata))
        if config.objective == "utility_risk":
            soft_authorization = torch.sigmoid(predicted_utility / config.policy_temperature_px)
            # Retained only as a documented diagnostic: direct action-risk
            # optimization collapsed to dense selection on validation.
            policy_risk = _category_mean(
                -soft_authorization * target.supervision_utility.clamp(-config.utility_clip_px, config.utility_clip_px),
                decision_strata,
            )
        else:
            policy_risk = predicted_utility.sum() * 0.0
        # Pixel-valued regression terms are naturally ~100x smaller than BCE
        # at cache resolution.  Fixed scaling makes the action objective
        # material without changing architecture or data.
        selective_risk = predicted_utility.sum() * 0.0
        selective_coverage = predicted_utility.sum() * 0.0
        selective_coverage_penalty = predicted_utility.sum() * 0.0
        if config.objective == "selective_utility":
            selection = output.memory_better_probability[valid]
            local_utility = target.supervision_utility[valid].clamp(-config.utility_clip_px, config.utility_clip_px)
            selective_coverage = selection.mean() if selection.numel() else selection.sum()
            # Normalization is essential: the rejected direct policy term
            # preferred dense selection whenever mean utility was positive.
            selective_risk = -(selection * local_utility).sum() / selection.sum().clamp_min(1e-6)
            selective_coverage_penalty = F.relu(config.selective_target_coverage - selective_coverage).square()
        total = (config.classification_weight * classification
                 + 5.0 * config.gain_weight * gain
                 + 5.0 * config.harm_weight * harm
                 + 1.0 * config.signed_utility_weight * signed_utility
                 + config.policy_risk_weight * policy_risk
                 + config.selective_risk_weight * selective_risk
                 + config.selective_coverage_weight * selective_coverage_penalty)
        return {
            "total": total, "classification": classification, "positive_gain": gain,
            "harmful_magnitude": harm, "signed_utility": signed_utility,
            "policy_risk": policy_risk, "selective_risk": selective_risk,
            "selective_coverage": selective_coverage,
            "selective_coverage_penalty": selective_coverage_penalty,
            "decisive_fraction": _mean(decisive.float(), valid),
        }

    label = target.memory_better.float()
    class_weight = torch.where(target.harmful_magnitude > 0, config.harmful_class_weight, 1.0)
    classification = _mean(
        F.binary_cross_entropy_with_logits(output.memory_better_logit, label, reduction="none") * class_weight,
        valid,
    )
    gain = _mean(F.huber_loss(output.expected_positive_gain, target.helpful_gain, reduction="none", delta=config.huber_delta_px), valid)
    harm = _mean(F.huber_loss(output.expected_harmful_magnitude, target.harmful_magnitude, reduction="none", delta=config.huber_delta_px), valid)
    harmful_selection = _mean(
        output.memory_better_probability * (target.harmful_magnitude > 0).float() * target.harmful_magnitude.clamp(max=5), valid
    )
    total = (config.classification_weight * classification + config.gain_weight * gain
             + config.harm_weight * harm + config.harmful_selection_weight * harmful_selection)
    return {
        "total": total, "classification": classification, "positive_gain": gain, "harmful_magnitude": harm,
        "harmful_selection": harmful_selection,
    }


__all__ = ["UtilitySelectorLossConfig", "utility_selector_losses"]
