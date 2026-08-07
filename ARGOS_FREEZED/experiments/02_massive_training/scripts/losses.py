"""Numerically identical loss logic extracted for ARGOS v2 campaign use."""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from argos_freezed.models.raw_multi_anchor_refiner import MultiAnchorEvidence, MultiAnchorOutput


@dataclass(frozen=True)
class MultiAnchorTargets:
    ground_truth: torch.Tensor
    raw_error: torch.Tensor
    candidate_error: torch.Tensor
    delta: torch.Tensor
    helpful: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class MultiAnchorLossConfig:
    margin_px: float = .10
    classification_weight: float = 1.0
    regression_weight: float = .50
    ranking_weight: float = .25
    fusion_weight: float = 1.0
    harmful_weight_regularizer: float = .20
    maximum_positive_weight: float = 50.0
    clipped_delta_px: float = 8.0


def multi_anchor_targets(raw, candidates, gt, gt_coverage, raw_valid, evidence, *, margin_px, coverage_threshold=.50):
    raw_error = (raw - gt).abs().detach(); candidate_error = (candidates - gt).abs().detach()
    delta = (raw_error - candidate_error).detach()
    base = (gt_coverage > coverage_threshold) & raw_valid.bool() & torch.isfinite(raw_error)
    valid = (base.expand_as(candidates) & evidence.available & torch.isfinite(delta)).detach()
    return MultiAnchorTargets(gt.detach(), raw_error, candidate_error, delta, (valid & (delta > margin_px)).detach(), valid)


def _masked_mean(value, mask):
    return value[mask].mean() if bool(mask.any()) else value.sum() * 0.0


def raw_multi_anchor_losses(output, evidence, targets, config, *, enable_fusion):
    valid = targets.valid; helpful = targets.helpful.float()
    positive_count = (targets.helpful & valid).sum().float(); negative_count = ((~targets.helpful) & valid).sum().float()
    positive_weight = (negative_count / positive_count.clamp_min(1)).clamp(1, config.maximum_positive_weight)
    classification = _masked_mean(F.binary_cross_entropy_with_logits(output.utility_logit, helpful, reduction="none", pos_weight=positive_weight), valid)
    clipped_delta = targets.delta.clamp(-config.clipped_delta_px, config.clipped_delta_px)
    balance = torch.where(targets.helpful, positive_weight, torch.ones_like(clipped_delta))
    regression = _masked_mean(F.smooth_l1_loss(output.predicted_delta, clipped_delta, reduction="none") * balance, valid)
    difference_true = targets.delta[:, :, None] - targets.delta[:, None, :]
    difference_pred = output.predicted_delta[:, :, None] - output.predicted_delta[:, None, :]
    pair_valid = valid[:, :, None] & valid[:, None, :] & (difference_true.abs() > config.margin_px)
    ranking = _masked_mean(F.relu(config.margin_px - difference_true.sign() * difference_pred), pair_valid)
    raw = evidence.raw.expand_as(evidence.candidates); prediction = raw + output.fusion_weight * (evidence.candidates - raw)
    fusion = _masked_mean(F.smooth_l1_loss(prediction, targets.ground_truth.expand_as(prediction), reduction="none") * balance, valid)
    weight_regularizer = _masked_mean(output.fusion_weight, valid & (targets.delta <= config.margin_px))
    total = config.classification_weight * classification + config.regression_weight * regression + config.ranking_weight * ranking
    if enable_fusion: total = total + config.fusion_weight * fusion + config.harmful_weight_regularizer * weight_regularizer
    return {"total": total, "classification": classification, "regression": regression, "ranking": ranking,
            "fusion": fusion, "weight_regularizer": weight_regularizer}
