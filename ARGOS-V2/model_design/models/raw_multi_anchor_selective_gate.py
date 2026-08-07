"""Post-hoc veto gates for the frozen ARGOS v2 raw multi-anchor proposal."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .raw_multi_anchor_refiner import MultiAnchorEvidence, MultiAnchorOutput, _masked_median_mad


GATE_FEATURE_NAMES = (
    "top1_score", "top1_top2_margin", "fusion_weight", "selected_age",
    "signed_anchor_raw", "absolute_anchor_raw", "update_magnitude",
    "temporal_median", "temporal_mad", "raw_median_deviation",
    "anchor_median_deviation", "valid_anchor_fraction", "agreement_fraction",
    "selected_support", "selected_fb_confidence", "local_disagreement_mean",
    "local_disagreement_variance", "local_update_mean", "local_update_variance",
    "selected_valid",
)
GATE_FEATURE_CHANNELS = len(GATE_FEATURE_NAMES)


@dataclass(frozen=True)
class FrozenProposal:
    proposed: torch.Tensor
    eligible: torch.Tensor
    chosen: torch.Tensor
    candidate: torch.Tensor
    fusion_weight: torch.Tensor
    top1_score: torch.Tensor
    top2_margin: torch.Tensor


@dataclass(frozen=True)
class GateOutput:
    harm_logit: torch.Tensor
    harm_probability: torch.Tensor
    predicted_gain: torch.Tensor | None


def frozen_proposal(
    evidence: MultiAnchorEvidence,
    output: MultiAnchorOutput,
    *,
    probability_threshold: float,
    utility_threshold_px: float,
) -> FrozenProposal:
    """Reconstruct the frozen soft proposal and its original authorization."""
    count = output.selection_score.shape[1]
    top = torch.topk(output.selection_score, k=min(2, count), dim=1)
    chosen = top.indices[:, :1]
    score = top.values[:, :1]
    runner_up = top.values[:, 1:2] if count > 1 else torch.zeros_like(score)
    margin = torch.nan_to_num(score - runner_up, nan=0.0, posinf=0.0, neginf=0.0)
    candidate = torch.gather(evidence.candidates, 1, chosen)
    probability = torch.gather(output.utility_probability, 1, chosen)
    weight = torch.gather(output.fusion_weight, 1, chosen)
    available = torch.gather(evidence.available, 1, chosen)
    eligible = available & (probability >= probability_threshold) & (score >= utility_threshold_px)
    proposed = evidence.raw + weight * (candidate - evidence.raw)
    return FrozenProposal(proposed, eligible, chosen, candidate, weight, score, margin)


def _local_mean_variance(value: torch.Tensor, kernel: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    mean = F.avg_pool2d(value, kernel, stride=1, padding=kernel // 2)
    square = F.avg_pool2d(value.square(), kernel, stride=1, padding=kernel // 2)
    return mean, (square - mean.square()).clamp_min(0)


def build_gate_features(evidence: MultiAnchorEvidence, proposal: FrozenProposal) -> torch.Tensor:
    """Return compact causal inference features ``[B,C,H,W]`` with no GT."""
    available = evidence.available
    median, mad, witness = _masked_median_mad(evidence.candidates, available)
    residual = proposal.candidate - evidence.raw
    absolute = residual.abs()
    update = (proposal.proposed - evidence.raw).abs()
    disagreement_mean, disagreement_variance = _local_mean_variance(absolute)
    update_mean, update_variance = _local_mean_variance(update)
    agreement = (((evidence.candidates - median).abs() <= mad + .10) & available).sum(dim=1, keepdim=True)
    age_map = evidence.ages
    if age_map.ndim == 1:
        age_map = age_map[None].expand(evidence.raw.shape[0], -1)
    age_map = age_map[:, :, None, None].expand(-1, -1, *evidence.raw.shape[-2:])
    selected_age = torch.gather(age_map, 1, proposal.chosen).float()
    support = torch.gather(evidence.warp_support, 1, proposal.chosen).float()
    fb = torch.gather(evidence.fb_confidence, 1, proposal.chosen)
    valid = torch.gather(evidence.available, 1, proposal.chosen).float()
    count = max(evidence.candidates.shape[1], 1)
    maps = (
        proposal.top1_score.clamp(-2, 2) / 2,
        proposal.top2_margin.clamp(-2, 2) / 2,
        proposal.fusion_weight,
        selected_age.clamp(0, 8) / 8,
        residual.clamp(-16, 16) / 16,
        absolute.clamp(0, 16) / 16,
        update.clamp(0, 8) / 8,
        median.clamp(0, 64) / 64,
        mad.clamp(0, 8) / 8,
        (evidence.raw - median).abs().clamp(0, 8) / 8,
        (proposal.candidate - median).abs().clamp(0, 8) / 8,
        witness.float() / count,
        agreement.float() / count,
        support,
        fb.clamp(0, 1),
        disagreement_mean.clamp(0, 8) / 8,
        disagreement_variance.clamp(0, 64) / 64,
        update_mean.clamp(0, 8) / 8,
        update_variance.clamp(0, 64) / 64,
        valid,
    )
    features = torch.cat(maps, dim=1)
    if features.shape[1] != GATE_FEATURE_CHANNELS:
        raise RuntimeError(f"unexpected gate feature shape {tuple(features.shape)}")
    return torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)


class LinearHarmGate(nn.Module):
    def __init__(self, channels: int = GATE_FEATURE_CHANNELS) -> None:
        super().__init__()
        self.head = nn.Conv2d(channels, 1, 1)

    def forward(self, features: torch.Tensor) -> GateOutput:
        logit = self.head(features)
        return GateOutput(logit, torch.sigmoid(logit), None)


class SelectiveRiskGate(nn.Module):
    """Small pointwise gate; spatial statistics are explicit frozen features."""

    def __init__(self, channels: int = GATE_FEATURE_CHANNELS, hidden: int = 64, depth: int = 3, joint: bool = False) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least two")
        layers: list[nn.Module] = [nn.Conv2d(channels, hidden, 1), nn.SiLU(inplace=True)]
        for _ in range(depth - 2):
            layers.extend((nn.Conv2d(hidden, hidden, 1), nn.SiLU(inplace=True)))
        self.encoder = nn.Sequential(*layers)
        self.harm_head = nn.Conv2d(hidden, 1, 1)
        self.gain_head = nn.Conv2d(hidden, 1, 1) if joint else None

    def forward(self, features: torch.Tensor) -> GateOutput:
        encoded = self.encoder(features)
        harm = self.harm_head(encoded)
        gain = self.gain_head(encoded) if self.gain_head is not None else None
        return GateOutput(harm, torch.sigmoid(harm), gain)


def apply_veto(raw: torch.Tensor, proposal: FrozenProposal, gate_accept: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a close-only veto and preserve raw bit-exactly on every rejection."""
    accepted = proposal.eligible & gate_accept.bool()
    return torch.where(accepted, proposal.proposed, raw), accepted


__all__ = [
    "GATE_FEATURE_CHANNELS", "GATE_FEATURE_NAMES", "FrozenProposal", "GateOutput",
    "LinearHarmGate", "SelectiveRiskGate", "apply_veto", "build_gate_features", "frozen_proposal",
]
