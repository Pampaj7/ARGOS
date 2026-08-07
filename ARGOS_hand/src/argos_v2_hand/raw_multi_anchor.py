"""Minimal non-recurrent raw-anchor retrieval and fusion for ARGOS v2."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


RAW_ANCHOR_AGES = (1, 2, 4, 8)
PROVENANCE_RAW = 0.0
PROVENANCE_CORRECTED = 1.0
FEATURE_CHANNELS = 17


@dataclass(frozen=True)
class MultiAnchorEvidence:
    raw: torch.Tensor
    candidates: torch.Tensor
    candidate_valid: torch.Tensor
    warp_support: torch.Tensor
    fb_confidence: torch.Tensor
    ages: torch.Tensor
    provenance: torch.Tensor

    @property
    def available(self) -> torch.Tensor:
        return self.candidate_valid.bool() & self.warp_support.bool()


@dataclass(frozen=True)
class MultiAnchorOutput:
    utility_logit: torch.Tensor
    utility_probability: torch.Tensor
    predicted_delta: torch.Tensor
    fusion_weight: torch.Tensor
    selection_score: torch.Tensor


def _gradient_magnitude(value: torch.Tensor) -> torch.Tensor:
    dx = F.pad(value[..., 1:] - value[..., :-1], (0, 1, 0, 0))
    dy = F.pad(value[..., 1:, :] - value[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt(dx.square() + dy.square() + 1e-12)


def _masked_median_mad(candidates: torch.Tensor, available: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    masked = torch.where(available, candidates, torch.full_like(candidates, torch.nan))
    median = torch.nanmedian(masked, dim=1, keepdim=True).values
    deviation = (candidates - median).abs()
    mad = torch.nanmedian(torch.where(available, deviation, torch.full_like(deviation, torch.nan)), dim=1, keepdim=True).values
    count = available.sum(dim=1, keepdim=True)
    return torch.nan_to_num(median), torch.nan_to_num(mad), count


def build_candidate_features(evidence: MultiAnchorEvidence) -> torch.Tensor:
    raw, candidates = evidence.raw, evidence.candidates
    if raw.ndim != 4 or candidates.ndim != 4 or raw.shape[1] != 1:
        raise ValueError("raw must be [B,1,H,W] and candidates [B,K,H,W]")
    if candidates.shape[0] != raw.shape[0] or candidates.shape[-2:] != raw.shape[-2:]:
        raise ValueError("raw/candidate spatial contracts differ")
    for value in (evidence.candidate_valid, evidence.warp_support, evidence.fb_confidence):
        if value.shape != candidates.shape:
            raise ValueError("candidate mask/confidence shapes must match candidates")
    batch, count, height, width = candidates.shape
    if evidence.ages.shape not in {(count,), (batch, count)}:
        raise ValueError("ages must be [K] or [B,K]")
    if evidence.provenance.shape not in {(count,), (batch, count)}:
        raise ValueError("provenance must be [K] or [B,K]")
    available = evidence.available
    median, mad, witness = _masked_median_mad(candidates, available)
    raw_k = raw.expand(-1, count, -1, -1)
    residual = candidates - raw_k
    abs_residual = residual.abs()
    local_mean = F.avg_pool2d(abs_residual.reshape(batch * count, 1, height, width), 5, 1, 2).reshape(batch, count, height, width)
    local_square = F.avg_pool2d(residual.square().reshape(batch * count, 1, height, width), 5, 1, 2).reshape(batch, count, height, width)
    local_std = (local_square - F.avg_pool2d(residual.reshape(batch * count, 1, height, width), 5, 1, 2).reshape(batch, count, height, width).square()).clamp_min(0).sqrt()
    agreement = ((candidates - median).abs() <= (mad + .10)) & available
    agreement_count = agreement.sum(dim=1, keepdim=True)
    age = evidence.ages if evidence.ages.ndim == 2 else evidence.ages[None].expand(batch, -1)
    provenance = evidence.provenance if evidence.provenance.ndim == 2 else evidence.provenance[None].expand(batch, -1)
    scalar = lambda value: value[:, :, None, None].expand(-1, -1, height, width)
    raw_gradient = _gradient_magnitude(raw).expand(-1, count, -1, -1)
    candidate_gradient = _gradient_magnitude(candidates.reshape(batch * count, 1, height, width)).reshape(batch, count, height, width)
    maps = (
        raw_k.clamp(0, 64) / 64, candidates.clamp(0, 64) / 64, residual.clamp(-16, 16) / 16,
        abs_residual.clamp(0, 16) / 16, local_mean.clamp(0, 8) / 8, local_std.clamp(0, 8) / 8,
        scalar(age.float().clamp(0, 8) / 8), evidence.candidate_valid.float(), evidence.warp_support.float(),
        evidence.fb_confidence.clamp(0, 1), witness.float().expand(-1, count, -1, -1) / max(count, 1),
        median.expand(-1, count, -1, -1).clamp(0, 64) / 64, mad.expand(-1, count, -1, -1).clamp(0, 8) / 8,
        (candidates - median).abs().clamp(0, 8) / 8, (raw - median).abs().expand(-1, count, -1, -1).clamp(0, 8) / 8,
        agreement_count.float().expand(-1, count, -1, -1) / max(count, 1), scalar(provenance.float()),
    )
    features = torch.stack(maps, dim=2)
    if features.shape != (batch, count, FEATURE_CHANNELS, height, width):
        raise RuntimeError(f"unexpected feature shape {tuple(features.shape)}")
    return features


class _SharedBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.GroupNorm(8, channels), nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.GroupNorm(8, channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.silu(value + self.net(value))


class RawMultiAnchorRefiner(nn.Module):
    def __init__(self, channels: int = 32, blocks: int = 3, max_delta_px: float = 8.0) -> None:
        super().__init__()
        if channels % 8:
            raise ValueError("channels must be divisible by 8")
        self.channels, self.blocks, self.max_delta_px = channels, blocks, float(max_delta_px)
        self.encoder = nn.Sequential(
            nn.Conv2d(FEATURE_CHANNELS, channels, 3, padding=1, bias=False), nn.GroupNorm(8, channels), nn.SiLU(inplace=True),
            *[_SharedBlock(channels) for _ in range(blocks)],
        )
        self.head = nn.Conv2d(channels, 3, 1)
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([-2.0, 0.0, -3.0]))

    def forward(self, evidence: MultiAnchorEvidence) -> MultiAnchorOutput:
        features = build_candidate_features(evidence)
        batch, count, channels, height, width = features.shape
        raw_output = self.head(self.encoder(features.reshape(batch * count, channels, height, width)))
        raw_output = raw_output.reshape(batch, count, 3, height, width)
        logit = raw_output[:, :, 0]
        delta = torch.tanh(raw_output[:, :, 1]) * self.max_delta_px
        weight = torch.sigmoid(raw_output[:, :, 2])
        probability = torch.sigmoid(logit)
        score = probability * delta
        unavailable = ~evidence.available
        return MultiAnchorOutput(logit, probability, delta, weight, score.masked_fill(unavailable, -torch.inf))


def retrieve_and_fuse(raw: torch.Tensor, evidence: MultiAnchorEvidence, output: MultiAnchorOutput, *, probability_threshold: float, utility_threshold_px: float, hard: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    score, chosen = output.selection_score.max(dim=1, keepdim=True)
    candidate = torch.gather(evidence.candidates, 1, chosen)
    probability = torch.gather(output.utility_probability, 1, chosen)
    weight = torch.gather(output.fusion_weight, 1, chosen)
    chosen_available = torch.gather(evidence.available, 1, chosen)
    accepted = chosen_available & (probability >= probability_threshold) & (score >= utility_threshold_px)
    if hard:
        prediction = torch.where(accepted, candidate, raw)
        effective_weight = accepted.to(raw.dtype)
    else:
        effective_weight = torch.where(accepted, weight, torch.zeros_like(weight))
        prediction = raw + effective_weight * (candidate - raw)
        prediction = torch.where(accepted, prediction, raw)
    return prediction, accepted, chosen, effective_weight
