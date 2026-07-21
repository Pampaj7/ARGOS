"""Compact, candidate-conditioned stereo correspondence evidence for ARGOS v2.

This module is deliberately a deterministic *measurement* frontend.  It does
not estimate or alter disparity, access a stereo backbone's internals, use a
future frame, or change causal/GT validity masks.  Given the current rectified
left/right pair and two finished left-disparity candidates, it measures a
small robust census cost curve around each candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from model_design.external_components.stereo_photometric import (
    _as_unit_rgb,
    ternary_census_cost,
    warp_right_to_left,
)


MatchingEvidenceMode = Literal["none", "cost", "shape", "full"]
OFFSETS: tuple[int, ...] = (-4, -2, -1, 0, 1, 2, 4)
_CENTER = OFFSETS.index(0)


@dataclass(frozen=True)
class CandidateCostStatistics:
    """Stereo support statistics for one candidate, all `[B,1,H,W]` except curve.

    ``curve`` is a seven-channel census-cost curve in the exact order in
    :data:`OFFSETS`; ``curve_support`` has the same channels.  All statistic
    maps are fixed-range measurements, not learned confidence values.
    """

    curve: torch.Tensor
    curve_support: torch.Tensor
    candidate_cost: torch.Tensor
    local_minimum_cost: torch.Tensor
    offset_to_local_minimum: torch.Tensor
    best_second_margin: torch.Tensor
    local_curvature: torch.Tensor
    local_sharpness: torch.Tensor
    local_support_fraction: torch.Tensor
    candidate_support: torch.Tensor


@dataclass(frozen=True)
class StereoMatchingEvidence:
    """Candidate-conditioned, current-frame stereo evidence.

    ``features`` contains only frozen, backbone-independent maps.  It must be
    treated as a selector input, never as an evaluation validity mask; the
    selector's canonical causal support remains defined by BiDA evidence.
    """

    raw: CandidateCostStatistics
    memory: CandidateCostStatistics
    image_boundary: torch.Tensor
    features: torch.Tensor
    mode: MatchingEvidenceMode


def stereo_matching_feature_channels(mode: MatchingEvidenceMode) -> int:
    """Number of fixed-range maps appended for a representation mode."""
    return {"none": 0, "cost": 6, "shape": 23, "full": 37}[mode]


def _luminance(image: torch.Tensor) -> torch.Tensor:
    rgb = _as_unit_rgb(image)
    return .299 * rgb[:, 0:1] + .587 * rgb[:, 1:2] + .114 * rgb[:, 2:3]


@torch.no_grad()
def _candidate_cost_statistics(
    left_rgb: torch.Tensor,
    right_rgb: torch.Tensor,
    candidate: torch.Tensor,
    *,
    offsets: tuple[int, ...],
    census_kernel: int,
    census_threshold: float,
    cost_temperature: float,
) -> CandidateCostStatistics:
    """Evaluate a local census cost curve for one positive-left candidate."""
    if candidate.ndim != 4 or candidate.shape[1] != 1:
        raise ValueError("candidate disparity must be [B,1,H,W]")
    if cost_temperature <= 0:
        raise ValueError("cost_temperature must be positive")
    left_luminance = _luminance(left_rgb)
    curves: list[torch.Tensor] = []
    supports: list[torch.Tensor] = []
    for offset in offsets:
        reconstructed, right_support = warp_right_to_left(right_rgb, candidate + float(offset))
        cost, support = ternary_census_cost(
            left_luminance, _luminance(reconstructed), right_support,
            kernel=census_kernel, threshold=census_threshold,
        )
        curves.append(cost)
        supports.append(support)
    curve = torch.cat(curves, dim=1).clamp(0.0, 1.0)
    support = torch.cat(supports, dim=1).bool()
    support_count = support.sum(dim=1, keepdim=True)
    safe_cost = torch.where(support, curve, torch.full_like(curve, float("inf")))
    minimum_cost, min_index = safe_cost.min(dim=1, keepdim=True)
    any_support = support_count > 0
    minimum_cost = torch.where(any_support, minimum_cost, torch.ones_like(minimum_cost))
    offset_values = torch.tensor(offsets, dtype=curve.dtype, device=curve.device).view(1, len(offsets), 1, 1).expand(curve.shape[0], -1, curve.shape[2], curve.shape[3])
    offset_to_minimum = offset_values.gather(1, min_index).div(float(max(abs(x) for x in offsets)))
    offset_to_minimum = torch.where(any_support, offset_to_minimum, torch.zeros_like(offset_to_minimum))

    sorted_cost = safe_cost.sort(dim=1).values
    has_two = support_count >= 2
    margin = (sorted_cost[:, 1:2] - sorted_cost[:, 0:1]).clamp(0.0, 1.0)
    margin = torch.where(has_two, margin, torch.zeros_like(margin))

    center = offsets.index(0)
    candidate_cost = torch.where(support[:, center:center + 1], curve[:, center:center + 1], torch.ones_like(curve[:, center:center + 1]))
    # The offsets -1, 0, +1 are fixed entries of the declared curve.  A
    # positive value means the centre is a local minimum; normalise its
    # mathematical range [-2, 2] to [-1, 1].
    minus, plus = offsets.index(-1), offsets.index(1)
    curvature_valid = support[:, minus:minus + 1] & support[:, center:center + 1] & support[:, plus:plus + 1]
    curvature = ((curve[:, minus:minus + 1] + curve[:, plus:plus + 1] - 2.0 * curve[:, center:center + 1]) / 2.0).clamp(-1.0, 1.0)
    curvature = torch.where(curvature_valid, curvature, torch.zeros_like(curvature))

    logits = torch.where(support, -curve / cost_temperature, torch.full_like(curve, -float("inf")))
    probabilities = torch.softmax(logits, dim=1)
    log_probabilities = torch.where(probabilities > 0, probabilities.log(), torch.zeros_like(probabilities))
    entropy = -(probabilities * log_probabilities).sum(dim=1, keepdim=True)
    entropy_normalizer = support_count.float().clamp_min(2).log()
    sharpness = (1.0 - entropy / entropy_normalizer).clamp(0.0, 1.0)
    sharpness = torch.where(has_two, sharpness, torch.zeros_like(sharpness))
    return CandidateCostStatistics(
        curve=curve,
        curve_support=support,
        candidate_cost=candidate_cost,
        local_minimum_cost=minimum_cost,
        offset_to_local_minimum=offset_to_minimum,
        best_second_margin=margin,
        local_curvature=curvature,
        local_sharpness=sharpness,
        local_support_fraction=support.float().mean(dim=1, keepdim=True),
        candidate_support=support[:, center:center + 1],
    )


def _image_boundary(height: int, width: int, *, radius: int, device: torch.device) -> torch.Tensor:
    boundary = torch.zeros((1, 1, height, width), dtype=torch.bool, device=device)
    if radius:
        boundary[..., :radius, :] = True
        boundary[..., -radius:, :] = True
        boundary[..., :, :radius] = True
        boundary[..., :, -radius:] = True
    return boundary


def _shape_features(raw: CandidateCostStatistics, memory: CandidateCostStatistics) -> list[torch.Tensor]:
    # Costs and margins are [0,1]; offsets and curvature are already [-1,1];
    # sharpness/support are [0,1].  Differences therefore retain sign in a
    # bounded, backbone-independent range.
    raw_stats = (
        raw.local_minimum_cost, raw.offset_to_local_minimum,
        raw.best_second_margin, raw.local_curvature, raw.local_sharpness,
        raw.local_support_fraction,
    )
    memory_stats = (
        memory.local_minimum_cost, memory.offset_to_local_minimum,
        memory.best_second_margin, memory.local_curvature, memory.local_sharpness,
        memory.local_support_fraction,
    )
    differences = tuple((memory_stats[index] - raw_stats[index]).clamp(-1.0, 1.0) for index in range(5))
    return [*raw_stats, *memory_stats, *differences]


@torch.no_grad()
def candidate_conditioned_stereo_matching_evidence(
    left_rgb: torch.Tensor,
    right_rgb: torch.Tensor,
    raw_disparity: torch.Tensor,
    aligned_memory_disparity: torch.Tensor,
    *,
    mode: MatchingEvidenceMode = "full",
    offsets: tuple[int, ...] = OFFSETS,
    census_kernel: int = 5,
    census_threshold: float = .02,
    cost_temperature: float = .10,
) -> StereoMatchingEvidence:
    """Return compact local stereo evidence for raw and aligned-memory maps.

    The function is causal: both candidates are evaluated only against the
    current rectified stereo pair.  It uses the canonical `x_right = x_left-d`
    convention through :func:`warp_right_to_left`.  No result from this
    function changes the canonical causal support, target or metric mask.
    """
    if mode not in {"none", "cost", "shape", "full"}:
        raise ValueError(f"unknown stereo matching evidence mode: {mode!r}")
    if offsets != OFFSETS:
        # Keep the feature-channel contract immutable for controlled runs.
        raise ValueError(f"offsets must be the fixed declared set {OFFSETS}")
    if census_kernel < 3 or census_kernel % 2 == 0:
        raise ValueError("census_kernel must be odd and >=3")
    if left_rgb.shape != right_rgb.shape or left_rgb.ndim != 4 or left_rgb.shape[1] != 3:
        raise ValueError("left/right RGB must share [B,3,H,W]")
    if raw_disparity.shape != aligned_memory_disparity.shape or raw_disparity.shape[1] != 1:
        raise ValueError("raw and aligned-memory disparity must share [B,1,H,W]")
    if raw_disparity.shape[0] != left_rgb.shape[0] or raw_disparity.shape[-2:] != left_rgb.shape[-2:]:
        raise ValueError("RGB and disparity must share batch/spatial dimensions")

    raw = _candidate_cost_statistics(
        left_rgb, right_rgb, raw_disparity, offsets=offsets,
        census_kernel=census_kernel, census_threshold=census_threshold, cost_temperature=cost_temperature,
    )
    memory = _candidate_cost_statistics(
        left_rgb, right_rgb, aligned_memory_disparity, offsets=offsets,
        census_kernel=census_kernel, census_threshold=census_threshold, cost_temperature=cost_temperature,
    )
    batch, _, height, width = raw_disparity.shape
    boundary = _image_boundary(height, width, radius=census_kernel // 2, device=raw_disparity.device).expand(batch, -1, -1, -1)
    cost_features = [
        raw.candidate_cost, memory.candidate_cost,
        (memory.candidate_cost - raw.candidate_cost).clamp(-1.0, 1.0),
        raw.candidate_support.float(), memory.candidate_support.float(), boundary.float(),
    ]
    if mode == "none":
        features = raw_disparity.new_empty((batch, 0, height, width))
    elif mode == "cost":
        features = torch.cat(cost_features, dim=1)
    elif mode == "shape":
        features = torch.cat([*cost_features, *_shape_features(raw, memory)], dim=1)
    else:
        features = torch.cat([*cost_features, *_shape_features(raw, memory), raw.curve, memory.curve], dim=1)
    expected = stereo_matching_feature_channels(mode)
    if features.shape[1] != expected:
        raise AssertionError(f"feature contract violation: expected {expected}, got {features.shape[1]}")
    if not torch.isfinite(features).all():
        raise FloatingPointError("candidate-conditioned stereo evidence is non-finite")
    return StereoMatchingEvidence(raw=raw, memory=memory, image_boundary=boundary, features=features.detach(), mode=mode)


__all__ = [
    "CandidateCostStatistics", "MatchingEvidenceMode", "OFFSETS", "StereoMatchingEvidence",
    "candidate_conditioned_stereo_matching_evidence", "stereo_matching_feature_channels",
]
