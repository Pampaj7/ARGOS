"""Post-hoc hybrid temporal-memory candidates and oracles for ARGOS v2.

This module contains no learned memory mechanism.  It only formalises candidate
provenance, common-support contracts, and diagnostic GT oracles.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from model_design.models.codd_style_fusion import convex_fusion_oracle


@dataclass(frozen=True)
class TemporalCandidate:
    name: str
    disparity: torch.Tensor
    valid: torch.Tensor
    warp_support: torch.Tensor
    fb_confidence: torch.Tensor
    age: int
    provenance: str
    source_frame_id: str
    source_backbone: str
    sequence: str

    @property
    def available(self) -> torch.Tensor:
        return self.valid.bool() & self.warp_support.bool()


@dataclass(frozen=True)
class OracleResult:
    error: torch.Tensor
    winner: torch.Tensor
    support: torch.Tensor
    names: tuple[str, ...]


def strict_intersection_support(base: torch.Tensor, candidates: Sequence[TemporalCandidate]) -> torch.Tensor:
    """Require every candidate in a comparison to be available."""
    support = base.bool().clone()
    for candidate in candidates:
        support &= candidate.available
    return support


def availability_aware_oracle(
    raw: torch.Tensor,
    ground_truth: torch.Tensor,
    base: torch.Tensor,
    candidates: Sequence[TemporalCandidate],
) -> OracleResult:
    """Raw is the fallback; temporal candidates compete only where available."""
    errors = [(raw - ground_truth).abs()]
    names = ["C0"]
    for candidate in candidates:
        error = (candidate.disparity - ground_truth).abs()
        errors.append(torch.where(candidate.available & base, error, torch.full_like(error, torch.inf)))
        names.append(candidate.name)
    stacked = torch.stack(errors, dim=0)
    minimum, winner = stacked.min(dim=0)
    return OracleResult(minimum, winner, base.bool(), tuple(names))


def strict_selection_oracle(
    raw: torch.Tensor,
    ground_truth: torch.Tensor,
    base: torch.Tensor,
    candidates: Sequence[TemporalCandidate],
) -> OracleResult:
    support = strict_intersection_support(base, candidates)
    errors = [(raw - ground_truth).abs()] + [(candidate.disparity - ground_truth).abs() for candidate in candidates]
    minimum, winner = torch.stack(errors, dim=0).min(dim=0)
    return OracleResult(minimum, winner, support, tuple(["C0"] + [candidate.name for candidate in candidates]))


def best_convex_oracle(
    raw: torch.Tensor,
    ground_truth: torch.Tensor,
    base: torch.Tensor,
    candidates: Sequence[TemporalCandidate],
    *,
    strict: bool,
) -> OracleResult:
    """Best independent raw-to-candidate convex segment; never averages the bank."""
    errors = [(raw - ground_truth).abs()]
    names = ["C0"]
    for candidate in candidates:
        fused, _weight = convex_fusion_oracle(raw, candidate.disparity, ground_truth)
        error = (fused - ground_truth).abs()
        if not strict:
            error = torch.where(candidate.available & base, error, torch.full_like(error, torch.inf))
        errors.append(error)
        names.append(candidate.name)
    minimum, winner = torch.stack(errors, dim=0).min(dim=0)
    support = strict_intersection_support(base, candidates) if strict else base.bool()
    return OracleResult(minimum, winner, support, tuple(names))


def progressive_oracles(
    raw: torch.Tensor,
    ground_truth: torch.Tensor,
    base: torch.Tensor,
    candidates: Mapping[str, TemporalCandidate],
    order: Sequence[str],
    *,
    strict: bool,
) -> list[tuple[tuple[str, ...], OracleResult]]:
    selected: list[TemporalCandidate] = []
    outputs = []
    for name in order:
        selected.append(candidates[name])
        oracle = (strict_selection_oracle if strict else availability_aware_oracle)(
            raw, ground_truth, base, selected
        )
        outputs.append((tuple(candidate.name for candidate in selected), oracle))
    return outputs


def temporal_median_mad(
    candidates: Sequence[TemporalCandidate],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Masked raw-anchor median, MAD and witness count."""
    if not candidates:
        raise ValueError("at least one candidate is required")
    values = torch.stack([candidate.disparity for candidate in candidates], dim=0)
    valid = torch.stack([candidate.available for candidate in candidates], dim=0)
    masked = torch.where(valid, values, torch.full_like(values, torch.nan))
    median = torch.nanmedian(masked, dim=0).values
    deviations = (values - median.unsqueeze(0)).abs()
    mad = torch.nanmedian(torch.where(valid, deviations, torch.full_like(deviations, torch.nan)), dim=0).values
    count = valid.sum(dim=0)
    return torch.nan_to_num(median), torch.nan_to_num(mad), count


__all__ = [
    "OracleResult", "TemporalCandidate", "availability_aware_oracle", "best_convex_oracle",
    "progressive_oracles", "strict_intersection_support", "strict_selection_oracle",
    "temporal_median_mad",
]
