from __future__ import annotations

import torch

from model_design.external_components.hybrid_temporal_memory import (
    TemporalCandidate,
    availability_aware_oracle,
    best_convex_oracle,
    strict_intersection_support,
    temporal_median_mad,
)


def candidate(name: str, value: float, valid: list[bool], age: int = 1, provenance: str = "raw") -> TemporalCandidate:
    disparity = torch.tensor(valid, dtype=torch.float32).view(1, 1, 1, -1) * 0 + value
    mask = torch.tensor(valid).view(1, 1, 1, -1)
    return TemporalCandidate(name, disparity, mask, mask, torch.ones_like(disparity), age, provenance, "past", "B", "S")


def test_availability_oracle_has_exact_raw_fallback() -> None:
    raw = torch.tensor([[[[2.0, 2.0]]]])
    gt = torch.tensor([[[[1.0, 1.0]]]])
    base = torch.ones_like(raw, dtype=torch.bool)
    memory = candidate("CS1", 1.0, [True, False])
    result = availability_aware_oracle(raw, gt, base, [memory])
    assert torch.equal(result.error, torch.tensor([[[[0.0, 1.0]]]]))
    assert torch.equal(result.winner, torch.tensor([[[[1, 0]]]]))


def test_strict_support_is_candidate_intersection() -> None:
    base = torch.ones(1, 1, 1, 3, dtype=torch.bool)
    first = candidate("CS1", 1, [True, True, False])
    second = candidate("CS2", 1, [True, False, True], age=2)
    assert torch.equal(strict_intersection_support(base, [first, second]), torch.tensor([[[[True, False, False]]]]))


def test_convex_oracle_can_beat_both_endpoints() -> None:
    raw = torch.tensor([[[[0.0]]]])
    gt = torch.tensor([[[[1.0]]]])
    base = torch.ones_like(raw, dtype=torch.bool)
    memory = candidate("CS1", 2.0, [True])
    result = best_convex_oracle(raw, gt, base, [memory], strict=False)
    assert result.error.item() == 0.0
    assert result.winner.item() == 1


def test_temporal_median_mad_respects_validity() -> None:
    values = [candidate("CS1", 1.0, [True]), candidate("CS2", 3.0, [True]), candidate("CS4", 100.0, [False])]
    median, mad, count = temporal_median_mad(values)
    assert median.item() == 1.0  # torch.nanmedian uses the lower middle value
    assert mad.item() == 0.0
    assert count.item() == 2
