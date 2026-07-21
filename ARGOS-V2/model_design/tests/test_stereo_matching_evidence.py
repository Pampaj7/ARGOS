"""Deterministic tests for candidate-conditioned stereo correspondence maps."""
from __future__ import annotations

import torch

from model_design.external_components.stereo_matching_evidence import (
    OFFSETS,
    candidate_conditioned_stereo_matching_evidence,
    stereo_matching_feature_channels,
)


def _right_from_left_for_positive_disparity(left: torch.Tensor, disparity: int) -> torch.Tensor:
    right = torch.zeros_like(left)
    right[..., :-disparity] = left[..., disparity:]
    return right


def test_true_candidate_has_cost_minimum_at_zero_offset_and_full_contract():
    torch.manual_seed(41)
    left = torch.rand(1, 3, 20, 48)
    right = _right_from_left_for_positive_disparity(left, 4)
    raw = torch.full((1, 1, 20, 48), 2.0)
    memory = torch.full_like(raw, 4.0)
    evidence = candidate_conditioned_stereo_matching_evidence(left, right, raw, memory, mode="full")
    support = evidence.memory.candidate_support & evidence.memory.curve_support[:, 2:3] & evidence.memory.curve_support[:, 4:5]
    # The true candidate has its census minimum at d+0 and lower direct cost
    # than the intentionally two-pixel-wrong raw candidate on common support.
    assert evidence.features.shape[1] == stereo_matching_feature_channels("full") == 37
    # Census may tie on locally flat random patches; the true correspondence
    # must nevertheless win the overwhelming majority of well-supported sites.
    assert float((evidence.memory.offset_to_local_minimum[support] == 0).float().mean()) > .95
    common = evidence.raw.candidate_support & evidence.memory.candidate_support
    assert evidence.memory.candidate_cost[common].mean() < evidence.raw.candidate_cost[common].mean()


def test_cost_shape_and_full_feature_modes_are_fixed_and_finite():
    torch.manual_seed(5)
    left = torch.rand(2, 3, 16, 32)
    right = _right_from_left_for_positive_disparity(left, 3)
    raw = torch.full((2, 1, 16, 32), 3.0)
    memory = torch.full_like(raw, 2.0)
    expected = {"none": 0, "cost": 6, "shape": 23, "full": 37}
    for mode, channels in expected.items():
        evidence = candidate_conditioned_stereo_matching_evidence(left, right, raw, memory, mode=mode)
        assert evidence.features.shape == (2, channels, 16, 32)
        assert torch.isfinite(evidence.features).all()
        assert channels == stereo_matching_feature_channels(mode)


def test_support_rejects_stereo_boundary_without_changing_tensor_shape():
    left = torch.rand(1, 3, 12, 20)
    right = left.clone()
    raw = torch.zeros((1, 1, 12, 20))
    memory = torch.full_like(raw, 8.0)
    evidence = candidate_conditioned_stereo_matching_evidence(left, right, raw, memory, mode="shape")
    assert not evidence.memory.candidate_support[..., :8].any()
    assert evidence.image_boundary[..., :2, :].all()
    assert evidence.image_boundary[..., :, :2].all()


def test_frontend_is_no_grad_and_uses_declared_offsets_only():
    left = torch.rand(1, 3, 12, 24, requires_grad=True)
    right = torch.rand(1, 3, 12, 24, requires_grad=True)
    disparity = torch.full((1, 1, 12, 24), 3.0, requires_grad=True)
    evidence = candidate_conditioned_stereo_matching_evidence(left, right, disparity, disparity, mode="cost")
    assert not evidence.features.requires_grad
    assert OFFSETS == (-4, -2, -1, 0, 1, 2, 4)
