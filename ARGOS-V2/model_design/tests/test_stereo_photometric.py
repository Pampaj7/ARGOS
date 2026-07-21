import torch

from model_design.external_components.stereo_photometric import (
    select_lower_stereo_cost,
    stereo_photometric_evidence,
    warp_right_to_left,
)


def _right_from_left_for_positive_disparity(left: torch.Tensor, disparity: int) -> torch.Tensor:
    """Synthetic rectified right image: right[x-d] equals left[x]."""
    right = torch.zeros_like(left)
    right[..., :-disparity] = left[..., disparity:]
    return right


def test_positive_left_disparity_samples_right_at_x_minus_d():
    left = torch.arange(8, dtype=torch.float32).view(1, 1, 1, 8).repeat(1, 3, 4, 1)
    right = _right_from_left_for_positive_disparity(left, 2)
    disparity = torch.full((1, 1, 4, 8), 2.0)
    sampled, support = warp_right_to_left(right, disparity)
    assert support[..., 2:].all()
    assert not support[..., :2].any()
    assert torch.allclose(sampled[..., 2:], left[..., 2:] / 255.0)


def test_true_disparity_has_lower_cost_than_wrong_disparity():
    torch.manual_seed(2)
    left = torch.rand(1, 3, 16, 32)
    right = _right_from_left_for_positive_disparity(left, 3)
    true = torch.full((1, 1, 16, 32), 3.0)
    wrong = torch.full((1, 1, 16, 32), 1.0)
    good = stereo_photometric_evidence(left, right, true, local_kernel=5)
    bad = stereo_photometric_evidence(left, right, wrong, local_kernel=5)
    common = good.right_support & bad.right_support
    assert good.local_rgb_l1[common].mean() < bad.local_rgb_l1[common].mean()
    census_common = good.census_support & bad.census_support
    assert good.ternary_census_cost[census_common].mean() < bad.ternary_census_cost[census_common].mean()


def test_ternary_census_is_offset_robust_and_rejects_patch_edges():
    torch.manual_seed(7)
    left = torch.rand(1, 3, 16, 32)
    right = _right_from_left_for_positive_disparity(left, 3)
    # A global exposure offset changes L1 but preserves local ternary ordering.
    right = (right + .15).clamp(0, 1)
    true = torch.full((1, 1, 16, 32), 3.0)
    evidence = stereo_photometric_evidence(left, right, true, local_kernel=5, census_kernel=5)
    # Three pixels of stereo displacement plus two census pixels are invalid.
    assert not evidence.census_support[..., :5].any()
    interior = evidence.census_support
    assert evidence.ternary_census_cost[interior].mean() < .12


def test_cost_selector_respects_margin_and_validity():
    raw = torch.tensor([[[[0.5, 0.5]]]])
    memory = torch.tensor([[[[0.4, 0.49]]]])
    valid = torch.tensor([[[[True, False]]]])
    assert select_lower_stereo_cost(raw, memory, valid, minimum_improvement=0.05).tolist() == [[[[True, False]]]]
    assert not select_lower_stereo_cost(raw, memory, valid, minimum_improvement=0.11).any()
