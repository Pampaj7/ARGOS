import torch

from model_design.external_components.stereo_lr_consistency import (
    flip_swap_stereo_pair,
    left_right_consistency,
    unflip_right_reference_disparity,
)


def test_flip_swap_and_unflip_preserve_positive_disparity_geometry():
    left = torch.arange(2 * 3 * 4 * 7, dtype=torch.float32).reshape(2, 3, 4, 7)
    right = left + 1000
    swapped_left, swapped_right = flip_swap_stereo_pair(left, right)
    assert torch.equal(swapped_left, right.flip(-1))
    assert torch.equal(swapped_right, left.flip(-1))
    disp = torch.arange(28, dtype=torch.float32).reshape(1, 1, 4, 7)
    assert torch.equal(unflip_right_reference_disparity(disp), disp.flip(-1))


def test_left_right_consistency_zero_for_constant_consistent_disparity():
    left = torch.full((1, 1, 3, 9), 2.0)
    right = torch.full_like(left, 2.0)
    evidence = left_right_consistency(left, right)
    # x<2 is out of support, all other pixels sample the same d_R=2.
    assert not evidence.right_support[..., :2].any()
    assert evidence.valid[..., 2:].all()
    assert torch.allclose(evidence.residual[evidence.valid], torch.zeros_like(evidence.residual[evidence.valid]))


def test_lrc_residual_and_validity_are_sampled_at_x_minus_left_disparity():
    left = torch.full((1, 1, 1, 7), 2.0)
    right = torch.tensor([[[[0.0, 0.0, 1.5, 2.0, 3.0, 4.0, 5.0]]]])
    right_valid = torch.tensor([[[[0, 0, 1, 1, 0, 1, 1]]]], dtype=torch.bool)
    evidence = left_right_consistency(left, right, right_valid=right_valid)
    # At x=4 the sample is right[2]=1.5 -> abs(2-1.5)=.5 and is valid.
    assert torch.isclose(evidence.right_disparity_sampled[0, 0, 0, 4], torch.tensor(1.5))
    assert torch.isclose(evidence.residual[0, 0, 0, 4], torch.tensor(.5))
    assert evidence.valid[0, 0, 0, 4]
    # At x=6 samples right[4], which is explicitly invalid.
    assert not evidence.sampled_right_valid[0, 0, 0, 6]
    assert not evidence.valid[0, 0, 0, 6]
