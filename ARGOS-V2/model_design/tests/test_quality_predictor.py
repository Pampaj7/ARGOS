from __future__ import annotations

import inspect

import pytest
import torch

from model_design.data.quality_prediction_dataset import (
    CANDIDATE_AGES,
    QualityCandidateBatch,
)
from model_design.losses.quality_losses import (
    QualityLossConfig,
    hard_negative_weights,
    laplace_uncertainty_loss,
    pairwise_ranking_loss,
    quality_prediction_losses,
)
from model_design.models.quality_predictor import ARCHITECTURES, FEATURE_CHANNELS, QualityPredictor


def candidates(*, invalid_last: bool = False) -> QualityCandidateBatch:
    b, k, h, w = 2, 5, 16, 20
    disparity = torch.rand(b, k, 1, h, w) * 8
    valid = torch.ones_like(disparity, dtype=torch.bool)
    if invalid_last:
        valid[:, -1] = False
    raw_error = (disparity[:, :1] - 3.0).abs()
    error = (disparity - 3.0).abs()
    one = torch.ones_like(disparity)
    zero = torch.zeros_like(disparity)
    return QualityCandidateBatch(
        disparity=disparity,
        candidate_valid=valid,
        warp_support=valid,
        forward_backward_error=zero,
        forward_backward_confidence=one,
        photometric_residual=zero,
        flow_magnitude=zero,
        ages=torch.tensor(CANDIDATE_AGES),
        consensus_median=disparity[:, 1:].median(dim=1).values,
        consensus_mad=torch.ones(b, 1, h, w) * 0.2,
        witness_count=torch.ones(b, 1, h, w) * 4,
        target_error=error,
        target_advantage=raw_error - error,
        target_valid=valid,
        failure_masks={},
    )


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_positive_mu_sigma_and_all_candidate_shapes(architecture: str) -> None:
    batch = candidates()
    model = QualityPredictor(architecture, channels=8)
    output = model(batch)
    assert output.mu.shape == output.sigma.shape == output.advantage.shape == (2, 5, 16, 20)
    assert torch.all(output.mu > 0) and torch.all(output.sigma > 0)
    assert model.normalized_inputs(batch).shape == (2, 5, FEATURE_CHANNELS, 16, 20)


def test_shared_candidate_encoder_is_one_module() -> None:
    model = QualityPredictor("q0_4", channels=8)
    assert not isinstance(model.shared_encoder, torch.nn.ModuleList)
    assert len(model.mu_heads) == 5
    encoder_ids = {id(parameter) for parameter in model.shared_encoder.parameters()}
    assert encoder_ids and not any(id(p) in encoder_ids for head in model.mu_heads for p in head.parameters())


def test_model_contract_has_no_backbone_rgb_or_future_input() -> None:
    names = set(inspect.signature(QualityPredictor.forward).parameters)
    assert names == {"self", "candidates"}
    source = inspect.getsource(QualityPredictor.normalized_inputs)
    assert "backbone" not in source and "rgb" not in source and "future" not in source


def test_invalid_candidate_does_not_change_regression_loss() -> None:
    batch = candidates(invalid_last=True)
    model = QualityPredictor("q0_2", channels=8)
    output = model(batch)
    first = quality_prediction_losses(output, batch, QualityLossConfig(target_mode="absolute"))["total"]
    changed = QualityCandidateBatch(**{
        **batch.__dict__, "target_error": batch.target_error.clone()
    })
    changed.target_error[:, -1].fill_(10000)
    second = quality_prediction_losses(output, changed, QualityLossConfig(target_mode="absolute"))["total"]
    torch.testing.assert_close(first, second)


def test_target_absolute_error_and_advantage_are_numerically_used() -> None:
    batch = candidates()
    expected = (batch.disparity - 3.0).abs()
    torch.testing.assert_close(batch.target_error, expected)
    torch.testing.assert_close(batch.target_advantage, expected[:, :1] - expected)


def test_ties_are_excluded_from_ranking() -> None:
    predicted = torch.zeros(1, 2, 2, 2)
    target = torch.ones_like(predicted)
    valid = torch.ones_like(predicted, dtype=torch.bool)
    loss, count = pairwise_ranking_loss(
        predicted, target, valid, indifference_margin_px=0.10, rank_margin_px=0.05
    )
    assert count == 0 and loss == 0


def test_ranking_loss_rewards_correct_order() -> None:
    target = torch.tensor([[[[0.0]], [[2.0]]]])
    valid = torch.ones_like(target, dtype=torch.bool)
    correct = torch.tensor([[[[0.1]], [[1.5]]]])
    wrong = correct.flip(1)
    good, _ = pairwise_ranking_loss(correct, target, valid, indifference_margin_px=0.1, rank_margin_px=0.1)
    bad, _ = pairwise_ranking_loss(wrong, target, valid, indifference_margin_px=0.1, rank_margin_px=0.1)
    assert good < bad


def test_uncertainty_loss_finite_and_inflation_penalized() -> None:
    target = torch.ones(1, 1, 2, 2) * 2
    mu = torch.ones_like(target)
    valid = torch.ones_like(target, dtype=torch.bool)
    normal, _, _ = laplace_uncertainty_loss(mu, target, torch.ones_like(target), valid)
    inflated, _, penalty = laplace_uncertainty_loss(mu, target, torch.ones_like(target) * 100, valid)
    assert torch.isfinite(normal) and torch.isfinite(inflated)
    assert inflated > normal and penalty > 0


def test_real_hard_negative_masks_receive_bounded_extra_weight() -> None:
    batch = candidates()
    hard = torch.zeros(2, 1, 16, 20, dtype=torch.bool)
    hard[:, :, :4, :4] = True
    batch = QualityCandidateBatch(**{
        **batch.__dict__, "failure_masks": {"minority_correct": hard}
    })
    valid = batch.target_valid[:, :, 0]
    weights = hard_negative_weights(
        batch, valid, indifference_margin_px=0.1, boost=1.0
    )
    assert weights.shape == valid.shape and torch.isfinite(weights).all()
    assert weights[:, :, :4, :4].mean() > weights[:, :, 8:, 8:].mean()
    assert float(weights.max()) <= 6.0


@pytest.mark.parametrize("patch_size", [1, 8, 16])
def test_losses_are_finite_for_pixel_and_regional_targets(patch_size: int) -> None:
    batch = candidates()
    output = QualityPredictor("q0_5", channels=8)(batch)
    losses = quality_prediction_losses(
        output, batch, QualityLossConfig(target_mode="uncertainty", patch_size=patch_size, ranking_weight=0.1)
    )
    assert all(torch.isfinite(value) for value in losses.values())


def test_gradients_reach_model_not_frozen_candidate_tensors() -> None:
    batch = candidates()
    model = QualityPredictor("q0_5", channels=8)
    output = model(batch)
    loss = quality_prediction_losses(output, batch, QualityLossConfig(target_mode="uncertainty"))["total"]
    loss.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.parameters() if parameter.requires_grad)
    assert batch.disparity.grad is None and batch.target_error.grad is None


def test_deterministic_validation_forward() -> None:
    torch.manual_seed(9)
    batch = candidates()
    model = QualityPredictor("q0_3", channels=8).eval()
    with torch.no_grad():
        first = model(batch); second = model(batch)
    torch.testing.assert_close(first.mu, second.mu, rtol=0, atol=0)
    torch.testing.assert_close(first.sigma, second.sigma, rtol=0, atol=0)
