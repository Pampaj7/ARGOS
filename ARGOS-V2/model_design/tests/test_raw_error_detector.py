from __future__ import annotations

import torch

from model_design.data.raw_error_dataset import RawErrorTargets
from model_design.losses.raw_error_losses import RawErrorLossConfig, raw_error_losses
from model_design.models.learned_t1_refiner import LearnedT1Refiner
from model_design.models.raw_error_detector import (
    FEATURE_CHANNELS,
    RawErrorDetector,
    RawErrorEvidence,
)


def evidence(batch: int = 2, height: int = 12, width: int = 16) -> RawErrorEvidence:
    one = torch.ones(batch, 1, height, width)
    raw = torch.rand_like(one) * 10
    return RawErrorEvidence(
        raw=raw,
        raw_valid=one.bool(),
        aligned=raw + torch.randn_like(raw),
        aligned_valid=one.bool(),
        warp_support=one.bool(),
        forward_backward_error=torch.rand_like(one),
        forward_backward_confidence=torch.rand_like(one),
        photometric_residual=torch.rand_like(one),
        flow_magnitude=torch.rand_like(one) * 4,
        a2_update=torch.rand_like(one) - 0.5,
        a2_error_gate=torch.rand_like(one),
        a2_memory_gate=torch.rand_like(one),
    )


def targets(shape: tuple[int, ...]) -> RawErrorTargets:
    error = torch.rand(shape) * 3
    valid = torch.ones(shape, dtype=torch.bool)
    return RawErrorTargets(
        error=error,
        label=(error > 0.5).float(),
        regression_valid=valid,
        classification_valid=valid,
        clean=error <= 0.5,
    )


def test_all_architectures_have_finite_positive_outputs_and_no_identity_input() -> None:
    torch.manual_seed(3)
    item = evidence()
    for architecture in ("s1", "s2", "s3", "s4"):
        model = RawErrorDetector(architecture, channels=8)
        output = model(item)
        assert model.normalized_inputs(item).shape[1] == FEATURE_CHANNELS
        assert output.mu.shape == item.raw.shape
        assert torch.isfinite(output.probability).all()
        assert torch.isfinite(output.mu).all() and (output.mu > 0).all()
        assert torch.isfinite(output.sigma).all() and (output.sigma > 0).all()
        assert "backbone" not in RawErrorEvidence.__dataclass_fields__


def test_shared_detector_gradients_do_not_enter_cached_evidence() -> None:
    model = RawErrorDetector("s2", channels=8)
    item = evidence()
    output = model(item)
    loss = raw_error_losses(output, targets(output.mu.shape), RawErrorLossConfig()).get("total")
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    for value in item.__dict__.values():
        assert value.grad is None


def test_invalid_pixels_do_not_affect_loss_and_uncertainty_is_bounded() -> None:
    model = RawErrorDetector("s2", channels=8)
    output = model(evidence(batch=1))
    target = targets(output.mu.shape)
    target.regression_valid[..., 0, 0] = False
    target.classification_valid[..., 0, 0] = False
    first = raw_error_losses(output, target, RawErrorLossConfig())
    target.error[..., 0, 0] = 1e6
    target.label[..., 0, 0] = 1 - target.label[..., 0, 0]
    second = raw_error_losses(output, target, RawErrorLossConfig())
    torch.testing.assert_close(first["total"], second["total"])

    # Very large predicted sigma is finite but explicitly penalized.
    with torch.no_grad():
        model.head_sigma.bias.fill_(100)
    large = raw_error_losses(model(evidence(batch=1)), targets(output.mu.shape), RawErrorLossConfig())
    assert torch.isfinite(large["total"])
    assert large["sigma_penalty"] > 0


def test_validated_a2_can_be_frozen() -> None:
    proposal = LearnedT1Refiner("A2")
    proposal.requires_grad_(False).eval()
    assert not proposal.training
    assert all(not parameter.requires_grad for parameter in proposal.parameters())

