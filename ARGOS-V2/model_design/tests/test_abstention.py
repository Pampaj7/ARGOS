from __future__ import annotations

import pytest
import torch

from model_design.models.abstention import (
    OperatingMode,
    authorization_mask,
    authorized_update,
    fit_temperature,
)
from model_design.models.raw_error_detector import RawErrorOutput


def output(probability: torch.Tensor, mu: float = 2.0, sigma: float = 0.2) -> RawErrorOutput:
    logits = torch.logit(probability.clamp(1e-5, 1 - 1e-5))
    return RawErrorOutput(
        probability=probability,
        mu=torch.full_like(probability, mu),
        sigma=torch.full_like(probability, sigma),
        logits=logits,
        raw_mu=torch.zeros_like(probability),
        raw_sigma=torch.zeros_like(probability),
    )


def test_abstention_is_bit_exact_identity_and_update_is_bounded() -> None:
    raw = torch.tensor([[[[1.0, 2.0, 3.0]]]])
    proposal = torch.tensor([[[[1.0, -1.0, 3.1]]]])
    valid = torch.ones_like(raw, dtype=torch.bool)
    mode = OperatingMode("safe", 0.8, 1.0, 0.5, 3.0)
    authorization = authorization_mask(
        output(torch.tensor([[[[0.9, 0.1, 0.99]]]])),
        mode=mode,
        temperature=1.0,
        aligned_valid=valid,
        warp_support=valid,
        proposal_update=proposal,
    )
    assert authorization.tolist() == [[[[True, False, False]]]]
    refined = authorized_update(raw, proposal, authorization)
    assert refined[0, 0, 0, 0] == 2
    assert torch.equal(refined[~authorization], raw[~authorization])


def test_probability_threshold_monotonically_reduces_coverage() -> None:
    probability = torch.linspace(0.05, 0.95, 19).reshape(1, 1, 1, -1)
    valid = torch.ones_like(probability, dtype=torch.bool)
    proposal = torch.ones_like(probability)
    coverages = []
    for threshold in (0.2, 0.5, 0.8):
        mode = OperatingMode("test", threshold, 0.0, 1.0)
        mask = authorization_mask(
            output(probability), mode=mode, temperature=1.0,
            aligned_valid=valid, warp_support=valid, proposal_update=proposal,
        )
        coverages.append(float(mask.float().mean()))
    assert coverages[0] >= coverages[1] >= coverages[2]


def test_calibration_is_validation_only_and_deterministic() -> None:
    logits = torch.tensor([-2.0, -0.5, 0.5, 2.0])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    valid = torch.ones(4, dtype=torch.bool)
    with pytest.raises(ValueError, match="validation-only"):
        fit_temperature(logits, labels, valid, split="test")
    first = fit_temperature(logits, labels, valid, split="validation")
    second = fit_temperature(logits, labels, valid, split="validation")
    assert first == pytest.approx(second)
    assert first > 0

