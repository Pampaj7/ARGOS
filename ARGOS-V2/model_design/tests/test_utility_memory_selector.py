"""Deterministic contracts for the ARGOS v2 raw-versus-memory selector."""
from __future__ import annotations

import torch

from model_design.data.utility_memory_selector_dataset import UtilityTargets
from model_design.losses.utility_memory_selector_losses import UtilitySelectorLossConfig, utility_selector_losses
from model_design.models.utility_memory_selector import (
    INPUT_CHANNELS, STEREO_PHOTOMETRIC_INPUT_CHANNELS, UtilityMemorySelector, UtilitySelectorEvidence, UtilitySelectorOutput,
    memory_authorization, utility_risk_authorization, select_raw_or_memory,
)
from model_design.external_components.stereo_matching_evidence import stereo_matching_feature_channels


def evidence(batch: int = 2, h: int = 8, w: int = 10) -> UtilitySelectorEvidence:
    raw = torch.rand(batch, 1, h, w) * 10
    return UtilitySelectorEvidence(
        raw=raw, aligned_memory=raw + .1, flow=torch.randn(batch, 2, h, w),
        flow_magnitude=torch.ones(batch, 1, h, w), forward_backward_confidence=torch.ones(batch, 1, h, w),
        warp_support=torch.ones(batch, 1, h, w, dtype=torch.bool), aligned_valid=torch.ones(batch, 1, h, w, dtype=torch.bool),
        raw_valid=torch.ones(batch, 1, h, w, dtype=torch.bool),
    )


def test_input_contract_output_shapes_and_positive_heads():
    model = UtilityMemorySelector()
    out = model(evidence())
    assert model.normalized_inputs(evidence()).shape[1] == INPUT_CHANNELS
    assert out.memory_better_probability.shape == (2, 1, 8, 10)
    assert bool((out.expected_positive_gain >= 0).all())
    assert bool((out.expected_harmful_magnitude >= 0).all())
    assert 100_000 <= sum(p.numel() for p in model.parameters()) <= 1_000_000


def test_stereo_photometric_contract_requires_and_consumes_only_universal_maps():
    ev = evidence()
    maps = {name: torch.rand_like(ev.raw) for name in (
        "raw_stereo_l1", "memory_stereo_l1", "raw_stereo_zncc", "memory_stereo_zncc",
    )}
    ev = UtilitySelectorEvidence(**{**ev.__dict__, **maps, "stereo_common_support": torch.ones_like(ev.raw, dtype=torch.bool)})
    model = UtilityMemorySelector(include_stereo_photometric=True)
    assert model.normalized_inputs(ev).shape[1] == STEREO_PHOTOMETRIC_INPUT_CHANNELS
    assert model(ev).memory_better_probability.shape == ev.raw.shape


def test_candidate_conditioned_stereo_contract_appends_input_only_maps():
    ev = evidence()
    channels = stereo_matching_feature_channels("full")
    maps = torch.rand(ev.raw.shape[0], channels, *ev.raw.shape[-2:])
    augmented = UtilitySelectorEvidence(**{**ev.__dict__, "stereo_matching_features": maps})
    model = UtilityMemorySelector(stereo_matching_feature_channels=channels)
    assert model.normalized_inputs(augmented).shape[1] == INPUT_CHANNELS + channels
    assert model(augmented).memory_better_probability.shape == ev.raw.shape
    try:
        model(ev)
    except ValueError as error:
        assert "candidate-conditioned" in str(error)
    else:
        raise AssertionError("missing matching evidence must fail closed")


def test_authorization_is_monotone_and_rejection_is_bit_exact():
    ev = evidence(1); model = UtilityMemorySelector(channels=32, blocks=1); out = model(ev)
    tight = memory_authorization(out, ev, probability_threshold=.99, utility_threshold_px=10., harm_threshold_px=0.)
    loose = memory_authorization(out, ev, probability_threshold=0., utility_threshold_px=-1., harm_threshold_px=10.)
    assert bool((tight <= loose).all())
    raw = ev.raw.clone(); selected = select_raw_or_memory(raw, ev.aligned_memory, tight)
    assert torch.equal(selected[~tight], raw[~tight])
    assert torch.equal(select_raw_or_memory(raw, ev.aligned_memory, loose)[loose], ev.aligned_memory[loose])
    unsupported = UtilitySelectorEvidence(**{**ev.__dict__, "stereo_common_support": torch.zeros_like(ev.raw_valid)})
    assert not memory_authorization(out, unsupported, probability_threshold=0., utility_threshold_px=-1., harm_threshold_px=10.).any()


def test_loss_excludes_invalid_and_backpropagates():
    ev = evidence(1); model = UtilityMemorySelector(channels=32, blocks=1); out = model(ev)
    target = UtilityTargets(
        utility=torch.ones_like(ev.raw), supervision_utility=torch.ones_like(ev.raw), raw_error=torch.ones_like(ev.raw), memory_error=torch.zeros_like(ev.raw),
        valid=torch.zeros_like(ev.raw, dtype=torch.bool), memory_better=torch.zeros_like(ev.raw, dtype=torch.bool),
        helpful_gain=torch.zeros_like(ev.raw), harmful_magnitude=torch.zeros_like(ev.raw),
    )
    losses = utility_selector_losses(out, target, __import__("model_design.losses.utility_memory_selector_losses", fromlist=["UtilitySelectorLossConfig"]).UtilitySelectorLossConfig())
    losses["total"].backward()
    assert float(losses["total"]) == 0.0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_conditional_expected_utility_matches_manual_decision_risk():
    logit = torch.tensor([[[[0.0]]]])
    output = UtilitySelectorOutput(
        memory_better_logit=logit,
        memory_better_probability=torch.sigmoid(logit),
        expected_positive_gain=torch.tensor([[[[2.0]]]]),
        expected_harmful_magnitude=torch.tensor([[[[1.0]]]]),
    )
    assert torch.allclose(output.conditional_expected_utility, torch.tensor([[[[0.5]]]]))
    assert torch.allclose(output.conditional_harm_risk, torch.tensor([[[[0.5]]]]))


def test_utility_risk_loss_balances_helpful_and_harmful_actions():
    logits = torch.zeros(1, 1, 1, 3, requires_grad=True)
    output = UtilitySelectorOutput(
        memory_better_logit=logits,
        memory_better_probability=torch.sigmoid(logits),
        expected_positive_gain=torch.ones_like(logits, requires_grad=True),
        expected_harmful_magnitude=torch.ones_like(logits, requires_grad=True),
    )
    utility = torch.tensor([[[[1.0, -1.0, 0.0]]]])
    target = UtilityTargets(
        utility=utility, supervision_utility=utility, raw_error=torch.ones_like(utility), memory_error=torch.ones_like(utility),
        valid=torch.ones_like(utility, dtype=torch.bool),
        memory_better=utility > 0.1,
        helpful_gain=torch.relu(utility - 0.1),
        harmful_magnitude=torch.relu(-utility - 0.1),
    )
    losses = utility_selector_losses(output, target, UtilitySelectorLossConfig(objective="utility_risk"))
    losses["total"].backward()
    assert torch.isfinite(losses["total"])
    assert "signed_utility" in losses and "policy_risk" in losses
    # Helpful probability must be pushed up, harmful probability down; the
    # indifferent pixel is absent from the classification term.
    assert logits.grad[0, 0, 0, 0] < 0
    assert logits.grad[0, 0, 0, 1] > 0


def test_utility_risk_authorization_is_single_score_monotone_and_supported():
    ev=evidence(1,h=1,w=2)
    probability=torch.tensor([[[[.8,.2]]]])
    output=UtilitySelectorOutput(
        memory_better_logit=torch.logit(probability),memory_better_probability=probability,
        expected_positive_gain=torch.tensor([[[[1.,1.]]]]),
        expected_harmful_magnitude=torch.tensor([[[[.1,.1]]]]),
    )
    loose=utility_risk_authorization(output,ev,utility_threshold_px=-1.)
    tight=utility_risk_authorization(output,ev,utility_threshold_px=.5)
    assert bool((tight <= loose).all()) and tight[0,0,0,0] and not tight[0,0,0,1]
    unsupported=UtilitySelectorEvidence(**{**ev.__dict__,"warp_support":torch.zeros_like(ev.warp_support)})
    assert not utility_risk_authorization(output,unsupported,utility_threshold_px=-1.).any()


def test_utility_weighted_bce_assigns_more_cost_to_large_harm_than_small_gain():
    logits = torch.zeros(1, 1, 1, 3, requires_grad=True)
    output = UtilitySelectorOutput(
        memory_better_logit=logits, memory_better_probability=torch.sigmoid(logits),
        expected_positive_gain=torch.ones_like(logits), expected_harmful_magnitude=torch.ones_like(logits),
    )
    utility = torch.tensor([[[[1.0, -3.0, 0.0]]]])
    target = UtilityTargets(
        utility=utility, supervision_utility=utility, raw_error=torch.ones_like(utility), memory_error=torch.ones_like(utility),
        valid=torch.ones_like(utility, dtype=torch.bool), memory_better=utility > .1,
        helpful_gain=torch.relu(utility-.1), harmful_magnitude=torch.relu(-utility-.1),
    )
    loss = utility_selector_losses(output, target, UtilitySelectorLossConfig(objective="utility_weighted"))
    expected_weight = ((1.9) + (1 + 4 * 2.9) + .25) / 3
    assert torch.allclose(loss["classification"], torch.tensor(expected_weight * 0.6931471805599453), atol=1e-5)
    loss["total"].backward()
    assert torch.isfinite(logits.grad).all()


def test_selective_utility_loss_is_finite_and_enforces_minimum_soft_coverage():
    logits = torch.full((1, 1, 1, 2), -8.0, requires_grad=True)
    output = UtilitySelectorOutput(
        memory_better_logit=logits, memory_better_probability=torch.sigmoid(logits),
        expected_positive_gain=torch.ones_like(logits), expected_harmful_magnitude=torch.ones_like(logits),
    )
    utility = torch.tensor([[[[1.0, -1.0]]]])
    target = UtilityTargets(
        utility=utility, supervision_utility=utility, raw_error=torch.ones_like(utility), memory_error=torch.ones_like(utility),
        valid=torch.ones_like(utility, dtype=torch.bool), memory_better=utility > .1,
        helpful_gain=torch.relu(utility-.1), harmful_magnitude=torch.relu(-utility-.1),
    )
    loss = utility_selector_losses(output, target, UtilitySelectorLossConfig(objective="selective_utility", selective_target_coverage=.02))
    assert float(loss["selective_coverage"]) < .02
    assert float(loss["selective_coverage_penalty"]) > 0
    loss["total"].backward()
    assert torch.isfinite(logits.grad).all()
