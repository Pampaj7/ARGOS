from __future__ import annotations

import torch

from model_design.models.raw_multi_anchor_refiner import MultiAnchorEvidence, MultiAnchorOutput
from model_design.models.raw_multi_anchor_selective_gate import apply_veto, frozen_proposal
from model_design.models.spatial_error_critic import (
    FAMILIES, LOG_SCALE_RANGE, CriticExtras, SpatialErrorCritic, build_critic_features,
    critic_losses, feature_channels, plane_sweep_features, selective_accept, stereo_residual,
)


def example(height=16, width=20):
    torch.manual_seed(0)
    raw = torch.rand(2, 1, height, width) * 10 + 2
    candidates = torch.stack([raw[:, 0] + value for value in (.4, -.6, 1.0, -1.2)], dim=1)
    valid = torch.ones_like(candidates, dtype=torch.bool)
    evidence = MultiAnchorEvidence(raw, candidates, valid, valid.clone(),
                                   torch.full_like(candidates, .8), torch.tensor([1, 2, 4, 8]), torch.zeros(4))
    score = torch.rand(2, 4, height, width)
    output = MultiAnchorOutput(torch.zeros_like(score), torch.full_like(score, .95),
                               torch.ones_like(score), torch.full_like(score, .5), score)
    proposal = frozen_proposal(evidence, output, probability_threshold=.9, utility_threshold_px=.01)
    extras = CriticExtras(left_rgb=torch.rand(2, 3, height, width) * 255,
                          right_rgb=torch.rand(2, 3, height, width) * 255,
                          temporal_residual=torch.rand(2, 4, height, width))
    return evidence, proposal, extras


def test_feature_families_are_cumulative_finite_and_sized():
    evidence, proposal, extras = example()
    previous = 0
    for family in FAMILIES:
        features, names = build_critic_features(evidence, proposal, extras, family)
        assert features.shape == (2, feature_channels(family), 16, 20)
        assert len(names) == feature_channels(family)
        assert torch.isfinite(features).all()
        assert features.shape[1] > previous
        previous = features.shape[1]
    assert "backbone" not in names and not any("gt" in name for name in names)


def test_features_do_not_mutate_frozen_inputs():
    evidence, proposal, extras = example()
    candidates = evidence.candidates.clone(); proposed = proposal.proposed.clone(); weight = proposal.fusion_weight.clone()
    build_critic_features(evidence, proposal, extras, "stereo")
    assert torch.equal(evidence.candidates, candidates)
    assert torch.equal(proposal.proposed, proposed)
    assert torch.equal(proposal.fusion_weight, weight)


def test_stereo_warp_sign_and_out_of_bounds():
    # Right image is the left image shifted right-to-left by the true disparity:
    # column x of left corresponds to column x-d of right (positive left disparity).
    height, width, disparity_true = 8, 32, 4.0
    ramp = torch.linspace(0, 1, width)[None, None, None, :].expand(1, 3, height, width).clone()
    left = ramp
    right = torch.roll(ramp, shifts=-int(disparity_true), dims=-1)
    disparity = torch.full((1, 1, height, width), disparity_true)
    residual, valid = stereo_residual(left, right, disparity)
    interior = residual[..., :, int(disparity_true) + 1:width - int(disparity_true) - 1]
    wrong = stereo_residual(left, right, disparity + 6.0)[0][..., :, 12:width - 12]
    assert float(interior.mean()) < float(wrong.mean())
    assert valid[..., :, 2].logical_not().all()  # x - d < 0 out of bounds
    assert not stereo_residual(left, right, torch.full_like(disparity, float("nan")))[1].any()


def test_plane_sweep_prefers_true_disparity():
    height, width = 8, 48
    image = torch.rand(1, 3, height, width)
    right = torch.roll(image, shifts=-3, dims=-1)
    good = plane_sweep_features(image, right, torch.full((1, 1, height, width), 3.0))
    bad = plane_sweep_features(image, right, torch.full((1, 1, height, width), 6.0))
    center = slice(8, width - 8)
    assert float(good[:, 0, :, center].mean()) < float(bad[:, 0, :, center].mean())


def test_critic_shapes_scale_clamp_and_parameter_budget():
    for family in FAMILIES:
        model = SpatialErrorCritic(feature_channels(family))
        features = torch.randn(2, feature_channels(family), 16, 20)
        output = model(features)
        for value in (output.mu_raw, output.mu_proposed, output.s_raw, output.s_proposed, output.harm_probability):
            assert value.shape == (2, 1, 16, 20)
        assert (output.mu_raw >= 0).all() and (output.mu_proposed >= 0).all()
        assert output.s_raw.min() >= LOG_SCALE_RANGE[0] and output.s_raw.max() <= LOG_SCALE_RANGE[1]
        assert (output.sigma_delta > 0).all()
        parameters = sum(parameter.numel() for parameter in model.parameters())
        assert 200_000 < parameters < 1_500_000


def test_losses_finite_and_backward():
    model = SpatialErrorCritic(feature_channels("geometry"), channels=16, dilations=(1, 2))
    features = torch.randn(2, feature_channels("geometry"), 12, 12)
    output = model(features)
    e_raw = torch.rand(2, 1, 12, 12); e_proposed = torch.rand(2, 1, 12, 12)
    mask = torch.rand(2, 1, 12, 12) > .3
    losses = critic_losses(output, e_raw, e_proposed, mask, harm_margin=.1)
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    empty = critic_losses(output, e_raw, e_proposed, torch.zeros_like(mask), harm_margin=.1)
    assert float(empty["total"]) == 0.0


def test_selective_policy_is_close_only_and_raw_bit_exact():
    evidence, proposal, extras = example()
    features, _ = build_critic_features(evidence, proposal, extras, "geometry")
    output = SpatialErrorCritic(feature_channels("geometry"), channels=16, dilations=(1,))(features)
    accept = selective_accept(output, proposal, lambda_uncertainty=1.0, tau_gain=0.0, tau_harm=.5)
    assert not (accept & ~proposal.eligible).any()  # never reopens a rejected pixel
    prediction, accepted = apply_veto(evidence.raw, proposal, accept)
    assert torch.equal(prediction[~accepted], evidence.raw[~accepted])
    assert torch.equal(prediction[accepted], proposal.proposed[accepted])
    # fully conservative thresholds -> universal exact raw fallback
    none = selective_accept(output, proposal, lambda_uncertainty=1.0, tau_gain=1e9, tau_harm=.5)
    prediction, accepted = apply_veto(evidence.raw, proposal, none)
    assert not accepted.any() and torch.equal(prediction, evidence.raw)


def test_invalid_proposal_always_rejected():
    evidence, proposal, extras = example()
    object.__setattr__(proposal, "eligible", torch.zeros_like(proposal.eligible))
    features, _ = build_critic_features(evidence, proposal, extras, "geometry")
    output = SpatialErrorCritic(feature_channels("geometry"), channels=16, dilations=(1,))(features)
    accept = selective_accept(output, proposal, lambda_uncertainty=0.0, tau_gain=-1e9, tau_harm=1.1)
    assert not accept.any()
