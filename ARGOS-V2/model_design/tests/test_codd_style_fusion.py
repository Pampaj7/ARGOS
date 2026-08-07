import torch

from model_design.losses.codd_fusion_losses import CODDFusionLossConfig, codd_fusion_losses_with_gt
from model_design.models.codd_style_fusion import CODDCues, CODDFusionOutput, CODDStyleFusionHead, _appearance_self_correlation, _cross_appearance_correlation, _disparity_self_correlation, build_codd_cues, convex_fusion_oracle, hard_endpoint_fusion


def test_codd_fusion_equation_and_raw_fallback():
    raw = torch.full((1, 1, 8, 8), 2.0)
    memory = torch.full((1, 1, 8, 8), 10.0)
    cues = CODDCues(torch.zeros(1, 8, 8, 8), torch.ones(1, 1, 8, 8, dtype=torch.bool), 8)
    model = CODDStyleFusionHead(8, width=16)
    output = model(cues, raw, memory)
    assert torch.allclose(output.fused_disparity, (1 - output.reset_weight * output.fusion_weight) * raw + output.reset_weight * output.fusion_weight * memory)
    manual = CODDFusionOutput(torch.zeros_like(raw), torch.ones_like(raw), torch.zeros_like(raw), raw)
    assert torch.equal(manual.fused_disparity, raw)


def test_codd_dead_bands_and_directional_labels():
    # e_M - e_S: +2 is worse, -2 better, zero tie in cache-grid units.
    raw = torch.tensor([[[[0.0, 2.0, 0.0]]]])
    memory = torch.tensor([[[[2.0, 0.0, 0.0]]]])
    gt = torch.zeros_like(raw); valid = torch.ones_like(raw, dtype=torch.bool)
    output = CODDFusionOutput(
        reset_weight=torch.tensor([[[[0.0, 1.0, 0.3]]]]),
        fusion_weight=torch.tensor([[[[0.0, 1.0, 0.5]]]]),
        temporal_weight=torch.zeros_like(raw), fused_disparity=raw,
    )
    losses = codd_fusion_losses_with_gt(output, raw=raw, memory=memory, gt=gt, valid=valid,
        config=CODDFusionLossConfig(tau_reset_px=1.0, tau_fusion_px=1.0, alpha_reg=.2))
    assert float(losses["reset"]) == 0.0
    assert float(losses["fusion"]) == 0.0
    assert abs(float(losses["fusion_tie_fraction"]) - 1 / 3) < 1e-6


def test_local_self_and_cross_correlations_are_finite_and_shaped():
    disparity = torch.arange(25, dtype=torch.float32).view(1, 1, 5, 5)
    feature = torch.randn(1, 8, 5, 5)
    assert _disparity_self_correlation(disparity).shape == (1, 8, 5, 5)
    assert _appearance_self_correlation(feature).shape == (1, 8, 5, 5)
    assert _cross_appearance_correlation(feature, feature).shape == (1, 9, 5, 5)
    assert torch.isfinite(_cross_appearance_correlation(feature, feature)).all()


def test_convex_oracle_distinguishes_interpolation_from_endpoint_selection():
    raw = torch.tensor([[[[0.0, 0.0, 2.0]]]])
    memory = torch.tensor([[[[2.0, 0.0, 4.0]]]])
    gt = torch.tensor([[[[1.0, 3.0, 1.0]]]])
    fused, weight = convex_fusion_oracle(raw, memory, gt)
    # GT lies between the first endpoint pair: interpolation reaches it exactly.
    assert torch.equal(weight[..., :1], torch.full_like(weight[..., :1], .5))
    assert torch.equal(fused[..., :1], gt[..., :1])
    # Equal endpoints use an explicit zero coefficient without NaN.
    assert weight[..., 1].item() == 0.0 and torch.isfinite(fused).all()
    # When GT is outside the segment the closest endpoint is selected.
    assert weight[..., 2].item() == 0.0


def test_hard_endpoint_fusion_is_bit_exact_at_raw_fallback():
    raw = torch.tensor([[[[1.25, -2.0]]]])
    memory = torch.tensor([[[[8.0, 5.0]]]])
    output, accepted = hard_endpoint_fusion(raw, memory, torch.tensor([[[[.49, .50]]]]), .50)
    assert not accepted[..., 0].item() and accepted[..., 1].item()
    assert torch.equal(output[..., :1], raw[..., :1])
    assert torch.equal(output[..., 1:], memory[..., 1:])


def test_no_learned_stereo_evidence_contract_has_no_extractor_dependency():
    raw = torch.ones(1, 1, 12, 16)
    cue = build_codd_cues(None, raw=raw, aligned_memory=raw * 1.2,
        current_rgb=torch.zeros(1, 3, 12, 16), current_right_rgb=torch.zeros(1, 3, 12, 16),
        past_rgb=torch.zeros(1, 3, 12, 16), flow_current_to_past=torch.zeros(1, 2, 12, 16),
        flow_magnitude=torch.zeros(1, 1, 12, 16), forward_backward_confidence=torch.ones(1, 1, 12, 16),
        warp_support=torch.ones(1, 1, 12, 16, dtype=torch.bool), aligned_valid=torch.ones(1, 1, 12, 16, dtype=torch.bool),
        include_learned_stereo_evidence=False)
    assert cue.channels == 38 and torch.isfinite(cue.values).all()
