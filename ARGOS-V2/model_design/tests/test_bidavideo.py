from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch

V2_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = V2_ROOT / "model_design/external_components/bidavideo.py"


def _load_argos_module():
    spec = importlib.util.spec_from_file_location("argos_bidavideo", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bida = _load_argos_module()


def _load_original_flow_warp():
    path = V2_ROOT.parent / "external/bidavideo/train_utils/losses.py"
    spec = importlib.util.spec_from_file_location("original_bidavideo_losses", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.flow_warp


def test_original_flow_warp_equivalence() -> None:
    torch.manual_seed(4)
    source = torch.randn(2, 3, 7, 9)
    flow = torch.randn(2, 2, 7, 9) * 0.75
    expected = _load_original_flow_warp()(source, flow)
    actual = bida.causal_warp(source, flow).warped
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_identity_flow_returns_original_and_full_support() -> None:
    source = torch.randn(2, 4, 5, 7)
    result = bida.causal_warp(source, torch.zeros(2, 2, 5, 7))
    torch.testing.assert_close(result.warped, source, rtol=1e-6, atol=1e-6)
    assert result.support.all()
    assert result.valid.all()


def test_integer_translation_and_out_of_bounds_support() -> None:
    source = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)
    flow = torch.zeros(1, 2, 3, 4)
    flow[:, 0] = 1.0
    result = bida.causal_warp(source, flow)
    expected = torch.tensor([[[[1, 2, 3, 0], [5, 6, 7, 0], [9, 10, 11, 0]]]], dtype=torch.float32)
    torch.testing.assert_close(result.warped, expected, rtol=0, atol=0)
    assert result.support[:, :, :, :3].all()
    assert not result.support[:, :, :, 3].any()


def test_mask_warp_is_conservative() -> None:
    source = torch.ones(1, 1, 2, 4)
    source_valid = torch.tensor([[[[1, 1, 0, 1], [1, 1, 0, 1]]]], dtype=torch.bool)
    flow = torch.zeros(1, 2, 2, 4)
    flow[:, 0] = 1.0
    result = bida.causal_warp(source, flow, source_valid=source_valid)
    expected = torch.tensor([[[[1, 0, 1, 0], [1, 0, 1, 0]]]], dtype=torch.bool)
    assert torch.equal(result.valid, expected)


def test_flow_resize_scales_components_independently() -> None:
    flow = torch.ones(1, 2, 2, 4)
    resized = bida.resize_flow(flow, (6, 8))
    assert resized.shape == (1, 2, 6, 8)
    torch.testing.assert_close(resized[:, 0], torch.full((1, 6, 8), 2.0))
    torch.testing.assert_close(resized[:, 1], torch.full((1, 6, 8), 3.0))


def test_forward_backward_consistency_for_inverse_translation() -> None:
    forward = torch.zeros(1, 2, 5, 8)
    backward = torch.zeros_like(forward)
    forward[:, 0] = 1.0
    backward[:, 0] = -1.0
    result = bida.forward_backward_consistency(forward, backward, absolute_threshold=0.01)
    assert result.support[:, :, :, :-1].all()
    assert result.valid[:, :, :, :-1].all()
    torch.testing.assert_close(result.error[:, :, :, :-1], torch.zeros(1, 1, 5, 7))
    assert not result.valid[:, :, :, -1].any()


def test_warp_is_differentiable_wrt_source_and_flow() -> None:
    source = torch.randn(1, 2, 4, 5, requires_grad=True)
    flow = (torch.randn(1, 2, 4, 5) * 0.1).requires_grad_()
    bida.causal_warp(source, flow).warped.square().mean().backward()
    assert source.grad is not None and torch.isfinite(source.grad).all()
    assert flow.grad is not None and torch.isfinite(flow.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for vendored SEA-RAFT")
def test_default_sea_raft_is_frozen_and_builds_no_graph() -> None:
    adapter = bida.BiDAFlowInferenceAdapter("sea_raft", device="cuda:0")
    assert not adapter.model.training
    assert all(not parameter.requires_grad for parameter in adapter.model.parameters())
    current = torch.rand(1, 3, 144, 180, device="cuda") * 255.0
    past = torch.rand_like(current)
    flow = adapter.current_to_past(current, past)
    assert flow.shape == (1, 2, 144, 180)
    assert not flow.requires_grad
    assert flow.grad_fn is None


def test_temporal_evidence_contract_and_causality() -> None:
    shape = (2, 1, 6, 8)
    current_disp = torch.ones(shape)
    past_disp = torch.full(shape, 2.0)
    valid = torch.ones(shape, dtype=torch.bool)
    current_rgb = torch.zeros(2, 3, 6, 8)
    past_rgb = torch.zeros_like(current_rgb)
    zero_flow = torch.zeros(2, 2, 6, 8)
    evidence = bida.temporal_disparity_evidence(
        current_disp,
        past_disp,
        zero_flow,
        zero_flow,
        current_valid=valid,
        past_valid=valid,
        current_rgb=current_rgb,
        past_rgb=past_rgb,
    )
    assert set(evidence.as_dict()) == {
        "aligned_past_disparity",
        "warp_support",
        "aligned_validity",
        "forward_backward_error",
        "forward_backward_confidence",
        "photometric_residual",
        "absolute_disparity_disagreement",
        "signed_disparity_disagreement",
        "flow_magnitude",
    }
    torch.testing.assert_close(evidence.signed_disparity_disagreement, torch.ones(shape))
