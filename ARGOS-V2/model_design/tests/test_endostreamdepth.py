from __future__ import annotations

import inspect

import pytest
import torch

from model_design.external_components.endostreamdepth import (
    CausalState,
    ExplicitMultiScaleState,
    initialize_state,
    reset_selected,
    state_statistics,
)
from model_design.models.latent_t1_refiner import LatentT1Refiner


def evidence(raw: torch.Tensor) -> dict[str, torch.Tensor]:
    ones = torch.ones_like(raw)
    zeros = torch.zeros_like(raw)
    return {
        "current_valid": ones.bool(),
        "aligned_past_disparity": raw + 0.25,
        "aligned_validity": ones.bool(),
        "warp_support": ones.bool(),
        "forward_backward_error": zeros,
        "forward_backward_confidence": ones,
        "photometric_residual": zeros,
        "flow_magnitude": zeros,
        "absolute_disparity_disagreement": torch.full_like(raw, 0.25),
    }


def features(batch: int = 2) -> dict[str, torch.Tensor]:
    return {
        "s4": torch.randn(batch, 4, 4, 5),
        "s8": torch.randn(batch, 4, 2, 3),
    }


def test_state_initializes_with_correct_contract_and_statistics() -> None:
    state = initialize_state(features(), ["a", "b"])
    assert state.scales == ("s4", "s8")
    assert state.sequence_ids == ("a", "b")
    assert state.frame_indices.tolist() == [-1, -1]
    assert all(not value.any() for value in state.tensors)
    assert set(state_statistics(state)) == {
        "s4_l2", "s4_mean_abs", "s4_max_abs", "s8_l2", "s8_mean_abs", "s8_max_abs", "update_counts"
    }


def test_selected_batch_elements_reset_independently() -> None:
    state = CausalState(
        ("s4",), (torch.ones(3, 2, 2, 2),), ("a", "b", "c"),
        torch.tensor([4, 5, 6]), torch.tensor([5, 6, 7]),
    )
    reset = reset_selected(state, torch.tensor([False, True, False]), sequence_ids=["a", "new", "c"])
    assert reset.sequence_ids == ("a", "new", "c")
    assert reset.tensors[0][0].all() and not reset.tensors[0][1].any() and reset.tensors[0][2].all()
    assert reset.frame_indices.tolist() == [4, -1, 6]
    assert reset.update_counts.tolist() == [5, 0, 7]


def test_state_serialization_restore_and_detach() -> None:
    value = torch.randn(1, 2, 2, 2, requires_grad=True) * 2
    state = CausalState(("s",), (value,), ("x",), torch.tensor([2]), torch.tensor([3]))
    detached = state.detach()
    assert detached.tensors[0].grad_fn is None and not detached.tensors[0].requires_grad
    restored = CausalState.restore(detached.serialize())
    torch.testing.assert_close(restored.tensors[0], value.detach())
    assert restored.sequence_ids == ("x",)


def test_no_future_or_reordered_frame_access() -> None:
    operator = ExplicitMultiScaleState(("s4", "s8"), 4)
    value = features()
    _, state = operator(value, None, sequence_ids=["a", "b"], frame_indices=torch.tensor([3, 3]))
    with pytest.raises(RuntimeError, match="strictly increasing"):
        operator(value, state, sequence_ids=["a", "b"], frame_indices=torch.tensor([2, 4]))


def test_sequence_crossing_requires_explicit_reset() -> None:
    operator = ExplicitMultiScaleState(("s4", "s8"), 4)
    value = features()
    _, state = operator(value, None, sequence_ids=["a", "b"], frame_indices=torch.tensor([0, 0]))
    with pytest.raises(RuntimeError, match="sequence crossing"):
        operator(value, state, sequence_ids=["other", "b"], frame_indices=torch.tensor([1, 1]))
    _, reset = operator(
        value, state, sequence_ids=["other", "b"], frame_indices=torch.tensor([1, 1]),
        reset_mask=torch.tensor([True, False]),
    )
    assert reset.sequence_ids == ("other", "b") and reset.update_counts.tolist() == [1, 2]


def test_deterministic_streaming_output() -> None:
    torch.manual_seed(4)
    model = LatentT1Refiner("E4", feature_channels=8, state_channels=4)
    frames = [torch.rand(2, 1, 16, 20) for _ in range(3)]

    def run():
        state = None; outputs = []
        for index, raw in enumerate(frames):
            result = model(raw, evidence(raw), state, sequence_ids=["a", "b"], frame_indices=torch.tensor([index, index]))
            state = result.state; outputs.append(result.disparity)
        return outputs

    for first, second in zip(run(), run(), strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_batch_streaming_matches_independent_sequence_streaming() -> None:
    torch.manual_seed(8)
    model = LatentT1Refiner("E3", feature_channels=8, state_channels=4).eval()
    frames = [torch.rand(2, 1, 16, 20) for _ in range(3)]
    batch_state = None; batch_outputs = []
    for index, raw in enumerate(frames):
        out = model(raw, evidence(raw), batch_state, sequence_ids=["a", "b"], frame_indices=torch.tensor([index, index]))
        batch_state = out.state; batch_outputs.append(out.disparity)
    for batch_index, sequence_id in enumerate(("a", "b")):
        state = None
        for index, raw_batch in enumerate(frames):
            raw = raw_batch[batch_index:batch_index + 1]
            out = model(raw, evidence(raw), state, sequence_ids=[sequence_id], frame_indices=torch.tensor([index]))
            state = out.state
            torch.testing.assert_close(out.disparity, batch_outputs[index][batch_index:batch_index + 1], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("variant,shapes", [
    ("E2", [(2, 4, 18, 22)]),
    ("E3", [(2, 4, 18, 22)]),
    ("E4", [(2, 4, 36, 45), (2, 4, 18, 22), (2, 4, 9, 11)]),
    ("E5", [(2, 4, 36, 45), (2, 4, 18, 22), (2, 4, 9, 11)]),
])
def test_state_shape_at_every_scale(variant: str, shapes: list[tuple[int, ...]]) -> None:
    model = LatentT1Refiner(variant, feature_channels=8, state_channels=4)
    raw = torch.rand(2, 1, 144, 180)
    out = model(raw, evidence(raw), None, sequence_ids=["a", "b"], frame_indices=torch.tensor([0, 0]))
    assert [tuple(item.shape) for item in out.state.tensors] == shapes


def test_gradients_flow_through_allowed_history_and_not_cached_input() -> None:
    torch.manual_seed(2)
    model = LatentT1Refiner("E3", feature_channels=8, state_channels=4)
    # Expose the state path while retaining bounded output.
    model.state_injection.data.fill_(0.2)
    model.head_delta.weight.data.normal_(0, 0.01)
    state = None; loss = 0
    raws = [torch.rand(1, 1, 12, 16) for _ in range(3)]
    for index, raw in enumerate(raws):
        out = model(raw, evidence(raw), state, sequence_ids=["s"], frame_indices=torch.tensor([index]))
        state = out.state; loss = loss + out.disparity.mean()
    loss.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.state_operator.parameters())
    assert all(raw.grad is None for raw in raws)
    assert state.tensors[0].grad_fn is not None


def test_identity_initialization_and_bounded_update() -> None:
    for variant in ("E2", "E3", "E4", "E5"):
        model = LatentT1Refiner(variant, feature_channels=8, state_channels=4)
        raw = torch.rand(2, 1, 16, 20) * 20
        out = model(raw, evidence(raw), None, sequence_ids=["a", "b"], frame_indices=torch.tensor([0, 0]))
        torch.testing.assert_close(out.disparity, raw, rtol=0, atol=0)
        assert out.update.abs().max() <= 3.0
        assert float(model.state_injection) == 0.0


def test_reset_prevents_stale_state_leakage() -> None:
    torch.manual_seed(12)
    model = LatentT1Refiner("E4", feature_channels=8, state_channels=4)
    model.state_injection.data.fill_(0.4); model.head_delta.weight.data.normal_(0, 0.02)
    old = torch.rand(1, 1, 16, 20)
    stale = model(old, evidence(old), None, sequence_ids=["old"], frame_indices=torch.tensor([0])).state
    current = torch.rand(1, 1, 16, 20)
    reset = model(current, evidence(current), stale, sequence_ids=["new"], frame_indices=torch.tensor([0]), reset_mask=torch.tensor([True]))
    fresh = model(current, evidence(current), None, sequence_ids=["new"], frame_indices=torch.tensor([0]))
    torch.testing.assert_close(reset.disparity, fresh.disparity, atol=1e-6, rtol=1e-6)
    for one, two in zip(reset.state.tensors, fresh.state.tensors, strict=True):
        torch.testing.assert_close(one, two, atol=1e-6, rtol=1e-6)


def test_wrong_sequence_state_is_detected() -> None:
    model = LatentT1Refiner("E2", feature_channels=8, state_channels=4)
    raw = torch.rand(1, 1, 12, 16)
    first = model(raw, evidence(raw), None, sequence_ids=["a"], frame_indices=torch.tensor([0]))
    with pytest.raises(RuntimeError, match="sequence crossing"):
        model(raw, evidence(raw), first.state, sequence_ids=["b"], frame_indices=torch.tensor([1]))


def test_forgetting_is_bounded_and_only_e5_exposes_maps() -> None:
    raw = torch.rand(1, 1, 16, 20)
    e5 = LatentT1Refiner("E5", feature_channels=8, state_channels=4)
    out = e5(raw, evidence(raw), None, sequence_ids=["a"], frame_indices=torch.tensor([0]))
    assert set(out.forget_maps) == {"s4", "s8", "s16"}
    assert all(torch.all((value >= 0) & (value <= 1)) for value in out.forget_maps.values())
    e4 = LatentT1Refiner("E4", feature_channels=8, state_channels=4)
    assert not e4(raw, evidence(raw), None, sequence_ids=["a"], frame_indices=torch.tensor([0])).forget_maps


def test_contract_has_no_backbone_or_future_input() -> None:
    parameters = set(inspect.signature(LatentT1Refiner.forward).parameters)
    assert "backbone" not in parameters and "backbone_id" not in parameters
    assert "future" not in parameters and "next_frame" not in parameters


def test_primary_unseen_backbone_is_rejected_during_tuning() -> None:
    from scripts.run_endostreamdepth_validation import validate_tuning_backbones

    with pytest.raises(ValueError, match="unseen backbones"):
        validate_tuning_backbones(["S2M2-S", "Fast-FoundationStereo"])


def test_variable_batch_change_requires_explicit_new_state() -> None:
    operator = ExplicitMultiScaleState(("s4", "s8"), 4)
    _, state = operator(features(2), None, sequence_ids=["a", "b"], frame_indices=torch.tensor([0, 0]))
    with pytest.raises(ValueError, match="batch contract"):
        operator(features(1), state, sequence_ids=["a"], frame_indices=torch.tensor([1]))
