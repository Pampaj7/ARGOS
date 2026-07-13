from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

V2_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V2_ROOT))
sys.path.insert(0, str(V2_ROOT / "scripts"))

from argos_v2.scared_c_data import load_sequence_info  # noqa: E402
from model_design.data.temporal_pair_dataset import (  # noqa: E402
    TemporalPairDataset,
    resize_gt_to_cache_masked,
)
from model_design.models.learned_t1_refiner import LearnedT1Refiner  # noqa: E402


def synthetic_evidence(batch: int = 2, height: int = 16, width: int = 20) -> dict[str, torch.Tensor]:
    one = torch.ones(batch, 1, height, width)
    return {
        "aligned_past_disparity": torch.rand_like(one) * 10,
        "current_valid": one.bool(),
        "aligned_validity": one.bool(),
        "warp_support": one.bool(),
        "forward_backward_error": torch.rand_like(one),
        "forward_backward_confidence": torch.rand_like(one),
        "photometric_residual": torch.rand_like(one),
        "flow_magnitude": torch.rand_like(one) * 3,
    }


def test_identity_initialization_and_bounded_update() -> None:
    torch.manual_seed(7)
    model = LearnedT1Refiner("A7", tau_px=3.0)
    raw = torch.rand(2, 1, 16, 20) * 20
    output = model(raw, synthetic_evidence(), torch.rand(2, 3, 16, 20) * 255)
    torch.testing.assert_close(output.disparity, raw, rtol=0, atol=0)
    assert float(output.g_error.max()) < 0.02
    assert float(output.update.abs().max()) <= 3.0

    with torch.no_grad():
        model.head_delta.bias.fill_(100)
        model.head_error.bias.fill_(100)
        model.head_memory.bias.fill_(100)
    bounded = model(raw, synthetic_evidence(), torch.rand(2, 3, 16, 20) * 255)
    assert float(bounded.update.abs().max()) <= 3.0 + 1e-6


def test_selector_gradients_but_not_cached_inputs_or_flow() -> None:
    model = LearnedT1Refiner("A5")
    raw = torch.rand(2, 1, 12, 16)
    evidence = synthetic_evidence(height=12, width=16)
    flow = torch.rand(2, 2, 12, 16)
    evidence["flow_magnitude"] = torch.linalg.vector_norm(flow, dim=1, keepdim=True)
    output = model(raw, evidence, torch.rand(2, 3, 12, 16) * 255)
    (output.disparity.mean() + output.g_error.mean() + output.c_memory.mean()).backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.parameters())
    assert raw.grad is None
    assert flow.grad is None
    assert all(value.grad is None for value in evidence.values() if value.is_floating_point())


@pytest.mark.parametrize("random_clip", [False, True])
def test_dataset_order_no_crossing_and_cache_alignment(random_clip: bool) -> None:
    sequence = "dataset_1_keyframe_2"
    dataset = TemporalPairDataset(
        ["S2M2-S"],
        [sequence],
        max_pairs_per_sequence=3,
        random_clip_start=random_clip,
        seed=19,
    )
    info = load_sequence_info(sequence)
    for record in dataset.records:
        assert record.sequence == sequence
        assert record.current_index == record.past_index + 1
        assert record.current_frame_id == info.frame_ids[record.current_index]
        assert record.past_frame_id == info.frame_ids[record.past_index]
    sample = dataset[0]
    assert sample["raw"].shape == (1, 144, 180)
    assert sample["gt"].shape == sample["raw"].shape
    assert sample["current_frame_id"] == dataset.records[0].current_frame_id
    assert 0.0 < float(sample["gt_coverage"].mean()) <= 1.0


def test_deterministic_validation_dataset() -> None:
    kwargs = dict(
        backbones=["S2M2-S"],
        sequences=["dataset_1_keyframe_2"],
        max_pairs_per_sequence=2,
        random_clip_start=False,
        seed=123,
    )
    first = TemporalPairDataset(**kwargs)
    second = TemporalPairDataset(**kwargs)
    assert first.records == second.records
    a, b = first[0], second[0]
    for key in ("raw", "past", "gt", "gt_coverage", "current_rgb", "past_rgb"):
        assert torch.equal(a[key], b[key])


def test_cache_gt_is_coverage_normalized() -> None:
    """A constant valid disparity must stay constant beside invalid regions."""
    import numpy as np

    disparity = np.zeros((288, 360), np.float32)
    valid = np.zeros((288, 360), bool)
    disparity[:, :181] = 8.0
    valid[:, :181] = True
    resized, coverage = resize_gt_to_cache_masked(disparity, valid)
    # Width halves from 360 to 180, so an 8 px disparity becomes 4 px.
    assert np.allclose(resized[coverage > 0], 4.0)
