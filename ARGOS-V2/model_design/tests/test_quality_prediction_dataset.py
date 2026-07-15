from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from model_design.data.quality_prediction_dataset import (  # noqa: E402
    CANDIDATE_AGES,
    CANDIDATE_NAMES,
    MEMORY_AGES,
    QualityPredictionDataset,
    assemble_quality_candidates,
    memory_consensus,
)
from model_design.data.temporal_pair_dataset import resize_gt_to_cache_masked  # noqa: E402
from model_design.external_components.bidavideo import BiDAFlowInferenceAdapter  # noqa: E402


@pytest.fixture(scope="module")
def real_dataset() -> QualityPredictionDataset:
    return QualityPredictionDataset(
        ["S2M2-S"], ["dataset_7_keyframe_1"], max_samples_per_sequence=1
    )


def test_candidate_order_ages_causality_and_exact_frame_mapping(real_dataset: QualityPredictionDataset) -> None:
    sample = real_dataset[0]
    assert tuple(sample["candidate_names"]) == CANDIDATE_NAMES
    assert tuple(sample["candidate_ages"].tolist()) == CANDIDATE_AGES
    assert tuple(sample["ages"].tolist()) == MEMORY_AGES
    assert sample["source_frame_ids"][0] == sample["current_frame_id"]
    assert all(int(sample["current_frame_id"]) - int(frame_id) == age for frame_id, age in zip(
        sample["source_frame_ids"][1:], MEMORY_AGES, strict=True
    ))
    assert all(age > 0 for age in MEMORY_AGES)


def test_no_sequence_crossing_or_backbone_switching(real_dataset: QualityPredictionDataset) -> None:
    sample = real_dataset[0]
    assert set(sample["source_sequences"]) == {sample["sequence"]}
    assert set(sample["source_backbones"]) == {sample["backbone"]}


def test_batch_loading_matches_individual_loading(real_dataset: QualityPredictionDataset) -> None:
    individual = real_dataset[0]
    batched = next(iter(DataLoader(real_dataset, batch_size=1, shuffle=False, num_workers=0)))
    for key in ("raw", "past", "source_disparity", "source_valid", "gt", "gt_coverage"):
        torch.testing.assert_close(batched[key][0], individual[key])


def test_weighted_gt_resize_ignores_invalid_extreme_values() -> None:
    disparity = np.ones((288, 360), np.float32) * 4
    valid = np.ones_like(disparity, dtype=bool)
    disparity[:2, :2] = 10000; valid[:2, :2] = False
    resized, coverage = resize_gt_to_cache_masked(disparity, valid)
    assert coverage.min() >= 0 and coverage.max() <= 1
    # Native width 360 -> cache width 180, therefore valid 4 px becomes 2 px.
    assert np.nanmax(resized[coverage == 1]) == pytest.approx(2.0, abs=1e-5)


def synthetic_assembly():
    raw = torch.tensor([[[[2.0, 4.0], [6.0, 8.0]]]])
    memory = torch.stack([raw + offset for offset in (1.0, -1.0, 2.0, -2.0)], dim=1)
    one = torch.ones_like(memory, dtype=torch.bool); zero = torch.zeros_like(memory)
    batch = {
        "raw": raw, "raw_valid": torch.ones_like(raw, dtype=torch.bool),
        "gt": torch.ones_like(raw) * 3, "gt_coverage": torch.ones_like(raw),
    }
    evidence = {
        "aligned_past_disparity": memory, "aligned_validity": one, "warp_support": one,
        "forward_backward_error": zero.float(), "forward_backward_confidence": one.float(),
        "photometric_residual": zero.float(), "flow_magnitude": zero.float(),
    }
    return batch, evidence


def test_candidate_masks_targets_and_advantage_are_exact() -> None:
    batch, evidence = synthetic_assembly()
    evidence["aligned_validity"][:, -1, :, 0, 0] = False
    candidates = assemble_quality_candidates(batch, evidence, coverage_threshold=0.5)
    assert candidates.disparity.shape == (1, 5, 1, 2, 2)
    assert not candidates.target_valid[0, -1, 0, 0, 0]
    expected = (candidates.disparity - batch["gt"][:, None]).abs()
    torch.testing.assert_close(candidates.target_error, expected)
    torch.testing.assert_close(candidates.target_advantage, expected[:, :1] - expected)


def test_consensus_even_median_mad_and_count() -> None:
    values = torch.tensor([1.0, 2.0, 3.0, 10.0]).view(1, 4, 1, 1, 1)
    valid = torch.ones_like(values, dtype=torch.bool)
    median, mad, count = memory_consensus(values, valid)
    assert median.item() == 2.5 and mad.item() == 1.0 and count.item() == 4


@pytest.mark.parametrize("backbone", ["Fast-FoundationStereo", "CREStereo"])
def test_unseen_backbones_are_rejected_before_loading(backbone: str) -> None:
    with pytest.raises(ValueError, match="must not touch unseen"):
        QualityPredictionDataset([backbone], ["dataset_7_keyframe_1"], max_samples_per_sequence=1)


def test_sea_raft_is_frozen_and_default_inference_needs_no_graph() -> None:
    adapter = BiDAFlowInferenceAdapter("sea_raft", device="cpu")
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in adapter.model.parameters())
