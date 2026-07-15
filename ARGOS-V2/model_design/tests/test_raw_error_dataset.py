from __future__ import annotations

import pytest
import torch

from model_design.data.raw_error_dataset import RawErrorDataset, raw_error_targets


def synthetic_batch() -> dict[str, torch.Tensor]:
    raw = torch.tensor([[[[1.0, 2.0, 4.0, 8.0]]]])
    gt = torch.tensor([[[[1.1, 2.5, 3.0, 4.0]]]])
    return {
        "raw": raw,
        "gt": gt,
        "gt_coverage": torch.tensor([[[[1.0, 0.6, 0.4, 1.0]]]]),
        "raw_valid": torch.tensor([[[[True, True, True, False]]]]),
    }


def test_target_errors_threshold_band_and_masks_are_exact() -> None:
    target = raw_error_targets(
        synthetic_batch(), epsilon_px=0.5, indifference_band_px=0.1,
        coverage_threshold=0.5, clean_threshold_px=0.5,
    )
    torch.testing.assert_close(target.error, torch.tensor([[[[0.1, 0.5, 1.0, 4.0]]]]))
    assert target.regression_valid.tolist() == [[[[True, True, False, False]]]]
    # 0.5 is inside the indifference band and is excluded.
    assert target.classification_valid.tolist() == [[[[True, False, False, False]]]]
    assert target.label.tolist() == [[[[0.0, 0.0, 1.0, 1.0]]]]
    assert target.clean.tolist() == [[[[True, True, False, False]]]]


@pytest.mark.parametrize("backbone", ["Fast-FoundationStereo", "CREStereo", "invented"])
def test_unseen_or_unknown_backbones_rejected_before_loading(backbone: str) -> None:
    with pytest.raises(ValueError):
        RawErrorDataset([backbone], ["dataset_7_keyframe_1"], max_pairs_per_sequence=1)


def test_real_records_are_exact_causal_and_do_not_cross_or_switch() -> None:
    dataset = RawErrorDataset(
        ["S2M2-S", "RAFT-Stereo"], ["dataset_7_keyframe_1"],
        max_pairs_per_sequence=2, random_clip_start=False,
    )
    for record in dataset.records:
        assert record.current_index == record.past_index + 1
        assert record.sequence == "dataset_7_keyframe_1"
        assert record.backbone in dataset.backbones
        assert record.current_frame_id != record.past_frame_id
    first = dataset[0]
    assert first["sequence"] == dataset.records[0].sequence
    assert first["backbone"] == dataset.records[0].backbone
    assert first["current_frame_id"] == dataset.records[0].current_frame_id


def test_validation_loading_is_deterministic() -> None:
    kwargs = dict(
        backbones=["S2M2-S"], sequences=["dataset_7_keyframe_1"],
        max_pairs_per_sequence=1, random_clip_start=False,
    )
    first, second = RawErrorDataset(**kwargs), RawErrorDataset(**kwargs)
    assert first.records == second.records
    for key in ("raw", "past", "gt", "gt_coverage", "raw_valid"):
        assert torch.equal(first[0][key], second[0][key])

