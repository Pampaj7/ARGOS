"""Integration checks for contiguous SCARED-C clip construction."""
from __future__ import annotations

import pytest

from model_design.data.temporal_clip_dataset import TemporalClipDataset


@pytest.fixture(scope="module")
def short_clip_dataset() -> TemporalClipDataset:
    return TemporalClipDataset(
        ["S2M2-S"],
        ["dataset_7_keyframe_1"],
        clip_length=4,
        clip_stride=4,
        max_clips_per_sequence=1,
    )


def test_clip_is_causal_contiguous_and_never_crosses_sequence(short_clip_dataset: TemporalClipDataset) -> None:
    record = short_clip_dataset.records[0]
    pairs = [short_clip_dataset.pairs.records[index] for index in record.pair_indices]
    assert {pair.sequence for pair in pairs} == {record.sequence}
    assert {pair.backbone for pair in pairs} == {record.backbone}
    assert [pair.current_index for pair in pairs] == list(
        range(record.first_current_index, record.last_current_index + 1)
    )
    assert all(pair.past_index < pair.current_index for pair in pairs)


def test_clip_tensor_and_exact_frame_id_mapping(short_clip_dataset: TemporalClipDataset) -> None:
    sample = short_clip_dataset[0]
    assert sample["raw"].shape == (4, 1, 144, 180)
    assert sample["current_rgb"].shape == (4, 3, 144, 180)
    assert sample["sequence"] == "dataset_7_keyframe_1"
    assert sample["backbone"] == "S2M2-S"
    assert all(int(current) - int(past) == 1 for past, current in zip(
        sample["past_frame_id"], sample["current_frame_id"], strict=True
    ))


def test_clip_selection_is_deterministic() -> None:
    kwargs = dict(
        backbones=["S2M2-S"],
        sequences=["dataset_7_keyframe_1"],
        clip_length=4,
        max_clips_per_sequence=3,
        random_clip_selection=True,
        seed=91,
    )
    first = TemporalClipDataset(**kwargs)
    second = TemporalClipDataset(**kwargs)
    assert first.records == second.records

