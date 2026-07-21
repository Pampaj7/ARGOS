"""Target and balancing tests without accessing external GPU/data state."""
from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import sys

import numpy as np
import torch

from model_design.data.utility_memory_selector_dataset import (
    BalancedSequenceSampler, HierarchicalDatasetSequenceSampler,
    dataset_id_from_sequence, utility_targets,
)
from model_design.data.temporal_pair_dataset import TemporalPairDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


def test_utility_targets_match_manual_errors_and_mask():
    batch = {
        "raw": torch.tensor([[[[2., 5.]]]]), "gt": torch.tensor([[[[1., 1.]]]]),
        "gt_coverage": torch.ones(1, 1, 1, 2), "raw_valid": torch.ones(1, 1, 1, 2, dtype=torch.bool),
    }
    aligned = torch.tensor([[[[1., 9.]]]])
    target = utility_targets(batch, aligned, torch.ones_like(aligned, dtype=torch.bool), torch.ones_like(aligned, dtype=torch.bool), epsilon_px=.1)
    assert torch.equal(target.utility, torch.tensor([[[[1., -4.]]]]))
    assert target.memory_better[0, 0, 0, 0]
    assert target.harmful_magnitude[0, 0, 0, 1] > 0


def test_invalid_memory_never_has_a_target():
    batch = {"raw": torch.ones(1,1,1,1), "gt": torch.zeros(1,1,1,1), "gt_coverage": torch.ones(1,1,1,1), "raw_valid": torch.ones(1,1,1,1,dtype=torch.bool)}
    t = utility_targets(batch, torch.zeros(1,1,1,1), torch.zeros(1,1,1,1,dtype=torch.bool), torch.ones(1,1,1,1,dtype=torch.bool))
    assert not t.valid.any() and not t.memory_better.any()


def test_additional_candidate_support_is_part_of_the_exact_target_mask():
    batch = {"raw": torch.ones(1,1,1,2), "gt": torch.zeros(1,1,1,2),
             "gt_coverage": torch.ones(1,1,1,2), "raw_valid": torch.ones(1,1,1,2,dtype=torch.bool)}
    aligned = torch.zeros(1,1,1,2)
    extra = torch.tensor([[[[True, False]]]])
    target = utility_targets(batch, aligned, torch.ones_like(extra), torch.ones_like(extra), additional_valid=extra)
    assert target.valid.tolist() == [[[[True, False]]]]


def test_regional_utility_target_pools_supervision_but_preserves_pixel_metric_target():
    batch = {
        "raw": torch.ones(1, 1, 3, 3), "gt": torch.zeros(1, 1, 3, 3),
        "gt_coverage": torch.ones(1, 1, 3, 3), "raw_valid": torch.ones(1, 1, 3, 3, dtype=torch.bool),
    }
    aligned = torch.zeros(1, 1, 3, 3)
    aligned[0, 0, 0, 0] = 3.0  # one harmful memory location: utility -2
    target = utility_targets(
        batch, aligned, torch.ones_like(aligned, dtype=torch.bool),
        torch.ones_like(aligned, dtype=torch.bool), regional_kernel=3,
    )
    assert target.utility[0, 0, 0, 0] == -2.0
    # The center regional label averages all nine valid pixel utilities.
    assert torch.allclose(target.supervision_utility[0, 0, 1, 1], target.utility.mean())
    assert target.supervision_utility[0, 0, 1, 1] < target.utility[0, 0, 1, 1]


def test_balanced_sampler_equalizes_sequence_groups_deterministically():
    records = [SimpleNamespace(backbone="A", sequence="long") for _ in range(5)] + [SimpleNamespace(backbone="B", sequence="short") for _ in range(2)]
    dataset = SimpleNamespace(records=records)
    sampler = BalancedSequenceSampler(dataset, seed=3)
    first = list(iter(sampler)); second = list(iter(sampler))
    assert len(first) == 10 and first == second
    assert sum(i < 5 for i in first) == 5 and sum(i >= 5 for i in first) == 5


def test_hierarchical_sampler_equalizes_backbone_session_and_sequence_deterministically():
    # dataset_1 has a short and a long sequence; dataset_2 has one sequence.
    # It must not receive less exposure merely because it has fewer keyframes.
    records = (
        [SimpleNamespace(backbone="A", sequence="dataset_1_keyframe_2") for _ in range(2)]
        + [SimpleNamespace(backbone="A", sequence="dataset_1_keyframe_3") for _ in range(4)]
        + [SimpleNamespace(backbone="A", sequence="dataset_2_keyframe_2") for _ in range(3)]
        + [SimpleNamespace(backbone="B", sequence="dataset_1_keyframe_2") for _ in range(2)]
        + [SimpleNamespace(backbone="B", sequence="dataset_1_keyframe_3") for _ in range(4)]
        + [SimpleNamespace(backbone="B", sequence="dataset_2_keyframe_2") for _ in range(3)]
    )
    dataset = SimpleNamespace(records=records)
    sampler = HierarchicalDatasetSequenceSampler(dataset, seed=11)
    first, second = list(sampler), list(sampler)
    assert first == second
    # max dataset group = dataset_1 with 2 sequences * 4 samples each.
    assert len(first) == 2 * 2 * 8
    counts = {}
    for index in first:
        record = records[index]
        key = (record.backbone, dataset_id_from_sequence(record.sequence))
        counts[key] = counts.get(key, 0) + 1
    assert set(counts.values()) == {8}
    # Full coverage is retained before any repeats.
    assert set(range(len(records))).issubset(first)


def test_dataset_id_parser_rejects_non_scared_sequence_names():
    assert dataset_id_from_sequence("dataset_7_keyframe_4") == "dataset_7"
    try:
        dataset_id_from_sequence("not_a_scared_sequence")
    except ValueError:
        pass
    else:
        raise AssertionError("malformed sequence must not become an implicit group")


def test_explicit_dataset_disjoint_split_is_accepted_and_session_leakage_is_rejected():
    from argparse import Namespace
    from run_utility_memory_selector import _validated_split
    valid = Namespace(
        train_sequences=["dataset_1_keyframe_2", "dataset_3_keyframe_1", "dataset_6_keyframe_1"],
        validation_sequences=["dataset_2_keyframe_2"], test_sequences=["dataset_7_keyframe_1"],
        strict_dataset_id_disjoint=True,
    )
    assert _validated_split(valid)[2] == ["dataset_7_keyframe_1"]
    leaked = Namespace(**{**valid.__dict__, "validation_sequences": ["dataset_7_keyframe_2"]})
    try:
        _validated_split(leaked)
    except ValueError as exc:
        assert "dataset-ID leakage" in str(exc)
    else:
        raise AssertionError("dataset-ID overlap across calibration/test must be rejected")


def test_optional_ram_preload_is_deterministic_and_compact(monkeypatch, tmp_path):
    dataset = object.__new__(TemporalPairDataset)
    dataset.sequences = ("synthetic",)
    dataset._infos = {"synthetic": SimpleNamespace(frame_ids=["0000", "0001"], seq_dir=tmp_path)}
    dataset._frame_data = {}
    monkeypatch.setattr(
        "model_design.data.temporal_pair_dataset.read_rgb",
        lambda _path: np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3),
    )
    monkeypatch.setattr(
        "model_design.data.temporal_pair_dataset.load_frame_gt",
        lambda _info, _frame: (np.ones((8, 10), np.float32) * 4, np.ones((8, 10), bool)),
    )
    summary = dataset.preload_frame_data(2)
    first = tuple(array.copy() for array in dataset._frame_data["synthetic"])
    dataset.preload_frame_data(2)
    second = dataset._frame_data["synthetic"]
    assert summary["frames"] == 2 and summary["bytes"] > 0
    assert first[0].dtype == np.uint8 and first[1].dtype == np.float32
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
