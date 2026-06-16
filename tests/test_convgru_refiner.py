from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.temporal_refinement.lib.models import ConvGRURefiner
from scripts.temporal_refinement.train_temporal_refiner_fastcache import (
    ClipIndexedTemporalRefinerDataset,
    loss_weights_for_epoch,
    parse_args,
)


CACHE_ROOT = Path("results/temporal_refinement_cache/large_v3_s2m2s512_fast")


def require_fast_cache() -> Path:
    if not (CACHE_ROOT / "index_s2m2l736.csv").exists():
        pytest.skip("ARGOS fast temporal-refinement cache is not available")
    return CACHE_ROOT


def test_convgru_shapes_and_hidden_state():
    model = ConvGRURefiner(in_channels=4, base_channels=8, hidden_channels=16, residual_clamp_px=2.0)
    x0 = torch.randn(2, 4, 64, 96)
    x1 = torch.randn(2, 4, 64, 96)

    delta0, hidden0 = model(x0)
    delta1, hidden1 = model(x1, hidden0)

    assert delta0.shape == (2, 1, 64, 96)
    assert delta1.shape == (2, 1, 64, 96)
    assert hidden0.shape == (2, 16, 16, 24)
    assert hidden1.shape == hidden0.shape
    assert torch.isfinite(delta0).all()
    assert torch.isfinite(delta1).all()


def make_clip_dataset(sequence_length: int = 5, random_crop: bool = False):
    cache_root = require_fast_cache()
    return ClipIndexedTemporalRefinerDataset(
        cache_root=cache_root,
        index_file="index_s2m2l736.csv",
        sample_ids=None,
        crop_size=(64, 96),
        random_crop=random_crop,
        backbone_prefix="s2m2_l736",
        spatial_teacher_prefix="s2m2_l736",
        temporal_teacher_prefix="sav",
        spatial_target="backbone",
        disp_norm=128.0,
        sequence_length=sequence_length,
    )


def test_clip_dataset_returns_causal_four_channel_inputs():
    dataset = make_clip_dataset(sequence_length=5)
    sample = dataset[0]

    assert sample["input"].shape == (5, 4, 64, 96)
    assert sample["s_center"].shape == (5, 1, 64, 96)
    assert sample["l_teacher"].shape == (5, 1, 64, 96)
    assert sample["sav_teacher"].shape == (5, 1, 64, 96)
    assert torch.isfinite(sample["input"]).all()
    assert torch.isfinite(sample["s_center"]).all()


def test_clip_dataset_keeps_sequence_boundaries_and_order():
    dataset = make_clip_dataset(sequence_length=5)
    for rows in dataset.clips[:25] + dataset.clips[-25:]:
        sequence_ids = {row["sequence_id"] for row in rows}
        frame_ids = [int(row["center_frame_id"]) for row in rows]
        assert len(sequence_ids) == 1
        assert frame_ids == list(range(frame_ids[0], frame_ids[0] + len(rows)))


def test_clip_dataset_supports_full_sequence_eval_mode():
    dataset = make_clip_dataset(sequence_length=0)
    assert len(dataset) > 0
    sample = dataset[0]
    assert sample["input"].ndim == 4
    assert sample["input"].shape[1:] == (4, 64, 96)
    assert isinstance(sample["source_sequence"], str)


def test_loss_schedule_weights():
    args = parse_args(
        [
            "--spatial-weight",
            "0.40",
            "--abs-sav-weight",
            "0.35",
            "--delta-sav-weight",
            "0.20",
            "--res-weight",
            "0.20",
            "--edge-weight",
            "0.05",
        ]
    )
    static = loss_weights_for_epoch(args, 17)
    assert static == {
        "spatial": 0.40,
        "abs_sav": 0.35,
        "delta_sav": 0.20,
        "res": 0.20,
        "edge": 0.05,
    }

    args = parse_args(
        [
            "--loss-schedule",
            "--schedule-warmup-epochs",
            "10",
            "--schedule-transition-epochs",
            "20",
            "--spatial-weight",
            "0.40",
            "--abs-sav-weight",
            "0.35",
            "--delta-sav-weight",
            "0.20",
            "--res-weight",
            "0.20",
            "--edge-weight",
            "0.05",
            "--final-spatial-weight",
            "0.25",
            "--final-abs-sav-weight",
            "0.25",
            "--final-delta-sav-weight",
            "0.40",
            "--final-res-weight",
            "0.10",
            "--final-edge-weight",
            "0.05",
        ]
    )
    initial = {"spatial": 0.40, "abs_sav": 0.35, "delta_sav": 0.20, "res": 0.20, "edge": 0.05}
    assert loss_weights_for_epoch(args, 1, initial) == initial
    assert loss_weights_for_epoch(args, 10, initial) == initial

    epoch_11 = loss_weights_for_epoch(args, 11, initial)
    assert epoch_11["spatial"] == pytest.approx(0.3925)
    assert epoch_11["abs_sav"] == pytest.approx(0.345)
    assert epoch_11["delta_sav"] == pytest.approx(0.21)
    assert epoch_11["res"] == pytest.approx(0.195)
    assert epoch_11["edge"] == pytest.approx(0.05)

    epoch_20 = loss_weights_for_epoch(args, 20, initial)
    assert epoch_20["spatial"] == pytest.approx(0.325)
    assert epoch_20["abs_sav"] == pytest.approx(0.30)
    assert epoch_20["delta_sav"] == pytest.approx(0.30)
    assert epoch_20["res"] == pytest.approx(0.15)
    assert epoch_20["edge"] == pytest.approx(0.05)

    final = {"spatial": 0.25, "abs_sav": 0.25, "delta_sav": 0.40, "res": 0.10, "edge": 0.05}
    assert loss_weights_for_epoch(args, 30, initial) == pytest.approx(final)
    assert loss_weights_for_epoch(args, 31, initial) == pytest.approx(final)
    assert loss_weights_for_epoch(args, 100, initial) == pytest.approx(final)
