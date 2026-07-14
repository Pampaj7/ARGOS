from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model_design.external_components.temporal_consensus import (
    ConsensusConfig,
    consensus_correction,
    consensus_fields,
    sweep_grid,
)


def test_fields_median_mad_and_count() -> None:
    aligned = np.stack(
        [np.full((2, 2), v, dtype=np.float32) for v in (1.0, 2.0, 3.0, 10.0)]
    )
    valid = np.ones_like(aligned, dtype=bool)
    valid[3, 0, 0] = False  # drop the outlier witness at one pixel
    fields = consensus_fields(aligned, valid)
    assert fields.count[0, 0] == 3 and fields.count[1, 1] == 4
    assert fields.median[0, 0] == 2.0  # median of {1,2,3}
    assert fields.median[1, 1] == 2.5  # median of {1,2,3,10}
    assert fields.spread[0, 0] == 1.0  # MAD of {1,2,3}


def test_no_witnesses_yields_nan_and_closed_gate() -> None:
    aligned = np.zeros((4, 3, 3), dtype=np.float32)
    valid = np.zeros_like(aligned, dtype=bool)
    fields = consensus_fields(aligned, valid)
    assert np.isnan(fields.median).all()
    refined, gate = consensus_correction(
        np.ones((3, 3), np.float32), fields, ConsensusConfig()
    )
    assert not gate.any()
    np.testing.assert_array_equal(refined, np.ones((3, 3), np.float32))


def test_gate_opens_only_on_tight_consensus_with_large_disagreement() -> None:
    h = w = 4
    raw = np.zeros((h, w), np.float32)
    # memories tightly agree at 5.0 -> raw is the outlier -> gate opens
    aligned = np.stack([np.full((h, w), 5.0 + e, np.float32) for e in (-0.1, 0.0, 0.1, 0.05)])
    valid = np.ones_like(aligned, dtype=bool)
    fields = consensus_fields(aligned, valid)
    config = ConsensusConfig(min_count=3, spread_max=0.5, disagree_min=1.0, kappa=1.0, bound=3.0)
    refined, gate = consensus_correction(raw, fields, config)
    assert gate.all()
    np.testing.assert_allclose(refined, 3.0, atol=1e-6)  # bounded at +3 px

    # clean pixel: raw agrees with consensus -> gate closed, identity
    refined2, gate2 = consensus_correction(fields.median.copy(), fields, config)
    assert not gate2.any()
    np.testing.assert_array_equal(refined2, fields.median)


def test_gate_closed_when_memories_disagree_among_themselves() -> None:
    h = w = 2
    raw = np.zeros((h, w), np.float32)
    aligned = np.stack([np.full((h, w), v, np.float32) for v in (2.0, 5.0, 8.0, 11.0)])
    valid = np.ones_like(aligned, dtype=bool)
    fields = consensus_fields(aligned, valid)
    config = ConsensusConfig(min_count=3, spread_max=0.5, disagree_min=1.0, kappa=1.0)
    _refined, gate = consensus_correction(raw, fields, config)
    assert not gate.any()  # spread (MAD=3.0) exceeds spread_max


def test_kappa_raises_threshold_with_spread() -> None:
    h = w = 1
    raw = np.zeros((h, w), np.float32)
    aligned = np.stack([np.full((h, w), v, np.float32) for v in (0.9, 1.0, 1.1, 1.0)])
    valid = np.ones_like(aligned, dtype=bool)
    fields = consensus_fields(aligned, valid)  # median 1.0, MAD 0.05
    open_cfg = ConsensusConfig(min_count=3, spread_max=0.5, disagree_min=0.5, kappa=0.0)
    closed_cfg = ConsensusConfig(min_count=3, spread_max=0.5, disagree_min=0.5, kappa=11.0)
    assert consensus_correction(raw, fields, open_cfg)[1].all()
    assert not consensus_correction(raw, fields, closed_cfg)[1].any()


def test_sweep_grid_matches_predeclared_size_and_labels_unique() -> None:
    grid = sweep_grid()
    assert len(grid) == 2 * 3 * 3 * 2
    labels = [config.label() for config in grid]
    assert len(set(labels)) == len(labels)
