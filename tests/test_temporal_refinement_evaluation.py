from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.temporal_refinement.evaluate_temporal_refinement import (
    disp_to_depth_mm,
    ema_baseline,
    enumerate_checkpoints,
    pareto_front,
    previous_blend,
    temporal_median,
)


def arrays(vals):
    return [np.full((2, 2), v, dtype=np.float32) for v in vals]


def test_disparity_to_depth_conversion():
    disp = np.array([[10.0, 20.0]], dtype=np.float32)
    depth = disp_to_depth_mm(disp, fx=100.0, baseline_mm=5.0)
    assert depth[0, 0] == pytest.approx(50.0)
    assert depth[0, 1] == pytest.approx(25.0)


def test_ema_and_previous_blend_baselines():
    disps = arrays([0.0, 10.0, 20.0])
    ema = ema_baseline(disps, alpha=0.5)
    assert float(ema[0][0, 0]) == pytest.approx(0.0)
    assert float(ema[1][0, 0]) == pytest.approx(5.0)
    assert float(ema[2][0, 0]) == pytest.approx(12.5)

    prev = previous_blend(disps, alpha=0.5)
    assert float(prev[2][0, 0]) == pytest.approx(15.0)


def test_temporal_median_baselines():
    disps = arrays([0.0, 100.0, 10.0, 20.0, 30.0])
    noncausal = temporal_median(disps, window=3, causal=False)
    causal = temporal_median(disps, window=3, causal=True)
    assert float(noncausal[1][0, 0]) == pytest.approx(10.0)
    assert float(causal[1][0, 0]) == pytest.approx(50.0)
    assert float(causal[2][0, 0]) == pytest.approx(10.0)


def test_pareto_front():
    rows = [
        {"name": "a", "x": 1.0, "y": 5.0},
        {"name": "b", "x": 2.0, "y": 3.0},
        {"name": "c", "x": 3.0, "y": 4.0},
        {"name": "d", "x": 0.5, "y": 6.0},
    ]
    front = pareto_front(rows, "x", "y")
    names = {r["name"] for r in front}
    assert names == {"a", "b", "d"}


def test_checkpoint_enumeration(tmp_path: Path):
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir()
    for name in ["best.pt", "latest.pt", "epoch_0010.pt", "epoch_0020.pt"]:
        (ckpt / name).write_text("x")
    assert [p.name for p in enumerate_checkpoints(tmp_path, "best")] == ["best.pt"]
    assert [p.name for p in enumerate_checkpoints(tmp_path, "latest")] == ["latest.pt"]
    assert [p.name for p in enumerate_checkpoints(tmp_path, "periodic")] == ["epoch_0010.pt", "epoch_0020.pt"]
    assert [p.name for p in enumerate_checkpoints(tmp_path, "all")] == [
        "best.pt",
        "latest.pt",
        "epoch_0010.pt",
        "epoch_0020.pt",
    ]
