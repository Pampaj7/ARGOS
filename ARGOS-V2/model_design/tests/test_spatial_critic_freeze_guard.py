"""Freeze/dataset-7 protocol guards for the spatial safety critic (ARGOS v2).

These are cheap CPU tests that do not build any dataset: they exercise the
hard protocol boundaries (dataset 7 locked until freeze; frozen refiner weights
unchanged) that the campaign relies on.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)


def test_frozen_refiner_sha_matches_expected():
    from run_raw_multi_anchor_selective_gate import EXPECTED_FROZEN_SHA256, FROZEN_CHECKPOINT
    from run_raw_multi_anchor_temporal_refiner import sha256

    assert sha256(FROZEN_CHECKPOINT) == EXPECTED_FROZEN_SHA256


def test_dataset7_blocked_before_freeze(tmp_path):
    import run_raw_multi_anchor_spatial_safety_critic as runner

    # output dir with no freeze manifest -> test split must refuse to proceed
    config = SimpleNamespace(output=tmp_path, split="test", device="cpu")
    with pytest.raises(RuntimeError, match="locked"):
        runner.evaluate(config)
    # the guard must fire before any dataset is touched
    assert not (tmp_path / "frozen_test").exists()
