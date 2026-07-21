"""Deterministic protocol checks for unchanged-A2 multi-domain training."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_multidomain_a2_training import selection_score, sources


def test_a2_sources_keep_d4d_to_the_geometrically_valid_s2m2_backbone():
    args = SimpleNamespace(seed=3, max_train_pairs=256, max_validation_pairs=160)
    _manifest, scared_train, scared_validation, d4d_train, d4d_validation = sources(args, smoke=True)
    assert set(record.backbone for record in scared_train.records) == {
        "S2M2-S", "RAFT-Stereo", "StereoAnywhere",
    }
    assert set(record.backbone for record in scared_validation.records) == {
        "S2M2-S", "RAFT-Stereo", "StereoAnywhere",
    }
    assert set(record.backbone for record in d4d_train.records) == {"S2M2-S"}
    assert set(record.backbone for record in d4d_validation.records) == {"S2M2-S"}
    assert set(record.specimen for record in d4d_train.records) == {"specimen_1"}
    assert set(record.specimen for record in d4d_validation.records) == {"specimen_2"}


def test_selection_score_gives_equal_weight_to_domains_not_pixel_count():
    scared = {"raw_epe": 1.0, "refined_epe": .9}
    d4d = {"raw_epe": 10.0, "refined_epe": 8.0}
    assert abs(selection_score(scared, d4d) - .85) < 1e-12
    # A large raw-error scale must not dominate selection merely by units.
    d4d_scaled = {"raw_epe": 100.0, "refined_epe": 80.0}
    assert abs(selection_score(scared, d4d_scaled) - .85) < 1e-12
