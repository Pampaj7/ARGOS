"""Protocol checks for the narrow multi-domain P4 control."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_multidomain_proposal_applicability import policy_metrics, split  # noqa: E402


def test_split_keeps_final_only_domains_and_backbones_out_of_selection() -> None:
    value = split()
    assert value["d4d_train"] == ["specimen_1"]
    assert value["d4d_calibration"] == ["specimen_2"]
    assert value["d4d_final_only"] == ["specimen_3"]
    assert value["d4d_backbones"] == ["S2M2-S"]
    assert {"SERV-CT", "StereoMIS", "Fast-FoundationStereo", "CREStereo"}.issubset(value["forbidden_before_freeze"])


def test_policy_metrics_uses_proposal_only_when_authorized() -> None:
    import numpy as np

    values = {
        "raw_error": np.array([1.0, 0.2, 2.0]),
        "proposal_error": np.array([0.2, 0.4, 3.0]),
        "utility": np.array([0.8, -0.2, -1.0]),
        "update": np.array([.5, .5, .5]),
    }
    result = policy_metrics(values, np.array([True, False, True]), .1)
    assert result["output_epe"] == (0.2 + 0.2 + 3.0) / 3
    assert result["coverage"] == 2 / 3
    assert result["precision"] == .5
    assert result["clean_degradation"] == 0.0
