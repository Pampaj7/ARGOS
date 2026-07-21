from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

V2_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = V2_ROOT / "scripts/run_large_scale_bida_signal_audit.py"


def _load():
    spec = importlib.util.spec_from_file_location("large_scale_bida_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit = _load()


def test_metric_sums_matches_manual_oracle() -> None:
    raw_error = np.array([[0.0, 2.0], [4.0, 1.0]], np.float32)
    memory_error = np.array([[1.0, 1.0], [5.0, 1.0]], np.float32)
    result = audit.finalize_sums(audit.metric_sums(raw_error, memory_error, np.ones((2, 2), bool)))
    assert result["raw_epe"] == 1.75
    assert result["memory_epe"] == 2.0
    assert result["oracle_epe"] == 1.5
    assert result["oracle_epe_gain"] == 0.25
    assert result["memory_better_fraction"] == 0.25
    assert result["memory_worse_fraction"] == 0.5
    assert result["tie_fraction"] == 0.25
    assert result["mean_helpful_magnitude"] == 1.0
    assert result["mean_harmful_magnitude"] == 1.0
    assert result["raw_bad3"] == 0.25
    assert result["oracle_bad3"] == 0.25


def test_oracle_is_never_worse_for_random_inputs() -> None:
    rng = np.random.default_rng(4)
    raw = rng.random((20, 30), dtype=np.float32) * 8
    memory = rng.random((20, 30), dtype=np.float32) * 8
    mask = rng.random((20, 30)) > 0.2
    result = audit.finalize_sums(audit.metric_sums(raw, memory, mask))
    assert result["oracle_epe"] <= result["raw_epe"]
    assert result["oracle_epe"] <= result["memory_epe"]


def test_temporal_difference_uses_disparity_not_error_difference() -> None:
    raw_error = np.array([[1.0]], np.float32)
    memory_error = np.array([[1.0]], np.float32)
    disparity_difference = np.array([[3.5]], np.float32)
    result = audit.finalize_sums(audit.metric_sums(
        raw_error, memory_error, np.ones((1, 1), bool),
        temporal_difference=disparity_difference,
    ))
    assert result["gt_relative_temporal_error_consistency"] == 0.0
    assert result["flow_warped_raw_temporal_difference"] == 3.5


def test_region_partitions_are_exact() -> None:
    shape = (3, 4)
    common = np.ones(shape, bool)
    boundary = np.zeros(shape, bool); boundary[:, 0] = True
    flow = np.zeros(shape, np.float32); flow[:, 2:] = 2
    fb = np.ones(shape, bool); fb[0, 0] = False
    raw_error = np.array([[0, .5, 2, 4]] * 3, np.float32)
    gt = np.array([[1, 3, 6, 9]] * 3, np.float32)
    regions = audit.region_masks(common, boundary, flow, fb, raw_error, gt)
    assert np.array_equal(regions["boundary"] | regions["non_boundary"], common)
    assert not (regions["boundary"] & regions["non_boundary"]).any()
    assert np.array_equal(regions["motion_low_lt_1px"] | regions["motion_high_ge_1px"], common)
    assert np.array_equal(regions["fb_consistent"] | regions["fb_inconsistent_occlusion_like"], common)
    raw_partition = (regions["raw_error_low_le_1px"].astype(int)
                     + regions["raw_error_mid_1_to_3px"].astype(int)
                     + regions["raw_error_high_gt_3px"].astype(int))
    assert np.array_equal(raw_partition, common.astype(int))


def test_bootstrap_uses_sequence_values_and_is_deterministic() -> None:
    values = np.array([0.1, 0.2, 0.3, 0.4])
    first = audit.bootstrap_mean(values, seed=9, replicates=2000)
    second = audit.bootstrap_mean(values, seed=9, replicates=2000)
    assert first == second
    assert first[0] > 0
    assert first[0] <= values.mean() <= first[1]


def test_no_learned_component_is_imported_or_referenced() -> None:
    source = SCRIPT.read_text()
    forbidden = (
        "learned_t1_refiner", "raw_error_detector", "proposal_applicability",
        "quality_predictor", "ppmstereo", "dinov3", "endostreamdepth",
    )
    assert not any(token in source.lower() for token in forbidden)
    assert "BiDAFlowInferenceAdapter" in source
    assert "causal_warp" in source


def test_metric_contract_declares_weighted_gt_and_cache_units() -> None:
    definitions = audit.metric_definitions()
    assert definitions["primary_gt_coverage_threshold"] == 0.50
    assert "resize(disparity * valid) / resize(valid)" in definitions["gt_resize"]
    assert "width 180" in definitions["disparity_units"]
    assert definitions["stereo_confidence_bins"].startswith("not evaluated")
