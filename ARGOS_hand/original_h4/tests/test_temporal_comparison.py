from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_design.comparison.definitive_evaluation import evaluate_scared_bundle, non_gt_applicability
from model_design.comparison.run_comparison import _group, atomic_json, check_adapter, default_output, drive, filter_d4d_windows, official_scared_protocol_mask, prepare_output, validate_cuda


class FakeAdapter:
    def __init__(self): self.calls = []
    def describe(self): return {"module": "fake", "checkpoint_sha256": "fake"}
    def start(self, frame): self.calls.append(("start", dict(frame))); return {"disparity": frame["raw"], "support": frame["raw_valid"], "reset": True, "state_age": 0, "diagnostics": {}}
    def step(self, frame):
        self.calls.append(("step", dict(frame)))
        return {"disparity": frame["raw"], "support": frame["raw_valid"], "reset": frame["reanchor"], "state_age": frame["state_age"], "diagnostics": {}}


class EmptySupportAdapter(FakeAdapter):
    def step(self, frame):
        result = super().step(frame); result["support"] = False
        return result


class TemporalComparisonTest(unittest.TestCase):
    def test_driver_uses_only_start_step_no_gt_and_h4_phase(self):
        adapter = FakeAdapter(); check_adapter(adapter)
        frames = [{"index": index, "raw": index, "raw_valid": True, "rgb": index, "right_rgb": index,
                   "gt": 99, "coverage": 99, "backbone": "forbidden", "protocol_mask": False,
                   "adapter_support": False, "sequence_id": "forbidden"} for index in range(7)]
        seen_past_indices = []
        def flow(current, past):
            seen_past_indices.append(past["index"])
            return (current["index"], past["rgb"]), (past["rgb"], current["index"])
        outputs = drive(adapter, frames, flow)
        self.assertEqual([index for index, _ in outputs], list(range(7)))
        self.assertEqual([name for name, _ in adapter.calls], ["start", "step", "step", "step", "step", "step", "step"])
        self.assertEqual([call[1]["reanchor"] for call in adapter.calls if call[0] == "step"], [True, False, False, False, True, False])
        self.assertEqual(seen_past_indices, [0, 1, 2, 3, 4, 5])
        forbidden = {"gt", "coverage", "backbone", "protocol_mask", "adapter_support", "sequence_id"}
        self.assertTrue(all(not (forbidden & value.keys()) for _, value in adapter.calls))
        self.assertEqual(set(adapter.calls[0][1]), {"raw", "raw_valid", "rgb", "right_rgb", "index"})

    def test_adapter_contract_and_cuda_guard(self):
        adapter = FakeAdapter(); check_adapter(adapter)
        self.assertEqual(adapter.describe()["checkpoint_sha256"], "fake")
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}, clear=True): self.assertEqual(validate_cuda("cuda:0"), "1")
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "CUDA_VISIBLE_DEVICES"): validate_cuda("cuda:0")

    def test_output_guard_refuses_existing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"; prepare_output(output)
            with self.assertRaises(FileExistsError): prepare_output(output)

    def test_smoke_default_output_is_distinct_from_full(self):
        root = Path("/tmp/results")
        base = SimpleNamespace(dataset="scared-d2", module="package.module:factory", smoke=False)
        smoke = SimpleNamespace(dataset="scared-d2", module="package.module:factory", smoke=True)
        self.assertEqual(default_output(root, base), root / "scared-d2" / "package_module__factory")
        self.assertNotEqual(default_output(root, base), default_output(root, smoke))

    def test_atomic_json_rejects_nan_and_cleans_temporary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            with self.assertRaises(ValueError): atomic_json(path, {"value": float("nan")})
            self.assertFalse(path.exists())
            self.assertEqual(list(path.parent.glob(".report.json.*")), [])
            with patch.object(Path, "replace", side_effect=OSError("replace failed")):
                with self.assertRaisesRegex(OSError, "replace failed"): atomic_json(path, {"value": 1})
            self.assertEqual(list(path.parent.glob(".report.json.*")), [])

    def test_scared_grouping_keeps_methods_and_weights_epe_by_pixels(self):
        rows = [
            {"backbone": "S", "method": "raw", "valid_pixel_count": 10, "error_sum": 10.0},
            {"backbone": "S", "method": "raw", "valid_pixel_count": 90, "error_sum": 18.0},
            {"backbone": "S", "method": "temporal_module", "valid_pixel_count": 100, "error_sum": 20.0},
        ]
        values = _group(rows, "backbone")
        self.assertEqual([(row["method"], row["frames"]) for row in values], [("raw", 2), ("temporal_module", 1)])
        self.assertAlmostEqual(values[0]["epe"], 0.28)
        self.assertAlmostEqual(values[1]["epe"], 0.20)

    def test_official_scared_support_ignores_empty_adapter_support(self):
        adapter = EmptySupportAdapter()
        outputs = dict(drive(adapter, [{"index": 0, "raw": 1, "raw_valid": True, "rgb": 1, "right_rgb": 1},
                                       {"index": 1, "raw": 1, "raw_valid": True, "rgb": 1, "right_rgb": 1}], lambda *_: (None, None)))
        self.assertFalse(outputs[1]["support"])
        self.assertTrue(official_scared_protocol_mask([[True]], [[True]], [[[True]]])[0, 0])

    def test_definitive_report_is_strict_json_without_dense_inputs(self):
        report = evaluate_scared_bundle({"dataset": "SCARED-C", "split": "d7", "protocol": "h4_only_common_support",
                                         "backbone": "fake", "sequence_id": "sequence", "frame_ids": ["a", "b"],
                                         "raw_disparity": np.ones((2, 1, 1)), "refined_disparity": np.full((2, 1, 1), 2.0),
                                         "gt_disparity": np.full((2, 1, 1), 2.0), "gt_valid": np.ones((2, 1, 1), bool),
                                         "protocol_mask": np.ones((2, 1, 1), bool), "adapter_support": np.zeros((2, 1, 1), bool),
                                         "reset_mask": np.array([False, False]), "keyframe_mask": np.array([True, False]), "calibration": None})
        encoded = json.dumps(report, allow_nan=False)
        self.assertEqual(report["spatial"]["disparity_px"]["raw"]["support_count"], 2)
        self.assertNotIn("raw_disparity", encoded)
        self.assertEqual(report["diagnostics"]["adapter_support_coverage"], 0.0)

    def test_d4d_and_servct_have_no_numeric_gt_metrics(self):
        for dataset in ("d4d", "servct"):
            report = non_gt_applicability(dataset)
            self.assertEqual(report["unified_gt_metric_families"], "NOT_APPLICABLE")
            self.assertIsNone(report["numeric_gt_metrics"])

    def test_d4d_context_is_past_to_present_in_source(self):
        source = (ROOT / "model_design/comparison/run_comparison.py").read_text()
        adapter = (ROOT / "model_design/comparison/canonical_h4.py").read_text()
        self.assertIn('context_stems"].split(";")[::-1]', source)
        self.assertIn("SEARAFTFlowAdapter", source)
        self.assertIn("flow_model.infer(torch.cat((current, past)), torch.cat((past, current)))", source)
        self.assertIn("argos_freezed.alignment.bida_pull_warp", adapter)
        self.assertNotIn("evaluate_scared", source)
        self.assertNotIn("evaluate_d4d", source)

    def test_d4d_filters_missing_source_before_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); left = root / "left.png"; right = root / "right.png"; left.touch(); right.touch()
            contexts = {"ok": {"context_stems": "a;b;c;d"}, "missing": {"context_stems": "e;f;g;h"}}
            windows = [{"sequence_id": "missing", "specimen": "specimen_1", "session": "s"}, {"sequence_id": "ok", "specimen": "specimen_2", "session": "s"}]
            paths = {f"specimen_2__s__{stem}": {"left_path": str(left), "right_path": str(right)} for stem in "abcd"}
            accepted, unavailable = filter_d4d_windows(windows, contexts, paths)
            self.assertEqual(accepted, [windows[1]])
            self.assertEqual(unavailable, {"specimen_1": 1})


if __name__ == "__main__": unittest.main()
