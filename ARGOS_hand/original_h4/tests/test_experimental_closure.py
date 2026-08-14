from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_design.comparison.experimental_closure import CHECKPOINT_SHA256, FULL_D2_METHODS, _oracle_bundle, endpoint_oracle, write_freeze
from model_design.comparison.experimental_policies import POLICIES, ExperimentalPolicy, PolicySpec
from model_design.comparison.extract_safety import extract
from model_design.comparison.run_comparison import drive


class _Adapter:
    def __init__(self, horizon=4): self.horizon, self.frames = horizon, []
    def describe(self): return {"module": "test"}
    def start(self, frame): self.frames.append(frame); return {"disparity": frame["raw"], "support": frame["raw_valid"], "reset": True, "state_age": 0, "diagnostics": {"update_magnitude": 0.0}}
    def step(self, frame): self.frames.append(frame); return {"disparity": frame["raw"], "support": frame["raw_valid"], "reset": frame["reanchor"], "state_age": frame["state_age"], "diagnostics": {"update_magnitude": 0.0}}


def _frames(n=9):
    return [{"index": i, "raw": float(i), "raw_valid": True, "rgb": i, "right_rgb": i, "gt": 7.0} for i in range(n)]


class ExperimentalClosureTest(unittest.TestCase):
    def test_checkpoint_and_declared_methods_are_frozen(self):
        self.assertEqual(CHECKPOINT_SHA256, "99c5745c164fd4903b8aa8acf8f57efccacd9cdcd0d4ed4305cd10609324d725")
        self.assertEqual(set(POLICIES), {"fixed_w0.1_h4", "fixed_w0.2_h4", "fixed_w0.3_h4", "fixed_w0.5_h4", "fb_confidence_h4", "warped_recurrent_h4", "warped_raw_previous_h1", "ema2_h4", "ema3_h4"})

    def test_reset_indices_all_horizons_and_true_boundary_isolation(self):
        expected = {1: [True] * 8, 2: [True, False, True, False, True, False, True, False],
                    4: [True, False, False, False, True, False, False, False],
                    6: [True, False, False, False, False, False, True, False],
                    8: [True, False, False, False, False, False, False, False],
                    None: [True, False, False, False, False, False, False, False]}
        for horizon, wanted in expected.items():
            adapter = _Adapter(horizon); drive(adapter, _frames(), lambda current, past: (None, None))
            self.assertEqual([item["reanchor"] for item in adapter.frames[1:]], wanted)
            self.assertNotIn("gt", adapter.frames[0])
        first, second = _Adapter(4), _Adapter(4)
        drive(first, _frames(3), lambda *_: (None, None)); drive(second, _frames(3), lambda *_: (None, None))
        self.assertTrue(second.frames[1]["reanchor"])

    def test_first_frame_identity_and_raw_previous_are_explicit(self):
        adapter = _Adapter(1); outputs = dict(drive(adapter, _frames(3), lambda *_: (None, None)))
        self.assertEqual(outputs[0]["disparity"], 0.0)
        self.assertEqual([frame["past_disparity"] for frame in adapter.frames[1:]], [0.0, 1.0])

    def test_horizon_adapter_is_not_generic_definitive_factory(self):
        from model_design.comparison import canonical_horizons
        self.assertFalse(hasattr(canonical_horizons, "factory"))
        self.assertFalse(any(name.endswith("_factory") for name in vars(canonical_horizons)))
        self.assertEqual(canonical_horizons.CanonicalHorizon(horizon=6, device="cpu").horizon, 6)

    def test_fixed_half_equals_ema2_and_policy_is_deterministic(self):
        import torch
        raw = torch.ones((1, 1, 2, 2)); memory = torch.full_like(raw, 3); rgb = torch.zeros((1, 3, 2, 2)); flow = torch.zeros((1, 2, 2, 2))
        frame = {"raw": raw, "raw_valid": raw.bool(), "past_disparity": memory, "past_valid": raw.bool(), "current_rgb": rgb, "past_rgb": rgb,
                 "forward_flow": flow, "backward_flow": flow, "reanchor": True, "state_age": 1, "horizon": 4}
        fixed, ema = ExperimentalPolicy(POLICIES["fixed_w0.5_h4"], device="cpu"), ExperimentalPolicy(POLICIES["ema2_h4"], device="cpu")
        one, two, again = fixed.step(frame), ema.step(frame), fixed.step(frame)
        self.assertTrue(torch.equal(one["disparity"], two["disparity"]))
        self.assertTrue(torch.equal(one["disparity"], again["disparity"]))
        self.assertEqual(fixed.describe()["horizon"], 4)

    def test_oracle_is_posthoc_and_only_changes_official_support(self):
        raw = np.array([[1., 7.]]) ; memory = np.array([[3., 1.]]) ; gt = np.array([[3., 7.]]) ; mask = np.array([[True, False]])
        np.testing.assert_array_equal(endpoint_oracle(raw, memory, gt, mask), [[3., 7.]])

    def test_oracle_bundle_schema_always_carries_aligned_memory(self):
        bundle = {"raw_disparity": np.array([[[1.]]]), "aligned_memory": np.array([[[3.]]]), "gt_disparity": np.array([[[3.]]]),
                  "gt_valid": np.ones((1, 1, 1), bool), "protocol_mask": np.ones((1, 1, 1), bool)}
        self.assertEqual(float(_oracle_bundle(bundle)["refined_disparity"][0, 0, 0]), 3.0)

    def test_default_h4_golden_drive_regression(self):
        class Golden(_Adapter):
            def step(self, frame):
                self.frames.append(frame); value = frame["past_disparity"] + .25 * frame["raw"]
                return {"disparity": value, "support": frame["raw_valid"], "reset": frame["reanchor"], "state_age": frame["state_age"], "diagnostics": {"update_magnitude": 0.0}}
        frames = [{"index": i, "raw": float((i + 1) * 10), "raw_valid": True, "rgb": i, "right_rgb": i} for i in range(6)]
        adapter = Golden() ; outputs = dict(drive(adapter, frames, lambda *_: (None, None)))
        self.assertEqual([outputs[i]["disparity"] for i in range(6)], [10., 15., 22.5, 32.5, 45., 65.])
        self.assertEqual([x["past_disparity"] for x in adapter.frames[1:]], [10., 15., 22.5, 32.5, 50.])

    def test_raw_previous_and_finite_guards(self):
        with self.assertRaisesRegex(ValueError, "raw_previous"):
            PolicySpec("bad", 4, 1.0, "raw_previous")
        import torch
        raw = torch.full((1, 1, 2, 2), float("nan")); rgb = torch.zeros((1, 3, 2, 2)); flow = torch.zeros((1, 2, 2, 2))
        frame = {"raw": raw, "raw_valid": torch.ones_like(raw, dtype=torch.bool), "past_disparity": raw.nan_to_num(), "past_valid": torch.ones_like(raw, dtype=torch.bool),
                 "current_rgb": rgb, "past_rgb": rgb, "forward_flow": flow, "backward_flow": flow, "reanchor": True, "state_age": 1, "horizon": 4}
        with self.assertRaisesRegex(ValueError, "non-finite adapter input"):
            ExperimentalPolicy(POLICIES["fixed_w0.5_h4"], device="cpu").step(frame)

    def test_freeze_refuses_overwrite_and_declares_no_d7(self):
        import model_design.comparison.experimental_closure as closure
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(closure, "FREEZE", root / "freeze.json"), patch.object(closure, "PROMOTION", root / "promotion.json"):
                path = write_freeze(); data = json.loads(path.read_text())
                self.assertEqual(data["status"], "FROZEN_D2_DEVELOPMENT")
                self.assertIn("SCARED-C D7", data["development"]["forbidden_until_promotion"])
                self.assertEqual(data["full_d2_method_grid"], list(FULL_D2_METHODS))
                self.assertIn("inference_frozen_sea_raft_adapter", data["immutable_inputs"])
                self.assertIn("sea_raft_checkpoint", data["immutable_inputs"])
                with self.assertRaises(FileExistsError): write_freeze()

    def test_promotion_pins_complete_grid_and_decision_bytes(self):
        import hashlib
        import model_design.comparison.experimental_closure as closure
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); freeze, promotion, decision = root / "freeze.json", root / "promotion.json", root / "decision.json"
            with patch.object(closure, "RESULTS", root), patch.object(closure, "FREEZE", freeze), patch.object(closure, "PROMOTION", promotion):
                write_freeze()
                d2 = root / "d2/run_manifest.json"; d2.parent.mkdir()
                required = json.loads(freeze.read_text())["required_d2_scope"]
                d2.write_text(json.dumps({"status": "COMPLETE", "methods_requested": list(FULL_D2_METHODS), "scope": required,
                                           "method_report_count": required["method_report_count"], "summary_row_count": required["summary_row_count"]}))
                decision.write_text(json.dumps({"selected_confirmation_method": "canonical_h4", "d2_manifest_sha256": hashlib.sha256(d2.read_bytes()).hexdigest()}))
                value = json.loads(write_freeze(promotion=True, decision=decision).read_text())
                self.assertEqual(value["selected_confirmation_method"], "canonical_h4")
                self.assertEqual(value["d2_decision"]["sha256"], hashlib.sha256(decision.read_bytes()).hexdigest())

    def test_canonical_closure_oracle_bundle_integration(self):
        import model_design.comparison.experimental_closure as closure
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); freeze = root / "freeze.json"; output = root / "d2_smoke"
            bundle = {"dataset": "SCARED-C", "split": "d2", "protocol": "paper_d2_strict_all_anchors", "backbone": "RAFT-Stereo", "sequence_id": "dataset_2_keyframe_2",
                      "raw_disparity": np.array([[[1.]]]), "refined_disparity": np.array([[[2.]]]), "aligned_memory": np.array([[[3.]]]), "gt_disparity": np.array([[[3.]]]),
                      "gt_valid": np.ones((1, 1, 1), bool), "protocol_mask": np.ones((1, 1, 1), bool), "adapter_support": np.ones((1, 1, 1), bool),
                      "reset_mask": np.array([True]), "keyframe_mask": np.array([True]), "frame_ids": ["000008"], "calibration": None}
            def fake_scared(config, adapter, sink): sink(bundle); return [], {}
            config = SimpleNamespace(dataset="scared-d2", methods=["canonical_h4"], smoke=True, device="cuda:0", output=output, backbones=("RAFT-Stereo",), sequences=None, flow_batch_size=1, max_frames=1)
            with patch.object(closure, "FREEZE", freeze), patch.object(closure, "RESULTS", root), patch.object(closure, "_scared", fake_scared), patch.object(closure, "validate_cuda", return_value="1"):
                write_freeze(); closure.run(config)
            self.assertTrue((output / "reports/raw_vs_aligned_memory_oracle/RAFT-Stereo/dataset_2_keyframe_2.json").is_file())
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "COMPLETE")
            self.assertIn("summary.csv", manifest["output_hashes"])

    def test_full_d2_scope_and_smoke_root_are_fail_closed(self):
        import model_design.comparison.experimental_closure as closure
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); required = closure._freeze()["required_d2_scope"]
            base = dict(dataset="scared-d2", methods=list(FULL_D2_METHODS), smoke=False, device="cuda:0", output=None,
                        backbones=closure.ALL_BACKBONES, sequences=None, flow_batch_size=1, max_frames=None)
            with patch.object(closure, "RESULTS", root), patch.object(closure, "load_freeze", return_value={"required_d2_scope": required}):
                with self.assertRaisesRegex(ValueError, "partial methods"):
                    closure.run(SimpleNamespace(**(base | {"backbones": ("RAFT-Stereo",)})))
                with self.assertRaisesRegex(ValueError, "collide"):
                    closure.run(SimpleNamespace(**(base | {"output": root / "d2"})))
                with self.assertRaisesRegex(ValueError, "smoke output"):
                    closure.run(SimpleNamespace(**(base | {"smoke": True, "output": root / "d2", "methods": ["canonical_h4"]})))

    def test_canonical_reference_report_is_bound_to_checkpoint(self):
        report = ROOT.parent / "results/definitive_evaluation/canonical_h4_baseline/runs/scared-d2/model_design_comparison_canonical_h4__factory/reports/RAFT-Stereo/dataset_2_keyframe_2.json"
        value = json.loads(report.read_text())
        self.assertAlmostEqual(value["aggregate"]["disparity_px"]["raw"]["MAE"]["macro_sequence"], .22000366741653446)
        self.assertAlmostEqual(value["aggregate"]["disparity_px"]["refined"]["MAE"]["macro_sequence"], .21292861175571365)
        run = report.parents[2] / "run_manifest.json"
        self.assertEqual(json.loads(run.read_text())["module_provenance"]["checkpoint_sha256"], CHECKPOINT_SHA256)

    def test_safety_extraction_renames_zero_threshold_fraction(self):
        leaf = lambda value: {"macro_sequence": value, "micro_pixel": value, "support_count": 2}
        report = {"dataset": "SCARED-C", "split": "d2", "backbone": "RAFT-Stereo", "method": "fixed_w0.5_h4",
                  "aggregate": {"disparity_px": {"raw": {"MAE": leaf(2)}, "refined": {"MAE": leaf(1)}}},
                  "safety": {"disparity_px": {"aggregate": {"HUR": leaf(.2), "HPlus": leaf(.4), "BPlus": leaf(.8), "BUR": leaf(.6),
                      "thresholds": {str(x) + ".0": {"NewBad": leaf(.1)} for x in (1, 3, 5)},
                      "FrameDegradation": {key: leaf(.3) for key in ("P95", "P99", "Worst", "PositiveFraction")}}}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "r.json"; path.write_text(json.dumps(report))
            rows = extract([path])
        names = {row["metric"] for row in rows}
        self.assertIn("frame_degradation_PositiveFraction", names)
        self.assertNotIn("CatastrophicFraction", names)
        ratio = next(row for row in rows if row["metric"] == "BPlus_over_HPlus" and row["aggregate"] == "macro_sequence")
        self.assertEqual(ratio["value"], 2.0)
        self.assertEqual({row["method"] for row in rows}, {"fixed_w0.5_h4"})


if __name__ == "__main__":
    unittest.main()
