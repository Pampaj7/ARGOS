"""Focused CPU checks for Definitive Evaluation wide tables and DRENDS input."""
from __future__ import annotations

import csv
import hashlib
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

from model_design.comparison import drends_evaluation as drends  # noqa: E402
from model_design.comparison.run_comparison import drive  # noqa: E402
from model_design.comparison.run_definitive_evaluation import (  # noqa: E402
    DATASET_METADATA, WIDE_METADATA, _reduce_scalars, _wide_metadata, build_wide_rows, finalize_output_hashes, write_wide_tables,
)
from model_design.metrics.unified_metrics import MetricConfig, evaluate_argos_prediction  # noqa: E402


def leaf(sequence: str, statistic: str, value: float | bool) -> dict[str, str]:
    return {"dataset": "SCARED-C", "split": "d2", "backbone": "CREStereo", "sequence": sequence,
            "section": "aggregate", "metric_path": "aggregate/disparity_px/raw/EPE", "method": "raw",
            "statistic": statistic, "value": str(value), "record_level": "aggregate",
            "protocol": "fixed", "metric_scope": "gt_referenced"}


class DefinitiveWideTest(unittest.TestCase):
    def test_unequal_support_reduction_and_worst_direction(self):
        rows = [leaf("a", "macro_sequence", 1), leaf("a", "micro_pixel", 1), leaf("a", "support_count", 1),
                leaf("a", "higher_is_better", False), leaf("a", "worst_sequence", 1),
                leaf("b", "macro_sequence", 3), leaf("b", "micro_pixel", 3), leaf("b", "support_count", 9),
                leaf("b", "higher_is_better", False), leaf("b", "worst_sequence", 3)]
        reduced = _reduce_scalars(rows)
        self.assertEqual(reduced["macro_sequence"], 2.0)
        self.assertEqual(reduced["micro_pixel"], 2.8)
        self.assertEqual(reduced["support_count"], 10.0)
        self.assertEqual(reduced["worst_sequence"], 3.0)

    def test_wide_metadata_labels_seen_unseen_and_ood_without_false_training(self):
        seen = _wide_metadata("SCARED-C", "d2", "RAFT-Stereo", {"module": "m"}, protocol="p", scope="gt")
        unseen = _wide_metadata("SCARED-C", "d7", "CREStereo", {"module": "m"}, protocol="p", scope="gt")
        ood = _wide_metadata("D4D", "", "RAFT-Stereo", {"module": "m"}, protocol="p", scope="no_reference")
        self.assertEqual(seen["backbone_status"], "training_seen")
        self.assertEqual(unseen["backbone_status"], "training_unseen")
        self.assertEqual(ood["domain_status"], "OOD")
        self.assertFalse(seen["evaluation_split_used_in_training"])
        self.assertEqual(DATASET_METADATA[("DRENDS", "pilot")][0], "OOD_metric")

    def test_wide_table_contains_every_aggregate_scalar_with_collision_free_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary)
            path = stage / "scared_aggregate_metrics.csv"
            fields = list(leaf("a", "macro_sequence", 1))
            rows = [leaf("a", "macro_sequence", 1), leaf("a", "support_count", 10), leaf("a", "higher_is_better", False),
                    leaf("a", "worst_sequence", 1), leaf("b", "macro_sequence", 3), leaf("b", "support_count", 30),
                    leaf("b", "higher_is_better", False), leaf("b", "worst_sequence", 3)]
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
            wide = build_wide_rows(stage, {"module": "m", "code_sha256": "c"}, ("scared-d2",), {})
            self.assertEqual(len(wide), 1)
            metric = "aggregate__aggregatex2fxdisparityx5fxpxx2fxrawx2fxEPE__raw__macrox5fxsequence"
            self.assertEqual(wide[0][metric], 2.0)
            self.assertTrue(set(WIDE_METADATA) <= set(wide[0]))
            write_wide_tables(stage, {"module": "m", "code_sha256": "c"}, ("scared-d2",), {})
            self.assertTrue((stage / "definitive_table.csv").is_file())
            self.assertTrue((stage / "tables/scared_c.csv").is_file())

    def test_drends_manifest_order_paths_timing_and_resize_units(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); curated = root / "curated"; workspace = root / "workspace"
            for frame_id in ("00000", "00001", "00002", "00003", "00004", "00005", "00006"):
                for name in ("left", "right", "depth", "mask"):
                    path = workspace / f"{frame_id}_{name}"; path.parent.mkdir(parents=True, exist_ok=True); path.touch()
            frames = []
            for index in range(7):
                frame_id = f"{index:05d}"; frames.append({"frame_id": frame_id, "rect_left": f"workspace/{frame_id}_left",
                    "rect_right": f"workspace/{frame_id}_right", "depth_left": f"workspace/{frame_id}_depth", "mask_left": f"workspace/{frame_id}_mask",
                    "helios": float(index + 1), "left": float(index + 1), "right": float(index + 1),
                    "left_right_offset_ms": 150.0 if index == 0 else 1.0, "left_helios_offset_ms": 1.0, "right_helios_offset_ms": 1.0})
            manifest = {"path_base": "repo_root", "sequence": {"recording": "Vid14_Pancreas_High", "frames": frames}}
            quality = {"rectified_projection": {"focal_baseline": 8.0, "independent_disparity_ground_truth": False}}
            curated.mkdir(); (curated / "temporal_pilot_manifest.json").write_text(json.dumps(manifest)); (curated / "temporal_pilot_quality_report.json").write_text(json.dumps(quality))
            with patch.object(drends, "CURATED", curated), patch.object(drends, "HAND", root):
                records, info = drends.load_drends_records("Vid14_Pancreas_High", max_frames=6)
            self.assertEqual(len(records), 6); self.assertEqual(info["excluded_timing_frame_ids"], ["00000"])
            self.assertAlmostEqual(drends.CANONICAL_SIZE[0] / 1280, 180 / 1280)
            source = Path(drends.__file__).read_text()
            self.assertNotIn("np.save", source); self.assertNotIn("torch.save", source)

    def test_drends_fake_flow_receives_only_canonical_inputs(self):
        import numpy as np
        import torch
        left = np.zeros((720, 1280, 3), dtype=np.uint8)
        right = np.ones((720, 1280, 3), dtype=np.uint8)
        raw = np.full((720, 1280), 12.0, dtype=np.float32)
        current = drends._canonical_frame(1, left, right, raw, device="cpu")
        past = drends._canonical_frame(0, left, right, raw, device="cpu")
        class FakeFlow:
            def current_to_anchor(self, a, b):
                self.forward = (tuple(a.shape), tuple(b.shape)); return torch.zeros((1, 2, 144, 180))
            def anchor_to_current(self, a, b):
                self.backward = (tuple(a.shape), tuple(b.shape)); return torch.zeros((1, 2, 144, 180))
        fake = FakeFlow()
        drends._validate_canonical_frame(current); drends._validate_canonical_frame(past)
        forward = fake.current_to_anchor(current["rgb"], past["rgb"])
        backward = fake.anchor_to_current(past["rgb"], current["rgb"])
        self.assertEqual(fake.forward, ((1, 3, 144, 180), (1, 3, 144, 180)))
        self.assertEqual(fake.backward, ((1, 3, 144, 180), (1, 3, 144, 180)))
        self.assertEqual(tuple(current["raw"].shape), (1, 1, 144, 180))
        self.assertEqual(tuple(forward.shape), (1, 2, 144, 180)); self.assertEqual(tuple(backward.shape), (1, 2, 144, 180))

    def test_drends_prediction_depth_range_is_symmetric_and_keeps_gt_support_fixed(self):
        product_mm = 1222.67
        disparity = np.array([[0.0, 1e-6, product_mm / 2000.0, product_mm / 0.5]])
        raw = drends._prediction_depth_mm(disparity, product_mm)
        refined = drends._prediction_depth_mm(disparity, product_mm)
        np.testing.assert_allclose(raw[:, 1:], [[1000.0, 1000.0, 1.0]])
        self.assertTrue(np.isnan(raw[0, 0]))
        np.testing.assert_equal(raw, refined)
        report = evaluate_argos_prediction(raw_depth=raw, refined_depth=refined,
            gt_depth=np.full_like(raw, 500.0), gt_valid=np.ones_like(raw, bool),
            protocol_mask=np.ones_like(raw, bool), config=MetricConfig())
        metrics = report["spatial"]["depth_mm"]["raw"]["depth_mm"]["prediction"]
        self.assertEqual(metrics["MAE"]["support_count"], 4)
        self.assertEqual(metrics["InvalidRate"]["value"], 0.25)

    def test_drends_depth_ignores_masked_nan_before_resize(self):
        depth = np.array([[100.0, np.nan], [0.0, 200.0]], dtype=np.float32)
        mask = np.array([[1, 0], [1, 1]], dtype=np.uint8)
        cv2 = SimpleNamespace(IMREAD_UNCHANGED=-1, INTER_AREA=3,
            imread=lambda path, _: depth if path.endswith("depth") else mask,
            resize=lambda value, size, interpolation: value)
        with patch.dict(sys.modules, {"cv2": cv2}):
            result, valid, _ = drends._depth(Path("depth"), Path("mask"), 1.0)
        np.testing.assert_allclose(result, [[100.0, 0.0], [0.0, 200.0]])
        self.assertFalse(valid[0, 1])

    def test_drends_flow_accepts_driver_rgb_only_past_frame(self):
        import torch

        class FakeAdapter:
            def start(self, frame):
                return {"disparity": frame["raw"], "support": frame["raw_valid"], "reset": True, "state_age": 0, "diagnostics": {}}
            def step(self, frame):
                return {"disparity": frame["raw"], "support": frame["raw_valid"], "reset": frame["reanchor"], "state_age": frame["state_age"], "diagnostics": {}}

        class FakeFlow:
            def current_to_anchor(self, current, past):
                self.past_keys = tuple(past.shape); return torch.zeros((1, 2, 144, 180))
            def anchor_to_current(self, past, current):
                return torch.zeros((1, 2, 144, 180))

        raw = torch.ones((1, 1, 144, 180)); rgb = torch.zeros((1, 3, 144, 180))
        frames = [{"index": index, "raw": raw, "raw_valid": raw.bool(), "rgb": rgb, "right_rgb": rgb} for index in range(2)]
        flow = FakeFlow()
        outputs = dict(drive(FakeAdapter(), frames, lambda current, past: drends._drends_flow(flow, current, past)))
        self.assertEqual(flow.past_keys, (1, 3, 144, 180)); self.assertEqual(set(outputs), {0, 1})

    def test_drends_native_stereo_is_loaded_predicted_and_discarded_per_frame(self):
        import numpy as np
        events = []
        records = [{"_rect_left": Path(f"left-{index}"), "_rect_right": Path(f"right-{index}")} for index in range(3)]

        def read(path, size):
            image = np.zeros((720, 1280, 3), dtype=np.uint8)
            events.append(str(path))
            return image

        def predict(left, right):
            events.append("predict")
            return (np.ones((720, 1280), dtype=np.float32),)

        def canonical(index, left, right, raw, *, device):
            events.append(f"canonical-{index}")
            return {"index": index}

        with patch.object(drends, "_rgb", side_effect=read), patch.object(drends, "_canonical_frame", side_effect=canonical):
            frames, shape = drends._canonical_frames(records, predict, device="cpu")
        self.assertEqual(shape, (720, 1280))
        self.assertEqual([frame["index"] for frame in frames], [0, 1, 2])
        self.assertEqual(events, ["left-0", "right-0", "predict", "canonical-0", "left-1", "right-1", "predict", "canonical-1", "left-2", "right-2", "predict", "canonical-2"])

    def test_final_manifest_hashes_match_durable_rewritten_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); output = root / "metric.csv"; output.write_text("source=/old/stage\n")
            (root / "run_manifest.json").write_text(json.dumps({"project": "ARGOS v2", "status": "COMPLETE", "output_hashes": {"metric.csv": "stale"}}))
            output.write_text("source=/published/output\n")
            finalize_output_hashes(root)
            manifest = json.loads((root / "run_manifest.json").read_text())
            self.assertEqual(manifest["output_hashes"]["metric.csv"], hashlib.sha256(output.read_bytes()).hexdigest())
            self.assertEqual(manifest["outputs"], ["metric.csv"])

    def test_d4d_and_serv_wide_rows_keep_no_reference_and_not_applicable(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary)
            fields = ["record_level", "backbone", "protocol", "metric_scope", "section", "metric_path", "method", "statistic", "value"]
            with (stage / "d4d_no_reference_summary.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
                writer.writerow({"record_level": "backbone", "backbone": "RAFT-Stereo", "protocol": "causal", "metric_scope": "no_reference_prediction_space", "section": "d4d", "metric_path": "update", "method": "refined", "statistic": "equal_frame_mean", "value": "1.0"})
            rows = build_wide_rows(stage, {"module": "m"}, ("d4d", "servct"), {"servct": {"backbones": ["StereoAnywhere"]}})
            d4d = next(row for row in rows if row["dataset"] == "D4D")
            serv = next(row for row in rows if row["dataset"] == "SERV-CT")
            self.assertEqual(d4d["metric_scope"], "no_reference_prediction_space")
            self.assertEqual(serv["temporal_applicability"], "NOT_APPLICABLE")

    def test_frozen_sources_remain_unchanged(self):
        expected = {
            "model_design/comparison/definitive_evaluation.py": "cefde2d5ce27f4a3d77df9fcaaee01f5bf79f3ca54931414fa30f43641c80432",
            "model_design/comparison/run_comparison.py": "f462edda4e5aedc9806295dbd9fa46baf4e950a0ba844fa78135c39e5414b31e",
            "model_design/comparison/build_paper_table.py": "7c37000f5720680de2bb624efdf60e208f394146187ae0c5cb1bb9f3dcc46c45",
            "model_design/metrics/unified_metrics.py": "1142e53f5f6865343ca2723f125789e1c592b68278751e177517b33add10139a",
            "model_design/checkpoints/codd_style_h4_best_validation.pt": "99c5745c164fd4903b8aa8acf8f57efccacd9cdcd0d4ed4305cd10609324d725",
            "model_design/checkpoints/codd_style_h4_policy.json": "3f20f0cf628e983990a3aa3b30189a218b628dc9a3b85b86c177844a12e2285d",
        }
        for relative, digest in expected.items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), digest, relative)


if __name__ == "__main__":
    unittest.main()
