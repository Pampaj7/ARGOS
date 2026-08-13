import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from model_design.comparison import build_paper_table as paper


class PaperTableTests(unittest.TestCase):
    def test_equal_sequence_macro_keeps_distinct_micro(self):
        result = paper.aggregate_sequence_metrics([
            {"macro_sequence": 1.0, "support_count": 1, "frame_count": 1},
            {"macro_sequence": 3.0, "support_count": 9, "frame_count": 2},
        ])
        self.assertEqual(result["macro_sequence"], 2.0)
        self.assertEqual(result["micro_pixel"], 2.8)
        self.assertEqual(result["sequence_count"], 2)

    def test_not_applicable_rows_preserve_family_separation(self):
        rows = paper.applicability_rows()
        self.assertEqual({row["dataset"] for row in rows}, {"StereoMIS", "joint unseen-backbone+OOD"})
        self.assertTrue(all(row["applicability"] == "NOT_APPLICABLE" for row in rows))
        self.assertTrue(all(row["metric_family"] == "applicability" for row in rows))
        self.assertTrue(all(row["candidate_value"] is None for row in rows))

    def test_manifest_hash_mismatch_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "scared-d2/canonical_h4/run_manifest.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"status": "COMPLETE", "module_provenance": {"checkpoint_sha256": "wrong"}, "unified_metrics": {"sha256": paper.EXPECTED_UNIFIED}}))
            with patch.object(paper, "RESULTS", root):
                with self.assertRaisesRegex(RuntimeError, "checkpoint hash mismatch"):
                    paper.validate_manifests()

    def test_actual_declared_hash_mismatch_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.txt"; path.write_text("actual")
            with self.assertRaisesRegex(RuntimeError, "artifact hash mismatch"):
                paper.validate_declared_artifact({"path": str(path), "sha256": "0" * 64}, "artifact")

    def test_strict_json_rejects_duplicate_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"x": 1, "x": 2}')
            with self.assertRaisesRegex(ValueError, "duplicate"):
                paper.read_json(path)
            path.write_text('{"x": NaN}')
            with self.assertRaisesRegex(ValueError, "non-finite"):
                paper.read_json(path)

    def test_d2_d7_labels_are_in_domain(self):
        self.assertEqual(paper.scared_context("d2")[:2], ("validation", "in_domain"))
        self.assertEqual(paper.scared_context("d7")[:2], ("heldout_test", "in_domain"))

    def test_d4d_schema_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root / "d4d/canonical_h4/d4d_diagnostics.csv"; path.parent.mkdir(parents=True)
            path.write_text("dataset,backbone\nD4D,RAFT-Stereo\n")
            with patch.object(paper, "RESULTS", root):
                with self.assertRaisesRegex(RuntimeError, "schema"):
                    paper.d4d_rows({})

    def test_serv_contract_is_currently_valid(self):
        rows, shards = paper._serv_contract({})
        self.assertEqual(len(rows), 16)
        self.assertEqual({key: value[0].shape for key, value in shards.items()}, {"honest_test__Experiment_2": (8, 144, 180), "honest_train__Experiment_1": (8, 144, 180)})

    def test_serv_native_gt_scales_to_cache_grid(self):
        rows = [{"width": "720"}]
        factor = paper.serv_cache_factor(rows, {"cache_width": 180, "disparity_units": "pixels_at_cache_resolution", "disparity_convention": "positive_left_disparity"})
        self.assertEqual(factor, .25)
        native_gt, prediction = np.full((1, 1, 1), 72.0), np.full((1, 1, 1), 18.0)
        report = paper.compute_spatial_metrics(prediction, native_gt * factor, np.ones_like(prediction, bool), np.ones_like(prediction, bool), paper.MetricConfig(fx_px=100.0 * factor, baseline_mm=5.0))
        self.assertEqual(report["aggregate"]["disparity_px"]["EPE"]["macro_sequence"], 0.0)
        self.assertEqual((native_gt * factor / factor).item(), 72.0)

    def test_complete_flatten_keeps_representative_sections(self):
        leaf = {"value": 1.0, "support_count": 3, "frame_count": 4, "sequence_count": 1}
        aggregate = {"primary_aggregate": "macro_sequence", "macro_sequence": 1.0, "micro_pixel": 2.0, "support_count": 3, "frame_count": 4, "sequence_count": 1}
        report = {"dataset": "SCARED-C", "backbone": "S2M2-S", "sequence_ids": ["s"], "protocol": "p",
                  "aggregate": {"disparity_px": {"raw": {"EPE": aggregate}}},
                  "spatial": {"disparity_px": {"raw": {"EPE": leaf}}},
                  "safety": {"disparity_px": {"aggregate": {"HUR": aggregate}}},
                  "temporal": {"disparity_px": {"1": {"diagnostic_grid_based": True, "methods": {"raw": {"Drift_px": {"MAE": leaf}}}}}}}
        rows = paper.flatten_scared_reports([report], "d2")
        self.assertEqual({row["section"] for row in rows}, {"aggregate", "spatial", "safety", "temporal"})
        self.assertTrue(any(row["horizon"] == "1" and 'diagnostic_grid_based' in row["diagnostic_tags"] for row in rows))

    def test_staged_failure_cleans_up(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "final"
            with self.assertRaisesRegex(RuntimeError, "boom"):
                paper._staged_directory(output, lambda _: (_ for _ in ()).throw(RuntimeError("boom")))
            self.assertFalse(output.exists())
            self.assertFalse((output.parent / ".final.lock").exists())

    def test_markdown_and_tex_share_rounded_values(self):
        row = paper.empty_row(dataset="SCARED-C", split="d2", backbone="S2M2-S", protocol="H4", metric="EPE", baseline_value=1.23456, candidate_value=1.0, delta=-0.23456, verdict="OBSERVED")
        md, tex = paper.render_markdown([row]), paper.render_tex([row])
        for value in ("1.2346", "1.0000", "-0.2346"):
            self.assertIn(value, md)
            self.assertIn(value, tex)

    def test_non_finite_values_are_refused(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                paper.finite(value)


if __name__ == "__main__":
    unittest.main()
