import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_design.comparison.run_definitive_evaluation import compile_results, scalar_metric_rows


SOURCE = ROOT.parent / "results/definitive_temporal_evaluation"


class DefinitiveComparisonExportTest(unittest.TestCase):
    def test_scalar_export_preserves_aggregate_metadata_and_boolean(self):
        rows = scalar_metric_rows({"raw": {"EPE": {"macro_sequence": 1.0, "median_sequence": 0.9, "P95_sequence": 1.5,
                                                       "worst_sequence": 2.0, "higher_is_better": False, "frame_pair_count": 7}}}, {})
        values = {(row["metric_path"], row["statistic"]): row["value"] for row in rows}
        self.assertEqual(values[("raw/EPE", "median_sequence")], 0.9)
        self.assertEqual(values[("raw/EPE", "P95_sequence")], 1.5)
        self.assertEqual(values[("raw/EPE", "worst_sequence")], 2.0)
        self.assertEqual(values[("raw/EPE", "higher_is_better")], False)
        self.assertEqual(values[("raw/EPE", "frame_pair_count")], 7.0)

    def test_compile_existing_canonical_d4d_and_servct_has_no_false_gt_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "compiled"
            compile_results(SOURCE, output, ("d4d", "servct"))
            with (output / "d4d_no_reference_metrics.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(rows)
            self.assertTrue(all(row["dataset"] == "D4D" and row["metric_scope"] == "no_reference_prediction_space" for row in rows))
            self.assertTrue(all("EPE" not in row["metric_path"] for row in rows))
            self.assertTrue({"specimen", "session", "window_index", "step_since_reset", "reset"} <= set(rows[0]))
            with (output / "d4d_no_reference_summary.csv").open(newline="") as stream:
                summaries = list(csv.DictReader(stream))
            self.assertTrue(summaries)
            self.assertEqual({"sequence", "session", "specimen", "backbone", "dataset_equal_frame_diagnostic"}, {row["record_level"] for row in summaries})
            self.assertTrue(all(row["metric_scope"] == "no_reference_prediction_space" for row in summaries))
            with (output / "applicability.csv").open(newline="") as stream:
                applicability = {row["dataset"]: row["applicability"] for row in csv.DictReader(stream)}
            self.assertEqual(applicability, {"D4D": "NOT_APPLICABLE", "SERV-CT": "NOT_APPLICABLE"})

    def test_compile_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "compiled"; output.mkdir()
            with self.assertRaises(FileExistsError):
                compile_results(SOURCE, output, ("d4d",))


if __name__ == "__main__":
    unittest.main()
