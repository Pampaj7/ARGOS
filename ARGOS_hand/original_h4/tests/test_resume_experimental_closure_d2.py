from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_design.comparison import experimental_closure as closure
from model_design.comparison import resume_experimental_closure_d2 as recovery


def _report(backbone: str, sequence: str) -> dict:
    metric = {"macro_sequence": 2.0}
    return {"dataset": "SCARED-C", "split": "d2", "backbone": backbone, "sequence_ids": [sequence],
            "protocol": recovery.PROTOCOL, "primary_aggregate": "macro_sequence",
            "aggregate": {"disparity_px": {"raw": {"MAE": metric}, "refined": {"MAE": {"macro_sequence": 1.0}}}}}


class ResumeExperimentalClosureD2Test(unittest.TestCase):
    def _state(self, output: Path, freeze: Path, *, oracle: bool, summary: bool) -> dict:
        required = {"method_report_count": 2, "summary_row_count": 3}
        output.mkdir()
        manifest = {"project": "ARGOS v2", "status": "INCOMPLETE", "dataset": "scared-d2", "freeze": {"path": str(freeze), "sha256": closure.sha256(freeze)},
                    "methods_requested": ["canonical_h4", "fixed_w0.1_h4"], "full_method_grid": ["canonical_h4", "fixed_w0.1_h4"], "scope": required,
                    "output": str(output.resolve()), "CUDA_VISIBLE_DEVICES": "1", "device": "physical cuda:1 remapped to logical cuda:0", "dense_predictions_written": False}
        (output / "run_manifest.json").write_text(json.dumps(manifest))
        for method in ("canonical_h4", "fixed_w0.1_h4"):
            report = _report("backbone", "sequence"); report["method"] = method
            path = recovery._path(output, recovery._target(method, "backbone", "sequence")); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(report))
        if oracle:
            report = _report("backbone", "sequence"); report["diagnostic"] = recovery.ORACLE_DIAGNOSTIC
            path = recovery._path(output, recovery._target(recovery.ORACLE, "backbone", "sequence")); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(report))
        if summary:
            canonical = _report("backbone", "sequence"); canonical["method"] = "canonical_h4"
            diagnostic = _report("backbone", "sequence"); diagnostic["diagnostic"] = recovery.ORACLE_DIAGNOSTIC
            fixed = _report("backbone", "sequence"); fixed["method"] = "fixed_w0.1_h4"
            recovery.comparison.atomic_csv(output / "summary.csv", [closure._row(canonical, "canonical_h4"),
                                                                        closure._row(diagnostic, recovery.ORACLE, diagnostic=True),
                                                                        closure._row(fixed, "fixed_w0.1_h4")])
        return required

    def test_reuses_valid_reports_completes_only_gap_and_refuses_unexpected_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output, freeze = Path(directory) / "d2", Path(directory) / "freeze.json"
            output.mkdir(); freeze.write_text("{}")
            required = {"method_report_count": 2, "summary_row_count": 3}
            manifest = {"project": "ARGOS v2", "status": "INCOMPLETE", "dataset": "scared-d2", "freeze": {"path": str(freeze), "sha256": closure.sha256(freeze)},
                        "methods_requested": ["canonical_h4", "fixed_w0.1_h4"], "full_method_grid": ["canonical_h4", "fixed_w0.1_h4"],
                        "scope": required, "output": str(output.resolve()), "CUDA_VISIBLE_DEVICES": "1",
                        "device": "physical cuda:1 remapped to logical cuda:0", "dense_predictions_written": False}
            (output / "run_manifest.json").write_text(json.dumps(manifest))
            canonical, oracle = _report("backbone", "sequence"), _report("backbone", "sequence")
            canonical["method"] = "canonical_h4"; oracle["diagnostic"] = recovery.ORACLE_DIAGNOSTIC
            for target, report in ((recovery._target("canonical_h4", "backbone", "sequence"), canonical), (recovery._target(recovery.ORACLE, "backbone", "sequence"), oracle)):
                path = recovery._path(output, target); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(report))
            unexpected = output / "unexpected.txt"; unexpected.write_text("no")
            with patch.object(recovery, "D2", output), patch.object(closure, "FREEZE", freeze), patch.object(closure, "FULL_D2_METHODS", ("canonical_h4", "fixed_w0.1_h4")), patch.object(closure, "ALL_BACKBONES", ("backbone",)), patch.object(closure, "_validation_sequences", return_value=("sequence",)), patch.object(closure, "load_freeze", return_value={"required_d2_scope": required}):
                with self.assertRaisesRegex(RuntimeError, "unexpected D2 recovery files"):
                    recovery.audit(output)
                unexpected.unlink()
                calls = []
                class Adapter:
                    def describe(self): return {"adapter": "fake"}
                def fake_scared(config, adapter, sink):
                    calls.append((tuple(config.backbones), tuple(config.sequences)))
                    sink({"backbone": "backbone", "sequence_id": "sequence", "protocol_mask": np.ones((1, 1), bool), "adapter_support": np.ones((1, 1), bool)})
                with patch.object(closure, "_adapter", return_value=Adapter()), patch.object(closure, "_scared", side_effect=fake_scared), patch.object(closure, "evaluate_scared_bundle", side_effect=lambda bundle: _report(bundle["backbone"], bundle["sequence_id"])), patch.object(recovery.comparison, "validate_cuda", return_value="1"):
                    result = recovery.resume(output=output)
            self.assertEqual(calls, [(("backbone",), ("sequence",))])
            self.assertEqual(result, {"reused": 2, "executed": 1, "summary_rows": 3})
            complete = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(complete["status"], "COMPLETE")
            self.assertEqual(complete["recovery"]["reused_normal_reports"], 1)
            self.assertEqual(complete["recovery"]["executed_normal_reports"], 1)

    def test_interrupted_summary_and_oracle_only_gap_are_recoverable(self):
        with tempfile.TemporaryDirectory() as directory:
            root, freeze = Path(directory), Path(directory) / "freeze.json"; freeze.write_text("{}")
            summary_output, oracle_output = root / "summary", root / "oracle"
            summary_required = self._state(summary_output, freeze, oracle=True, summary=True)
            oracle_required = self._state(oracle_output, freeze, oracle=False, summary=False)
            common = (patch.object(closure, "FREEZE", freeze), patch.object(closure, "FULL_D2_METHODS", ("canonical_h4", "fixed_w0.1_h4")),
                      patch.object(closure, "ALL_BACKBONES", ("backbone",)), patch.object(closure, "_validation_sequences", return_value=("sequence",)),
                      patch.object(closure, "load_freeze", side_effect=[{"required_d2_scope": summary_required}, {"required_d2_scope": summary_required},
                                                                          {"required_d2_scope": oracle_required}, {"required_d2_scope": oracle_required}]),
                      patch.object(recovery.comparison, "validate_cuda", return_value="1"))
            with common[0], common[1], common[2], common[3], common[4], common[5]:
                with patch.object(closure, "_scared", side_effect=AssertionError("summary recovery must not infer")):
                    recovery.resume(output=summary_output)
                calls = []
                class Adapter:
                    def describe(self): return {"adapter": "fake"}
                def fake_scared(config, adapter, sink):
                    calls.append((tuple(config.backbones), tuple(config.sequences)))
                    sink({"backbone": "backbone", "sequence_id": "sequence"})
                with patch.object(closure, "_adapter", return_value=Adapter()), patch.object(closure, "_scared", side_effect=fake_scared), patch.object(closure, "_oracle_bundle", side_effect=lambda bundle: bundle), patch.object(closure, "evaluate_scared_bundle", side_effect=lambda bundle: _report(bundle["backbone"], bundle["sequence_id"])):
                    recovery.resume(output=oracle_output)
            self.assertEqual(calls, [(("backbone",), ("sequence",))])
            self.assertTrue(recovery._path(oracle_output, recovery._target(recovery.ORACLE, "backbone", "sequence")).is_file())
            self.assertEqual(json.loads((oracle_output / "run_manifest.json").read_text())["recovery"]["rerun_canonical_for_oracle_reports"], 1)


if __name__ == "__main__":
    unittest.main()
