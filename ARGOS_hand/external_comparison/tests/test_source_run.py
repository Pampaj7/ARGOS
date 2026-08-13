import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bridge import read_output_snapshot, write_input, write_output
import package_source_run
import compile_external_results
import run_bidastabilizer_d2_full


class SourceRunPackageTest(unittest.TestCase):
    def test_final_compiled_manifest_has_no_incomplete_paths(self):
        compiled = ROOT / "results/bidastabilizer_raftstereo_robust/d2_full/compiled_test_only"
        if not compiled.is_dir():
            self.skipTest("full D2 artifact is not present")
        run_bidastabilizer_d2_full.validate_compiled_manifest(compiled)

    def test_compile_rejects_publishable_run_without_bridge_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "sources/scared-d2/nvds_plus_forward_clip4"; run.mkdir(parents=True)
            external = {"method": run.name, "publication": "PUBLISHABLE", "execution_manifest_id": "ok",
                        "execution_manifest_sha256": "e" * 64, "input_sha256": "a" * 64, "prediction_sha256": "b" * 64}
            (run / "external_method.json").write_text(json.dumps(external))
            (run / "run_manifest.json").write_text(json.dumps({"publication": "PUBLISHABLE", "execution_manifest_id": "ok",
                                                                  "execution_manifest_sha256": "e" * 64, "external_method": external}))
            with patch.object(compile_external_results, "_execution_manifest", return_value={"sha256": "e" * 64}):
                with self.assertRaisesRegex(ValueError, "source bridge provenance mismatch"):
                    compile_external_results.validate(run.parents[1], ["scared-d2"])

    def test_output_rejects_wrong_source_snapshot_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); input_path = root / "input.npz"; prediction = root / "prediction.npz"
            values = {"rgb_left": np.ones((2, 3, 4, 4), np.float32), "rgb_right": np.ones((2, 3, 4, 4), np.float32),
                      "raw_disparity": np.full((2, 1, 4, 4), 2, np.float32), "raw_valid": np.ones((2, 1, 4, 4), bool),
                      "frame_ids": np.array(["frame-0", "frame-1"])}
            info = write_input(input_path, values)
            write_output(prediction, values["raw_disparity"], values, info, "nvds_plus_forward_clip4")
            meta = json.loads(prediction.with_suffix(".json").read_text()); meta["source_rgb_input_sha256"] = "0" * 64
            prediction.with_suffix(".json").write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "does not match"):
                read_output_snapshot(prediction, values, info, "nvds_plus_forward_clip4")

    def test_compile_rejects_forged_publishable_attestation_before_subprocess(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); run = root / "sources/scared-d2/nvds_plus_forward_clip4"
            run.mkdir(parents=True)
            external = {"method": run.name, "publication": "PUBLISHABLE", "execution_manifest_id": "forged",
                        "execution_manifest_sha256": "f" * 64, "input_sha256": "a" * 64, "source_input_sha256": "a" * 64,
                        "rgb_input_sha256": "c" * 64, "source_rgb_input_sha256": "c" * 64, "frame_ids": ["frame-0"],
                        "prediction_sha256": "b" * 64}
            (run / "external_method.json").write_text(json.dumps(external))
            (run / "run_manifest.json").write_text(json.dumps({"publication": "PUBLISHABLE", "execution_manifest_id": "forged",
                                                                  "execution_manifest_sha256": "f" * 64, "external_method": external}))
            command = ["compile_external_results.py", "--source-root", str(root / "sources"), "--datasets", "scared-d2", "--output", str(root / "published")]
            with patch.object(sys, "argv", command), patch.object(compile_external_results.subprocess, "run") as subprocess_run:
                with self.assertRaisesRegex(ValueError, "unknown execution manifest"):
                    compile_external_results.main()
                subprocess_run.assert_not_called()

    def test_identity_output_compiles_through_frozen_driver(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); input_path = root / "input.npz"; prediction = root / "prediction.npz"
            values = {"rgb_left": np.ones((5, 3, 4, 4), np.float32), "rgb_right": np.ones((5, 3, 4, 4), np.float32),
                      "raw_disparity": np.full((5, 1, 4, 4), 2, np.float32), "raw_valid": np.ones((5, 1, 4, 4), bool),
                      "frame_ids": np.array([f"frame-{index}" for index in range(5)])}
            info = write_input(input_path, values)
            write_output(prediction, values["raw_disparity"].copy(), values, info, "nvds_plus_forward_clip4")
            env = os.environ | {"PYTHONDONTWRITEBYTECODE": "1"}
            subprocess.run([sys.executable, str(ROOT / "package_source_run.py"), "--source-root", str(root / "sources"), "--method", "nvds_plus_forward_clip4", "--input", str(input_path), "--prediction", str(prediction), "--evaluation", "TEST_FIXTURE"], check=True, env=env)
            manifest = json.loads((root / "sources/scared-d2/nvds_plus_forward_clip4/external_method.json").read_text())
            self.assertEqual(manifest["evaluation_artifact_id"], "TEST_FIXTURE")
            self.assertEqual(manifest["publication"], "TEST_ONLY")
            self.assertIsNone(manifest["execution_manifest_id"])
            self.assertIsNone(manifest["execution_manifest_sha256"])
            self.assertEqual(len(manifest["evaluation_npz_sha256"]), 64)
            self.assertEqual(len(manifest["evaluation_json_sha256"]), 64)
            run_manifest = json.loads((root / "sources/scared-d2/nvds_plus_forward_clip4/run_manifest.json").read_text())
            self.assertEqual(run_manifest["publication"], run_manifest["external_method"]["publication"])
            self.assertIsNone(run_manifest["execution_manifest_id"])
            rejected = subprocess.run([sys.executable, str(ROOT / "compile_external_results.py"), "--source-root", str(root / "sources"), "--datasets", "scared-d2", "--output", str(root / "published")], env=env, text=True, capture_output=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("not publishable", rejected.stderr)
            compiled = root / "compiled"
            subprocess.run([sys.executable, str(ROOT.parent / "original_h4/model_design/comparison/run_definitive_evaluation.py"), "--compile-from", str(root / "sources"), "--datasets", "scared-d2", "--output", str(compiled)], check=True, env=env)
            self.assertTrue((compiled / "run_manifest.json").is_file())

    def test_rejects_unknown_or_hash_mismatched_evaluation_artifact(self):
        with self.assertRaisesRegex(ValueError, "unknown evaluation artifact"):
            package_source_run._evaluation_artifact("UNKNOWN", "fixture-input")
        expected_input = json.loads((ROOT / "evaluation_artifacts.lock.json").read_text())["artifacts"]["TEST_FIXTURE"]["input_sha256"]
        with tempfile.TemporaryDirectory() as temporary:
            lock = Path(temporary) / "evaluation_artifacts.lock.json"
            value = json.loads((ROOT / "evaluation_artifacts.lock.json").read_text())
            value["artifacts"]["TEST_FIXTURE"]["npz_sha256"] = "0" * 64
            lock.write_text(json.dumps(value))
            with patch.object(package_source_run, "EVALUATION_LOCK", lock):
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    package_source_run._evaluation_artifact("TEST_FIXTURE", expected_input)

    def test_rejects_publishable_artifact_without_execution_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); input_path = root / "input.npz"; prediction = root / "prediction.npz"
            values = {"rgb_left": np.ones((5, 3, 4, 4), np.float32), "rgb_right": np.ones((5, 3, 4, 4), np.float32),
                      "raw_disparity": np.full((5, 1, 4, 4), 2, np.float32), "raw_valid": np.ones((5, 1, 4, 4), bool),
                      "frame_ids": np.array([f"frame-{index}" for index in range(5)])}
            info = write_input(input_path, values)
            write_output(prediction, values["raw_disparity"], values, info, "nvds_plus_forward_clip4")
            evaluation = package_source_run._evaluation_artifact("TEST_FIXTURE", info["input_sha256"])
            with patch.object(package_source_run, "_evaluation_artifact", return_value=(evaluation[0], evaluation[1], evaluation[2] | {"publication": "PUBLISHABLE"})):
                with self.assertRaisesRegex(ValueError, "execution manifest required"):
                    package_source_run.package(root / "sources", "nvds_plus_forward_clip4", input_path, prediction, "TEST_FIXTURE")

    def test_rejects_execution_manifest_bound_to_wrong_upstream(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "protocols").mkdir()
            protocol = root / "protocols/nvds_plus_forward_clip4.json"
            protocol.write_text("{}")
            execution = root / "execution.json"
            execution.write_text(json.dumps({"method": "nvds_plus_forward_clip4", "protocol_sha256": package_source_run.sha256(protocol),
                                             "upstream": {"name": "bidavideo", "commit": "b"},
                                             "checkpoint": {"id": "nvds_plus", "sha256": "c"},
                                             "input_sha256": "input", "output_prediction_sha256": "prediction"}))
            lock = root / "execution_manifests.lock.json"
            lock.write_text(json.dumps({"manifests": {"bad": {"path": "execution.json", "sha256": package_source_run.sha256(execution)}}}))
            upstreams = root / "upstreams.lock.json"
            upstreams.write_text(json.dumps({"upstreams": {"nvds": {"commit": "n"}, "bidavideo": {"commit": "b"}}}))
            checkpoints = root / "checkpoints.lock.json"
            checkpoints.write_text(json.dumps({"checkpoints": {"nvds_plus": {"status": "READY", "sha256": "c"}}}))
            with patch.object(package_source_run, "ROOT", root), patch.object(package_source_run, "EXECUTION_MANIFEST_LOCK", lock), patch.object(package_source_run, "UPSTREAM_LOCK", upstreams), patch.object(package_source_run, "CHECKPOINT_LOCK", checkpoints):
                with self.assertRaisesRegex(ValueError, "upstream mismatch"):
                    package_source_run._execution_manifest("bad", "nvds_plus_forward_clip4", "input", "prediction")
