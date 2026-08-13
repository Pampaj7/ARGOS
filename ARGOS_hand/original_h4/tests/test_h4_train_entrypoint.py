"""CPU-only checks for the canonical H4 training entrypoint."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model_design"
sys.path.insert(0, str(MODEL))
import train  # noqa: E402


class H4TrainingEntrypointTest(unittest.TestCase):
    def test_dry_run_is_stdlib_only_and_reports_locked_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "never-created"
            run = subprocess.run([sys.executable, str(MODEL / "train.py"), "--dry-run", "--output", str(output)],
                                 check=True, capture_output=True, text=True, env={"PATH": os.environ["PATH"]})
            report = json.loads(run.stdout)
            self.assertFalse(output.exists())
            self.assertFalse(report["torch_imported"])
            self.assertEqual(report["locked_training"]["train_dataset_ids"], [1, 3, 6])
            self.assertEqual(report["locked_training"]["validation_dataset_ids"], [2])
            self.assertEqual(report["locked_training"]["epochs"], 12)
            self.assertEqual(report["locked_training"]["batch_size"], 4)
            self.assertEqual(report["locked_training"]["learning_rate"], 0.0002)
            self.assertEqual(report["locked_training"]["memory_state"], "recurrent")
            self.assertEqual(report["device"], "cuda:0 (logical device after the single-GPU remap)")

    def test_actual_launch_is_fail_closed_before_torch_import(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "CUDA_VISIBLE_DEVICES"):
                train.validate_cuda()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "not-created"
            run = subprocess.run([sys.executable, str(MODEL / "train.py"), "--output", str(output)],
                                 capture_output=True, text=True, env={"PATH": os.environ["PATH"]})
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("CUDA_VISIBLE_DEVICES", run.stderr)
            self.assertFalse(output.exists())

    def test_output_protects_canonical_artifacts_and_no_resume_existing_directory(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "canonical checkpoint"):
            train.validate_output(train.FROZEN_CHECKPOINT, resume=True)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "output exists"):
                train.validate_output(Path(temporary), resume=False)

    def test_provenance_locks_config_and_source_hashes_without_geometry_config(self) -> None:
        value = train.metadata()
        train.verify_sources(value)
        self.assertEqual(hashlib.sha256(train.FROZEN_CHECKPOINT.read_bytes()).hexdigest(), value["canonical_checkpoint"]["sha256"])
        self.assertEqual(value["locked_training"]["optimizer"], "AdamW")
        self.assertIsNone(value["locked_training"]["scheduler"])
        self.assertNotIn("geometry", json.dumps(value).lower())
        source = (MODEL / "train.py").read_text().lower()
        self.assertNotIn("geometry", source)


if __name__ == "__main__":
    unittest.main()
