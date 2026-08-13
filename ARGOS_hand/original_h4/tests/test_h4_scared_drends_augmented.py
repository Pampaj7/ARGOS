"""CPU contracts for the locked mixed-dataset H4 experiment."""
from __future__ import annotations
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("mixed_runner", ROOT / "scripts/run_h4_scared_drends_augmented.py")
runner = importlib.util.module_from_spec(SPEC); assert SPEC.loader; SPEC.loader.exec_module(runner)
from model_design.comparison.drends_evaluation import _canonical_disparity, load_drends_records  # noqa: E402
from model_design.data.drends_temporal_dataset import build_raft_cache  # noqa: E402


class _Record:
    def __init__(self, backbone): self.backbone = backbone
class _Data:
    def __init__(self, records): self.records = records
    def __len__(self): return len(self.records)


class MixedH4ContractTest(unittest.TestCase):
    def test_split_scale_timing_and_heldout_are_locked(self):
        audit = runner.split(None)
        self.assertNotIn("Vid14_Pancreas_High", audit["drends_train"] + audit["drends_validation"])
        self.assertFalse(set(audit["scared_train"]) & set(audit["scared_validation"]))
        self.assertFalse(set(audit["drends_train"]) & set(audit["drends_validation"]))
        self.assertAlmostEqual(_canonical_disparity(np.full((720,1280), 1280., np.float32), 180/1280).mean(), 180., places=3)
        records, info = load_drends_records("Vid10_Liver_Med", max_frames=6)
        self.assertTrue(records); self.assertTrue(all(max(row["left_right_offset_ms"], row["left_helios_offset_ms"], row["right_helios_offset_ms"]) <= 100 for row in records)); self.assertIn("manifest_sha256", info)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError): build_raft_cache(Path(temporary), ("Vid14_Pancreas_High",), "cpu")

    def test_fixed_sampler_is_75_25_and_deterministic(self):
        scared = _Data([_Record(name) for name in runner.SEEN_BACKBONES for _ in range(9)])
        drends = _Data([_Record("RAFT-Stereo") for _ in range(9)])
        a, b = runner.FixedMixedBatchSampler(scared, drends, steps=2), runner.FixedMixedBatchSampler(scared, drends, steps=2)
        first, second = list(a), list(b); self.assertEqual(first, second)
        for batch in first:
            self.assertEqual(len(batch), 32); self.assertEqual(sum(i >= len(scared) for i in batch), 8)
            self.assertEqual([sum(scared.records[i].backbone == name for i in batch[:24]) for name in runner.SEEN_BACKBONES], [8,8,8])

    def test_macro_and_cpu_dry_run(self):
        self.assertEqual(runner.validation_macro({"ratio": .8}, {"ratio": 1.2}), 1.0)
        result = subprocess.run([sys.executable, str(ROOT / "scripts/run_h4_scared_drends_augmented.py"), "--mode", "dry-run", "--device", "cpu"], cwd=ROOT, env={**__import__("os").environ, "PYTHONPATH": str(ROOT)}, capture_output=True, text=True, check=True)
        payload = json.loads(result.stdout); self.assertEqual(payload["sampling"]["steps"], 252)

    def test_fresh_run_refuses_resume_collision_before_cache_or_dataset_open(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "run"; (out / "checkpoints").mkdir(parents=True); (out / "checkpoints/last.pt").write_bytes(b"collision")
            c = runner.arguments.__globals__["argparse"].Namespace(mode="train", output=out, cache_root=Path(temporary) / "cache", device="cpu", seed=0, epochs=150, patience=10, batch_size=32, workers=20, preload_workers=20, learning_rate=2e-4, weight_decay=1e-4, clip_length=4, coverage_threshold=.5, tau_reset_native_px=5., tau_fusion_native_px=1., alpha_reg=.2, memory_state="recurrent", disable_learned_stereo_evidence=False)
            with self.assertRaises(FileExistsError): runner.train(c)


if __name__ == "__main__": unittest.main()
