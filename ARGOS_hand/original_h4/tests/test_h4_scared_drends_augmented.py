"""CPU contracts for the locked mixed-dataset H4 experiment."""
from __future__ import annotations
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("mixed_runner", ROOT / "scripts/run_h4_scared_drends_augmented.py")
runner = importlib.util.module_from_spec(SPEC); assert SPEC.loader; SPEC.loader.exec_module(runner)
from model_design.comparison.drends_evaluation import _canonical_disparity, load_drends_records  # noqa: E402
from model_design.data import drends_temporal_dataset as drends  # noqa: E402
from model_design.data.drends_temporal_dataset import build_raft_cache  # noqa: E402


class _Record:
    def __init__(self, backbone): self.backbone = backbone
class _Data:
    def __init__(self, records): self.records = records
    def __len__(self): return len(self.records)


class _DrendsClip(Dataset):
    def __len__(self): return 1
    def __getitem__(self, index):
        value = torch.full((4, 1, 1, 1), 2.0)
        return {"raw": value, "gt": torch.ones_like(value), "gt_depth_mm": torch.full((4, 1, 1, 1), 5.0),
                "focal_baseline_mm": torch.full((4,), 10.0), "gt_valid": torch.ones_like(value, dtype=torch.bool)}


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

    def test_real_scared_manifest_and_construction_work_from_clean_cwd(self):
        code = """
import importlib.util, os, sys
from pathlib import Path
root = Path(sys.argv[1])
os.chdir(sys.argv[2])
sys.path.insert(0, str(root))
spec = importlib.util.spec_from_file_location('mixed_clean', root / 'scripts/run_h4_scared_drends_augmented.py')
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
from argos_v2.scared_c_data import load_sequence_info
info = load_sequence_info('dataset_1_keyframe_2')
config = type('Config', (), {'clip_length': 4, 'coverage_threshold': .5, 'seed': 0})()
data = runner.make_scared(('dataset_1_keyframe_2',), config)
assert info.frame_ids and len(data) > 0
"""
        with tempfile.TemporaryDirectory() as temporary:
            subprocess.run([sys.executable, "-c", code, str(ROOT), temporary], cwd=temporary, env={**__import__("os").environ, "PYTHONPATH": ""}, check=True)

    def test_fresh_run_refuses_resume_collision_before_cache_or_dataset_open(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "run"; (out / "checkpoints").mkdir(parents=True); (out / "checkpoints/last.pt").write_bytes(b"collision")
            c = runner.arguments.__globals__["argparse"].Namespace(mode="train", output=out, cache_root=Path(temporary) / "cache", device="cpu", seed=0, epochs=150, patience=10, batch_size=32, workers=20, preload_workers=20, learning_rate=2e-4, weight_decay=1e-4, clip_length=4, coverage_threshold=.5, tau_reset_native_px=5., tau_fusion_native_px=1., alpha_reg=.2, memory_state="recurrent", disable_learned_stereo_evidence=False)
            with self.assertRaises(FileExistsError): runner.train(c)

    def test_cache_publishes_under_backbone_parent_and_rejects_corrupt_reuse(self):
        records = [{"frame_id": "f0", "_rect_left": "left", "_rect_right": "right"}]
        info = {"manifest_sha256": "manifest", "quality_sha256": "quality", "excluded_timing_frame_ids": []}
        native = np.ones((1, 720, 1280), np.float32)
        with tempfile.TemporaryDirectory() as temporary, patch.object(drends, "load_drends_records", return_value=(records, info)), patch.object(drends, "_load_raft", return_value=("fake", lambda left, right: native)), patch.object(drends, "_rgb", return_value=np.zeros((720, 1280, 3), np.uint8)):
            root = Path(temporary) / "cache"
            report = build_raft_cache(root, ("Vid10_Liver_Med",), "cpu")
            target = drends.cache_path(root, "Vid10_Liver_Med")
            self.assertEqual(report["Vid10_Liver_Med"]["status"], "built")
            self.assertTrue((target / ".complete").is_file())
            self.assertTrue((target / "disparity.npy").is_file())
            np.save(target / "disparity.npy", np.ones((1, 1, 1), np.float16))
            with self.assertRaisesRegex(RuntimeError, "corrupt or mismatched"):
                build_raft_cache(root, ("Vid10_Liver_Med",), "cpu")

    def test_dataloader_frame_and_drends_rmse_accept_tensor_focal_baseline(self):
        batch = next(iter(DataLoader(_DrendsClip(), batch_size=1)))
        self.assertTrue(torch.is_tensor(batch["focal_baseline_mm"]))
        self.assertEqual(tuple(runner.frame(batch, 0)["focal_baseline_mm"].shape), (1,))
        class Model(torch.nn.Module):
            def __init__(self): super().__init__(); self.weight = torch.nn.Parameter(torch.zeros(()))
        def fake_run_clip(model, extractor, adapter, clip, c, training):
            item = runner.frame(clip, 0)
            return torch.zeros(()), [{"item": item, "output": SimpleNamespace(fused_disparity=item["raw"]), "raw_memory_valid": torch.ones_like(item["raw"], dtype=torch.bool)}]
        with patch.object(runner, "run_clip", side_effect=fake_run_clip):
            result = runner.evaluate_domain(Model(), None, None, _DrendsClip(), SimpleNamespace(workers=0), "drends")
        self.assertEqual(result["raw_depth_rmse_mm"], 0.0)
        self.assertEqual(result["fused_depth_rmse_mm"], 0.0)

    def test_drends_depth_matches_definitive_range_and_invalid_handling(self):
        class Model(torch.nn.Module):
            def __init__(self): super().__init__(); self.weight = torch.nn.Parameter(torch.zeros(()))
        class Clip(Dataset):
            def __len__(self): return 1
            def __getitem__(self, index):
                value = torch.full((4, 1, 1, 4), 2.0)
                return {"raw": value, "gt": torch.ones_like(value), "gt_depth_mm": torch.full_like(value, 5.0), "focal_baseline_mm": torch.full((4,), 10.0), "gt_valid": torch.ones_like(value, dtype=torch.bool)}
        def fake_run_clip(model, extractor, adapter, batch, c, training):
            item = runner.frame(batch, 0); fused = torch.tensor([[[[0.0, 1e-6, .005, 20.0]]]])
            return torch.zeros(()), [{"item": item, "output": SimpleNamespace(fused_disparity=fused), "raw_memory_valid": torch.zeros_like(fused, dtype=torch.bool)}]
        with patch.object(runner, "run_clip", side_effect=fake_run_clip):
            result = runner.evaluate_domain(Model(), None, None, Clip(), SimpleNamespace(workers=0), "drends")
        expected = np.sqrt(np.mean(np.square([10000.0, 995.0, 995.0, -4.0])))
        self.assertAlmostEqual(result["fused_depth_rmse_mm"], expected)
        self.assertEqual(result["valid_count"], 4)

    def test_revalidation_source_binds_epoch_checkpoint_and_snapshot_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary); (out / "checkpoints").mkdir()
            c = runner.arguments.__globals__["argparse"].Namespace(mode="revalidate", output=out, cache_root=out / "cache", device="cpu", seed=0, epochs=150, patience=10, batch_size=32, workers=20, preload_workers=20, learning_rate=2e-4, weight_decay=1e-4, clip_length=4, coverage_threshold=.5, tau_reset_native_px=5., tau_fusion_native_px=1., alpha_reg=.2, memory_state="recurrent", disable_learned_stereo_evidence=False)
            state = {"epoch": 27, "config_identity": runner.identity(c), "split": runner.split(c)}; checkpoint = out / "checkpoints/best_validation.pt"; torch.save(state, checkpoint)
            snapshot = out / "source_snapshot.sha256"; snapshot.write_text("training source snapshot\n")
            with patch.object(runner, "EXPECTED_BEST_CHECKPOINT_SHA256", runner.sha256(checkpoint)), patch.object(runner, "EXPECTED_TRAINING_SOURCE_SNAPSHOT_SHA256", runner.sha256(snapshot)):
                self.assertEqual(runner.revalidation_source(c)[0]["epoch"], 27)
                with patch.object(runner, "EXPECTED_BEST_EPOCH", 28), self.assertRaisesRegex(RuntimeError, "epoch"):
                    runner.revalidation_source(c)
                with patch.object(runner, "EXPECTED_BEST_CHECKPOINT_SHA256", "0" * 64), self.assertRaisesRegex(RuntimeError, "hash"):
                    runner.revalidation_source(c)

    def test_strict_d2_metric_uses_frozen_protocol_mask(self):
        from model_design.comparison import run_comparison
        bundle = {"gt_valid": np.array([[[True, True]]]), "protocol_mask": np.array([[[False, True]]]), "raw_disparity": np.array([[[1., 2.]]]), "refined_disparity": np.array([[[999., 0.]]]), "gt_disparity": np.array([[[1., 1.]]])}
        c = SimpleNamespace(device="cpu", batch_size=1)
        def frozen(config, adapter, sink): sink(bundle); return [], {}
        with patch.object(run_comparison, "_scared", side_effect=frozen):
            result = runner.evaluate_scared_d2(None, None, c)
        self.assertEqual(result["valid_count"], 1)
        self.assertEqual(result["raw_epe"], 1.)
        self.assertEqual(result["fused_epe"], runner.MetricConfig().invalid_penalty_px)
        self.assertEqual(result["protocol"], "paper_d2_strict_all_anchors")

    def test_revalidation_no_go_does_not_publish(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            c = runner.arguments.__globals__["argparse"].Namespace(mode="revalidate", output=out, cache_root=out / "cache", device="cpu", seed=0, epochs=150, patience=10, batch_size=32, workers=20, preload_workers=20, learning_rate=2e-4, weight_decay=1e-4, clip_length=4, coverage_threshold=.5, tau_reset_native_px=5., tau_fusion_native_px=1., alpha_reg=.2, memory_state="recurrent", disable_learned_stereo_evidence=False)
            state, audit = {"cue_channels": 1, "model": {}}, runner.split(c)
            source, snapshot = out / "checkpoint", out / "snapshot"; source.write_bytes(b"checkpoint"); snapshot.write_text("snapshot\n")
            class Fake:
                def to(self, device): return self
                def load_state_dict(self, value): pass
            class Data:
                pairs = SimpleNamespace(preload_frame_data=lambda workers: {})
                def preload_frame_data(self, workers): return {}
            with patch.object(runner, "revalidation_source", return_value=(state, audit, source, snapshot)), patch.object(runner, "DrendsTemporalClipDataset", return_value=Data()), patch.object(runner, "FrozenResNet18Layer1", return_value=Fake()), patch.object(runner, "BiDAFlowInferenceAdapter", return_value=Fake()), patch.object(runner, "CODDStyleFusionHead", return_value=Fake()), patch.object(runner, "evaluate_scared_d2", return_value={"ratio": 1.01}), patch.object(runner, "evaluate_domain", return_value={"ratio": .9, "raw_depth_rmse_mm": 1., "fused_depth_rmse_mm": .9}), patch.object(runner, "publish_final_bundle") as publish:
                with self.assertRaisesRegex(RuntimeError, "NO-GO"):
                    runner.revalidate(c)
            publish.assert_not_called()

    def test_cache_exception_records_failed_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "run"
            c = runner.arguments.__globals__["argparse"].Namespace(mode="train", output=out, cache_root=Path(temporary) / "cache", device="cpu", seed=0, epochs=150, patience=10, batch_size=32, workers=20, preload_workers=20, learning_rate=2e-4, weight_decay=1e-4, clip_length=4, coverage_threshold=.5, tau_reset_native_px=5., tau_fusion_native_px=1., alpha_reg=.2, memory_state="recurrent", disable_learned_stereo_evidence=False)
            with patch.object(runner, "build_raft_cache", side_effect=RuntimeError("cache failure")), self.assertRaisesRegex(RuntimeError, "cache failure"):
                runner.train(c)
            status = json.loads((out / "status.json").read_text())
            self.assertEqual((status["state"], status["phase"], status["error_type"]), ("failed", "caching", "RuntimeError"))

    def test_main_records_post_cache_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "run"
            config = runner.arguments.__globals__["argparse"].Namespace(mode="train", output=out, cache_root=Path(temporary) / "cache", device="cpu", seed=0, epochs=150, patience=10, batch_size=32, workers=20, preload_workers=20, learning_rate=2e-4, weight_decay=1e-4, clip_length=4, coverage_threshold=.5, tau_reset_native_px=5., tau_fusion_native_px=1., alpha_reg=.2, memory_state="recurrent", disable_learned_stereo_evidence=False)
            with patch.object(runner, "arguments", return_value=config), patch.object(runner, "train", side_effect=RuntimeError("setup failure")), patch.object(sys, "argv", ["runner"]), self.assertRaisesRegex(RuntimeError, "setup failure"):
                runner.main()
            status = json.loads((out / "status.json").read_text())
            self.assertEqual((status["state"], status["phase"], status["error_type"]), ("failed", "post_cache", "RuntimeError"))

    def test_final_bundle_is_published_together(self):
        with tempfile.TemporaryDirectory() as temporary:
            out, final = Path(temporary) / "run", Path(temporary) / "final"
            (out / "checkpoints").mkdir(parents=True)
            torch.save({"state": "test"}, out / "checkpoints/best_validation.pt")
            (out / "training_history.csv").write_text("epoch\n1\n")
            c = SimpleNamespace(output=out)
            runner.publish_final_bundle(c, {"split": "test"}, {"gate": {"passed": True}}, final)
            self.assertTrue(all((final / name).is_file() for name in ("best_validation.pt", "provenance.json", "configuration.json", "split_audit.json", "training_history.csv")))
            self.assertEqual(json.loads((final / "provenance.json").read_text())["configuration_identity"], runner.identity(c))


if __name__ == "__main__": unittest.main()
