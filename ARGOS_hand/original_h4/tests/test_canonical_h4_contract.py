"""Small CPU-only contract checks for the frozen winning H4 baseline."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V2 = Path("/dtu/p1/leopam/ARGOS/ARGOS-V2")
EXPECTED = "99c5745c164fd4903b8aa8acf8f57efccacd9cdcd0d4ed4305cd10609324d725"
SOURCE = "4b1a22c638214371055012087f93e9faf8a3b087a40a0b969341f33e05187a49"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from model_design.models.codd_bounded_memory import BoundedMemoryPolicy, advance_state_age  # noqa: E402
from frozen_transfer_eval import check_frozen_inputs, validate_cuda_assignment  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CanonicalH4ContractTest(unittest.TestCase):
    def test_checkpoint_and_model_are_exact_canonical_artifacts(self) -> None:
        self.assertEqual(digest(ROOT / "model_design/checkpoints/codd_style_h4_best_validation.pt"), EXPECTED)
        self.assertEqual(digest(ROOT / "model_design/models/codd_style_fusion.py"), SOURCE)
        self.assertEqual(digest(V2 / "model_design/models/codd_style_fusion.py"), SOURCE)
        self.assertEqual(
            check_frozen_inputs(
                ROOT / "model_design/checkpoints/codd_style_h4_best_validation.pt",
                ROOT / "model_design/checkpoints/codd_style_h4_policy.json",
            )["checkpoint_sha256"],
            EXPECTED,
        )
        with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
            check_frozen_inputs(ROOT / "model_design/models/codd_style_fusion.py", ROOT / "model_design/checkpoints/codd_style_h4_policy.json")
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "policy.json"
            copied.write_text((ROOT / "model_design/checkpoints/codd_style_h4_policy.json").read_text())
            with self.assertRaisesRegex(RuntimeError, "policy path mismatch"):
                check_frozen_inputs(ROOT / "model_design/checkpoints/codd_style_h4_best_validation.pt", copied)

    def test_policy_is_dataset2_selected_soft_fixed_h4(self) -> None:
        policy = json.loads((ROOT / "model_design/checkpoints/codd_style_h4_policy.json").read_text())
        self.assertEqual(policy["selection_split"], "dataset_2_validation")
        self.assertEqual(policy["checkpoint_sha256"], EXPECTED)
        self.assertEqual(policy["hard_threshold"], None)
        self.assertEqual(policy["policy"], BoundedMemoryPolicy(name="fixed_h4", max_age=4).to_dict())

    def test_h4_recurrence_uses_only_preceding_raw_or_fused_state(self) -> None:
        policy = BoundedMemoryPolicy(name="fixed_h4", max_age=4)
        raw = [10, 20, 30, 40, 50, 60]
        state = None
        age = 0
        memories = []
        for current in range(1, len(raw)):
            reset = state is None or policy.pre_reset(age=age, accumulated_update=0.0)
            memory = raw[current - 1] if reset else state
            memories.append(memory)
            state = memory + 1  # deterministic stand-in for the frozen fused output
            age = advance_state_age(age, reset=reset)
        self.assertEqual(memories, [10, 11, 12, 13, 50])
        self.assertFalse(policy.pre_reset(age=3, accumulated_update=0.0))
        self.assertTrue(policy.pre_reset(age=4, accumulated_update=0.0))

    def test_tiny_scope_preserves_requested_unseen_backbone(self) -> None:
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("requires the ARGOS PyTorch environment; static invariant is covered below")
        from run_codd_style_bounded_memory_validation import evaluation_scope
        sequences, backbones, pairs = evaluation_scope(SimpleNamespace(
            split="validation", sequences=("dataset_2_keyframe_2",), backbones=("CREStereo",), tiny=True,
        ))
        self.assertEqual(sequences, ("dataset_2_keyframe_2",))
        self.assertEqual(backbones, ("CREStereo",))
        self.assertEqual(pairs, 4)

    def test_direct_runner_accepts_only_manifest_verified_policy(self) -> None:
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("requires the ARGOS PyTorch environment")
        from run_codd_style_bounded_memory_validation import CANONICAL_CHECKPOINT, CANONICAL_POLICY, policy_from_args
        policy, hard_threshold = policy_from_args(SimpleNamespace(
            checkpoint=CANONICAL_CHECKPOINT, frozen_policy=CANONICAL_POLICY,
        ))
        self.assertEqual(policy, BoundedMemoryPolicy(name="fixed_h4", max_age=4))
        self.assertIsNone(hard_threshold)

    def test_cuda_assignment_requires_single_gpu_remap(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "CUDA_VISIBLE_DEVICES"):
                validate_cuda_assignment("cuda:0")
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "1"}, clear=True):
            self.assertEqual(validate_cuda_assignment("cuda:0"), "1")
            with self.assertRaisesRegex(RuntimeError, "logical --device cuda:0"):
                validate_cuda_assignment("cuda:1")

    def test_transfer_evaluator_is_raw_plus_canonical_h4_only(self) -> None:
        source = (ROOT / "scripts/frozen_transfer_eval.py").read_text()
        runner = (ROOT / "scripts/run_codd_style_bounded_memory_validation.py").read_text()
        self.assertIn('"canonical_bounded_h4"', source)
        self.assertIn('"method": "raw"', source)
        self.assertIn('default="cuda:0"', source)
        self.assertIn('CUDA_VISIBLE_DEVICES', source)
        self.assertIn('PAPER_PROTOCOL = "paper_d2_strict_all_anchors"', source)
        self.assertIn('STRICT_SUPPORT = "GT coverage > 0.5 & raw valid & H4 support & all CS1/2/4/8 aligned-valid & warp support"', source)
        self.assertIn('sequences, backbones, max_pairs = (sequences[0],), (backbones[0],), 4', runner)
        self.assertIn('memory_state != "recurrent"', runner)
        self.assertIn('disable_learned_stereo_evidence', runner)
        self.assertNotIn("RawMultiAnchor" + "Refiner", source)
        self.assertNotIn("retrieve_and_" + "fuse", source)
        self.assertNotIn("argos_" + "frozen", source)
        manifest = (ROOT / "model_design/checkpoints/inference_manifest.json").read_text()
        self.assertNotIn('"bounded_h4_runner"', manifest)
        self.assertIn('"fusion_probe_helpers"', manifest)
        self.assertIn('"scared_c_paths"', manifest)
        self.assertIn('"runtime_sources"', source)


if __name__ == "__main__":
    unittest.main()
