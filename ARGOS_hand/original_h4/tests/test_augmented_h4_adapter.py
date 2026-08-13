"""Focused contracts for the promoted augmented H=4 evaluation adapter."""
from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_design.comparison.augmented_h4 import (  # noqa: E402
    CHECKPOINT, CHECKPOINT_SHA256, PARAMETERS, PROVENANCE, PROVENANCE_SHA256,
    AugmentedH4, factory,
)
from model_design.comparison.canonical_h4 import CanonicalH4, factory as canonical_factory  # noqa: E402
from model_design.comparison.run_definitive_evaluation import _wide_metadata  # noqa: E402


class AugmentedH4AdapterTest(unittest.TestCase):
    def test_factory_reuses_canonical_h4_state_machine_and_declares_exact_artifacts(self) -> None:
        self.assertIs(AugmentedH4.start, CanonicalH4.start)
        self.assertIs(AugmentedH4.step, CanonicalH4.step)
        self.assertEqual(hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(), CHECKPOINT_SHA256)
        self.assertEqual(hashlib.sha256(PROVENANCE.read_bytes()).hexdigest(), PROVENANCE_SHA256)
        description = factory(device="cpu").describe()
        self.assertEqual(description["checkpoint"], str(CHECKPOINT))
        self.assertEqual(description["policy"], str(PROVENANCE))
        self.assertEqual(description["training_profile"], "h4_augmented")
        self.assertEqual(description["reset_protocol"], "fixed H=4; raw t-1 after re-anchor; preceding fused otherwise")
        self.assertEqual(description["evaluation_split_roles"], {"SCARED-C/d2": "mixed_train_validation_exposed", "SCARED-C/d7": "heldout_test"})

    def test_augmented_and_canonical_wide_split_metadata(self) -> None:
        augmented = factory(device="cpu").describe()
        d2 = _wide_metadata("SCARED-C", "d2", "RAFT-Stereo", augmented, protocol="p", scope="gt")
        d7 = _wide_metadata("SCARED-C", "d7", "RAFT-Stereo", augmented, protocol="p", scope="gt")
        canonical = _wide_metadata("SCARED-C", "d2", "RAFT-Stereo", canonical_factory(device="cpu").describe(), protocol="p", scope="gt")
        self.assertEqual(d2["dataset_role"], "mixed_train_validation_exposed")
        self.assertTrue(d2["evaluation_split_used_in_training"])
        self.assertEqual(d7["dataset_role"], "heldout_test")
        self.assertFalse(d7["evaluation_split_used_in_training"])
        self.assertEqual(canonical["dataset_role"], "D2_validation")
        self.assertFalse(canonical["evaluation_split_used_in_training"])

    def test_target_environment_load_is_strict_and_frozen(self) -> None:
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("requires the ARGOS PyTorch environment")
        adapter = factory(device="cpu")
        model, extractor, _ = adapter._load()
        self.assertEqual(model.full[0].in_channels, 142)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), PARAMETERS)
        self.assertFalse(model.training)
        self.assertFalse(extractor.training)
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in extractor.parameters()))


if __name__ == "__main__":
    unittest.main()
