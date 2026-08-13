"""CPU-only contract tests for the one augmented H4 training profile."""
from __future__ import annotations

import argparse
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("h4_augmented_runner", ROOT / "scripts/run_h4_augmented_fusion_probe.py")
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def config(**changes):
    value = dict(mode="train", profile="h4_augmented", output=Path("/tmp/h4_augmented"), seed=0, device="cpu", epochs=150,
                 overfit_epochs=100, batch_size=32, workers=20, preload_workers=20, learning_rate=2e-4, weight_decay=1e-4,
                 clip_length=4, max_clips_per_sequence=None, patience=10, resume=True, coverage_threshold=.50,
                 tau_reset_native_px=5.0, tau_fusion_native_px=1.0, alpha_reg=.2, memory_state="recurrent",
                 disable_learned_stereo_evidence=False, dry_run=False)
    value.update(changes)
    return argparse.Namespace(**value)


class H4AugmentedTrainingTest(unittest.TestCase):
    def test_split_patience_and_resume_contract(self) -> None:
        value = config()
        runner.validate_config(value)
        self.assertEqual(value.batch_size, 32)
        train, validation, test = runner.split_for(value)
        self.assertEqual(train[-2:], ("dataset_2_keyframe_2", "dataset_2_keyframe_3"))
        self.assertEqual(validation, ("dataset_2_keyframe_4",))
        self.assertFalse(set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test))
        self.assertFalse(runner.should_stop(10, 9)); self.assertTrue(runner.should_stop(10, 10))
        split = {"train": list(train), "validation": list(validation)}
        state = {"split": split, "config_identity": runner.config_identity(value)}
        self.assertTrue(runner.resume_compatible(state, split, value))
        self.assertFalse(runner.resume_compatible(state, split, config(seed=1)))

    def test_legacy_config_without_new_fields_still_validates(self) -> None:
        runner.validate_config(argparse.Namespace())


if __name__ == "__main__":
    unittest.main()
