import json
from pathlib import Path
import unittest
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bridge import (bidastabilizer_to_positive_left, positive_left_to_bidastabilizer,
                    read_input, resize_disparity, rgb_input_sha256, validate_input, write_input)
import workers.bidastabilizer as bidastabilizer


def sample():
    return {"rgb_left": np.ones((2, 3, 2, 3), np.float32), "rgb_right": np.ones((2, 3, 2, 3), np.float32),
            "raw_disparity": np.full((2, 1, 2, 3), 2, np.float32), "raw_valid": np.ones((2, 1, 2, 3), bool),
            "frame_ids": np.array(["a", "b"])}


class ContractTest(unittest.TestCase):
    def test_npz_json_contract_and_hash(self):
        path = Path(self._testMethodName + ".npz")
        try:
            info = write_input(path, sample()); values, loaded = read_input(path)
            self.assertEqual(loaded["input_sha256"], info["input_sha256"])
            self.assertEqual(loaded["rgb_input_sha256"], rgb_input_sha256(values))
            self.assertEqual(values["rgb_left"].dtype, np.float32)
            self.assertEqual(json.loads(path.with_suffix(".json").read_text())["frame_ids"], ["a", "b"])
        finally:
            path.unlink(missing_ok=True); path.with_suffix(".json").unlink(missing_ok=True)

    def test_positive_left_bida_roundtrip_and_resize_scale(self):
        disparity = sample()["raw_disparity"]
        self.assertTrue(np.array_equal(bidastabilizer_to_positive_left(positive_left_to_bidastabilizer(disparity)), disparity))
        self.assertTrue(np.all(resize_disparity(disparity, 4, 6) == 4))

    def test_rejects_bad_dtype_and_duplicate_frames(self):
        values = sample(); values["rgb_left"] = values["rgb_left"].astype(np.float64)
        with self.assertRaises(ValueError): validate_input(values)
        values = sample(); values["frame_ids"] = np.array(["a", "a"])
        with self.assertRaises(ValueError): validate_input(values)

    def test_bidastabilizer_safe_derived_checkpoint_gate(self):
        _, item = bidastabilizer._lock()
        paths = bidastabilizer._derived(item)
        self.assertEqual(set(paths), {"raftstereo", "stabilizer", "sea_raft"})
        with tempfile.TemporaryDirectory() as temporary:
            lock = json.loads((Path(__file__).resolve().parents[1] / "checkpoints.lock.json").read_text())
            lock["checkpoints"]["bidastabilizer_raftstereo_robust"]["derived"]["raftstereo"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "derived checkpoint hash mismatch"):
                bidastabilizer._derived(lock["checkpoints"]["bidastabilizer_raftstereo_robust"])
