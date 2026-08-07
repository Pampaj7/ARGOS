#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from metrics.temporal_metrics import gt_tce, nr_tce, pull_warp, stereo_photo


class TemporalMetricsTest(unittest.TestCase):
    def setUp(self):
        self.flow = np.zeros((3, 4, 2), np.float32)
        self.valid = np.ones((3, 4), bool)

    def test_identical_sequence_is_zero(self):
        value = np.full((3, 4), 2.0, np.float32)
        self.assertEqual(gt_tce(value, value, value, value, self.flow, current_gt_valid=self.valid, past_gt_valid=self.valid)["gt_tce"], 0.0)
        self.assertEqual(nr_tce(value, value, self.flow, current_valid=self.valid, past_valid=self.valid)["nr_tce"], 0.0)

    def test_known_change_matching_gt_is_zero(self):
        past = np.full((3, 4), 2.0, np.float32); current = past + 3
        self.assertEqual(gt_tce(current, past, current, past, self.flow, current_gt_valid=self.valid, past_gt_valid=self.valid)["gt_tce"], 0.0)

    def test_constant_oversmoothing_is_penalized(self):
        past = np.full((3, 4), 2.0, np.float32); gt = past + 3; constant = past
        self.assertGreater(gt_tce(constant, past, gt, past, self.flow, current_gt_valid=self.valid, past_gt_valid=self.valid)["gt_tce"], 2.9)

    def test_invalid_is_excluded_and_scale_normalizes(self):
        past = np.full((3, 4), 2.0, np.float32); current = past + 1; valid = self.valid.copy(); valid[0, 0] = False
        result = gt_tce(current, past, current, past, self.flow, current_gt_valid=valid, past_gt_valid=valid)
        self.assertEqual(result["count"], 11); self.assertEqual(result["rgt_tce"], 0.0)

    def test_flow_sign_reversal_fails(self):
        past = np.tile(np.arange(4, dtype=np.float32) + 1, (3, 1)); current = np.roll(past, -1, axis=1); current[:, -1] = 1
        correct = np.zeros_like(self.flow); correct[..., 0] = 1
        wrong = -correct
        a = gt_tce(current, past, current, past, correct, current_gt_valid=self.valid, past_gt_valid=self.valid)["gt_tce"]
        b = gt_tce(current, past, past, past, wrong, current_gt_valid=self.valid, past_gt_valid=self.valid)["gt_tce"]
        self.assertEqual(a, 0.0); self.assertGreater(b, 0.1)

    def test_positive_left_stereo_warp(self):
        left = np.tile(np.arange(5, dtype=np.float32), (2, 1)); right = np.roll(left, 1, axis=1); right[:, 0] = 0
        score = stereo_photo(left[..., None], right[..., None], np.ones((2, 5), np.float32))
        self.assertLess(score["stereo_photo"], 0.01)


if __name__ == "__main__":
    unittest.main()
