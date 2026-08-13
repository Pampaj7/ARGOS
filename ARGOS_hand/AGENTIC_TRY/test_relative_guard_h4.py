from __future__ import annotations

import unittest
from unittest.mock import patch
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "original_h4"))

from hard_h4 import HardH4
from model_design.comparison.run_comparison import drive
from relative_guard_h4 import KAPPA, TAU, factory, relative_guard_endpoint


class RelativeGuardH4Test(unittest.TestCase):
    def test_endpoint_accepts_boundaries_and_rejects_invalid_or_unsupported_values(self) -> None:
        raw = torch.tensor([[[[4., 4., 4., 4., 4., 0., float("nan"), 4., 4., 4.]]]])
        hard = torch.tensor([[[[2., 8., 1.9, 8.1, float("nan"), 4., 4., 4., 0., -1.]]]])
        support = torch.tensor([[[[True, True, True, True, True, True, True, False, True, True]]]])
        disparity, accepted = relative_guard_endpoint(raw, hard, support)

        self.assertEqual(disparity.shape, raw.shape)
        self.assertEqual(disparity.dtype, raw.dtype)
        self.assertTrue(torch.equal(accepted, torch.tensor([[[[True, True, False, False, False, False, False, False, False, False]]]])))
        torch.testing.assert_close(disparity, torch.where(accepted, hard, raw), equal_nan=True)

    def test_guarded_output_is_recurrent_driver_state_and_resets_stay_fixed(self) -> None:
        adapter = factory(device="cpu")
        self.assertEqual((adapter.threshold, KAPPA, TAU), (.35, 2.0, .35))
        raw = torch.tensor([[[[4.]]]])
        valid = torch.ones_like(raw, dtype=torch.bool)
        frames = [{"raw": raw, "raw_valid": valid, "rgb": torch.zeros(1), "right_rgb": torch.zeros(1), "index": index}
                  for index in range(6)]
        seen = []

        def hard_step(frame):
            seen.append((frame["reanchor"], frame["past_disparity"].clone()))
            return {"disparity": frame["raw"] * 2, "support": frame["raw_valid"], "reset": frame["reanchor"],
                    "state_age": frame["state_age"], "diagnostics": {"update_magnitude": 4.0}}

        with patch.object(HardH4, "step", side_effect=hard_step):
            outputs = dict(drive(adapter, frames, lambda _current, _past: (None, None)))
        self.assertTrue(outputs[0]["reset"])
        self.assertTrue(torch.equal(outputs[0]["disparity"], raw))
        self.assertTrue(torch.equal(seen[1][1], raw * 2))
        self.assertTrue(seen[4][0])
        self.assertTrue(torch.equal(seen[4][1], raw))

    def test_step_rejects_out_of_ratio_and_keeps_finite_update_diagnostic(self) -> None:
        adapter = factory(device="cpu")
        raw = torch.tensor([[[[4., 4.]]]])
        parent_result = {"disparity": torch.tensor([[[[8., 9.]]]]), "support": torch.ones_like(raw, dtype=torch.bool),
                         "reset": False, "state_age": 1, "diagnostics": {"update_magnitude": 5.0}}
        with patch.object(HardH4, "step", return_value=parent_result):
            result = adapter.step({"raw": raw})
        torch.testing.assert_close(result["disparity"], torch.tensor([[[[8., 4.]]]]))
        self.assertEqual(result["diagnostics"]["update_magnitude"], 2.0)
        parent_result["disparity"] = torch.full_like(raw, 9.)
        with patch.object(HardH4, "step", return_value=parent_result):
            rejected = adapter.step({"raw": raw})
        self.assertEqual(rejected["diagnostics"]["update_magnitude"], 0.0)

    def test_step_nonfinite_supported_values_leave_raw_and_finite_diagnostic(self) -> None:
        adapter = factory(device="cpu")
        raw = torch.tensor([[[[float("nan"), float("inf"), 4., 4.]]]])
        parent_result = {"disparity": torch.tensor([[[[4., 4., float("nan"), float("inf")]]]]),
                         "support": torch.ones_like(raw, dtype=torch.bool), "reset": False, "state_age": 1,
                         "diagnostics": {"update_magnitude": 5.0}}
        with patch.object(HardH4, "step", return_value=parent_result):
            result = adapter.step({"raw": raw})
        torch.testing.assert_close(result["disparity"], raw, equal_nan=True)
        self.assertEqual(result["diagnostics"]["update_magnitude"], 0.0)


if __name__ == "__main__":
    unittest.main()
