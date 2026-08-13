from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "original_h4"))
sys.path.insert(0, str(ROOT))

from hard_h4 import HardH4
from lower_guard_h4 import LOWER_RATIO, TAU, factory, lower_guard_endpoint
from model_design.comparison.run_comparison import drive


class LowerGuardH4Test(unittest.TestCase):
    def test_endpoint_boundaries_and_rejections(self) -> None:
        raw = torch.tensor([[[[4., 4., 4., 4., 0., float("nan"), 4., 4.]]]])
        hard = torch.tensor([[[[2., 1.9, 8., float("nan"), 4., 4., 4., -1.]]]])
        support = torch.tensor([[[[True, True, True, True, True, True, False, True]]]])
        disparity, accepted = lower_guard_endpoint(raw, hard, support)
        self.assertEqual(disparity.shape, raw.shape)
        self.assertEqual(disparity.dtype, raw.dtype)
        self.assertTrue(torch.equal(accepted, torch.tensor([[[[True, False, True, False, False, False, False, False]]]])))
        torch.testing.assert_close(disparity, torch.where(accepted, hard, raw), equal_nan=True)

    def test_step_keeps_support_and_finite_update_for_invalid_raw_fallback(self) -> None:
        adapter = factory(device="cpu")
        raw = torch.tensor([[[[float("nan"), float("inf"), 4., 4.]]]])
        support = torch.ones_like(raw, dtype=torch.bool)
        parent = {"disparity": torch.tensor([[[[4., 4., float("nan"), 1.]]]]), "support": support,
                  "reset": False, "state_age": 1, "diagnostics": {"update_magnitude": 5.0}}
        with patch.object(HardH4, "step", return_value=parent):
            result = adapter.step({"raw": raw})
        self.assertIs(result["support"], support)
        torch.testing.assert_close(result["disparity"], raw, equal_nan=True)
        self.assertEqual(result["diagnostics"]["update_magnitude"], 0.0)

    def test_driver_uses_guarded_state_and_fixed_reanchors(self) -> None:
        adapter = factory(device="cpu")
        self.assertEqual((adapter.threshold, TAU, LOWER_RATIO), (.35, .35, .5))
        raw = [torch.tensor([[[[value]]]]) for value in (4., 8., 16., 32., 64., 128., 256.)]
        frames = [{"raw": value, "raw_valid": torch.ones_like(value, dtype=torch.bool), "rgb": torch.zeros(1),
                   "right_rgb": torch.zeros(1), "index": index} for index, value in enumerate(raw)]
        seen = []

        def hard_step(frame):
            seen.append((frame["reanchor"], frame["past_disparity"].clone()))
            hard = frame["raw"] * (.25 if float(frame["raw"].item()) == 16. else 1.5)
            return {"disparity": hard, "support": frame["raw_valid"], "reset": frame["reanchor"],
                    "state_age": frame["state_age"], "diagnostics": {"update_magnitude": 3.0}}

        with patch.object(HardH4, "step", side_effect=hard_step):
            outputs = dict(drive(adapter, frames, lambda _current, _past: (None, None)))
        self.assertTrue(outputs[1]["reset"])
        self.assertTrue(torch.equal(outputs[1]["disparity"], torch.tensor([[[[12.]]]])))
        self.assertTrue(torch.equal(seen[1][1], torch.tensor([[[[12.]]]])))
        self.assertTrue(torch.equal(outputs[2]["disparity"], torch.tensor([[[[16.]]]])))
        self.assertTrue(torch.equal(seen[2][1], torch.tensor([[[[16.]]]])))
        self.assertTrue(outputs[5]["reset"])
        self.assertTrue(torch.equal(seen[4][1], torch.tensor([[[[64.]]]])))

    def test_provenance_and_preregistration_hashes(self) -> None:
        adapter = factory(device="cpu")
        with patch.object(HardH4, "describe", return_value={"code": "hard_h4.py", "code_sha256": "parent"}):
            description = adapter.describe()
        self.assertEqual(description["module"], "lower_guard_h4")
        preregistration = json.loads((ROOT / "cycle3_preregister.json").read_text())
        for key, name in (("candidate_code", "lower_guard_h4.py"), ("test_code", "test_lower_guard_h4.py"),
                          ("probe_code", "probe_lower_guard_d2.py")):
            self.assertEqual(preregistration["hashes"][key]["sha256"], hashlib.sha256((ROOT / name).read_bytes()).hexdigest())
        self.assertEqual(preregistration["candidate"]["factory"], "factory")


if __name__ == "__main__":
    unittest.main()
