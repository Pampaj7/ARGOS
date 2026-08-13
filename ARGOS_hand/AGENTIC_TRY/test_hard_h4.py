from __future__ import annotations

import unittest

import torch

from hard_h4 import factory_035, factory_050, hard_endpoint_fusion


class HardH4Test(unittest.TestCase):
    def test_endpoint_selects_memory_at_or_above_threshold(self) -> None:
        raw = torch.tensor([[[[1.25, -2.0]]]])
        memory = torch.tensor([[[[8.0, 5.0]]]])
        disparity, accepted = hard_endpoint_fusion(raw, memory, torch.tensor([[[[.49, .50]]]]), .50)
        self.assertFalse(accepted[..., 0].item())
        self.assertTrue(accepted[..., 1].item())
        self.assertTrue(torch.equal(disparity[..., :1], raw[..., :1]))
        self.assertTrue(torch.equal(disparity[..., 1:], memory[..., 1:]))

    def test_factories_expose_requested_thresholds_without_loading_a_model(self) -> None:
        self.assertEqual(factory_035(device="cpu").threshold, .35)
        self.assertEqual(factory_050(device="cpu").threshold, .50)


if __name__ == "__main__":
    unittest.main()
