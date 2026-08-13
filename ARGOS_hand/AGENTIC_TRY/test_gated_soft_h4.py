from __future__ import annotations

import unittest

import torch

from gated_soft_h4 import factory_035, gated_soft_endpoint_fusion


class GatedSoftH4Test(unittest.TestCase):
    def test_endpoint_selects_raw_below_and_canonical_soft_at_threshold(self) -> None:
        raw = torch.tensor([[[[1.25, -2.0]]]])
        fused = torch.tensor([[[[8.0, 5.0]]]])
        disparity, accepted = gated_soft_endpoint_fusion(raw, fused, torch.tensor([[[[.34, .35]]]]), .35)
        self.assertFalse(accepted[..., 0].item())
        self.assertTrue(accepted[..., 1].item())
        self.assertTrue(torch.equal(disparity[..., :1], raw[..., :1]))
        self.assertTrue(torch.equal(disparity[..., 1:], fused[..., 1:]))

    def test_factory_uses_requested_threshold_without_loading_a_model(self) -> None:
        self.assertEqual(factory_035(device="cpu").threshold, .35)


if __name__ == "__main__":
    unittest.main()
