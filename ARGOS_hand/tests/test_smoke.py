import inspect

import torch

from argos_v2_hand.codd import CODDStyleFusionHead, FrozenResNet18Layer1
from argos_v2_hand.losses import CODDFusionLossConfig, MultiAnchorLossConfig
from argos_v2_hand.raw_multi_anchor import RawMultiAnchorRefiner
from argos_v2_hand.training import codd_training_step, raw_multi_anchor_training_step


def test_frozen_resnet_checkpoint_is_explicit():
    assert inspect.signature(FrozenResNet18Layer1).parameters["checkpoint"].default is inspect.Parameter.empty


def test_cpu_forward_loss_backward_for_both_heads():
    torch.manual_seed(0)
    raw = torch.rand(2, 1, 8, 8)
    raw_model = RawMultiAnchorRefiner(channels=16, blocks=1)
    raw_losses, _ = raw_multi_anchor_training_step(raw_model, {
        "raw": raw, "candidates": torch.stack((raw[:, 0] + .2, raw[:, 0] - .1), 1),
        "candidate_valid": torch.ones(2, 2, 8, 8, dtype=torch.bool), "warp_support": torch.ones(2, 2, 8, 8, dtype=torch.bool),
        "fb_confidence": torch.ones(2, 2, 8, 8), "ages": torch.tensor([1, 2]), "provenance": torch.zeros(2),
        "gt": raw + .15, "gt_coverage": torch.ones_like(raw), "raw_valid": torch.ones_like(raw, dtype=torch.bool),
    }, MultiAnchorLossConfig())
    raw_losses["total"].backward()
    assert torch.isfinite(raw_losses["total"]) and any(p.grad is not None for p in raw_model.parameters())

    codd_model = CODDStyleFusionHead(8, width=16)
    codd_losses, _ = codd_training_step(codd_model, {
        "cues": torch.rand(2, 8, 8, 8), "support": torch.ones(2, 1, 8, 8, dtype=torch.bool), "raw": raw,
        "aligned_memory": raw + .2, "gt": raw + .1, "valid": torch.ones_like(raw, dtype=torch.bool),
    }, CODDFusionLossConfig(tau_reset_px=.1, tau_fusion_px=.1))
    codd_losses["total"].backward()
    assert torch.isfinite(codd_losses["total"]) and any(p.grad is not None for p in codd_model.parameters())
