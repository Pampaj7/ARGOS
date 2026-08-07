"""Single-step training helpers over already-prepared tensor batches."""
from __future__ import annotations

import torch

from .codd import CODDCues, CODDFusionOutput, CODDStyleFusionHead
from .losses import CODDFusionLossConfig, MultiAnchorLossConfig, codd_fusion_losses_with_gt, multi_anchor_targets, raw_multi_anchor_losses
from .raw_multi_anchor import MultiAnchorEvidence, RawMultiAnchorRefiner


def raw_multi_anchor_training_step(model: RawMultiAnchorRefiner, batch: dict[str, torch.Tensor], config: MultiAnchorLossConfig, *, enable_fusion: bool = True) -> tuple[dict[str, torch.Tensor], MultiAnchorEvidence]:
    """Forward/loss only; callers own device movement, backward, and optimizer."""
    evidence = MultiAnchorEvidence(batch["raw"].float(), batch["candidates"].float(), batch["candidate_valid"].bool(), batch["warp_support"].bool(), batch["fb_confidence"].float(), batch["ages"], batch["provenance"])
    targets = multi_anchor_targets(evidence.raw, evidence.candidates, batch["gt"].float(), batch["gt_coverage"].float(), batch["raw_valid"].bool(), evidence, margin_px=config.margin_px)
    return raw_multi_anchor_losses(model(evidence), evidence, targets, config, enable_fusion=enable_fusion), evidence


def codd_training_step(model: CODDStyleFusionHead, batch: dict[str, torch.Tensor], config: CODDFusionLossConfig) -> tuple[dict[str, torch.Tensor], CODDFusionOutput]:
    """Forward/loss only; `cues`, `raw`, `aligned_memory`, `gt`, and `valid` are prepared tensors."""
    cues = CODDCues(batch["cues"], batch["support"].bool(), batch["cues"].shape[1])
    output = model(cues, batch["raw"], batch["aligned_memory"])
    return codd_fusion_losses_with_gt(output, raw=batch["raw"], memory=batch["aligned_memory"], gt=batch["gt"], valid=batch["valid"], config=config), output
