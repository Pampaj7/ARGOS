"""Validated ARGOS v2 train steps over already-prepared tensor batches."""
from __future__ import annotations

import torch

from .codd import CODDCues, CODDFusionOutput, CODDStyleFusionHead
from .losses import (
    CODDFusionLossConfig,
    MultiAnchorLossConfig,
    codd_fusion_losses_with_gt,
    multi_anchor_targets,
    raw_multi_anchor_losses,
)
from .raw_multi_anchor import MultiAnchorEvidence, RawMultiAnchorRefiner


def raw_multi_anchor_training_step(
    model: RawMultiAnchorRefiner,
    batch: dict[str, torch.Tensor],
    config: MultiAnchorLossConfig,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    enable_fusion: bool = True,
    scaler: torch.amp.GradScaler | torch.cuda.amp.GradScaler | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> tuple[dict[str, torch.Tensor], MultiAnchorEvidence]:
    """Run ARGOS's validated raw-anchor forward/loss/update sequence.

    The batch must already contain the cache-grid tensors named below; caller
    owns data loading and device transfer.  An omitted optimizer leaves the
    differentiable loss for an external backward call.
    """
    model.train()
    evidence = MultiAnchorEvidence(
        batch["raw"].float(), batch["candidates"].float(), batch["candidate_valid"].bool(),
        batch["warp_support"].bool(), batch["fb_confidence"].float(), batch["ages"], batch["provenance"],
    )
    targets = multi_anchor_targets(
        evidence.raw, evidence.candidates, batch["gt"].float(), batch["gt_coverage"].float(),
        batch["raw_valid"].bool(), evidence, margin_px=config.margin_px,
    )
    device_type = evidence.raw.device.type
    with torch.autocast(device_type=device_type, enabled=device_type == "cuda"):
        output = model(evidence)
        losses = raw_multi_anchor_losses(output, evidence, targets, config, enable_fusion=enable_fusion)
    if optimizer is not None:
        # Extracted from run_raw_multi_anchor_temporal_refiner.py::train.
        optimizer.zero_grad(set_to_none=True)
        if scaler is None:
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        else:
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        if scheduler is not None:
            scheduler.step()
    return losses, evidence


def codd_training_step(
    model: CODDStyleFusionHead,
    batch: dict[str, torch.Tensor],
    config: CODDFusionLossConfig,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    scaler: torch.amp.GradScaler | torch.cuda.amp.GradScaler | None = None,
) -> tuple[dict[str, torch.Tensor], CODDFusionOutput]:
    """Run ARGOS's validated H4 CODD forward/loss/update sequence."""
    model.train()
    cues = CODDCues(batch["cues"], batch["support"].bool(), batch["cues"].shape[1])
    device_type = batch["raw"].device.type
    with torch.autocast(device_type=device_type, enabled=device_type == "cuda"):
        output = model(cues, batch["raw"], batch["aligned_memory"])
        losses = codd_fusion_losses_with_gt(
            output, raw=batch["raw"], memory=batch["aligned_memory"], gt=batch["gt"],
            valid=batch["valid"], config=config,
        )
    if optimizer is not None:
        # Extracted from run_codd_style_fusion_probe.py::train.
        optimizer.zero_grad(set_to_none=True)
        if scaler is None:
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        else:
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
    return losses, output
