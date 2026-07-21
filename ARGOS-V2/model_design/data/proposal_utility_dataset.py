"""Causal SCARED-C pairs and proposal-utility targets for ARGOS v2."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from model_design.data.raw_error_dataset import RawErrorDataset


@dataclass(frozen=True)
class ProposalUtilityTargets:
    """GT-derived targets. None of these tensors may enter model evidence."""

    utility: torch.Tensor
    raw_error: torch.Tensor
    proposal_error: torch.Tensor
    classes: torch.Tensor
    regression_valid: torch.Tensor
    classification_valid: torch.Tensor
    helpful: torch.Tensor
    indifferent: torch.Tensor
    harmful: torch.Tensor


class ProposalUtilityDataset(Dataset):
    """Thin guarded reuse of the validated mmap temporal-pair dataset."""

    def __init__(
        self,
        backbones: Sequence[str],
        sequences: Sequence[str],
        *,
        coverage_threshold: float = 0.50,
        max_pairs_per_sequence: int | None = None,
        random_clip_start: bool = False,
        seed: int = 20260715,
    ) -> None:
        self.base = RawErrorDataset(
            backbones,
            sequences,
            coverage_threshold=coverage_threshold,
            max_pairs_per_sequence=max_pairs_per_sequence,
            random_clip_start=random_clip_start,
            seed=seed,
        )

    @property
    def records(self):
        return self.base.records

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict:
        return self.base[index]

    def describe(self) -> dict:
        return self.base.describe() | {"target": "frozen A2 proposal utility"}


def proposal_utility_targets(
    batch: dict,
    proposal_disparity: torch.Tensor,
    *,
    aligned_valid: torch.Tensor,
    warp_support: torch.Tensor,
    epsilon_px: float,
    coverage_threshold: float = 0.50,
) -> ProposalUtilityTargets:
    """Construct exact continuous and three-class utility targets.

    Class indices are harmful=0, indifferent=1, helpful=2. Invalid pixels get
    the ignore index -100 and never contribute to regression or classification.
    """
    if epsilon_px < 0:
        raise ValueError("epsilon_px must be non-negative")
    raw_error = (batch["raw"] - batch["gt"]).abs().detach()
    proposal_error = (proposal_disparity - batch["gt"]).abs().detach()
    utility = (raw_error - proposal_error).detach()
    valid = (
        (batch["gt_coverage"] > coverage_threshold)
        & batch["raw_valid"].bool()
        & aligned_valid.bool()
        & warp_support.bool()
        & torch.isfinite(utility)
        & torch.isfinite(proposal_disparity)
    ).detach()
    helpful = (valid & (utility > epsilon_px)).detach()
    harmful = (valid & (utility < -epsilon_px)).detach()
    indifferent = (valid & ~(helpful | harmful)).detach()
    classes = torch.full_like(utility, -100, dtype=torch.long)
    classes[harmful] = 0
    classes[indifferent] = 1
    classes[helpful] = 2
    return ProposalUtilityTargets(
        utility=utility,
        raw_error=raw_error,
        proposal_error=proposal_error,
        classes=classes,
        regression_valid=valid,
        classification_valid=valid,
        helpful=helpful,
        indifferent=indifferent,
        harmful=harmful,
    )


def boundary_mask(disparity: torch.Tensor) -> torch.Tensor:
    dx = F.pad((disparity[..., 1:] - disparity[..., :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((disparity[..., 1:, :] - disparity[..., :-1, :]).abs(), (0, 0, 0, 1))
    return F.max_pool2d(((dx > 1) | (dy > 1)).float(), 3, 1, 1).bool()


def stratified_training_targets(
    target: ProposalUtilityTargets,
    proposal_update: torch.Tensor,
    gt: torch.Tensor,
    *,
    maximum_pixels: int = 32768,
) -> ProposalUtilityTargets:
    """Deterministically balance class/error/update/boundary strata.

    Dataset records are already balanced by backbone and sequence. This function
    prevents the natural indifferent class from dominating optimizer pixels.
    Natural validation and test targets must not call this function.
    """
    if maximum_pixels <= 0:
        return target
    raw_wrong = target.raw_error > 0.5
    updated = proposal_update.detach().abs() > 0.05
    boundary = boundary_mask(gt.detach())
    # 3 classes x 2 raw-error bins x 2 update bins x 2 boundary bins.
    code = target.classes.clamp_min(0) * 8 + raw_wrong.long() * 4 + updated.long() * 2 + boundary.long()
    present = torch.unique(code[target.regression_valid])
    if not present.numel():
        return target
    quota = max(1, maximum_pixels // int(present.numel()))
    selected = torch.zeros_like(target.regression_valid)
    flat_selected = selected.flatten()
    flat_code = code.flatten()
    flat_valid = target.regression_valid.flatten()
    for value in present.tolist():
        indices = torch.nonzero(flat_valid & (flat_code == value), as_tuple=False).flatten()
        if indices.numel() > quota:
            positions = torch.linspace(0, indices.numel() - 1, quota, device=indices.device).long()
            indices = indices[positions]
        flat_selected[indices] = True
    return replace(
        target,
        regression_valid=selected,
        classification_valid=selected,
        helpful=target.helpful & selected,
        indifferent=target.indifferent & selected,
        harmful=target.harmful & selected,
    )


__all__ = [
    "ProposalUtilityDataset",
    "ProposalUtilityTargets",
    "boundary_mask",
    "proposal_utility_targets",
    "stratified_training_targets",
]
