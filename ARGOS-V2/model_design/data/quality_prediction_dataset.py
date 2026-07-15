"""Exact-age causal Q0 records and aligned candidate target construction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch.utils.data import Dataset

from model_design.data.temporal_memory_dataset import TemporalMemoryDataset


MEMORY_AGES = (1, 2, 4, 8)
CANDIDATE_AGES = (0, 1, 2, 4, 8)
CANDIDATE_NAMES = ("raw", "t-1", "t-2", "t-4", "t-8")
SEEN_BACKBONES = ("S2M2-S", "RAFT-Stereo", "StereoAnywhere")
FORBIDDEN_Q0_BACKBONES = ("Fast-FoundationStereo", "CREStereo")


@dataclass(frozen=True)
class QualityCandidateBatch:
    """Aligned Q0 candidates and targets, all on the cache grid.

    Candidate maps are ``[B,K=5,1,H,W]``. Consensus maps are
    ``[B,1,H,W]``. Invalid candidate targets contain finite placeholder values
    but are excluded exactly by ``target_valid``.
    """

    disparity: torch.Tensor
    candidate_valid: torch.Tensor
    warp_support: torch.Tensor
    forward_backward_error: torch.Tensor
    forward_backward_confidence: torch.Tensor
    photometric_residual: torch.Tensor
    flow_magnitude: torch.Tensor
    ages: torch.Tensor
    consensus_median: torch.Tensor
    consensus_mad: torch.Tensor
    witness_count: torch.Tensor
    target_error: torch.Tensor
    target_advantage: torch.Tensor
    target_valid: torch.Tensor
    failure_masks: dict[str, torch.Tensor]


class QualityPredictionDataset(Dataset):
    """Thin Q0 wrapper over the validated exact-age temporal-memory dataset.

    RGB is returned only for frozen SEA-RAFT/BiDA construction. It is not a Q0
    model input. Metadata retains source backbone/sequence/frame IDs for audit.
    """

    def __init__(
        self,
        backbones: Sequence[str],
        sequences: Sequence[str],
        *,
        max_samples_per_sequence: int | None = None,
        random_clip_start: bool = False,
        seed: int = 20260713,
    ) -> None:
        forbidden = set(backbones) & set(FORBIDDEN_Q0_BACKBONES)
        if forbidden:
            raise ValueError(f"Q0 must not touch unseen backbones: {sorted(forbidden)}")
        unknown = set(backbones) - set(SEEN_BACKBONES)
        if unknown:
            raise ValueError(f"Q0 training/selection accepts only seen backbones: {sorted(unknown)}")
        self.base = TemporalMemoryDataset(
            backbones,
            sequences,
            ages=MEMORY_AGES,
            max_samples_per_sequence=max_samples_per_sequence,
            random_clip_start=random_clip_start,
            seed=seed,
        )
        self.backbones = tuple(backbones)
        self.sequences = tuple(sequences)

    @property
    def records(self):
        return self.base.records

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict:
        sample = self.base[index]
        if tuple(sample["ages"].tolist()) != MEMORY_AGES:
            raise RuntimeError("validated memory dataset returned an unexpected age order")
        source_disparity = torch.cat((sample["raw"][None], sample["past"]), dim=0)
        source_valid = torch.cat((sample["raw_valid"][None], sample["past_valid"]), dim=0)
        frame_ids = [sample["current_frame_id"], *sample["past_frame_ids"]]
        return sample | {
            "source_disparity": source_disparity,
            "source_valid": source_valid,
            "candidate_ages": torch.tensor(CANDIDATE_AGES, dtype=torch.long),
            "candidate_names": list(CANDIDATE_NAMES),
            "source_frame_ids": frame_ids,
            "source_sequences": [sample["sequence"]] * len(CANDIDATE_AGES),
            "source_backbones": [sample["backbone"]] * len(CANDIDATE_AGES),
        }

    def describe(self) -> dict:
        return {
            "candidate_order": list(CANDIDATE_NAMES),
            "candidate_ages": list(CANDIDATE_AGES),
            "backbones": list(self.backbones),
            "sequences": list(self.sequences),
            "sample_count": len(self),
            "causal": True,
            "rgb_policy": "alignment only; excluded from Q0 model inputs",
        }


def masked_median(values: torch.Tensor, valid: torch.Tensor, *, dim: int = 1) -> torch.Tensor:
    """Exact even/odd masked median with NaN for zero witnesses."""
    if values.shape != valid.shape:
        raise ValueError("values and valid must share shape")
    filled = values.masked_fill(~valid.bool(), torch.inf)
    ordered = filled.sort(dim=dim).values
    count = valid.sum(dim=dim, keepdim=True)
    lower = ((count - 1).clamp_min(0) // 2).long()
    upper = (count.clamp_min(1) // 2).long()
    lo = ordered.gather(dim, lower)
    hi = ordered.gather(dim, upper)
    median = 0.5 * (lo + hi)
    return torch.where(count > 0, median, torch.full_like(median, torch.nan))


def memory_consensus(
    aligned_memory: torch.Tensor,
    memory_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return CMC-compatible median, MAD and witness count over four memories."""
    median_keepdim = masked_median(aligned_memory, memory_valid, dim=1)
    deviation = (aligned_memory - median_keepdim).abs()
    mad_keepdim = masked_median(deviation, memory_valid, dim=1)
    count = memory_valid.sum(dim=1).to(aligned_memory.dtype)
    return median_keepdim.squeeze(1), mad_keepdim.squeeze(1), count


def assemble_quality_candidates(
    batch: Mapping[str, torch.Tensor],
    memory_evidence: Mapping[str, torch.Tensor],
    *,
    coverage_threshold: float = 0.50,
    margin: float = 0.10,
) -> QualityCandidateBatch:
    """Combine raw and four aligned memories and construct exact Q0 targets."""
    raw = batch["raw"]
    raw_valid = batch["raw_valid"].bool()
    memory = memory_evidence["aligned_past_disparity"]
    if memory.ndim != 5 or memory.shape[1] != len(MEMORY_AGES):
        raise ValueError("aligned memory must be [B,4,1,H,W]")
    memory_support = memory_evidence["warp_support"].bool()
    memory_valid = memory_evidence["aligned_validity"].bool() & memory_support
    disparity = torch.cat((raw[:, None], memory), dim=1)
    candidate_valid = torch.cat((raw_valid[:, None], memory_valid), dim=1)
    support = torch.cat((raw_valid[:, None], memory_support), dim=1)
    zeros = torch.zeros_like(raw[:, None])
    ones = torch.ones_like(raw[:, None])

    def with_raw(name: str, raw_value: torch.Tensor) -> torch.Tensor:
        return torch.cat((raw_value, memory_evidence[name]), dim=1)

    fb_error = with_raw("forward_backward_error", zeros)
    fb_confidence = with_raw("forward_backward_confidence", ones)
    photo = with_raw("photometric_residual", zeros)
    flow = with_raw("flow_magnitude", zeros)
    median, mad, count = memory_consensus(memory, memory_valid)
    gt = batch["gt"][:, None]
    gt_valid = (batch["gt_coverage"] > coverage_threshold)[:, None]
    target_valid = gt_valid & raw_valid[:, None] & candidate_valid & support
    error = (disparity - gt).abs()
    raw_error = error[:, :1]
    advantage = raw_error - error

    median_finite = torch.isfinite(median)
    median_safe = torch.nan_to_num(median)
    median_error = (median_safe - batch["gt"]).abs()
    memory_better = (error[:, 1:] + margin < raw_error) & target_valid[:, 1:]
    minority_correct = (
        memory_better.any(dim=1)
        & median_finite
        & (median_error + margin >= raw_error[:, 0])
        & gt_valid[:, 0]
        & raw_valid
    )
    correlated_wrong = (
        (count >= 3)
        & (torch.nan_to_num(mad, nan=torch.inf) <= 0.25)
        & median_finite
        & (median_error > raw_error[:, 0] + margin)
        & gt_valid[:, 0]
        & raw_valid
    )
    raw_clean = (raw_error[:, 0] <= 0.50) & gt_valid[:, 0] & raw_valid
    plausible_worse = (
        raw_clean
        & ((error[:, 1:] > raw_error + margin) & target_valid[:, 1:]).any(dim=1)
    )
    low_fb_wrong = (
        (memory_evidence["forward_backward_error"] <= 0.50)
        & (error[:, 1:] > raw_error + margin)
        & target_valid[:, 1:]
    ).any(dim=1)
    low_photo_wrong = (
        (memory_evidence["photometric_residual"] <= 0.10)
        & (error[:, 1:] > raw_error + margin)
        & target_valid[:, 1:]
    ).any(dim=1)
    return QualityCandidateBatch(
        disparity=disparity,
        candidate_valid=candidate_valid,
        warp_support=support,
        forward_backward_error=fb_error,
        forward_backward_confidence=fb_confidence,
        photometric_residual=photo,
        flow_magnitude=flow,
        ages=torch.tensor(CANDIDATE_AGES, device=raw.device),
        consensus_median=median,
        consensus_mad=mad,
        witness_count=count,
        target_error=error.detach(),
        target_advantage=advantage.detach(),
        target_valid=target_valid.detach(),
        failure_masks={
            "minority_correct": minority_correct.detach(),
            "correlated_consensus_wrong": correlated_wrong.detach(),
            "raw_clean_memory_worse": plausible_worse.detach(),
            "low_fb_wrong": low_fb_wrong.detach(),
            "low_photometric_wrong": low_photo_wrong.detach(),
        },
    )


__all__ = [
    "CANDIDATE_AGES",
    "CANDIDATE_NAMES",
    "FORBIDDEN_Q0_BACKBONES",
    "MEMORY_AGES",
    "QualityCandidateBatch",
    "QualityPredictionDataset",
    "assemble_quality_candidates",
    "masked_median",
    "memory_consensus",
]
