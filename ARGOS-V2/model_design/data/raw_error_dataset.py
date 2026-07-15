"""Causal t-1 records and exact raw-error targets for ARGOS v2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch.utils.data import Dataset

from model_design.data.temporal_pair_dataset import (
    SEEN_BACKBONES,
    TemporalPairDataset,
)


FORBIDDEN_SELECTION_BACKBONES = ("Fast-FoundationStereo", "CREStereo")
CALIBRATION_SEQUENCES = ("dataset_7_keyframe_1", "dataset_7_keyframe_2")
TEST_SEQUENCES = ("dataset_7_keyframe_3", "dataset_7_keyframe_4")


@dataclass(frozen=True)
class RawErrorTargets:
    error: torch.Tensor
    label: torch.Tensor
    regression_valid: torch.Tensor
    classification_valid: torch.Tensor
    clean: torch.Tensor


class RawErrorDataset(Dataset):
    """Guarded thin wrapper around the validated mmap temporal-pair loader."""

    def __init__(
        self,
        backbones: Sequence[str],
        sequences: Sequence[str],
        *,
        coverage_threshold: float = 0.50,
        max_pairs_per_sequence: int | None = None,
        random_clip_start: bool = False,
        seed: int = 20260713,
    ) -> None:
        forbidden = set(backbones) & set(FORBIDDEN_SELECTION_BACKBONES)
        if forbidden:
            raise ValueError(f"raw-error selection must not touch unseen backbones: {sorted(forbidden)}")
        unknown = set(backbones) - set(SEEN_BACKBONES)
        if unknown:
            raise ValueError(f"only seen backbones are permitted before promotion: {sorted(unknown)}")
        self.base = TemporalPairDataset(
            backbones,
            sequences,
            coverage_threshold=coverage_threshold,
            max_pairs_per_sequence=max_pairs_per_sequence,
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
        return self.base[index]

    def describe(self) -> dict:
        return self.base.describe() | {
            "causal_age": 1,
            "forbidden_selection_backbones": list(FORBIDDEN_SELECTION_BACKBONES),
        }


def raw_error_targets(
    batch: dict,
    *,
    epsilon_px: float,
    indifference_band_px: float,
    coverage_threshold: float,
    clean_threshold_px: float = 0.50,
) -> RawErrorTargets:
    error = (batch["raw"] - batch["gt"]).abs().detach()
    regression_valid = (
        (batch["gt_coverage"] > coverage_threshold) & batch["raw_valid"].bool()
    ).detach()
    classification_valid = (
        regression_valid & ((error - epsilon_px).abs() > indifference_band_px)
    ).detach()
    return RawErrorTargets(
        error=error,
        label=(error > epsilon_px).to(error.dtype),
        regression_valid=regression_valid,
        classification_valid=classification_valid,
        clean=(regression_valid & (error <= clean_threshold_px)).detach(),
    )


__all__ = [
    "CALIBRATION_SEQUENCES",
    "FORBIDDEN_SELECTION_BACKBONES",
    "RawErrorDataset",
    "RawErrorTargets",
    "TEST_SEQUENCES",
    "raw_error_targets",
]
