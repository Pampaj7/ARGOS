"""Causal t-1 data and GT utility targets for the ARGOS v2 memory selector.

The dataset deliberately contains no proposal, detector, backbone identity or
future-frame input.  Alignment evidence is produced later, on the GPU, by the
canonical BiDA adapter.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

from model_design.data.temporal_pair_dataset import SEEN_BACKBONES, TemporalPairDataset


@dataclass(frozen=True)
class UtilityTargets:
    """GT-only utilities; none of these tensors may enter model inputs."""

    # Per-pixel utility remains the only evaluation target.
    utility: torch.Tensor
    # Optional local pooled utility is a training-only regional label.
    supervision_utility: torch.Tensor
    raw_error: torch.Tensor
    memory_error: torch.Tensor
    valid: torch.Tensor
    memory_better: torch.Tensor
    helpful_gain: torch.Tensor
    harmful_magnitude: torch.Tensor


class UtilityMemorySelectorDataset(Dataset):
    """Thin, mmap-backed wrapper of the validated causal pair loader."""

    def __init__(
        self,
        backbones: Sequence[str],
        sequences: Sequence[str],
        *,
        coverage_threshold: float = 0.50,
        max_pairs_per_sequence: int | None = None,
        random_clip_start: bool = False,
        seed: int = 20260717,
        selection_only: bool = True,
        include_right_rgb: bool = False,
    ) -> None:
        if selection_only:
            forbidden = set(backbones) - set(SEEN_BACKBONES)
            if forbidden:
                raise ValueError(f"selection data cannot load unseen backbones: {sorted(forbidden)}")
        self.base = TemporalPairDataset(
            backbones, sequences, coverage_threshold=coverage_threshold,
            max_pairs_per_sequence=max_pairs_per_sequence,
            random_clip_start=random_clip_start, seed=seed, include_right_rgb=include_right_rgb,
        )
        self.backbones, self.sequences = tuple(backbones), tuple(sequences)
        self.selection_only = selection_only
        self.include_right_rgb = bool(include_right_rgb)

    @property
    def records(self):
        return self.base.records

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict:
        return self.base[index]

    def describe(self) -> dict:
        return self.base.describe() | {
            "causal_pair": "t-1 -> t; no future access",
            "selection_only": self.selection_only,
            "include_right_rgb": self.include_right_rgb,
        }


def utility_targets(
    batch: dict,
    aligned_memory: torch.Tensor,
    aligned_valid: torch.Tensor,
    warp_support: torch.Tensor,
    *,
    epsilon_px: float = 0.10,
    coverage_threshold: float = 0.50,
    regional_kernel: int = 1,
    additional_valid: torch.Tensor | None = None,
) -> UtilityTargets:
    """Return exact raw-versus-warped-memory utility at the cache grid."""
    if regional_kernel < 1 or regional_kernel % 2 == 0:
        raise ValueError("regional_kernel must be a positive odd integer")
    raw_error = (batch["raw"] - batch["gt"]).abs().detach()
    memory_error = (aligned_memory - batch["gt"]).abs().detach()
    utility = (raw_error - memory_error).detach()
    valid = (
        (batch["gt_coverage"] > coverage_threshold)
        & batch["raw_valid"].bool()
        & aligned_valid.bool()
        & warp_support.bool()
        & torch.isfinite(utility)
    ).detach()
    if additional_valid is not None:
        if additional_valid.shape != valid.shape:
            raise ValueError("additional_valid must match the cache-grid target shape")
        valid = (valid & additional_valid.bool()).detach()
    if regional_kernel == 1:
        supervision_utility = utility
    else:
        # Region-level supervision asks whether a local causal memory region
        # has positive *net* utility.  It never changes per-pixel evaluation,
        # uses only valid neighbouring supervision, and has no inference-time
        # GT dependency.
        support = F.avg_pool2d(valid.float(), regional_kernel, 1, regional_kernel // 2).clamp_min(1e-6)
        supervision_utility = F.avg_pool2d(utility * valid, regional_kernel, 1, regional_kernel // 2) / support
        supervision_utility = supervision_utility.detach()
    return UtilityTargets(
        utility=utility,
        supervision_utility=supervision_utility,
        raw_error=raw_error,
        memory_error=memory_error,
        valid=valid,
        memory_better=(valid & (supervision_utility > epsilon_px)).detach(),
        helpful_gain=torch.relu(supervision_utility - epsilon_px).detach(),
        harmful_magnitude=torch.relu(-supervision_utility - epsilon_px).detach(),
    )


class BalancedSequenceSampler(Sampler[int]):
    """Epoch sampler with equal backbone/sequence exposure and full coverage.

    Every (backbone, sequence) group is shuffled deterministically and repeated
    only to the longest group length. Thus no large sequence dominates merely
    because it contains many correlated adjacent frames, while every original
    pair occurs at least once per epoch.
    """

    def __init__(self, dataset: UtilityMemorySelectorDataset, *, seed: int) -> None:
        self.seed, self.epoch = int(seed), 0
        groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, record in enumerate(dataset.records):
            groups[(record.backbone, record.sequence)].append(index)
        if not groups:
            raise ValueError("cannot sample an empty dataset")
        self.groups = dict(sorted(groups.items()))
        self.group_length = max(map(len, self.groups.values()))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.groups) * self.group_length

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        expanded: list[list[int]] = []
        for _, values in self.groups.items():
            order = torch.randperm(len(values), generator=generator).tolist()
            shuffled = [values[i] for i in order]
            expanded.append([shuffled[i % len(shuffled)] for i in range(self.group_length)])
        # Interleave groups so a worker batch is not a single sequence.
        group_order = torch.randperm(len(expanded), generator=generator).tolist()
        for offset in range(self.group_length):
            round_order = torch.randperm(len(group_order), generator=generator).tolist()
            for idx in round_order:
                yield expanded[group_order[idx]][offset]


def dataset_id_from_sequence(sequence: str) -> str:
    """Return the acquisition/session identifier without exposing it to a model."""
    prefix, marker, _ = str(sequence).partition("_keyframe_")
    if not marker or not prefix.startswith("dataset_"):
        raise ValueError(f"unsupported SCARED-C sequence identifier: {sequence!r}")
    return prefix


class HierarchicalDatasetSequenceSampler(Sampler[int]):
    """Deterministic full-coverage sampler balanced by backbone, session and sequence.

    This sampler is deliberately an *input-order* operation.  It does not add
    a group feature to the selector.  A `(backbone, dataset-ID)` group is first
    expanded to the size of the largest such group.  Within it, each complete
    keyframe sequence is expanded to the longest sequence in that dataset.
    Thus every original pair occurs at least once per epoch while short
    sessions/keyframes are repeated only to eliminate exposure imbalance.
    """

    def __init__(self, dataset: UtilityMemorySelectorDataset, *, seed: int) -> None:
        self.seed, self.epoch = int(seed), 0
        nested: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for index, record in enumerate(dataset.records):
            session = dataset_id_from_sequence(record.sequence)
            nested[(record.backbone, session)][record.sequence].append(index)
        if not nested:
            raise ValueError("cannot sample an empty dataset")
        self.groups = {
            key: {sequence: values for sequence, values in sorted(per_sequence.items())}
            for key, per_sequence in sorted(nested.items())
        }
        self.sequence_length = {
            key: max(map(len, per_sequence.values())) for key, per_sequence in self.groups.items()
        }
        self.dataset_length = {
            key: len(per_sequence) * self.sequence_length[key] for key, per_sequence in self.groups.items()
        }
        self.group_length = max(self.dataset_length.values())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.groups) * self.group_length

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        expanded: list[list[int]] = []
        for key, per_sequence in self.groups.items():
            per_group: list[int] = []
            target = self.sequence_length[key]
            for _, values in per_sequence.items():
                order = torch.randperm(len(values), generator=generator).tolist()
                shuffled = [values[i] for i in order]
                per_group.extend(shuffled[i % len(shuffled)] for i in range(target))
            order = torch.randperm(len(per_group), generator=generator).tolist()
            shuffled = [per_group[i] for i in order]
            expanded.append([shuffled[i % len(shuffled)] for i in range(self.group_length)])
        group_order = torch.randperm(len(expanded), generator=generator).tolist()
        for offset in range(self.group_length):
            round_order = torch.randperm(len(group_order), generator=generator).tolist()
            for idx in round_order:
                yield expanded[group_order[idx]][offset]


__all__ = [
    "BalancedSequenceSampler", "HierarchicalDatasetSequenceSampler", "UtilityMemorySelectorDataset",
    "UtilityTargets", "dataset_id_from_sequence", "utility_targets",
]
