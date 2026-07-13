"""Mmap-backed exact-age causal clips for ARGOS v2 long-memory experiments."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from argos_v2.cache_io import load_sequence_cache
from argos_v2.scared_c_data import load_frame_gt, load_sequence_info, read_rgb
from model_design.data.temporal_pair_dataset import TemporalPairDataset, resize_gt_to_cache_masked


@dataclass(frozen=True)
class TemporalMemoryRecord:
    backbone: str
    sequence: str
    current_index: int
    current_frame_id: str


class TemporalMemoryDataset(Dataset):
    """Return current frame and exact past ages without sequence crossing."""

    def __init__(
        self,
        backbones: Sequence[str],
        sequences: Sequence[str],
        *,
        ages: Sequence[int] = (1, 2, 4, 8),
        max_samples_per_sequence: int | None = None,
        random_clip_start: bool = False,
        seed: int = 20260713,
    ) -> None:
        self.backbones = tuple(backbones)
        self.sequences = tuple(sequences)
        self.ages = tuple(int(age) for age in ages)
        if not self.ages or tuple(sorted(self.ages)) != self.ages or self.ages[0] <= 0:
            raise ValueError("ages must be positive and ascending")
        self.max_samples_per_sequence = max_samples_per_sequence
        self.random_clip_start = random_clip_start
        self.seed = seed
        self._infos = {sequence: load_sequence_info(sequence) for sequence in self.sequences}
        self._handles: dict[tuple[str, str], tuple] = {}
        self.records = self._build_records()
        counts = {
            backbone: sum(record.backbone == backbone for record in self.records)
            for backbone in self.backbones
        }
        if len(set(counts.values())) != 1:
            raise RuntimeError(f"unbalanced backbone records: {counts}")

    def _build_records(self) -> list[TemporalMemoryRecord]:
        records = []
        first_index = max(self.ages)
        for sequence in self.sequences:
            frame_ids = self._infos[sequence].frame_ids
            indices = list(range(first_index, len(frame_ids)))
            if self.max_samples_per_sequence is not None:
                take = min(len(indices), self.max_samples_per_sequence)
                start = 0
                if self.random_clip_start and take < len(indices):
                    digest = hashlib.sha256(f"{self.seed}:{sequence}".encode()).digest()
                    start = int.from_bytes(digest[:8], "little") % (len(indices) - take + 1)
                indices = indices[start : start + take]
            for backbone in self.backbones:
                _d, _v, cache_ids, _m = load_sequence_cache(backbone, sequence)
                if [str(item) for item in cache_ids] != frame_ids:
                    raise RuntimeError(f"frame-ID mismatch for {backbone}/{sequence}")
                records.extend(
                    TemporalMemoryRecord(backbone, sequence, index, frame_ids[index])
                    for index in indices
                )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def _cache(self, backbone: str, sequence: str):
        key = (backbone, sequence)
        if key not in self._handles:
            self._handles[key] = load_sequence_cache(backbone, sequence)
        return self._handles[key]

    def __getitem__(self, item: int) -> dict:
        record = self.records[item]
        disparities, validity, _ids, _metadata = self._cache(record.backbone, record.sequence)
        info = self._infos[record.sequence]
        current_id = info.frame_ids[record.current_index]
        past_indices = [record.current_index - age for age in self.ages]
        past_ids = [info.frame_ids[index] for index in past_indices]
        gt_native, gt_valid_native = load_frame_gt(info, current_id)
        gt, coverage = resize_gt_to_cache_masked(gt_native, gt_valid_native)
        current_rgb = read_rgb(info.seq_dir / "left" / f"{current_id}.png")
        past_rgb = [read_rgb(info.seq_dir / "left" / f"{frame_id}.png") for frame_id in past_ids]
        return {
            "raw": torch.from_numpy(np.asarray(disparities[record.current_index], dtype=np.float32).copy())[None],
            "raw_valid": torch.from_numpy((np.asarray(validity[record.current_index]) > 0).copy())[None],
            "past": torch.stack(
                [torch.from_numpy(np.asarray(disparities[index], dtype=np.float32).copy())[None] for index in past_indices]
            ),
            "past_valid": torch.stack(
                [torch.from_numpy((np.asarray(validity[index]) > 0).copy())[None] for index in past_indices]
            ),
            "current_rgb": TemporalPairDataset._rgb_cache(current_rgb),
            "past_rgb": torch.stack([TemporalPairDataset._rgb_cache(rgb) for rgb in past_rgb]),
            "gt": torch.from_numpy(np.ascontiguousarray(gt))[None],
            "gt_coverage": torch.from_numpy(np.ascontiguousarray(coverage))[None],
            "backbone": record.backbone,
            "sequence": record.sequence,
            "current_index": record.current_index,
            "current_frame_id": current_id,
            "past_frame_ids": past_ids,
            "ages": torch.tensor(self.ages, dtype=torch.long),
        }
