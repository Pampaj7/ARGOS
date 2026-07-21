"""Contiguous causal clips built from the validated mmap temporal-pair loader."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Sequence

import torch
from torch.utils.data import Dataset

from model_design.data.temporal_pair_dataset import TemporalPairDataset


@dataclass(frozen=True)
class TemporalClipRecord:
    """Indices of consecutive ``t-1 -> t`` pairs from one stream."""

    backbone: str
    sequence: str
    pair_indices: tuple[int, ...]
    first_current_index: int
    last_current_index: int


class TemporalClipDataset(Dataset):
    """Return ``T`` ordered pairs without backbone or sequence crossing.

    Tensor fields have an added leading time dimension ``[T,...]``.  Each clip
    starts from a freshly reset state; no sample contains a future frame.
    ``clip_stride`` controls overlap between clips and is independent of the
    frame stride used by the validated pair loader.
    """

    def __init__(
        self,
        backbones: Sequence[str],
        sequences: Sequence[str],
        *,
        clip_length: int,
        coverage_threshold: float = 0.50,
        frame_stride: int = 1,
        clip_stride: int | None = None,
        max_clips_per_sequence: int | None = None,
        random_clip_selection: bool = False,
        seed: int = 20260713,
        include_right_rgb: bool = False,
    ) -> None:
        if clip_length < 2:
            raise ValueError("clip_length must be >=2")
        self.clip_length = int(clip_length)
        self.clip_stride = int(clip_stride or clip_length)
        if self.clip_stride < 1:
            raise ValueError("clip_stride must be >=1")
        self.max_clips_per_sequence = max_clips_per_sequence
        self.random_clip_selection = bool(random_clip_selection)
        self.seed = int(seed)
        self.include_right_rgb = bool(include_right_rgb)
        self.pairs = TemporalPairDataset(
            backbones,
            sequences,
            coverage_threshold=coverage_threshold,
            frame_stride=frame_stride,
            max_pairs_per_sequence=None,
            random_clip_start=False,
            seed=seed, include_right_rgb=include_right_rgb,
        )
        self.records = self._build_records()
        counts = {
            backbone: sum(record.backbone == backbone for record in self.records)
            for backbone in backbones
        }
        if len(set(counts.values())) != 1:
            raise RuntimeError(f"clip sampling is not backbone-balanced: {counts}")

    def _select(self, backbone: str, sequence: str, candidates: list[TemporalClipRecord]) -> list[TemporalClipRecord]:
        limit = self.max_clips_per_sequence
        if limit is None or len(candidates) <= limit:
            return candidates
        if not self.random_clip_selection:
            return candidates[:limit]
        digest = hashlib.sha256(f"{self.seed}:{backbone}:{sequence}".encode()).digest()
        generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))
        chosen = torch.randperm(len(candidates), generator=generator)[:limit].sort().values.tolist()
        return [candidates[index] for index in chosen]

    def _build_records(self) -> list[TemporalClipRecord]:
        groups: dict[tuple[str, str], list[int]] = {}
        for pair_index, record in enumerate(self.pairs.records):
            groups.setdefault((record.backbone, record.sequence), []).append(pair_index)
        clips: list[TemporalClipRecord] = []
        for (backbone, sequence), indices in groups.items():
            candidates = []
            for offset in range(0, len(indices) - self.clip_length + 1, self.clip_stride):
                selected = indices[offset : offset + self.clip_length]
                pair_records = [self.pairs.records[index] for index in selected]
                current = [record.current_index for record in pair_records]
                expected_step = self.pairs.frame_stride
                if any(right - left != expected_step for left, right in zip(current, current[1:])):
                    raise RuntimeError("non-contiguous temporal clip")
                candidates.append(TemporalClipRecord(
                    backbone=backbone,
                    sequence=sequence,
                    pair_indices=tuple(selected),
                    first_current_index=current[0],
                    last_current_index=current[-1],
                ))
            clips.extend(self._select(backbone, sequence, candidates))
        return clips

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        clip = self.records[index]
        frames = [self.pairs[pair_index] for pair_index in clip.pair_indices]
        result: dict = {}
        for key in frames[0]:
            values = [frame[key] for frame in frames]
            result[key] = torch.stack(values) if torch.is_tensor(values[0]) else values
        # Collate these once per clip rather than as repeated per-frame strings.
        result["backbone"] = clip.backbone
        result["sequence"] = clip.sequence
        return result

    def describe(self) -> dict:
        return {
            "clip_length": self.clip_length,
            "clip_stride": self.clip_stride,
            "max_clips_per_sequence": self.max_clips_per_sequence,
            "random_clip_selection": self.random_clip_selection,
            "include_right_rgb": self.include_right_rgb,
            "seed": self.seed,
            "clip_count": len(self),
            "pair_contract": self.pairs.describe(),
            "records": [asdict(record) for record in self.records[:3]],
        }


__all__ = ["TemporalClipDataset", "TemporalClipRecord"]
