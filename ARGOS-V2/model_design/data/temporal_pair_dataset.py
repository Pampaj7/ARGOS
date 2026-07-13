"""Mmap-backed causal SCARED-C temporal pairs for frozen stereo caches."""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

V2_ROOT = Path(__file__).resolve().parents[2]
if str(V2_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(V2_ROOT / "scripts"))

from argos_v2.cache_io import load_sequence_cache  # noqa: E402
from argos_v2.paths import CACHE_HEIGHT, CACHE_WIDTH  # noqa: E402
from argos_v2.scared_c_data import (  # noqa: E402
    load_frame_gt,
    load_sequence_info,
    read_rgb,
)
from argos_v2.sequences import accepted_sequences  # noqa: E402


SEEN_BACKBONES = ("S2M2-S", "RAFT-Stereo", "StereoAnywhere")
PRIMARY_UNSEEN_BACKBONE = "Fast-FoundationStereo"
DEFAULT_VALIDATION_SEQUENCES = (
    "dataset_7_keyframe_1",
    "dataset_7_keyframe_2",
    "dataset_7_keyframe_3",
    "dataset_7_keyframe_4",
)


@dataclass(frozen=True)
class TemporalPairRecord:
    backbone: str
    sequence: str
    past_index: int
    current_index: int
    past_frame_id: str
    current_frame_id: str


def resize_gt_to_cache_masked(
    disparity: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Coverage-normalized area resize; returns cache disparity and coverage."""
    coverage = cv2.resize(
        valid.astype(np.float32), (CACHE_WIDTH, CACHE_HEIGHT), interpolation=cv2.INTER_AREA
    )
    numerator = cv2.resize(
        disparity.astype(np.float32) * valid.astype(np.float32),
        (CACHE_WIDTH, CACHE_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    cache = numerator / np.maximum(coverage, 1e-6)
    cache *= CACHE_WIDTH / disparity.shape[1]
    return cache.astype(np.float32), coverage


def build_split_manifest(
    *,
    seed: int = 20260713,
    coverage_threshold: float = 0.50,
    frame_stride: int = 1,
    validation_sequences: Sequence[str] = DEFAULT_VALIDATION_SEQUENCES,
) -> dict:
    """Build a deterministic group-held-out split with no dataset_7 leakage."""
    sequences = accepted_sequences()
    validation = list(validation_sequences)
    missing = sorted(set(validation) - set(sequences))
    if missing:
        raise ValueError(f"validation sequences are not accepted SCARED-C: {missing}")
    train = [sequence for sequence in sequences if sequence not in validation]
    return {
        "version": 1,
        "seed": seed,
        "split_unit": "complete SCARED-C sequence",
        "train_sequences": train,
        "validation_sequences": validation,
        "training_backbones": list(SEEN_BACKBONES),
        "primary_unseen_backbone": PRIMARY_UNSEEN_BACKBONE,
        "primary_cache_coverage_threshold": coverage_threshold,
        "evaluation_coverage_thresholds": [0.05, 0.25, 0.50, 0.90],
        "frame_stride": frame_stride,
        "causal_pair": "t-1 -> t; no future frame",
        "balance_policy": "equal record count per seen backbone",
        "selection_policy": (
            "All dataset_7 keyframes are held out together; Fast-FoundationStereo is "
            "excluded from training, validation selection, architecture selection, and tuning."
        ),
    }


class TemporalPairDataset(Dataset):
    """Return consecutive cache-grid pairs without crossing a sequence boundary.

    ``max_pairs_per_sequence`` selects one deterministic contiguous clip per
    sequence. Set ``random_clip_start=False`` for deterministic prefix validation.
    Cache arrays remain numpy memmaps and are opened lazily in each worker.
    """

    def __init__(
        self,
        backbones: Sequence[str],
        sequences: Sequence[str],
        *,
        coverage_threshold: float = 0.50,
        frame_stride: int = 1,
        max_pairs_per_sequence: int | None = None,
        random_clip_start: bool = False,
        seed: int = 20260713,
    ) -> None:
        if not 0 <= coverage_threshold <= 1:
            raise ValueError("coverage_threshold must be in [0,1]")
        if frame_stride < 1:
            raise ValueError("frame_stride must be >=1")
        self.backbones = tuple(backbones)
        self.sequences = tuple(sequences)
        self.coverage_threshold = float(coverage_threshold)
        self.frame_stride = int(frame_stride)
        self.max_pairs_per_sequence = max_pairs_per_sequence
        self.random_clip_start = random_clip_start
        self.seed = seed
        self._handles: dict[tuple[str, str], tuple] = {}
        self._infos = {sequence: load_sequence_info(sequence) for sequence in self.sequences}
        self.records = self._build_records()
        counts = {backbone: sum(r.backbone == backbone for r in self.records) for backbone in self.backbones}
        if len(set(counts.values())) != 1:
            raise RuntimeError(f"backbone sampling is not balanced: {counts}")

    def _clip_start(self, sequence: str, available: int, take: int) -> int:
        if not self.random_clip_start or take >= available:
            return 0
        digest = hashlib.sha256(f"{self.seed}:{sequence}".encode()).digest()
        return int.from_bytes(digest[:8], "little") % (available - take + 1)

    def _build_records(self) -> list[TemporalPairRecord]:
        records: list[TemporalPairRecord] = []
        for sequence in self.sequences:
            frame_ids = self._infos[sequence].frame_ids
            current_indices = list(range(self.frame_stride, len(frame_ids), self.frame_stride))
            if self.max_pairs_per_sequence is not None:
                take = min(len(current_indices), self.max_pairs_per_sequence)
                start = self._clip_start(sequence, len(current_indices), take)
                current_indices = current_indices[start : start + take]
            for backbone in self.backbones:
                _disp, _valid, cache_frame_ids, _metadata = load_sequence_cache(backbone, sequence)
                cache_ids = [str(value) for value in cache_frame_ids]
                if cache_ids != frame_ids:
                    raise RuntimeError(f"frame-ID mismatch for {backbone}/{sequence}")
                for current_index in current_indices:
                    past_index = current_index - self.frame_stride
                    records.append(
                        TemporalPairRecord(
                            backbone=backbone,
                            sequence=sequence,
                            past_index=past_index,
                            current_index=current_index,
                            past_frame_id=frame_ids[past_index],
                            current_frame_id=frame_ids[current_index],
                        )
                    )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def _cache(self, backbone: str, sequence: str):
        key = (backbone, sequence)
        if key not in self._handles:
            self._handles[key] = load_sequence_cache(backbone, sequence)
        return self._handles[key]

    @staticmethod
    def _rgb_cache(rgb: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(rgb, (CACHE_WIDTH, CACHE_HEIGHT), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(np.ascontiguousarray(resized)).permute(2, 0, 1).float()

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        disparities, validity, _frame_ids, _metadata = self._cache(record.backbone, record.sequence)
        raw = np.asarray(disparities[record.current_index], dtype=np.float32)
        past = np.asarray(disparities[record.past_index], dtype=np.float32)
        raw_valid = np.asarray(validity[record.current_index], dtype=np.uint8) > 0
        past_valid = np.asarray(validity[record.past_index], dtype=np.uint8) > 0

        info = self._infos[record.sequence]
        current_rgb = read_rgb(info.seq_dir / "left" / f"{record.current_frame_id}.png")
        past_rgb = read_rgb(info.seq_dir / "left" / f"{record.past_frame_id}.png")
        gt_native, gt_valid_native = load_frame_gt(info, record.current_frame_id)
        gt, coverage = resize_gt_to_cache_masked(gt_native, gt_valid_native)
        gt_valid = coverage > self.coverage_threshold

        return {
            "raw": torch.from_numpy(raw.copy())[None],
            "past": torch.from_numpy(past.copy())[None],
            "raw_valid": torch.from_numpy(raw_valid.copy())[None],
            "past_valid": torch.from_numpy(past_valid.copy())[None],
            "current_rgb": self._rgb_cache(current_rgb),
            "past_rgb": self._rgb_cache(past_rgb),
            "gt": torch.from_numpy(np.ascontiguousarray(gt))[None],
            "gt_coverage": torch.from_numpy(np.ascontiguousarray(coverage))[None],
            "gt_valid": torch.from_numpy(np.ascontiguousarray(gt_valid))[None],
            "backbone": record.backbone,
            "sequence": record.sequence,
            "past_frame_id": record.past_frame_id,
            "current_frame_id": record.current_frame_id,
            "past_index": record.past_index,
            "current_index": record.current_index,
        }

    def describe(self) -> dict:
        return {
            "backbones": list(self.backbones),
            "sequences": list(self.sequences),
            "coverage_threshold": self.coverage_threshold,
            "frame_stride": self.frame_stride,
            "max_pairs_per_sequence": self.max_pairs_per_sequence,
            "random_clip_start": self.random_clip_start,
            "seed": self.seed,
            "pair_count": len(self),
            "records": [asdict(record) for record in self.records[:3]],
        }


def write_split_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
