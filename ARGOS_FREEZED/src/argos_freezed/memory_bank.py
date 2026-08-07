"""Immutable raw-only causal memory for frozen ARGOS v2 geometry-v1."""
from __future__ import annotations

import torch

from .constants import ANCHOR_AGES
from .types import RawAnchor


class RawAnchorBank:
    """Stores only independently generated raw stereo states."""

    def __init__(self, anchor_ages: tuple[int, ...] = ANCHOR_AGES) -> None:
        if tuple(anchor_ages) != ANCHOR_AGES:
            raise ValueError(f"frozen anchor ages must be {ANCHOR_AGES}")
        self._frames: dict[int, RawAnchor] = {}
        self._last_index: int | None = None

    def append_raw(self, disparity: torch.Tensor, validity: torch.Tensor, left_rgb: torch.Tensor, *,
                   frame_id: str, frame_index: int, timestamp: float | None = None,
                   provenance: str = "independent_frozen_stereo") -> None:
        if provenance != "independent_frozen_stereo":
            raise ValueError("long-term memory accepts only independently generated raw stereo")
        if self._last_index is not None and frame_index <= self._last_index:
            raise ValueError("raw frames must be appended in strictly causal order")
        if frame_index in self._frames:
            raise ValueError(f"frame index already stored: {frame_index}")
        if disparity.ndim != 4 or disparity.shape[1] != 1 or validity.shape != disparity.shape:
            raise ValueError("disparity and validity must share [B,1,H,W]")
        if disparity.shape[0] != 1 or left_rgb.shape != (1, 3, *disparity.shape[-2:]):
            raise ValueError("raw bank stores one causal stream with matching [1,3,H,W] RGB")
        valid = validity.bool()
        if not torch.isfinite(disparity[valid]).all() or bool((disparity[valid] <= 0).any()):
            raise ValueError("valid disparity must be finite positive-left")
        self._frames[frame_index] = RawAnchor(disparity.detach().clone(), valid.detach().clone(),
            left_rgb.detach().clone(), frame_id, frame_index, timestamp, provenance)
        self._last_index = frame_index
        oldest = frame_index - max(ANCHOR_AGES)
        self._frames = {index: frame for index, frame in self._frames.items() if index >= oldest}

    def anchor(self, current_index: int, age: int) -> RawAnchor | None:
        if age not in ANCHOR_AGES:
            raise ValueError(f"unsupported anchor age: {age}")
        return self._frames.get(current_index - age)

    def __len__(self) -> int:
        return len(self._frames)
