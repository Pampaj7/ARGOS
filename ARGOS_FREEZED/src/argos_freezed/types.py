"""Public frozen ARGOS v2 data contracts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class RawAnchor:
    disparity: torch.Tensor
    validity: torch.Tensor
    left_rgb: torch.Tensor
    frame_id: str
    frame_index: int
    timestamp: float | None
    provenance: str = "independent_frozen_stereo"


@dataclass(frozen=True)
class RefinementResult:
    output_disparity: torch.Tensor
    raw_disparity: torch.Tensor
    proposal_disparity: torch.Tensor
    selected_anchor_age: torch.Tensor
    selected_aligned_anchor: torch.Tensor
    selection_score: torch.Tensor
    fusion_weight: torch.Tensor
    accepted_mask: torch.Tensor
    support_mask: torch.Tensor
    validity_mask: torch.Tensor
    forward_backward_consistency: torch.Tensor
    update_magnitude: torch.Tensor
    diagnostic_metadata: dict[str, Any]
