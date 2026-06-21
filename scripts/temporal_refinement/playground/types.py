from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class TemporalBatch:
    """Canonical tensors for playground experiments.

    Shapes:
      rgb:            [B, T, 3, H, W]
      s2m2_s_disp:    [B, T, 1, H, W]
      s2m2_l_disp:    [B, T, 1, H, W]
      sav_disp:       [B, T, 1, H, W]
      gt_disp/depth:  optional [B, T, 1, H, W]
    """

    rgb: torch.Tensor
    s2m2_s_disp: torch.Tensor
    s2m2_l_disp: torch.Tensor
    sav_disp: torch.Tensor
    gt_disp: torch.Tensor | None = None
    gt_depth_mm: torch.Tensor | None = None
    valid_mask: torch.Tensor | None = None
    semantic_masks: torch.Tensor | None = None
    motion: dict[str, torch.Tensor] | None = None
    sequence_ids: list[str] = field(default_factory=list)


@dataclass
class FusionOutput:
    fused_disparity: torch.Tensor
    source_weights: torch.Tensor
    alpha_map: torch.Tensor
    reset_map: torch.Tensor
    residual_map: torch.Tensor
    uncertainty_map: torch.Tensor
    hidden_state: torch.Tensor | tuple[torch.Tensor, ...] | None
    diagnostics: dict[str, torch.Tensor | float | str]
