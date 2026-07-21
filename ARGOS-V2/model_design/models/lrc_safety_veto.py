"""Frozen frame-relative left--right-consistency safety veto for ARGOS v2.

This module deliberately has no parameters and cannot open a temporal update.
It receives an authorization produced by the existing frozen utility selector
and can only close it when the current raw disparity is relatively
stereo-consistent within its own frame.  Frame-relative ranking avoids an
absolute LRC scale or a backbone identifier.
"""
from __future__ import annotations

import torch


def frame_relative_lrc_gate(
    residual: torch.Tensor,
    valid: torch.Tensor,
    *,
    quantile: float,
) -> torch.Tensor:
    """Return pixels with raw LRC at/above each frame's valid quantile.

    Args:
        residual: Non-negative LRC residual, shape ``[B,1,H,W]``.
        valid: LRC support mask with exactly the same shape.
        quantile: Closed interval ``[0, 1]``.  ``.90`` retains the upper
            tenth of the valid residual ranking in each frame.

    Invalid/non-finite values are always rejected.  The operation is
    deterministic and uses no backbone or sequence identity.
    """
    if residual.ndim != 4 or residual.shape[1] != 1:
        raise ValueError("residual must be [B,1,H,W]")
    if valid.shape != residual.shape:
        raise ValueError("valid must match residual shape")
    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("quantile must be in [0,1]")
    support = valid.bool() & torch.isfinite(residual)
    result = torch.zeros_like(support)
    for index in range(residual.shape[0]):
        mask = support[index, 0]
        if not mask.any():
            continue
        values = residual[index, 0][mask]
        threshold = torch.quantile(values, float(quantile))
        result[index, 0] = mask & (residual[index, 0] >= threshold)
    return result


def lrc_safety_veto(
    base_authorization: torch.Tensor,
    raw_lrc_residual: torch.Tensor,
    raw_lrc_valid: torch.Tensor,
    *,
    quantile: float,
) -> torch.Tensor:
    """Apply a one-way LRC veto to a frozen base authorization.

    This is ``base_authorization AND frame_relative_lrc_gate``.  It can never
    change a rejected pixel into an accepted pixel.
    """
    if base_authorization.shape != raw_lrc_residual.shape:
        raise ValueError("base authorization and LRC residual must match")
    return base_authorization.bool() & frame_relative_lrc_gate(
        raw_lrc_residual, raw_lrc_valid, quantile=quantile,
    )


__all__ = ["frame_relative_lrc_gate", "lrc_safety_veto"]
