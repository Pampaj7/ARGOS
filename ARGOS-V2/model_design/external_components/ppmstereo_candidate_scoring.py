"""ARGOS-candidate memory-scoring variant, NOT a faithful PPMStereo reproduction.

See ppmstereo.py in this same directory for the formula that matches the original
PPMStereo repository exactly (verified via assert against the original class method).

This module is a separate, ARGOS-specific scoring variant inspired by PPMStereo's general
idea (quality + similarity - redundancy, then top-K + play weights) but adds redundancy and
age terms the original does not have. It exists to test whether that variant outperforms the
faithful formula in the component-combination probe — an ablation candidate, not a port.

Original repository reference: https://github.com/cocowy1/PPMStereo
Original path inspected: models/core/ppmstereo.py
Source commit inspected: d0ccf7705145502c1eea49e7be0ddeafbcfd6a08
"""
from __future__ import annotations

import torch


def score_memory(
    quality: torch.Tensor,
    similarity: torch.Tensor,
    redundancy: torch.Tensor,
    validity: torch.Tensor,
    age: torch.Tensor | None = None,
    age_weight: float = 0.05,
) -> torch.Tensor:
    """Score memory candidates. All tensors broadcast to [B, M]. Invalid candidates
    receive a large negative score."""
    score = quality + similarity - redundancy
    if age is not None:
        score = score - age_weight * age
    return score.masked_fill(validity <= 0, -1e9)


def select_topk_and_weights(scores: torch.Tensor, k: int, min_score: float = -1e8) -> tuple[torch.Tensor, torch.Tensor]:
    """Select top-K indices and normalized play weights. If all candidates are poor/invalid
    for a batch item, weights are zero but indices remain deterministic."""
    if scores.ndim != 2:
        raise ValueError(f"scores must be [B,M], got {tuple(scores.shape)}")
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(k, scores.shape[1])
    values, indices = torch.topk(scores, k=k, dim=1)
    good = values > min_score
    safe_values = values.masked_fill(~good, -1e9)
    weights = torch.softmax(safe_values, dim=1).masked_fill(~good, 0.0)
    denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return indices, weights / denom


def aggregate_memory(memory: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Aggregate selected memory. memory: [B, M, C, ...], indices: [B, K], weights: [B, K]."""
    if memory.ndim < 3:
        raise ValueError("memory must be [B,M,C,...]")
    b, _m = memory.shape[:2]
    gather_shape = (b, indices.shape[1]) + (1,) * (memory.ndim - 2)
    expanded_idx = indices.view(gather_shape).expand((b, indices.shape[1]) + memory.shape[2:])
    selected = torch.gather(memory, dim=1, index=expanded_idx)
    weight_shape = (b, indices.shape[1]) + (1,) * (memory.ndim - 2)
    return (selected * weights.view(weight_shape)).sum(dim=1)
