"""Causal PPMStereo-derived memory primitives for ARGOS v2.

This module is the single canonical implementation of reusable long-memory logic.
It deliberately separates:

* ``*_faithful`` helpers reproducing released PPMStereo score mechanics;
* the deployable, backbone-agnostic ARGOS causal adapter;
* optional learned selectors, which must not be described as original QAM.

The original repository is ``external/PPMStereo`` at commit
``d0ccf7705145502c1eea49e7be0ddeafbcfd6a08``.  Its full read-out depends on
stereo cost volumes, context features and recurrent hidden states and is
non-causal.  See ``model_design/PPMSTEREO_AUDIT.md``.

Tensor contracts
----------------
Spatial tensors use ``[B,C,H,W]``. Candidate stacks use ``[B,M,C,H,W]``;
candidate scores use ``[B,M]``; selected indices/weights use ``[B,K]``.
Validity tensors are boolean ``[B,1,H,W]``. Disparity is positive-left in
pixels of its own grid. No operation here changes disparity units.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

from . import bidavideo as bida

ROOT = Path(__file__).resolve().parents[3]
PPM_ROOT = ROOT / "external/PPMStereo"


@dataclass(frozen=True)
class MemoryEntry:
    """One causal memory candidate.

    ``disparity`` and ``validity`` are the stored source-frame maps. Fields
    prefixed by ``aligned`` and all evidence maps live on the current target
    grid. Scores are optional ``[B]`` tensors populated by a selector.
    """

    sequence_id: str
    frame_index: int
    age: int
    disparity: torch.Tensor
    validity: torch.Tensor
    rgb: torch.Tensor | None = None
    aligned_disparity: torch.Tensor | None = None
    aligned_validity: torch.Tensor | None = None
    warp_support: torch.Tensor | None = None
    forward_backward_error: torch.Tensor | None = None
    forward_backward_confidence: torch.Tensor | None = None
    photometric_residual: torch.Tensor | None = None
    disparity_disagreement: torch.Tensor | None = None
    absolute_disparity_disagreement: torch.Tensor | None = None
    flow_magnitude: torch.Tensor | None = None
    learned_feature: torch.Tensor | None = None
    quality_score: torch.Tensor | None = None
    similarity_score: torch.Tensor | None = None
    redundancy_score: torch.Tensor | None = None
    final_selection_score: torch.Tensor | None = None
    selection_count: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if self.age < 0:
            raise ValueError("age must be non-negative")
        if self.disparity.ndim != 4 or self.disparity.shape[1] != 1:
            raise ValueError("disparity must be [B,1,H,W]")
        validity = _valid_b1hw(self.validity)
        if validity.shape != self.disparity.shape:
            raise ValueError("validity must match disparity [B,1,H,W]")

    @property
    def current_grid_disparity(self) -> torch.Tensor:
        return self.aligned_disparity if self.aligned_disparity is not None else self.disparity

    @property
    def current_grid_validity(self) -> torch.Tensor:
        value = self.aligned_validity if self.aligned_validity is not None else self.validity
        return _valid_b1hw(value)


@dataclass(frozen=True)
class TopKSelection:
    indices: torch.Tensor
    scores: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class AggregatedMemory:
    value: torch.Tensor
    valid: torch.Tensor
    effective_weights: torch.Tensor
    selected_values: torch.Tensor
    selected_validity: torch.Tensor


class CausalMemoryBank:
    """Deterministic sequence-local bank; entries are append-only in time."""

    def __init__(self, max_entries: int | None = None) -> None:
        if max_entries is not None and max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self._sequence_id: str | None = None
        self._entries: list[MemoryEntry] = []

    @property
    def sequence_id(self) -> str | None:
        return self._sequence_id

    @property
    def entries(self) -> tuple[MemoryEntry, ...]:
        return tuple(self._entries)

    def reset(self, sequence_id: str | None = None) -> None:
        self._entries.clear()
        self._sequence_id = sequence_id

    def append(self, entry: MemoryEntry) -> None:
        """Append one observed frame, resetting automatically on a new sequence."""
        if entry.age != 0:
            raise ValueError("stored entries must have age=0; age is assigned at query time")
        if self._sequence_id != entry.sequence_id:
            self.reset(entry.sequence_id)
        if self._entries and entry.frame_index <= self._entries[-1].frame_index:
            raise ValueError("memory frames must be appended in strictly causal order")
        self._entries.append(entry)
        if self.max_entries is not None and len(self._entries) > self.max_entries:
            del self._entries[: len(self._entries) - self.max_entries]

    def candidates(
        self, current_frame_index: int, ages: Sequence[int] | None = None
    ) -> tuple[MemoryEntry, ...]:
        """Return past-only candidates, optionally at exact requested ages."""
        if ages is not None and any(age <= 0 for age in ages):
            raise ValueError("candidate ages must be positive")
        requested = None if ages is None else set(int(age) for age in ages)
        result: list[MemoryEntry] = []
        for entry in self._entries:
            age = current_frame_index - entry.frame_index
            if age <= 0:
                continue
            if requested is None or age in requested:
                result.append(replace(entry, age=age))
        result.sort(key=lambda item: item.age)
        return tuple(result)


def _valid_b1hw(valid: torch.Tensor) -> torch.Tensor:
    if valid.ndim == 3:
        valid = valid[:, None]
    if valid.ndim != 4 or valid.shape[1] != 1:
        raise ValueError(f"validity must be [B,1,H,W], got {tuple(valid.shape)}")
    return valid.bool()


def _import_original_ppm_class():
    sys.path.insert(0, str(PPM_ROOT))
    try:
        from models.core.ppmstereo import PPMStereo  # type: ignore
    finally:
        try:
            sys.path.remove(str(PPM_ROOT))
        except ValueError:
            pass
    return PPMStereo


def compute_qk_similarity_faithful(q: torch.Tensor, k: torch.Tensor, t: int) -> torch.Tensor:
    """Exact standalone ``PPMStereo.compute_qk_similarity`` reproduction.

    Args:
        q, k: ``[B,C,T,H,W]`` original-style context embeddings.
        t: temporal length, equal to ``T``.
    Returns:
        Full query/key matrix ``[B,1,T,T]``.
    """
    if q.shape != k.shape or q.ndim != 5 or q.shape[2] != t:
        raise ValueError("q and k must share [B,C,T,H,W] with T=t")
    b, c, _t, h, w = q.shape
    q_flat = q.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    k_flat = k.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    pool = torch.nn.AdaptiveMaxPool2d((max(h // 4, 1), max(w // 4, 1)))
    q_pool = pool(q_flat).mean(dim=1).reshape(b, t, -1)
    k_pool = pool(k_flat).mean(dim=1).reshape(b, t, -1)
    return F.cosine_similarity(q_pool.unsqueeze(1), k_pool.unsqueeze(2), dim=-1).unsqueeze(1)


# Backward-compatible name used by the earlier component probe.
compute_qk_similarity_exact = compute_qk_similarity_faithful


def quality_aware_scores_faithful(
    similarity: torch.Tensor,
    frame_confidence: torch.Tensor,
    strive_time: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Released-code score formula for one query and ``M`` candidates."""
    if similarity.shape != frame_confidence.shape or similarity.ndim != 2:
        raise ValueError("similarity and confidence must share [B,M]")
    if strive_time is None:
        strive_time = torch.ones_like(similarity)
    if strive_time.shape != similarity.shape:
        raise ValueError("strive_time must match similarity")
    m = similarity.shape[1]
    penalty = torch.exp(-strive_time / (strive_time.sum(dim=1, keepdim=True) + m))
    return penalty * similarity + frame_confidence, penalty


quality_aware_scores = quality_aware_scores_faithful


def faithful_play_modulation(scores: torch.Tensor) -> torch.Tensor:
    """Released-code selected-score/mean modulation, not disparity weights."""
    if scores.ndim != 2:
        raise ValueError("scores must be [B,K]")
    return scores / scores.mean().clamp_min(1e-6)


def align_entry_with_bida(
    entry: MemoryEntry,
    *,
    current_frame_index: int,
    current_disparity: torch.Tensor,
    current_validity: torch.Tensor,
    current_rgb: torch.Tensor,
    flow_current_to_memory: torch.Tensor,
    flow_memory_to_current: torch.Tensor,
) -> MemoryEntry:
    """Align a past entry by calling the canonical BiDA evidence implementation."""
    age = current_frame_index - entry.frame_index
    if age <= 0:
        raise ValueError("memory entry must strictly precede current frame")
    if entry.rgb is None:
        raise ValueError("entry.rgb is required for BiDA photometric evidence")
    evidence = bida.temporal_disparity_evidence(
        current_disparity,
        entry.disparity,
        flow_current_to_memory,
        flow_memory_to_current,
        current_valid=current_validity,
        past_valid=entry.validity,
        current_rgb=current_rgb,
        past_rgb=entry.rgb,
    )
    return replace(
        entry,
        age=age,
        aligned_disparity=evidence.aligned_past_disparity,
        aligned_validity=evidence.aligned_validity,
        warp_support=evidence.warp_support,
        forward_backward_error=evidence.forward_backward_error,
        forward_backward_confidence=evidence.forward_backward_confidence,
        photometric_residual=evidence.photometric_residual,
        disparity_disagreement=evidence.signed_disparity_disagreement,
        absolute_disparity_disagreement=evidence.absolute_disparity_disagreement,
        flow_magnitude=evidence.flow_magnitude,
    )


def masked_spatial_cosine(
    query: torch.Tensor,
    memory: torch.Tensor,
    support: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine similarity on spatially corresponding valid support.

    ``query`` is ``[B,C,H,W]``; ``memory`` is ``[B,M,C,H,W]``; ``support`` is
    ``[B,M,1,H,W]``. Returns similarity ``[B,M]`` and valid candidates ``[B,M]``.
    """
    if query.ndim != 4 or memory.ndim != 5 or support.ndim != 5:
        raise ValueError("expected query [B,C,H,W], memory/support [B,M,C,H,W]")
    if memory.shape[:1] + memory.shape[2:] != query.shape:
        raise ValueError("query and memory spatial/channel shapes differ")
    if support.shape[:2] != memory.shape[:2] or support.shape[2] != 1 or support.shape[-2:] != memory.shape[-2:]:
        raise ValueError("support must be [B,M,1,H,W]")
    mask = support.to(memory.dtype)
    q = query[:, None].expand_as(memory)
    numerator = (q * memory * mask).sum(dim=(2, 3, 4))
    q_norm = (q.square() * mask).sum(dim=(2, 3, 4)).sqrt()
    m_norm = (memory.square() * mask).sum(dim=(2, 3, 4)).sqrt()
    support_count = support.sum(dim=(2, 3, 4))
    valid = support_count > 0
    similarity = numerator / (q_norm * m_norm).clamp_min(eps)
    return similarity.masked_fill(~valid, 0.0), valid


def spatial_redundancy_matrix(
    features: torch.Tensor,
    support: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pairwise cosine redundancy on each candidates' intersecting support.

    Args:
        features: ``[B,M,C,H,W]`` aligned universal features.
        support: ``[B,M,1,H,W]`` aligned validity/support.
    Returns:
        cosine matrix and pair-valid matrix, both ``[B,M,M]``.
    """
    if features.ndim != 5 or support.ndim != 5:
        raise ValueError("features/support must be candidate stacks")
    if support.shape[:2] != features.shape[:2] or support.shape[2] != 1 or support.shape[-2:] != features.shape[-2:]:
        raise ValueError("support must be [B,M,1,H,W]")
    left = features[:, :, None]
    right = features[:, None, :]
    intersection = support[:, :, None] & support[:, None, :]
    mask = intersection.to(features.dtype)
    numerator = (left * right * mask).sum(dim=(3, 4, 5))
    left_norm = (left.square() * mask).sum(dim=(3, 4, 5)).sqrt()
    right_norm = (right.square() * mask).sum(dim=(3, 4, 5)).sqrt()
    pair_valid = intersection.sum(dim=(3, 4, 5)) > 0
    cosine = numerator / (left_norm * right_norm).clamp_min(eps)
    return cosine.masked_fill(~pair_valid, 0.0), pair_valid


def max_off_diagonal_redundancy(
    matrix: torch.Tensor, pair_valid: torch.Tensor
) -> torch.Tensor:
    """Maximum similarity to another valid candidate, returning ``[B,M]``."""
    if matrix.shape != pair_valid.shape or matrix.ndim != 3:
        raise ValueError("matrix and pair_valid must share [B,M,M]")
    m = matrix.shape[1]
    diagonal = torch.eye(m, device=matrix.device, dtype=torch.bool)[None]
    eligible = pair_valid & ~diagonal
    values = matrix.masked_fill(~eligible, -torch.inf)
    result = values.max(dim=2).values
    return torch.where(torch.isfinite(result), result, torch.zeros_like(result))


def deterministic_argos_scores(
    quality: torch.Tensor,
    similarity: torch.Tensor,
    redundancy: torch.Tensor,
    candidate_valid: torch.Tensor,
    *,
    redundancy_weight: float = 0.25,
) -> torch.Tensor:
    """Universal adapted score; this is not the original PPMStereo QAM."""
    if not (quality.shape == similarity.shape == redundancy.shape == candidate_valid.shape):
        raise ValueError("all score inputs must share [B,M]")
    score = quality + similarity - redundancy_weight * redundancy
    return score.masked_fill(~candidate_valid.bool(), -torch.inf)


def deterministic_topk(
    scores: torch.Tensor,
    k: int,
    *,
    ages: torch.Tensor | None = None,
    candidate_valid: torch.Tensor | None = None,
) -> TopKSelection:
    """Stable top-K; exact score ties prefer smaller age, then input order."""
    if scores.ndim != 2 or k <= 0:
        raise ValueError("scores must be [B,M] and k positive")
    b, m = scores.shape
    k = min(k, m)
    if candidate_valid is None:
        candidate_valid = torch.isfinite(scores)
    if candidate_valid.shape != scores.shape:
        raise ValueError("candidate_valid must match scores")
    if ages is None:
        ages = torch.arange(m, device=scores.device).expand(b, m)
    elif ages.ndim == 1:
        ages = ages[None].expand(b, -1)
    if ages.shape != scores.shape:
        raise ValueError("ages must be [M] or [B,M]")
    age_order = torch.argsort(ages, dim=1, descending=False, stable=True)
    ordered_scores = scores.gather(1, age_order)
    ordered_valid = candidate_valid.gather(1, age_order)
    ordered_scores = ordered_scores.masked_fill(~ordered_valid, -torch.inf)
    rank = torch.argsort(ordered_scores, dim=1, descending=True, stable=True)[:, :k]
    indices = age_order.gather(1, rank)
    selected_scores = scores.gather(1, indices)
    selected_valid = candidate_valid.gather(1, indices) & torch.isfinite(selected_scores)
    return TopKSelection(indices, selected_scores, selected_valid)


def normalized_play_weights(
    selection: TopKSelection,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Softmax play weights over valid selected candidates; all-invalid -> zero."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = selection.scores / temperature
    logits = logits.masked_fill(~selection.valid, -torch.inf)
    all_invalid = ~selection.valid.any(dim=1, keepdim=True)
    logits = torch.where(all_invalid, torch.zeros_like(logits), logits)
    weights = torch.softmax(logits, dim=1).masked_fill(~selection.valid, 0.0)
    return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)


def aggregate_selected_memory(
    memory: torch.Tensor,
    validity: torch.Tensor,
    selection: TopKSelection,
    weights: torch.Tensor,
) -> AggregatedMemory:
    """Validity-renormalized selected-memory aggregation on the current grid."""
    if memory.ndim != 5 or validity.ndim != 5:
        raise ValueError("memory and validity must be [B,M,C,H,W]/[B,M,1,H,W]")
    b, _m, c, h, w = memory.shape
    if validity.shape != (b, memory.shape[1], 1, h, w):
        raise ValueError("validity shape mismatch")
    if weights.shape != selection.indices.shape:
        raise ValueError("weights must match selected indices [B,K]")
    k = selection.indices.shape[1]
    value_idx = selection.indices[:, :, None, None, None].expand(b, k, c, h, w)
    valid_idx = selection.indices[:, :, None, None, None].expand(b, k, 1, h, w)
    selected = memory.gather(1, value_idx)
    selected_valid = validity.bool().gather(1, valid_idx) & selection.valid[:, :, None, None, None]
    spatial_weights = weights[:, :, None, None, None] * selected_valid.to(memory.dtype)
    denominator = spatial_weights.sum(dim=1, keepdim=True)
    effective = spatial_weights / denominator.clamp_min(1e-12)
    value = (selected * effective).sum(dim=1)
    valid = denominator[:, 0] > 0
    return AggregatedMemory(value, valid, effective, selected, selected_valid)


def stack_entry_maps(
    entries: Iterable[MemoryEntry], field: str
) -> torch.Tensor:
    """Stack a required current-grid entry field into candidate dimension ``M``."""
    values = []
    for entry in entries:
        value = getattr(entry, field)
        if value is None:
            raise ValueError(f"entry age={entry.age} has no {field}")
        values.append(value)
    if not values:
        raise ValueError("at least one memory entry is required")
    return torch.stack(values, dim=1)


def topk_with_modulation(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compatibility helper for the earlier faithful component probe."""
    selection = deterministic_topk(scores, k)
    modulation = faithful_play_modulation(selection.scores)
    weights = modulation.clamp_min(0)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return selection.indices, selection.scores, weights


def update_strive_time(strive_time: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    updated = strive_time.clone()
    updated.scatter_add_(1, indices, torch.ones_like(indices, dtype=strive_time.dtype))
    return updated


def self_check() -> None:
    torch.manual_seed(0)
    q = torch.randn(1, 4, 3, 8, 8)
    k = torch.randn(1, 4, 3, 8, 8)
    local = compute_qk_similarity_faithful(q, k, t=3)
    PPMStereo = _import_original_ppm_class()
    original = PPMStereo.compute_qk_similarity(None, q, k, t=3)
    assert torch.allclose(local, original, atol=1e-6)


if __name__ == "__main__":
    self_check()
    print("ppmstereo causal adapter ok")
