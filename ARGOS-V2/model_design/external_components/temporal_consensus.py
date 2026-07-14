"""Cross-Memory Consensus Correction (CMC) — ARGOS-original, zero parameters.

Formulation and predeclared gates: ``model_design/CONSENSUS_AUDIT.md``.

Inputs are BiDA-aligned past disparities on the current cache grid (the
``aligned_past_disparity`` / ``aligned_validity`` / ``warp_support`` fields of
``bidavideo.temporal_disparity_evidence``). Everything here is causal by
construction because the alignment inputs are.

All functions are pure numpy on ``[K,H,W]`` stacks; no torch, no state.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConsensusFields:
    """Per-pixel consensus statistics over K aligned memories."""

    median: np.ndarray  # [H,W] float32, NaN where count == 0
    spread: np.ndarray  # [H,W] float32 MAD, NaN where count == 0
    count: np.ndarray   # [H,W] int16 valid witness count


@dataclass(frozen=True)
class ConsensusConfig:
    """One CMC operating point. Defaults are placeholders; the validated
    values come from the stage-1 sweep, never hand-picked."""

    min_count: int = 3
    spread_max: float = 0.5
    disagree_min: float = 1.0
    kappa: float = 1.0
    bound: float = 3.0

    def label(self) -> str:
        return (
            f"n{self.min_count}_s{self.spread_max:g}"
            f"_d{self.disagree_min:g}_k{self.kappa:g}"
        )


def consensus_fields(
    aligned: np.ndarray, aligned_valid: np.ndarray
) -> ConsensusFields:
    """Median / MAD / count over the valid witnesses at each pixel.

    ``aligned``: [K,H,W] float; ``aligned_valid``: [K,H,W] bool.
    """
    if aligned.ndim != 3 or aligned.shape != aligned_valid.shape:
        raise ValueError(
            f"aligned {aligned.shape} and aligned_valid {aligned_valid.shape} "
            "must share [K,H,W]"
        )
    values = np.where(aligned_valid, aligned.astype(np.float32), np.nan)
    count = aligned_valid.sum(axis=0).astype(np.int16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(values, axis=0)
        spread = np.nanmedian(np.abs(values - median[None]), axis=0)
    return ConsensusFields(
        median.astype(np.float32), spread.astype(np.float32), count
    )


def consensus_correction(
    raw: np.ndarray,
    fields: ConsensusFields,
    config: ConsensusConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Bounded correction toward the consensus where raw is the outlier.

    Returns ``(refined [H,W] float32, gate [H,W] bool)``. Pixels with an open
    gate move toward the median by at most ``config.bound`` px; all other
    pixels return raw unchanged (identity-preserving by construction).
    """
    if raw.shape != fields.median.shape:
        raise ValueError(f"raw {raw.shape} vs fields {fields.median.shape}")
    disagreement = np.abs(raw - fields.median)
    gate = (
        (fields.count >= config.min_count)
        & np.isfinite(fields.median)
        & np.isfinite(fields.spread)
        & (fields.spread <= config.spread_max)
        & (disagreement >= config.disagree_min + config.kappa * fields.spread)
    )
    delta = np.clip(
        np.nan_to_num(fields.median - raw), -config.bound, config.bound
    )
    refined = np.where(gate, raw + delta, raw).astype(np.float32)
    return refined, gate


def sweep_grid() -> list[ConsensusConfig]:
    """The predeclared stage-1 grid — must match CONSENSUS_AUDIT.md."""
    return [
        ConsensusConfig(min_count=n, spread_max=s, disagree_min=d, kappa=k)
        for n in (3, 4)
        for s in (0.25, 0.5, 1.0)
        for d in (0.5, 1.0, 2.0)
        for k in (0.0, 1.0)
    ]
