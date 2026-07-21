"""Minimal frozen feature-support guard for ARGOS v2.

The guard is not a learned model.  It fits descriptive statistics only on
SCARED-C training penultimate detector features and accepts a pixel when its
feature vector lies below a frozen support-score threshold.  Tensor features
use ``[B,C,H,W]``; scores and masks use ``[B,1,H,W]``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from torch import nn


FIT_DATASET = "SCARED-C"
FIT_SPLIT = "training"
THRESHOLD_SPLIT = "calibration"
METHODS = ("diagonal", "shrinkage", "knn")


@dataclass(frozen=True)
class SupportProvenance:
    dataset: str
    split: str
    backbones: tuple[str, ...]
    sequences: tuple[str, ...]
    seed: int


@dataclass(frozen=True)
class SupportReference:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    precision: np.ndarray
    reference_bank: np.ndarray
    shrinkage: float
    knn_k: int
    provenance: SupportProvenance

    @property
    def feature_dim(self) -> int:
        return int(self.mean.size)

    @property
    def memory_bytes(self) -> int:
        return int(
            self.mean.nbytes + self.std.nbytes + self.precision.nbytes
            + self.reference_bank.nbytes
        )


def validate_fit_provenance(provenance: SupportProvenance) -> None:
    if provenance.dataset != FIT_DATASET or provenance.split != FIT_SPLIT:
        raise ValueError("support statistics may be fitted only on SCARED-C training")
    forbidden = {"Fast-FoundationStereo", "CREStereo"}
    if forbidden.intersection(provenance.backbones):
        raise ValueError("unseen backbones cannot enter support fitting")


def validate_threshold_provenance(provenance: SupportProvenance) -> None:
    if provenance.dataset != FIT_DATASET or provenance.split != THRESHOLD_SPLIT:
        raise ValueError("support thresholds may be selected only on SCARED-C calibration")
    forbidden = {"Fast-FoundationStereo", "CREStereo"}
    if forbidden.intersection(provenance.backbones):
        raise ValueError("unseen backbones cannot enter threshold selection")


def deterministic_bank_indices(length: int, bank_size: int, seed: int) -> np.ndarray:
    if length < 1 or bank_size < 1:
        raise ValueError("length and bank_size must be positive")
    take = min(length, bank_size)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(length, size=take, replace=False))


def fit_support_reference(
    features: np.ndarray,
    *,
    feature_names: Sequence[str],
    provenance: SupportProvenance,
    bank_size: int = 4096,
    knn_k: int = 5,
    variance_floor: float = 1e-6,
) -> SupportReference:
    """Fit diagonal, shrinkage-covariance and compact k-NN references."""
    validate_fit_provenance(provenance)
    value = np.asarray(features, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != len(feature_names):
        raise ValueError("features must be [N,C] and match feature_names")
    if value.shape[0] <= value.shape[1] or not np.isfinite(value).all():
        raise ValueError("finite support samples with N>C are required")
    mean = value.mean(axis=0)
    std = np.maximum(value.std(axis=0), float(variance_floor))
    standardized = (value - mean) / std
    covariance = LedoitWolf(store_precision=True, assume_centered=False).fit(standardized)
    indices = deterministic_bank_indices(len(value), bank_size, provenance.seed)
    return SupportReference(
        feature_names=tuple(feature_names),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        precision=covariance.precision_.astype(np.float32),
        reference_bank=standardized[indices].astype(np.float32),
        shrinkage=float(covariance.shrinkage_),
        knn_k=min(int(knn_k), len(indices)),
        provenance=provenance,
    )


def quantile_threshold(
    scores: np.ndarray,
    quantile: float,
    *,
    provenance: SupportProvenance,
) -> float:
    validate_threshold_provenance(provenance)
    value = np.asarray(scores, dtype=np.float64)
    value = value[np.isfinite(value)]
    if not len(value) or not 0 < quantile < 1:
        raise ValueError("finite scores and quantile in (0,1) are required")
    return float(np.quantile(value, quantile))


def save_reference(path: str | Path, reference: SupportReference) -> None:
    path = Path(path)
    np.savez_compressed(
        path,
        feature_names=np.asarray(reference.feature_names),
        mean=reference.mean,
        std=reference.std,
        precision=reference.precision,
        reference_bank=reference.reference_bank,
        shrinkage=np.asarray(reference.shrinkage),
        knn_k=np.asarray(reference.knn_k),
        dataset=np.asarray(reference.provenance.dataset),
        split=np.asarray(reference.provenance.split),
        backbones=np.asarray(reference.provenance.backbones),
        sequences=np.asarray(reference.provenance.sequences),
        seed=np.asarray(reference.provenance.seed),
    )


def load_reference(path: str | Path) -> SupportReference:
    value = np.load(Path(path), allow_pickle=False)
    provenance = SupportProvenance(
        dataset=str(value["dataset"]), split=str(value["split"]),
        backbones=tuple(str(x) for x in value["backbones"]),
        sequences=tuple(str(x) for x in value["sequences"]),
        seed=int(value["seed"]),
    )
    validate_fit_provenance(provenance)
    return SupportReference(
        feature_names=tuple(str(x) for x in value["feature_names"]),
        mean=value["mean"].astype(np.float32), std=value["std"].astype(np.float32),
        precision=value["precision"].astype(np.float32),
        reference_bank=value["reference_bank"].astype(np.float32),
        shrinkage=float(value["shrinkage"]), knn_k=int(value["knn_k"]),
        provenance=provenance,
    )


class SupportGuard(nn.Module):
    """Vectorized score computation with no backbone identity or trainable state."""

    def __init__(self, reference: SupportReference, *, chunk_size: int = 4096) -> None:
        super().__init__()
        self.feature_names = reference.feature_names
        self.knn_k = reference.knn_k
        self.chunk_size = int(chunk_size)
        self.register_buffer("mean", torch.from_numpy(reference.mean)[None, :, None, None])
        self.register_buffer("std", torch.from_numpy(reference.std)[None, :, None, None])
        self.register_buffer("precision", torch.from_numpy(reference.precision))
        self.register_buffer("reference_bank", torch.from_numpy(reference.reference_bank))

    def standardized(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 4 or features.shape[1] != self.mean.shape[1]:
            raise ValueError("features must be [B,C,H,W] with the fitted channel count")
        finite = torch.isfinite(features).all(dim=1, keepdim=True)
        safe = torch.where(torch.isfinite(features), features, self.mean)
        return (safe - self.mean) / self.std, finite

    def score(self, features: torch.Tensor, method: str) -> torch.Tensor:
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        z, finite = self.standardized(features)
        b, c, h, w = z.shape
        flat = z.permute(0, 2, 3, 1).reshape(-1, c)
        if method == "diagonal":
            score = flat.square().sum(dim=1)
        elif method == "shrinkage":
            score = torch.einsum("nc,cd,nd->n", flat, self.precision, flat)
        else:
            chunks = []
            bank = self.reference_bank.to(dtype=flat.dtype)
            for start in range(0, len(flat), self.chunk_size):
                distance = torch.cdist(flat[start:start + self.chunk_size], bank)
                chunks.append(distance.topk(self.knn_k, largest=False, dim=1).values.mean(dim=1))
            score = torch.cat(chunks)
        result = score.reshape(b, 1, h, w)
        return torch.where(finite, result, torch.full_like(result, float("inf")))

    @staticmethod
    def accept(score: torch.Tensor, threshold: float) -> torch.Tensor:
        if not math_isfinite_positive(threshold):
            raise ValueError("threshold must be finite and positive")
        return torch.isfinite(score) & (score <= float(threshold))

    def forward(self, features: torch.Tensor, method: str, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
        score = self.score(features, method)
        return score, self.accept(score, threshold)


def math_isfinite_positive(value: float) -> bool:
    return bool(np.isfinite(value) and value > 0)


def guarded_output(
    raw: torch.Tensor,
    proposal_update: torch.Tensor,
    error_authorization: torch.Tensor,
    support_authorization: torch.Tensor,
) -> torch.Tensor:
    """Bit-exact raw for either rejection; unchanged A2 update for acceptance."""
    accepted = error_authorization.bool() & support_authorization.bool()
    return torch.where(accepted, raw + proposal_update, raw)


def support_mask(score: torch.Tensor, threshold: float, granularity: str = "pixel") -> torch.Tensor:
    """Return pixel support or one deterministic median frame decision."""
    pixel = SupportGuard.accept(score, threshold)
    if granularity == "pixel":
        return pixel
    if granularity != "frame":
        raise ValueError("granularity must be pixel or frame")
    flat = score.flatten(1)
    finite = torch.isfinite(flat)
    safe = torch.where(finite, flat, torch.full_like(flat, float("nan")))
    medians = torch.nanmedian(safe, dim=1).values
    accepted = torch.isfinite(medians) & (medians <= float(threshold))
    return accepted[:, None, None, None].expand_as(score)


__all__ = [
    "FIT_DATASET", "FIT_SPLIT", "THRESHOLD_SPLIT", "METHODS",
    "SupportProvenance", "SupportReference", "SupportGuard",
    "deterministic_bank_indices", "fit_support_reference", "quantile_threshold",
    "save_reference", "load_reference", "guarded_output",
    "support_mask",
    "validate_fit_provenance", "validate_threshold_provenance",
]
