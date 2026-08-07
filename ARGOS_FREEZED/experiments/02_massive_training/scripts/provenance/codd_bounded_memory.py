"""Minimal deterministic state-reset policies for ARGOS v2 CODD-style fusion.

These policies do not learn, alter disparities, or inspect ground truth.  They
only decide whether the recurrent fused state must be re-anchored to the raw
previous-frame disparity before the current causal fusion step.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResetEvidence:
    age: int
    accumulated_update: float
    disagreement: float
    warp_support: float
    fb_confidence: float
    temporal_activation: float
    update_magnitude: float


@dataclass(frozen=True)
class BoundedMemoryPolicy:
    """Interpretable reset contract; every threshold is optional and causal."""

    name: str
    max_age: int | None = None
    accumulated_update_max: float | None = None
    disagreement_max: float | None = None
    warp_support_min: float | None = None
    fb_confidence_min: float | None = None
    temporal_activation_max: float | None = None
    update_magnitude_max: float | None = None

    def pre_reset(self, *, age: int, accumulated_update: float) -> bool:
        if age < 0:
            raise ValueError("state age must be non-negative")
        return bool(
            (self.max_age is not None and age >= self.max_age)
            or (
                self.accumulated_update_max is not None
                and accumulated_update > self.accumulated_update_max
            )
        )

    def evidence_reset(self, evidence: ResetEvidence) -> bool:
        return bool(
            (self.disagreement_max is not None and evidence.disagreement > self.disagreement_max)
            or (self.warp_support_min is not None and evidence.warp_support < self.warp_support_min)
            or (self.fb_confidence_min is not None and evidence.fb_confidence < self.fb_confidence_min)
            or (
                self.temporal_activation_max is not None
                and evidence.temporal_activation > self.temporal_activation_max
            )
            or (
                self.update_magnitude_max is not None
                and evidence.update_magnitude > self.update_magnitude_max
            )
        )

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, value: dict) -> "BoundedMemoryPolicy":
        allowed = set(cls.__dataclass_fields__)
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown reset-policy fields: {sorted(unknown)}")
        return cls(**value)


def advance_state_age(age: int, *, reset: bool) -> int:
    """Current output becomes a one-step-old state for the next pair."""
    if age < 0:
        raise ValueError("state age must be non-negative")
    return 1 if reset else age + 1


__all__ = ["ResetEvidence", "BoundedMemoryPolicy", "advance_state_age"]
