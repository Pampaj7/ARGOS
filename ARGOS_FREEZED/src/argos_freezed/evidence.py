"""Validated universal candidate-evidence API."""
from .models.raw_multi_anchor_refiner import FEATURE_CHANNELS, MultiAnchorEvidence, build_candidate_features

__all__ = ["FEATURE_CHANNELS", "MultiAnchorEvidence", "build_candidate_features"]
