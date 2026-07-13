"""ARGOS v2 training objectives."""

from .safety_losses import SafetyLossConfig, learned_t1_losses

__all__ = ["SafetyLossConfig", "learned_t1_losses"]
