"""Clean-room ARGOS v2 streaming state helper.

Original repository reference: https://github.com/MedICL-VU/EndoStreamDepth
Original paths inspected:
- endostreamdepth/mamba.py
- endostreamdepth/model.py
Source commit inspected: 5abe89d9c0e09f64fdc5276d21bb5a34aa815cc6

Reason for export: preserve the per-sequence reset and one-frame-at-a-time state
lifecycle without depending on DepthAnything/DPT/Mamba internals.
Copied unchanged: no. This is a minimal clean-room implementation.
"""

from __future__ import annotations

import torch


class StreamingState:
    """Small state container for one-step causal smoke tests."""

    def __init__(self) -> None:
        self.value: torch.Tensor | None = None
        self.steps = 0

    def reset(self) -> None:
        self.value = None
        self.steps = 0

    def step(self, current: torch.Tensor, update_weight: float = 0.5) -> torch.Tensor:
        """Update state with one current feature tensor and return the new state."""
        if self.value is None:
            self.value = current.clone()
        else:
            self.value = (1.0 - update_weight) * self.value + update_weight * current
        self.steps += 1
        return self.value

    def detach(self) -> None:
        if self.value is not None:
            self.value = self.value.detach()
