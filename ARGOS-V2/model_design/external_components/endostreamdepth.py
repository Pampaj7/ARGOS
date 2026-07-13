"""Explicit causal multi-scale state interface for ARGOS v2.

This is an ARGOS adaptation of EndoStreamDepth's persistent multi-level state,
not its DINO/DPT/Mamba architecture. The concrete E2-E5 operator is a generic
ConvGRU baseline and is named accordingly.

Contracts
---------
* scale features/states: ``[B,C,H_s,W_s]``;
* reset mask: ``[B]`` bool;
* forget maps: ``[B,1,H_s,W_s]`` in ``[0,1]``;
* frame indices: ``[B]`` integer, strictly increasing within each sequence;
* sequence IDs: one explicit string per batch element.

State is passed in and returned explicitly. Modules contain no global temporal
cache, so sequence crossing cannot happen silently.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class CausalState:
    """Serializable explicit streaming state for one ordered batch."""

    scales: tuple[str, ...]
    tensors: tuple[torch.Tensor, ...]
    sequence_ids: tuple[str, ...]
    frame_indices: torch.Tensor
    update_counts: torch.Tensor

    @property
    def batch_size(self) -> int:
        return len(self.sequence_ids)

    def by_scale(self) -> dict[str, torch.Tensor]:
        return dict(zip(self.scales, self.tensors, strict=True))

    def detach(self) -> "CausalState":
        return CausalState(
            self.scales,
            tuple(value.detach() for value in self.tensors),
            self.sequence_ids,
            self.frame_indices.detach(),
            self.update_counts.detach(),
        )

    def clone(self) -> "CausalState":
        return CausalState(
            self.scales,
            tuple(value.clone() for value in self.tensors),
            tuple(self.sequence_ids),
            self.frame_indices.clone(),
            self.update_counts.clone(),
        )

    def serialize(self) -> dict:
        return {
            "version": 1,
            "scales": self.scales,
            "tensors": tuple(value.detach().cpu() for value in self.tensors),
            "sequence_ids": self.sequence_ids,
            "frame_indices": self.frame_indices.detach().cpu(),
            "update_counts": self.update_counts.detach().cpu(),
        }

    @classmethod
    def restore(cls, payload: Mapping, *, device: torch.device | str | None = None) -> "CausalState":
        if payload.get("version") != 1:
            raise ValueError("unsupported state serialization version")
        target = torch.device(device) if device is not None else None
        tensors = tuple(value.to(target) if target is not None else value for value in payload["tensors"])
        frame_indices = payload["frame_indices"].to(target) if target is not None else payload["frame_indices"]
        update_counts = payload["update_counts"].to(target) if target is not None else payload["update_counts"]
        return cls(tuple(payload["scales"]), tensors, tuple(payload["sequence_ids"]), frame_indices, update_counts)


def initialize_state(
    features: Mapping[str, torch.Tensor],
    sequence_ids: Sequence[str],
) -> CausalState:
    """Create zero state matching explicit per-scale input feature maps."""
    if not features:
        raise ValueError("at least one state scale is required")
    scales = tuple(features)
    batch_sizes = {value.shape[0] for value in features.values()}
    if len(batch_sizes) != 1:
        raise ValueError("all scale features must share a batch size")
    batch_size = batch_sizes.pop()
    if len(sequence_ids) != batch_size:
        raise ValueError("one sequence ID is required per batch element")
    for scale, value in features.items():
        if value.ndim != 4:
            raise ValueError(f"{scale} feature must be [B,C,H,W]")
    first = next(iter(features.values()))
    return CausalState(
        scales=scales,
        tensors=tuple(torch.zeros_like(features[scale]) for scale in scales),
        sequence_ids=tuple(str(item) for item in sequence_ids),
        frame_indices=torch.full((batch_size,), -1, dtype=torch.long, device=first.device),
        update_counts=torch.zeros(batch_size, dtype=torch.long, device=first.device),
    )


def reset_selected(
    state: CausalState,
    reset_mask: torch.Tensor,
    *,
    sequence_ids: Sequence[str] | None = None,
) -> CausalState:
    """Zero selected batch elements without disturbing other streams."""
    mask = reset_mask.to(state.frame_indices.device, dtype=torch.bool).flatten()
    if mask.shape != (state.batch_size,):
        raise ValueError("reset_mask must be [B]")
    identifiers = list(state.sequence_ids)
    if sequence_ids is not None:
        if len(sequence_ids) != state.batch_size:
            raise ValueError("sequence_ids must have length B")
        for index, reset in enumerate(mask.tolist()):
            if reset:
                identifiers[index] = str(sequence_ids[index])
    tensors = []
    for value in state.tensors:
        updated = value.clone()
        updated[mask] = 0
        tensors.append(updated)
    frame_indices = state.frame_indices.clone()
    update_counts = state.update_counts.clone()
    frame_indices[mask] = -1
    update_counts[mask] = 0
    return CausalState(state.scales, tuple(tensors), tuple(identifiers), frame_indices, update_counts)


def state_statistics(state: CausalState) -> dict[str, torch.Tensor]:
    stats = {}
    for scale, value in state.by_scale().items():
        flattened = value.float().flatten(1)
        stats[f"{scale}_l2"] = torch.linalg.vector_norm(flattened, dim=1)
        stats[f"{scale}_mean_abs"] = flattened.abs().mean(dim=1)
        stats[f"{scale}_max_abs"] = flattened.abs().amax(dim=1)
    stats["update_counts"] = state.update_counts
    return stats


class ConvGRUCell(nn.Module):
    """Generic spatial gated-state baseline; not an EndoStreamDepth Mamba."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.gates = nn.Conv2d(2 * channels, 2 * channels, 3, padding=1)
        self.candidate = nn.Conv2d(2 * channels, channels, 3, padding=1)

    def forward(self, current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        if current.shape != previous.shape or current.shape[1] != self.channels:
            raise ValueError("current and previous must share [B,C,H,W] state shape")
        reset, update = torch.sigmoid(self.gates(torch.cat((current, previous), dim=1))).chunk(2, dim=1)
        candidate = torch.tanh(self.candidate(torch.cat((current, reset * previous), dim=1)))
        return (1.0 - update) * previous + update * candidate


class ExplicitMultiScaleState(nn.Module):
    """Causal one-frame update over explicit scale-keyed ConvGRU states."""

    def __init__(self, scales: Sequence[str], channels: int) -> None:
        super().__init__()
        self.scales = tuple(scales)
        if not self.scales or len(set(self.scales)) != len(self.scales):
            raise ValueError("scales must be non-empty and unique")
        self.channels = int(channels)
        self.cells = nn.ModuleDict({scale: ConvGRUCell(channels) for scale in self.scales})

    def forward(
        self,
        features: Mapping[str, torch.Tensor],
        state: CausalState | None,
        *,
        sequence_ids: Sequence[str],
        frame_indices: torch.Tensor,
        reset_mask: torch.Tensor | None = None,
        forget: Mapping[str, torch.Tensor] | None = None,
    ) -> tuple[dict[str, torch.Tensor], CausalState]:
        if tuple(features) != self.scales:
            raise ValueError(f"expected ordered scales {self.scales}, got {tuple(features)}")
        if any(value.shape[1] != self.channels for value in features.values()):
            raise ValueError(f"every scale must have {self.channels} channels")
        if state is None:
            state = initialize_state(features, sequence_ids)
        if state.scales != self.scales or state.batch_size != next(iter(features.values())).shape[0]:
            raise ValueError("state scale or batch contract changed; initialize explicitly")
        incoming_ids = tuple(str(item) for item in sequence_ids)
        indices = frame_indices.to(state.frame_indices.device, dtype=torch.long).flatten()
        if indices.shape != (state.batch_size,):
            raise ValueError("frame_indices must be [B]")
        reset = torch.zeros(state.batch_size, dtype=torch.bool, device=indices.device)
        if reset_mask is not None:
            reset = reset_mask.to(indices.device, dtype=torch.bool).flatten()
            if reset.shape != (state.batch_size,):
                raise ValueError("reset_mask must be [B]")
        changed = torch.tensor(
            [old != new for old, new in zip(state.sequence_ids, incoming_ids, strict=True)],
            device=indices.device,
            dtype=torch.bool,
        )
        if torch.any(changed & ~reset):
            raise RuntimeError("sequence crossing requires an explicit reset")
        if reset.any():
            state = reset_selected(state, reset, sequence_ids=incoming_ids)
        previous_indices = state.frame_indices
        initialized = previous_indices >= 0
        if torch.any(initialized & (indices <= previous_indices)):
            raise RuntimeError("frames must be strictly increasing; future/reordered access rejected")

        previous = state.by_scale()
        outputs = {}
        for scale in self.scales:
            old = previous[scale]
            if old.shape != features[scale].shape:
                raise ValueError(f"state shape changed at {scale}")
            if forget is not None and scale in forget:
                strength = forget[scale].to(old).clamp(0, 1)
                if strength.shape != (old.shape[0], 1, old.shape[2], old.shape[3]):
                    raise ValueError(f"forget map for {scale} must be [B,1,H,W]")
                old = old * (1.0 - strength)
            outputs[scale] = self.cells[scale](features[scale], old)
        new_state = CausalState(
            self.scales,
            tuple(outputs[scale] for scale in self.scales),
            incoming_ids,
            indices.clone(),
            state.update_counts + 1,
        )
        return outputs, new_state


__all__ = [
    "CausalState",
    "ConvGRUCell",
    "ExplicitMultiScaleState",
    "initialize_state",
    "reset_selected",
    "state_statistics",
]
