"""Controlled frozen-representation ranker with an explicit raw/null option."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


VARIANTS = ("P0", "P1", "P2", "P3", "P4", "P5", "P6")


@dataclass(frozen=True)
class RepresentationSelectorOutput:
    logits: torch.Tensor  # [B,1+M,H,W], index 0 is raw/null
    probabilities: torch.Tensor
    candidate_logits: torch.Tensor


def _fixed_projection(out_channels: int, in_channels: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(out_channels, in_channels, generator=generator)
    return F.normalize(matrix, dim=1)


class LocalRGBPairEncoder(nn.Module):
    """Small local RGB CNN used only by P1/P5, at the shared patch grid."""

    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(9, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, current: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((current, memory, (current - memory).abs()), dim=1))


class DINORepresentationSelector(nn.Module):
    """Same shared ranker for P0-P6; only its 64-channel descriptor changes.

    Inputs:
    * ``geom``: ``[B,M,12,H,W]`` normalized universal BiDA evidence;
    * ``rgb``: ``[B,1+M,3,H,W]`` RGB in [0,1];
    * ``dino``: ``[B,L,M,64,H,W]`` frozen pair descriptors for layers
      `(5,11,17,23)`;
    * ``candidate_valid``: ``[B,M,1,H,W]``.

    P2 uses layer 23, P3 layer 5, P4 fuses all four layers, P5 adds the local
    RGB CNN and P6 adds geometry. No backbone identity can enter this contract.
    """

    def __init__(self, variant: str, channels: int = 32) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        self.variant = variant
        self.rgb_encoder = LocalRGBPairEncoder(64) if variant in {"P1", "P5"} else None
        self.register_buffer("geom_projection", _fixed_projection(64, 12, 20260713))
        self.ranker = nn.Sequential(
            nn.Conv2d(64, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 1, 1),
        )
        nn.init.zeros_(self.ranker[-1].weight)
        nn.init.constant_(self.ranker[-1].bias, -1.5)

    @staticmethod
    def _unit_channels(value: torch.Tensor) -> torch.Tensor:
        return F.normalize(value, dim=2, eps=1e-6)

    def descriptors(self, geom: torch.Tensor, rgb: torch.Tensor, dino: torch.Tensor) -> torch.Tensor:
        b, m, _channels, h, w = geom.shape
        geom_descriptor = torch.einsum("oc,bmchw->bmohw", self.geom_projection, geom)
        if self.variant == "P0":
            return geom_descriptor
        rgb_descriptor = None
        if self.rgb_encoder is not None:
            current = rgb[:, :1].expand(-1, m, -1, -1, -1).reshape(b * m, 3, h, w)
            memory = rgb[:, 1:].reshape(b * m, 3, h, w)
            rgb_descriptor = self.rgb_encoder(current, memory).reshape(b, m, 64, h, w)
        if self.variant == "P1":
            assert rgb_descriptor is not None
            return rgb_descriptor
        layer_descriptor = dino[:, 3] if self.variant == "P2" else dino[:, 0]
        if self.variant in {"P4", "P5", "P6"}:
            layer_descriptor = F.normalize(dino, dim=3, eps=1e-6).mean(dim=1)
        if self.variant == "P5":
            assert rgb_descriptor is not None
            return self._unit_channels(layer_descriptor) + self._unit_channels(rgb_descriptor)
        if self.variant == "P6":
            return self._unit_channels(layer_descriptor) + self._unit_channels(geom_descriptor)
        return layer_descriptor

    def forward(
        self,
        geom: torch.Tensor,
        rgb: torch.Tensor,
        dino: torch.Tensor,
        candidate_valid: torch.Tensor,
    ) -> RepresentationSelectorOutput:
        descriptor = self.descriptors(geom, rgb, dino)
        b, m, _channels, h, w = descriptor.shape
        candidate_logits = self.ranker(descriptor.reshape(b * m, 64, h, w)).reshape(b, m, h, w)
        valid = candidate_valid[:, :, 0].bool()
        candidate_logits = candidate_logits.masked_fill(~valid, -20.0)
        raw_logit = torch.zeros((b, 1, h, w), dtype=candidate_logits.dtype, device=candidate_logits.device)
        logits = torch.cat((raw_logit, candidate_logits), dim=1)
        return RepresentationSelectorOutput(logits, torch.softmax(logits, dim=1), candidate_logits)


def selector_targets(
    errors: torch.Tensor,
    candidate_valid: torch.Tensor,
    *,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw-aware best class and per-memory usefulness labels."""
    if errors.ndim != 5 or errors.shape[1] != candidate_valid.shape[1] + 1:
        raise ValueError("errors must be [B,1+M,1,H,W]")
    raw = errors[:, 0]
    memory = errors[:, 1:].masked_fill(~candidate_valid.bool(), torch.inf)
    best_memory_error, best_memory = memory[:, :, 0].min(dim=1)
    use_memory = best_memory_error + margin < raw[:, 0]
    target = torch.where(use_memory, best_memory + 1, torch.zeros_like(best_memory))
    useful = (memory + margin < raw[:, None]).to(errors.dtype)
    return target, useful


__all__ = [
    "DINORepresentationSelector",
    "RepresentationSelectorOutput",
    "VARIANTS",
    "selector_targets",
]
