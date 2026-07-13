"""Small universal learned selector for BiDA-aligned long-memory candidates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from model_design.models.learned_t1_refiner import LearnedT1Refiner


@dataclass
class LearnedPPMOutput:
    disparity: torch.Tensor
    update: torch.Tensor
    candidate_logits: torch.Tensor
    play_weights: torch.Tensor
    raw_abstain_weight: torch.Tensor
    aggregated_memory: torch.Tensor
    aggregated_validity: torch.Tensor
    g_error: torch.Tensor
    c_memory: torch.Tensor
    delta: torch.Tensor
    tau: torch.Tensor
    error_logits: torch.Tensor
    memory_logits: torch.Tensor


class CandidateEncoder(nn.Module):
    def __init__(self, channels: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(12, channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -1.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LearnedPPMSelectorRefiner(nn.Module):
    """Per-pixel memory play weights plus the validated bounded residual form.

    Inputs are candidate stacks ``[B,M,C,H,W]``. A fixed raw-abstain logit of
    zero competes with every memory. The correction branch is the same A2 CNN
    used by the validated t-1 refiner. Final output remains exactly identity at
    initialization because its delta head is zero initialized.
    """

    def __init__(self, channels: int = 16, tau_px: float = 3.0) -> None:
        super().__init__()
        self.selector = CandidateEncoder(channels)
        self.refiner = LearnedT1Refiner("A2", tau_px=tau_px)

    @staticmethod
    def normalized_candidate_inputs(
        raw: torch.Tensor,
        current_valid: torch.Tensor,
        evidence: Mapping[str, torch.Tensor],
        ages: torch.Tensor,
    ) -> torch.Tensor:
        memory = evidence["aligned_past_disparity"]
        b, m, _c, h, w = memory.shape
        raw_m = raw[:, None].expand(-1, m, -1, -1, -1)
        valid_m = current_valid[:, None].expand(-1, m, -1, -1, -1)
        disagreement = raw_m - memory
        age_map = (ages.to(raw).view(1, m, 1, 1, 1) / 8.0).expand(b, m, 1, h, w)
        tensors = (
            raw_m.clamp(0, 64) / 64,
            memory.clamp(0, 64) / 64,
            disagreement.clamp(-16, 16) / 16,
            disagreement.abs().clamp(0, 16) / 16,
            valid_m.float(),
            evidence["aligned_validity"].float(),
            evidence["warp_support"].float(),
            evidence["forward_backward_error"].clamp(0, 8) / 8,
            evidence["forward_backward_confidence"].clamp(0, 1),
            evidence["photometric_residual"].clamp(0, 1),
            evidence["flow_magnitude"].clamp(0, 32) / 32,
            age_map,
        )
        return torch.cat(tensors, dim=2)

    def forward(
        self,
        raw: torch.Tensor,
        current_valid: torch.Tensor,
        evidence: Mapping[str, torch.Tensor],
        ages: torch.Tensor,
    ) -> LearnedPPMOutput:
        memory = evidence["aligned_past_disparity"]
        b, m, _c, h, w = memory.shape
        inputs = self.normalized_candidate_inputs(raw, current_valid, evidence, ages)
        logits = self.selector(inputs.reshape(b * m, 12, h, w)).reshape(b, m, 1, h, w)
        candidate_valid = evidence["aligned_validity"].bool() & evidence["warp_support"].bool()
        logits = logits.masked_fill(~candidate_valid, -20.0)
        raw_logit = torch.zeros((b, 1, 1, h, w), device=raw.device, dtype=raw.dtype)
        all_weights = torch.softmax(torch.cat((raw_logit, logits), dim=1), dim=1)
        raw_weight = all_weights[:, 0]
        play_weights = all_weights[:, 1:] * candidate_valid.to(all_weights.dtype)
        memory_mass = play_weights.sum(dim=1)
        normalized_memory_weights = play_weights / memory_mass[:, None].clamp_min(1e-8)
        aggregated = (normalized_memory_weights * memory).sum(dim=1)
        aggregated_valid = candidate_valid.any(dim=1)
        aggregated = torch.where(aggregated_valid, aggregated, raw)

        t1_style = {
            "aligned_past_disparity": aggregated,
            "current_valid": current_valid,
            "aligned_validity": aggregated_valid,
            "warp_support": aggregated_valid,
        }
        base = self.refiner(raw, t1_style)
        update = base.update * memory_mass
        return LearnedPPMOutput(
            disparity=raw + update,
            update=update,
            candidate_logits=logits,
            play_weights=play_weights,
            raw_abstain_weight=raw_weight,
            aggregated_memory=aggregated,
            aggregated_validity=aggregated_valid,
            g_error=base.g_error,
            c_memory=base.c_memory * memory_mass,
            delta=base.delta,
            tau=base.tau,
            error_logits=base.error_logits,
            memory_logits=base.memory_logits,
        )
