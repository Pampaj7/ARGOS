"""Adaptive Safe-Step Proposal Refiner.

Keeps MPC's proposal generator, but separates the final correction into:
    residual = small_residual + alpha_safe * large_residual

The safe-step head predicts alpha/risk from counterfactual proposal context. It is
initialized to reproduce MPC nearly exactly, then trained with the existing CPV
counterfactual loss.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from experimental_refiner_vx import GRAD_CHANNELS, _norm
from magnitude_proposal_critic_refiner import MagnitudeProposalCriticRefiner


class AdaptiveSafeStepProposalRefiner(MagnitudeProposalCriticRefiner):
    LARGE_EXPERT = 4

    def __init__(self, in_channels: int = 16, residual_scale: float = 32.0, base: int = 80):
        super().__init__(in_channels=in_channels, residual_scale=residual_scale, base=base)
        c1 = base
        mem_ch = self.memory_cell.convz.out_channels
        self.verifier = nn.Sequential(
            nn.Conv2d(c1 + mem_ch + 6, c1 // 2, 3, padding=1, bias=False),
            _norm(c1 // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1 // 2, 4, 1),
        )
        nn.init.zeros_(self.verifier[-1].weight)
        with torch.no_grad():
            # benefit high, risk very low, alpha nearly one, expected-gain neutral.
            self.verifier[-1].bias.copy_(torch.tensor([4.0, -6.0, 6.0, 0.0]))

    def _fixed_point(self, residual: torch.Tensor, valve: torch.Tensor) -> torch.Tensor:
        r = torch.zeros_like(residual)
        for _ in range(self.update_steps):
            r = r + valve * self.update_step_size * (residual - r)
        return r

    def _edge_field(self, residual: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
        smoothed = F.avg_pool2d(residual, 3, stride=1, padding=1)
        residual = (1.0 - boundary) * smoothed + boundary * residual
        return residual * (1.0 - torch.sigmoid(self.boundary_atten) * boundary)

    def forward(self, x: torch.Tensor, residual_scale: float | None = None):
        scale = float(self.large_scale if residual_scale is None else residual_scale)
        f1 = self.stage1(self.stem(x))
        f2 = self.stage2(self.down1(f1))
        f3 = self.stage3(self.down2(f2))

        mem = self._temporal_memory(x)
        mem_at_f2 = F.interpolate(mem, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.up2(torch.cat([F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False), f2, mem_at_f2], dim=1))
        u1 = self.up1(torch.cat([F.interpolate(u2, size=f1.shape[-2:], mode="bilinear", align_corners=False), f1], dim=1))
        feat = self.refine(u1)

        bad_logit = self.bad_head(feat)
        p_bad = torch.sigmoid(bad_logit)
        mem_full = F.interpolate(mem, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        damping = torch.sigmoid(self.damping_head(torch.cat([feat, mem_full], dim=1)))
        boundary = torch.sigmoid(self.boundary_head(self.boundary_branch(torch.cat([x[:, GRAD_CHANNELS], f1], dim=1))))
        trust = torch.sigmoid(self.trust_head(torch.cat([feat, mem_full, boundary], dim=1)))

        local_scale = min(3.0, scale)
        local = [local_scale * torch.tanh(head(feat)) for head in self.expert_heads]
        large_ctx = self.large_context(torch.cat([f3, mem], dim=1))
        large_sign = torch.tanh(self.large_sign_head(large_ctx))
        large_mag = scale * torch.sigmoid(self.large_mag_head(large_ctx))
        large = F.interpolate(large_sign * large_mag, size=feat.shape[-2:], mode="bilinear", align_corners=False)

        router = torch.softmax(self.router_head(feat), dim=1)
        small_mix = sum(router[:, k : k + 1] * local[k] for k in range(len(local)))
        large_mix = router[:, self.LARGE_EXPERT : self.LARGE_EXPERT + 1] * large
        small_mix = self._edge_field(small_mix, boundary)
        large_mix = self._edge_field(large_mix, boundary)

        grid = F.adaptive_avg_pool2d(f3, (8, 10))
        delta = 0.3 * torch.tanh(self.threshold_head(grid))
        delta = F.interpolate(delta, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        gate = torch.sigmoid((p_bad - (self.base_threshold + delta)) / self.gate_tau)
        base_valve = gate * damping * trust
        small_residual = self._fixed_point(small_mix, base_valve)
        large_residual = self._fixed_point(large_mix, base_valve)
        pre_residual = small_residual + large_residual

        logits = self.verifier(torch.cat([
            feat,
            mem_full,
            boundary,
            large_residual.abs() / max(scale, 1e-6),
            small_residual.abs() / max(scale, 1e-6),
            trust,
            damping,
            gate,
        ], dim=1))
        benefit = torch.sigmoid(logits[:, 0:1])
        new_bad3_risk = torch.sigmoid(logits[:, 1:2])
        alpha = torch.sigmoid(logits[:, 2:3])
        expected_gain = scale * torch.tanh(logits[:, 3:4])
        alpha_safe = alpha * (1.0 - new_bad3_risk)
        residual = small_residual + alpha_safe * large_residual

        diagnostics = {
            "damping": damping,
            "trust": trust,
            "router_weights": router,
            "boundary_confidence": boundary,
            "dynamic_threshold": self.base_threshold + delta,
            "temporal_memory": mem,
            "mixture_residual": small_mix + large_mix,
            "large_proposal": large,
            "large_magnitude": F.interpolate(large_mag, size=feat.shape[-2:], mode="bilinear", align_corners=False),
            "gate": gate,
            "small_residual": small_residual,
            "large_residual": large_residual,
            "pre_verifier_residual": pre_residual,
            "verifier_benefit": benefit,
            "verifier_new_bad3_risk": new_bad3_risk,
            "verifier_safe_alpha": alpha,
            "verifier_expected_gain": expected_gain,
            "verifier_logits": logits,
            "verifier_safe": alpha_safe,
        }
        return bad_logit, p_bad, residual, diagnostics


def adaptive_safe_step_proposal_refiner(in_channels: int = 16, residual_scale: float = 32.0) -> AdaptiveSafeStepProposalRefiner:
    return AdaptiveSafeStepProposalRefiner(in_channels=in_channels, residual_scale=residual_scale, base=80)
