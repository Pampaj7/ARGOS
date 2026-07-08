"""Counterfactual Proposal Verifier (CPV) refiner.

CPV keeps MPC's large proposal generator and adds an explicit verifier that predicts:
benefit, new-Bad3 risk, and safe step fraction for the already-proposed correction.
The new head is safe-initialized near pass-through, so an MPC checkpoint can warm-start
without throwing away its oracle-gap recovery.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from experimental_refiner_vx import GRAD_CHANNELS, _norm
from magnitude_proposal_critic_refiner import MagnitudeProposalCriticRefiner


class CounterfactualProposalVerifierRefiner(MagnitudeProposalCriticRefiner):
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
            # benefit high, risk low, alpha high, expected-gain neutral.
            self.verifier[-1].bias.copy_(torch.tensor([4.0, -4.0, 4.0, 0.0]))

    def _fixed_point(self, mixture: torch.Tensor, valve: torch.Tensor) -> torch.Tensor:
        r = torch.zeros_like(mixture)
        for _ in range(self.update_steps):
            r = r + valve * self.update_step_size * (mixture - r)
        return r

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

        candidates = [*local, large, torch.zeros_like(large)]
        router = torch.softmax(self.router_head(feat), dim=1)
        mixture = sum(router[:, k : k + 1] * candidates[k] for k in range(self.N_EXPERTS))
        smoothed = F.avg_pool2d(mixture, 3, stride=1, padding=1)
        mixture = (1.0 - boundary) * smoothed + boundary * mixture
        mixture = mixture * (1.0 - torch.sigmoid(self.boundary_atten) * boundary)

        grid = F.adaptive_avg_pool2d(f3, (8, 10))
        delta = 0.3 * torch.tanh(self.threshold_head(grid))
        delta = F.interpolate(delta, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        gate = torch.sigmoid((p_bad - (self.base_threshold + delta)) / self.gate_tau)

        base_valve = gate * damping * trust
        pre_verifier_residual = self._fixed_point(mixture, base_valve)
        verifier_in = torch.cat([
            feat,
            mem_full,
            boundary,
            mixture.abs() / max(scale, 1e-6),
            large.abs() / max(scale, 1e-6),
            trust,
            damping,
            gate,
        ], dim=1)
        v = self.verifier(verifier_in)
        benefit = torch.sigmoid(v[:, 0:1])
        new_bad3_risk = torch.sigmoid(v[:, 1:2])
        safe_alpha = torch.sigmoid(v[:, 2:3])
        expected_gain = scale * torch.tanh(v[:, 3:4])
        verifier_safe = benefit * (1.0 - new_bad3_risk) * safe_alpha
        residual = self._fixed_point(mixture, base_valve * verifier_safe)

        diagnostics = {
            "damping": damping,
            "trust": trust,
            "router_weights": router,
            "boundary_confidence": boundary,
            "dynamic_threshold": self.base_threshold + delta,
            "temporal_memory": mem,
            "mixture_residual": mixture,
            "large_proposal": large,
            "large_magnitude": F.interpolate(large_mag, size=feat.shape[-2:], mode="bilinear", align_corners=False),
            "gate": gate,
            "pre_verifier_residual": pre_verifier_residual,
            "verifier_benefit": benefit,
            "verifier_new_bad3_risk": new_bad3_risk,
            "verifier_safe_alpha": safe_alpha,
            "verifier_expected_gain": expected_gain,
            "verifier_logits": v,
            "verifier_safe": verifier_safe,
        }
        return bad_logit, p_bad, residual, diagnostics


def counterfactual_proposal_verifier_refiner(in_channels: int = 16, residual_scale: float = 32.0) -> CounterfactualProposalVerifierRefiner:
    return CounterfactualProposalVerifierRefiner(in_channels=in_channels, residual_scale=residual_scale, base=80)
