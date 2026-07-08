"""Magnitude Proposal-Critic Refiner (MPC).

Audit-driven change over EGBM: keep EGBM's detector/damping/router safety machinery,
but add a separate low-frequency large-magnitude proposal. The oracle-gap audit showed
that >92% of the remaining selected-clip gap needs |delta| > 6px while EGBM's effective
update is ~2.25px. Sign/support are mostly right; magnitude is too small.

Interface matches EGBM:
    forward(x, residual_scale) -> bad_logit, p_bad, residual, diagnostics
where residual is already gated/damped/trusted and refined = raw + residual.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from experimental_refiner_vx import EGBMRefiner, GRAD_CHANNELS, _norm


class MagnitudeProposalCriticRefiner(EGBMRefiner):
    N_EXPERTS = 5  # local temporal, local boundary, local catastrophic, large proposal, identity

    def __init__(self, in_channels: int = 16, residual_scale: float = 32.0, base: int = 80):
        super().__init__(in_channels=in_channels, base=base, residual_scale=residual_scale)
        c1, c2, c3 = base, base * 2, base * 4
        mem_ch = self.memory_cell.convz.out_channels
        self.large_scale = residual_scale

        self.router_head = nn.Conv2d(c1, self.N_EXPERTS, 1)
        self.large_context = nn.Sequential(
            nn.Conv2d(c3 + mem_ch, c2, 3, padding=1, bias=False), _norm(c2), nn.SiLU(inplace=True),
            nn.Conv2d(c2, c1, 3, padding=1, bias=False), _norm(c1), nn.SiLU(inplace=True),
        )
        self.large_sign_head = nn.Conv2d(c1, 1, 1)
        self.large_mag_head = nn.Conv2d(c1, 1, 1)
        self.trust_head = nn.Conv2d(c1 + mem_ch + 1, 1, 1)
        for head in (self.router_head, self.large_sign_head, self.large_mag_head, self.trust_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

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

        # Three EGBM local experts stay small. The fourth expert is a coarse large proposal.
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

        r = torch.zeros_like(mixture)
        valve = gate * damping * trust
        for _ in range(self.update_steps):
            r = r + valve * self.update_step_size * (mixture - r)

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
        }
        return bad_logit, p_bad, r, diagnostics


def magnitude_proposal_critic_refiner(in_channels: int = 16, residual_scale: float = 32.0) -> MagnitudeProposalCriticRefiner:
    return MagnitudeProposalCriticRefiner(in_channels=in_channels, residual_scale=residual_scale, base=80)

