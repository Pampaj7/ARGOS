"""Candidate-Fusion Refiner (CFR) — fable wildcard experiment.

Hypothesis
----------
Every refiner in the v3 -> v4 -> SOG -> EGBM line sees only S2M2-derived features and
tries to *re-invent* the correction. But the oracle whose gap we are chasing
(oracle_all_available, selected-clip MAE 6.81 vs raw 11.32) is literally a per-pixel
argmin over candidate disparity maps that ALREADY EXIST on disk for the selected clips:
raw S2M2, fixed-EMA, adaptive-no-raft, RAFT-small-warped-EMA, and StereoAnyVideo.
No model has ever been given those candidates as input. CFR tests whether *learning the
oracle's selection* (a soft per-pixel mixture over candidates, plus a small free
residual) recovers substantially more oracle gap than disparity-only refiners can.

Architecture
------------
v4-style 3-scale inverted-residual encoder-decoder (base 96, ~6M params — capacity in
the fusion decision, not brute force), over an input stack of:
  16 v3 feature channels + 5 candidate maps /64 + 4 |candidate - raw|/64 diffs
  + 5 per-candidate availability flags (constant maps) = 30 channels.

Heads:
  * weight head (5ch) -> masked softmax over candidates (unavailable candidates get
    -inf logits, so probability mass renormalizes over what exists);
  * residual head -> bounded free residual (tanh * scale) for corrections no candidate
    provides;
  * gate head -> global correction gate g in [0,1].

    fused    = sum_k W_k * candidate_k
    refined  = raw + g * ((fused - raw) + residual)

Identity at init: weight head zero-init + a fixed +4 logit bias on the raw channel
(raw weight ~0.982), residual zero-init, gate logit bias -4 (g ~0.018) -> refined ~= raw
to within ~1e-3 px. Verified in the benchmark.

Candidate-availability robustness (the deployment story)
--------------------------------------------------------
forward(x, scale) with no candidates degrades gracefully: candidates default to raw
copies with zero availability flags, the masked softmax collapses onto raw, and CFR
acts as a plain raw-only refiner (this is also how it trains/evals on full-GT shards,
which carry no candidate maps). During training, random candidate dropout teaches the
same robustness for partial candidate sets — deployment can supply only the cheap
online candidates (fixed-EMA / adaptive are trivial temporal filters) and skip
RAFT/SAV, or supply everything offline.

Interface: forward(x, scale, candidates=None, flags=None)
  -> (gate_logit, gate, residual_out, diagnostics)
where refined = raw + residual_out (gating already applied), matching the EGBM-style
convention so the existing full-GT eval helper works unchanged.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

DISP_SCALE = 64.0
N_CANDIDATES = 5  # raw, fixed_ema, adaptive_no_raft, raftsmall, sav
CANDIDATE_KEYS = ("raw_disp", "fixed_ema_disp", "adaptive_no_raft_disp", "raftsmall_disp", "sav_disp")


def _norm(ch: int) -> nn.Module:
    return nn.GroupNorm(min(8, ch), ch)


class InvertedResidual(nn.Module):
    def __init__(self, ch: int, expansion: float = 2.5):
        super().__init__()
        hidden = int(ch * expansion)
        self.block = nn.Sequential(
            nn.Conv2d(ch, hidden, 1, bias=False), _norm(hidden), nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False), _norm(hidden), nn.SiLU(inplace=True),
            nn.Conv2d(hidden, ch, 1, bias=False), _norm(ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class CandidateFusionRefiner(nn.Module):
    def __init__(
        self,
        in_features: int = 16,
        n_candidates: int = N_CANDIDATES,
        base: int = 96,
        depths: tuple[int, int, int] = (2, 3, 4),
        expansion: float = 2.5,
        residual_scale: float = 3.0,
        raw_bias: float = 4.0,
        gate_bias: float = -4.0,
    ):
        super().__init__()
        self.n_candidates = n_candidates
        self.residual_scale = residual_scale
        in_ch = in_features + n_candidates + (n_candidates - 1) + n_candidates
        c1, c2, c3 = base, base * 2, base * 4
        self.stem = nn.Sequential(nn.Conv2d(in_ch, c1, 3, padding=1, bias=False), _norm(c1), nn.SiLU(inplace=True))
        self.stage1 = nn.Sequential(*[InvertedResidual(c1, expansion) for _ in range(depths[0])])
        self.down1 = nn.Sequential(nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False), _norm(c2), nn.SiLU(inplace=True))
        self.stage2 = nn.Sequential(*[InvertedResidual(c2, expansion) for _ in range(depths[1])])
        self.down2 = nn.Sequential(nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False), _norm(c3), nn.SiLU(inplace=True))
        self.stage3 = nn.Sequential(*[InvertedResidual(c3, expansion) for _ in range(depths[2])])
        self.up2 = nn.Sequential(nn.Conv2d(c3 + c2, c2, 3, padding=1, bias=False), _norm(c2), nn.SiLU(inplace=True))
        self.up1 = nn.Sequential(nn.Conv2d(c2 + c1, c1, 3, padding=1, bias=False), _norm(c1), nn.SiLU(inplace=True))
        self.refine = InvertedResidual(c1, expansion)
        self.weight_head = nn.Conv2d(c1, n_candidates, 1)
        self.residual_head = nn.Conv2d(c1, 1, 1)
        self.gate_head = nn.Conv2d(c1, 1, 1)
        for head in (self.weight_head, self.residual_head, self.gate_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        bias = torch.zeros(1, n_candidates, 1, 1)
        bias[0, 0] = raw_bias  # softmax mass ~= raw at init
        self.register_buffer("raw_bias", bias)
        self.register_buffer("gate_bias", torch.tensor(float(gate_bias)))

    def default_candidates(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """No-candidate mode: raw copies, only the raw flag set (full-GT / online-minimal)."""
        raw = x[:, 0:1] * DISP_SCALE
        candidates = raw.expand(-1, self.n_candidates, -1, -1).contiguous()
        flags = torch.zeros(x.shape[0], self.n_candidates, device=x.device, dtype=x.dtype)
        flags[:, 0] = 1.0
        return candidates, flags

    def forward(self, x: torch.Tensor, residual_scale: float | None = None, candidates: torch.Tensor | None = None, flags: torch.Tensor | None = None):
        scale = self.residual_scale if residual_scale is None else residual_scale
        if candidates is None:
            candidates, flags = self.default_candidates(x)
        if flags is None:
            flags = torch.ones(x.shape[0], self.n_candidates, device=x.device, dtype=x.dtype)
        raw = candidates[:, 0:1]
        flag_maps = flags[:, :, None, None].expand(-1, -1, x.shape[-2], x.shape[-1])
        diffs = (candidates[:, 1:] - raw).abs() / DISP_SCALE
        inp = torch.cat([x, candidates / DISP_SCALE, diffs, flag_maps], dim=1)

        f1 = self.stage1(self.stem(inp))
        f2 = self.stage2(self.down1(f1))
        f3 = self.stage3(self.down2(f2))
        u2 = self.up2(torch.cat([F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False), f2], dim=1))
        u1 = self.up1(torch.cat([F.interpolate(u2, size=f1.shape[-2:], mode="bilinear", align_corners=False), f1], dim=1))
        feat = self.refine(u1)

        logits = self.weight_head(feat) + self.raw_bias
        logits = logits.masked_fill(flag_maps <= 0, float("-inf"))
        weights = torch.softmax(logits, dim=1)
        fused = (weights * candidates).sum(dim=1, keepdim=True)
        residual = scale * torch.tanh(self.residual_head(feat))
        gate_logit = self.gate_head(feat) + self.gate_bias
        gate = torch.sigmoid(gate_logit)
        residual_out = gate * ((fused - raw) + residual)
        diagnostics = {"weights": weights, "fused": fused, "gate": gate, "free_residual": residual}
        return gate_logit, gate, residual_out, diagnostics


def cfr_medium(in_features: int = 16, residual_scale: float = 3.0) -> CandidateFusionRefiner:
    return CandidateFusionRefiner(in_features, base=96, depths=(2, 3, 4), expansion=2.5, residual_scale=residual_scale)
