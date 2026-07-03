"""EGBM-Refiner: Event-Gated Boundary Memory Refiner (experimental branch).

An intentionally unconventional refiner for frozen S2M2 disparity, designed around the
two observed pathological failure modes of the v3/v4 line:

  * high_temporal_flicker  -> corrections chase temporally unstable pixels
  * high_boundary_error    -> residuals bleed across depth boundaries and overshoot

Five ideas composed into one executable architecture:

1. EVENT-GATED TEMPORAL MEMORY (ConvGRU at 1/4 resolution).
   The 4-frame disparity stack is replayed oldest->newest through a small ConvGRU whose
   per-step input is (disparity, valid, |temporal delta| event map). The final hidden
   state is an *instability memory*: pixels that flickered across the window leave a
   trace. The memory feeds the damping head, so flicker history directly closes the
   correction valve — the mechanism v3 lacked.

2. BOUNDARY-AWARE CORRECTION FIELD.
   A boundary branch predicts boundary confidence B in [0,1] from the spatial-gradient
   feature channels. The mixture residual is smoothed with an edge-stopping 3x3 average
   ((1-B) selects smoothing, B keeps the value sharp), then attenuated at boundaries by
   a learned global factor. This suppresses cross-edge residual bleeding instead of
   asking a plain conv stack to figure it out.

3. DYNAMIC ABSTENTION CONTROLLER.
   Instead of one global threshold, a coarse-grid head (adaptive-pooled encoder state)
   predicts a per-region threshold offset delta in [-0.3, 0.3]. The soft gate is
   sigmoid((p_bad - (0.7 + delta)) / tau): regions with heavy flicker/boundary activity
   can locally *raise* their own bar for correction.

4. MIXTURE-OF-CORRECTION EXPERTS.
   Four experts produce candidate residuals: temporal-smooth, boundary, catastrophic,
   and a hard-wired IDENTITY expert (residual exactly 0). A softmax router mixes them
   per pixel; routing mass on the identity expert is *learned abstention*, giving the
   model a way to say "do nothing here" that is separate from the p_bad gate.

5. ENERGY-STYLE ITERATIVE UPDATE.
   The refined disparity is produced by two damped, gated fixed-point steps
       r <- r + gate * damping * 0.5 * (R_mix - r)
   rather than one direct addition. Each step pulls the running correction toward the
   mixture residual but is re-scaled by gate*damping, so weakly-supported corrections
   converge to a fraction of their candidate magnitude.

Interface: forward(x, residual_scale) -> (bad_logit, p_bad, residual, diagnostics)
where refined = raw + residual (gating/damping already applied inside). The first three
outputs match the v3/v4 convention so existing eval code can consume out[:3]. The
diagnostics dict exposes damping, router weights, boundary confidence, dynamic
threshold map, and the temporal memory for analysis.

Identity at initialization: all expert residual heads are zero-initialized, so the
mixture residual is 0 and both update steps leave r = 0 -> refined == raw exactly,
regardless of gate state. Input format is the unchanged 16-channel v3 feature tensor
(4-frame raw stack, valid masks, temporal stats, spatial gradients). No RGB, no
pretrained weights, no teacher inference.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


# 16-channel v3 feature layout (see make_features_from_raws):
#  0-3  raw disparity stack (t, t-1, t-2, t-3) / 64
#  4-7  valid masks
#  8    |raw_t - raw_{t-1}| / 64          (dt1)
#  9    temporal mean / 64
# 10    temporal median / 64
# 11    temporal variance / 64^2
# 12    |raw - median| / 64
# 13-14 spatial gradients gx, gy / 64
# 15    gradient magnitude (edge) / 64
RAW_STACK = slice(0, 4)
VALID_STACK = slice(4, 8)
GRAD_CHANNELS = slice(13, 16)


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


class ConvGRUCell(nn.Module):
    """Minimal ConvGRU used for the low-resolution instability memory."""

    def __init__(self, in_ch: int, hidden_ch: int):
        super().__init__()
        self.convz = nn.Conv2d(in_ch + hidden_ch, hidden_ch, 3, padding=1)
        self.convr = nn.Conv2d(in_ch + hidden_ch, hidden_ch, 3, padding=1)
        self.convq = nn.Conv2d(in_ch + hidden_ch, hidden_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        xh = torch.cat([x, h], dim=1)
        z = torch.sigmoid(self.convz(xh))
        r = torch.sigmoid(self.convr(xh))
        q = torch.tanh(self.convq(torch.cat([x, r * h], dim=1)))
        return (1 - z) * h + z * q


class EGBMRefiner(nn.Module):
    N_EXPERTS = 4  # temporal-smooth, boundary, catastrophic, identity

    def __init__(
        self,
        in_channels: int = 16,
        base: int = 80,
        depths: tuple[int, int, int] = (2, 2, 4),
        expansion: float = 2.5,
        memory_channels: int = 32,
        residual_scale: float = 3.0,
        base_threshold: float = 0.7,
        gate_tau: float = 0.1,
        update_steps: int = 2,
        update_step_size: float = 0.5,
    ):
        super().__init__()
        self.residual_scale = residual_scale
        self.gate_tau = gate_tau
        self.update_steps = update_steps
        self.update_step_size = update_step_size
        self.register_buffer("base_threshold", torch.tensor(float(base_threshold)))
        c1, c2, c3 = base, base * 2, base * 4

        # --- multi-scale encoder ---
        self.stem = nn.Sequential(nn.Conv2d(in_channels, c1, 3, padding=1, bias=False), _norm(c1), nn.SiLU(inplace=True))
        self.stage1 = nn.Sequential(*[InvertedResidual(c1, expansion) for _ in range(depths[0])])
        self.down1 = nn.Sequential(nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False), _norm(c2), nn.SiLU(inplace=True))
        self.stage2 = nn.Sequential(*[InvertedResidual(c2, expansion) for _ in range(depths[1])])
        self.down2 = nn.Sequential(nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False), _norm(c3), nn.SiLU(inplace=True))
        self.stage3 = nn.Sequential(*[InvertedResidual(c3, expansion) for _ in range(depths[2])])

        # --- event-gated temporal memory (1/4 resolution ConvGRU) ---
        self.memory_cell = ConvGRUCell(3, memory_channels)
        self.memory_proj = nn.Sequential(nn.Conv2d(memory_channels, memory_channels, 3, padding=1), _norm(memory_channels), nn.SiLU(inplace=True))

        # --- boundary branch (from gradient channels + full-res features) ---
        self.boundary_branch = nn.Sequential(
            nn.Conv2d(3 + c1, 32, 3, padding=1), _norm(32), nn.SiLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1), _norm(16), nn.SiLU(inplace=True),
        )
        self.boundary_head = nn.Conv2d(16, 1, 1)
        # learned global boundary-attenuation strength (sigmoid -> [0,1])
        self.boundary_atten = nn.Parameter(torch.tensor(0.0))

        # --- decoder with memory injection ---
        self.up2 = nn.Sequential(nn.Conv2d(c3 + c2 + memory_channels, c2, 3, padding=1, bias=False), _norm(c2), nn.SiLU(inplace=True))
        self.up1 = nn.Sequential(nn.Conv2d(c2 + c1, c1, 3, padding=1, bias=False), _norm(c1), nn.SiLU(inplace=True))
        self.refine = InvertedResidual(c1, expansion)

        # --- heads ---
        self.bad_head = nn.Conv2d(c1, 1, 1)
        self.expert_heads = nn.ModuleList([nn.Conv2d(c1, 1, 1) for _ in range(self.N_EXPERTS - 1)])  # identity expert is hard-wired 0
        self.router_head = nn.Conv2d(c1, self.N_EXPERTS, 1)
        self.damping_head = nn.Conv2d(c1 + memory_channels, 1, 1)  # sees the instability memory
        # dynamic abstention: coarse grid threshold offset from pooled deep features
        self.threshold_head = nn.Sequential(nn.Conv2d(c3, 32, 1), nn.SiLU(inplace=True), nn.Conv2d(32, 1, 1))
        for head in [self.bad_head, self.damping_head, self.router_head, *self.expert_heads]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.zeros_(self.threshold_head[-1].weight)
        nn.init.zeros_(self.threshold_head[-1].bias)
        nn.init.zeros_(self.boundary_head.weight)
        nn.init.zeros_(self.boundary_head.bias)

    def _temporal_memory(self, x: torch.Tensor) -> torch.Tensor:
        """Replay the 4-frame stack oldest->newest through the low-res ConvGRU."""
        raws = x[:, RAW_STACK]
        valids = x[:, VALID_STACK]
        b, _, h, w = raws.shape
        hs, ws = h // 4, w // 4
        mem = raws.new_zeros(b, self.memory_cell.convz.out_channels, hs, ws)
        prev = None
        for t in range(3, -1, -1):  # oldest (t-3) -> newest (t)
            r = F.adaptive_avg_pool2d(raws[:, t : t + 1], (hs, ws))
            v = F.adaptive_avg_pool2d(valids[:, t : t + 1], (hs, ws))
            event = torch.tanh(4.0 * (r - prev).abs()) if prev is not None else torch.zeros_like(r)
            mem = self.memory_cell(torch.cat([r, v, event], dim=1), mem)
            prev = r
        return self.memory_proj(mem)

    def forward(self, x: torch.Tensor, residual_scale: float | None = None):
        scale = self.residual_scale if residual_scale is None else residual_scale
        f1 = self.stage1(self.stem(x))
        f2 = self.stage2(self.down1(f1))
        f3 = self.stage3(self.down2(f2))

        mem = self._temporal_memory(x)  # (B, M, H/4, W/4)
        mem_at_f2 = F.interpolate(mem, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.up2(torch.cat([F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False), f2, mem_at_f2], dim=1))
        u1 = self.up1(torch.cat([F.interpolate(u2, size=f1.shape[-2:], mode="bilinear", align_corners=False), f1], dim=1))
        feat = self.refine(u1)

        # detection + damping (damping sees the flicker memory explicitly)
        bad_logit = self.bad_head(feat)
        p_bad = torch.sigmoid(bad_logit)
        mem_full = F.interpolate(mem, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        damping = torch.sigmoid(self.damping_head(torch.cat([feat, mem_full], dim=1)))

        # mixture-of-correction experts; expert N-1 is the hard-wired identity (residual 0)
        router = torch.softmax(self.router_head(feat), dim=1)  # (B, 4, H, W)
        candidates = [scale * torch.tanh(head(feat)) for head in self.expert_heads]
        candidates.append(torch.zeros_like(candidates[0]))  # identity expert
        mixture = sum(router[:, k : k + 1] * candidates[k] for k in range(self.N_EXPERTS))

        # boundary-aware correction field: edge-stopping smoothing + boundary attenuation
        boundary = torch.sigmoid(self.boundary_head(self.boundary_branch(torch.cat([x[:, GRAD_CHANNELS], f1], dim=1))))
        smoothed = F.avg_pool2d(mixture, 3, stride=1, padding=1)
        mixture = (1.0 - boundary) * smoothed + boundary * mixture  # do not average across edges
        mixture = mixture * (1.0 - torch.sigmoid(self.boundary_atten) * boundary)  # attenuate at boundaries

        # dynamic abstention: coarse per-region threshold offset in [-0.3, 0.3]
        grid = F.adaptive_avg_pool2d(f3, (8, 10))
        delta = 0.3 * torch.tanh(self.threshold_head(grid))
        delta = F.interpolate(delta, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        gate = torch.sigmoid((p_bad - (self.base_threshold + delta)) / self.gate_tau)

        # energy-style damped fixed-point update (2 steps)
        r = torch.zeros_like(mixture)
        for _ in range(self.update_steps):
            r = r + gate * damping * self.update_step_size * (mixture - r)

        diagnostics = {
            "damping": damping,
            "router_weights": router,
            "boundary_confidence": boundary,
            "dynamic_threshold": self.base_threshold + delta,
            "temporal_memory": mem,
            "mixture_residual": mixture,
            "gate": gate,
        }
        return bad_logit, p_bad, r, diagnostics


def egbm_refiner(in_channels: int = 16, residual_scale: float = 3.0) -> EGBMRefiner:
    return EGBMRefiner(in_channels, base=80, depths=(2, 2, 4), expansion=2.5, residual_scale=residual_scale)


def egbm_refiner_large(in_channels: int = 16, residual_scale: float = 3.0) -> EGBMRefiner:
    return EGBMRefiner(in_channels, base=112, depths=(2, 3, 6), expansion=2.5, residual_scale=residual_scale)
