"""EGBM-v2-CARE: Change-Aware Reliability Encoding with Predictive Temporal Memory.

Architecture refinement of EGBM-v1 (`experimental_refiner_vx.EGBMRefiner`), which
reached oracle-gap 20.37% with patho new-Bad3 1.30%. EGBM-v1's damping separation is
strong on high_boundary_error (hard-neg 0.186 vs hard-pos 0.792) but weak on
high_temporal_flicker (0.387 vs 0.733): the v1 instability memory only *accumulates*
change — it cannot tell unpredictable stereo flicker (suppress!) from predictable
motion or a real scene change (do not blindly suppress!).

CARE adds a predictive pathway that makes exactly that distinction:

  1. FRAME RELIABILITY ENCODER  z_t = f(disp_t, valid_t, event_t) at 1/4 resolution.
  2. PREDICTIVE TEMPORAL MEMORY M_t: a ConvGRU over (z_t, surprise_t). ConvGRU is kept
     (proven in v1, cheap, causal); a *predictor* head g(M) forecasts z_hat_{t+1}.
  3. SURPRISE e_t = |z_t - z_hat_t| (charbonnier-flavored abs). Predictable content —
     smooth camera/tissue motion the memory can forecast — produces LOW surprise even
     when frame-to-frame change is large; stereo flicker produces HIGH surprise.
  4. CHANGE-AWARE RELIABILITY HEAD: softmax over 5 change types
     (artifact_like, real_scene_change, occlusion_tool_like, boundary_change,
     stable_predictable) from (z_t, z_hat_t, e_t, M_t).
  5. MODULATION: a compressed CARE context (care probs + surprise + memory) is injected
     into the decoder (hence all v1 heads see it) and explicitly into the damping head,
     so correction strength can key on *why* a region changed, not just *that* it did.

The whole temporal pathway replays the existing 4-frame feature window causally
(oldest -> newest), exactly like v1's memory — so the crop-based training pipeline,
splits, and eval protocol carry over unchanged, and no future frames are used.
Longer-horizon streaming memory across full clips is architecture-compatible (the
recurrence has no window-length dependence) but is left untrained/uneval'd here to
avoid a train/test mismatch; documented as future work.

Warm start: all v1 module names/shapes are preserved (stem, stages, downs, boundary
branch, memory_cell/memory_proj, up1, refine, all heads except damping) so v1's best
checkpoint loads with strict=False; only `up2` (wider input: +CARE context) and
`damping_head` (wider input) re-initialize, plus the new CARE modules. Expert/router/
bad/threshold heads keep v1 zero-init semantics -> exact identity at init regardless
of CARE outputs.

No RGB, no external pretrained models, no new teacher inference.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from experimental_refiner_vx import RAW_STACK, VALID_STACK, GRAD_CHANNELS, ConvGRUCell, InvertedResidual, _norm


CARE_CLASSES = ("artifact_like", "real_scene_change", "occlusion_tool_like", "boundary_change", "stable_predictable")
N_CARE = len(CARE_CLASSES)


class EGBMv2CARE(nn.Module):
    N_EXPERTS = 4  # temporal-smooth, boundary, catastrophic, identity (as v1)

    def __init__(
        self,
        in_channels: int = 16,
        base: int = 80,
        depths: tuple[int, int, int] = (2, 2, 4),
        expansion: float = 2.5,
        memory_channels: int = 32,
        care_z_channels: int = 32,
        care_memory_channels: int = 48,
        care_context_channels: int = 32,
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
        zc, mc = care_z_channels, care_memory_channels

        # ---- v1 backbone (names preserved for warm start) ----
        self.stem = nn.Sequential(nn.Conv2d(in_channels, c1, 3, padding=1, bias=False), _norm(c1), nn.SiLU(inplace=True))
        self.stage1 = nn.Sequential(*[InvertedResidual(c1, expansion) for _ in range(depths[0])])
        self.down1 = nn.Sequential(nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False), _norm(c2), nn.SiLU(inplace=True))
        self.stage2 = nn.Sequential(*[InvertedResidual(c2, expansion) for _ in range(depths[1])])
        self.down2 = nn.Sequential(nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False), _norm(c3), nn.SiLU(inplace=True))
        self.stage3 = nn.Sequential(*[InvertedResidual(c3, expansion) for _ in range(depths[2])])
        self.memory_cell = ConvGRUCell(3, memory_channels)
        self.memory_proj = nn.Sequential(nn.Conv2d(memory_channels, memory_channels, 3, padding=1), _norm(memory_channels), nn.SiLU(inplace=True))
        self.boundary_branch = nn.Sequential(
            nn.Conv2d(3 + c1, 32, 3, padding=1), _norm(32), nn.SiLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1), _norm(16), nn.SiLU(inplace=True),
        )
        self.boundary_head = nn.Conv2d(16, 1, 1)
        self.boundary_atten = nn.Parameter(torch.tensor(0.0))
        # up2 widened by CARE context -> re-initialized (cannot warm-start this layer)
        self.up2 = nn.Sequential(nn.Conv2d(c3 + c2 + memory_channels + care_context_channels, c2, 3, padding=1, bias=False), _norm(c2), nn.SiLU(inplace=True))
        self.up1 = nn.Sequential(nn.Conv2d(c2 + c1, c1, 3, padding=1, bias=False), _norm(c1), nn.SiLU(inplace=True))
        self.refine = InvertedResidual(c1, expansion)
        self.bad_head = nn.Conv2d(c1, 1, 1)
        self.expert_heads = nn.ModuleList([nn.Conv2d(c1, 1, 1) for _ in range(self.N_EXPERTS - 1)])
        self.router_head = nn.Conv2d(c1, self.N_EXPERTS, 1)
        self.damping_head = nn.Conv2d(c1 + memory_channels + care_context_channels, 1, 1)
        self.threshold_head = nn.Sequential(nn.Conv2d(c3, 32, 1), nn.SiLU(inplace=True), nn.Conv2d(32, 1, 1))
        for head in [self.bad_head, self.damping_head, self.router_head, *self.expert_heads]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.zeros_(self.threshold_head[-1].weight)
        nn.init.zeros_(self.threshold_head[-1].bias)
        nn.init.zeros_(self.boundary_head.weight)
        nn.init.zeros_(self.boundary_head.bias)

        # ---- CARE pathway ----
        self.care_encoder = nn.Sequential(
            nn.Conv2d(3, zc, 3, padding=1), _norm(zc), nn.SiLU(inplace=True),
            nn.Conv2d(zc, zc, 3, padding=1), _norm(zc),
        )
        self.care_gru = ConvGRUCell(zc + zc, mc)  # input: (z_t, surprise_t)
        self.care_predictor = nn.Sequential(nn.Conv2d(mc, mc, 3, padding=1), _norm(mc), nn.SiLU(inplace=True), nn.Conv2d(mc, zc, 1))
        self.care_head = nn.Sequential(
            nn.Conv2d(zc * 3 + mc, 64, 3, padding=1), _norm(64), nn.SiLU(inplace=True),
            nn.Conv2d(64, N_CARE, 1),
        )
        self.care_context = nn.Sequential(
            nn.Conv2d(N_CARE + zc + mc, care_context_channels, 1), _norm(care_context_channels), nn.SiLU(inplace=True),
        )

    def _temporal_pathways(self, x: torch.Tensor):
        """Causal replay of the 4-frame window: v1 memory + CARE predictive memory."""
        raws = x[:, RAW_STACK]
        valids = x[:, VALID_STACK]
        b, _, h, w = raws.shape
        hs, ws = h // 4, w // 4
        mem = raws.new_zeros(b, self.memory_cell.convz.out_channels, hs, ws)
        m_care = raws.new_zeros(b, self.care_gru.convz.out_channels, hs, ws)
        zc = self.care_encoder[0].out_channels
        z_hat = raws.new_zeros(b, zc, hs, ws)
        prev = None
        z_t = z_hat
        surprise = z_hat
        pred_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []  # (z_hat_before, z_actual) per predicted step
        for t in range(3, -1, -1):  # oldest (t-3) -> newest (t)
            r = F.adaptive_avg_pool2d(raws[:, t : t + 1], (hs, ws))
            v = F.adaptive_avg_pool2d(valids[:, t : t + 1], (hs, ws))
            event = torch.tanh(4.0 * (r - prev).abs()) if prev is not None else torch.zeros_like(r)
            step_in = torch.cat([r, v, event], dim=1)
            mem = self.memory_cell(step_in, mem)
            z_t = self.care_encoder(step_in)
            surprise = (z_t - z_hat).abs() if prev is not None else torch.zeros_like(z_t)
            if prev is not None:
                pred_pairs.append((z_hat, z_t))
            m_care = self.care_gru(torch.cat([z_t, surprise], dim=1), m_care)
            z_hat = self.care_predictor(m_care)  # forecast for next step
            prev = r
        mem = self.memory_proj(mem)
        # change-type reasoning at the current frame
        z_hat_cur = pred_pairs[-1][0] if pred_pairs else torch.zeros_like(z_t)
        care_logits = self.care_head(torch.cat([z_t, z_hat_cur, surprise, m_care], dim=1))
        care_probs = torch.softmax(care_logits, dim=1)
        care_ctx = self.care_context(torch.cat([care_probs, surprise, m_care], dim=1))
        return mem, care_ctx, care_probs, surprise, pred_pairs, m_care

    def forward(self, x: torch.Tensor, residual_scale: float | None = None):
        scale = self.residual_scale if residual_scale is None else residual_scale
        f1 = self.stage1(self.stem(x))
        f2 = self.stage2(self.down1(f1))
        f3 = self.stage3(self.down2(f2))

        mem, care_ctx, care_probs, surprise, pred_pairs, m_care = self._temporal_pathways(x)
        mem_f2 = F.interpolate(mem, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        ctx_f2 = F.interpolate(care_ctx, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.up2(torch.cat([F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False), f2, mem_f2, ctx_f2], dim=1))
        u1 = self.up1(torch.cat([F.interpolate(u2, size=f1.shape[-2:], mode="bilinear", align_corners=False), f1], dim=1))
        feat = self.refine(u1)

        bad_logit = self.bad_head(feat)
        p_bad = torch.sigmoid(bad_logit)
        mem_full = F.interpolate(mem, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        ctx_full = F.interpolate(care_ctx, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        damping = torch.sigmoid(self.damping_head(torch.cat([feat, mem_full, ctx_full], dim=1)))

        router = torch.softmax(self.router_head(feat), dim=1)
        candidates = [scale * torch.tanh(head(feat)) for head in self.expert_heads]
        candidates.append(torch.zeros_like(candidates[0]))
        mixture = sum(router[:, k : k + 1] * candidates[k] for k in range(self.N_EXPERTS))

        boundary = torch.sigmoid(self.boundary_head(self.boundary_branch(torch.cat([x[:, GRAD_CHANNELS], f1], dim=1))))
        smoothed = F.avg_pool2d(mixture, 3, stride=1, padding=1)
        mixture = (1.0 - boundary) * smoothed + boundary * mixture
        mixture = mixture * (1.0 - torch.sigmoid(self.boundary_atten) * boundary)

        grid = F.adaptive_avg_pool2d(f3, (8, 10))
        delta = 0.3 * torch.tanh(self.threshold_head(grid))
        delta = F.interpolate(delta, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        gate = torch.sigmoid((p_bad - (self.base_threshold + delta)) / self.gate_tau)

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
            "care_probs": care_probs,          # (B, 5, H/4, W/4)
            "surprise": surprise,              # (B, zc, H/4, W/4) current-frame surprise
            "care_memory": m_care,
            "pred_pairs": pred_pairs,          # [(z_hat, z_actual)] for the prediction loss
        }
        return bad_logit, p_bad, r, diagnostics


def egbm_v2_care(in_channels: int = 16, residual_scale: float = 3.0) -> EGBMv2CARE:
    return EGBMv2CARE(in_channels, base=80, depths=(2, 2, 4), expansion=2.5, residual_scale=residual_scale)


def load_v1_warm_start(model: EGBMv2CARE, v1_checkpoint: str) -> tuple[int, int]:
    """Load EGBM-v1 weights so the warm-started v2 reproduces v1 behavior EXACTLY at init.

    Shape-matching tensors load directly. The two widened layers (up2 conv,
    damping_head) get their v1 weights copied into the original input channels and
    ZEROS on the new CARE-context channels — so the CARE pathway contributes nothing
    until training moves it, and the warm model's outputs equal v1's.

    Returns (n_loaded_tensors, n_model_tensors); the two partially-loaded tensors are
    counted as loaded."""
    ck = torch.load(v1_checkpoint, map_location="cpu", weights_only=False)
    state = ck["model_state_dict"]
    own = model.state_dict()
    loadable = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(loadable, strict=False)
    n = len(loadable)
    with torch.no_grad():
        for key in ("up2.0.weight", "damping_head.weight"):
            if key in state and key in own and own[key].shape != state[key].shape:
                v1_w = state[key]
                param = model.get_parameter(key)
                param.zero_()
                param[:, : v1_w.shape[1]] = v1_w
                n += 1
        if "damping_head.bias" in state:
            model.get_parameter("damping_head.bias").copy_(state["damping_head.bias"])
    return n, len(own)
