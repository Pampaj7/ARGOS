#!/usr/bin/env python3
"""Causal NVDS-lite: explicit target-to-history local-correlation matching (not implicit
recurrent memory), RGB-D input, bounded gated residual. See module docstring in
train_nvds_lite.py for the full hypothesis this architecture is built to test.

Everything runs at half resolution (stride-2 encoder) to keep the local-correlation matcher
cheap; gate/residual are predicted at half-res and bilinear-upsampled to full-res before being
applied to the full-res raw disparity.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

DISP_SCALE = 64.0


def geo_channels(raw, valid):
    """raw/valid: [B,1,H,W] -> [raw,gx,gy,edge,valid] / DISP_SCALE (valid unscaled), 5ch."""
    gx = torch.zeros_like(raw)
    gy = torch.zeros_like(raw)
    gx[..., 1:] = raw[..., 1:] - raw[..., :-1]
    gy[..., 1:, :] = raw[..., 1:, :] - raw[..., :-1, :]
    edge = torch.sqrt(gx * gx + gy * gy)
    return torch.cat([raw / DISP_SCALE, gx / DISP_SCALE, gy / DISP_SCALE, edge / DISP_SCALE, valid], 1)


class Encoder(nn.Module):
    """Shared siamese encoder: 8ch (3 RGB + 5 geometric) or 5ch (disparity-only) -> hid, stride 2."""

    def __init__(self, in_ch=8, hid=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hid, 3, stride=2, padding=1), nn.GroupNorm(8, hid), nn.SiLU(True),
            nn.Conv2d(hid, hid, 3, padding=1), nn.GroupNorm(8, hid), nn.SiLU(True),
            nn.Conv2d(hid, hid, 3, padding=1), nn.GroupNorm(8, hid), nn.SiLU(True),
        )

    def forward(self, x):
        return self.net(x)


class LocalCorrMatcher(nn.Module):
    """Parameter-free local correlation between current (query) and one historical (key/value)
    feature map. Exposes a soft target-to-history matched feature and a confidence map
    (max normalized correlation) — the explicit matching mechanism the hypothesis calls for,
    as opposed to concatenation or implicit recurrent state.
    """

    def __init__(self, disp=3):
        super().__init__()
        self.d = disp

    def forward(self, f_cur, f_hist):
        # Low-memory local correlation: displacement slices are views of the padded history (no
        # copies), correlation accumulated per displacement, matched feature as a running weighted
        # sum. Avoids materializing a [B,K,C,H,W] cost volume (which OOMs at useful batch sizes
        # because autograd would retain it for every matcher call in the clip).
        B, C, H, W = f_cur.shape
        d = self.d
        f_hist_pad = F.pad(f_hist, [d, d, d, d])
        shifts = [f_hist_pad[:, :, d + dy:d + dy + H, d + dx:d + dx + W]
                  for dy in range(-d, d + 1) for dx in range(-d, d + 1)]
        corr = torch.cat([(f_cur * s).sum(1, keepdim=True) / (C ** 0.5) for s in shifts], 1)  # [B,K,H,W]
        weights = torch.softmax(corr, dim=1)
        matched = sum(w.unsqueeze(1) * s for w, s in zip(weights.unbind(1), shifts))  # [B,C,H,W]
        confidence = corr.amax(dim=1, keepdim=True)
        return matched, confidence


class FusionHead(nn.Module):
    """Fuse current + matched-history features -> confidence gate + bounded residual (half-res)."""

    def __init__(self, in_ch, hid=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hid, 3, padding=1), nn.GroupNorm(8, hid), nn.SiLU(True),
            nn.Conv2d(hid, hid, 3, padding=1), nn.GroupNorm(8, hid), nn.SiLU(True),
        )
        self.gate_head = nn.Conv2d(hid, 1, 1)
        self.res_head = nn.Conv2d(hid, 1, 1)
        nn.init.zeros_(self.res_head.weight)
        nn.init.zeros_(self.res_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, 0.0)  # gate~0.5 at init (residual still zero-init,
        # so refined==raw at step 0); -2.0 was too conservative and drove identity collapse.

    def forward(self, x):
        h = self.net(x)
        gate = torch.sigmoid(self.gate_head(h))
        res = torch.tanh(self.res_head(h))
        return gate, res


def hist_indices(t, mode, rng):
    """Causal history-frame selection for target frame t. Never references t' > t.

    full_history        [t-1, t-2, t-3] clamped to 0 (repeat earliest frame near sequence start)
    current_frame_only  [t, t, t] -- degenerate self-match, carries zero cross-frame information
                         but keeps tensor statistics/shapes identical to the other modes
    shuffled_history    3 draws (with repetition if pool<3) from a shuffled pool of frames < t;
                         corrupts only history order/content, current frame t is untouched,
                         GT ordering at t is untouched, no frame > t is ever used
    """
    if mode == "current_frame_only":
        return [t, t, t]
    if mode == "full_history":
        return [max(t - 1, 0), max(t - 2, 0), max(t - 3, 0)]
    if mode == "shuffled_history":
        pool = list(range(t))
        rng.shuffle(pool)
        return [pool[i] if i < len(pool) else t for i in range(3)]
    raise ValueError(mode)


class CausalNVDSLite(nn.Module):
    """Explicit local-correlation matching against up to 3 causal history frames."""

    def __init__(self, use_rgb=True, enc_hid=96, fus_hid=192, disp=3, res_scale=3.0):
        super().__init__()
        self.use_rgb = use_rgb
        in_ch = 8 if use_rgb else 5
        self.encoder = Encoder(in_ch, enc_hid)
        self.matcher = LocalCorrMatcher(disp)
        fusion_in = enc_hid + enc_hid * 3 + 3 + 3 + 1 + 1  # cur + 3*matched + 3*conf + 3*diff + raw_lr + valid_lr
        self.fusion = FusionHead(fusion_in, fus_hid)
        self.res_scale = res_scale

    def frame_input(self, raw, valid, rgb):
        geo = geo_channels(raw, valid)
        return torch.cat([rgb, geo], 1) if self.use_rgb else geo

    def forward(self, raw_clip, valid_clip, rgb_clip, temporal_mode, rng):
        """raw_clip/valid_clip: [B,T,1,H,W]; rgb_clip: [B,T,3,H,W] in [0,1] (ignored if use_rgb=False).
        Returns refined [B,T,1,H,W] and, for the leakage test, the per-t chosen history indices.
        """
        B, T = raw_clip.shape[:2]
        H, W = raw_clip.shape[-2:]
        feats = [self.encoder(self.frame_input(raw_clip[:, t], valid_clip[:, t], rgb_clip[:, t])) for t in range(T)]
        raw_lr = [F.avg_pool2d(raw_clip[:, t], 2) for t in range(T)]
        valid_lr = [F.avg_pool2d(valid_clip[:, t], 2) for t in range(T)]
        refined = [None] * T
        gates, residuals = [], []
        used_idx = []
        for t in range(T):
            idxs = hist_indices(t, temporal_mode, rng)
            used_idx.append(idxs)
            matched, conf, diff = [], [], []
            for i in idxs:
                m, c = self.matcher(feats[t], feats[i])
                matched.append(m)
                conf.append(c)
                diff.append((feats[t] - m).abs().mean(1, keepdim=True))
            fusion_in = torch.cat([feats[t], *matched, *conf, *diff, raw_lr[t] / DISP_SCALE, valid_lr[t]], 1)
            gate, res = self.fusion(fusion_in)
            gate_up = F.interpolate(gate, size=(H, W), mode="bilinear", align_corners=False)
            res_up = F.interpolate(res, size=(H, W), mode="bilinear", align_corners=False)
            applied = gate_up * self.res_scale * res_up
            refined[t] = raw_clip[:, t] + applied
            gates.append(gate_up.detach())
            residuals.append(applied.detach())
        self.last_diag = {"gate": torch.stack(gates, 1), "residual": torch.stack(residuals, 1)}
        return torch.stack(refined, 1), used_idx


class ConcatBaseline(nn.Module):
    """Config F: same param budget, causal frames concatenated channel-wise (no explicit
    matching) -- tests whether local correlation actually beats naive concatenation.
    """

    def __init__(self, use_rgb=True, enc_hid=96, fus_hid=192, res_scale=3.0):
        super().__init__()
        self.use_rgb = use_rgb
        in_ch = 8 if use_rgb else 5
        self.encoder = Encoder(in_ch, enc_hid)
        fusion_in = enc_hid * 4 + 1 + 1  # cur + 3 history feats concatenated + raw_lr + valid_lr
        self.fusion = FusionHead(fusion_in, fus_hid)
        self.res_scale = res_scale

    def frame_input(self, raw, valid, rgb):
        geo = geo_channels(raw, valid)
        return torch.cat([rgb, geo], 1) if self.use_rgb else geo

    def forward(self, raw_clip, valid_clip, rgb_clip, temporal_mode, rng):
        B, T = raw_clip.shape[:2]
        H, W = raw_clip.shape[-2:]
        feats = [self.encoder(self.frame_input(raw_clip[:, t], valid_clip[:, t], rgb_clip[:, t])) for t in range(T)]
        raw_lr = [F.avg_pool2d(raw_clip[:, t], 2) for t in range(T)]
        valid_lr = [F.avg_pool2d(valid_clip[:, t], 2) for t in range(T)]
        refined = [None] * T
        gates, residuals = [], []
        used_idx = []
        for t in range(T):
            idxs = hist_indices(t, temporal_mode, rng)
            used_idx.append(idxs)
            fusion_in = torch.cat([feats[t], *[feats[i] for i in idxs], raw_lr[t] / DISP_SCALE, valid_lr[t]], 1)
            gate, res = self.fusion(fusion_in)
            gate_up = F.interpolate(gate, size=(H, W), mode="bilinear", align_corners=False)
            res_up = F.interpolate(res, size=(H, W), mode="bilinear", align_corners=False)
            applied = gate_up * self.res_scale * res_up
            refined[t] = raw_clip[:, t] + applied
            gates.append(gate_up.detach())
            residuals.append(applied.detach())
        self.last_diag = {"gate": torch.stack(gates, 1), "residual": torch.stack(residuals, 1)}
        return torch.stack(refined, 1), used_idx


def build_model(name, use_rgb=True):
    if name == "nvds_lite":
        return CausalNVDSLite(use_rgb=use_rgb)
    if name == "concat_baseline":
        return ConcatBaseline(use_rgb=use_rgb)
    raise ValueError(name)


if __name__ == "__main__":
    # smoke: shapes + param count + causal leakage self-check
    rng = np.random.default_rng(0)
    B, T, H, W = 1, 6, 64, 80
    raw = torch.randn(B, T, 1, H, W).abs() * 10
    valid = (torch.rand(B, T, 1, H, W) > 0.1).float()
    rgb = torch.rand(B, T, 3, H, W)
    for name in ("nvds_lite", "concat_baseline"):
        for use_rgb in (True, False):
            m = build_model(name, use_rgb)
            n_params = sum(p.numel() for p in m.parameters())
            for mode in ("full_history", "current_frame_only", "shuffled_history"):
                out, idxs = m(raw, valid, rgb, mode, rng)
                assert out.shape == raw.shape
                for t, ii in enumerate(idxs):
                    assert all(i <= t for i in ii), f"FUTURE LEAK t={t} idxs={ii}"
            print(f"{name} use_rgb={use_rgb}: params={n_params:,} OK (all modes causal)")
