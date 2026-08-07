"""Spatial comparative error critic for the frozen ARGOS v2 multi-anchor proposal.

Veto-only adapter: predicts expected raw error, expected proposal error, their
aleatoric scales, and a harm probability from local spatial evidence, then either
ACCEPTs the frozen proposal or returns raw bit-exact. It never changes the frozen
selected anchor, fusion weight, or proposal, and never reopens a rejected pixel.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .raw_multi_anchor_refiner import MultiAnchorEvidence, _masked_median_mad
from .raw_multi_anchor_selective_gate import FrozenProposal

FAMILIES = ("geometry", "temporal", "stereo", "plane_sweep")
LOG_SCALE_RANGE = (-4.0, 3.0)
PLANE_SWEEP_OFFSETS = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)


@dataclass(frozen=True)
class CriticExtras:
    """Causal image evidence at cache resolution. No GT, no future frames.

    ``left_rgb``/``right_rgb``: [B,3,H,W] in [0,255] or [0,1].
    ``temporal_residual``: [B,K,H,W] robust photometric residual per anchor age,
    from the validated BiDA alignment path (already computed causally).
    """

    left_rgb: torch.Tensor
    right_rgb: torch.Tensor
    temporal_residual: torch.Tensor


@dataclass(frozen=True)
class CriticOutput:
    mu_raw: torch.Tensor
    mu_proposed: torch.Tensor
    s_raw: torch.Tensor
    s_proposed: torch.Tensor
    harm_logit: torch.Tensor
    harm_probability: torch.Tensor

    @property
    def delta_hat(self) -> torch.Tensor:
        return self.mu_raw - self.mu_proposed

    @property
    def sigma_delta(self) -> torch.Tensor:
        return torch.sqrt(torch.exp(2 * self.s_raw) + torch.exp(2 * self.s_proposed))


def _gray(rgb: torch.Tensor) -> torch.Tensor:
    value = rgb.float()
    if float(value.detach().max()) > 1.5:
        value = value / 255.0
    return value.mean(dim=1, keepdim=True)


def _gradient(value: torch.Tensor) -> torch.Tensor:
    dx = F.pad(value[..., 1:] - value[..., :-1], (0, 1, 0, 0))
    dy = F.pad(value[..., 1:, :] - value[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt(dx.square() + dy.square() + 1e-12)


def _local_mean_variance(value: torch.Tensor, kernel: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    mean = F.avg_pool2d(value, kernel, stride=1, padding=kernel // 2)
    square = F.avg_pool2d(value.square(), kernel, stride=1, padding=kernel // 2)
    return mean, (square - mean.square()).clamp_min(0)


def stereo_residual(left_rgb: torch.Tensor, right_rgb: torch.Tensor, disparity: torch.Tensor,
                    *, robust_scale: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """Robust |I_L - warp_stereo(I_R, d)| for positive-left disparity in pixels.

    Samples the right image at ``x - d`` (rectified pair, cache resolution).
    Returns (robust residual in [0,1], in-bounds validity), both [B,1,H,W].
    Residual is zeroed where sampling leaves the image; the validity channel
    carries that information to the model instead of a fake low residual.
    """
    if disparity.ndim != 4 or disparity.shape[1] != 1:
        raise ValueError("disparity must be [B,1,H,W]")
    left = left_rgb.float()
    right = right_rgb.float()
    if max(float(left.detach().max()), float(right.detach().max())) > 1.5:
        left, right = left / 255.0, right / 255.0
    b, _c, h, w = left.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, device=left.device, dtype=left.dtype),
        torch.arange(w, device=left.device, dtype=left.dtype), indexing="ij")
    sample_x = xs[None] + torch.zeros_like(disparity[:, 0]) - disparity[:, 0]
    valid = ((sample_x >= 0) & (sample_x <= w - 1) & torch.isfinite(disparity[:, 0]) & (disparity[:, 0] >= 0))[:, None]
    norm_x = 2.0 * sample_x / max(w - 1, 1) - 1.0
    norm_y = (2.0 * ys / max(h - 1, 1) - 1.0)[None].expand_as(norm_x)
    grid = torch.stack((norm_x, norm_y), dim=-1)
    warped = F.grid_sample(right, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    l1 = (left - warped).abs().mean(dim=1, keepdim=True)
    robust = l1 / (l1 + robust_scale)
    return robust * valid.to(robust.dtype), valid


def plane_sweep_features(left_rgb: torch.Tensor, right_rgb: torch.Tensor, disparity: torch.Tensor,
                         *, offsets: tuple[float, ...] = PLANE_SWEEP_OFFSETS) -> torch.Tensor:
    """External plane-sweep-style confidence around one hypothesis (Lee et al. 2024,
    approximated: a small perturbation neighborhood instead of a full disparity
    volume, fully external to the stereo backbones). Returns [B,3,H,W]:
    center-minus-min residual (ambiguity), best signed offset, fraction of
    perturbations strictly better than the center hypothesis.
    """
    center, _ = stereo_residual(left_rgb, right_rgb, disparity)
    profile = torch.cat([stereo_residual(left_rgb, right_rgb, disparity + o)[0] for o in offsets], dim=1)
    best, best_index = profile.min(dim=1, keepdim=True)
    offset_map = disparity.new_tensor(offsets)[best_index]
    better = (profile < center - 1e-3).float().mean(dim=1, keepdim=True)
    return torch.cat(((center - best).clamp(0, 1), offset_map / max(abs(o) for o in offsets), better), dim=1)


def build_critic_features(evidence: MultiAnchorEvidence, proposal: FrozenProposal,
                          extras: CriticExtras, family: str) -> tuple[torch.Tensor, list[str]]:
    """Dense causal feature tensor [B,C,H,W]; ``family`` is cumulative:
    geometry < temporal < stereo < plane_sweep. No GT, no future, no backbone ID.
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    level = FAMILIES.index(family)
    raw, candidates = evidence.raw, evidence.candidates
    available = evidence.available
    count = max(candidates.shape[1], 1)
    median, mad, witness = _masked_median_mad(candidates, available)
    anchor = proposal.candidate
    proposed = proposal.proposed
    signed = anchor - raw
    update = proposed - raw
    agreement = (((candidates - median).abs() <= mad + .10) & available).sum(dim=1, keepdim=True)
    age_map = evidence.ages
    if age_map.ndim == 1:
        age_map = age_map[None].expand(raw.shape[0], -1)
    age_map = age_map[:, :, None, None].expand(-1, -1, *raw.shape[-2:])
    selected_age = torch.gather(age_map, 1, proposal.chosen).float()
    support = torch.gather(evidence.warp_support, 1, proposal.chosen).float()
    fb = torch.gather(evidence.fb_confidence, 1, proposal.chosen)
    valid = torch.gather(available, 1, proposal.chosen).float()
    disagreement_mean, disagreement_variance = _local_mean_variance(signed.abs())
    update_mean, update_variance = _local_mean_variance(update.abs())
    gray = _gray(extras.left_rgb)
    named: list[tuple[str, torch.Tensor]] = [
        ("raw_disparity", raw.clamp(0, 64) / 64),
        ("anchor_disparity", anchor.clamp(0, 64) / 64),
        ("proposed_disparity", proposed.clamp(0, 64) / 64),
        ("signed_disagreement", signed.clamp(-16, 16) / 16),
        ("absolute_disagreement", signed.abs().clamp(0, 16) / 16),
        ("signed_update", update.clamp(-8, 8) / 8),
        ("absolute_update", update.abs().clamp(0, 8) / 8),
        ("fusion_weight", proposal.fusion_weight),
        ("top1_score", proposal.top1_score.clamp(-2, 2) / 2),
        ("top1_top2_margin", proposal.top2_margin.clamp(-2, 2) / 2),
        ("selected_age", selected_age.clamp(0, 8) / 8),
        ("selected_valid", valid),
        ("selected_support", support),
        ("selected_fb_confidence", fb.clamp(0, 1)),
        ("temporal_median", median.clamp(0, 64) / 64),
        ("temporal_mad", mad.clamp(0, 8) / 8),
        ("raw_median_deviation", (raw - median).abs().clamp(0, 8) / 8),
        ("anchor_median_deviation", (anchor - median).abs().clamp(0, 8) / 8),
        ("valid_anchor_fraction", witness.float() / count),
        ("agreement_fraction", agreement.float() / count),
        ("image_gradient", _gradient(gray).clamp(0, 1)),
        ("raw_gradient", _gradient(raw).clamp(0, 8) / 8),
        ("anchor_gradient", _gradient(anchor).clamp(0, 8) / 8),
        ("proposed_gradient", _gradient(proposed).clamp(0, 8) / 8),
        ("local_disagreement_mean", disagreement_mean.clamp(0, 8) / 8),
        ("local_disagreement_variance", disagreement_variance.clamp(0, 64) / 64),
        ("local_update_mean", update_mean.clamp(0, 8) / 8),
        ("local_update_variance", update_variance.clamp(0, 64) / 64),
    ]
    if level >= 1:
        r_temporal = torch.gather(extras.temporal_residual, 1, proposal.chosen)
        t_mean, t_variance = _local_mean_variance(r_temporal)
        named += [
            ("temporal_residual", r_temporal.clamp(0, 1)),
            ("temporal_residual_mean", t_mean.clamp(0, 1)),
            ("temporal_residual_variance", t_variance.clamp(0, 1)),
            ("temporal_residual_charbonnier", torch.sqrt(r_temporal.square() + 1e-4).clamp(0, 1)),
        ]
    if level >= 2:
        r_raw, ok_raw = stereo_residual(extras.left_rgb, extras.right_rgb, raw)
        r_anchor, ok_anchor = stereo_residual(extras.left_rgb, extras.right_rgb, anchor)
        r_prop, ok_prop = stereo_residual(extras.left_rgb, extras.right_rgb, proposed)
        m_raw, v_raw = _local_mean_variance(r_raw)
        m_anchor, v_anchor = _local_mean_variance(r_anchor)
        m_prop, v_prop = _local_mean_variance(r_prop)
        named += [
            ("stereo_residual_raw", r_raw), ("stereo_residual_anchor", r_anchor),
            ("stereo_residual_proposed", r_prop),
            ("stereo_raw_minus_proposed", (r_raw - r_prop).clamp(-1, 1)),
            ("stereo_raw_minus_anchor", (r_raw - r_anchor).clamp(-1, 1)),
            ("stereo_anchor_minus_proposed", (r_anchor - r_prop).clamp(-1, 1)),
            ("stereo_raw_local_mean", m_raw), ("stereo_anchor_local_mean", m_anchor),
            ("stereo_proposed_local_mean", m_prop),
            ("stereo_raw_local_variance", v_raw.clamp(0, 1)),
            ("stereo_anchor_local_variance", v_anchor.clamp(0, 1)),
            ("stereo_proposed_local_variance", v_prop.clamp(0, 1)),
            ("stereo_inbounds", (ok_raw & ok_anchor & ok_prop).float()),
        ]
    if level >= 3:
        for label, hypothesis in (("raw", raw), ("proposed", proposed)):
            sweep = plane_sweep_features(extras.left_rgb, extras.right_rgb, hypothesis)
            named += [(f"sweep_{label}_ambiguity", sweep[:, 0:1]), (f"sweep_{label}_best_offset", sweep[:, 1:2]),
                      (f"sweep_{label}_better_fraction", sweep[:, 2:3])]
    features = torch.cat([value for _, value in named], dim=1)
    return torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0), [name for name, _ in named]


def feature_channels(family: str) -> int:
    return {"geometry": 28, "temporal": 32, "stereo": 45, "plane_sweep": 51}[family]


class _DilatedBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(8, channels), nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(8, channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.silu(value + self.net(value))


class SpatialErrorCritic(nn.Module):
    """Small FCN: 3x3 projection + dilated residual body + 5-channel head.

    Heads: mu_raw, mu_proposed (softplus, non-negative), s_raw, s_proposed
    (clamped log scales), harm logit. ~0.4M parameters at channels=64.
    """

    def __init__(self, in_channels: int, channels: int = 64, dilations: tuple[int, ...] = (1, 2, 4, 2, 1)) -> None:
        super().__init__()
        if channels % 8:
            raise ValueError("channels must be divisible by 8")
        self.in_channels = in_channels
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels), nn.SiLU(inplace=True),
            *[_DilatedBlock(channels, d) for d in dilations],
        )
        self.head = nn.Conv2d(channels, 5, 1)
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([0.0, 0.0, 0.0, 0.0, -2.0]))

    def forward(self, features: torch.Tensor) -> CriticOutput:
        raw_output = self.head(self.body(features))
        low, high = LOG_SCALE_RANGE
        return CriticOutput(
            mu_raw=F.softplus(raw_output[:, 0:1]),
            mu_proposed=F.softplus(raw_output[:, 1:2]),
            s_raw=raw_output[:, 2:3].clamp(low, high),
            s_proposed=raw_output[:, 3:4].clamp(low, high),
            harm_logit=raw_output[:, 4:5],
            harm_probability=torch.sigmoid(raw_output[:, 4:5]),
        )


def critic_losses(output: CriticOutput, e_raw: torch.Tensor, e_proposed: torch.Tensor,
                  supervised: torch.Tensor, *, harm_margin: float, lambda_harm: float = 1.0,
                  lambda_rank: float = 0.1, rank_margin: float = 0.05) -> dict[str, torch.Tensor]:
    """Heteroscedastic Laplace NLL + class-weighted harm BCE + ranking hinge.

    ``supervised`` masks pixels with GT and an eligible frozen proposal.
    False-safe (missed harm) is weighted above false-reject via pos_weight.
    """
    mask = supervised.bool()
    if not mask.any():
        zero = output.harm_logit.sum() * 0
        return {"total": zero, "error": zero.detach(), "harm": zero.detach(), "rank": zero.detach()}
    nll = ((e_raw - output.mu_raw).abs() * torch.exp(-output.s_raw) + output.s_raw
           + (e_proposed - output.mu_proposed).abs() * torch.exp(-output.s_proposed) + output.s_proposed)
    error = nll[mask].mean()
    harm_target = (e_proposed > e_raw + harm_margin).float()
    labels = harm_target[mask]
    positive = labels.sum()
    pos_weight = ((labels.numel() - positive) / positive.clamp_min(1)).clamp(.5, 4.0).detach() * 2.0
    harm = F.binary_cross_entropy_with_logits(output.harm_logit[mask], labels, pos_weight=pos_weight)
    rank_sign = torch.sign(e_raw - e_proposed)
    rank = F.relu(rank_margin - rank_sign * output.delta_hat)[mask].mean()
    total = error + lambda_harm * harm + lambda_rank * rank
    return {"total": total, "error": error.detach(), "harm": harm.detach(), "rank": rank.detach()}


def selective_accept(output: CriticOutput, proposal: FrozenProposal, *, lambda_uncertainty: float,
                     tau_gain: float, tau_harm: float) -> torch.Tensor:
    """Conservative LCB acceptance; only within the frozen refiner's own
    nonzero-intervention eligibility (close-only veto, never reopens)."""
    lcb = output.delta_hat - lambda_uncertainty * output.sigma_delta
    return proposal.eligible & (lcb > tau_gain) & (output.harm_probability < tau_harm)


__all__ = [
    "FAMILIES", "PLANE_SWEEP_OFFSETS", "CriticExtras", "CriticOutput", "SpatialErrorCritic",
    "build_critic_features", "critic_losses", "feature_channels", "plane_sweep_features",
    "selective_accept", "stereo_residual",
]
