"""Faithful-as-possible CODD fusion head for the ARGOS v2 BiDA diagnostic.

This is a *baseline reproduction*, not a new ARGOS architecture.  It keeps the
current raw stereo disparity and causal BiDA-aligned previous fused disparity
fixed, replacing only CODD's RAFT3D projection with canonical BiDA warping.
CODD obtains matching features from its stereo network; cached ARGOS backbones
do not expose those features, so this file uses a frozen local ImageNet
ResNet-18 layer-1 map as the closest common learned left/right feature space.
The extractor is never trained and is intentionally excluded from the fusion
parameter count.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18

from argos_freezed.alignment.bida_pull_warp import causal_warp, resize_flow
from .stereo_photometric import warp_right_to_left


RESNET18_CHECKPOINT = Path.home() / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth"


@dataclass(frozen=True)
class CODDCues:
    """CODD-like explicit cues at the cache grid, ``[B,C,H,W]``.

    The cue tensor comprises candidate feature-match L1 curves, local
    disparity/appearance self-correlation, local cross-frame correlations,
    motion/visibility, raw/memory disparity context, RGB, and frozen context
    features.  `support` is separate and never alters the geometry mask.
    """

    values: torch.Tensor
    support: torch.Tensor
    channels: int


@dataclass(frozen=True)
class CODDFusionOutput:
    reset_weight: torch.Tensor
    fusion_weight: torch.Tensor
    temporal_weight: torch.Tensor
    fused_disparity: torch.Tensor


class FrozenResNet18Layer1(nn.Module):
    """Frozen 1/4-resolution learned RGB feature map, with no download path."""

    output_channels = 64

    def __init__(self, checkpoint: Path = RESNET18_CHECKPOINT) -> None:
        super().__init__()
        if not checkpoint.exists():
            raise FileNotFoundError(f"required local frozen ResNet-18 checkpoint is missing: {checkpoint}")
        net = resnet18(weights=None)
        net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1), persistent=False)
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        image = rgb.float()
        if image.detach().amax() > 1.5:
            image = image / 255.0
        return self.stem((image - self.mean) / self.std)


def _up(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(value, size=size, mode="bilinear", align_corners=True)


def _feature_disparity(disparity: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resize positive-left disparity and scale its horizontal displacement."""
    height, width = disparity.shape[-2:]
    feature = _up(disparity, size)
    return feature * (size[1] / width)


def _sample_right_feature(right: torch.Tensor, disparity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Stereo sample a feature map using the canonical positive-left convention."""
    # The RGB helper is channel-agnostic despite its argument name.
    return warp_right_to_left(right, disparity)


def _patches(value: torch.Tensor, *, window: int = 3, dilation: int = 2) -> torch.Tensor:
    if window != 3:
        raise ValueError("this faithful CODD probe fixes W=3")
    batch, channels, height, width = value.shape
    unfolded = F.unfold(value, kernel_size=window, dilation=dilation, padding=dilation).view(
        batch, channels, window * window, height, width
    )
    return unfolded


def _without_center(values: torch.Tensor) -> torch.Tensor:
    return torch.cat((values[:, :, :4], values[:, :, 5:]), dim=2)


def _disparity_self_correlation(disparity: torch.Tensor) -> torch.Tensor:
    patch = _without_center(_patches(disparity))
    center = disparity[:, :, None]
    # CODD calls this a correlation; bounded local similarity is the scalar
    # disparity analogue of its dot-product semantic correlation.
    return (1.0 - (patch - center).abs().clamp_max(4.0) / 4.0).squeeze(1)


def _appearance_self_correlation(feature: torch.Tensor) -> torch.Tensor:
    patch = _without_center(_patches(F.normalize(feature, dim=1, eps=1e-6)))
    center = F.normalize(feature, dim=1, eps=1e-6)[:, :, None]
    return (patch * center).sum(dim=1)


def _cross_disparity_correlation(current: torch.Tensor, aligned_past: torch.Tensor) -> torch.Tensor:
    patch = _patches(aligned_past)
    return (1.0 - (patch - current[:, :, None]).abs().clamp_max(4.0) / 4.0).squeeze(1)


def _cross_appearance_correlation(current: torch.Tensor, aligned_past: torch.Tensor) -> torch.Tensor:
    current = F.normalize(current, dim=1, eps=1e-6)
    patch = _patches(F.normalize(aligned_past, dim=1, eps=1e-6))
    return (patch * current[:, :, None]).sum(dim=1)


@torch.no_grad()
def build_codd_cues(
    extractor: FrozenResNet18Layer1 | None,
    *,
    raw: torch.Tensor,
    aligned_memory: torch.Tensor,
    current_rgb: torch.Tensor,
    current_right_rgb: torch.Tensor,
    past_rgb: torch.Tensor,
    flow_current_to_past: torch.Tensor,
    flow_magnitude: torch.Tensor,
    forward_backward_confidence: torch.Tensor,
    warp_support: torch.Tensor,
    aligned_valid: torch.Tensor,
    include_learned_stereo_evidence: bool = True,
) -> CODDCues:
    """Build CODD Sect. 3.3.2-style explicit frozen cues.

    Feature confidence samples right features at d-1/d/d+1.  The previous
    left feature is causally warped using BiDA's `grid + target_to_source`
    convention before cross-frame correlations are formed.
    """
    height, width = raw.shape[-2:]
    # The no-learned-stereo-evidence ablation deliberately retains only
    # disparity/motion/RGB evidence.  It neither evaluates the frozen ResNet
    # nor leaves a zero-valued learned feature block that the head could use as
    # an architecture-side domain marker.
    feature_size = (height // 4, width // 4)
    raw_f = _feature_disparity(raw, feature_size)
    memory_f = _feature_disparity(aligned_memory, feature_size)
    if include_learned_stereo_evidence:
        if extractor is None:
            raise ValueError("learned stereo evidence requires the frozen extractor")
        current_feature = extractor(current_rgb)
        right_feature = extractor(current_right_rgb)
        past_feature = extractor(past_rgb)
        feature_size = current_feature.shape[-2:]
        raw_f = _feature_disparity(raw, feature_size)
        memory_f = _feature_disparity(aligned_memory, feature_size)
        flow_f = resize_flow(flow_current_to_past, feature_size)
        aligned_past_feature = causal_warp(past_feature, flow_f).warped

        def confidence_curve(candidate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            curve, supports = [], []
            for offset in (-1.0, 0.0, 1.0):
                sampled, support = _sample_right_feature(right_feature, candidate + offset)
                curve.append((current_feature - sampled).abs().mean(dim=1, keepdim=True).clamp(0, 4) / 4)
                supports.append(support.float())
            return torch.cat(curve, dim=1), torch.cat(supports, dim=1)

        raw_conf, raw_conf_support = confidence_curve(raw_f)
        memory_conf, memory_conf_support = confidence_curve(memory_f)
        # All correlations are formed at 1/4, then upsampled to the cache grid.
        low = torch.cat((
            raw_conf, memory_conf, raw_conf - memory_conf,
            _disparity_self_correlation(raw_f), _disparity_self_correlation(memory_f),
            _appearance_self_correlation(current_feature), _appearance_self_correlation(aligned_past_feature),
            _cross_disparity_correlation(raw_f, memory_f),
            _cross_appearance_correlation(current_feature, aligned_past_feature),
            raw_conf_support, memory_conf_support,
            torch.tanh(current_feature / 3.0),
        ), dim=1)
    else:
        low = torch.cat((
            _disparity_self_correlation(raw_f), _disparity_self_correlation(memory_f),
            _cross_disparity_correlation(raw_f, memory_f),
        ), dim=1)
    low = _up(low, (height, width))
    residual = (aligned_memory - raw).clamp(-16, 16) / 16
    motion = torch.cat((
        flow_current_to_past[:, 0:1].clamp(-32, 32) / 32,
        flow_current_to_past[:, 1:2].clamp(-32, 32) / 32,
        flow_magnitude.clamp(0, 32) / 32,
        forward_backward_confidence.clamp(0, 1),
        warp_support.float(), aligned_valid.float(),
    ), dim=1)
    image = current_rgb.float() / (255.0 if current_rgb.detach().amax() > 1.5 else 1.0)
    values = torch.cat((
        low, raw.clamp(0, 64) / 64, aligned_memory.clamp(0, 64) / 64,
        residual, residual.abs(), motion, image * 2.0 - 1.0,
    ), dim=1)
    if not torch.isfinite(values).all():
        raise ValueError("non-finite CODD cue")
    return CODDCues(values=values, support=(warp_support & aligned_valid), channels=values.shape[1])


class _Residual(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.GroupNorm(8, channels), nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.GroupNorm(8, channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.act(value + self.body(value))


class CODDStyleFusionHead(nn.Module):
    """~0.2M dual-weight fusion head following CODD Appendix A.3.

    Reset is predicted at full resolution; fusion is predicted at 1/4 and
    bilinearly upsampled for context-aware aggregation.
    """

    def __init__(self, cue_channels: int, width: int = 48) -> None:
        super().__init__()
        if width % 8:
            raise ValueError("width must be divisible by 8")
        self.full = nn.Sequential(
            nn.Conv2d(cue_channels, width // 2, 3, padding=1, bias=False), nn.GroupNorm(8, width // 2), nn.SiLU(inplace=True),
            _Residual(width // 2),
        )
        self.coarse = nn.Sequential(
            nn.Conv2d(width // 2, width, 3, stride=2, padding=1, bias=False), nn.GroupNorm(8, width), nn.SiLU(inplace=True),
            _Residual(width), _Residual(width), _Residual(width),
        )
        self.reset = nn.Conv2d(width // 2 + width, 1, 3, padding=1)
        self.fusion = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.reset.weight); nn.init.constant_(self.reset.bias, -2.0)
        nn.init.zeros_(self.fusion.weight); nn.init.constant_(self.fusion.bias, 0.0)

    def forward(self, cues: CODDCues, raw: torch.Tensor, aligned_memory: torch.Tensor) -> CODDFusionOutput:
        fine = self.full(cues.values)
        coarse = self.coarse(F.avg_pool2d(fine, 2, 2))
        coarse_full = _up(coarse, raw.shape[-2:])
        reset = torch.sigmoid(self.reset(torch.cat((fine, coarse_full), dim=1)))
        fusion_low = torch.sigmoid(self.fusion(coarse))
        fusion = _up(fusion_low, raw.shape[-2:])
        temporal = reset * fusion
        fused = (1.0 - temporal) * raw + temporal * aligned_memory
        return CODDFusionOutput(reset, fusion, temporal, fused)


def convex_fusion_oracle(
    raw: torch.Tensor,
    memory: torch.Tensor,
    ground_truth: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GT-only best convex interpolation and its coefficient.

    This is diagnostic-only: it identifies the minimum absolute scalar
    disparity error on the segment from ``raw`` to ``memory``.  Equal endpoint
    disparities use an exactly-zero coefficient to avoid a division by zero.
    """
    denominator = memory - raw
    safe = denominator.abs() > epsilon
    # Compute the ratio only at safe elements with an explicit masked
    # assignment: a denominator can be negative, so ``clamp_min`` is invalid.
    weight = torch.zeros_like(raw)
    weight[safe] = ((ground_truth - raw)[safe] / denominator[safe]).clamp(0.0, 1.0)
    fused = (1.0 - weight) * raw + weight * memory
    return fused, weight


def hard_endpoint_fusion(
    raw: torch.Tensor,
    memory: torch.Tensor,
    temporal_weight: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bit-exact raw-or-memory endpoint decision used only for the audit."""
    accepted = temporal_weight >= threshold
    return torch.where(accepted, memory, raw), accepted


__all__ = [
    "CODDCues", "CODDFusionOutput", "CODDStyleFusionHead", "FrozenResNet18Layer1",
    "RESNET18_CHECKPOINT", "build_codd_cues", "convex_fusion_oracle", "hard_endpoint_fusion",
]
