from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from scripts.temporal_refinement.lib.models import ConvBlock, ConvGRUCell
from .registry import Registry
from .types import FusionOutput, TemporalBatch


STEREO_SOURCES: Registry[nn.Module] = Registry("stereo_source")
MOTION_ESTIMATORS: Registry[nn.Module] = Registry("motion_estimator")
WARPERS: Registry[nn.Module] = Registry("warper")
MEMORY_MODULES: Registry[nn.Module] = Registry("memory")
UNCERTAINTY_ESTIMATORS: Registry[nn.Module] = Registry("uncertainty")
TEACHERS: Registry[nn.Module] = Registry("teacher")
SEMANTIC_PRIORS: Registry[nn.Module] = Registry("semantic_prior")
FUSION_HEADS: Registry[nn.Module] = Registry("fusion")


def _zeros_like_disp(x: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(x)


def _grid_warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp `x` with pixel-space backward flow [B,2,H,W]."""
    b, _c, h, w = x.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=x.device, dtype=x.dtype),
        torch.arange(w, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    grid_x = xx.unsqueeze(0).expand(b, -1, -1) + flow[:, 0]
    grid_y = yy.unsqueeze(0).expand(b, -1, -1) + flow[:, 1]
    grid = torch.stack(
        [
            2.0 * grid_x / max(w - 1, 1) - 1.0,
            2.0 * grid_y / max(h - 1, 1) - 1.0,
        ],
        dim=-1,
    )
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)


class BaseStereoSource(nn.Module):
    def forward(self, batch: TemporalBatch) -> torch.Tensor:
        raise NotImplementedError


@STEREO_SOURCES.register("s2m2_s512")
class S2M2SSource(BaseStereoSource):
    def forward(self, batch: TemporalBatch) -> torch.Tensor:
        return batch.s2m2_s_disp


@STEREO_SOURCES.register("s2m2_l736")
class S2M2LSource(BaseStereoSource):
    def forward(self, batch: TemporalBatch) -> torch.Tensor:
        return batch.s2m2_l_disp


class BaseMotionEstimator(nn.Module):
    def forward(self, batch: TemporalBatch) -> dict[str, torch.Tensor]:
        raise NotImplementedError


@MOTION_ESTIMATORS.register("zero")
class ZeroMotionEstimator(BaseMotionEstimator):
    def forward(self, batch: TemporalBatch) -> dict[str, torch.Tensor]:
        b, t, _c, h, w = batch.rgb.shape
        flow = batch.rgb.new_zeros(b, t, 2, h, w)
        valid = batch.rgb.new_ones(b, t, 1, h, w)
        magnitude = batch.rgb.new_zeros(b, t, 1, h, w)
        return {"flow": flow, "valid": valid, "magnitude": magnitude}


@MOTION_ESTIMATORS.register("rgb_difference")
class RGBDifferenceMotionEstimator(BaseMotionEstimator):
    """Lightweight local proxy used for smoke tests when no learned flow is wired."""

    def forward(self, batch: TemporalBatch) -> dict[str, torch.Tensor]:
        b, t, _c, h, w = batch.rgb.shape
        flow = batch.rgb.new_zeros(b, t, 2, h, w)
        diff = batch.rgb.new_zeros(b, t, 1, h, w)
        if t > 1:
            diff[:, 1:] = torch.mean(torch.abs(batch.rgb[:, 1:] - batch.rgb[:, :-1]), dim=2, keepdim=True)
        valid = (diff < 0.35).float()
        valid[:, 0].fill_(1.0)
        return {"flow": flow, "valid": valid, "magnitude": diff}


@MOTION_ESTIMATORS.register("precomputed")
class PrecomputedMotionEstimator(BaseMotionEstimator):
    def forward(self, batch: TemporalBatch) -> dict[str, torch.Tensor]:
        if batch.motion is None:
            raise RuntimeError("precomputed motion requested but TemporalBatch.motion is None")
        return batch.motion


class TorchvisionRAFTBase(BaseMotionEstimator):
    """Frozen torchvision RAFT optical-flow wrapper.

    Stored flow is backward flow current->previous, so `warper(previous, flow_t)`
    samples previous disparity in the current frame coordinate system.
    """

    def __init__(
        self,
        model_size: str = "large",
        weights: str = "default",
        iters: int = 12,
        fb_threshold_px: float = 1.5,
    ):
        super().__init__()
        try:
            from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights, raft_large, raft_small
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("torchvision RAFT is required for raft_frozen motion") from exc
        self.iters = int(iters)
        self.fb_threshold_px = float(fb_threshold_px)
        if model_size == "small":
            weight_obj = Raft_Small_Weights.DEFAULT if weights == "default" else None
            self.transforms = weight_obj.transforms() if weight_obj is not None else None
            self.model = raft_small(weights=weight_obj, progress=False)
        else:
            weight_obj = Raft_Large_Weights.DEFAULT if weights == "default" else None
            self.transforms = weight_obj.transforms() if weight_obj is not None else None
            self.model = raft_large(weights=weight_obj, progress=False)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def _prepare_pair(self, image1: torch.Tensor, image2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        _, _, h, w = image1.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h or pad_w:
            image1 = F.pad(image1, (0, pad_w, 0, pad_h), mode="replicate")
            image2 = F.pad(image2, (0, pad_w, 0, pad_h), mode="replicate")
        if self.transforms is not None:
            image1, image2 = self.transforms(image1, image2)
        else:
            image1 = image1 * 2.0 - 1.0
            image2 = image2 * 2.0 - 1.0
        return image1, image2, (h, w)

    @torch.no_grad()
    def _flow(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        image1, image2, (h, w) = self._prepare_pair(image1, image2)
        preds = self.model(image1, image2, num_flow_updates=self.iters)
        flow = preds[-1][..., :h, :w]
        return flow

    @torch.no_grad()
    def forward(self, batch: TemporalBatch) -> dict[str, torch.Tensor]:
        b, t, _c, h, w = batch.rgb.shape
        flows = batch.rgb.new_zeros(b, t, 2, h, w)
        valid = batch.rgb.new_ones(b, t, 1, h, w)
        fb_error = batch.rgb.new_zeros(b, t, 1, h, w)
        for i in range(1, t):
            prev = batch.rgb[:, i - 1]
            curr = batch.rgb[:, i]
            forward = self._flow(prev, curr)
            backward = self._flow(curr, prev)
            flows[:, i] = backward
            warped_forward = _grid_warp(forward, backward)
            closure = backward + warped_forward
            err = torch.linalg.vector_norm(closure, dim=1, keepdim=True)
            fb_error[:, i] = err
            valid[:, i] = torch.exp(-err / max(self.fb_threshold_px, 1e-6))
        magnitude = torch.linalg.vector_norm(flows, dim=2, keepdim=True)
        return {"flow": flows, "valid": valid, "magnitude": magnitude, "fb_error": fb_error}


@MOTION_ESTIMATORS.register("raft_frozen")
class RAFTFrozenMotionEstimator(TorchvisionRAFTBase):
    pass


@MOTION_ESTIMATORS.register("raft_fb_consistency")
class RAFTForwardBackwardConsistencyMotionEstimator(TorchvisionRAFTBase):
    pass


class BaseWarper(nn.Module):
    def forward(self, previous: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


@WARPERS.register("identity")
class IdentityWarper(BaseWarper):
    def forward(self, previous: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        return previous


@WARPERS.register("flow")
class FlowWarper(BaseWarper):
    def forward(self, previous: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        return _grid_warp(previous, flow)


@MEMORY_MODULES.register("none")
class NoMemory(nn.Module):
    def forward(self, current: torch.Tensor, previous: torch.Tensor | None = None) -> torch.Tensor:
        return current


@MEMORY_MODULES.register("enabled")
class EnabledMemoryMarker(nn.Module):
    """Declarative marker for fusions that own their memory implementation."""

    def forward(self, current: torch.Tensor, previous: torch.Tensor | None = None) -> torch.Tensor:
        return current if previous is None else previous


@MEMORY_MODULES.register("ema")
class EMAMemory(nn.Module):
    def __init__(self, decay: float = 0.95):
        super().__init__()
        self.decay = float(decay)

    def forward(self, current: torch.Tensor, previous: torch.Tensor | None = None) -> torch.Tensor:
        if previous is None:
            return current
        return self.decay * previous + (1.0 - self.decay) * current


@UNCERTAINTY_ESTIMATORS.register("none")
class NoUncertainty(nn.Module):
    def forward(self, batch: TemporalBatch, raw: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(raw)


@UNCERTAINTY_ESTIMATORS.register("model_disagreement")
class ModelDisagreementUncertainty(nn.Module):
    def __init__(self, normalize: float = 8.0):
        super().__init__()
        self.normalize = float(normalize)

    def forward(self, batch: TemporalBatch, raw: torch.Tensor) -> torch.Tensor:
        disagreement = torch.abs(batch.s2m2_l_disp - batch.sav_disp)
        return torch.clamp(disagreement / self.normalize, 0.0, 1.0)


@TEACHERS.register("none")
class NoTeacher(nn.Module):
    def forward(self, batch: TemporalBatch) -> torch.Tensor | None:
        return None


@TEACHERS.register("stereoanyvideo")
class StereoAnyVideoTeacher(nn.Module):
    def forward(self, batch: TemporalBatch) -> torch.Tensor:
        return batch.sav_disp


@SEMANTIC_PRIORS.register("none")
class NoSemanticPrior(nn.Module):
    def forward(self, batch: TemporalBatch) -> torch.Tensor:
        b, t, _c, h, w = batch.rgb.shape
        return batch.rgb.new_zeros(b, t, 3, h, w)


@SEMANTIC_PRIORS.register("optional_masks")
class OptionalSemanticMasks(nn.Module):
    """Interface for tools/tissue/specularity masks.

    Channel convention is [tool, tissue, specular]. Missing masks become zeros.
    """

    def forward(self, batch: TemporalBatch) -> torch.Tensor:
        if batch.semantic_masks is None:
            b, t, _c, h, w = batch.rgb.shape
            return batch.rgb.new_zeros(b, t, 3, h, w)
        return batch.semantic_masks


class TinyFusionEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_channels, base_channels),
            ConvBlock(base_channels, base_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BaseFusionHead(nn.Module):
    def forward(
        self,
        batch: TemporalBatch,
        raw: torch.Tensor,
        motion: dict[str, torch.Tensor],
        warper: BaseWarper,
        uncertainty: torch.Tensor,
        semantic: torch.Tensor,
    ) -> FusionOutput:
        raise NotImplementedError

    @staticmethod
    def _empty_output(
        fused: torch.Tensor,
        source_weights: torch.Tensor,
        alpha: torch.Tensor | None = None,
        reset: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        uncertainty: torch.Tensor | None = None,
        hidden: torch.Tensor | None = None,
        diagnostics: dict[str, torch.Tensor | float | str] | None = None,
    ) -> FusionOutput:
        return FusionOutput(
            fused_disparity=fused,
            source_weights=source_weights,
            alpha_map=torch.ones_like(fused) if alpha is None else alpha,
            reset_map=torch.zeros_like(fused) if reset is None else reset,
            residual_map=torch.zeros_like(fused) if residual is None else residual,
            uncertainty_map=torch.zeros_like(fused) if uncertainty is None else uncertainty,
            hidden_state=hidden,
            diagnostics={} if diagnostics is None else diagnostics,
        )


@FUSION_HEADS.register("raw_s2m2_s")
class RawS2M2Fusion(BaseFusionHead):
    def forward(self, batch, raw, motion, warper, uncertainty, semantic) -> FusionOutput:
        weights = raw.new_zeros(raw.shape[0], raw.shape[1], 3, raw.shape[-2], raw.shape[-1])
        weights[:, :, 0:1].fill_(1.0)
        return self._empty_output(raw, weights, uncertainty=uncertainty, diagnostics={"mode": "raw"})


@FUSION_HEADS.register("fixed_ema")
class FixedEMAFusion(BaseFusionHead):
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, batch, raw, motion, warper, uncertainty, semantic) -> FusionOutput:
        fused_frames, alpha_frames = [], []
        previous = raw[:, 0]
        for i in range(raw.shape[1]):
            if i == 0:
                current = raw[:, i]
            else:
                previous = warper(previous, motion["flow"][:, i])
                current = self.alpha * raw[:, i] + (1.0 - self.alpha) * previous
            fused_frames.append(current)
            alpha_frames.append(torch.full_like(current, self.alpha))
            previous = current
        fused = torch.stack(fused_frames, dim=1)
        alpha = torch.stack(alpha_frames, dim=1)
        weights = fused.new_zeros(fused.shape[0], fused.shape[1], 3, fused.shape[-2], fused.shape[-1])
        weights[:, :, 0:1] = alpha
        weights[:, :, 1:2] = 1.0 - alpha
        diagnostics = {
            "mode": "fixed_ema",
            "flow_magnitude": motion.get("magnitude"),
            "flow_confidence": motion.get("valid"),
            "fb_error": motion.get("fb_error"),
        }
        return self._empty_output(fused, weights, alpha=alpha, uncertainty=uncertainty, diagnostics=diagnostics)


class LearnedAlphaBase(BaseFusionHead):
    def __init__(
        self,
        in_extra_channels: int = 0,
        base_channels: int = 32,
        residual_clamp_px: float = 2.0,
        use_gru: bool = False,
        hidden_channels: int = 64,
    ):
        super().__init__()
        self.residual_clamp_px = float(residual_clamp_px)
        in_channels = 3 + 1 + 1 + 1 + 1 + 1 + in_extra_channels
        self.encoder = TinyFusionEncoder(in_channels, base_channels)
        self.use_gru = bool(use_gru)
        self.gru = ConvGRUCell(base_channels, hidden_channels) if use_gru else None
        head_channels = hidden_channels if use_gru else base_channels
        self.proj = nn.Conv2d(head_channels, base_channels, 3, padding=1) if use_gru else nn.Identity()
        self.head = nn.Conv2d(base_channels, 3, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def step(self, x: torch.Tensor, hidden: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        feat = self.encoder(x)
        if self.gru is not None:
            hidden = self.gru(feat, hidden)
            feat = self.proj(hidden)
        return self.head(feat), hidden

    def forward(self, batch, raw, motion, warper, uncertainty, semantic) -> FusionOutput:
        fused_frames, alpha_frames, reset_frames, residual_frames, warped_frames = [], [], [], [], []
        previous = raw[:, 0]
        hidden = None
        for i in range(raw.shape[1]):
            warped_prev = raw[:, i] if i == 0 else warper(previous, motion["flow"][:, i])
            raw_diff = torch.abs(raw[:, i] - warped_prev)
            x = torch.cat(
                [
                    batch.rgb[:, i],
                    raw[:, i] / 128.0,
                    warped_prev / 128.0,
                    raw_diff / 16.0,
                    motion["magnitude"][:, i],
                    motion["valid"][:, i],
                    uncertainty[:, i],
                    semantic[:, i],
                ],
                dim=1,
            )
            head, hidden = self.step(x, hidden)
            alpha = torch.sigmoid(head[:, 0:1])
            reset = torch.sigmoid(head[:, 1:2])
            residual = self.residual_clamp_px * torch.tanh(head[:, 2:3])
            memory_weight = (1.0 - alpha) * (1.0 - reset)
            fused = alpha * raw[:, i] + memory_weight * warped_prev + residual
            fused = torch.clamp(fused, min=0.0)
            fused_frames.append(fused)
            alpha_frames.append(alpha)
            reset_frames.append(reset)
            residual_frames.append(residual)
            warped_frames.append(warped_prev)
            previous = fused
        fused_all = torch.stack(fused_frames, dim=1)
        alpha_all = torch.stack(alpha_frames, dim=1)
        reset_all = torch.stack(reset_frames, dim=1)
        residual_all = torch.stack(residual_frames, dim=1)
        warped_all = torch.stack(warped_frames, dim=1)
        weights = fused_all.new_zeros(fused_all.shape[0], fused_all.shape[1], 3, fused_all.shape[-2], fused_all.shape[-1])
        weights[:, :, 0:1] = alpha_all
        weights[:, :, 1:2] = (1.0 - alpha_all) * (1.0 - reset_all)
        diagnostics = {
            "warped_memory": warped_all,
            "mode": "learned_alpha_gru" if self.use_gru else "learned_alpha",
            "flow_magnitude": motion.get("magnitude"),
            "flow_confidence": motion.get("valid"),
            "fb_error": motion.get("fb_error"),
        }
        return self._empty_output(fused_all, weights, alpha_all, reset_all, residual_all, uncertainty, hidden, diagnostics)


@FUSION_HEADS.register("learned_alpha")
class LearnedAlphaFusion(LearnedAlphaBase):
    def __init__(self, base_channels: int = 32, residual_clamp_px: float = 2.0):
        super().__init__(in_extra_channels=4, base_channels=base_channels, residual_clamp_px=residual_clamp_px)


@FUSION_HEADS.register("flow_warped_adaptive")
class FlowWarpedAdaptiveFusion(LearnedAlphaBase):
    def __init__(self, base_channels: int = 40, residual_clamp_px: float = 2.0):
        super().__init__(in_extra_channels=4, base_channels=base_channels, residual_clamp_px=residual_clamp_px)


@FUSION_HEADS.register("flow_warped_convgru")
class FlowWarpedConvGRUFusion(LearnedAlphaBase):
    def __init__(self, base_channels: int = 40, hidden_channels: int = 64, residual_clamp_px: float = 2.0):
        super().__init__(
            in_extra_channels=4,
            base_channels=base_channels,
            residual_clamp_px=residual_clamp_px,
            use_gru=True,
            hidden_channels=hidden_channels,
        )


@FUSION_HEADS.register("dual_memory")
class DualMemoryFusion(BaseFusionHead):
    def __init__(self, base_channels: int = 40, long_decay: float = 0.95, residual_clamp_px: float = 2.0):
        super().__init__()
        self.long_decay = float(long_decay)
        self.residual_clamp_px = float(residual_clamp_px)
        in_channels = 3 + 1 + 1 + 1 + 1 + 1 + 1 + 3
        self.encoder = TinyFusionEncoder(in_channels, base_channels)
        self.head = nn.Conv2d(base_channels, 5, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, batch, raw, motion, warper, uncertainty, semantic) -> FusionOutput:
        fused_frames, weight_frames, reset_frames, residual_frames = [], [], [], []
        warped_short_frames, warped_long_frames = [], []
        short = raw[:, 0]
        long = raw[:, 0]
        for i in range(raw.shape[1]):
            warped_short = raw[:, i] if i == 0 else warper(short, motion["flow"][:, i])
            warped_long = raw[:, i] if i == 0 else warper(long, motion["flow"][:, i])
            x = torch.cat(
                [
                    batch.rgb[:, i],
                    raw[:, i] / 128.0,
                    warped_short / 128.0,
                    warped_long / 128.0,
                    torch.abs(raw[:, i] - warped_short) / 16.0,
                    motion["magnitude"][:, i],
                    uncertainty[:, i],
                    semantic[:, i],
                ],
                dim=1,
            )
            head = self.head(self.encoder(x))
            weights = torch.softmax(head[:, 0:3], dim=1)
            reset = torch.sigmoid(head[:, 3:4])
            residual = self.residual_clamp_px * torch.tanh(head[:, 4:5])
            fused = weights[:, 0:1] * raw[:, i] + weights[:, 1:2] * warped_short + weights[:, 2:3] * warped_long + residual
            fused = torch.clamp(fused, min=0.0)
            short = fused
            long = self.long_decay * warped_long + (1.0 - self.long_decay) * fused
            fused_frames.append(fused)
            weight_frames.append(weights)
            reset_frames.append(reset)
            residual_frames.append(residual)
            warped_short_frames.append(warped_short)
            warped_long_frames.append(warped_long)
        fused_all = torch.stack(fused_frames, dim=1)
        weights_all = torch.stack(weight_frames, dim=1)
        reset_all = torch.stack(reset_frames, dim=1)
        residual_all = torch.stack(residual_frames, dim=1)
        warped_short_all = torch.stack(warped_short_frames, dim=1)
        warped_long_all = torch.stack(warped_long_frames, dim=1)
        return self._empty_output(
            fused_all,
            weights_all,
            alpha=weights_all[:, :, 0:1],
            reset=reset_all,
            residual=residual_all,
            uncertainty=uncertainty,
            diagnostics={
                "mode": "dual_memory",
                "flow_magnitude": motion.get("magnitude"),
                "flow_confidence": motion.get("valid"),
                "fb_error": motion.get("fb_error"),
                "warped_short_memory": warped_short_all,
                "warped_long_memory": warped_long_all,
                "warped_memory": warped_short_all,
            },
        )


def build_component(registry: Registry[nn.Module], cfg_name: str, params: dict[str, Any]) -> nn.Module:
    return registry.build(cfg_name, **params)
