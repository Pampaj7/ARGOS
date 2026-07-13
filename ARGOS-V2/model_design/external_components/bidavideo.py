"""Backbone-agnostic causal alignment utilities derived from BiDAStabilizer.

The original non-causal reference is
``external/bidavideo/models/core/bidastabilizer.py::BiDAStabilizer`` at commit
``dae817df1ceaafcb865ebd9c7aa44b16c535e856``.  Its sampling convention is
preserved exactly: a target pixel samples the source at ``grid + flow`` using
bilinear ``grid_sample``, zero padding, and ``align_corners=True``.

ARGOS tensor contracts
----------------------
* images/features/disparities/masks: ``[B,C,H,W]``
* optical flow: ``[B,2,H,W]`` in pixels of its own grid
* flow ``target_to_source``: coordinates in the target frame point into source
* cached disparity: positive-left pixels at cache width 180; warping does not
  alter its magnitude because source and target share the same spatial grid

Only pairwise, past-only operations in this file are deployably causal.  The
original ``BiDAStabilizer.forward`` remains the non-causal reproduction baseline;
it is deliberately not duplicated here.
"""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
BIDAVIDEO_ROOT = ROOT / "external/bidavideo"
SEA_RAFT_CHECKPOINT = (
    BIDAVIDEO_ROOT
    / "third_party/SEA-RAFT/models/Tartan-C-T-TSKH-spring540x960-S.pth"
)
RAFT_CHECKPOINT = BIDAVIDEO_ROOT / "third_party/RAFT/models/raft-things.pth"


@dataclass(frozen=True)
class WarpResult:
    """Result of target-to-source sampling, all masks shaped ``[B,1,H,W]``."""

    warped: torch.Tensor
    support: torch.Tensor
    sampled_source_valid: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class ForwardBackwardResult:
    """Round-trip consistency on the target grid."""

    error: torch.Tensor
    support: torch.Tensor
    valid: torch.Tensor
    confidence: torch.Tensor
    threshold: torch.Tensor


@dataclass(frozen=True)
class PhotometricResult:
    """Source RGB aligned to target plus universal photometric evidence."""

    aligned_source_rgb: torch.Tensor
    l1_residual: torch.Tensor
    robust_normalized_residual: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class TemporalDisparityEvidence:
    """Causal evidence from one past frame, expressed on the current grid."""

    aligned_past_disparity: torch.Tensor
    warp_support: torch.Tensor
    aligned_validity: torch.Tensor
    forward_backward_error: torch.Tensor
    forward_backward_confidence: torch.Tensor
    photometric_residual: torch.Tensor
    absolute_disparity_disagreement: torch.Tensor
    signed_disparity_disagreement: torch.Tensor
    flow_magnitude: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _flow_bchw(flow: torch.Tensor) -> torch.Tensor:
    if flow.ndim != 4:
        raise ValueError(f"flow must be rank 4, got {tuple(flow.shape)}")
    if flow.shape[1] == 2:
        return flow
    if flow.shape[-1] == 2:
        return flow.permute(0, 3, 1, 2)
    raise ValueError(f"flow must be [B,2,H,W] or [B,H,W,2], got {tuple(flow.shape)}")


def resize_flow(
    flow: torch.Tensor,
    size: tuple[int, int],
    *,
    mode: Literal["bilinear", "nearest"] = "bilinear",
) -> torch.Tensor:
    """Resize ``[B,2,Hs,Ws]`` flow and preserve geometric displacement.

    Horizontal and vertical components are scaled independently by
    ``Wt/Ws`` and ``Ht/Hs``.  This function never scales disparity.
    """
    flow = _flow_bchw(flow)
    source_h, source_w = flow.shape[-2:]
    target_h, target_w = size
    if (source_h, source_w) == (target_h, target_w):
        return flow
    kwargs = {"mode": mode}
    if mode == "bilinear":
        kwargs["align_corners"] = True
    resized = F.interpolate(flow, size=size, **kwargs)
    scale = resized.new_tensor([target_w / source_w, target_h / source_h]).view(1, 2, 1, 1)
    return resized * scale


def causal_warp(
    source: torch.Tensor,
    flow_target_to_source: torch.Tensor,
    *,
    source_valid: torch.Tensor | None = None,
    mode: Literal["bilinear", "nearest"] = "bilinear",
    valid_threshold: float = 0.999,
) -> WarpResult:
    """Warp a past/source tensor to the current/target coordinate system.

    This is exactly BiDA's ``grid + flow`` convention.  ``support`` marks
    continuous sampling coordinates inside the source image.  Source validity is
    sampled bilinearly by default and conservatively thresholded, so invalid
    neighbours cannot silently enter a supposedly valid disparity comparison.
    """
    if source.ndim != 4:
        raise ValueError(f"source must be [B,C,H,W], got {tuple(source.shape)}")
    flow = _flow_bchw(flow_target_to_source).to(device=source.device, dtype=source.dtype)
    b, _c, h, w = source.shape
    if flow.shape != (b, 2, h, w):
        raise ValueError(f"flow {tuple(flow.shape)} incompatible with source {tuple(source.shape)}")

    ys, xs = torch.meshgrid(
        torch.arange(h, device=source.device, dtype=source.dtype),
        torch.arange(w, device=source.device, dtype=source.dtype),
        indexing="ij",
    )
    sample_x = xs[None] + flow[:, 0]
    sample_y = ys[None] + flow[:, 1]
    support = (
        (sample_x >= 0)
        & (sample_x <= max(w - 1, 0))
        & (sample_y >= 0)
        & (sample_y <= max(h - 1, 0))
    )[:, None]
    norm_x = 2.0 * sample_x / max(w - 1, 1) - 1.0
    norm_y = 2.0 * sample_y / max(h - 1, 1) - 1.0
    grid = torch.stack((norm_x, norm_y), dim=-1)
    warped = F.grid_sample(
        source,
        grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=True,
    )

    if source_valid is None:
        sampled_valid = torch.ones((b, 1, h, w), dtype=torch.bool, device=source.device)
    else:
        if source_valid.ndim == 3:
            source_valid = source_valid[:, None]
        if source_valid.shape != (b, 1, h, w):
            raise ValueError(
                f"source_valid must be [B,1,H,W], got {tuple(source_valid.shape)}"
            )
        valid_sample = F.grid_sample(
            source_valid.to(source.dtype),
            grid,
            mode=mode,
            padding_mode="zeros",
            align_corners=True,
        )
        sampled_valid = valid_sample >= valid_threshold
    combined = support & sampled_valid
    return WarpResult(warped, support, sampled_valid, combined)


def forward_backward_consistency(
    flow_target_to_source: torch.Tensor,
    flow_source_to_target: torch.Tensor,
    *,
    absolute_threshold: float = 0.5,
    relative_threshold: float = 0.01,
) -> ForwardBackwardResult:
    """Check ``f_t->s + warp(f_s->t, f_t->s)`` on the target grid.

    A pixel is consistent when round-trip error is below
    ``absolute_threshold + relative_threshold * (|f_t->s| + |f_s->t_aligned|)``.
    Confidence is ``exp(-error/threshold)`` within supported pixels and zero
    outside, providing a bounded universal evidence channel.
    """
    forward = _flow_bchw(flow_target_to_source)
    backward = _flow_bchw(flow_source_to_target).to(forward)
    if forward.shape != backward.shape:
        raise ValueError(f"flow shapes differ: {tuple(forward.shape)} vs {tuple(backward.shape)}")
    aligned_backward = causal_warp(backward, forward)
    cycle = forward + aligned_backward.warped
    error = torch.linalg.vector_norm(cycle, dim=1, keepdim=True)
    magnitude = torch.linalg.vector_norm(forward, dim=1, keepdim=True)
    magnitude = magnitude + torch.linalg.vector_norm(aligned_backward.warped, dim=1, keepdim=True)
    threshold = absolute_threshold + relative_threshold * magnitude
    support = aligned_backward.support
    valid = support & (error <= threshold)
    confidence = torch.exp(-error / threshold.clamp_min(1e-6)) * support.to(error.dtype)
    return ForwardBackwardResult(error, support, valid, confidence, threshold)


def photometric_consistency(
    target_rgb: torch.Tensor,
    source_rgb: torch.Tensor,
    flow_target_to_source: torch.Tensor,
    *,
    target_valid: torch.Tensor | None = None,
    source_valid: torch.Tensor | None = None,
    robust_scale: float = 0.1,
) -> PhotometricResult:
    """Compute valid-aware RGB evidence after aligning source to target.

    RGB may be in ``[0,1]`` or ``[0,255]``; it is normalized to ``[0,1]``.
    The robust signal is ``r/(r+robust_scale)`` and therefore lies in ``[0,1]``.
    Invalid output pixels are zeroed and separately identified by ``valid``.
    """
    if target_rgb.shape != source_rgb.shape or target_rgb.ndim != 4:
        raise ValueError("target_rgb and source_rgb must share [B,C,H,W]")
    target = target_rgb.float()
    source = source_rgb.float()
    if max(float(target.detach().max()), float(source.detach().max())) > 1.5:
        target = target / 255.0
        source = source / 255.0
    aligned = causal_warp(source, flow_target_to_source, source_valid=source_valid)
    valid = aligned.valid
    if target_valid is not None:
        if target_valid.ndim == 3:
            target_valid = target_valid[:, None]
        valid = valid & target_valid.bool()
    l1 = (target - aligned.warped).abs().mean(dim=1, keepdim=True)
    robust = l1 / (l1 + robust_scale)
    valid_f = valid.to(l1.dtype)
    return PhotometricResult(aligned.warped, l1 * valid_f, robust * valid_f, valid)


def temporal_disparity_evidence(
    current_disparity: torch.Tensor,
    past_disparity: torch.Tensor,
    flow_current_to_past: torch.Tensor,
    flow_past_to_current: torch.Tensor,
    *,
    current_valid: torch.Tensor,
    past_valid: torch.Tensor,
    current_rgb: torch.Tensor,
    past_rgb: torch.Tensor,
    fb_absolute_threshold: float = 0.5,
    fb_relative_threshold: float = 0.01,
    photometric_robust_scale: float = 0.1,
) -> TemporalDisparityEvidence:
    """Build backbone-independent causal t-k evidence on the current frame grid."""
    aligned = causal_warp(
        past_disparity,
        flow_current_to_past,
        source_valid=past_valid,
    )
    current_valid_b = current_valid[:, None] if current_valid.ndim == 3 else current_valid
    aligned_validity = aligned.valid & current_valid_b.bool()
    fb = forward_backward_consistency(
        flow_current_to_past,
        flow_past_to_current,
        absolute_threshold=fb_absolute_threshold,
        relative_threshold=fb_relative_threshold,
    )
    photo = photometric_consistency(
        current_rgb,
        past_rgb,
        flow_current_to_past,
        target_valid=current_valid_b,
        source_valid=past_valid,
        robust_scale=photometric_robust_scale,
    )
    signed = aligned.warped - current_disparity
    valid_f = aligned_validity.to(signed.dtype)
    flow_magnitude = torch.linalg.vector_norm(
        _flow_bchw(flow_current_to_past), dim=1, keepdim=True
    )
    return TemporalDisparityEvidence(
        aligned_past_disparity=aligned.warped,
        warp_support=aligned.support,
        aligned_validity=aligned_validity,
        forward_backward_error=fb.error,
        forward_backward_confidence=fb.confidence,
        photometric_residual=photo.robust_normalized_residual,
        absolute_disparity_disagreement=signed.abs() * valid_f,
        signed_disparity_disagreement=signed * valid_f,
        flow_magnitude=flow_magnitude,
    )


class BiDAFlowInferenceAdapter:
    """Frozen pairwise flow wrapper around BiDAVideo's vendored implementations.

    ``infer(target, source)`` returns target-to-source flow.  Supplying
    ``precomputed_flow`` bypasses the model and is useful for resumable evaluation.
    Model parameters are frozen, evaluation mode is deterministic, and default
    inference uses ``torch.inference_mode`` so no flow-model graph is retained.
    """

    def __init__(
        self,
        kind: Literal["sea_raft", "raft"] = "sea_raft",
        *,
        checkpoint: str | Path | None = None,
        device: str | torch.device | None = None,
        iterations: int | None = None,
    ) -> None:
        self.kind = kind
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.iterations = iterations or (4 if kind == "sea_raft" else 20)
        default_checkpoint = SEA_RAFT_CHECKPOINT if kind == "sea_raft" else RAFT_CHECKPOINT
        self.checkpoint = Path(checkpoint or default_checkpoint)
        if not self.checkpoint.exists():
            raise FileNotFoundError(
                f"{kind} checkpoint not found: {self.checkpoint}. "
                "RAFT here means optical RAFT, not RAFT-Stereo."
            )
        self.model, self._padder_cls = self._load_model()
        self.model.to(self.device).eval()
        self.model.requires_grad_(False)

    @staticmethod
    def _ensure_external_import_path() -> None:
        external_root = str(ROOT / "external")
        if external_root not in sys.path:
            sys.path.insert(0, external_root)

    def _load_model(self) -> tuple[torch.nn.Module, type]:
        self._ensure_external_import_path()
        if self.kind == "sea_raft":
            raft_module = importlib.import_module("bidavideo.third_party.SEA-RAFT.core.raft")
            utils_module = importlib.import_module(
                "bidavideo.third_party.SEA-RAFT.core.utils.utils"
            )
            args = SimpleNamespace(
                use_var=True,
                var_min=0,
                var_max=10,
                pretrain="resnet18",
                initial_dim=64,
                block_dims=[64, 128, 256],
                radius=4,
                dim=128,
                num_blocks=2,
                iters=self.iterations,
            )
        elif self.kind == "raft":
            raft_module = importlib.import_module("bidavideo.third_party.RAFT.core.raft")
            utils_module = importlib.import_module(
                "bidavideo.third_party.RAFT.core.utils.utils"
            )
            args = SimpleNamespace(mixed_precision=False, small=False, dropout=0.0)
        else:
            raise ValueError(f"unsupported flow model {self.kind!r}")
        model = raft_module.RAFT(args)
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        state = {key.removeprefix("module."): value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
        return model, utils_module.InputPadder

    def infer(
        self,
        target_rgb: torch.Tensor,
        source_rgb: torch.Tensor,
        *,
        precomputed_flow: torch.Tensor | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Return ``target -> source`` flow as ``[B,2,H,W]`` pixels."""
        if precomputed_flow is not None:
            flow = _flow_bchw(precomputed_flow).to(self.device, dtype=torch.float32)
            return resize_flow(flow, output_size) if output_size is not None else flow
        if target_rgb.shape != source_rgb.shape or target_rgb.ndim != 4:
            raise ValueError("target_rgb and source_rgb must share [B,3,H,W]")
        target = target_rgb.to(self.device, dtype=torch.float32)
        source = source_rgb.to(self.device, dtype=torch.float32)
        padder = self._padder_cls(target.shape)
        target_pad, source_pad = padder.pad(target, source)
        with torch.inference_mode():
            if self.kind == "sea_raft":
                output = self.model(
                    target_pad.contiguous(),
                    source_pad.contiguous(),
                    iters=self.iterations,
                    test_mode=True,
                )
                flow = output["flow"][-1]
            else:
                _low, flow = self.model(
                    target_pad.contiguous(),
                    source_pad.contiguous(),
                    iters=self.iterations,
                    test_mode=True,
                )
            flow = padder.unpad(flow)
        if output_size is not None:
            flow = resize_flow(flow, output_size)
        return flow

    def current_to_past(self, current_rgb: torch.Tensor, past_rgb: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.infer(current_rgb, past_rgb, **kwargs)

    def past_to_current(self, past_rgb: torch.Tensor, current_rgb: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.infer(past_rgb, current_rgb, **kwargs)


def original_noncausal_bidastabilizer_reference():
    """Import the original reproduction baseline; this path uses future frames."""
    BiDAFlowInferenceAdapter._ensure_external_import_path()
    module = importlib.import_module("bidavideo.models.core.bidastabilizer")
    return module.BiDAStabilizer
