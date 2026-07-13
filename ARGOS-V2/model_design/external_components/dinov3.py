"""Frozen official DINOv3 ViT-L/16 adapter for ARGOS v2.

The implementation is imported from ``external/dinov3`` and the local
LVD-1689M checkpoint is loaded strictly.  No model code or weights are copied
into ARGOS v2 and no network fallback exists.

Tensor contract
---------------
* RGB input: ``[B,3,H,W]`` uint8 in [0,255], or floating point in [0,1]
  (floating [0,255] is accepted only when its range makes that unambiguous).
* preprocessed input: aspect-preserving bilinear resize, optional symmetric
  zero padding to an explicitly requested patch-divisible ``(H,W)``, followed
  by ImageNet mean/std normalization.
* patch maps: one ``[B,1024,H/16,W/16]`` tensor per requested zero-based block.
* optical flow: ``[B,2,H,W]`` target-to-source pixels on its own grid.

The validated ARGOS BiDA convention is reused for feature warping: current
pixels sample past features at ``grid + flow``, with ``align_corners=True``.
"""
from __future__ import annotations

import contextlib
import hashlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from model_design.external_components.bidavideo import WarpResult, causal_warp, resize_flow


ROOT = Path(__file__).resolve().parents[3]
DINOV3_ROOT = ROOT / "external/dinov3"
DINOV3_CHECKPOINT = (
    DINOV3_ROOT / "checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
)
EXPECTED_CHECKPOINT_SHA256 = "8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DINOInputMetadata:
    """Geometry of the deterministic aspect-preserving preprocessing."""

    source_size: tuple[int, int]
    resized_size: tuple[int, int]
    input_size: tuple[int, int]
    padding_ltrb: tuple[int, int, int, int]
    patch_grid: tuple[int, int]


@dataclass(frozen=True)
class DINOFeatureOutput:
    """Normalized dense patch maps and their input geometry."""

    layers: tuple[int, ...]
    feature_maps: tuple[torch.Tensor, ...]
    metadata: DINOInputMetadata

    def by_layer(self) -> dict[int, torch.Tensor]:
        return dict(zip(self.layers, self.feature_maps, strict=True))


@dataclass(frozen=True)
class DINORuntime:
    latency_ms_per_frame: float
    peak_gpu_memory_bytes: int


def checkpoint_sha256(path: Path = DINOV3_CHECKPOINT) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess_rgb(
    rgb: torch.Tensor,
    input_size: tuple[int, int],
) -> tuple[torch.Tensor, DINOInputMetadata]:
    """Resize without stretching, pad symmetrically, and normalize for LVD.

    ``input_size`` must be divisible by 16. Padding is black (zero in [0,1]
    RGB) before normalization, hence it takes the normalized black-pixel value;
    it is not valid image support. ARGOS experiments use exact 4:5 canvases
    (256x320, 320x400), so SCARED-C frames require no padding.
    """
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError(f"rgb must be [B,3,H,W], got {tuple(rgb.shape)}")
    target_h, target_w = map(int, input_size)
    if target_h <= 0 or target_w <= 0 or target_h % 16 or target_w % 16:
        raise ValueError("DINO input height and width must be positive multiples of 16")
    source_h, source_w = rgb.shape[-2:]
    scale = min(target_h / source_h, target_w / source_w)
    resized_h = max(1, min(target_h, round(source_h * scale)))
    resized_w = max(1, min(target_w, round(source_w * scale)))

    if rgb.is_floating_point():
        image = rgb.float()
        maximum = float(image.detach().amax()) if image.numel() else 0.0
        minimum = float(image.detach().amin()) if image.numel() else 0.0
        if minimum < 0 or maximum > 255.0 + 1e-3:
            raise ValueError("floating RGB must be in [0,1] or [0,255]")
        if maximum > 1.0 + 1e-3:
            image = image / 255.0
    else:
        image = rgb.float() / 255.0
    image = F.interpolate(image, (resized_h, resized_w), mode="bilinear", align_corners=False, antialias=True)
    pad_h, pad_w = target_h - resized_h, target_w - resized_w
    top, left = pad_h // 2, pad_w // 2
    bottom, right = pad_h - top, pad_w - left
    image = F.pad(image, (left, right, top, bottom), mode="constant", value=0.0)
    mean = image.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = image.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    image = (image - mean) / std
    metadata = DINOInputMetadata(
        source_size=(source_h, source_w),
        resized_size=(resized_h, resized_w),
        input_size=(target_h, target_w),
        padding_ltrb=(left, top, right, bottom),
        patch_grid=(target_h // 16, target_w // 16),
    )
    return image, metadata


class FrozenDINOv3(nn.Module):
    """Official local ViT-L/16 with frozen, deterministic feature extraction."""

    patch_size = 16
    embed_dim = 1024
    depth = 24
    n_storage_tokens = 4

    def __init__(
        self,
        *,
        checkpoint: Path | str = DINOV3_CHECKPOINT,
        device: torch.device | str = "cpu",
        verify_hash: bool = True,
        default_no_grad: bool = True,
        autocast_dtype: torch.dtype | None = torch.bfloat16,
    ) -> None:
        super().__init__()
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if verify_hash and checkpoint_sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
            raise RuntimeError("DINOv3 checkpoint SHA-256 does not match the validated LVD-1689M file")
        if not DINOV3_ROOT.is_dir():
            raise FileNotFoundError(DINOV3_ROOT)
        if str(DINOV3_ROOT) not in sys.path:
            sys.path.insert(0, str(DINOV3_ROOT))
        from dinov3.hub.backbones import dinov3_vitl16

        # The official constructor converts this local path to file:// and loads
        # with strict=True; it therefore has no download path in this adapter.
        model = dinov3_vitl16(pretrained=True, weights=str(checkpoint))
        actual = (model.patch_size, model.embed_dim, model.n_blocks, model.n_storage_tokens)
        expected = (self.patch_size, self.embed_dim, self.depth, self.n_storage_tokens)
        if actual != expected:
            raise RuntimeError(f"unexpected DINO architecture {actual}, expected {expected}")
        model.eval()
        model.requires_grad_(False)
        self.model = model.to(device)
        self.default_no_grad = bool(default_no_grad)
        self.autocast_dtype = autocast_dtype
        self.checkpoint = checkpoint
        self.eval()

    def train(self, mode: bool = True):
        # Frozen DINO stays deterministic even when a parent selector trains.
        super().train(False)
        self.model.eval()
        return self

    def extract(
        self,
        rgb: torch.Tensor,
        *,
        layers: Sequence[int] = (23,),
        input_size: tuple[int, int] = (256, 320),
        no_grad: bool | None = None,
    ) -> DINOFeatureOutput:
        selected = tuple(int(layer) for layer in layers)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("layers must be non-empty and unique")
        if any(layer < 0 or layer >= self.depth for layer in selected):
            raise ValueError(f"layer indices must be in [0,{self.depth - 1}]")
        image, metadata = preprocess_rgb(rgb, input_size)
        image = image.to(next(self.model.parameters()).device)
        use_no_grad = self.default_no_grad if no_grad is None else bool(no_grad)
        grad_context = torch.inference_mode() if use_no_grad else contextlib.nullcontext()
        amp_enabled = image.is_cuda and self.autocast_dtype is not None
        amp_context = torch.autocast("cuda", dtype=self.autocast_dtype, enabled=amp_enabled)
        self.model.eval()
        with grad_context, amp_context:
            maps = self.model.get_intermediate_layers(
                image, n=selected, reshape=True, norm=True
            )
        return DINOFeatureOutput(selected, tuple(maps), metadata)

    def measure(
        self,
        rgb: torch.Tensor,
        *,
        layers: Sequence[int] = (23,),
        input_size: tuple[int, int] = (256, 320),
        warmup: int = 2,
        repeats: int = 5,
    ) -> tuple[DINOFeatureOutput, DINORuntime]:
        if repeats < 1 or warmup < 0:
            raise ValueError("invalid timing iteration count")
        for _ in range(warmup):
            output = self.extract(rgb, layers=layers, input_size=input_size)
        device = next(self.model.parameters()).device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(repeats):
            output = self.extract(rgb, layers=layers, input_size=input_size)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device)
        else:
            peak = 0
        latency = 1000.0 * (time.perf_counter() - start) / (repeats * rgb.shape[0])
        return output, DINORuntime(latency, peak)


def warp_dino_feature(
    memory_feature: torch.Tensor,
    flow_current_to_memory: torch.Tensor,
    *,
    memory_valid: torch.Tensor | None = None,
) -> WarpResult:
    """Align a past DINO map using only canonical BiDA resize and warp helpers."""
    flow = resize_flow(flow_current_to_memory, memory_feature.shape[-2:])
    if memory_valid is not None and memory_valid.shape[-2:] != memory_feature.shape[-2:]:
        memory_valid = F.interpolate(memory_valid.float(), memory_feature.shape[-2:], mode="nearest") > 0.5
    return causal_warp(memory_feature, flow, source_valid=memory_valid)


__all__ = [
    "DINOV3_CHECKPOINT",
    "DINOFeatureOutput",
    "DINOInputMetadata",
    "DINORuntime",
    "FrozenDINOv3",
    "checkpoint_sha256",
    "preprocess_rgb",
    "warp_dino_feature",
]
