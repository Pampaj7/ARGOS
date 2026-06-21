"""
lib/flow.py – Frozen RAFT-based optical flow wrapper.

Provides a thin, typed interface over the local RAFT implementation bundled
with StereoAnyVideo. The model is always loaded in eval mode and its
parameters are frozen; only the fusion refiner is trained.

Features:
- Confidence estimation via flow magnitude and forward-backward consistency
- Occlusion mask from forward-backward check
- Bilinear warping utility used by the refiner at training and inference time
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# RAFT import – we reuse the copy bundled inside StereoAnyVideo's third_party.
# We prefer this over a fresh clone to avoid duplication.
# ──────────────────────────────────────────────────────────────────────────────
_RAFT_ROOT = Path("/dtu/p1/leopam/ARGOS/external/video_stereo_repos/stereoanyvideo/third_party/RAFT")

if str(_RAFT_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAFT_ROOT))

from core.raft import RAFT  # noqa: E402
from core.utils.utils import bilinear_sampler  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Public warp utility
# ──────────────────────────────────────────────────────────────────────────────

def warp_disp(disp: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp a disparity map with an optical-flow field.

    Args:
        disp:  (B, 1, H, W) disparity map to be warped.
        flow:  (B, 2, H, W) forward optical flow (dx, dy) in pixels.

    Returns:
        Warped disparity of the same shape as *disp*.
    """
    B, _, H, W = disp.shape
    # Build sampling grid: pixel coordinates shifted by flow
    device = disp.device
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij",
    )
    grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
    coords = grid + flow  # (B, 2, H, W)

    # Normalise to [-1, 1] for grid_sample
    norm_x = 2.0 * coords[:, 0] / (W - 1) - 1.0
    norm_y = 2.0 * coords[:, 1] / (H - 1) - 1.0
    grid_norm = torch.stack([norm_x, norm_y], dim=-1)  # (B, H, W, 2)

    return F.grid_sample(disp, grid_norm, mode="bilinear", padding_mode="border", align_corners=True)


def flow_confidence(
    flow_fwd: torch.Tensor,
    flow_bwd: torch.Tensor,
    magnitude_threshold: float = 20.0,
    consistency_threshold: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute flow confidence and occlusion mask.

    Uses the forward-backward consistency check: a pixel is considered
    occluded when |flow_fwd(p) + flow_bwd(p + flow_fwd(p))| > threshold.

    Args:
        flow_fwd:  (B, 2, H, W) forward flow (frame t-1 → t).
        flow_bwd:  (B, 2, H, W) backward flow (frame t → t-1).
        magnitude_threshold: maximum reliable flow magnitude (pixels).
        consistency_threshold: FB error threshold for occlusion (pixels).

    Returns:
        confidence:  (B, 1, H, W) in [0, 1]. Higher = more reliable.
        occlusion:   (B, 1, H, W) binary. 1 = occluded pixel.
    """
    B, _, H, W = flow_fwd.shape
    device = flow_fwd.device

    # Warp backward flow with forward flow to get cyclic consistency error
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij",
    )
    base = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
    coords_fwd = base + flow_fwd  # where fwd flow lands
    norm_x = 2.0 * coords_fwd[:, 0] / (W - 1) - 1.0
    norm_y = 2.0 * coords_fwd[:, 1] / (H - 1) - 1.0
    grid_norm = torch.stack([norm_x, norm_y], dim=-1)

    flow_bwd_at_fwd = F.grid_sample(
        flow_bwd, grid_norm, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    # Cyclic error: should be ≈ 0 for non-occluded pixels
    fb_error = torch.linalg.vector_norm(flow_fwd + flow_bwd_at_fwd, dim=1, keepdim=True)
    occlusion = (fb_error > consistency_threshold).float()

    # Magnitude-based confidence: decays for large flows
    magnitude = torch.linalg.vector_norm(flow_fwd, dim=1, keepdim=True)
    mag_conf = torch.clamp(1.0 - magnitude / magnitude_threshold, min=0.0, max=1.0)

    # FB-based confidence
    fb_conf = torch.clamp(1.0 - fb_error / consistency_threshold, min=0.0, max=1.0)

    confidence = mag_conf * fb_conf * (1.0 - occlusion)
    return confidence, occlusion


# ──────────────────────────────────────────────────────────────────────────────
# Frozen RAFT wrapper
# ──────────────────────────────────────────────────────────────────────────────

class FrozenRAFT(nn.Module):
    """Frozen RAFT optical-flow model.

    Wraps the local RAFT implementation. Parameters are kept frozen during
    fusion-refiner training. Accepts uint8-range or [0,1]-range RGB images.

    Args:
        checkpoint:  Path to a RAFT *_things.pth or similar checkpoint.
                     If None, the model is initialised with random weights
                     (useful for smoke tests / debugging).
        iters:       Number of RAFT recurrent update iterations.
        small:       Use the smaller RAFT-Small variant.
    """

    def __init__(
        self,
        checkpoint: Optional[Path] = None,
        iters: int = 12,
        small: bool = False,
    ) -> None:
        super().__init__()
        args = SimpleNamespace(
            small=small,
            mixed_precision=False,
            alternate_corr=False,
        )
        self.raft = RAFT(args)
        self.iters = iters
        self._checkpoint = checkpoint

        if checkpoint is not None and Path(checkpoint).exists():
            state = torch.load(checkpoint, map_location="cpu")
            # RAFT checkpoints may be wrapped in DataParallel
            if any(k.startswith("module.") for k in state):
                state = {k[len("module."):]: v for k, v in state.items()}
            self.raft.load_state_dict(state, strict=True)

        # Freeze all parameters — only the refiner is trained
        for p in self.raft.parameters():
            p.requires_grad_(False)
        self.raft.eval()

    @torch.no_grad()
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Estimate forward optical flow from img1 to img2.

        Args:
            img1:  (B, 3, H, W) RGB frame t-1, values in [0, 1].
            img2:  (B, 3, H, W) RGB frame t,   values in [0, 1].

        Returns:
            flow:  (B, 2, H, W) forward flow in pixels.
        """
        # RAFT internally normalises by dividing by 255 and then shifting
        # to [-1, 1], so we pass images scaled to [0, 255]
        img1_u8 = (img1 * 255.0).clamp(0, 255)
        img2_u8 = (img2 * 255.0).clamp(0, 255)
        _, flow_up = self.raft(img1_u8, img2_u8, iters=self.iters, test_mode=True)
        return flow_up

    def train(self, mode: bool = True) -> "FrozenRAFT":
        # Always keep RAFT in eval mode regardless of the outer context
        self.raft.eval()
        return self
