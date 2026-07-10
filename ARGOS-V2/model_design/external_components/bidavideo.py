"""BiDAVideo / BiDAStabilizer alignment wrapper.

Original repository: external/bidavideo
Commit: dae817df1ceaafcb865ebd9c7aa44b16c535e856
Original source paths:
- models/core/bidastabilizer.py: BiDAStabilizer.flow_warp, bidirectional forward/backward path
- train_utils/losses.py: standalone flow_warp and bidirectional_alignment

Tensor convention:
- input tensor: [B, C, H, W]
- flow: [B, 2, H, W] or [B, H, W, 2]
- flow is target-to-source in pixels, sampled as grid + flow
- grid_sample uses mode=bilinear, padding_mode=zeros, align_corners=True

Adaptation:
- imports the original standalone flow_warp unchanged;
- adds a support mask with the same coordinate convention for measurement only.

Causal status:
- flow_warp itself is causal if called with past-only source and current-to-past flow.
- BiDAStabilizer.forward is not causal: it uses future frames and backward propagation.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
BIDA_ROOT = ROOT / "external/bidavideo"
BIDA_LOSSES = BIDA_ROOT / "train_utils/losses.py"


def _import_original_flow_warp():
    sys.path.insert(0, str(BIDA_ROOT))
    try:
        from train_utils.losses import flow_warp  # type: ignore
    finally:
        try:
            sys.path.remove(str(BIDA_ROOT))
        except ValueError:
            pass
    return flow_warp


ORIGINAL_FLOW_WARP = _import_original_flow_warp()


def source_sha256() -> str:
    return hashlib.sha256(BIDA_LOSSES.read_bytes()).hexdigest()


def warp_disparity_original(source: torch.Tensor, flow_t_to_source: torch.Tensor) -> torch.Tensor:
    """Call BiDAVideo's original flow_warp implementation."""
    return ORIGINAL_FLOW_WARP(source, flow_t_to_source)


def support_mask(source: torch.Tensor, flow_t_to_source: torch.Tensor) -> torch.Tensor:
    """Same grid convention as BiDA flow_warp, returned as [B,1,H,W] float."""
    if flow_t_to_source.shape[-1] != 2:
        flow = flow_t_to_source.permute(0, 2, 3, 1)
    else:
        flow = flow_t_to_source
    b, _c, h, w = source.shape
    if flow.shape != (b, h, w, 2):
        raise ValueError(f"flow shape {tuple(flow.shape)} incompatible with source {tuple(source.shape)}")
    ys, xs = torch.meshgrid(
        torch.arange(h, device=source.device, dtype=source.dtype),
        torch.arange(w, device=source.device, dtype=source.dtype),
        indexing="ij",
    )
    grid = torch.stack((xs, ys), dim=-1).unsqueeze(0) + flow.to(source.dtype)
    mask = (grid[..., 0] >= 0) & (grid[..., 0] <= w - 1) & (grid[..., 1] >= 0) & (grid[..., 1] <= h - 1)
    return mask[:, None].to(source.dtype)


def self_check() -> None:
    x = torch.arange(9, dtype=torch.float32).view(1, 1, 3, 3)
    zero = torch.zeros(1, 2, 3, 3)
    warped = warp_disparity_original(x, zero)
    assert torch.allclose(warped, x), "BiDA zero-flow warp changed input"
    assert support_mask(x, zero).sum().item() == 9


if __name__ == "__main__":
    self_check()
    print("bidavideo_wrapper ok")

