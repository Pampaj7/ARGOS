"""PPMStereo Pick-and-Play scoring wrapper.

Original repository: external/PPMStereo
Commit: d0ccf7705145502c1eea49e7be0ddeafbcfd6a08
Original source paths:
- models/core/ppmstereo.py: compute_qk_similarity, quality-aware memory assessment,
  top-k selection, dynamic memory modulation, flash_attn readout
- models/core/ppmtereo_update.py: Attention_qk, temporal positional encoding

Tensor convention in the original Pick-and-Play block:
- query/key: [B, C, T, H, W]
- sim_score: [B, 1, T, T]
- frame_confidence: [B, 1, 1, T]
- frame_score = exp(-strive_time / (strive_time.sum(dim=3)+T)) * sim_score + frame_confidence
- selected_score_norm = selected_score / selected_score.mean()
- dynamic modulation multiplies selected key channels by selected_score_norm and adds temporal PE.

Adaptation:
- the full readout remains reference-only because it is cost-volume/update-block/flash-attn coupled;
- this wrapper preserves the executable scoring/top-k/modulation math and accepts ARGOS feature adapters.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
PPM_ROOT = ROOT / "external/PPMStereo"


def _import_original_ppm_class():
    sys.path.insert(0, str(PPM_ROOT))
    try:
        from models.core.ppmstereo import PPMStereo  # type: ignore
    finally:
        try:
            sys.path.remove(str(PPM_ROOT))
        except ValueError:
            pass
    return PPMStereo


def compute_qk_similarity_exact(q: torch.Tensor, k: torch.Tensor, t: int) -> torch.Tensor:
    """Faithful standalone version of PPMStereo.compute_qk_similarity."""
    q = q.permute(0, 2, 1, 3, 4).reshape(q.shape[0] * t, q.shape[1], q.shape[3], q.shape[4])
    k = k.permute(0, 2, 1, 3, 4).reshape(k.shape[0] * t, k.shape[1], k.shape[3], k.shape[4])
    bt, _channels, height, width = q.shape
    b = bt // t
    pool = torch.nn.AdaptiveMaxPool2d((max(height // 4, 1), max(width // 4, 1)))
    q_, k_ = pool(q), pool(k)
    q_ = q_.mean(dim=1).reshape(b, t, -1)
    k_ = k_.mean(dim=1).reshape(b, t, -1)
    return F.cosine_similarity(q_.unsqueeze(1), k_.unsqueeze(2), dim=-1).unsqueeze(1)


def quality_aware_scores(
    similarity: torch.Tensor,
    frame_confidence: torch.Tensor,
    strive_time: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PPMStereo score formula for a single current frame's memory candidates.

    Args:
        similarity: [B, M]
        frame_confidence: [B, M]
        strive_time: [B, M], initialized to ones in the original block
    """
    if strive_time is None:
        strive_time = torch.ones_like(similarity)
    add_item = similarity.shape[1]
    penalty = torch.exp(-strive_time / (strive_time.sum(dim=1, keepdim=True) + add_item))
    return penalty * similarity + frame_confidence, penalty


def topk_with_modulation(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return original top-k indices plus diagnostic play weights from selected_score_norm."""
    k = min(k, scores.shape[1])
    values, indices = torch.topk(scores, k=k, dim=1)
    selected_score_norm = values / values.mean(dim=1, keepdim=True).clamp_min(1e-6)
    weights = selected_score_norm.clamp_min(0)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return indices, values, weights


def update_strive_time(strive_time: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    updated = strive_time.clone()
    updated.scatter_add_(1, indices, torch.ones_like(indices, dtype=strive_time.dtype))
    return updated


def self_check() -> None:
    torch.manual_seed(0)
    q = torch.randn(1, 4, 3, 8, 8)
    k = torch.randn(1, 4, 3, 8, 8)
    local = compute_qk_similarity_exact(q, k, t=3)
    try:
        PPMStereo = _import_original_ppm_class()
        original = PPMStereo.compute_qk_similarity(None, q, k, t=3)
        assert torch.allclose(local, original, atol=1e-6), "PPM similarity drifted from original method"
    except Exception as exc:
        raise AssertionError(f"could not compare against original PPMStereo method: {exc}") from exc
    score, penalty = quality_aware_scores(torch.ones(1, 5), torch.zeros(1, 5))
    assert score.shape == penalty.shape == (1, 5)


if __name__ == "__main__":
    self_check()
    print("ppmstereo_wrapper ok")

