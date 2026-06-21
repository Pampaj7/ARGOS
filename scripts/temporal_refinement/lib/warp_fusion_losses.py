"""
lib/warp_fusion_losses.py – Loss functions for CausalWarpedFusionRefiner.

Key design choices:
- No pixel-aligned temporal difference loss (was the root cause of alpha collapse).
- Motion-compensated temporal consistency: compare fused deltas to SAV deltas
  after accounting for flow (warp-aligned).
- Geometric supervision uses SAV teacher; S2M2-L (spatial) is used only when
  reliable (gated by a NaN / outlier check).
- Reset supervision from flow confidence + occlusion (no heuristic RGB diff).
- Weak alpha prior around 0.5 decays over training.
- Anti-collapse loss: penalise std(alpha) < epsilon.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from scripts.temporal_refinement.lib.losses import edge_aware_smoothness
from scripts.temporal_refinement.lib.flow import warp_disp


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _is_reliable(tensor: torch.Tensor, outlier_quantile: float = 0.98) -> torch.Tensor:
    """Boolean mask: True where values are finite and below 99th percentile."""
    finite = torch.isfinite(tensor)
    if finite.any():
        vals = tensor[finite].float()
        if vals.numel() > 10_000_000:
            step = vals.numel() // 10_000_000 + 1
            vals = vals[::step]
        q = torch.quantile(vals, outlier_quantile)
        return finite & (tensor < q)
    return finite


def balanced_bce(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Computes binary cross entropy with inverse frequency weighting."""
    pred = pred.float()
    target = target.float()
    
    pos_mask = target > 0.5
    neg_mask = target <= 0.5
    
    pos_count = pos_mask.sum()
    neg_count = neg_mask.sum()
    
    # Avoid zero division
    pos_weight = 1.0 / (pos_count + 1e-8)
    neg_weight = 1.0 / (neg_count + 1e-8)
    
    weight = torch.where(pos_mask, pos_weight, neg_weight)
    
    return F.binary_cross_entropy(
        pred, target, weight=weight, reduction="sum"
    ) / 2.0


# ──────────────────────────────────────────────────────────────────────────────
# Main loss
# ──────────────────────────────────────────────────────────────────────────────

def warp_fusion_loss(
    out: dict[str, torch.Tensor],
    epoch: int,
    *,
    # loss weights
    sav_weight: float = 0.40,
    spatial_weight: float = 0.15,
    raw_fidelity_weight: float = 0.10,
    motion_comp_weight: float = 0.25,
    residual_l1_weight: float = 0.08,
    edge_weight: float = 0.04,
    alpha_prior_weight: float = 0.02,
    reset_weight: float = 0.04,
    anti_collapse_weight: float = 0.05,
    # hyper-parameters
    alpha_prior: float = 0.5,
    alpha_prior_decay_epochs: int = 20,
    alpha_collapse_std_min: float = 0.02,
    disp_norm: float = 128.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the composite training loss.

    Args:
        out:   dict produced by ``train_warp_fusion.forward_sequence``.
               Keys: fused, raw, spatial, sav, alpha, reset, residual,
                     rgb, flow, confidence, occlusion, warped_prev.
        epoch: current training epoch (1-indexed).

    Returns:
        (scalar loss tensor, dict of float metrics for logging)
    """
    fused    = out["fused"]        # (B, T, 1, H, W)
    raw      = out["raw"]
    spatial  = out["spatial"]
    sav      = out["sav"]
    alpha    = out["alpha"]
    reset    = out["reset"]
    residual = out["residual"]
    rgb      = out["rgb"]
    flow     = out["flow"]              # (B, T, 2, H, W) — zero at t=0
    conf     = out["confidence"]        # (B, T, 1, H, W)
    occ      = out["occlusion"]         # (B, T, 1, H, W)
    warped_prev = out["warped_prev"]    # (B, T, 1, H, W)

    B, T, _, H, W = fused.shape

    # ── Geometric losses ─────────────────────────────────────────────────────

    # SAV teacher (always used)
    loss_sav = F.smooth_l1_loss(fused, sav)

    # Spatial / S2M2-L supervision only when reliable
    spatial_mask = _is_reliable(spatial).float()
    if spatial_mask.any():
        loss_spatial = F.smooth_l1_loss(fused * spatial_mask, spatial * spatial_mask)
    else:
        loss_spatial = fused.new_tensor(0.0)

    # Raw fidelity: stay close to the S2M2-S input
    loss_raw = F.smooth_l1_loss(fused, raw)

    # ── Motion-compensated temporal consistency ───────────────────────────────
    # Compare (fused_t - warped_fused_{t-1}) to (sav_t - warped_sav_{t-1}).
    # Only at t ≥ 1, and only in confident / non-occluded regions.
    if T > 1:
        # warped_prev already contains warp(fused_{t-1}, flow_{t→t-1})
        # Build same for SAV: warp(sav_{t-1}, flow)
        flow_flat  = flow[:, 1:].reshape(B * (T - 1), 2, H, W)   # (B*(T-1), 2, H, W)
        sav_prev   = sav[:, :-1].reshape(B * (T - 1), 1, H, W)
        fused_prev = fused[:, :-1].reshape(B * (T - 1), 1, H, W)

        warped_sav_prev   = warp_disp(sav_prev,   flow_flat)
        warped_fused_prev = warp_disp(fused_prev,  flow_flat)

        sav_curr   = sav[:, 1:].reshape(B * (T - 1), 1, H, W)
        fused_curr = fused[:, 1:].reshape(B * (T - 1), 1, H, W)
        conf_curr  = conf[:, 1:].reshape(B * (T - 1), 1, H, W)
        occ_curr   = occ[:, 1:].reshape(B * (T - 1), 1, H, W)

        valid = conf_curr * (1.0 - occ_curr)   # (B*(T-1), 1, H, W)
        valid_sum = valid.sum().clamp(min=1.0)

        delta_fused = fused_curr - warped_fused_prev
        delta_sav   = sav_curr   - warped_sav_prev

        loss_motion = (valid * torch.abs(delta_fused - delta_sav)).sum() / valid_sum
    else:
        loss_motion = fused.new_tensor(0.0)

    # ── Residual regularisation ───────────────────────────────────────────────
    loss_residual = torch.mean(torch.abs(residual))

    # Edge-aware residual smoothness
    loss_edge = torch.stack([
        edge_aware_smoothness(residual[:, t], rgb[:, t])
        for t in range(T)
    ]).mean()

    # ── Alpha prior (decays to zero) ──────────────────────────────────────────
    prior_scale = max(0.0, 1.0 - float(epoch - 1) / max(1, alpha_prior_decay_epochs))
    loss_alpha_prior = prior_scale * F.smooth_l1_loss(
        alpha, torch.full_like(alpha, alpha_prior)
    )

    # ── Anti-collapse: penalise near-zero spatial variance of alpha ───────────
    alpha_std = alpha.std(dim=[-2, -1], keepdim=True)   # per (B, T, 1)
    loss_anti_collapse = F.relu(alpha_collapse_std_min - alpha_std).mean()

    # ── Reset supervision from flow confidence + occlusion ───────────────────
    if T > 1:
        # High occlusion or low confidence → should reset
        reset_target = torch.clamp(
            occ[:, 1:] + (1.0 - conf[:, 1:]),
            min=0.0, max=1.0,
        )
        loss_reset = balanced_bce(reset[:, 1:], reset_target)
    else:
        loss_reset = fused.new_tensor(0.0)

    # ── Total ─────────────────────────────────────────────────────────────────
    loss = (
        sav_weight             * loss_sav
        + spatial_weight       * loss_spatial
        + raw_fidelity_weight  * loss_raw
        + motion_comp_weight   * loss_motion
        + residual_l1_weight   * loss_residual
        + edge_weight          * loss_edge
        + alpha_prior_weight   * loss_alpha_prior
        + anti_collapse_weight * loss_anti_collapse
        + reset_weight         * loss_reset
    )

    metrics: dict[str, float] = {
        "loss":             float(loss.detach().cpu()),
        "loss_sav":         float(loss_sav.detach().cpu()),
        "loss_spatial":     float(loss_spatial.detach().cpu()),
        "loss_raw":         float(loss_raw.detach().cpu()),
        "loss_motion":      float(loss_motion.detach().cpu()),
        "loss_residual":    float(loss_residual.detach().cpu()),
        "loss_edge":        float(loss_edge.detach().cpu()),
        "loss_alpha_prior": float(loss_alpha_prior.detach().cpu()),
        "loss_anti_collapse": float(loss_anti_collapse.detach().cpu()),
        "loss_reset":       float(loss_reset.detach().cpu()),
        "prior_scale":      prior_scale,
        "alpha_mean":       float(alpha.mean().detach().cpu()),
        "alpha_std":        float(alpha.std().detach().cpu()),
        "reset_mean":       float(reset.mean().detach().cpu()),
        "residual_abs_mean":float(loss_residual.detach().cpu()),
    }
    return loss, metrics
