from __future__ import annotations

import torch


def teacher_metrics(s2m2: torch.Tensor, refined: torch.Tensor, teacher: torch.Tensor, delta: torch.Tensor) -> dict[str, float]:
    before = torch.mean(torch.abs(s2m2 - teacher))
    after = torch.mean(torch.abs(refined - teacher))
    return {
        "teacher_mae_before": float(before.detach().cpu()),
        "teacher_mae_after": float(after.detach().cpu()),
        "residual_mean": float(delta.mean().detach().cpu()),
        "residual_std": float(delta.std().detach().cpu()),
        "residual_min": float(delta.min().detach().cpu()),
        "residual_max": float(delta.max().detach().cpu()),
    }


def gt_metrics(refined: torch.Tensor, gt_disp: torch.Tensor, gt_depth: torch.Tensor, valid: torch.Tensor, fx: torch.Tensor, baseline_mm: torch.Tensor) -> dict[str, float]:
    mask = valid.bool() & torch.isfinite(gt_disp) & torch.isfinite(gt_depth) & (gt_disp > 0) & (gt_depth > 0)
    if mask.sum() == 0:
        return {}
    disp_err = torch.abs(refined[mask] - gt_disp[mask])
    pred_depth = fx.reshape(-1, 1, 1, 1) * baseline_mm.reshape(-1, 1, 1, 1) / torch.clamp(refined, min=1e-6)
    depth_err = torch.abs(pred_depth[mask] - gt_depth[mask])
    return {
        "gt_disp_mae": float(disp_err.mean().detach().cpu()),
        "gt_depth_mae": float(depth_err.mean().detach().cpu()),
        "gt_bad_2mm": float((depth_err > 2.0).float().mean().detach().cpu() * 100.0),
    }

