from __future__ import annotations

import torch
import torch.nn.functional as F


def edge_aware_smoothness(delta: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(delta[..., :, 1:] - delta[..., :, :-1])
    dy = torch.abs(delta[..., 1:, :] - delta[..., :-1, :])
    rgb_dx = torch.mean(torch.abs(rgb[..., :, 1:] - rgb[..., :, :-1]), dim=1, keepdim=True)
    rgb_dy = torch.mean(torch.abs(rgb[..., 1:, :] - rgb[..., :-1, :]), dim=1, keepdim=True)
    wx = torch.exp(-10.0 * rgb_dx)
    wy = torch.exp(-10.0 * rgb_dy)
    return (dx * wx).mean() + (dy * wy).mean()


def refiner_loss(
    refined: torch.Tensor,
    teacher: torch.Tensor,
    delta: torch.Tensor,
    rgb: torch.Tensor,
    disp_window: torch.Tensor | None = None,
    teacher_weight: float = 1.0,
    temporal_weight: float = 0.0,
    residual_l1_weight: float = 0.05,
    smoothness_weight: float = 0.05,
    disp_scale: float = 128.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    teacher_loss = F.smooth_l1_loss(refined, teacher)
    if disp_window is not None and temporal_weight > 0:
        temporal_target = torch.median(disp_window * disp_scale, dim=1, keepdim=True).values
        temporal_loss = F.smooth_l1_loss(refined, temporal_target)
    else:
        temporal_loss = refined.new_tensor(0.0)
    residual_l1 = torch.mean(torch.abs(delta))
    smooth = edge_aware_smoothness(delta, rgb)
    loss = (
        teacher_weight * teacher_loss
        + temporal_weight * temporal_loss
        + residual_l1_weight * residual_l1
        + smoothness_weight * smooth
    )
    return loss, {
        "loss_teacher": float(teacher_loss.detach().cpu()),
        "loss_temporal": float(temporal_loss.detach().cpu()),
        "loss_residual_l1": float(residual_l1.detach().cpu()),
        "loss_smooth": float(smooth.detach().cpu()),
    }
