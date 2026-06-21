from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import ExperimentConfig
from .types import FusionOutput, TemporalBatch


def _valid_mean_abs(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    finite = torch.isfinite(x)
    if mask is not None:
        finite = finite & (mask > 0)
    if not finite.any():
        return x.new_tensor(0.0)
    return torch.mean(torch.abs(x[finite]))


def compute_playground_loss(
    config: ExperimentConfig,
    output: FusionOutput,
    batch: TemporalBatch,
) -> tuple[torch.Tensor, dict[str, float]]:
    fused = output.fused_disparity
    weights = config.losses
    parts: dict[str, torch.Tensor] = {}

    parts["teacher"] = F.smooth_l1_loss(fused, batch.sav_disp)
    parts["spatial_teacher"] = F.smooth_l1_loss(fused, batch.s2m2_l_disp)
    parts["raw_fidelity"] = F.smooth_l1_loss(fused, batch.s2m2_s_disp)
    parts["residual"] = torch.mean(torch.abs(output.residual_map))
    parts["alpha_prior"] = F.smooth_l1_loss(output.alpha_map, torch.full_like(output.alpha_map, 0.5))
    parts["uncertainty"] = torch.mean(output.uncertainty_map)
    source_weights = torch.clamp(output.source_weights, 1e-6, 1.0)
    parts["source_weight_entropy"] = torch.mean(torch.sum(source_weights * torch.log(source_weights), dim=2, keepdim=True))

    if fused.shape[1] > 1:
        parts["temporal"] = F.smooth_l1_loss(fused[:, 1:] - fused[:, :-1], batch.sav_disp[:, 1:] - batch.sav_disp[:, :-1])
        parts["motion_compensated"] = parts["temporal"]
    else:
        parts["temporal"] = fused.new_tensor(0.0)
        parts["motion_compensated"] = fused.new_tensor(0.0)

    if batch.gt_disp is not None:
        parts["gt"] = _valid_mean_abs(fused - batch.gt_disp, batch.valid_mask)
    else:
        parts["gt"] = fused.new_tensor(0.0)

    loss = (
        weights.teacher * parts["teacher"]
        + weights.spatial_teacher * parts["spatial_teacher"]
        + weights.raw_fidelity * parts["raw_fidelity"]
        + weights.temporal * parts["temporal"]
        + weights.motion_compensated * parts["motion_compensated"]
        + weights.residual * parts["residual"]
        + weights.alpha_prior * parts["alpha_prior"]
        + weights.uncertainty * parts["uncertainty"]
        + weights.source_weight_entropy * parts["source_weight_entropy"]
        + weights.gt * parts["gt"]
    )
    metrics = {f"loss_{k}": float(v.detach().cpu()) for k, v in parts.items()}
    metrics["loss"] = float(loss.detach().cpu())
    return loss, metrics


@torch.no_grad()
def compute_playground_metrics(output: FusionOutput, batch: TemporalBatch) -> dict[str, float]:
    fused = output.fused_disparity
    metrics = {
        "fused_to_sav_mae": float(torch.mean(torch.abs(fused - batch.sav_disp)).cpu()),
        "fused_to_s2m2_l_mae": float(torch.mean(torch.abs(fused - batch.s2m2_l_disp)).cpu()),
        "raw_s2m2_s_to_sav_mae": float(torch.mean(torch.abs(batch.s2m2_s_disp - batch.sav_disp)).cpu()),
        "fixed_ema_reference_to_sav_mae": float(torch.mean(torch.abs(_fixed_ema(batch.s2m2_s_disp) - batch.sav_disp)).cpu()),
        "alpha_mean": float(output.alpha_map.mean().cpu()),
        "alpha_std": float(output.alpha_map.std().cpu()),
        "reset_mean": float(output.reset_map.mean().cpu()),
        "residual_abs_mean": float(torch.mean(torch.abs(output.residual_map)).cpu()),
        "uncertainty_mean": float(output.uncertainty_map.mean().cpu()),
        "w_raw_mean": float(output.source_weights[:, :, 0:1].mean().cpu()),
        "w_short_mean": float(output.source_weights[:, :, 1:2].mean().cpu()),
        "w_long_mean": float(output.source_weights[:, :, 2:3].mean().cpu()),
    }
    if fused.shape[1] > 1:
        metrics["raw_temporal_error"] = float(torch.mean(torch.abs(fused[:, 1:] - fused[:, :-1])).cpu())
        metrics["motion_compensated_temporal_error"] = metrics["raw_temporal_error"]
    else:
        metrics["raw_temporal_error"] = 0.0
        metrics["motion_compensated_temporal_error"] = 0.0
    if batch.gt_disp is not None:
        metrics["gt_disp_mae"] = float(_valid_mean_abs(fused - batch.gt_disp, batch.valid_mask).cpu())
    return metrics


def _fixed_ema(raw: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    frames = []
    previous = raw[:, 0]
    for i in range(raw.shape[1]):
        current = raw[:, i] if i == 0 else alpha * raw[:, i] + (1.0 - alpha) * previous
        frames.append(current)
        previous = current
    return torch.stack(frames, dim=1)
