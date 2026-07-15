"""Calibration and exact identity-preserving authorization for ARGOS v2."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from model_design.models.raw_error_detector import RawErrorOutput


@dataclass(frozen=True)
class OperatingMode:
    name: str
    probability_threshold: float
    error_threshold_px: float
    uncertainty_threshold_px: float
    maximum_update_px: float = 3.0

    def as_dict(self) -> dict:
        return asdict(self)


def calibrated_probability(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return torch.sigmoid(logits / float(temperature))


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid: torch.Tensor,
    *,
    split: str,
) -> float:
    """Fit one scalar only when the caller proves validation provenance."""
    if split != "validation":
        raise ValueError("temperature fitting is validation-only")
    selected_logits = logits.detach()[valid.bool()].float()
    selected_labels = labels.detach()[valid.bool()].float()
    if not selected_logits.numel():
        return 1.0
    log_temperature = torch.zeros((), device=selected_logits.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            selected_logits / temperature, selected_labels
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20))


def authorization_mask(
    output: RawErrorOutput,
    *,
    mode: OperatingMode,
    temperature: float,
    aligned_valid: torch.Tensor,
    warp_support: torch.Tensor,
    proposal_update: torch.Tensor,
) -> torch.Tensor:
    probability = calibrated_probability(output.logits, temperature)
    bounded = torch.isfinite(proposal_update) & (
        proposal_update.abs() <= mode.maximum_update_px + 1e-6
    )
    return (
        (probability >= mode.probability_threshold)
        & (output.mu >= mode.error_threshold_px)
        & (output.sigma <= mode.uncertainty_threshold_px)
        & aligned_valid.bool()
        & warp_support.bool()
        & bounded
    )


def authorized_update(
    raw: torch.Tensor,
    proposal_update: torch.Tensor,
    authorization: torch.Tensor,
) -> torch.Tensor:
    """Rejected pixels are bit-exact raw; no soft blend is introduced."""
    return torch.where(authorization.bool(), raw + proposal_update, raw)


__all__ = [
    "OperatingMode", "authorization_mask", "authorized_update",
    "calibrated_probability", "fit_temperature",
]
