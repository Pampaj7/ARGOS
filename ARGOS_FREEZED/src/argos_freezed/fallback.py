"""Exact raw fallback; kept explicit for auditability."""
import torch


def exact_raw_fallback(raw: torch.Tensor, proposal: torch.Tensor, accepted: torch.Tensor) -> torch.Tensor:
    return torch.where(accepted, proposal, raw)
