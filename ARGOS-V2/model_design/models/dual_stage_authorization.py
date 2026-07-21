"""Deterministic veto-only composition for ARGOS v2 authorization."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from model_design.models.proposal_applicability_detector import ProposalApplicabilityOutput


@dataclass(frozen=True)
class VetoPolicy:
    """A frozen logical policy; unset thresholds do not contribute a veto."""

    name: str
    maximum_update_px: float | None = None
    patch_mean_maximum_update_px: float | None = None
    patch_kernel: int = 5
    harmful_probability_threshold: float | None = None
    predicted_utility_ceiling_px: float | None = None
    uncertainty_floor_px: float | None = None
    require_harmful_class: bool = False
    p4_logic: str = "any"

    def __post_init__(self) -> None:
        if self.p4_logic not in {"any", "all"}:
            raise ValueError("p4_logic must be 'any' or 'all'")
        if self.patch_kernel < 1 or self.patch_kernel % 2 == 0:
            raise ValueError("patch_kernel must be a positive odd integer")
        for value in (self.maximum_update_px, self.patch_mean_maximum_update_px):
            if value is not None and value < 0:
                raise ValueError("magnitude thresholds must be non-negative")

    def as_dict(self) -> dict:
        return asdict(self)


def p4_harmful_probability(output: ProposalApplicabilityOutput) -> torch.Tensor:
    if output.class_logits is None:
        raise ValueError("P4 harmful probability requires class logits")
    return torch.softmax(output.class_logits, dim=1)[:, 0:1]


def veto_mask(
    raw_error_authorization: torch.Tensor,
    proposal_update: torch.Tensor,
    p4_output: ProposalApplicabilityOutput,
    policy: VetoPolicy,
) -> torch.Tensor:
    """Return a veto subset of the existing Raw Error authorization."""
    raw_authorized = raw_error_authorization.bool()
    if proposal_update.shape != raw_authorized.shape:
        raise ValueError("proposal update and authorization must have identical shapes")
    magnitude_veto = torch.zeros_like(raw_authorized)
    if policy.maximum_update_px is not None:
        magnitude_veto |= proposal_update.abs() > policy.maximum_update_px
    if policy.patch_mean_maximum_update_px is not None:
        local_mean = F.avg_pool2d(
            proposal_update.abs(), policy.patch_kernel, stride=1,
            padding=policy.patch_kernel // 2,
        )
        magnitude_veto |= local_mean > policy.patch_mean_maximum_update_px

    p4_masks: list[torch.Tensor] = []
    if policy.harmful_probability_threshold is not None:
        p4_masks.append(p4_harmful_probability(p4_output) >= policy.harmful_probability_threshold)
    if policy.predicted_utility_ceiling_px is not None:
        p4_masks.append(p4_output.utility <= policy.predicted_utility_ceiling_px)
    if policy.uncertainty_floor_px is not None:
        p4_masks.append(p4_output.sigma >= policy.uncertainty_floor_px)
    if policy.require_harmful_class:
        if p4_output.class_logits is None:
            raise ValueError("harmful-class veto requires P4 class logits")
        p4_masks.append(p4_output.class_logits.argmax(dim=1, keepdim=True) == 0)

    p4_veto = torch.zeros_like(raw_authorized)
    if p4_masks:
        p4_veto = p4_masks[0]
        for signal in p4_masks[1:]:
            p4_veto = p4_veto | signal if policy.p4_logic == "any" else p4_veto & signal
        finite = torch.isfinite(p4_output.utility) & torch.isfinite(p4_output.sigma)
        if p4_output.class_logits is not None:
            finite &= torch.isfinite(p4_output.class_logits).all(dim=1, keepdim=True)
        p4_veto |= ~finite
    return raw_authorized & (magnitude_veto | p4_veto)


def cascade_authorization(
    raw_error_authorization: torch.Tensor,
    proposal_update: torch.Tensor,
    p4_output: ProposalApplicabilityOutput,
    policy: VetoPolicy,
) -> torch.Tensor:
    return raw_error_authorization.bool() & ~veto_mask(
        raw_error_authorization, proposal_update, p4_output, policy
    )


def apply_cascade(
    raw: torch.Tensor,
    proposal: torch.Tensor,
    authorization: torch.Tensor,
) -> torch.Tensor:
    """Rejected pixels are raw bit-exactly; accepted pixels are A2 exactly."""
    if raw.shape != proposal.shape or raw.shape != authorization.shape:
        raise ValueError("raw, proposal, and authorization must have identical shapes")
    return torch.where(authorization.bool(), proposal, raw)


__all__ = [
    "VetoPolicy", "apply_cascade", "cascade_authorization",
    "p4_harmful_probability", "veto_mask",
]
