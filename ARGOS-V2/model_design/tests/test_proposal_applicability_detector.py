from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest
import torch

from model_design.data.proposal_utility_dataset import proposal_utility_targets
from model_design.losses.proposal_utility_losses import (
    ProposalUtilityLossConfig,
    proposal_utility_losses,
)
from model_design.models.learned_t1_refiner import LearnedT1Refiner
from model_design.models.proposal_applicability_detector import (
    FEATURE_CHANNELS,
    RECEPTIVE_FIELDS,
    ProposalApplicabilityDetector,
    ProposalEvidence,
    apply_frozen_proposal,
    proposal_authorization_mask,
)


ROOT = Path(__file__).resolve().parents[2]
A2_CHECKPOINT = ROOT / "results/learned_t1_refiner/ablations/A2/checkpoints/best_validation.pt"


def evidence(batch: int = 2, height: int = 12, width: int = 16) -> ProposalEvidence:
    one = torch.ones(batch, 1, height, width)
    raw = torch.rand_like(one) * 10
    update = (torch.rand_like(one) - 0.5) * 2
    return ProposalEvidence(
        raw=raw,
        aligned=raw + torch.randn_like(raw),
        proposal=raw + update,
        update=update,
        a2_error_gate=torch.rand_like(one),
        a2_memory_gate=torch.rand_like(one),
        a2_delta=torch.randn_like(one),
        raw_valid=one.bool(),
        aligned_valid=one.bool(),
        warp_support=one.bool(),
        flow_magnitude=torch.rand_like(one) * 8,
        photometric_residual=torch.rand_like(one),
        forward_backward_error=torch.rand_like(one),
        forward_backward_confidence=torch.rand_like(one),
    )


def targets(item: ProposalEvidence, epsilon: float = 0.1):
    gt = item.raw + torch.randn_like(item.raw) * 0.3
    batch = {
        "raw": item.raw,
        "gt": gt,
        "gt_coverage": torch.ones_like(gt),
        "raw_valid": item.raw_valid,
    }
    return proposal_utility_targets(
        batch, item.proposal, aligned_valid=item.aligned_valid,
        warp_support=item.warp_support, epsilon_px=epsilon,
    )


@pytest.mark.parametrize("variant", ["P1", "P2", "P3", "P4"])
def test_outputs_are_finite_bounded_and_have_no_identity_or_gt_input(variant: str) -> None:
    item = evidence()
    model = ProposalApplicabilityDetector(variant, channels=24)
    output = model(item)
    assert model.normalized_inputs(item).shape[1] == FEATURE_CHANNELS
    assert output.utility.shape == item.raw.shape
    assert output.utility.abs().max() <= 3.0
    assert torch.isfinite(output.utility).all()
    assert torch.isfinite(output.sigma).all() and (output.sigma > 0).all()
    assert (output.class_logits is not None) == (variant == "P4")
    assert "backbone" not in ProposalEvidence.__dataclass_fields__
    assert "gt" not in ProposalEvidence.__dataclass_fields__
    assert sum(p.numel() for p in model.parameters()) < 100_000


def test_local_variants_have_real_spatial_receptive_field() -> None:
    assert RECEPTIVE_FIELDS["P1"] == 2
    assert all(RECEPTIVE_FIELDS[name] == 8 for name in ("P2", "P3", "P4"))
    model = ProposalApplicabilityDetector("P2", channels=24)
    with torch.no_grad():
        model.head_utility.weight.fill_(1.0)
    item = evidence(batch=1, height=9, width=9)
    item.raw.requires_grad_()
    model(item).utility[..., 4, 4].backward()
    support = item.raw.grad[0, 0].nonzero()
    assert (support[:, 0].min(), support[:, 0].max()) == (1, 8)
    assert (support[:, 1].min(), support[:, 1].max()) == (1, 8)


def test_gradients_enter_only_detector_not_frozen_evidence() -> None:
    item = evidence()
    model = ProposalApplicabilityDetector("P4", channels=24)
    output = model(item)
    losses = proposal_utility_losses(
        output, targets(item), ProposalUtilityLossConfig(
            heteroscedastic_weight=.25, classification_weight=.5,
            harmful_as_helpful_weight=.5,
        ),
    )
    losses["total"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert all(value.grad is None for value in item.__dict__.values())


def test_invalid_pixels_do_not_contribute_to_losses() -> None:
    item = evidence(batch=1)
    output = ProposalApplicabilityDetector("P4", channels=24)(item)
    target = targets(item)
    target.regression_valid[..., 0, 0] = False
    target.classification_valid[..., 0, 0] = False
    target.helpful[..., 0, 0] = False
    target.harmful[..., 0, 0] = False
    first = proposal_utility_losses(output, target, ProposalUtilityLossConfig(classification_weight=1))
    target.utility[..., 0, 0] = 1e6
    target.classes[..., 0, 0] = 2
    second = proposal_utility_losses(output, target, ProposalUtilityLossConfig(classification_weight=1))
    torch.testing.assert_close(first["total"], second["total"])


def test_abstention_is_raw_bit_exact_and_acceptance_is_a2_exact() -> None:
    raw = torch.tensor([[[[1.0, 2.0]]]], dtype=torch.float32)
    proposal = torch.tensor([[[[1.25, 1.5]]]], dtype=torch.float32)
    decision = torch.tensor([[[[False, True]]]])
    output = apply_frozen_proposal(raw, proposal, decision)
    assert torch.equal(output[..., 0], raw[..., 0])
    assert torch.equal(output[..., 1], proposal[..., 1])


def test_authorization_respects_uncertainty_support_and_bound() -> None:
    item = evidence(batch=1, height=2, width=4)
    model = ProposalApplicabilityDetector("P3", channels=24)
    with torch.no_grad():
        model.head_utility.bias.fill_(1.0)
        model.head_sigma.bias.fill_(-5.0)
        item.warp_support[..., 0] = False
        item.update[..., 1] = 3.1
    output = model(item)
    authorized = proposal_authorization_mask(
        output, item, utility_margin_px=.1, uncertainty_threshold_px=.5,
    )
    assert not authorized[..., 0].any()
    assert not authorized[..., 1].any()
    assert authorized[..., 2:].all()


def test_validated_a2_hash_and_freezing() -> None:
    assert hashlib.sha256(A2_CHECKPOINT.read_bytes()).hexdigest() == (
        "6cd29277397001333ef3ce630b2f3bc04ec393cdc72e65aa5eb087afd3b389ea"
    )
    payload = torch.load(A2_CHECKPOINT, map_location="cpu", weights_only=False)
    model = LearnedT1Refiner("A2", tau_px=float(payload.get("tau_px", 3.0)))
    model.load_state_dict(payload["model"], strict=True)
    model.eval().requires_grad_(False)
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_checkpoint_roundtrip_is_deterministic() -> None:
    torch.manual_seed(9)
    item = evidence(batch=1)
    model = ProposalApplicabilityDetector("P4", channels=24).eval()
    expected = model(item)
    buffer = io.BytesIO(); torch.save(model.state_dict(), buffer); buffer.seek(0)
    restored = ProposalApplicabilityDetector("P4", channels=24).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    actual = restored(item)
    torch.testing.assert_close(actual.utility, expected.utility, rtol=0, atol=0)
    torch.testing.assert_close(actual.sigma, expected.sigma, rtol=0, atol=0)
    torch.testing.assert_close(actual.class_logits, expected.class_logits, rtol=0, atol=0)


def test_runner_cannot_load_ood_and_writes_no_dense_cache() -> None:
    source = (ROOT / "scripts/run_proposal_applicability.py").read_text()
    assert "SERV-CT" not in source and "D4D" not in source and "StereoMIS" not in source
    assert "np.save(" not in source and "torch.save(proposal" not in source
