from __future__ import annotations

import torch

from model_design.losses.raw_multi_anchor_losses import (
    MultiAnchorLossConfig, multi_anchor_targets, raw_multi_anchor_losses,
)
from model_design.models.raw_multi_anchor_refiner import (
    FEATURE_CHANNELS, MultiAnchorEvidence, RawMultiAnchorRefiner,
    build_candidate_features, retrieve_and_fuse,
)


def evidence() -> MultiAnchorEvidence:
    raw = torch.ones(2, 1, 8, 9)
    candidates = torch.stack([raw[:, 0] + value for value in (.1, -.2, .3, -.4)], dim=1)
    valid = torch.ones_like(candidates, dtype=torch.bool)
    return MultiAnchorEvidence(
        raw=raw, candidates=candidates, candidate_valid=valid,
        warp_support=valid.clone(), fb_confidence=torch.full_like(candidates, .8),
        ages=torch.tensor([1, 2, 4, 8]), provenance=torch.zeros(4),
    )


def test_feature_contract_and_no_backbone_identity() -> None:
    item = evidence(); features = build_candidate_features(item)
    assert features.shape == (2, 4, FEATURE_CHANNELS, 8, 9)
    assert torch.isfinite(features).all()
    assert "backbone" not in MultiAnchorEvidence.__dataclass_fields__


def test_shared_candidate_encoder_and_shapes() -> None:
    model = RawMultiAnchorRefiner(channels=16, blocks=1)
    output = model(evidence())
    assert output.selection_score.shape == (2, 4, 8, 9)
    assert 0 < output.utility_probability.min() < output.utility_probability.max() + 1e-6 < 1
    assert 0 < output.fusion_weight.min() < output.fusion_weight.max() + 1e-6 < 1
    assert not any(name.startswith("candidate_") for name, _ in model.named_modules())


def test_invalid_candidate_never_selected() -> None:
    item = evidence()
    invalid = item.candidate_valid.clone(); invalid[:, 0] = False
    item = MultiAnchorEvidence(**{**item.__dict__, "candidate_valid": invalid})
    output = RawMultiAnchorRefiner(channels=16, blocks=1)(item)
    assert torch.isneginf(output.selection_score[:, 0]).all()


def test_exact_raw_fallback_and_hard_endpoint() -> None:
    item = evidence(); model = RawMultiAnchorRefiner(channels=16, blocks=1)
    output = model(item)
    fallback, accepted, _, _ = retrieve_and_fuse(
        item.raw, item, output, probability_threshold=1.1, utility_threshold_px=99, hard=False,
    )
    assert not accepted.any()
    assert torch.equal(fallback, item.raw)
    hard, accepted, chosen, _ = retrieve_and_fuse(
        item.raw, item, output, probability_threshold=0, utility_threshold_px=-99, hard=True,
    )
    selected = torch.gather(item.candidates, 1, chosen)
    assert accepted.all() and torch.equal(hard, selected)


def test_targets_and_losses_exclude_invalid() -> None:
    item = evidence(); gt = item.raw + .1
    coverage = torch.ones_like(item.raw); raw_valid = torch.ones_like(item.raw, dtype=torch.bool)
    targets = multi_anchor_targets(item.raw, item.candidates, gt, coverage, raw_valid, item, margin_px=.05)
    assert targets.helpful[:, 0].all()
    model = RawMultiAnchorRefiner(channels=16, blocks=1); output = model(item)
    losses = raw_multi_anchor_losses(output, item, targets, MultiAnchorLossConfig(), enable_fusion=True)
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_non_recurrent_contract_and_ttl_provenance() -> None:
    fields = set(MultiAnchorEvidence.__dataclass_fields__)
    assert "state" not in fields and "previous_output" not in fields
    item = evidence()
    assert item.ages.max().item() == 8
    assert item.provenance.eq(0).all()
