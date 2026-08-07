import torch
from argos_freezed.models.raw_multi_anchor_refiner import MultiAnchorEvidence, RawMultiAnchorRefiner


def test_invalid_anchor_scores_negative_infinity():
    raw = torch.ones(1, 1, 5, 6)
    evidence = MultiAnchorEvidence(raw, raw.expand(-1, 4, -1, -1), torch.zeros(1, 4, 5, 6, dtype=torch.bool),
        torch.ones(1, 4, 5, 6, dtype=torch.bool), torch.ones(1, 4, 5, 6), torch.tensor([1, 2, 4, 8]), torch.zeros(4))
    assert torch.isneginf(RawMultiAnchorRefiner()(evidence).selection_score).all()
