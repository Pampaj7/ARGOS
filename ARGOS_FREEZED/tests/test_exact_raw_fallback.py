import torch
from argos_freezed.models.raw_multi_anchor_refiner import MultiAnchorEvidence, RawMultiAnchorRefiner, retrieve_and_fuse


def test_fallback_is_bit_exact():
    torch.manual_seed(4); raw = torch.rand(1, 1, 7, 9) + 1
    unavailable = torch.zeros(1, 4, 7, 9, dtype=torch.bool)
    evidence = MultiAnchorEvidence(raw, torch.rand(1, 4, 7, 9) + 1, unavailable, unavailable, torch.zeros(1, 4, 7, 9), torch.tensor([1, 2, 4, 8]), torch.zeros(4))
    output = RawMultiAnchorRefiner()(evidence)
    result, accepted, _, weight = retrieve_and_fuse(raw, evidence, output, probability_threshold=.9, utility_threshold_px=.1, hard=False)
    assert torch.equal(result, raw) and not accepted.any() and torch.equal(weight, torch.zeros_like(weight))
