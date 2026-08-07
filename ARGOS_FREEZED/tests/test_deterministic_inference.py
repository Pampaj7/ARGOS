import torch
from argos_freezed.models.raw_multi_anchor_refiner import MultiAnchorEvidence, RawMultiAnchorRefiner


def test_model_inference_is_deterministic():
    torch.manual_seed(8); raw = torch.rand(1, 1, 8, 9) + 1; candidate = torch.rand(1, 4, 8, 9) + 1
    mask = torch.ones_like(candidate, dtype=torch.bool)
    evidence = MultiAnchorEvidence(raw, candidate, mask, mask, torch.rand_like(candidate), torch.tensor([1, 2, 4, 8]), torch.zeros(4))
    model = RawMultiAnchorRefiner().eval()
    first = model(evidence); second = model(evidence)
    assert all(torch.equal(getattr(first, name), getattr(second, name)) for name in first.__dataclass_fields__)
