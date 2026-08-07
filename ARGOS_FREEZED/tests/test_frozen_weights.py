import torch

from argos_freezed.models.raw_multi_anchor_refiner import MultiAnchorEvidence
from argos_freezed.pipeline import FrozenArgosGeometryRefiner
from conftest import ZeroFlow, raw, rgb
from argos_freezed.memory_bank import RawAnchorBank


def test_exact_parameter_count_and_frozen_weights():
    refiner = FrozenArgosGeometryRefiner(device="cpu", flow_adapter=ZeroFlow())
    assert sum(p.numel() for p in refiner.model.parameters()) == 60739
    assert not any(p.requires_grad for p in refiner.model.parameters())
    before = {k: v.clone() for k, v in refiner.model.state_dict().items()}
    refiner.step(rgb(), rgb(), raw(), torch.ones_like(raw(), dtype=torch.bool), RawAnchorBank(), frame_id="0", frame_index=0)
    assert all(torch.equal(value, before[key]) for key, value in refiner.model.state_dict().items())
