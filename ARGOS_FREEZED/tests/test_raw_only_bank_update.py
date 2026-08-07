import pytest
import torch
from argos_freezed.memory_bank import RawAnchorBank
from conftest import raw, rgb


def test_raw_only_append_and_provenance_rejection():
    bank = RawAnchorBank(); disparity = raw(3); valid = torch.ones_like(disparity, dtype=torch.bool)
    bank.append_raw(disparity, valid, rgb(), frame_id="raw", frame_index=0)
    disparity.fill_(9)
    assert torch.all(bank.anchor(1, 1).disparity == 3)
    with pytest.raises(ValueError, match="independently generated raw"):
        bank.append_raw(raw(), valid, rgb(), frame_id="fused", frame_index=1, provenance="fused_output")
