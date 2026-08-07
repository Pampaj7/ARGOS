import pytest
import torch
from argos_freezed.memory_bank import RawAnchorBank
from conftest import raw, rgb


def test_positive_left_disparity_is_enforced():
    bank = RawAnchorBank(); valid = torch.ones_like(raw(), dtype=torch.bool)
    bank.append_raw(raw(1), valid, rgb(), frame_id="ok", frame_index=0)
    with pytest.raises(ValueError, match="positive-left"):
        bank.append_raw(raw(-1), valid, rgb(), frame_id="bad", frame_index=1)
