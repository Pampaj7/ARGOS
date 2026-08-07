from argos_freezed.constants import ANCHOR_AGES
from argos_freezed.memory_bank import RawAnchorBank
from conftest import raw, rgb
import pytest
import torch


def test_only_past_indices_are_addressable():
    bank = RawAnchorBank()
    for index in range(9):
        bank.append_raw(raw(index + 1), torch.ones_like(raw(), dtype=torch.bool), rgb(index), frame_id=str(index), frame_index=index)
    assert [bank.anchor(9, age).frame_index for age in ANCHOR_AGES] == [8, 7, 5, 1]
    assert all(bank.anchor(9, age).frame_index < 9 for age in ANCHOR_AGES)
    with pytest.raises(ValueError, match="causal order"):
        bank.append_raw(raw(), torch.ones_like(raw(), dtype=torch.bool), rgb(), frame_id="late", frame_index=7)
