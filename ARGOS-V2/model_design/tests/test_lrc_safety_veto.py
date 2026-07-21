import pytest
import torch

from model_design.models.lrc_safety_veto import frame_relative_lrc_gate, lrc_safety_veto


def test_frame_relative_lrc_gate_is_per_frame_and_invalid_safe():
    residual = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]], [[[10.0, 20.0, 30.0, 40.0]]]])
    valid = torch.tensor([[[[1, 1, 1, 0]]], [[[1, 1, 0, 1]]]], dtype=torch.bool)
    gate = frame_relative_lrc_gate(residual, valid, quantile=.5)
    assert torch.equal(gate[0, 0], torch.tensor([[False, True, True, False]]))
    assert torch.equal(gate[1, 0], torch.tensor([[False, True, False, True]]))


def test_veto_can_close_but_never_open():
    base = torch.tensor([[[[True, False, True, True]]]])
    residual = torch.tensor([[[[.1, .2, .3, .4]]]])
    valid = torch.ones_like(base)
    output = lrc_safety_veto(base, residual, valid, quantile=.75)
    assert torch.equal(output, torch.tensor([[[[False, False, False, True]]]]))
    assert not (output & ~base).any()


def test_rejects_invalid_contracts():
    value = torch.ones((1, 1, 2, 2))
    with pytest.raises(ValueError):
        frame_relative_lrc_gate(value, value.bool(), quantile=1.1)
    with pytest.raises(ValueError):
        frame_relative_lrc_gate(value[:, 0], value.bool(), quantile=.5)
