import torch
from argos_freezed.memory_bank import RawAnchorBank
from argos_freezed.pipeline import FrozenArgosGeometryRefiner
from conftest import ZeroFlow, raw, rgb


def test_each_anchor_gets_one_direct_pairwise_flow():
    bank = RawAnchorBank()
    for index in (0, 4, 6, 7):
        bank.append_raw(raw(index + 1), torch.ones_like(raw(), dtype=torch.bool), rgb(index), frame_id=str(index), frame_index=index)
    flow = ZeroFlow(); refiner = FrozenArgosGeometryRefiner(device="cpu", flow_adapter=flow)
    refiner.step(rgb(100), rgb(101), raw(10), torch.ones_like(raw(), dtype=torch.bool), bank, frame_id="8", frame_index=8)
    assert [name for name, *_ in flow.calls] == [item for _ in range(4) for item in ("current_to_anchor", "anchor_to_current")]
    assert [call[2] for call in flow.calls[::2]] == [7.0, 6.0, 4.0, 0.0]
