from argos_freezed.memory_bank import RawAnchorBank
from argos_freezed.models.raw_multi_anchor_refiner import RawMultiAnchorRefiner


def test_no_fused_writeback_api():
    forbidden = ("add_fused", "update_with_output", "write_prediction")
    assert not any(hasattr(RawAnchorBank, name) for name in forbidden)
    assert not any("recurrent" in name or "hidden" in name or "state" == name for name in vars(RawMultiAnchorRefiner()))
