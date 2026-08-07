import inspect
from argos_freezed.pipeline import FrozenArgosGeometryRefiner


def test_public_api_has_no_backbone_identity_or_internal_features():
    parameters = inspect.signature(FrozenArgosGeometryRefiner.step).parameters
    assert not any("backbone" in name or "cost_volume" in name or "feature" in name or "hidden" in name for name in parameters)
