import inspect
from argos_freezed import pipeline


def test_pipeline_has_no_composition_operation():
    source = inspect.getsource(pipeline.FrozenArgosGeometryRefiner.step)
    assert "current_to_anchor" in source
    assert "previous_flow" not in source and "compose(" not in source
