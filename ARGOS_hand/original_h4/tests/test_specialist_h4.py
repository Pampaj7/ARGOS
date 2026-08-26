"""The specialist evaluator must read the specialist's weights, not A2's.

SpecialistH4 reaches its inference path by constructing AblationH4 at variant A2
first and then redirecting the checkpoint. That order is what makes the head
class and cue builder correct by inheritance, and it is also what could leave
A2's checkpoint or a cached model in place -- scoring the shipped head under a
specialist's name and reporting a difference of exactly zero.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BACKBONE = "RAFT-Stereo"
RUN = ROOT / f"model_design/training_runs/specialist_{BACKBONE}"


def _module():
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from model_design.comparison import specialist_h4
    return specialist_h4


def test_undeclared_backbone_refused():
    with pytest.raises(ValueError):
        _module().SpecialistH4(backbone="ResNet-18", device="cpu")


@pytest.mark.skipif(not (RUN / "checkpoints/best_validation.pt").is_file(),
                    reason="specialist not trained yet")
def test_checkpoint_is_the_specialist_not_a2():
    specialist_h4 = _module()
    model = specialist_h4.SpecialistH4(backbone=BACKBONE, device="cpu")
    assert model.checkpoint == RUN / "checkpoints/best_validation.pt"
    assert "ablation_A2" not in str(model.checkpoint)
    assert model._model is None, "a model cached from A2 would be scored instead"

    described = model.describe()
    assert described["backbone"] == BACKBONE
    assert described["training_provenance"]["deviation"]["backbones"]["to"] == [BACKBONE]
    assert described["training_provenance"]["deviation"]["learned_stereo_evidence"] is False
    # The reported hash must be the file it will actually load.
    import hashlib
    digest = hashlib.sha256(model.checkpoint.read_bytes()).hexdigest()
    assert described["checkpoint_sha256"] == digest


@pytest.mark.skipif(not (RUN / "specialist_provenance.json").is_file(),
                    reason="specialist not started yet")
def test_provenance_names_one_backbone():
    record = json.loads((RUN / "specialist_provenance.json").read_text())
    assert record["backbone"] == BACKBONE
    assert record["deviation"]["backbones"]["to"] == [BACKBONE]
    assert record["deviation"]["epochs"]["used"] == 36
    audit = json.loads((RUN / "split_audit.json").read_text())
    assert audit["backbones"] == [BACKBONE], "the run recorded a different training set"
    assert audit["learned_stereo_evidence"] is False
