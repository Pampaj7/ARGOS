"""The specialist patch must reach both places the runner reads the backbone list.

If it reaches only make_dataset, the run trains on one backbone and writes a
split_audit.json claiming three -- the same class of silent provenance lie that
the shared SHIPPED_BASE_VARIANTS tuple was introduced to prevent.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_patch_reaches_datasets_and_manifest():
    specialist = _load("train_specialist", ROOT / "model_design/train_specialist.py")
    train = _load("canonical_h4_train", ROOT / "model_design/train.py")
    runner = train.load_runner()

    assert tuple(runner.SEEN_BACKBONES) == ("S2M2-S", "RAFT-Stereo", "StereoAnywhere")
    patch = specialist.install(runner, "CREStereo")
    assert patch["to"] == ["CREStereo"]

    config = argparse.Namespace(
        clip_length=4, memory_state="recurrent", disable_learned_stereo_evidence=True,
        coverage_threshold=0.5, tau_reset_native_px=5.0, tau_fusion_native_px=1.0,
    )
    assert runner.manifest(config)["backbones"] == ["CREStereo"]

    # make_dataset builds a TemporalClipDataset over the module global, so the
    # records it enumerates carry exactly one backbone name.
    dataset = runner.make_dataset(("dataset_1_keyframe_2",), argparse.Namespace(
        clip_length=4, coverage_threshold=0.5, max_clips_per_sequence=1, seed=0))
    assert {record.backbone for record in dataset.records} == {"CREStereo"}


def test_undeclared_backbone_refused():
    specialist = _load("train_specialist", ROOT / "model_design/train_specialist.py")
    train = _load("canonical_h4_train", ROOT / "model_design/train.py")
    runner = train.load_runner()
    try:
        specialist.install(runner, "ResNet-18")
    except SystemExit:
        return
    raise AssertionError("install accepted a backbone with no cache")


if __name__ == "__main__":
    test_patch_reaches_datasets_and_manifest()
    test_undeclared_backbone_refused()
    print("ok")
