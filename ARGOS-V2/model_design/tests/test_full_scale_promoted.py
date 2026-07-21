"""Regression tests for the architecture-frozen ARGOS v2 scale protocol."""
from __future__ import annotations

import csv
import hashlib
import inspect
import json
from pathlib import Path

import torch

from model_design.models.learned_t1_refiner import LearnedT1Refiner
from model_design.models.raw_error_detector import RawErrorDetector
from scripts.run_raw_error_abstention import load_a2
from scripts.summarize_full_scale_promoted import aggregate_frames


ROOT = Path(__file__).resolve().parents[2]
FROZEN_SOURCE_HASHES = {
    "model_design/models/learned_t1_refiner.py": "6b0b8de0616506058e889c05c5a35af7ec40cc6464fc3ae38357f19ad6dc6bde",
    "model_design/models/raw_error_detector.py": "fd0fc03be3b3b02d61d047ba415e09e3e09aa1601f043ef6ce004b9cc1b9829f",
    "model_design/losses/safety_losses.py": "9954a830cb54f6c6f9ec1fbee96c2b39222f2de527b15b5a60a24b37bdaa1a4a",
    "model_design/losses/raw_error_losses.py": "ef4fc3409a3effc9f7f6345344b6ab779d22f2ec2a7956a4bfc42152cecefa3c",
    "model_design/models/abstention.py": "ff8af2cbf28a9b85cd83c5d449473f52734c531a5710993408fd3b01a1f1a9f6",
    "model_design/external_components/bidavideo.py": "133a13f8a4dd89065f736484f1dba1811b40e0f1272d0bbec87d74074bf5c530",
    "model_design/data/temporal_pair_dataset.py": "a3eca9b4dd78034033a9d1d7545d23df330643bbe2cc84ac357d75b43f44b99d",
    "model_design/data/raw_error_dataset.py": "9592afcbb68701253c1d3e4d8baf8c6dad68d234fd4483c3b7f88cc4a179963a",
}


def test_scientific_sources_are_byte_identical_to_promoted_pipeline() -> None:
    actual = {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in FROZEN_SOURCE_HASHES
    }
    assert actual == FROZEN_SOURCE_HASHES


def test_promoted_architectures_and_parameter_counts_are_unchanged() -> None:
    a2 = LearnedT1Refiner("A2", tau_px=3.0)
    detector = RawErrorDetector("s1", channels=24)
    assert sum(parameter.numel() for parameter in a2.parameters()) == 39299
    assert sum(parameter.numel() for parameter in detector.parameters()) == 1107


def test_explicit_a2_checkpoint_loader_is_frozen(tmp_path: Path) -> None:
    original = ROOT / "results/learned_t1_refiner/ablations/A2/checkpoints/best_validation.pt"
    payload = torch.load(original, map_location="cpu", weights_only=False)
    checkpoint = tmp_path / "a2.pt"
    torch.save(payload, checkpoint)
    model = load_a2(torch.device("cpu"), checkpoint)
    assert model.variant == "A2" and not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_aggregator_uses_identical_primary_mask_counts(tmp_path: Path) -> None:
    path = tmp_path / "frame_metrics.csv"
    fields = (
        "coverage_threshold", "method", "valid_count", "clean_count", "changed_count",
        "helpful_count", "false_update_count", "clean_degradation_count", "epe", "raw_epe",
        "bad3", "boundary_epe", "refined_minus_raw_epe",
    )
    rows = [
        dict(zip(fields, (.5, "authorized_balanced", 10, 8, 4, 3, 2, 1, .8, 1., .1, 2., -.2))),
        dict(zip(fields, (.5, "authorized_balanced", 20, 12, 6, 4, 3, 2, .9, 1., .2, 3., .1))),
        dict(zip(fields, (.25, "authorized_balanced", 999, 999, 999, 999, 999, 999, 99., 99., 1., 99., 99.))),
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    (tmp_path / "runtime_summary.json").write_text(json.dumps({
        "wall_ms_per_frame": 4.0, "peak_gpu_memory_mb": 5.0,
    }))
    result = aggregate_frames(path, dataset="seen", seed=0)
    assert result["valid_count"] == 30
    assert result["epe"] == (10 * .8 + 20 * .9) / 30
    assert result["false_update_rate"] == 5 / 20
    assert result["clean_degradation"] == 3 / 20
    assert result["intervention_coverage"] == 10 / 30
    assert result["intervention_precision"] == 7 / 10


def test_scale_runner_adds_no_architecture_or_training_objective() -> None:
    source = (ROOT / "scripts/summarize_full_scale_promoted.py").read_text().lower()
    for forbidden in ("dinov3", "mamba", "convgru", "ppmstereo", "support_guard", "proposal_applicability"):
        assert forbidden not in source
    assert "optimizer" not in source and ".backward(" not in source
    assert "serv-ct" not in source and "d4d" not in source and "stereomis" not in source


def test_unseen_choice_is_evaluation_plumbing_only() -> None:
    source = inspect.getsource(__import__("scripts.run_raw_error_abstention", fromlist=["*"]))
    assert 'choices=(PRIMARY_UNSEEN_BACKBONE, "CREStereo")' in source
    assert "seen GO is required before unseen evaluation" in source
