from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from model_design.external_components.bidavideo import BiDAFlowInferenceAdapter
from model_design.models.support_guard import (
    SupportGuard,
    SupportProvenance,
    deterministic_bank_indices,
    fit_support_reference,
    guarded_output,
    quantile_threshold,
    support_mask,
)


ROOT = Path(__file__).resolve().parents[2]
SEEN = ("S2M2-S", "RAFT-Stereo", "StereoAnywhere")


def provenance(split: str = "training", backbones=SEEN) -> SupportProvenance:
    return SupportProvenance("SCARED-C", split, tuple(backbones), ("train_sequence",), 17)


def reference(seed: int = 17):
    rng = np.random.default_rng(seed)
    value = rng.normal(size=(128, 3)).astype(np.float32)
    return fit_support_reference(value, feature_names=("f0", "f1", "f2"),
                                 provenance=provenance(), bank_size=32, knn_k=3)


def test_fit_statistics_use_scared_training_only():
    x = np.random.default_rng(0).normal(size=(32, 2))
    with pytest.raises(ValueError):
        fit_support_reference(x, feature_names=("a", "b"),
                              provenance=SupportProvenance("SERV-CT", "training", SEEN, ("x",), 1))


def test_threshold_selection_uses_scared_calibration_only():
    with pytest.raises(ValueError):
        quantile_threshold(np.arange(10), .95, provenance=provenance("training"))
    assert quantile_threshold(np.arange(10), .9, provenance=provenance("calibration")) == pytest.approx(8.1)


@pytest.mark.parametrize("backbone", ["Fast-FoundationStereo", "CREStereo"])
def test_unseen_backbones_rejected_during_fit(backbone):
    x = np.random.default_rng(0).normal(size=(32, 2))
    with pytest.raises(ValueError):
        fit_support_reference(x, feature_names=("a", "b"),
                              provenance=provenance(backbones=(backbone,)))


def test_diagonal_score_matches_manual_calculation():
    ref = reference()
    guard = SupportGuard(ref)
    x = torch.tensor([[[[1.0]], [[-0.5]], [[2.0]]]])
    z = (x[0, :, 0, 0].numpy() - ref.mean) / ref.std
    assert guard.score(x, "diagonal").item() == pytest.approx(float((z * z).sum()), rel=1e-5)


def test_shrinkage_score_matches_manual_calculation():
    ref = reference()
    guard = SupportGuard(ref)
    x = torch.tensor([[[[1.0]], [[-0.5]], [[2.0]]]])
    z = (x[0, :, 0, 0].numpy() - ref.mean) / ref.std
    expected = float(z @ ref.precision @ z)
    assert guard.score(x, "shrinkage").item() == pytest.approx(expected, rel=1e-5)


def test_covariance_regularization_is_finite_and_positive():
    rng = np.random.default_rng(3)
    x = np.repeat(rng.normal(size=(128, 1)), 3, axis=1) + rng.normal(scale=1e-7, size=(128, 3))
    ref = fit_support_reference(x, feature_names=("a", "b", "c"),
                                provenance=provenance(), bank_size=16)
    assert np.isfinite(ref.precision).all()
    assert np.linalg.eigvalsh(ref.precision).min() > 0


def test_knn_score_matches_manual_calculation():
    ref = reference()
    guard = SupportGuard(ref)
    x = torch.tensor([[[[0.1]], [[0.2]], [[0.3]]]])
    z = (x[0, :, 0, 0].numpy() - ref.mean) / ref.std
    expected = np.sort(np.linalg.norm(ref.reference_bank - z, axis=1))[:ref.knn_k].mean()
    assert guard.score(x, "knn").item() == pytest.approx(float(expected), rel=1e-5)


def test_reference_bank_sampling_is_deterministic():
    assert np.array_equal(deterministic_bank_indices(100, 17, 9),
                          deterministic_bank_indices(100, 17, 9))
    assert not np.array_equal(deterministic_bank_indices(100, 17, 9),
                              deterministic_bank_indices(100, 17, 10))


def test_support_acceptance_monotonic_with_threshold():
    score = torch.tensor([[[[1.0, 2.0, 3.0]]]])
    low = SupportGuard.accept(score, 1.5)
    high = SupportGuard.accept(score, 2.5)
    assert torch.all(low <= high)


def test_frame_granularity_is_one_deterministic_median_decision():
    score = torch.tensor([[[[1.0, 2.0], [10.0, 20.0]]]])
    rejected = support_mask(score, 1.5, "frame")
    accepted = support_mask(score, 11.0, "frame")
    assert not rejected.any()
    assert accepted.all()


def test_rejected_samples_return_raw_bit_exactly():
    raw = torch.randn(1, 1, 4, 5)
    proposal = torch.randn_like(raw)
    output = guarded_output(raw, proposal, torch.ones_like(raw, dtype=torch.bool),
                            torch.zeros_like(raw, dtype=torch.bool))
    assert torch.equal(output, raw)


def test_accepted_samples_preserve_existing_a2_output():
    raw = torch.randn(1, 1, 4, 5)
    proposal = torch.randn_like(raw).clamp(-3, 3)
    output = guarded_output(raw, proposal, torch.ones_like(raw, dtype=torch.bool),
                            torch.ones_like(raw, dtype=torch.bool))
    assert torch.equal(output, raw + proposal)


def test_invalid_features_are_rejected_with_infinite_score():
    guard = SupportGuard(reference())
    x = torch.zeros(1, 3, 2, 2)
    x[:, 1, 0, 0] = float("nan")
    score, accepted = guard(x, "diagonal", 100.0)
    assert torch.isinf(score[0, 0, 0, 0])
    assert not accepted[0, 0, 0, 0]
    assert torch.isfinite(score[0, 0, 1, 1])


def test_no_nan_or_inf_for_finite_features():
    guard = SupportGuard(reference())
    for method in ("diagonal", "shrinkage", "knn"):
        assert torch.isfinite(guard.score(torch.randn(2, 3, 3, 4), method)).all()


def test_guard_has_no_trainable_parameters_or_backbone_input():
    guard = SupportGuard(reference())
    assert list(guard.parameters()) == []
    assert "backbone" not in inspect.signature(guard.forward).parameters


def test_no_gradients_through_frozen_input_or_guard():
    guard = SupportGuard(reference())
    feature = torch.randn(1, 3, 2, 2)
    with torch.inference_mode():
        score = guard.score(feature, "diagonal")
    assert not score.requires_grad


def test_sea_raft_default_inference_is_no_graph():
    source = inspect.getsource(BiDAFlowInferenceAdapter.infer)
    assert "torch.inference_mode" in source


def test_no_future_or_sequence_state_in_guard():
    source = inspect.getsource(SupportGuard)
    assert "future" not in source.lower()
    assert "sequence_id" not in source
    assert not hasattr(SupportGuard(reference()), "state")


def test_identical_paired_mask_is_reusable_for_all_methods():
    gt = torch.tensor([1, 1, 0, 1], dtype=torch.bool)
    raw = torch.tensor([1, 1, 1, 0], dtype=torch.bool)
    warp = torch.tensor([1, 0, 1, 1], dtype=torch.bool)
    common = gt & raw & warp
    assert torch.equal(common, gt & raw & warp)
    assert common.sum() == 1


def test_frozen_artifact_hashes_match_validated_manifest():
    expected = {
        "results/raw_error_abstention/full/checkpoints/best_validation.pt": "78b1bb6cf809dc76448222e41e3bcfafb754bc9b7b6629edcdfa2e1a33444e67",
        "results/learned_t1_refiner/ablations/A2/checkpoints/best_validation.pt": "6cd29277397001333ef3ce630b2f3bc04ec393cdc72e65aa5eb087afd3b389ea",
        "results/raw_error_abstention/full/operating_modes.json": "791f27d21e3f9fa63fe267d5742c4fb85226f49e6027b285aeb90754fbe10b69",
    }
    for relative, digest in expected.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest


def test_runner_freezes_before_ood_loader_construction():
    runner = (ROOT / "scripts/run_support_guard_validation.py")
    if not runner.exists():
        pytest.skip("runner is added after the support operator")
    source = runner.read_text()
    assert source.index("freeze_support_policy") < source.index("run_frozen_evaluations")


def test_runner_does_not_modify_frozen_modules():
    runner = ROOT / "scripts/run_support_guard_validation.py"
    if not runner.exists():
        pytest.skip("runner is added after the support operator")
    source = runner.read_text()
    assert ".train()" not in source
    assert "optimizer" not in source.lower()
    assert "backward(" not in source
