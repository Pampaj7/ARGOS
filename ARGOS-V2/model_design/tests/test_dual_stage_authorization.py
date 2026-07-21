from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest
import torch

from model_design.models.dual_stage_authorization import (
    VetoPolicy,
    apply_cascade,
    cascade_authorization,
    p4_harmful_probability,
    veto_mask,
)
from model_design.models.proposal_applicability_detector import ProposalApplicabilityOutput


ROOT = Path(__file__).resolve().parents[2]


def output(
    utility: torch.Tensor | None = None,
    sigma: torch.Tensor | None = None,
    logits: torch.Tensor | None = None,
) -> ProposalApplicabilityOutput:
    utility = torch.tensor([[[[-.2, .1, -.3, .4]]]]) if utility is None else utility
    sigma = torch.tensor([[[[.1, .2, .3, .4]]]]) if sigma is None else sigma
    if logits is None:
        logits = torch.tensor([[[[3., 0., 3., 0.]], [[0., 3., 0., 1.]], [[0., 0., 0., 3.]]]])
    return ProposalApplicabilityOutput(utility, sigma, logits, utility, sigma)


def test_c0_is_exactly_the_existing_authorization() -> None:
    raw_auth = torch.tensor([[[[True, False, True, False]]]])
    update = torch.tensor([[[[.1, .2, .3, .4]]]])
    actual = cascade_authorization(raw_auth, update, output(), VetoPolicy("C0"))
    assert torch.equal(actual, raw_auth)


def test_veto_can_only_close_and_never_open_authorization() -> None:
    raw_auth = torch.tensor([[[[True, False, True, False]]]])
    update = torch.tensor([[[[.1, .2, .3, .4]]]])
    policy = VetoPolicy("C2", harmful_probability_threshold=.5)
    veto = veto_mask(raw_auth, update, output(), policy)
    final = cascade_authorization(raw_auth, update, output(), policy)
    assert not (veto & ~raw_auth).any()
    assert not (final & ~raw_auth).any()
    assert torch.equal(final, raw_auth & ~veto)


def test_magnitude_veto_matches_manual_threshold() -> None:
    raw_auth = torch.tensor([[[[True, True, False, True]]]])
    update = torch.tensor([[[[.1, -.6, .8, .5]]]])
    policy = VetoPolicy("C1", maximum_update_px=.5)
    expected_veto = raw_auth & (update.abs() > .5)
    assert torch.equal(veto_mask(raw_auth, update, output(), policy), expected_veto)


def test_patch_mean_veto_matches_manual_pooling() -> None:
    raw_auth = torch.ones(1, 1, 3, 3, dtype=torch.bool)
    update = torch.zeros(1, 1, 3, 3); update[..., 1, 1] = .9
    policy = VetoPolicy("C1_patch", patch_mean_maximum_update_px=.09, patch_kernel=3)
    expected = torch.nn.functional.avg_pool2d(update.abs(), 3, 1, 1) > .09
    assert torch.equal(veto_mask(raw_auth, update, output(
        utility=torch.zeros_like(update), sigma=torch.ones_like(update),
        logits=torch.zeros(1, 3, 3, 3),
    ), policy), expected)


def test_p4_veto_matches_manual_probability_utility_and_uncertainty() -> None:
    raw_auth = torch.ones(1, 1, 1, 4, dtype=torch.bool)
    update = torch.zeros_like(raw_auth, dtype=torch.float32)
    p4 = output()
    harmful = p4_harmful_probability(p4) >= .5
    utility = p4.utility <= -.1
    uncertainty = p4.sigma >= .35
    for policy, expected in (
        (VetoPolicy("harm", harmful_probability_threshold=.5), harmful),
        (VetoPolicy("utility", predicted_utility_ceiling_px=-.1), utility),
        (VetoPolicy("sigma", uncertainty_floor_px=.35), uncertainty),
        (VetoPolicy("joint", harmful_probability_threshold=.5,
                    predicted_utility_ceiling_px=-.1, p4_logic="all"), harmful & utility),
    ):
        assert torch.equal(veto_mask(raw_auth, update, p4, policy), expected)


def test_combined_veto_is_logical_or() -> None:
    raw_auth = torch.ones(1, 1, 1, 4, dtype=torch.bool)
    update = torch.tensor([[[[.1, .8, .1, .1]]]])
    policy = VetoPolicy("C3", maximum_update_px=.5, harmful_probability_threshold=.5)
    expected = (update.abs() > .5) | (p4_harmful_probability(output()) >= .5)
    assert torch.equal(veto_mask(raw_auth, update, output(), policy), expected)


def test_rejection_is_raw_exact_and_acceptance_is_a2_exact() -> None:
    raw = torch.tensor([[[[1., 2., 3., 4.]]]])
    proposal = torch.tensor([[[[1.1, 1.8, 3.2, 3.9]]]])
    authorization = torch.tensor([[[[False, True, False, True]]]])
    result = apply_cascade(raw, proposal, authorization)
    assert torch.equal(result[..., (0, 2)], raw[..., (0, 2)])
    assert torch.equal(result[..., (1, 3)], proposal[..., (1, 3)])


def test_update_bound_and_finite_invalid_p4_fail_closed() -> None:
    raw_auth = torch.ones(1, 1, 1, 4, dtype=torch.bool)
    update = torch.tensor([[[[.1, .2, .3, 3.1]]]])
    p4 = output(); p4.utility[..., 1] = float("nan")
    policy = VetoPolicy("C3", maximum_update_px=3.0, harmful_probability_threshold=.99)
    final = cascade_authorization(raw_auth, update, p4, policy)
    assert not final[..., 1].item() and not final[..., 3].item()
    assert torch.isfinite(apply_cascade(torch.zeros_like(update), update.nan_to_num(), final)).all()


def test_policy_is_deterministic_and_manual_logic_matches() -> None:
    raw_auth = torch.rand(2, 1, 4, 5) > .3
    update = torch.randn(2, 1, 4, 5)
    p4 = output(
        utility=torch.randn_like(update), sigma=torch.rand_like(update),
        logits=torch.randn(2, 3, 4, 5),
    )
    policy = VetoPolicy("C2", harmful_probability_threshold=.5,
                        predicted_utility_ceiling_px=0, p4_logic="all")
    first = cascade_authorization(raw_auth, update, p4, policy)
    second = cascade_authorization(raw_auth, update, p4, policy)
    manual = raw_auth & ~((p4_harmful_probability(p4) >= .5) & (p4.utility <= 0))
    assert torch.equal(first, second) and torch.equal(first, manual)


def test_frozen_artifact_hashes() -> None:
    paths = {
        "a2": ROOT / "results/learned_t1_refiner/ablations/A2/checkpoints/best_validation.pt",
        "raw": ROOT / "results/raw_error_abstention/full/checkpoints/best_validation.pt",
        "p4": ROOT / "results/proposal_applicability/P4/checkpoints/best_validation.pt",
    }
    expected = {
        "a2": "6cd29277397001333ef3ce630b2f3bc04ec393cdc72e65aa5eb087afd3b389ea",
        "raw": "78b1bb6cf809dc76448222e41e3bcfafb754bc9b7b6629edcdfa2e1a33444e67",
        "p4": "c4d7d732b44ede1bb831b7789d6791907412b99345105f998c49c1cecde5bd2b",
    }
    assert {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()} == expected


def test_p4_loader_returns_only_frozen_eval_parameters() -> None:
    from scripts.run_dual_stage_authorization import load_p4

    p4 = load_p4(torch.device("cpu"))
    assert not p4.training
    assert all(not parameter.requires_grad for parameter in p4.parameters())
    first = p4.state_dict()
    second = load_p4(torch.device("cpu")).state_dict()
    assert first.keys() == second.keys()
    assert all(torch.equal(first[key], second[key]) for key in first)


def test_policy_serialization_roundtrip_is_resume_stable() -> None:
    from scripts.run_dual_stage_authorization import policy_from_dict

    policy = VetoPolicy(
        "frozen", maximum_update_px=2.0, predicted_utility_ceiling_px=0.0,
        harmful_probability_threshold=.5, p4_logic="all",
    )
    payload = json.loads(json.dumps(policy.as_dict()))
    assert policy_from_dict(payload) == policy


def test_threshold_choice_is_deterministic() -> None:
    from scripts.run_dual_stage_authorization import choose

    rows = [
        {"policy": "low", "gain_retained": .85, "false_update_rate": .01,
         "clean_degradation": .005, "intervention_precision": .82,
         "intervention_coverage": .01, "epe_gain": .02},
        {"policy": "high", "gain_retained": .90, "false_update_rate": .01,
         "clean_degradation": .005, "intervention_precision": .83,
         "intervention_coverage": .01, "epe_gain": .03},
    ]
    assert choose(rows) == choose(list(reversed(rows)))
    assert choose(rows)["policy"] == "high"


def test_runner_enforces_split_and_promotion_barriers_before_unseen() -> None:
    from model_design.data.raw_error_dataset import CALIBRATION_SEQUENCES, TEST_SEQUENCES

    assert set(CALIBRATION_SEQUENCES).isdisjoint(TEST_SEQUENCES)
    source = (ROOT / "scripts/run_dual_stage_authorization.py").read_text()
    assert 'get("promotion_passed")' in source
    assert 'raise RuntimeError("unseen backbones are blocked until final seen promotion")' in source
    assert 'if (args.output / "unseen_complete.json").exists()' in source


def test_runner_has_no_training_ood_or_dense_cache_path() -> None:
    source = (ROOT / "scripts/run_dual_stage_authorization.py")
    if not source.exists():
        pytest.skip("runner added after policy unit tests")
    text = source.read_text()
    assert "optimizer" not in text and ".backward(" not in text
    assert "SERV-CT" not in text and "D4D" not in text and "StereoMIS" not in text
    assert "np.save(" not in text


def test_policy_has_no_trainable_module_or_backbone_identity() -> None:
    source = inspect.getsource(__import__(
        "model_design.models.dual_stage_authorization", fromlist=["*"]
    ))
    assert "nn.Module" not in source
    assert "backbone" not in source
