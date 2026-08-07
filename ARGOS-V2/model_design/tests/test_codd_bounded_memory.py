from model_design.models.codd_bounded_memory import (
    BoundedMemoryPolicy,
    ResetEvidence,
    advance_state_age,
)


def evidence(**overrides):
    values = dict(age=3, accumulated_update=.2, disagreement=.3, warp_support=.9,
                  fb_confidence=.8, temporal_activation=.4, update_magnitude=.1)
    values.update(overrides)
    return ResetEvidence(**values)


def test_fixed_horizon_resets_before_next_recurrent_candidate():
    policy = BoundedMemoryPolicy("h4", max_age=4)
    assert not policy.pre_reset(age=3, accumulated_update=10)
    assert policy.pre_reset(age=4, accumulated_update=0)
    assert advance_state_age(4, reset=True) == 1


def test_h1_is_no_recurrent_fused_state():
    policy = BoundedMemoryPolicy("h1", max_age=1)
    assert policy.pre_reset(age=1, accumulated_update=0)


def test_cumulative_and_evidence_thresholds_are_strict_and_deterministic():
    policy = BoundedMemoryPolicy("adaptive", accumulated_update_max=.5,
        disagreement_max=.8, warp_support_min=.5, fb_confidence_min=.4,
        temporal_activation_max=.9, update_magnitude_max=.7)
    assert not policy.pre_reset(age=2, accumulated_update=.5)
    assert policy.pre_reset(age=2, accumulated_update=.5001)
    assert not policy.evidence_reset(evidence())
    assert policy.evidence_reset(evidence(warp_support=.49))
    assert policy.evidence_reset(evidence(disagreement=.81))


def test_policy_round_trip_contains_no_dataset_or_backbone_identity():
    policy = BoundedMemoryPolicy("hybrid", max_age=6, disagreement_max=1.2)
    restored = BoundedMemoryPolicy.from_dict(policy.to_dict())
    assert restored == policy
    assert "dataset" not in policy.to_dict() and "backbone" not in policy.to_dict()
