from __future__ import annotations

import torch

from model_design.models.raw_multi_anchor_refiner import MultiAnchorEvidence, MultiAnchorOutput
from model_design.models.raw_multi_anchor_selective_gate import (
    GATE_FEATURE_CHANNELS, LinearHarmGate, SelectiveRiskGate, apply_veto,
    build_gate_features, frozen_proposal,
)


def example():
    raw=torch.tensor([[[[1.,2.],[3.,4.]]]])
    candidates=torch.stack((raw[:,0]+1,raw[:,0]-1,raw[:,0]+.5,raw[:,0]-.5),dim=1)
    evidence=MultiAnchorEvidence(raw,candidates,torch.ones_like(candidates,dtype=torch.bool),torch.ones_like(candidates,dtype=torch.bool),torch.ones_like(candidates),torch.tensor([1,2,4,8]),torch.zeros(4))
    score=torch.tensor([[[[.5,.5],[.5,.5]],[[.2,.2],[.2,.2]],[[.1,.1],[.1,.1]],[[.0,.0],[.0,.0]]]])
    output=MultiAnchorOutput(torch.zeros_like(score),torch.full_like(score,.95),torch.ones_like(score),torch.full_like(score,.25),score)
    return evidence,output


def test_frozen_candidate_and_weight_are_unchanged():
    evidence,output=example(); proposal=frozen_proposal(evidence,output,probability_threshold=.9,utility_threshold_px=.1)
    assert torch.equal(proposal.chosen,torch.zeros_like(proposal.chosen))
    assert torch.equal(proposal.candidate,evidence.candidates[:,0:1])
    assert torch.equal(proposal.fusion_weight,output.fusion_weight[:,0:1])
    assert torch.equal(proposal.proposed,evidence.raw+.25*(proposal.candidate-evidence.raw))


def test_feature_contract_is_finite_and_contains_no_target():
    evidence,output=example(); proposal=frozen_proposal(evidence,output,probability_threshold=.9,utility_threshold_px=.1)
    features=build_gate_features(evidence,proposal)
    assert features.shape==(1,GATE_FEATURE_CHANNELS,2,2)
    assert torch.isfinite(features).all()


def test_veto_is_close_only_and_raw_fallback_is_bit_exact():
    evidence,output=example(); proposal=frozen_proposal(evidence,output,probability_threshold=.9,utility_threshold_px=.1)
    gate=torch.tensor([[[[True,False],[False,True]]]])
    prediction,accepted=apply_veto(evidence.raw,proposal,gate)
    assert torch.equal(accepted,proposal.eligible&gate)
    assert torch.equal(prediction[~accepted],evidence.raw[~accepted])
    assert torch.equal(prediction[accepted],proposal.proposed[accepted])
    closed=proposal.eligible&~gate
    assert not accepted[closed].any()


def test_invalid_candidate_is_never_eligible():
    evidence,output=example(); invalid=torch.zeros_like(evidence.candidate_valid)
    evidence=MultiAnchorEvidence(evidence.raw,evidence.candidates,invalid,evidence.warp_support,evidence.fb_confidence,evidence.ages,evidence.provenance)
    proposal=frozen_proposal(evidence,output,probability_threshold=.9,utility_threshold_px=.1)
    prediction,accepted=apply_veto(evidence.raw,proposal,torch.ones_like(proposal.eligible))
    assert not accepted.any()
    assert torch.equal(prediction,evidence.raw)


def test_gate_sizes_and_positive_probability():
    features=torch.randn(2,GATE_FEATURE_CHANNELS,8,9)
    for model in (LinearHarmGate(),SelectiveRiskGate(hidden=32,depth=3),SelectiveRiskGate(hidden=64,depth=3,joint=True)):
        output=model(features)
        assert output.harm_probability.shape==(2,1,8,9)
        assert ((output.harm_probability>=0)&(output.harm_probability<=1)).all()
        assert sum(parameter.numel() for parameter in model.parameters())<100_000
    assert SelectiveRiskGate(joint=True)(features).predicted_gain is not None


def test_gate_cannot_write_anchor_bank():
    evidence,output=example(); original=evidence.candidates.clone(); proposal=frozen_proposal(evidence,output,probability_threshold=.9,utility_threshold_px=.1)
    apply_veto(evidence.raw,proposal,torch.ones_like(proposal.eligible))
    assert torch.equal(evidence.candidates,original)
