"""Exact `infer` extraction from run_codd_style_bounded_memory_validation.py."""
from __future__ import annotations
from argos_freezed.alignment.bida_pull_warp import temporal_disparity_evidence
from .codd_style_fusion import build_codd_cues

def infer(model, extractor, item, state, forward, backward, *, include_learned: bool):
    evidence = temporal_disparity_evidence(
        item["raw"], state["disparity"], forward, backward,
        current_valid=item["raw_valid"], past_valid=state["valid"],
        current_rgb=item["current_rgb"], past_rgb=item["past_rgb"],
    )
    cues = build_codd_cues(
        extractor, raw=item["raw"], aligned_memory=evidence.aligned_past_disparity,
        current_rgb=item["current_rgb"], current_right_rgb=item["current_right_rgb"],
        past_rgb=item["past_rgb"], flow_current_to_past=forward,
        flow_magnitude=evidence.flow_magnitude,
        forward_backward_confidence=evidence.forward_backward_confidence,
        warp_support=evidence.warp_support, aligned_valid=evidence.aligned_validity,
        include_learned_stereo_evidence=include_learned,
    )
    return evidence, model(cues, item["raw"], evidence.aligned_past_disparity)
