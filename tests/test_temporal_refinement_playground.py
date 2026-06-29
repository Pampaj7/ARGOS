from __future__ import annotations

from pathlib import Path

import torch

from scripts.temporal_refinement.playground.config import load_experiment_config
from scripts.temporal_refinement.playground.losses import compute_playground_loss
from scripts.temporal_refinement.playground.model import PlaygroundModel
from scripts.temporal_refinement.playground.modules import (
    FUSION_HEADS,
    MOTION_ESTIMATORS,
    SEMANTIC_PRIORS,
    STEREO_SOURCES,
    TEACHERS,
    UNCERTAINTY_ESTIMATORS,
    WARPERS,
)
from scripts.temporal_refinement.playground.runner import make_synthetic_batch


def test_required_playground_modules_are_registered() -> None:
    assert {"s2m2_s512", "s2m2_l736"} <= set(STEREO_SOURCES.names())
    assert {"zero", "rgb_difference"} <= set(MOTION_ESTIMATORS.names())
    assert {"identity", "flow"} <= set(WARPERS.names())
    assert {"none", "model_disagreement"} <= set(UNCERTAINTY_ESTIMATORS.names())
    assert {"none", "stereoanyvideo"} <= set(TEACHERS.names())
    assert {"none", "optional_masks"} <= set(SEMANTIC_PRIORS.names())
    assert {
        "raw_s2m2_s",
        "fixed_ema",
        "learned_alpha",
        "flow_warped_adaptive",
        "flow_warped_convgru",
        "dual_memory",
    } <= set(FUSION_HEADS.names())


def test_all_playground_configs_forward_backward() -> None:
    config_dir = Path("configs/temporal_refinement/playground")
    for path in sorted(config_dir.glob("*.yaml")):
        cfg = load_experiment_config(path)
        batch = make_synthetic_batch(batch_size=1, sequence_length=3, height=32, width=48)
        model = PlaygroundModel(cfg)
        has_params = any(p.requires_grad for p in model.parameters())
        if not has_params:
            batch.s2m2_s_disp.requires_grad_(True)
        output = model(batch)
        loss, _metrics = compute_playground_loss(cfg, output, batch)
        assert output.fused_disparity.shape == batch.s2m2_s_disp.shape
        assert output.source_weights.shape[:2] == batch.s2m2_s_disp.shape[:2]
        assert output.source_weights.shape[2] == 3
        assert output.alpha_map.shape == batch.s2m2_s_disp.shape
        assert output.reset_map.shape == batch.s2m2_s_disp.shape
        assert output.residual_map.shape == batch.s2m2_s_disp.shape
        assert output.uncertainty_map.shape == batch.s2m2_s_disp.shape
        loss.backward()
        if has_params:
            assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
