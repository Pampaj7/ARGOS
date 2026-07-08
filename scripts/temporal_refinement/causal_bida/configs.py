"""ARGOS v2 one-seed ladder configs."""

from __future__ import annotations

BASE = {
    "seed": 0,
    "clip_len": 8,
    "steps": 1200,
    "batch_size": 1,
    "lr": 3e-4,
    "eval_every": 200,
    "spatial_weight": 1.0,
    "tgm_weight": 1.0,
    "warp_weight": 0.2,
    "safe_weight": 0.2,
    "sparse_weight": 0.02,
    "residual_bound": 3.0,
    "selection_metric": "val_refined_mae",
}

CONFIGS = {
    "raw_s2m2": {"model": "raw", "mode": "full", "eval_only": True, "safe_losses": False},
    "current_only": {"model": "current_only", "mode": "current_only", "safe_losses": True},
    "aligned_local_faithful": {"model": "aligned_local_faithful", "mode": "full", "safe_losses": False},
    "aligned_local_safe": {"model": "aligned_local_safe", "mode": "full", "safe_losses": True},
    "faithful_causal_bida": {"model": "faithful_causal_bida", "mode": "full", "safe_losses": False},
    "faithful_causal_bida_state_reset": {"model": "faithful_causal_bida", "mode": "state_reset", "eval_only": True},
    "faithful_causal_bida_shuffled_history": {"model": "faithful_causal_bida", "mode": "shuffled_history", "eval_only": True},
    "safe_causal_bida": {"model": "safe_causal_bida", "mode": "full", "safe_losses": True},
}


def resolved_config(name: str, **overrides):
    cfg = dict(BASE)
    cfg.update(CONFIGS[name])
    cfg["name"] = name
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg
