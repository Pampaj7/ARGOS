#!/usr/bin/env python3
"""Recover the canonical ARGOS v2 recipe and materialize exact scratch initializations."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from campaign_common import CAMPAIGN, SEEDS, atomic_json, atomic_torch_save, sha256, tensor_state_sha256, verify_frozen_core

V2 = Path("/dtu/p1/leopam/ARGOS/ARGOS-V2")


def main() -> None:
    verify_frozen_core(); sys.path[:0] = [str(V2), str(V2 / "scripts")]
    from model_design.external_components.bidavideo import BiDAFlowInferenceAdapter
    from model_design.models.raw_multi_anchor_refiner import RawMultiAnchorRefiner
    from run_hybrid_temporal_memory_oracle_audit import load_model as load_corrected_model
    from run_raw_multi_anchor_temporal_refiner import BalancedGroupSampler, seed_all
    initial = {}
    for seed in SEEDS:
        seed_all(seed)
        first = BiDAFlowInferenceAdapter("sea_raft", device="cpu"); del first
        second = BiDAFlowInferenceAdapter("sea_raft", device="cpu"); corrected, _ = load_corrected_model(torch.device("cpu")); del second, corrected
        model = RawMultiAnchorRefiner(32, 3)
        payload = {"project": "ARGOS v2", "seed": seed, "initialization": "scratch_after_canonical_train_and_validation_bank_dependency_construction",
                   "model": model.state_dict(), "tensor_sha256": tensor_state_sha256(model.state_dict())}
        path = CAMPAIGN / f"initial_weights/seed_{seed}.pt"; atomic_torch_save(path, payload)
        initial[str(seed)] = {"path": str(path), "file_sha256": sha256(path), "tensor_sha256": payload["tensor_sha256"]}
    recipe = {
        "project": "ARGOS v2", "status": "FROZEN_BEFORE_FULL_TRAINING", "source_runner": str(V2 / "scripts/run_raw_multi_anchor_temporal_refiner.py"),
        "train_ids": [1, 3, 6], "validation_ids": [2], "test_id_locked": 7,
        "train_sequences": ["dataset_1_keyframe_2", "dataset_1_keyframe_3", "dataset_3_keyframe_1", "dataset_3_keyframe_2", "dataset_3_keyframe_3", "dataset_3_keyframe_4", "dataset_6_keyframe_1", "dataset_6_keyframe_2", "dataset_6_keyframe_3", "dataset_6_keyframe_4"],
        "validation_sequences": ["dataset_2_keyframe_2", "dataset_2_keyframe_3", "dataset_2_keyframe_4"],
        "backbones": ["S2M2-S", "RAFT-Stereo", "StereoAnywhere"], "raw_examples": 25608,
        "sampler": "BalancedGroupSampler: 30 backbone-sequence groups oversampled to max group 1589; deterministic seed+epoch interleaving",
        "samples_per_epoch": 47670, "batch_size": 12, "effective_batch_size": 12, "steps_per_epoch": 3973,
        "canonical_epochs": 10, "canonical_optimizer_steps": 39730,
        "budgets": {"1x": {"epochs": 10, "steps": 39730}, "3x": {"epochs": 30, "steps": 119190}, "6x": {"epochs": 60, "steps": 238380}},
        "optimizer": {"name": "AdamW", "lr": .002, "weight_decay": .0001, "betas": [.9, .999], "eps": 1e-8, "amsgrad": False},
        "scheduler": {"name": "CosineAnnealingLR", "step_unit": "optimizer_step", "T_max": "complete budget optimizer steps", "eta_min": .0001},
        "warmup": None, "amp": "torch.autocast CUDA + GradScaler enabled", "gradient_clip_norm": 5.0,
        "crop": {"training": [64, 80], "one_fixed_random_crop_per_frame_backbone": True, "validation": [144, 180], "other_augmentation": None},
        "validation_cadence": {"epochs": 1, "optimizer_steps": 3973},
        "checkpoint_selection": "maximum validation_best_gain; per-epoch D2 stride-64 policy grid p=[.3..9], u=[-.05,0,.01,.02,.05,.1], feasible coverage>=.005 and harm-cost<=.25 else maximum gain",
        "deployment_thresholds": {"probability": .9, "utility_px": .1}, "coverage_threshold": .5, "margin_px": .1,
        "loss": {"classification": 1.0, "regression": .5, "ranking": .25, "fusion": 1.0, "harmful_weight_regularizer": .2, "maximum_positive_weight": 50.0, "clipped_delta_px": 8.0},
        "workers": 48, "prefetch_factor": 4, "persistent_workers": True, "pin_memory": True, "flow_batch_size": 32,
        "cache_behavior": "validated stereo caches reused; SEA-RAFT live frozen; evidence bank held only in RAM; no persistent dense predictions",
        "initialization": "from scratch after canonical dependency construction; never initialized from canonical trained checkpoint", "initial_weights": initial,
    }
    atomic_json(CAMPAIGN / "protocol_audit/canonical_recipe_audit.json", recipe)
    print(json.dumps(recipe, indent=2))


if __name__ == "__main__": main()
