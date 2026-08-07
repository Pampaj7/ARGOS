#!/usr/bin/env python3
"""Phase-1 audit of the already-trained spatial-critic families (ARGOS v2).

Verifies, for the geometry/temporal/stereo checkpoints trained before
plane_sweep: checkpoint presence and hashes, feature-schema (in_channels)
match, best-validation selection consistency after the checkpoint-selection
bug fix (best_validation.pt epoch == argmax of the unconstrained selection
score), frozen-refiner SHA embedded in each checkpoint, and that dataset 7 has
not been opened. Writes the three protocol_audit JSONs the freeze depends on.

Read-only: loads checkpoints on CPU, never trains or mutates weights.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from run_raw_multi_anchor_spatial_safety_critic import (  # noqa: E402
    ENSEMBLE_SEEDS, FAMILY_DIRS, OUTPUT, family_root,
)
from model_design.models.spatial_error_critic import feature_channels  # noqa: E402
from run_raw_multi_anchor_selective_gate import EXPECTED_FROZEN_SHA256, FROZEN_CHECKPOINT  # noqa: E402
from run_raw_multi_anchor_temporal_refiner import TEST, TRAIN, VALIDATION, sha256  # noqa: E402

TRAINED = ("geometry", "temporal", "stereo", "plane_sweep")
SEED = ENSEMBLE_SEEDS[0]
CONFIG = SimpleNamespace(output=OUTPUT)


def read_history(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def audit() -> dict:
    audit_dir = OUTPUT / "protocol_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    family_audit: dict = {}
    checkpoint_hashes: dict = {"frozen_refiner": sha256(FROZEN_CHECKPOINT),
                               "frozen_refiner_expected": EXPECTED_FROZEN_SHA256,
                               "frozen_refiner_match": sha256(FROZEN_CHECKPOINT) == EXPECTED_FROZEN_SHA256}
    all_ok = checkpoint_hashes["frozen_refiner_match"]

    for family in TRAINED:
        root = family_root(CONFIG, family, SEED)
        best_path = root / "checkpoints/best_validation.pt"
        last_path = root / "checkpoints/last.pt"
        history_path = root / "training_history.csv"
        entry: dict = {"dir": FAMILY_DIRS[family], "expected_in_channels": feature_channels(family)}

        entry["best_checkpoint_exists"] = best_path.exists()
        entry["last_checkpoint_exists"] = last_path.exists()
        if not best_path.exists():
            entry["ok"] = False
            family_audit[family] = entry
            all_ok = False
            continue

        payload = torch.load(best_path, map_location="cpu", weights_only=False)
        checkpoint_hashes[family] = sha256(best_path)
        if last_path.exists():
            checkpoint_hashes[f"{family}_last"] = sha256(last_path)

        entry["checkpoint_in_channels"] = int(payload["in_channels"])
        entry["schema_match"] = payload["in_channels"] == feature_channels(family)
        entry["checkpoint_epoch"] = int(payload["epoch"])
        entry["harm_margin"] = float(payload["harm_margin"])
        entry["frozen_sha_in_checkpoint"] = payload["frozen_checkpoint_sha256"]
        entry["frozen_sha_match"] = payload["frozen_checkpoint_sha256"] == EXPECTED_FROZEN_SHA256
        # parameter count from state_dict
        entry["parameters"] = int(sum(t.numel() for t in payload["model"].values()))

        # best-validation selection consistency (unconstrained selection score)
        history = read_history(history_path)
        gains = [(int(row["epoch"]), float(row["validation_best_selective_gain"])) for row in history]
        best_epoch, best_gain = max(gains, key=lambda pair: pair[1])
        entry["history_epochs"] = len(history)
        entry["argmax_selection_epoch"] = best_epoch
        entry["best_selective_gain"] = best_gain
        entry["selection_consistent"] = best_epoch == payload["epoch"]
        # confirm the fix: selection score is NOT frozen at epoch 1 (bug symptom)
        entry["selection_not_frozen_epoch1"] = best_epoch != 1 or len(gains) == 1
        auroc_at_best = next(float(row["validation_harm_auroc"]) for row in history
                             if int(row["epoch"]) == best_epoch)
        entry["harm_auroc_at_selected"] = auroc_at_best

        entry["ok"] = bool(entry["best_checkpoint_exists"] and entry["schema_match"]
                           and entry["frozen_sha_match"] and entry["selection_consistent"])
        all_ok = all_ok and entry["ok"]
        family_audit[family] = entry

    family_audit["_all_ok"] = all_ok
    checkpoint_hashes["_all_frozen_match"] = all(
        family_audit[f].get("frozen_sha_match", False) for f in TRAINED)

    # dataset-access audit: D7 disjoint from train/val, and D7 not opened yet
    train_val = set(TRAIN) | set(VALIDATION)
    test_set = set(TEST)
    frozen_test_dir = OUTPUT / "frozen_test"
    freeze_manifest = OUTPUT / "calibration/freeze_manifest.json"
    test_opened_marker = OUTPUT / "protocol_audit/test_opened.json"
    dataset_access = {
        "train_sequences": list(TRAIN), "validation_sequences": list(VALIDATION),
        "test_sequences": list(TEST), "test_dataset_ids": [7],
        "test_disjoint_from_train_val": len(train_val & test_set) == 0,
        "frozen_test_dir_exists": frozen_test_dir.exists(),
        "freeze_manifest_exists": freeze_manifest.exists(),
        "test_opened_marker_exists": test_opened_marker.exists(),
        "dataset7_opened": frozen_test_dir.exists() or test_opened_marker.exists(),
        "note": "dataset 7 must remain closed until the freeze manifest is written",
    }

    save = lambda name, obj: (audit_dir / name).write_text(json.dumps(obj, indent=2))
    save("existing_family_audit.json", family_audit)
    save("checkpoint_hashes.json", checkpoint_hashes)
    save("dataset_access_audit.json", dataset_access)
    return {"family_audit": family_audit, "checkpoint_hashes": checkpoint_hashes,
            "dataset_access": dataset_access}


if __name__ == "__main__":
    result = audit()
    # self-check: the audit is only meaningful if every trained family passed.
    fa = result["family_audit"]
    assert fa["_all_ok"], f"family audit failed: {json.dumps(fa, indent=2)}"
    assert not result["dataset_access"]["dataset7_opened"], "dataset 7 already opened before freeze"
    print(json.dumps({
        "all_ok": fa["_all_ok"],
        "families": {f: {"epoch": fa[f]["checkpoint_epoch"],
                         "in_channels": fa[f]["checkpoint_in_channels"],
                         "selection_consistent": fa[f]["selection_consistent"],
                         "harm_auroc": round(fa[f]["harm_auroc_at_selected"], 4),
                         "best_gain": round(fa[f]["best_selective_gain"], 4)} for f in TRAINED},
        "dataset7_opened": result["dataset_access"]["dataset7_opened"],
    }, indent=2))
