"""Shared fail-closed utilities for the ARGOS v2 budget campaign."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path("/dtu/p1/leopam/ARGOS/ARGOS_FREEZED")
CAMPAIGN = ROOT / "experiments/02_massive_training"
REPLICATION = ROOT / "experiments/01_multiseed_replication"
FROZEN_MANIFEST_SHA = "bc8ced366e80c406c42d28657f8d70fc724c223739388880bc0deebbb1f63aed"
MODEL_SHA = "40526a32ef6e9a62a3ea2b59e6751a60c441b8190f9b96522e3b12b35895d5cd"
SEA_SHA = "1a21575ed6ca2c6945fb8e25c4169d241cf59ee5d12b8802c01c965206268cac"
SEEDS = (20260722, 20260723, 20260724)
BUDGET_EPOCHS = {1: 10, 3: 30, 6: 60}
TRAIN_IDS = (1, 3, 6)
VALIDATION_IDS = (2,)
ANCHOR_AGES = (1, 2, 4, 8)
SEEN_BACKBONES = ("S2M2-S", "RAFT-Stereo", "StereoAnywhere")
TRAIN_SEQUENCES = (
    "dataset_1_keyframe_2", "dataset_1_keyframe_3",
    "dataset_3_keyframe_1", "dataset_3_keyframe_2", "dataset_3_keyframe_3", "dataset_3_keyframe_4",
    "dataset_6_keyframe_1", "dataset_6_keyframe_2", "dataset_6_keyframe_3", "dataset_6_keyframe_4",
)
VALIDATION_SEQUENCES = ("dataset_2_keyframe_2", "dataset_2_keyframe_3", "dataset_2_keyframe_4")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_state_sha256(state: dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode()); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(""); return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields); writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def atomic_torch_save(path: Path, value: object) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(value, temporary); temporary.replace(path)


def verify_frozen_core() -> None:
    if sha256(ROOT / "FREEZE_MANIFEST.json") != FROZEN_MANIFEST_SHA:
        raise RuntimeError("frozen manifest hash changed")
    if sha256(ROOT / "checkpoints/raw_multi_anchor_best_validation.pt") != MODEL_SHA:
        raise RuntimeError("canonical checkpoint hash changed")
    subprocess.run([sys.executable, str(ROOT / "scripts/verify_freeze.py")], check=True)


def guard_no_d7(*values: object) -> None:
    def visit(value: object):
        if isinstance(value, dict):
            for key, item in value.items(): yield from visit(key); yield from visit(item)
        elif isinstance(value, (list, tuple, set)):
            for item in value: yield from visit(item)
        elif isinstance(value, int) and value == 7: yield "dataset_id=7"
        else: yield str(value)
    for text in visit(values):
        lower = text.lower()
        if "dataset_7" in lower or re.search(r"(?:dataset|dataset_id|id)[^0-9]*7(?:\D|$)", lower):
            raise RuntimeError(f"D7 ACCESS DENIED before recipe freeze: {text}")


def strict_common_support(*masks):
    if not masks:
        raise ValueError("strict common support needs at least one mask")
    result = masks[0].bool().clone()
    for mask in masks[1:]:
        if mask.shape != result.shape:
            raise ValueError("strict common-support shapes differ")
        result &= mask.bool()
    return result


def run_directory(budget: int, seed: int) -> Path:
    if budget not in BUDGET_EPOCHS or seed not in SEEDS:
        raise ValueError("unregistered budget/seed")
    root = REPLICATION / "budget_1x" if budget == 1 else CAMPAIGN / f"budget_{budget}x"
    result = (root / f"seed_{seed}").resolve()
    result.relative_to((ROOT / "experiments").resolve())
    return result


def recipe_hashes() -> dict[str, str]:
    return {
        "frozen_manifest": FROZEN_MANIFEST_SHA,
        "campaign_config": sha256(CAMPAIGN / "campaign_config.yaml"),
        "selection_protocol": sha256(CAMPAIGN / "selection_protocol.json"),
        "dependency_manifest": sha256(CAMPAIGN / "dependency_manifest.json"),
        "data_manifest": sha256(CAMPAIGN / "data_manifest.json"),
    }


def verify_training_dependencies() -> None:
    manifest = json.loads((CAMPAIGN / "dependency_manifest.json").read_text())
    for section in ("frozen_runtime_sources", "experiment_training_sources", "initial_weight_artifacts"):
        entries = manifest[section]
        if isinstance(entries, dict):
            entries = ({"path": f"initial_weights/seed_{seed}.pt", "sha256": item["file_sha256"]} for seed, item in entries.items())
        for entry in entries:
            path = Path(entry["path"])
            if not path.is_absolute(): path = CAMPAIGN / path
            if sha256(path) != entry["sha256"]: raise RuntimeError(f"training dependency hash changed: {path}")
