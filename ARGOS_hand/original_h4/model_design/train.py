#!/usr/bin/env python3
"""Train only the exact dataset-2-selected canonical CODD-style H4 run."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "model_design/checkpoints/training_provenance.json"
RUNNER = ROOT / "scripts/run_codd_style_fusion_probe.py"
DEFAULT_OUTPUT = ROOT / "model_design/training_runs/canonical_h4_seed_0"
FROZEN_CHECKPOINT = ROOT / "model_design/checkpoints/codd_style_h4_best_validation.pt"
CANONICAL_RESULTS = Path("/dtu/p1/leopam/ARGOS/ARGOS-V2/results/codd_style_fusion_probe/bida_memory_phase1/full_phase1/seed_0")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata() -> dict:
    value = json.loads(PROVENANCE.read_text())
    locked = value["locked_training"]
    expected = {
        "seed": 0, "train_dataset_ids": [1, 3, 6], "validation_dataset_ids": [2],
        "backbones": ["S2M2-S", "RAFT-Stereo", "StereoAnywhere"], "clip_length": 4,
        "memory_state": "recurrent", "epochs": 12, "batch_size": 4,
        "learning_rate": 0.0002, "optimizer": "AdamW", "weight_decay": 0.0001,
        "scheduler": None, "gradient_clip_norm": 1.0, "mixed_precision_cuda": True,
        "coverage_threshold": 0.5, "tau_reset_native_px": 5.0,
        "tau_fusion_native_px": 1.0, "alpha_reg": 0.2,
        "learned_stereo_evidence": True,
    }
    if locked != expected:
        raise RuntimeError("training provenance is not the locked canonical H4 configuration")
    if sha256(FROZEN_CHECKPOINT) != value["canonical_checkpoint"]["sha256"]:
        raise RuntimeError("canonical H4 checkpoint hash mismatch")
    reference = Path(value["reference_configuration"]["path"])
    if not reference.is_file() or sha256(reference) != value["reference_configuration"]["sha256"]:
        raise RuntimeError("canonical checkpoint configuration metadata hash mismatch")
    return value


def verify_sources(value: dict) -> None:
    for relative, expected in value["runtime_sources"].items():
        path = Path(relative)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"canonical training source hash mismatch: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="new or resumable local run directory (default: %(default)s)")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--preload-workers", type=int, default=20)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true", help="verify and print only; imports no torch and writes nothing")
    return parser.parse_args()


def validate_output(output: Path, *, resume: bool) -> Path:
    resolved = output.expanduser().resolve()
    protected = (FROZEN_CHECKPOINT.parent.resolve(), CANONICAL_RESULTS.resolve())
    if any(resolved == path or resolved in path.parents or path in resolved.parents for path in protected):
        raise RuntimeError("output must not overwrite a canonical checkpoint or canonical results")
    if resolved == ROOT or resolved == (ROOT / "model_design").resolve():
        raise RuntimeError("output must be a dedicated run directory")
    if resolved.exists() and not resume:
        raise RuntimeError("output exists; use --resume or choose a new --output")
    return resolved


def validate_cuda() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not re.fullmatch(r"[0-9]+", visible):
        raise RuntimeError("actual training requires exactly one numeric CUDA_VISIBLE_DEVICES value, e.g. CUDA_VISIBLE_DEVICES=1")
    return visible


def config(value: dict, output: Path, arguments: argparse.Namespace) -> SimpleNamespace:
    locked = value["locked_training"]
    return SimpleNamespace(
        mode="train", output=output, device="cuda:0", seed=locked["seed"], epochs=locked["epochs"],
        overfit_epochs=100, batch_size=locked["batch_size"], workers=arguments.workers,
        preload_workers=arguments.preload_workers, learning_rate=locked["learning_rate"],
        weight_decay=locked["weight_decay"], clip_length=locked["clip_length"],
        max_clips_per_sequence=None, resume=arguments.resume,
        coverage_threshold=locked["coverage_threshold"],
        tau_reset_native_px=locked["tau_reset_native_px"],
        tau_fusion_native_px=locked["tau_fusion_native_px"], alpha_reg=locked["alpha_reg"],
        memory_state=locked["memory_state"],
        disable_learned_stereo_evidence=not locked["learned_stereo_evidence"],
    )


def print_dry_run(value: dict, output: Path, arguments: argparse.Namespace) -> None:
    print(json.dumps({
        "dry_run": True, "writes": False, "torch_imported": False,
        "canonical_checkpoint_sha256": value["canonical_checkpoint"]["sha256"],
        "locked_training": value["locked_training"],
        "operational": {"output": str(output), "workers": arguments.workers,
                        "preload_workers": arguments.preload_workers, "resume": arguments.resume},
        "actual_launch": "CUDA_VISIBLE_DEVICES=<one numeric GPU> python train.py [--output PATH] [--resume|--no-resume]",
        "device": "cuda:0 (logical device after the single-GPU remap)",
        "runner": str(RUNNER), "source_hashes_verified": True,
    }, indent=2, sort_keys=True))


def load_runner():
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("canonical_h4_training_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load canonical runner: {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    arguments = parse_args()
    value = metadata()
    verify_sources(value)
    output = validate_output(arguments.output, resume=arguments.resume)
    if arguments.dry_run:
        print_dry_run(value, output, arguments)
        return
    gpu = validate_cuda()
    print(f"Launching canonical H4 training on physical GPU {gpu} as logical cuda:0; output={output}", flush=True)
    load_runner().train(config(value, output, arguments))


if __name__ == "__main__":
    main()
