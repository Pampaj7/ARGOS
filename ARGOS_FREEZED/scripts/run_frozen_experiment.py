#!/usr/bin/env python3
"""Verified, fail-closed launcher for ARGOS v2 experiment-local commands."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = (ROOT / "experiments").resolve()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent)
        return True
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--mode", choices=("train", "validation", "test"), required=True)
    args = parser.parse_args()
    subprocess.run([sys.executable, str(ROOT / "scripts/verify_freeze.py")], check=True)
    config_path = args.config.resolve()
    if not inside(config_path, EXPERIMENTS):
        raise SystemExit("config must live under ARGOS_FREEZED/experiments")
    config = yaml.safe_load(config_path.read_text())
    experiment_dir = config_path.parent.resolve()
    output = (ROOT / config["output_path"]).resolve()
    if not inside(output, experiment_dir) or not inside(output, EXPERIMENTS):
        raise SystemExit("output must remain inside the selected experiment directory")
    freeze_sha = sha256(ROOT / "FREEZE_MANIFEST.json")
    experiment_manifest = experiment_dir / "manifest.json"
    if args.mode == "test":
        if not experiment_manifest.is_file():
            raise SystemExit("test refused: experiment manifest is missing")
        frozen = json.loads(experiment_manifest.read_text())
        if not frozen.get("validation_decision_frozen") or frozen.get("frozen_manifest_sha256") != freeze_sha:
            raise SystemExit("test refused: validation decision/core hash is not frozen")
    command = config.get("commands", {}).get(args.mode)
    if not isinstance(command, list) or not command:
        raise SystemExit(f"no explicit {args.mode} command in config; nothing was run")
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "project": "ARGOS v2",
        "mode": args.mode,
        "experiment_id": config["experiment_id"],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "frozen_manifest_sha256": freeze_sha,
        "source_commit": json.loads((ROOT / "FREEZE_MANIFEST.json").read_text())["source_git_commit"],
        "checkpoint_hashes": json.loads((ROOT / "evidence/checkpoint_hashes.json").read_text()),
        "dataset_cache_paths": config.get("dataset_cache_paths", []),
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "command": command,
        "dense_caches": False,
    }
    (output / f"{args.mode}_run_manifest.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    environment = os.environ.copy()
    environment["ARGOS_FREEZED_ROOT"] = str(ROOT)
    environment["ARGOS_EXPERIMENT_OUTPUT"] = str(output)
    with (output / f"{args.mode}.log").open("a") as log:
        completed = subprocess.run(command, cwd=experiment_dir, env=environment, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
