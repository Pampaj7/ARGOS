#!/usr/bin/env python3
"""Canonical H4 training with the seed as the only deviation from the locked recipe.

`train.py` hard-fails unless the locked provenance matches the canonical configuration
byte for byte, including `seed: 0`. That guard is correct and is left in place: this
sibling imports `train.py` and reuses its `metadata()`, `verify_sources()`,
`validate_output()` and `config()` unchanged, then overrides exactly one field.

Everything else -- dataset ids, backbones, clip length, memory state, epochs, batch size,
learning rate, optimiser, weight decay, gradient clipping, mixed precision, thresholds,
the regularisation weight, the learned stereo evidence flag and the checkpoint-selection
rule -- comes from the same locked provenance and is verified at launch.

Pre-registered in `model_design/multiseed_preregister.json` before any run.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREREGISTER = ROOT / "model_design/multiseed_preregister.json"


def _train_module():
    path = ROOT / "model_design/train.py"
    spec = importlib.util.spec_from_file_location("canonical_h4_train", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--preload-workers", type=int, default=20)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    prereg = json.loads(PREREGISTER.read_text())
    allowed = prereg["deviation_from_locked_recipe"]["new_values"]
    if args.seed not in allowed:
        raise SystemExit(f"seed {args.seed} is not pre-registered; allowed: {allowed}")

    train = _train_module()
    value = train.metadata()          # same locked-recipe and checkpoint-hash guards
    train.verify_sources(value)       # same runtime-source hash guard
    output = train.validate_output(
        args.output or ROOT / f"model_design/training_runs/canonical_h4_seed_{args.seed}",
        resume=args.resume)

    settings = train.config(value, output, args)
    if settings.seed != value["locked_training"]["seed"]:
        raise RuntimeError("unexpected: base config seed already deviates from the locked recipe")
    settings.seed = args.seed         # the one and only deviation

    record = {
        "canonical_checkpoint_sha256": value["canonical_checkpoint"]["sha256"],
        "locked_training": value["locked_training"],
        "deviation": {"seed": {"locked": value["locked_training"]["seed"], "used": args.seed}},
        "preregistration": str(PREREGISTER),
        "output": str(output),
    }
    if args.dry_run:
        print(json.dumps(record | {"dry_run": True, "writes": False}, indent=2, sort_keys=True))
        return

    gpu = train.validate_cuda()
    output.mkdir(parents=True, exist_ok=True)
    (output / "seed_provenance.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"Canonical H4 recipe, seed {args.seed}, physical GPU {gpu} as logical cuda:0; "
          f"output={output}", flush=True)
    train.load_runner().train(settings)


if __name__ == "__main__":
    main()
