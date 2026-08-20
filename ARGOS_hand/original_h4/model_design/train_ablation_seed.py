#!/usr/bin/env python3
"""Train a pre-registered ablation at a pre-registered seed.

Composes the two existing sibling trainers rather than duplicating either: the ablation
patch comes from train_ablation.py, the seed deviation from multiseed_preregister.json.
Both guards still apply, so this refuses an undeclared variant and an undeclared seed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = ROOT / "model_design/multiseed_preregister.json"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--preload-workers", type=int, default=20)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    allowed = json.loads(SEEDS.read_text())["deviation_from_locked_recipe"]["new_values"]
    if args.seed not in allowed:
        raise SystemExit(f"seed {args.seed} is not pre-registered; allowed: {allowed}")

    ablation = _module("train_ablation", ROOT / "model_design/train_ablation.py")
    if args.variant not in ablation.VARIANTS:
        raise SystemExit(f"{args.variant} is not a pre-registered variant")
    prereg = json.loads(ablation.PREREGISTER.read_text())["deviations_from_locked_recipe"]
    declared = prereg[ablation.VARIANTS[args.variant]]

    train = ablation._train_module()
    value = train.metadata()
    train.verify_sources(value)
    output = train.validate_output(
        ROOT / f"model_design/training_runs/ablation_{args.variant}_seed_{args.seed}",
        resume=args.resume)

    settings = train.config(value, output, args)
    settings.seed = args.seed
    if args.variant in ablation.SHIPPED_BASE_VARIANTS:
        settings.disable_learned_stereo_evidence = True

    gpu = train.validate_cuda()
    runner = train.load_runner()
    patch = ablation.install(runner, args.variant)
    output.mkdir(parents=True, exist_ok=True)
    (output / "ablation_provenance.json").write_text(json.dumps({
        "canonical_checkpoint_sha256": value["canonical_checkpoint"]["sha256"],
        "locked_training": value["locked_training"],
        "variant": args.variant, "declared": declared,
        "deviation": {"seed": {"locked": value["locked_training"]["seed"], "used": args.seed}},
        "preregistration": [str(ablation.PREREGISTER), str(SEEDS)],
        "patch": patch, "output": str(output),
    }, indent=2, sort_keys=True) + "\n")
    print(f"{args.variant} seed {args.seed} on physical GPU {gpu}; patch={patch}", flush=True)
    runner.train(settings)


if __name__ == "__main__":
    main()
