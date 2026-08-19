#!/usr/bin/env python3
"""Does the twelve-epoch budget bind? Train the shipped model twice as long and look.

The paper argues the budget is not the binding constraint from the epoch at which
best-validation selection fires: 8, 11 and 11 across the three seeds. That argument does not
survive a hostile reading, and should not -- two of three seeds selecting at epoch 11 of 12
is exactly what an undertrained model looks like. The curve says something better: from
epoch 5 the epoch-to-epoch wobble (0.0025--0.0046 EPE) is as large as everything still to be
gained (0.0041--0.0054), so the minimum lands where it does by noise rather than by trend.
But that is still an argument about a curve, and the only thing that settles it is a longer
curve.

So this runs the locked recipe unchanged except for the epoch count, doubled to 24.

Two constraints make it honest. It writes to its own directory, and its checkpoints are
never eligible for anything: the shipped model stays the one selected under the pre-declared
recipe, whatever this probe shows. If it turns out the model keeps improving to epoch 24,
that is a limitation to report, not a new model to adopt -- adopting it would be selection
on evidence gathered after the fact, which is the practice the paper's own seed study exists
to warn about.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_design.train_ablation import PREREGISTER, VARIANTS, _train_module, install

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--variant", default="A2_no_learned_evidence", choices=sorted(VARIANTS))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--preload-workers", type=int, default=12)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    train = _train_module()
    value = train.metadata()          # the locked-recipe guard still has to pass
    train.verify_sources(value)
    locked_epochs = value["locked_training"]["epochs"]
    if args.epochs <= locked_epochs:
        raise SystemExit(f"a convergence probe must run longer than the locked {locked_epochs}")

    output = train.validate_output(
        ROOT / f"model_design/training_runs/convergence_probe_{args.variant}_{args.epochs}ep",
        resume=args.resume)
    settings = train.config(value, output, args)
    if args.variant == "A2_no_learned_evidence":
        settings.disable_learned_stereo_evidence = True
    settings.epochs = args.epochs      # the one and only deviation from the locked recipe

    record = {
        "purpose": "convergence probe: is the locked 12-epoch budget the binding constraint?",
        "locked_training": value["locked_training"],
        "deviation": {"field": "epochs", "locked": locked_epochs, "probe": args.epochs},
        "variant": args.variant,
        "eligible_for_selection": False,
        "why_not": ("the shipped checkpoint is selected under the pre-declared recipe; adopting "
                    "a checkpoint from this run would be selection on evidence gathered after "
                    "the fact. A probe that keeps improving is a limitation to report."),
        "preregistration": str(PREREGISTER),
        "output": str(output),
    }
    if args.dry_run:
        print(json.dumps(record | {"dry_run": True, "writes": False}, indent=2, sort_keys=True))
        return

    gpu = train.validate_cuda()
    runner = train.load_runner()
    record["patch"] = install(runner, args.variant)
    output.mkdir(parents=True, exist_ok=True)
    (output / "convergence_probe.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"Convergence probe {args.variant}, {args.epochs} epochs against a locked "
          f"{locked_epochs}, on physical GPU {gpu}; output={output}", flush=True)
    runner.train(settings)


if __name__ == "__main__":
    main()
