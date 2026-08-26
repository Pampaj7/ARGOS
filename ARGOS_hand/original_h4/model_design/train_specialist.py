#!/usr/bin/env python3
"""Train the per-backbone specialist control: the shipped head, one backbone only.

Same relationship to `train.py` as `train_ablation.py`: that file's metadata,
source-hash verification, output validation and locked configuration are reused
unchanged, and exactly one name is rebound in the runner's namespace. Nothing on
disk is edited, so the canonical hash guard still means what it says.

The rebound name is SEEN_BACKBONES, which the runner reads twice -- once to build
the datasets and once to write split_audit.json -- so the recorded provenance
cannot disagree with the data the run actually saw.

Refuses any backbone or epoch count not declared in specialist_control_declaration.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECLARATION = ROOT / "model_design/specialist_control_declaration.json"


def _train_module():
    path = ROOT / "model_design/train.py"
    spec = importlib.util.spec_from_file_location("canonical_h4_train", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install(runner, backbone: str) -> dict:
    """Rebind exactly one name in the runner's namespace and report what changed."""
    canonical = tuple(runner.SEEN_BACKBONES)
    if backbone not in ("S2M2-S", "RAFT-Stereo", "StereoAnywhere",
                        "CREStereo", "Fast-FoundationStereo"):
        raise SystemExit(f"{backbone} is not a SCARED-C cached backbone")
    runner.SEEN_BACKBONES = (backbone,)
    return {"patched": "SEEN_BACKBONES", "from": list(canonical), "to": [backbone]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--preload-workers", type=int, default=20)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    declared = json.loads(DECLARATION.read_text())
    arm = f"specialist_{args.backbone}"
    if arm not in declared["arms"]:
        raise SystemExit(f"{arm} is not a declared arm; allowed: {declared['arms']}")
    epochs = declared["deviations_from_locked_recipe"]["epochs"]["used"]

    train = _train_module()
    value = train.metadata()
    train.verify_sources(value)
    output = train.validate_output(
        ROOT / f"model_design/training_runs/{arm}", resume=args.resume)

    settings = train.config(value, output, args)
    settings.epochs = epochs
    # The control is against the head we ship, which is A2: no learned stereo evidence.
    settings.disable_learned_stereo_evidence = True

    gpu = train.validate_cuda()
    runner = train.load_runner()
    patch = install(runner, args.backbone)
    output.mkdir(parents=True, exist_ok=True)
    (output / "specialist_provenance.json").write_text(json.dumps({
        "canonical_checkpoint_sha256": value["canonical_checkpoint"]["sha256"],
        "locked_training": value["locked_training"],
        "arm": arm, "backbone": args.backbone,
        "deviation": {"backbones": patch,
                      "epochs": {"locked": value["locked_training"]["epochs"], "used": epochs},
                      "learned_stereo_evidence": False},
        "declaration": str(DECLARATION),
        "output": str(output),
    }, indent=2, sort_keys=True) + "\n")
    print(f"{arm} on physical GPU {gpu}; {epochs} epochs; patch={patch}", flush=True)
    runner.train(settings)


if __name__ == "__main__":
    main()
