#!/usr/bin/env python3
"""Train one pre-registered architecture ablation.

Same relationship to `train.py` as `train_seed.py`: it reuses that file's metadata,
source-hash verification, output validation and locked configuration unchanged, and
changes exactly one thing. `train.py` and the pinned `codd_style_fusion.py` are not
modified; the variant is installed by rebinding the name the runner imported.

Refuses any variant that is not declared in `ablation_preregister.json`.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREREGISTER = ROOT / "model_design/ablation_preregister.json"
VARIANTS = {
    "A1_no_appearance": "A1_no_appearance_channels",
    "A2_no_learned_evidence": "A2_no_learned_stereo_evidence",
    "A3_single_resolution": "A3_single_resolution",
    "A4_relaxed_convexity": "A4_relaxed_convexity",
    "A5_no_fb_cue": "A5_no_fb_cue",
    "A6_geometry_only": "A6_geometry_only",
    "A7_half_width": "A7_half_width",
    "A3b_single_resolution_38ch": "A3b_single_resolution_38ch",
    "A4b_relaxed_convexity_38ch": "A4b_relaxed_convexity_38ch",
}
# Variants that deviate from the shipped 38-channel head and therefore inherit its
# disable-learned-evidence flag. One list, imported by train_ablation_seed.py too:
# the seed script carrying its own copy is how an A3b seed run would silently have
# trained on 142 channels while writing provenance that says 38.
SHIPPED_BASE_VARIANTS = ("A2_no_learned_evidence", "A5_no_fb_cue", "A6_geometry_only",
                         "A7_half_width", "A3b_single_resolution_38ch",
                         "A4b_relaxed_convexity_38ch")


def _train_module():
    path = ROOT / "model_design/train.py"
    spec = importlib.util.spec_from_file_location("canonical_h4_train", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install(runner, variant: str) -> dict:
    """Rebind exactly one name in the runner's namespace and report what changed."""
    from model_design.comparison.ablation_variants import (
        HalfWidthHead, RelaxedConvexityHead, SingleResolutionHead,
        build_cues_geometry_only, build_cues_without_appearance,
        build_cues_without_fb_confidence,
    )
    if variant == "A1_no_appearance":
        canonical = runner.build_codd_cues
        runner.build_codd_cues = build_cues_without_appearance
        return {"patched": "build_codd_cues", "from": canonical.__name__,
                "to": "build_cues_without_appearance", "cue_channels": 78}
    if variant == "A4_relaxed_convexity":
        canonical = runner.CODDStyleFusionHead
        runner.CODDStyleFusionHead = RelaxedConvexityHead
        return {"patched": "CODDStyleFusionHead", "from": canonical.__name__,
                "to": "RelaxedConvexityHead", "extra_parameters": 49}
    if variant == "A3_single_resolution":
        canonical = runner.CODDStyleFusionHead
        runner.CODDStyleFusionHead = SingleResolutionHead
        return {"patched": "CODDStyleFusionHead", "from": canonical.__name__,
                "to": "SingleResolutionHead", "extra_parameters": 0}
    if variant == "A2_no_learned_evidence":
        # Nothing is patched: the runner already honours the config flag.
        return {"patched": None, "config_field": "disable_learned_stereo_evidence",
                "value": True, "cue_channels": 38}
    # A5 to A7 are deviations from the SHIPPED 38-channel head, not from the 142-channel
    # configuration the first four were registered against. Their builders force the
    # learned-evidence flag themselves so the comparison is against A2 and nothing else.
    if variant == "A5_no_fb_cue":
        canonical = runner.build_codd_cues
        runner.build_codd_cues = build_cues_without_fb_confidence
        return {"patched": "build_codd_cues", "from": canonical.__name__,
                "to": "build_cues_without_fb_confidence", "cue_channels": 38,
                "note": "channel 32 held at 1.0 throughout training"}
    if variant == "A6_geometry_only":
        canonical = runner.build_codd_cues
        runner.build_codd_cues = build_cues_geometry_only
        return {"patched": "build_codd_cues", "from": canonical.__name__,
                "to": "build_cues_geometry_only", "cue_channels": 13}
    if variant == "A7_half_width":
        canonical = runner.CODDStyleFusionHead
        runner.CODDStyleFusionHead = HalfWidthHead
        return {"patched": "CODDStyleFusionHead", "from": canonical.__name__,
                "to": "HalfWidthHead", "width": 32, "trainable_parameters": 70994}
    # A3b and A4b are A3 and A4 re-asked of the shipped head: the same head class, but
    # built on the 38-channel evidence space instead of the 142-channel one. The cue
    # builder needs no patch -- the learned-evidence flag set in main() is what produces
    # the 38 channels, exactly as it does for A2.
    if variant == "A3b_single_resolution_38ch":
        canonical = runner.CODDStyleFusionHead
        runner.CODDStyleFusionHead = SingleResolutionHead
        return {"patched": "CODDStyleFusionHead", "from": canonical.__name__,
                "to": "SingleResolutionHead", "extra_parameters": 0, "cue_channels": 38}
    if variant == "A4b_relaxed_convexity_38ch":
        canonical = runner.CODDStyleFusionHead
        runner.CODDStyleFusionHead = RelaxedConvexityHead
        return {"patched": "CODDStyleFusionHead", "from": canonical.__name__,
                "to": "RelaxedConvexityHead", "extra_parameters": 49, "cue_channels": 38}
    raise ValueError(variant)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--preload-workers", type=int, default=20)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    prereg = json.loads(PREREGISTER.read_text())
    key = VARIANTS[args.variant]
    if key not in prereg["deviations_from_locked_recipe"]:
        raise SystemExit(f"{args.variant} is not pre-registered")
    declared = prereg["deviations_from_locked_recipe"][key]

    train = _train_module()
    value = train.metadata()          # same locked-recipe and checkpoint-hash guards
    train.verify_sources(value)       # same runtime-source hash guard
    output = train.validate_output(
        args.output or ROOT / f"model_design/training_runs/ablation_{args.variant}",
        resume=args.resume)

    settings = train.config(value, output, args)
    if settings.seed != value["locked_training"]["seed"]:
        raise RuntimeError("unexpected: base config seed already deviates from the locked recipe")
    if settings.disable_learned_stereo_evidence:
        raise RuntimeError("unexpected: the locked recipe already disables learned stereo evidence")
    if args.variant in SHIPPED_BASE_VARIANTS:
        # A5-A7 and the A3b/A4b twins deviate from the shipped 38-channel head, so they
        # inherit its one flag.
        settings.disable_learned_stereo_evidence = True

    record = {
        "canonical_checkpoint_sha256": value["canonical_checkpoint"]["sha256"],
        "locked_training": value["locked_training"],
        "variant": args.variant,
        "declared": declared,
        "preregistration": str(PREREGISTER),
        "output": str(output),
    }

    if args.dry_run:
        runner_names = {"note": "runner not loaded in dry-run; patch target verified by name only",
                        "target": "build_codd_cues" if args.variant == "A1_no_appearance"
                                  else "CODDStyleFusionHead" if args.variant in
                                  ("A4_relaxed_convexity", "A3_single_resolution",
                                   "A7_half_width", "A3b_single_resolution_38ch",
                                   "A4b_relaxed_convexity_38ch")
                                  else None}
        print(json.dumps(record | {"dry_run": True, "writes": False, "patch": runner_names},
                         indent=2, sort_keys=True))
        return

    gpu = train.validate_cuda()
    runner = train.load_runner()
    record["patch"] = install(runner, args.variant)
    output.mkdir(parents=True, exist_ok=True)
    (output / "ablation_provenance.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"Ablation {args.variant} on physical GPU {gpu} as logical cuda:0; "
          f"patch={record['patch']}; output={output}", flush=True)
    runner.train(settings)


if __name__ == "__main__":
    main()
