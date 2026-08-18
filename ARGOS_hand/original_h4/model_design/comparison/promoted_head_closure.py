#!/usr/bin/env python3
"""The development closure re-run for a promoted head, without touching the frozen one.

`experimental_closure.py` and `canonical_horizons.py` are both pinned by the closure's
freeze manifest. Adding a `--head` argument to the first and a class to the second made
the protocol refuse to run -- correctly, because a frozen experiment whose script can be
edited in place is not frozen. So the canonical closure keeps its bytes and this runs
beside it, importing the frozen module rather than modifying it.

What is re-run, and what is not
-------------------------------
Only the six learned-head rows. The fifteen baseline policies blend raw against
flow-aligned raw with a fixed, EMA or forward-backward-confidence weight and load no
checkpoint at all, so they produce identical numbers whichever head the paper ships; the
raw-versus-memory oracle is computed from ground truth and never runs the adapter. Both
are copied forward from the canonical closure by reference, not recomputed, and this
script refuses to start if that source is missing.

The comparison this supports is a like-for-like one: same sequences, same backbones, same
support, same protocol, same evaluation code. The only thing that changes is which trained
head produces the canonical_* rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from model_design.comparison.ablation_horizons import AblationHorizon
from model_design.comparison.definitive_evaluation import evaluate_scared_bundle
from model_design.comparison.experimental_closure import (CANONICAL_HORIZONS, RESULTS, _row,
                                                          _scope, load_freeze)
from model_design.comparison.run_comparison import (ALL_BACKBONES, _scared, atomic_csv,
                                                    atomic_json, prepare_output, sha256,
                                                    validate_cuda)

CANONICAL_D2 = RESULTS / "d2"


def run(config: argparse.Namespace) -> None:
    # The freeze still has to verify: this script does not modify the protocol, it runs a
    # different head under it, and a drifted protocol invalidates the comparison either way.
    freeze = load_freeze()
    if not (CANONICAL_D2 / "summary.csv").is_file():
        raise RuntimeError("the canonical D2 closure must exist first; its baseline policy "
                           "rows and its oracle are the reference this run is compared against")
    suffix = config.head + (f"_seed{config.head_seed}" if config.head_seed is not None else "")
    output = config.output or RESULTS / f"d2_{suffix}"
    if output.resolve() == CANONICAL_D2.resolve():
        raise ValueError("a promoted head may not write the canonical closure root")

    gpu = validate_cuda(config.device)
    prepare_output(output)
    manifest = {"project": "ARGOS v2", "status": "INCOMPLETE", "dataset": "scared-d2",
                "CUDA_VISIBLE_DEVICES": gpu, "head": config.head, "head_seed": config.head_seed,
                "methods_requested": list(config.methods), "output": str(output),
                "freeze": {"path": str(freeze.get("path", "")), "verified": True},
                "canonical_closure": {"path": str(CANONICAL_D2),
                                      "summary_sha256": sha256(CANONICAL_D2 / "summary.csv")},
                "not_rerun": {"baseline_policies": "model-free; identical under any head",
                              "raw_vs_aligned_memory_oracle": "GT-only; never runs the adapter"},
                "scope": freeze["required_d2_scope"], "dense_predictions_written": False,
                "module_provenance": {}}
    atomic_json(output / "run_manifest.json", manifest)

    rows: list[dict[str, Any]] = []
    try:
        for name in config.methods:
            adapter = AblationHorizon(horizon=CANONICAL_HORIZONS[name], variant=config.head,
                                      seed=config.head_seed, device=config.device)
            manifest["module_provenance"][name] = adapter.describe()
            reports: list[tuple[dict[str, Any], Mapping[str, Any]]] = []

            def save(bundle: Mapping[str, Any], _name: str = name) -> None:
                report = evaluate_scared_bundle(bundle)
                reports.append((report, bundle))
                atomic_json(output / "reports" / _name / bundle["backbone"] / f"{bundle['sequence_id']}.json",
                            report | {"method": _name, "head": config.head})

            _scared(config, adapter, save)
            rows.extend(_row(report, name) for report, _ in reports)
            atomic_csv(output / "summary.csv", rows)   # partial, so a kill costs one method
            print(f"{name}: {len(reports)} reports", flush=True)
        atomic_csv(output / "summary.csv", rows)
        load_freeze()   # TOCTOU guard before publishing a COMPLETE result
        atomic_json(output / "run_manifest.json",
                    manifest | {"status": "COMPLETE", "summary_row_count": len(rows)})
    except BaseException as error:
        atomic_json(output / "run_manifest.json", manifest | {"error": f"{type(error).__name__}: {error}"})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--head", required=True, help="pre-registered variant name, e.g. A2_no_learned_evidence")
    parser.add_argument("--head-seed", type=int, help="seed of that head, when it has more than one")
    parser.add_argument("--methods", nargs="+", default=list(CANONICAL_HORIZONS),
                        choices=list(CANONICAL_HORIZONS))
    parser.add_argument("--backbones", nargs="+", default=list(ALL_BACKBONES))
    parser.add_argument("--sequences", nargs="+")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--flow-batch-size", type=int, default=32)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset", default="scared-d2", choices=("scared-d2",))
    parser.add_argument("--smoke", action="store_true")
    config = parser.parse_args()
    if not config.smoke and (tuple(config.methods) != tuple(CANONICAL_HORIZONS)
                             or tuple(config.backbones) != tuple(ALL_BACKBONES)
                             or config.sequences is not None or config.max_frames is not None):
        raise SystemExit("a complete re-run forbids partial methods, custom scope and frame limits; "
                         "pass --smoke to explore")
    run(config)


if __name__ == "__main__":
    main()
