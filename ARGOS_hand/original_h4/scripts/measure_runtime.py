#!/usr/bin/env python3
"""Stage-by-stage runtime and peak VRAM of the frozen temporal refiner.

Measures only what the temporal module adds on top of a frozen stereo estimator:
SEA-RAFT forward, SEA-RAFT reverse, BiDA alignment, evidence construction (including the
frozen ResNet-18 features when the measured module uses them), and the fusion head.
Frozen stereo latency is not re-measured here: it is already recorded per backbone in the
cache metadata, and is combined with these numbers to report end-to-end FPS.

`--module` selects what is measured, in the same `module:factory` form the evaluation
runners take.  This matters for exactly one pre-registered variant: A2 drops the learned
stereo evidence and with it the frozen ResNet-18, so it is the only variant whose latency
can differ from the canonical model rather than only its parameter count.  The evidence
stage is named after the channel width actually built, and the output file is named after
the module, so a variant cannot silently overwrite the canonical row.

No claim of real-time operation is made by this script; it only produces the numbers.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ARGOS = ROOT.parents[1]
COMPARISON = ROOT / "model_design/comparison"
CACHE = ARGOS / "ARGOS-V2/cache_scaredc_backbones"
OUT = ROOT.parent / "results" / "runtime"
BACKBONES = ("S2M2-S", "RAFT-Stereo", "StereoAnywhere", "CREStereo", "Fast-FoundationStereo")
HEIGHT, WIDTH = 144, 180


def timed(function, iterations: int, warmup: int, device: torch.device) -> dict:
    for _ in range(warmup):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        stop.record()
        torch.cuda.synchronize(device)
        samples.append(start.elapsed_time(stop))
    return {"mean_ms": statistics.mean(samples), "median_ms": statistics.median(samples),
            "p95_ms": sorted(samples)[int(0.95 * len(samples)) - 1], "iterations": iterations}


def backbone_latency() -> dict:
    values = {}
    for backbone in BACKBONES:
        path = CACHE / backbone / "dataset_2_keyframe_2/metadata.json"
        if not path.is_file():
            continue
        meta = json.loads(path.read_text())
        values[backbone] = {
            "median_ms": 1000.0 * float(meta["median_runtime_per_frame_s"]),
            "mean_ms": 1000.0 * float(meta["avg_runtime_per_frame_s"]),
            "p95_ms": 1000.0 * float(meta["p95_runtime_per_frame_s"]),
            "inference_resolution": [meta.get("model_inference_height"), meta.get("model_inference_width")],
            "note": meta.get("model_inference_note"),
            "source": "cache metadata (frozen stereo pass), not re-measured here",
        }
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    # A2 is the only pre-registered variant that removes the frozen ResNet-18 from the
    # deployed system rather than only narrowing the first convolution, so it is the only
    # one whose latency can differ from canonical. That was unmeasurable while this script
    # named its module in an import statement.
    parser.add_argument("--module", default="model_design.comparison.ablation_h4:factory_a2",
                        help="module:factory to measure, as accepted by the evaluation runners")
    parser.add_argument("--output", type=Path, default=None,
                        help="defaults to results/runtime/runtime.json for the canonical module "
                             "and a module-derived name otherwise, so variants cannot overwrite it")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("runtime measurement requires a CUDA device")

    for path in (str(COMPARISON), str(ROOT / "scripts"), str(ARGOS / "ARGOS_FREEZED/src"),
                 str(ARGOS / "ARGOS_FREEZED/experiments/02_massive_training/scripts")):
        if path not in sys.path:
            sys.path.insert(0, path)
    import importlib
    from argos_freezed.alignment.bida_pull_warp import temporal_disparity_evidence
    from argos_freezed.alignment.sea_raft_adapter import SEARAFTFlowAdapter

    module_name, _, factory_name = args.module.partition(":")
    factory = getattr(importlib.import_module(module_name), factory_name or "factory")

    torch.cuda.set_device(device)          # peak-memory stats need an initialised context
    torch.cuda.reset_peak_memory_stats(device)
    adapter = factory(device=str(device))
    model, extractor, build_codd_cues = adapter._load()
    flow = SEARAFTFlowAdapter(device=device)

    generator = torch.Generator(device="cpu").manual_seed(0)
    left = torch.rand(1, 3, HEIGHT, WIDTH, generator=generator).mul(255).to(device)
    right = torch.rand(1, 3, HEIGHT, WIDTH, generator=generator).mul(255).to(device)
    past = torch.rand(1, 3, HEIGHT, WIDTH, generator=generator).mul(255).to(device)
    raw = torch.rand(1, 1, HEIGHT, WIDTH, generator=generator).mul(30).add(5).to(device)
    memory = torch.rand(1, 1, HEIGHT, WIDTH, generator=generator).mul(30).add(5).to(device)
    valid = torch.ones(1, 1, HEIGHT, WIDTH, dtype=torch.bool, device=device)

    stages: dict[str, dict] = {}
    with torch.inference_mode():
        stages["sea_raft_forward"] = timed(lambda: flow.infer(left, past), args.iterations, args.warmup, device)
        stages["sea_raft_reverse"] = timed(lambda: flow.infer(past, left), args.iterations, args.warmup, device)
        forward = flow.infer(left, past)
        backward = flow.infer(past, left)
        stages["bida_alignment"] = timed(
            lambda: temporal_disparity_evidence(raw, memory, forward, backward, current_valid=valid,
                                                past_valid=valid, current_rgb=left, past_rgb=past),
            args.iterations, args.warmup, device)
        evidence = temporal_disparity_evidence(raw, memory, forward, backward, current_valid=valid,
                                               past_valid=valid, current_rgb=left, past_rgb=past)
        build = lambda: build_codd_cues(
            extractor, raw=raw, aligned_memory=evidence.aligned_past_disparity, current_rgb=left,
            current_right_rgb=right, past_rgb=past, flow_current_to_past=forward,
            flow_magnitude=evidence.flow_magnitude,
            forward_backward_confidence=evidence.forward_backward_confidence,
            warp_support=evidence.warp_support, aligned_valid=evidence.aligned_validity,
            # A variant that drops the learned evidence also drops the extractor, and its
            # cue builder would refuse the flag being true. Deriving it here keeps the
            # measured configuration identical to the evaluated one.
            include_learned_stereo_evidence=extractor is not None)
        cues = build()
        # Named after what was actually built, not after the canonical width: a variant
        # measured under a "142ch" label is how a latency table acquires a wrong row.
        stages[f"evidence_{cues.channels}ch"] = timed(build, args.iterations, args.warmup, device)
        stages["fusion_head"] = timed(lambda: model(cues, raw, evidence.aligned_past_disparity),
                                      args.iterations, args.warmup, device)

    overhead = sum(stage["median_ms"] for stage in stages.values())
    peak_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2
    backbones = backbone_latency()
    end_to_end = {
        name: {
            "stereo_median_ms": value["median_ms"],
            "temporal_overhead_ms": overhead,
            "total_ms": value["median_ms"] + overhead,
            "fps": 1000.0 / (value["median_ms"] + overhead),
            "overhead_fraction": overhead / (value["median_ms"] + overhead),
        }
        for name, value in backbones.items()
    }

    OUT.mkdir(parents=True, exist_ok=True)
    default = "runtime.json" if module_name.endswith("canonical_h4") and factory_name in ("", "factory") \
        else f"runtime_{module_name.rsplit('.', 1)[-1]}_{factory_name or 'factory'}.json"
    destination = args.output or (OUT / default)
    record = {
        "project": "ARGOS v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device_name": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cache_grid": [HEIGHT, WIDTH],
        "module": args.module,
        "cue_channels": cues.channels,
        "frozen_extractor_in_runtime": extractor is not None,
        "module_provenance": adapter.describe(),
        "stages_ms": stages,
        "temporal_overhead_median_ms": overhead,
        "peak_vram_mb": peak_mb,
        "frozen_stereo_latency": backbones,
        "end_to_end": end_to_end,
        "caveat": ("frozen stereo latency comes from the cache metadata and was measured on the "
                   "caching hardware at each backbone's own inference resolution; the temporal "
                   "stages are measured here at the 144x180 cache grid"),
        "realtime_claim": None,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {destination}")
    print(json.dumps({k: record[k] for k in ("device_name", "module", "cue_channels",
                                             "frozen_extractor_in_runtime", "stages_ms",
                                             "temporal_overhead_median_ms", "peak_vram_mb",
                                             "end_to_end")}, indent=2))


if __name__ == "__main__":
    main()
