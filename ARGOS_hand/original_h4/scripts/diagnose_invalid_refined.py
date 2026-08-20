#!/usr/bin/env python3
"""Locate and characterise the pixels where the refiner invalidates a valid raw prediction.

Across DRENDS the raw backbone never produces an invalid disparity on protocol support
(InvalidRate 0.000e+00) while the refined output does (up to 1.3e-05), and the size of that
rate tracks the metric-depth regression recording by recording.  Each such pixel is charged
`MetricConfig.invalid_penalty_mm = 10000.0`, so a handful of them decide the external-OOD gates.

This runs the frozen canonical H4 exactly as the DRENDS evaluator does and, for every pixel
where the fused disparity is not finite-and-positive, records the state that produced it:
raw, aligned memory, recovered fusion weight, the three evidence flags and the state age.

Nothing is trained, no threshold is tuned, and no pinned module is modified.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "results" / "invalid_refined_diagnosis"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recording", default="Vid10_Liver_Med")
    parser.add_argument("--backbone", default="RAFT-Stereo")
    parser.add_argument("--max-frames", type=int, help="default is the whole recording")
    parser.add_argument("--module", default="model_design.comparison.ablation_h4:factory_a2")
    parser.add_argument("--tag", default="", help="suffix for the output file names")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    # The 142-channel results already live at the bare path. A different head must not
    # land on them: this is how a stale table survived a recanonicalisation once.
    out = OUT if "canonical_h4" in args.module else OUT / "a2"

    import torch
    for path in (str(ROOT), str(ROOT / "scripts"), str(ROOT.parents[1] / "ARGOS_FREEZED/src")):
        if path not in sys.path:
            sys.path.insert(0, path)
    from argos_freezed.alignment.sea_raft_adapter import SEARAFTFlowAdapter
    from model_design.comparison import drends_evaluation as base
    from model_design.comparison.drends_backbone_transfer import _load_backbone
    from model_design.comparison.run_comparison import drive, load_factory

    device = torch.device(args.device)
    adapter = load_factory(args.module)(device=args.device)
    flow_model = SEARAFTFlowAdapter(device=device)
    _, predict = _load_backbone(args.backbone, device)

    records, info = base.load_drends_records(args.recording, args.max_frames)
    frames, (height, width) = base._canonical_frames(records, predict, device=args.device)

    def flow(current, past):
        return base._drends_flow(flow_model, current, past)

    outputs = dict(drive(adapter, frames, flow))

    offenders = []
    per_frame = Counter()
    total_support = 0
    for index in range(len(frames)):
        result = outputs[index]
        fused = result["disparity"][0, 0].detach().cpu().numpy()
        raw = frames[index]["raw"][0, 0].cpu().numpy()
        support = result["support"][0, 0].detach().cpu().numpy().astype(bool)
        raw_valid = frames[index]["raw_valid"][0, 0].cpu().numpy().astype(bool)
        memory = result.get("aligned_memory")
        memory = memory[0, 0].detach().cpu().numpy() if memory is not None else np.full_like(raw, np.nan)
        total_support += int(raw_valid.sum())
        bad = raw_valid & ~(np.isfinite(fused) & (fused > 0))
        if not bad.any():
            continue
        per_frame[index] = int(bad.sum())
        span = memory - raw
        weight = np.where(np.abs(span) > 1e-3, (fused - raw) / np.where(np.abs(span) > 1e-3, span, 1.0), np.nan)
        for y, x in zip(*np.nonzero(bad)):
            offenders.append({
                "frame_index": int(index), "frame_id": records[index]["frame_id"], "y": int(y), "x": int(x),
                "raw": float(raw[y, x]), "aligned_memory": float(memory[y, x]), "fused": float(fused[y, x]),
                "recovered_weight": float(weight[y, x]), "in_module_support": bool(support[y, x]),
                "state_age": int(result["state_age"]), "reset": bool(result["reset"]),
            })

    out.mkdir(parents=True, exist_ok=True)
    memory_values = [item["aligned_memory"] for item in offenders]
    summary = {
        "project": "ARGOS v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "recording": args.recording, "backbone": args.backbone,
        "module": args.module, "module_provenance": adapter.describe(),
        "frames": len(frames), "support_pixels": total_support,
        "invalid_refined_pixels": len(offenders),
        "invalid_rate": len(offenders) / max(total_support, 1),
        "frames_with_invalid": len(per_frame),
        "worst_frames": per_frame.most_common(10),
        "aligned_memory_of_offenders": {
            "non_positive": int(sum(1 for v in memory_values if not (v > 0))),
            "non_finite": int(sum(1 for v in memory_values if not np.isfinite(v))),
            "min": float(np.min(memory_values)) if memory_values else None,
            "max": float(np.max(memory_values)) if memory_values else None,
        },
        "inside_module_support": int(sum(1 for item in offenders if item["in_module_support"])),
        "state_age_histogram": dict(Counter(item["state_age"] for item in offenders)),
        "invalid_penalty_mm": 10000.0,
        "training_performed": False,
    }
    (out / f"{args.recording}_{args.backbone}{args.tag}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / f"{args.recording}_{args.backbone}{args.tag}_pixels.json").write_text(json.dumps(offenders[:500], indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
