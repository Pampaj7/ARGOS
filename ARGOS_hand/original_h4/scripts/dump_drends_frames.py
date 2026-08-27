#!/usr/bin/env python3
"""Dump the per-frame arrays a DRENDS video needs, without touching the evaluator.

`drends_backbone_transfer` produces numbers the paper cites, so a dump path is not
added to it. This sibling reuses the same validated record loading, canonical-frame
contract, depth conversion and flow adapter, drives the same module through the same
`drive()`, and writes the arrays that evaluator discards.

Nothing here is a measurement. The metrics printed alongside are recomputed from the
dumped arrays purely so the video's on-screen numbers cannot drift from its pixels;
the paper's DRENDS figures come from the evaluator and are not touched.

Arrays are float16 because the video reads them through a colormap, and RGB is uint8.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import sys

import numpy as np

# Run as a script, sys.path[0] is scripts/, so the package root and the frozen
# component trees have to be added the way the sibling scripts here already do.
_ROOT = Path(__file__).resolve().parents[1]
_ARGOS = _ROOT.parents[1]
for _p in (str(_ROOT), str(_ROOT / "scripts"), str(_ARGOS / "ARGOS_FREEZED/src"),
           str(_ARGOS / "ARGOS-V2/scripts"),
           str(_ARGOS / "ARGOS_FREEZED/experiments/02_massive_training/scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_design.comparison import drends_backbone_transfer as transfer
from model_design.comparison import drends_evaluation as base
from model_design.comparison.run_comparison import drive, load_factory

OUT = transfer.ROOT.parent / "results" / "drends_video_frames"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", default="RAFT-Stereo")
    parser.add_argument("--module", default="model_design.comparison.ablation_h4:factory_a2")
    parser.add_argument("--recordings", nargs="+", default=list(transfer.ALL_RECORDINGS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()

    import torch
    from argos_freezed.alignment.sea_raft_adapter import SEARAFTFlowAdapter

    device = torch.device(args.device)
    flow_model = SEARAFTFlowAdapter(device=device)
    checkpoint, predict = transfer._load_backbone(args.backbone, device)
    adapter = load_factory(args.module)(device=args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    index_record = {"backbone": args.backbone, "module": args.module,
                    "backbone_checkpoint": checkpoint, "recordings": {}}
    for recording in args.recordings:
        records, info = base.load_drends_records(recording, args.max_frames)
        frames, (height, width) = base._canonical_frames(records, predict, device=args.device)
        scale = transfer.CANONICAL_SIZE[0] / width
        raw = [frame["raw"][0, 0].cpu().numpy() for frame in frames]

        def flow(current: Mapping[str, Any], past: Mapping[str, Any]):
            return base._drends_flow(flow_model, current, past)

        with torch.inference_mode():
            outputs = dict(drive(adapter, frames, flow))
        refined = [outputs[i]["disparity"][0, 0].detach().cpu().numpy() for i in range(len(records))]
        support = [outputs[i]["support"][0, 0].detach().cpu().numpy() for i in range(len(records))]

        depth_values, valid_values, _cov = zip(
            *(base._depth(item["_depth_left"], item["_mask_left"], scale) for item in records))
        product_mm = info["focal_baseline_native_px_m"] * 1000.0 * scale
        gt = [product_mm / np.maximum(value, 1e-6) for value in depth_values]

        rgb = np.stack([f["rgb"][0].detach().cpu().numpy() for f in frames])
        rgb = np.clip(rgb if rgb.max() > 1.5 else rgb * 255.0, 0, 255).astype(np.uint8)

        valid = np.asarray(valid_values, dtype=bool)
        raw_a, ref_a, gt_a = (np.asarray(x, dtype=np.float32) for x in (raw, refined, gt))
        # Recomputed only so the video's caption cannot disagree with its own pixels.
        m = valid & np.isfinite(gt_a) & (gt_a > 0)
        epe_raw = float(np.abs(raw_a - gt_a)[m].mean())
        epe_ref = float(np.abs(ref_a - gt_a)[m].mean())

        np.savez_compressed(
            args.output / f"{args.backbone}__{recording}.npz",
            rgb=rgb, raw=raw_a.astype(np.float16), refined=ref_a.astype(np.float16),
            gt=gt_a.astype(np.float16), gt_valid=valid,
            support=np.asarray(support, dtype=bool),
            reset=np.asarray([bool(outputs[i]["reset"]) for i in range(len(records))]))
        index_record["recordings"][recording] = {
            "frames": len(records), "epe_raw_px": epe_raw, "epe_refined_px": epe_ref,
            "reduction_pct": 100.0 * (epe_raw - epe_ref) / epe_raw,
            "native_resolution": [width, height], "disparity_scale": scale}
        print(f"{recording}: {len(records)} frames, EPE {epe_raw:.4f} -> {epe_ref:.4f} "
              f"({100*(epe_raw-epe_ref)/epe_raw:+.2f}%)", flush=True)

    (args.output / "index.json").write_text(json.dumps(index_record, indent=2) + "\n")


if __name__ == "__main__":
    main()
