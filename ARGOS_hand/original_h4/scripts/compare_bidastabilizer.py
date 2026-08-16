#!/usr/bin/env python3
"""Head-to-head Raw / BiDAStabilizer / TETHER on identical inputs and identical support.

The published BiDAStabilizer reproduction and our own evaluation were not comparable for
two independent reasons: BiDA ran on RAFT-Stereo *robust* while we ran RAFT-Stereo
*middlebury*, and it was scored under `d2_full_diagnostic_raw_valid` while we use
`paper_d2_strict_all_anchors`. Subtracting those numbers would be meaningless.

This removes both differences. It reads the frozen NPZ boundary the external comparison
already wrote --- the exact RGB and the exact RAFT-Stereo robust disparity BiDA consumed ---
runs our module on that same input, and scores raw, BiDA and ours against the same ground
truth on one prediction-independent support: GT coverage AND raw validity, which neither
method influences.

The remaining difference is the one that matters and is reported rather than removed:
BiDAStabilizer is bidirectional and consumes future frames; ours is strictly causal.

Nothing is trained and no threshold is tuned.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARGOS = ROOT.parents[1]
BIDA = ARGOS / "ARGOS_hand/external_comparison/results/bidastabilizer_raftstereo_robust/d2_full"
OUT = ROOT.parent / "results" / "bida_common_support"
SEQUENCES = ("dataset_2_keyframe_2", "dataset_2_keyframe_3", "dataset_2_keyframe_4")


def metrics(prediction: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict:
    """EPE, bad-pixel rates and RMSE on a fixed, prediction-independent support.

    An invalid prediction on valid support is kept and counted, never dropped.
    """
    valid = mask & np.isfinite(gt) & (gt > 0)
    if not valid.any():
        return {}
    error = np.abs(prediction[valid] - gt[valid])
    finite = np.isfinite(prediction[valid]) & (prediction[valid] > 0)
    error = np.where(finite, error, 1000.0)          # same invalid penalty as the framework
    return {"pixels": int(valid.sum()),
            "EPE": float(error.mean()),
            "Bad1": float((error > 1).mean()),
            "Bad3": float((error > 3).mean()),
            "RMSE": float(np.sqrt((error ** 2).mean())),
            "P95": float(np.percentile(error, 95)),
            "InvalidRate": float((~finite).mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--module", default="model_design.comparison.canonical_h4_masked:factory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sequences", nargs="+", default=list(SEQUENCES))
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()

    import torch
    for path in (str(ROOT), str(ROOT / "scripts"), str(ARGOS / "ARGOS_FREEZED/src"),
                 str(ARGOS / "ARGOS-V2/scripts")):
        if path not in sys.path:
            sys.path.insert(0, path)
    from argos_freezed.alignment.sea_raft_adapter import SEARAFTFlowAdapter
    from model_design.comparison.run_comparison import drive, load_factory

    device = torch.device(args.device)
    adapter = load_factory(args.module)(device=args.device)
    flow_model = SEARAFTFlowAdapter(device=device)
    rows = []

    for sequence in args.sequences:
        raw_npz = np.load(BIDA / sequence / "raw.npz", allow_pickle=False)
        refined_npz = np.load(BIDA / sequence / "refined.npz", allow_pickle=False)
        evaluation = np.load(BIDA / sequence / "evaluation.npz", allow_pickle=False)
        ids = [str(v) for v in raw_npz["frame_ids"]]
        if [str(v) for v in refined_npz["frame_ids"]] != ids:
            raise RuntimeError(f"frame-ID mismatch between raw and refined: {sequence}")
        T = min(len(ids), args.max_frames) if args.max_frames else len(ids)

        raw = raw_npz["raw_disparity"][:T, 0].astype(np.float64)
        raw_valid = raw_npz["raw_valid"][:T, 0].astype(bool)
        bida = refined_npz["disparity"][:T, 0].astype(np.float64)
        gt = evaluation["gt_disparity"][:T, 0].astype(np.float64)
        gt_valid = evaluation["gt_valid"][:T, 0].astype(bool)

        left = [torch.from_numpy(raw_npz["rgb_left"][i:i + 1].copy()).float().to(device) for i in range(T)]
        right = [torch.from_numpy(raw_npz["rgb_right"][i:i + 1].copy()).float().to(device) for i in range(T)]
        frames = [{"index": i,
                   "raw": torch.from_numpy(raw_npz["raw_disparity"][i:i + 1].copy()).float().to(device),
                   "raw_valid": torch.from_numpy(raw_npz["raw_valid"][i:i + 1].copy()).to(device),
                   "rgb": left[i], "right_rgb": right[i]} for i in range(T)]

        def flow(current, past):
            a, b = current["index"], past["index"]
            return flow_model.infer(left[a], left[b]), flow_model.infer(left[b], left[a])

        outputs = dict(drive(adapter, frames, flow))
        ours = np.stack([outputs[i]["disparity"][0, 0].detach().cpu().numpy() for i in range(T)]).astype(np.float64)

        # One support for all three conditions; neither method can influence it.
        support = gt_valid & raw_valid
        for name, prediction in (("raw", raw), ("bidastabilizer", bida), ("tether", ours)):
            row = metrics(prediction, gt, support)
            if row:
                rows.append({"sequence": sequence, "frames": T, "method": name,
                             "causal": {"raw": "n/a", "bidastabilizer": "no", "tether": "yes"}[name]} | row)
        print(f"{sequence}: {T} frames, support {int(support.sum())} px", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "bida_common_support.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Pixel-weighted pooling over sequences, per method.
    pooled = {}
    for name in ("raw", "bidastabilizer", "tether"):
        sub = [r for r in rows if r["method"] == name]
        n = sum(r["pixels"] for r in sub)
        pooled[name] = {m: sum(r[m] * r["pixels"] for r in sub) / n
                        for m in ("EPE", "Bad1", "Bad3", "RMSE", "P95", "InvalidRate")} | {"pixels": n}
    (OUT / "pooled.json").write_text(json.dumps(pooled, indent=2) + "\n")
    (OUT / "run_manifest.json").write_text(json.dumps({
        "project": "ARGOS v2", "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Raw / BiDAStabilizer / TETHER on identical inputs and identical support",
        "input_boundary": str(BIDA), "module": args.module,
        "module_provenance": adapter.describe(),
        "stereo_backbone": "RAFT-Stereo robust, as consumed by the BiDAStabilizer reproduction",
        "support": "GT coverage AND raw validity; prediction-independent",
        "invalid_penalty_px": 1000.0,
        "causality": {"bidastabilizer": "bidirectional, uses future frames",
                      "tether": "strictly causal"},
        "training_performed": False, "threshold_tuning_performed": False,
    }, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "pooled": pooled}, indent=2))


if __name__ == "__main__":
    main()
