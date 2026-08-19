#!/usr/bin/env python3
"""Raw / BiDAStabilizer / TETHER on DRENDS, out of domain for both methods.

The published head-to-head runs on SCARED-C D2, which is the split our checkpoint was
selected on while BiDAStabilizer had never seen the domain. The paper declares that and
reads the margin as an upper bound on our advantage rather than an estimate of it. This
is the neutral arena: DRENDS is synthetic colonoscopy, our training data is ex-vivo
porcine tissue, and neither method has seen it.

The shared input comes from BiDA, not from us. Its worker ignores the disparity in the
seed boundary and computes its own RAFT-Stereo robust prediction, writing it to raw.npz;
TETHER is then driven on that exact array. That ordering is forced, not chosen, and it is
why these numbers are not comparable with the middlebury figures in the cross-backbone
table -- the same caveat the D2 comparison already carries.

Scoring is on one prediction-independent support: ground-truth validity AND raw validity,
which neither method influences. An invalid prediction on valid support is penalised at
1000 px rather than dropped, matching the framework everywhere else.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
ARGOS = ROOT.parents[1]
BIDA = ARGOS / "ARGOS_hand/external_comparison/results/bidastabilizer_drends"
OUT = ROOT.parent / "results" / "bida_drends_common_support"
RECORDINGS = ("Vid10_Liver_Med", "Vid11_Liver_High", "Vid12_Pancreas_Ext",
              "Vid13_Pancreas_Med", "Vid14_Pancreas_High")

sys.path.insert(0, str(ROOT / "scripts"))
from compare_bidastabilizer import metrics  # noqa: E402  same invalid-penalty definition


def ground_truth(recording: str, frames: int) -> tuple[np.ndarray, np.ndarray]:
    """DRENDS ground-truth disparity and validity on the canonical grid.

    `frames` is passed down as the record cap rather than used only to slice: loading all
    ~1500 depth maps to keep the first 24 makes a smoke test as slow as a real run, which
    defeats the point of having one.
    """
    from model_design.comparison import drends_evaluation as base
    records, info = base.load_drends_records(recording, frames)
    scale = base.CANONICAL_SIZE[0] / 1280.0
    depth, valid, _coverage = zip(*(base._depth(item["_depth_left"], item["_mask_left"], scale)
                                    for item in records))
    product_mm = info["focal_baseline_native_px_m"] * 1000.0 * scale
    disparity = np.asarray([product_mm / np.maximum(value, 1e-6) for value in depth])[:frames]
    return disparity.astype(np.float64), np.asarray(valid)[:frames].astype(bool)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--module", default="model_design.comparison.canonical_h4_masked:factory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--recordings", nargs="+", default=list(RECORDINGS))
    parser.add_argument("--max-frames", type=int, help="smoke cap; a capped run is not a result")
    # A second module must not land on the first one's record: the default path holds the
    # canonical comparison the paper cites, and overwriting it would leave two different
    # models behind one file name.
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output

    import torch
    for path in (str(ROOT), str(ROOT / "scripts"), str(ARGOS / "ARGOS_FREEZED/src"),
                 str(ARGOS / "ARGOS-V2/scripts")):
        if path not in sys.path:
            sys.path.insert(0, path)
    from argos_freezed.alignment.sea_raft_adapter import SEARAFTFlowAdapter
    from model_design.comparison.run_comparison import drive, load_factory
    from temporal_competitor_metrics import dtce

    device = torch.device(args.device)
    adapter = load_factory(args.module)(device=args.device)
    flow_model = SEARAFTFlowAdapter(device=device)
    rows = []

    for recording in args.recordings:
        raw_npz = np.load(BIDA / recording / "raw.npz", allow_pickle=False)
        refined_npz = np.load(BIDA / recording / "refined.npz", allow_pickle=False)
        ids = [str(v) for v in raw_npz["frame_ids"]]
        if [str(v) for v in refined_npz["frame_ids"]] != ids:
            raise RuntimeError(f"frame-ID mismatch between raw and refined: {recording}")
        count = min(len(ids), args.max_frames) if args.max_frames else len(ids)

        raw = raw_npz["raw_disparity"][:count, 0].astype(np.float64)
        raw_valid = raw_npz["raw_valid"][:count, 0].astype(bool)
        bida = refined_npz["disparity"][:count, 0].astype(np.float64)
        gt, gt_valid = ground_truth(recording, count)
        if gt.shape != raw.shape:
            raise RuntimeError(f"{recording}: ground truth {gt.shape} does not match raw {raw.shape}; "
                               "the boundary and the records are on different grids")

        left = [torch.from_numpy(raw_npz["rgb_left"][i:i + 1].copy()).float().to(device) for i in range(count)]
        right = [torch.from_numpy(raw_npz["rgb_right"][i:i + 1].copy()).float().to(device) for i in range(count)]
        frames = [{"index": i,
                   "raw": torch.from_numpy(raw_npz["raw_disparity"][i:i + 1].copy()).float().to(device),
                   "raw_valid": torch.from_numpy(raw_npz["raw_valid"][i:i + 1].copy()).to(device),
                   "rgb": left[i], "right_rgb": right[i]} for i in range(count)]

        def flow(current, past):
            a, b = current["index"], past["index"]
            return flow_model.infer(left[a], left[b]), flow_model.infer(left[b], left[a])

        outputs = dict(drive(adapter, frames, flow))
        tether = np.asarray([outputs[i]["disparity"][0, 0].detach().cpu().numpy() for i in range(count)],
                            dtype=np.float64)

        # One support for all three, chosen before any prediction is seen and influenced
        # by none of them.
        support = gt_valid & raw_valid

        # Temporal change error beside the geometry. A table that scores temporal methods
        # only per frame cannot say whether either made the sequence more consistent, which
        # is the whole reason both exist. The bidirectional baseline consumes future frames
        # and should win this column; a causal method staying close to it is the strongest
        # form of the comparison, and losing is a finding the metric section already warns
        # about. Not printing it was the only indefensible option.
        stacks = {"raw": raw, "bidastabilizer": bida, "tether": tether}
        temporal = dtce(stacks, gt, support)

        scored = {name: metrics(values, gt, support)
                  | {f"DTCE_k{k}": temporal.get(name, {}).get(k, {}).get("DTCE_px")
                     for k in (1, 2, 4, 8)}
                  for name, values in stacks.items()}
        rows.append({"recording": recording, "frames": count,
                     "pixels": int(support.sum())} | scored)
        current = rows[-1]
        # Flushed, and the partial result is written after every recording. A five-hour
        # run whose only output arrives at the end is unobservable while it matters and
        # unrecoverable if it dies: block buffering behind a pipe hid all progress on the
        # first full run, so there was no way to tell slow from stuck.
        print(f"{recording:<24} n={count:>5} px={current['pixels']:>10,}  "
              f"EPE raw {current['raw']['EPE']:.4f}  BiDA {current['bidastabilizer']['EPE']:.4f}  "
              f"TETHER {current['tether']['EPE']:.4f}", flush=True)
        out.mkdir(parents=True, exist_ok=True)
        (out / "partial.json").write_text(json.dumps(
            {"complete": False, "recordings_done": [r["recording"] for r in rows],
             "capped": args.max_frames is not None, "per_recording": rows}, indent=2) + "\n")

    def pooled(method: str, metric: str) -> float:
        """Pixel-weighted over recordings, so a long recording is not worth a short one."""
        weight = np.array([row["pixels"] for row in rows], dtype=np.float64)
        value = np.array([row[method][metric] for row in rows], dtype=np.float64)
        return float((weight * value).sum() / weight.sum())

    record = {"project": "ARGOS v2", "generated_at": datetime.now(timezone.utc).isoformat(),
              "dataset": "drends", "in_domain_for": [],
              "note": ("neither method has seen DRENDS; the shared raw is BiDA's own RAFT-Stereo "
                       "robust prediction, so these numbers are not comparable with the middlebury "
                       "cross-backbone table"),
              "module": args.module, "capped": args.max_frames is not None,
              "per_recording": rows,
              "pooled": {method: {metric: pooled(method, metric)
                                  for metric in ("EPE", "Bad1", "Bad3", "RMSE",
                                                 "DTCE_k1", "DTCE_k2", "DTCE_k4", "DTCE_k8")
                                  if all(row[method].get(metric) is not None for row in rows)}
                         for method in ("raw", "bidastabilizer", "tether")}}
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(record, indent=2) + "\n")
    (out / "partial.json").unlink(missing_ok=True)   # a leftover partial would outrank nothing
    print("\npooled (pixel-weighted):", flush=True)
    for method in ("raw", "bidastabilizer", "tether"):
        values = record["pooled"][method]
        print(f"  {method:<16} EPE {values['EPE']:.4f}  Bad1 {100 * values['Bad1']:.3f}%  "
              f"Bad3 {100 * values['Bad3']:.3f}%  RMSE {values['RMSE']:.4f}")
    if args.max_frames is not None:
        print("\nCAPPED RUN: smoke test only, not a result.")


if __name__ == "__main__":
    main()
