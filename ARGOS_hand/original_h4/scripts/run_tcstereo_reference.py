#!/usr/bin/env python3
"""TC-Stereo (ECCV 2024) as the integrated-architecture reference row.

TC-Stereo is the closest published temporal stereo method to ours in intent and the
furthest in construction: it is causal, but it *replaces* the stereo network and carries
its own hidden state, so it cannot be attached to a frozen backbone. Its temporal
propagation (`core/tc_stereo.py:119-137`) additionally warps the previous disparity by a
rigid 6-DoF reprojection built from per-frame camera poses, intrinsics and baseline.
SCARED-C *does* supply per-frame poses -- correcting them is what the dataset is for, and
this file previously claimed the opposite -- so the temporal path is runnable and only the
rigid-scene assumption remains questionable on deforming tissue, which is a result to
measure rather than a reason to skip.

What *this* run reports is the frame path only: `params=None` on every frame, its
single-frame stereo path, an integrated 16.7M-parameter architecture evaluated zero-shot
and not a temporal rival. The temporal run is a separate entry point because it needs
per-frame state carried across the sequence -- K, T, previous_T, baseline, last_disp,
last_net_list and fmap1 -- rather than a flag on this one.

The point of the row is to answer the obvious reviewer question ("why refine a frozen
network instead of using a better one?") on identical frames and identical ground truth.
It consumes the same frozen NPZ boundary the BiDAStabilizer comparison already wrote, so
no frame, no crop and no ground-truth pixel differs between the methods being compared.

Nothing is trained and no threshold is tuned.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARGOS = ROOT.parents[1]
BIDA = ARGOS / "ARGOS_hand/external_comparison/results/bidastabilizer_raftstereo_robust/d2_full"
TCSTEREO = ARGOS / "external/comparison_methods/Temporally-Consistent-Stereo-Matching"
OUT = ROOT.parent / "results" / "tcstereo_reference"
SEQUENCES = ("dataset_2_keyframe_2", "dataset_2_keyframe_3", "dataset_2_keyframe_4")

# The load configuration found by search: 0 missing / 0 unexpected keys on all three
# released checkpoints. Recorded in external/comparison_methods/STATUS.md.
TC_ARGS = dict(hidden_dims=[128] * 3, corr_implementation="reg", corr_levels=4,
               corr_radius=4, n_downsample=2, slow_fast_gru=False, n_gru_layers=3,
               mixed_precision=False, init_thres=0.5, context_norm="instance",
               shared_backbone=True)

INVALID_PENALTY_PX = 1000.0


def metrics(prediction: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict:
    """EPE, bad-pixel rates and RMSE on a fixed support.

    An invalid prediction on valid support is penalised, never dropped: a method does not
    get to improve its average by declining to answer.
    """
    valid = mask & np.isfinite(gt) & (gt > 0)
    if not valid.any():
        return {}
    error = np.abs(prediction[valid] - gt[valid])
    finite = np.isfinite(prediction[valid]) & (prediction[valid] > 0)
    error = np.where(finite, error, INVALID_PENALTY_PX)
    return {"pixels": int(valid.sum()),
            "EPE": float(error.mean()),
            "Bad1": float((error > 1).mean()),
            "Bad3": float((error > 3).mean()),
            "RMSE": float(np.sqrt((error ** 2).mean())),
            "P95": float(np.percentile(error, 95)),
            "InvalidRate": float((~finite).mean())}


def load_tcstereo(checkpoint: Path, device):
    import torch
    if str(TCSTEREO) not in sys.path:
        sys.path.insert(0, str(TCSTEREO))
    from core.tc_stereo import TCStereo
    model = TCStereo(SimpleNamespace(**TC_ARGS))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weights = state["model"] if "model" in state else state
    weights = {k.replace("module.", "", 1): v for k, v in weights.items()}
    missing, unexpected = model.load_state_dict(weights, strict=True), None
    return model.to(device).eval().requires_grad_(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=str(TCSTEREO / "checkpoints/sceneflow.pth"),
                        help="released TC-Stereo weights; sceneflow is its most general pretraining")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iters", type=int, default=32, help="its evaluation default")
    parser.add_argument("--sequences", nargs="+", default=list(SEQUENCES))
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()

    import torch
    if str(TCSTEREO) not in sys.path:
        sys.path.insert(0, str(TCSTEREO))
    from core.utils.utils import InputPadder

    device = torch.device(args.device)
    model = load_tcstereo(Path(args.checkpoint), device)
    rows = []

    for sequence in args.sequences:
        raw_npz = np.load(BIDA / sequence / "raw.npz", allow_pickle=False)
        refined_npz = np.load(BIDA / sequence / "refined.npz", allow_pickle=False)
        evaluation = np.load(BIDA / sequence / "evaluation.npz", allow_pickle=False)
        T = len(raw_npz["frame_ids"])
        if args.max_frames:
            T = min(T, args.max_frames)

        raw = raw_npz["raw_disparity"][:T, 0].astype(np.float64)
        raw_valid = raw_npz["raw_valid"][:T, 0].astype(bool)
        bida = refined_npz["disparity"][:T, 0].astype(np.float64)
        gt = evaluation["gt_disparity"][:T, 0].astype(np.float64)
        gt_valid = evaluation["gt_valid"][:T, 0].astype(bool)

        tc = np.empty_like(raw)
        for i in range(T):
            left = torch.from_numpy(raw_npz["rgb_left"][i:i + 1].copy()).float().to(device)
            right = torch.from_numpy(raw_npz["rgb_right"][i:i + 1].copy()).float().to(device)
            if float(left.max()) <= 1.5:          # the boundary stores [0,1]; TC-Stereo expects [0,255]
                left, right = left * 255.0, right * 255.0
            padder = InputPadder(left.shape, divis_by=32)
            left_p, right_p = padder.pad(left, right)
            with torch.inference_mode():
                output = model(left_p, right_p, iters=args.iters, params=None, test_mode=True)
            disparity = padder.unpad(-output["flow"])          # it emits negative flow
            tc[i] = disparity[0, 0].float().cpu().numpy()
            if i % 100 == 0:
                print(f"  {sequence} {i}/{T}", flush=True)

        # Two supports, both prediction-independent, reported side by side because they
        # answer different questions. `gt` is fair to a method that never sees the frozen
        # backbone; `gt_and_raw` is the common support on which BiDA was scored.
        for support_name, support in (("gt", gt_valid), ("gt_and_raw", gt_valid & raw_valid)):
            for name, prediction in (("raw_raftstereo_robust", raw), ("bidastabilizer", bida),
                                     ("tcstereo_sceneflow", tc)):
                row = metrics(prediction, gt, support)
                if row:
                    rows.append({"sequence": sequence, "frames": T, "support": support_name,
                                 "method": name, "causal": {"raw_raftstereo_robust": "n/a",
                                                            "bidastabilizer": "no",
                                                            "tcstereo_sceneflow": "yes"}[name],
                                 "frozen_backbone": {"raw_raftstereo_robust": "n/a",
                                                     "bidastabilizer": "yes",
                                                     "tcstereo_sceneflow": "no"}[name]} | row)
        print(f"{sequence}: {T} frames done", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "tcstereo_reference.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    pooled = {}
    for support_name in ("gt", "gt_and_raw"):
        for name in ("raw_raftstereo_robust", "bidastabilizer", "tcstereo_sceneflow"):
            sub = [r for r in rows if r["method"] == name and r["support"] == support_name]
            n = sum(r["pixels"] for r in sub)
            pooled[f"{support_name}/{name}"] = {
                m: sum(r[m] * r["pixels"] for r in sub) / n
                for m in ("EPE", "Bad1", "Bad3", "RMSE", "P95", "InvalidRate")} | {"pixels": n}
    (OUT / "pooled.json").write_text(json.dumps(pooled, indent=2) + "\n")
    (OUT / "run_manifest.json").write_text(json.dumps({
        "project": "ARGOS v2", "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "TC-Stereo as the integrated-architecture reference on identical frames and GT",
        "input_boundary": str(BIDA),
        "tcstereo_checkpoint": args.checkpoint, "tcstereo_args": TC_ARGS, "iters": args.iters,
        "temporal_path_disabled": {
            "params": None,
            "reason": "its propagation warps the previous disparity by a rigid 6-DoF "
                      "reprojection from per-frame pose, intrinsics and baseline; SCARED-C "
                      "does supply per-frame poses, so this run reports the frame path by "
                      "choice of entry point, not by unavailability",
            "consequence": "this is TC-Stereo's single-frame stereo path, reported as an "
                           "integrated-architecture reference and not as a temporal rival"},
        "invalid_penalty_px": INVALID_PENALTY_PX,
        "training_performed": False, "threshold_tuning_performed": False,
    }, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "pooled": pooled}, indent=2))


if __name__ == "__main__":
    main()
