#!/usr/bin/env python3
"""Dump the per-pixel arrays a qualitative figure needs, for chosen frames.

The evaluation pipeline stores per-frame metrics and discards the dense predictions
(`dense_predictions_written: false` in every manifest), so the panels have to be
regenerated. Nothing needs re-running at scale: the frozen NPZ boundary already holds
the RGB, the raw disparity and the ground truth for all 4249 D2 frames, and the module's
`step()` already returns the aligned memory as a full tensor.

Frames are not hand-picked for flattery. The script ranks every frame by the signed
change in masked EPE and dumps the requested quantiles, so the selection rule is written
down and includes the worst frame in the sequence by construction.

Nothing is trained and no threshold is tuned.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARGOS = ROOT.parents[1]
BIDA = ARGOS / "ARGOS_hand/external_comparison/results/bidastabilizer_raftstereo_robust/d2_full"
OUT = ROOT.parent / "results" / "qualitative_panels"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence", default="dataset_2_keyframe_4")
    parser.add_argument("--module", default="model_design.comparison.ablation_h4:factory_a2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-frames", type=int, help="limit the driven prefix, for smoke runs")
    args = parser.parse_args()
    # The 142-channel results already live at the bare path. A different head must not
    # land on them: this is how a stale table survived a recanonicalisation once.
    out = OUT if "canonical_h4" in args.module else OUT / "a2"

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

    raw_npz = np.load(BIDA / args.sequence / "raw.npz", allow_pickle=False)
    evaluation = np.load(BIDA / args.sequence / "evaluation.npz", allow_pickle=False)
    T = len(raw_npz["frame_ids"])
    if args.max_frames:
        T = min(T, args.max_frames)

    # Each `npz[key]` inflates the whole array; hoisted so it happens once, not per frame.
    rgb_left, rgb_right = raw_npz["rgb_left"], raw_npz["rgb_right"]
    raw_disparity, raw_valid_all = raw_npz["raw_disparity"], raw_npz["raw_valid"]
    left = [torch.from_numpy(rgb_left[i:i + 1].copy()).float().to(device) for i in range(T)]
    right = [torch.from_numpy(rgb_right[i:i + 1].copy()).float().to(device) for i in range(T)]
    frames = [{"index": i,
               "raw": torch.from_numpy(raw_disparity[i:i + 1].copy()).float().to(device),
               "raw_valid": torch.from_numpy(raw_valid_all[i:i + 1].copy()).to(device),
               "rgb": left[i], "right_rgb": right[i]} for i in range(T)]

    def flow(current, past):
        a, b = current["index"], past["index"]
        return flow_model.infer(left[a], left[b]), flow_model.infer(left[b], left[a])

    outputs = dict(drive(adapter, frames, flow))

    raw = raw_npz["raw_disparity"][:T, 0].astype(np.float64)
    gt = evaluation["gt_disparity"][:T, 0].astype(np.float64)
    support = evaluation["gt_valid"][:T, 0].astype(bool) & raw_npz["raw_valid"][:T, 0].astype(bool)

    def array(index: int, key: str) -> np.ndarray:
        value = outputs[index].get(key)
        return None if value is None else value[0, 0].detach().cpu().numpy().astype(np.float64)

    fused = np.stack([array(i, "disparity") for i in range(T)])
    memory = np.stack([array(i, "aligned_memory") if outputs[i].get("aligned_memory") is not None
                       else np.full_like(raw[i], np.nan) for i in range(T)])

    # Per-frame signed change in masked EPE: negative means the module helped.
    delta = np.full(T, np.nan)
    for i in range(T):
        m = support[i]
        if m.any():
            delta[i] = np.abs(fused[i][m] - gt[i][m]).mean() - np.abs(raw[i][m] - gt[i][m]).mean()
    order = np.argsort(np.where(np.isnan(delta), np.inf, delta))
    ranked = [int(i) for i in order if np.isfinite(delta[i])]
    # Written-down selection rule: best, two typical, and the single worst frame.
    chosen = {"best": ranked[0], "median": ranked[len(ranked) // 2],
              "upper_quartile": ranked[int(0.75 * len(ranked))], "worst": ranked[-1]}

    out.mkdir(parents=True, exist_ok=True)
    panels = {}
    for label, i in chosen.items():
        np.savez_compressed(
            out / f"{args.sequence}_{label}.npz",
            rgb_left=rgb_left[i], raw=raw[i], aligned_memory=memory[i], fused=fused[i],
            gt=gt[i], support=support[i],
            error_raw=np.where(support[i], np.abs(raw[i] - gt[i]), np.nan),
            error_fused=np.where(support[i], np.abs(fused[i] - gt[i]), np.nan),
            # w recovered exactly from the convex form, undefined where the candidates agree
            weight=np.where(np.abs(memory[i] - raw[i]) > 1e-3,
                            (fused[i] - raw[i]) / np.where(np.abs(memory[i] - raw[i]) > 1e-3,
                                                           memory[i] - raw[i], 1.0), np.nan))
        panels[label] = {"frame": i, "frame_id": str(raw_npz["frame_ids"][i]),
                         "delta_epe_px": float(delta[i]),
                         "epe_raw_px": float(np.abs(raw[i][support[i]] - gt[i][support[i]]).mean()),
                         "epe_fused_px": float(np.abs(fused[i][support[i]] - gt[i][support[i]]).mean())}

    (out / "run_manifest.json").write_text(json.dumps({
        "project": "ARGOS v2", "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "dense panels for the qualitative figure",
        "input_boundary": str(BIDA / args.sequence), "sequence": args.sequence, "frames_driven": T,
        "module": args.module, "module_provenance": adapter.describe(),
        "selection_rule": "frames ranked by signed change in masked EPE; best, median, "
                          "upper quartile and worst are dumped, so the worst frame in the "
                          "sequence is included by construction and not by choice",
        "support": "GT coverage AND raw validity; prediction-independent",
        "weight_recovery": "w = (fused - raw) / (memory - raw), undefined where |memory - raw| <= 1e-3",
        "panels": panels, "training_performed": False, "threshold_tuning_performed": False,
    }, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "panels": panels}, indent=2))


if __name__ == "__main__":
    main()
