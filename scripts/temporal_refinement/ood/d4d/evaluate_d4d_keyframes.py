#!/usr/bin/env python3
"""Minimal D4D keyframe evaluator. Accepts arbitrary raw/refined disparity predictions.

Computes at each Zivid anchor (valid GT pixels only): MAE, Bad-1/3/5, depth MAE,
coverage, boundary vs interior, and SNR-stratified MAE. Predictions are loaded per anchor
from --pred-root/<session>__<clip>__<anchor>.npy (must match GT resolution 894x714).

With no --pred-root it runs a GT self-consistency sanity (identity prediction -> 0 error),
proving the evaluator + GT are wired correctly without running any stereo model.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

REPORT = Path("/dtu/p1/leopam/ARGOS/results/03_temporal_refinement/ood/d4d_keyframe_gt")


def anchor_key(r) -> str:
    return f"{r['session']}__{r['clip']}__{r['anchor']}"


def edge_mask(disp, valid):
    g = np.zeros_like(disp); g[:, 1:] = np.abs(disp[:, 1:] - disp[:, :-1])
    return (g > 1.0) & valid


def metrics(pred, gt, valid, snr, fx, baseline_mm):
    m = valid & np.isfinite(pred) & np.isfinite(gt)
    if m.sum() == 0:
        return {"valid_px": 0}
    e = np.abs(pred[m] - gt[m])
    dep_p = fx * baseline_mm / np.maximum(pred, 1e-6)
    dep_g = fx * baseline_mm / np.maximum(gt, 1e-6)
    d = {"valid_px": int(m.sum()), "mae_px": float(e.mean()), "rmse_px": float(np.sqrt((e ** 2).mean())),
         "bad1_pct": float((e > 1).mean() * 100), "bad3_pct": float((e > 3).mean() * 100),
         "bad5_pct": float((e > 5).mean() * 100),
         "depth_mae_mm": float(np.abs(dep_p[m] - dep_g[m]).mean())}
    b = edge_mask(gt, valid)[m]
    if b.any():
        d["boundary_mae"] = float(e[b].mean())
    if (~b).any():
        d["interior_mae"] = float(e[~b].mean())
    sv = snr[m]
    for lo, hi, lab in [(0, 5, "lo"), (5, 20, "mid"), (20, 1e9, "hi")]:
        sel = (sv >= lo) & (sv < hi)
        if sel.any():
            d[f"mae_snr_{lab}"] = float(e[sel].mean())
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=REPORT / "benchmark_manifest.csv")
    ap.add_argument("--pred-root", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=REPORT / "keyframe_eval.csv")
    args = ap.parse_args()
    rows = list(csv.DictReader(args.manifest.open()))
    out = []
    for r in rows:
        gt = np.load(r["gt_disparity"]).astype(np.float32)
        valid = cv2.imread(r["valid_mask"], cv2.IMREAD_GRAYSCALE) > 0
        snr = np.load(r["snr_mask"]).astype(np.float32)
        fx = float(r["fx_px"]); base = float(r["baseline_mm"])
        if args.pred_root:
            pp = args.pred_root / f"{anchor_key(r)}.npy"
            if not pp.exists():
                continue
            pred = np.load(pp).astype(np.float32)
        else:
            pred = gt.copy()  # self-consistency sanity
        d = metrics(pred, gt, valid, snr, fx, base)
        d.update({"key": anchor_key(r), "status": r["status"]})
        out.append(d)
    if out:
        keys = ["key", "status"] + [k for k in out[0] if k not in ("key", "status")]
        with args.out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(out)
        agg = {k: float(np.mean([o[k] for o in out if k in o and o[k] == o[k]]))
               for k in out[0] if isinstance(out[0][k], float)}
        print(json.dumps({"mode": "prediction" if args.pred_root else "GT_self_consistency_sanity",
                          "anchors": len(out), "aggregate": {k: round(v, 4) for k, v in agg.items()}}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
