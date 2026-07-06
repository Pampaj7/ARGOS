#!/usr/bin/env python3
"""D4D zero-shot refiner evaluation. Reuses the OOD harness (frame_metrics, run_model,
model registry) on the D4D causal-context shards, adds D4D stratification + safety.

Raw S2M2-S is the baseline; SCARED-trained refiners are applied zero-shot (no D4D tuning).
Establishes the 0%-target-data point for the few-shot adaptation study.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/dtu/p1/leopam/ARGOS")
for p in ("scripts/temporal_refinement/ood/eval", "scripts/temporal_refinement",
          "scripts/temporal_refinement/models", "scripts/temporal_refinement/eval_scripts"):
    sys.path.insert(0, str(ROOT / p))
from evaluate_ood_refiners import frame_metrics, run_model, load_samples_and_shards, EPS  # noqa: E402
from model_registry import build_registry  # noqa: E402

OUT = ROOT / "results/03_temporal_refinement/ood/d4d_s2m2_zero_shot"
RAW_ERR_BINS = [0, 1, 3, 6, 12, 1e9]
RAW_ERR_LABELS = ["<1", "1-3", "3-6", "6-12", ">12"]
MAIN = ["raw", "EGBM-v1", "EGBM-v2-CARE", "EGBM-v3-CARE-S"]  # paper-relevant


def wcsv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(""); return
    keys = sorted({k for r in rows for k in r})
    lead = [k for k in ("model", "specimen", "session", "quality", "convention", "bin", "anchor_id") if k in keys]
    keys = lead + [k for k in keys if k not in lead]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def wmean(dicts, key):
    v = [d[key] for d in dicts if key in d and d[key] == d[key]]
    return float(np.mean(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=Path, default=OUT / "d4d_index.csv")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    samples, shards, index_rows = load_samples_and_shards(args.index)
    meta = {r["sequence_id"]: r for r in index_rows}
    registry = build_registry()
    runs = [("raw", None)] + [(e.name, e) for e in registry]
    print(f"D4D zero-shot: {len(samples)} anchors, {len(registry)} refiners + raw, device={device}")

    anchor_rows = []
    raw_err_acc = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0.0]))  # model->bin->[raw_e_sum, ref_e_sum, n]
    per_model_arrays = {}   # model -> list of (d, meta) for aggregation
    blocked = []
    for name, entry in runs:
        try:
            rec = run_model(entry, samples, shards, device)
        except Exception as e:  # e.g. MPC/CPV large-proposal path needs even dims at D4D grid
            blocked.append({"model": name, "reason": f"{type(e).__name__}: {str(e)[:120]}"})
            print(f"  {name:15s} BLOCKED at D4D resolution: {str(e)[:80]}")
            continue
        fdicts = []
        for (d, raw, refined, gt, valid, edge, aux) in rec:
            sid = d["sequence_id"]; m = meta[sid]
            row = {"model": name, "anchor_id": sid, "specimen": m["specimen"], "session": m["session"],
                   "quality": m["quality"], "convention": m["convention"], **d}
            anchor_rows.append(row); fdicts.append(row)
            # pixel-level raw-error-bin accumulation
            vm = valid
            er = np.abs(raw - gt); ef = np.abs(refined - gt)
            idx = np.digitize(er[vm], RAW_ERR_BINS)
            for b in range(1, len(RAW_ERR_BINS)):
                sel = idx == b
                if sel.any():
                    acc = raw_err_acc[name][RAW_ERR_LABELS[b - 1]]
                    acc[0] += er[vm][sel].sum(); acc[1] += ef[vm][sel].sum(); acc[2] += int(sel.sum())
        fdicts = [d for d in fdicts if d.get("valid_px", 0) > 0]  # exclude empty-valid anchors from stats
        per_model_arrays[name] = fdicts
        mm = {"model": name, "anchors": len(fdicts),
              "raw_mae": wmean(fdicts, "raw_mae"), "refined_mae": wmean(fdicts, "refined_mae"),
              "delta_mae": wmean(fdicts, "delta_mae"), "refined_bad1": wmean(fdicts, "refined_bad1"),
              "refined_bad3": wmean(fdicts, "refined_bad3"), "refined_bad5": wmean(fdicts, "refined_bad5"),
              "new_bad3_pct": wmean(fdicts, "new_bad3_pct_of_rawgood"),
              "harmful_rate": wmean(fdicts, "harmful_rate"), "beneficial_rate": wmean(fdicts, "beneficial_rate"),
              "modified_pixel_ratio": wmean(fdicts, "modified_pixel_ratio"),
              "boundary_refined_mae": wmean(fdicts, "boundary_refined_mae"),
              "runtime_ms": wmean(fdicts, "runtime_ms"),
              "pct_anchors_improved": float(np.mean([1.0 if d["delta_mae"] > EPS else 0.0 for d in fdicts]) * 100),
              "pct_anchors_harmed": float(np.mean([1.0 if d["delta_mae"] < -EPS else 0.0 for d in fdicts]) * 100)}
        print(f"  {name:15s} refMAE={mm['refined_mae']:.3f} dMAE={mm['delta_mae']:+.3f} "
              f"bad3={mm['refined_bad3']:.1f} newBad3={mm['new_bad3_pct']:.1f} "
              f"harmful={mm['harmful_rate']:.2f} improved={mm['pct_anchors_improved']:.0f}%")

    wcsv(args.out / "metrics" / "anchor_metrics.csv", anchor_rows)
    if blocked:
        wcsv(args.out / "metrics" / "blocked_models.csv", blocked)

    # aggregate + stratified
    def strat(field):
        rows = []
        for name in per_model_arrays:
            groups = defaultdict(list)
            for d in per_model_arrays[name]:
                groups[d[field]].append(d)
            for gv, ds in sorted(groups.items()):
                rows.append({"model": name, field: gv, "n": len(ds),
                             "raw_mae": round(wmean(ds, "raw_mae"), 3), "refined_mae": round(wmean(ds, "refined_mae"), 3),
                             "delta_mae": round(wmean(ds, "delta_mae"), 3), "refined_bad3": round(wmean(ds, "refined_bad3"), 2),
                             "new_bad3_pct": round(wmean(ds, "new_bad3_pct_of_rawgood"), 2),
                             "harmful_rate": round(wmean(ds, "harmful_rate"), 3)})
        return rows
    wcsv(args.out / "metrics" / "specimen_metrics.csv", strat("specimen"))
    wcsv(args.out / "metrics" / "session_metrics.csv", strat("session"))
    wcsv(args.out / "metrics" / "quality_status_metrics.csv", strat("quality"))
    wcsv(args.out / "metrics" / "tf_convention_metrics.csv", strat("convention"))

    # raw-error-bin (pixel level): does the refiner harm raw-good pixels?
    reb = []
    for name in raw_err_acc:
        for lab in RAW_ERR_LABELS:
            a = raw_err_acc[name][lab]
            if a[2] > 0:
                reb.append({"model": name, "raw_error_bin": lab, "pixels": int(a[2]),
                            "raw_mae": round(a[0] / a[2], 3), "refined_mae": round(a[1] / a[2], 3),
                            "delta_mae": round((a[0] - a[1]) / a[2], 3)})
    wcsv(args.out / "metrics" / "raw_error_bin_metrics.csv", reb)

    # model comparison + primary table
    comp = []
    for name in per_model_arrays:
        fd = per_model_arrays[name]
        comp.append({"model": name, "anchors": len(fd),
                     "disp_MAE": round(wmean(fd, "refined_mae"), 3), "Bad1": round(wmean(fd, "refined_bad1"), 2),
                     "Bad3": round(wmean(fd, "refined_bad3"), 2), "Bad5": round(wmean(fd, "refined_bad5"), 2),
                     "boundary_MAE": round(wmean(fd, "boundary_refined_mae"), 3),
                     "new_Bad3": round(wmean(fd, "new_bad3_pct_of_rawgood"), 2),
                     "harmful_rate": round(wmean(fd, "harmful_rate"), 3),
                     "pct_improved": round(float(np.mean([d["delta_mae"] > EPS for d in fd]) * 100), 1),
                     "modified_ratio": round(wmean(fd, "modified_pixel_ratio"), 3),
                     "runtime_ms": round(wmean(fd, "runtime_ms"), 2)})
    wcsv(args.out / "metrics" / "aggregate_metrics.csv", comp)
    wcsv(args.out / "metrics" / "correction_safety_metrics.csv",
         [{"model": c["model"], "new_Bad3": c["new_Bad3"], "harmful_rate": c["harmful_rate"],
           "pct_improved": c["pct_improved"], "modified_ratio": c["modified_ratio"]} for c in comp])
    order = {n: i for i, n in enumerate(MAIN)}
    primary = sorted([c for c in comp if c["model"] in MAIN], key=lambda c: order[c["model"]])
    (args.out / "reports").mkdir(parents=True, exist_ok=True)
    (args.out / "reports" / "primary_table.json").write_text(json.dumps(primary, indent=2) + "\n")
    (args.out / "model_comparison.json").write_text(json.dumps(comp, indent=2) + "\n")
    (args.out / "aggregate_summary.json").write_text(json.dumps(
        {"n_anchors": len(samples), "models": [c["model"] for c in comp], "primary": primary,
         "device": str(device)}, indent=2) + "\n")
    print("\nPRIMARY:", json.dumps(primary, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
