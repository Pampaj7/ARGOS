#!/usr/bin/env python3
"""Aggregate the VDPP confirmation study: factorial SCARED table (mean±std over seeds),
D4D per-clip/per-specimen + paired bootstrap CI vs raw, TGM-weight sweep, decision JSON.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

B = Path("/dtu/p1/leopam/ARGOS/results/03_temporal_refinement/vdpp_style_causal_confirmation")
GEO_KEYS = ["refined_mae", "delta_mae", "refined_bad3", "new_bad3_pct_of_rawgood", "harmful_rate", "modified_pixel_ratio"]
TMP_KEYS = ["tgm_error", "terr_jitter", "hf_error_energy", "boundary_tgm"]


def load_runs():
    runs = []
    for d in sorted((B / "runs").glob("*/config.json")):
        runs.append(json.loads(d.read_text()))
    return runs


def factorial_and_meanstd(runs):
    # group by (temporal_input_mode, loss_mode, lam_tgm)
    g = defaultdict(list)
    for c in runs:
        g[(c["temporal_input_mode"], c["loss_mode"], c["lam_tgm"])].append(c)
    per_seed, meanstd = [], []
    for (tm, lm, lam), cs in sorted(g.items()):
        for c in cs:
            row = {"temporal_input_mode": tm, "loss_mode": lm, "lam_tgm": lam, "seed": c["run_id"].split("seed")[-1]}
            row.update({f"geo_{k}": c["test_geometric"].get(k) for k in GEO_KEYS})
            row.update({f"tmp_{k}": c["test_temporal"].get(k) for k in TMP_KEYS})
            per_seed.append(row)
        ms = {"temporal_input_mode": tm, "loss_mode": lm, "lam_tgm": lam, "n_seeds": len(cs)}
        for k in GEO_KEYS:
            v = [c["test_geometric"][k] for c in cs if k in c["test_geometric"]]
            ms[f"geo_{k}_mean"] = round(float(np.mean(v)), 4); ms[f"geo_{k}_std"] = round(float(np.std(v)), 4)
        for k in TMP_KEYS:
            v = [c["test_temporal"][k] for c in cs if k in c["test_temporal"]]
            ms[f"tmp_{k}_mean"] = round(float(np.mean(v)), 4); ms[f"tmp_{k}_std"] = round(float(np.std(v)), 4)
        meanstd.append(ms)
    return per_seed, meanstd


def d4d_bootstrap():
    """Per-clip paired diff vs raw, per temporal variant; bootstrap CI over clips."""
    per_clip, per_spec, ci = [], [], {}
    for vd in sorted((B / "d4d").glob("*")):
        tm = vd.name.split("__")[0]
        tcsv = vd / "d4d_temporal_metrics.csv"
        if not tcsv.exists():
            continue
        rows = list(csv.DictReader(tcsv.open()))
        raw = {(r["specimen"], r["clip"]): r for r in rows if r["config"] == "raw"}
        ref = {(r["specimen"], r["clip"]): r for r in rows if r["config"] == "vdpp_tgm"}
        diffs = defaultdict(list)
        for key in ref:
            if key not in raw:
                continue
            for m in ("mc_inconsistency", "hf_energy", "depth_mc_mm", "boundary_mc"):
                try:
                    d = float(ref[key][m]) - float(raw[key][m])
                except (ValueError, KeyError):
                    continue
                diffs[m].append(d)
                per_clip.append({"variant": tm, "run": vd.name, "specimen": key[0], "clip": key[1],
                                 "metric": m, "raw": raw[key][m], "vdpp": ref[key][m], "diff": round(d, 4)})
        civar = {}
        for m, ds in diffs.items():
            ds = np.array(ds)
            boot = [np.mean(np.random.choice(ds, len(ds), replace=True)) for _ in range(2000)]
            civar[m] = {"mean_diff": round(float(ds.mean()), 4), "ci95": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
                        "pct_clips_improved": round(float((ds < 0).mean() * 100), 1), "n_clips": len(ds)}
        ci[vd.name] = civar
    return per_clip, ci


def main():
    runs = load_runs()
    per_seed, meanstd = factorial_and_meanstd(runs)
    def w(name, rows):
        if not rows: return
        with (B / name).open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    w("per_seed_scared_metrics.csv", per_seed)
    w("scared_mean_std.csv", meanstd)
    w("factorial_ablation.csv", meanstd)
    # tgm weight sweep table (full_history)
    sweep = [m for m in meanstd if m["temporal_input_mode"] == "full_history" and m["loss_mode"] == "spatial_plus_tgm"]
    w("tgm_weight_sweep.csv", sweep)
    per_clip, ci = d4d_bootstrap()
    w("d4d_per_clip_metrics.csv", per_clip)
    (B / "d4d_bootstrap_ci.json").write_text(json.dumps(ci, indent=2) + "\n")
    # per specimen
    spec = defaultdict(lambda: defaultdict(list))
    for r in per_clip:
        spec[(r["variant"], r["specimen"], r["metric"])]["diff"].append(float(r["diff"]))
    ps = [{"variant": k[0], "specimen": k[1], "metric": k[2], "mean_diff": round(float(np.mean(v["diff"])), 4),
           "n": len(v["diff"])} for k, v in sorted(spec.items())]
    w("d4d_per_specimen_metrics.csv", ps)

    # decision
    def get(tm, lm, lam, key):
        for m in meanstd:
            if m["temporal_input_mode"] == tm and m["loss_mode"] == lm and m["lam_tgm"] == lam:
                return m.get(key)
        return None
    full_tgm = get("full_history", "spatial_plus_tgm", 1.0, "tmp_tgm_error_mean")
    full_tgm_std = get("full_history", "spatial_plus_tgm", 1.0, "tmp_tgm_error_std")
    full_sp = get("full_history", "spatial_only", 1.0, "tmp_tgm_error_mean")
    cur_tgm = get("current_frame_only", "spatial_plus_tgm", 1.0, "tmp_tgm_error_mean")
    cur_tgm_std = get("current_frame_only", "spatial_plus_tgm", 1.0, "tmp_tgm_error_std")
    shf_tgm = get("shuffled_history", "spatial_plus_tgm", 1.0, "tmp_tgm_error_mean")
    shf_tgm_std = get("shuffled_history", "spatial_plus_tgm", 1.0, "tmp_tgm_error_std")
    # per-seed raw values for overlap check (std-based, not just mean comparison)
    per_seed_vals = defaultdict(list)
    for c in runs:
        if c["lam_tgm"] != 1.0:
            continue  # core-matrix gates use lam=1.0 only; sweep lambdas are a separate analysis
        per_seed_vals[(c["temporal_input_mode"], c["loss_mode"])].append(c["test_temporal"]["tgm_error"])
    full_vals = per_seed_vals.get(("full_history", "spatial_plus_tgm"), [])
    shf_vals = per_seed_vals.get(("shuffled_history", "spatial_plus_tgm"), [])
    cur_vals = per_seed_vals.get(("current_frame_only", "spatial_plus_tgm"), [])
    # overlap: does full's worst seed beat the ablation's best seed? if not, margin is not robust.
    full_vs_shf_robust = bool(full_vals and shf_vals and max(full_vals) < min(shf_vals))
    full_vs_cur_robust = bool(full_vals and cur_vals and max(full_vals) < min(cur_vals))
    fullci = ci.get("full_history__seed0", {})
    d4d_sign_stable = None
    # sign stability across seeds for full_history mc_inconsistency
    signs = []
    for s in (0, 1, 2):
        c = ci.get(f"full_history__seed{s}", {}).get("mc_inconsistency", {})
        if c:
            signs.append(c["mean_diff"] < 0)
    d4d_sign_stable = (len(signs) >= 2 and all(signs))
    gate = {
        "g1_full_tgm_beats_full_spatial": {"full_tgm_mean": full_tgm, "full_spatial_mean": full_sp,
            "pass": full_tgm is not None and full_sp is not None and full_tgm < full_sp},
        "g2_full_tgm_beats_current_tgm": {"full_mean": full_tgm, "full_std": full_tgm_std,
            "current_mean": cur_tgm, "current_std": cur_tgm_std, "full_seeds": full_vals, "current_seeds": cur_vals,
            "pass_on_mean": full_tgm is not None and cur_tgm is not None and full_tgm < cur_tgm,
            "robust_no_overlap": full_vs_cur_robust},
        "g3_full_tgm_beats_shuffled_tgm": {"full_mean": full_tgm, "full_std": full_tgm_std,
            "shuffled_mean": shf_tgm, "shuffled_std": shf_tgm_std, "full_seeds": full_vals, "shuffled_seeds": shf_vals,
            "pass_on_mean": full_tgm is not None and shf_tgm is not None and full_tgm < shf_tgm,
            "robust_no_overlap": full_vs_shf_robust,
            "note": "full_history+TGM's worst seed (%.4f) is WORSE than shuffled_history+TGM's best seed (%s) -> means differ but per-seed distributions overlap heavily" % (max(full_vals) if full_vals else float('nan'), min(shf_vals) if shf_vals else None)},
        "g4_d4d_sign_stable_across_seeds": {"mc_inconsistency_improves_all_seeds": d4d_sign_stable},
        "g5_no_identity_collapse_or_unsafe": "see d4d anchor modified/harmful",
    }
    mean_pass = all(gate[k].get("pass_on_mean", gate[k].get("pass")) for k in ("g1_full_tgm_beats_full_spatial", "g2_full_tgm_beats_current_tgm", "g3_full_tgm_beats_shuffled_tgm"))
    robust_pass = mean_pass and full_vs_shf_robust and full_vs_cur_robust
    if robust_pass:
        gate["VERDICT"] = "CONFIRMED temporal usage (robust: no per-seed overlap)"
    elif mean_pass:
        gate["VERDICT"] = ("MARGINAL — means favor full_history+TGM but per-seed distributions "
                           "overlap with the shuffled/current-frame ablations (high seed variance, "
                           "esp. full_history+TGM std=%.3f driven by one high-error seed). "
                           "Do NOT claim confirmed temporal usage from this pilot scale; the original "
                           "pilot's clean separation was an artefact of the loss confound." % (full_tgm_std or 0))
    else:
        gate["VERDICT"] = "NOT CONFIRMED — TGM regularizes but history not exploited"
    (B / "temporal_usage_decision.json").write_text(json.dumps(gate, indent=2) + "\n")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
