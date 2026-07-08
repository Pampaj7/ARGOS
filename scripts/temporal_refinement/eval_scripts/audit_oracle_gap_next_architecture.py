#!/usr/bin/env python3
"""Mandatory pre-architecture oracle-gap audit.

Decomposes the unrecovered oracle gap on the 502 selected frames along five axes
(failure mode, raw-error bin, oracle-delta magnitude, boundary distance, temporal
instability) and analyzes HOW the current models (EGBM-v1/v2/v3-window) fail on
oracle-beneficial pixels: missed support, wrong sign, or under-estimated magnitude.
Also computes counterfactual gains: how much gap each failure class costs if fixed
in isolation. All conclusions feed the next architecture choice.

Definitions (per valid pixel):
  raw_err     = |raw - gt|
  oracle_err  = |oracle_all_available - gt|
  gap_px      = raw_err - oracle_err          (positive = oracle-beneficial)
  delta       = oracle - raw                  (the correction the oracle applies)
  pred        = refined_model - raw           (the correction the model applied)
  recovered   = raw_err - |refined - gt|      (positive = model helped)

Recovery fraction within a bin = sum(recovered) / sum(gap_px, positive part only).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "models"))
sys.path.insert(0, str(SCRIPT_DIR / "eval_scripts"))

from train_tiny_refiner_v3_2_hybrid_oracle import load_clips  # noqa: E402
from calibrate_v3_3_failure_mode_thresholds import PATHOLOGICAL_MODES  # noqa: E402
from train_tiny_refiner_v3_1_staged_abstention import make_features_from_raws  # noqa: E402
from experimental_refiner_vx import egbm_refiner  # noqa: E402
from egbm_v2_care_refiner import egbm_v2_care  # noqa: E402
from egbm_v3_care_streaming_refiner import egbm_v3_care_streaming  # noqa: E402


OUT = Path("results/03_temporal_refinement/analysis/oracle_gap_next_architecture")
CKPTS = {
    "egbm_v1": ("results/03_temporal_refinement/training/experimental_refiner_vx_training/checkpoints/best.pt", egbm_refiner),
    "egbm_v2_care": ("results/03_temporal_refinement/training/egbm_v2_care/checkpoints/best.pt", egbm_v2_care),
    "egbm_v3_window": ("results/03_temporal_refinement/training/egbm_v3_care_streaming/checkpoints/best.pt", egbm_v3_care_streaming),
}
SUPPORT_THRESH = 0.25   # |pred| above this counts as "model attempted a correction"
BENEFIT_THRESH = 0.5    # gap_px above this counts as an oracle-beneficial pixel

RAW_ERR_BINS = [(0, 1), (1, 3), (3, 6), (6, 12), (12, 24), (24, np.inf)]
DELTA_BINS = [(0, 0.5), (0.5, 1.5), (1.5, 3), (3, 6), (6, 12), (12, np.inf)]
BOUND_BINS = [(0, 1), (1, 3), (3, 8), (8, np.inf)]
TEMP_BINS = [(0, 0.5), (0.5, 2), (2, 8), (8, np.inf)]


def write_csv(path, rows):
    import csv as _csv
    if not rows:
        path.write_text("")
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


@torch.no_grad()
def predict(model, clip, device, batch=16):
    outs = []
    n = len(clip.frame_ids)
    for s in range(0, n, batch):
        e = min(n, s + batch)
        xs = []
        for i in range(s, e):
            ids = [max(0, i - k) for k in range(4)]
            xf, _a, _b = make_features_from_raws(clip.raws[ids], clip.valids[ids])
            xs.append(xf)
        xb = torch.from_numpy(np.stack(xs)).to(device)
        r = model(xb, 3.0)[2]
        outs.append((torch.from_numpy(clip.raws[s:e]).to(device) + r[:, 0]).cpu().numpy())
    return np.concatenate(outs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    args_ns = p.parse_args()
    device = torch.device(args_ns.device if args_ns.device == "cpu" or torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diagnostics").mkdir(exist_ok=True)

    ns = argparse.Namespace(oracle_min_improvement_px=1.0, oracle_hard_only=False, bad_threshold_px=3.0)
    clips = load_clips(Path("results/03_temporal_refinement/evaluation/distillation_targets_selected_clips"), ns)

    # ---- model predictions ----
    preds = {}
    for name, (ck_path, ctor) in CKPTS.items():
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        m = ctor().to(device)
        m.load_state_dict(ck["model_state_dict"])
        m.eval()
        preds[name] = {c.clip_id: predict(m, c, device) for c in clips}
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- flatten per-pixel arrays with annotations ----
    flat = {k: [] for k in ("raw_err", "oracle_err", "gap", "delta", "bdist", "dt", "mode_idx", "clip_idx")}
    model_pred = {name: [] for name in CKPTS}
    modes = []
    for ci, clip in enumerate(clips):
        if clip.failure_mode not in modes:
            modes.append(clip.failure_mode)
        T = len(clip.frame_ids)
        # boundary distance from GT gradient edges (per frame)
        for t in range(T):
            valid = clip.valids[t] > 0
            if not valid.any():
                continue
            gt = clip.gts[t]
            gy, gx = np.gradient(gt)
            edge = (np.hypot(gx, gy) > 2.0) & valid
            bdist = distance_transform_edt(~edge) if edge.any() else np.full_like(gt, 99.0)
            dt = np.abs(clip.raws[t] - clip.raws[max(0, t - 1)])
            raw_err = np.abs(clip.raws[t] - gt)
            oracle_err = np.abs(clip.oracle[t] - gt)
            v = valid.reshape(-1)
            sel = np.flatnonzero(v)
            flat["raw_err"].append(raw_err.reshape(-1)[sel])
            flat["oracle_err"].append(oracle_err.reshape(-1)[sel])
            flat["gap"].append((raw_err - oracle_err).reshape(-1)[sel])
            flat["delta"].append((clip.oracle[t] - clip.raws[t]).reshape(-1)[sel])
            flat["bdist"].append(bdist.reshape(-1)[sel])
            flat["dt"].append(dt.reshape(-1)[sel])
            flat["mode_idx"].append(np.full(sel.size, modes.index(clip.failure_mode), dtype=np.int8))
            flat["clip_idx"].append(np.full(sel.size, ci, dtype=np.int8))
            for name in CKPTS:
                model_pred[name].append((preds[name][clip.clip_id][t] - clip.raws[t]).reshape(-1)[sel])
    F = {k: np.concatenate(v) for k, v in flat.items()}
    P = {name: np.concatenate(v) for name, v in model_pred.items()}
    n_px = F["raw_err"].size
    gap_pos = np.clip(F["gap"], 0, None)
    total_gap = gap_pos.sum()

    def recovered(name):
        ref_err = np.abs(F["raw_err"] - 0)  # placeholder replaced below
        return None

    rec = {}
    for name in CKPTS:
        ref_err = np.abs((P[name] + 0) - F["delta"] + F["oracle_err"] * np.sign(1))  # not usable; compute directly
    # direct: refined_err = |raw + pred - gt| = |pred - (gt - raw)|; gt - raw = -(raw - gt).
    # raw_err = |raw - gt| => gt - raw = -(raw-gt) but we lost sign. Use delta & oracle_err identity instead:
    # store signed gt residual: gt - raw = delta + (gt - oracle) — sign of (gt - oracle) unknown too.
    # Simplest correct route: recompute refined_err per clip during flatten. Do that now instead.
    rec = {name: [] for name in CKPTS}
    for ci, clip in enumerate(clips):
        T = len(clip.frame_ids)
        for t in range(T):
            valid = clip.valids[t] > 0
            if not valid.any():
                continue
            sel = np.flatnonzero(valid.reshape(-1))
            raw_err = np.abs(clip.raws[t] - clip.gts[t]).reshape(-1)[sel]
            for name in CKPTS:
                ref_err = np.abs(preds[name][clip.clip_id][t] - clip.gts[t]).reshape(-1)[sel]
                rec[name].append(raw_err - ref_err)
    REC = {name: np.concatenate(v) for name, v in rec.items()}

    # ---- helper: binned decomposition ----
    def bin_rows(values, bins, label):
        rows = []
        for lo, hi in bins:
            m = (values >= lo) & (values < hi)
            share = float(gap_pos[m].sum() / max(total_gap, 1e-9))
            row = {label: f"[{lo},{hi})", "px_fraction": float(m.mean()), "gap_share": share,
                   "gap_px_mean": float(gap_pos[m].mean()) if m.any() else 0.0}
            for name in CKPTS:
                denom = max(gap_pos[m].sum(), 1e-9)
                row[f"{name}_recovered_frac"] = float(REC[name][m].sum() / denom)
            rows.append(row)
        return rows

    rows_raw = bin_rows(F["raw_err"], RAW_ERR_BINS, "raw_err_bin")
    rows_delta = bin_rows(np.abs(F["delta"]), DELTA_BINS, "delta_mag_bin")
    rows_bound = bin_rows(F["bdist"], BOUND_BINS, "boundary_dist_bin")
    rows_temp = bin_rows(F["dt"], TEMP_BINS, "temporal_dt_bin")
    write_csv(OUT / "oracle_gap_by_raw_error_bin.csv", rows_raw)
    write_csv(OUT / "oracle_gap_by_oracle_delta_magnitude.csv", rows_delta)
    write_csv(OUT / "oracle_gap_by_boundary_distance.csv", rows_bound)
    write_csv(OUT / "oracle_gap_by_temporal_instability.csv", rows_temp)

    rows_mode = []
    for mi, mode in enumerate(modes):
        m = F["mode_idx"] == mi
        row = {"failure_mode": mode, "px_fraction": float(m.mean()),
               "gap_share": float(gap_pos[m].sum() / max(total_gap, 1e-9)),
               "raw_mae": float(F["raw_err"][m].mean()), "oracle_mae": float(F["oracle_err"][m].mean())}
        for name in CKPTS:
            row[f"{name}_recovered_frac"] = float(REC[name][m].sum() / max(gap_pos[m].sum(), 1e-9))
        rows_mode.append(row)
    write_csv(OUT / "oracle_gap_by_failure_mode.csv", rows_mode)

    # ---- support / sign / magnitude analysis on oracle-beneficial pixels ----
    B = gap_pos > BENEFIT_THRESH  # oracle-beneficial px
    sign_rows, supp_rows, model_rows = [], [], []
    counterfactuals = {}
    for name in CKPTS:
        pred = P[name]
        supported = np.abs(pred) > SUPPORT_THRESH
        right_sign = np.sign(pred) == np.sign(F["delta"])
        mag_ratio = np.abs(pred) / np.clip(np.abs(F["delta"]), 1e-6, None)
        # on beneficial pixels:
        b_sup = supported & B
        supp_recall = float(b_sup.sum() / max(B.sum(), 1))
        sign_acc = float((right_sign & b_sup).sum() / max(b_sup.sum(), 1))
        undersized = right_sign & b_sup & (mag_ratio < 0.5)
        under_frac = float(undersized.sum() / max(B.sum(), 1))
        missed = B & ~supported
        missed_frac = float(missed.sum() / max(B.sum(), 1))
        wrong_sign = B & supported & ~right_sign
        med_ratio = float(np.median(mag_ratio[right_sign & b_sup])) if (right_sign & b_sup).any() else 0.0
        # gap loss attribution (px sums)
        gap_missed = float(gap_pos[missed].sum() / total_gap)
        gap_wrong_sign = float(gap_pos[wrong_sign].sum() / total_gap)
        gap_under = float(gap_pos[undersized].sum() / total_gap)
        recovered_total = float(REC[name].sum() / total_gap)
        sign_rows.append({"model": name, "support_recall_on_beneficial": supp_recall,
                          "sign_accuracy_when_supported": sign_acc,
                          "undersized_frac_of_beneficial(sign_ok,mag<50%)": under_frac,
                          "median_magnitude_ratio_when_sign_ok": med_ratio})
        supp_rows.append({"model": name, "missed_support_frac_of_beneficial": missed_frac,
                          "gap_share_lost_to_missed_support": gap_missed,
                          "gap_share_lost_to_wrong_sign": gap_wrong_sign,
                          "gap_share_lost_to_undersized_magnitude": gap_under,
                          "gap_share_recovered_total": recovered_total})
        # counterfactuals: fix one failure class at a time, holding the rest at model behavior
        ref_err_now = F["raw_err"] - REC[name]
        # fix magnitude: where sign ok & supported & beneficial, assume oracle_err achieved
        cf_mag = ref_err_now.copy()
        fix = right_sign & b_sup
        cf_mag[fix] = F["oracle_err"][fix]
        # fix support: where missed & beneficial, assume oracle_err achieved
        cf_sup = ref_err_now.copy()
        cf_sup[missed] = F["oracle_err"][missed]
        # fix sign: where wrong sign, assume oracle achieved
        cf_sign = ref_err_now.copy()
        cf_sign[wrong_sign] = F["oracle_err"][wrong_sign]
        counterfactuals[name] = {
            "current_recovered_frac": recovered_total,
            "if_magnitude_fixed_recovered_frac": float((F["raw_err"] - cf_mag).sum() / total_gap),
            "if_support_fixed_recovered_frac": float((F["raw_err"] - cf_sup).sum() / total_gap),
            "if_sign_fixed_recovered_frac": float((F["raw_err"] - cf_sign).sum() / total_gap),
        }
        model_rows.append({"model": name, "selected_recovered_gap_frac": recovered_total,
                           "harm_px_frac": float((REC[name] < -0.1).mean()),
                           "benefit_px_frac": float((REC[name] > 0.1).mean())})
        # magnitude calibration curve data for plots
    write_csv(OUT / "correction_sign_accuracy_analysis.csv", sign_rows)
    write_csv(OUT / "correction_support_analysis.csv", supp_rows)
    write_csv(OUT / "current_models_recovered_gap_analysis.csv", model_rows)

    decomposition = {
        "n_valid_pixels": int(n_px),
        "total_gap_px_sum": float(total_gap),
        "selected_raw_mae": float(F["raw_err"].mean()),
        "selected_oracle_mae": float(F["oracle_err"].mean()),
        "beneficial_px_frac(gap>0.5px)": float(B.mean()),
        "gap_share_from_delta_gt_3px": float(gap_pos[np.abs(F["delta"]) >= 3].sum() / total_gap),
        "gap_share_from_delta_gt_6px": float(gap_pos[np.abs(F["delta"]) >= 6].sum() / total_gap),
        "gap_share_within_3px_of_boundary": float(gap_pos[F["bdist"] <= 3].sum() / total_gap),
        "gap_share_high_temporal_dt_gt_2px": float(gap_pos[F["dt"] >= 2].sum() / total_gap),
        "counterfactual_gains": counterfactuals,
        "residual_scale_note": "all EGBM variants bound the additive residual to ~<=2.25px effective (scale=3 tanh, 2-step damped update); compare with gap_share_from_delta_gt_3px",
    }
    (OUT / "oracle_gap_decomposition.json").write_text(json.dumps(decomposition, indent=2) + "\n")

    # ---- diagnostics plots ----
    def bar(rows, key, valkeys, title, fname, rotate=20):
        fig, ax = plt.subplots(figsize=(9, 4.2))
        x = np.arange(len(rows))
        width = 0.8 / max(len(valkeys), 1)
        for i, vk in enumerate(valkeys):
            ax.bar(x + i * width, [r[vk] for r in rows], width, label=vk.replace("_recovered_frac", ""))
        ax.set_xticks(x + width * (len(valkeys) - 1) / 2)
        ax.set_xticklabels([str(r[key]) for r in rows], rotation=rotate, ha="right", fontsize=8)
        ax.set_title(title)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(OUT / "diagnostics" / fname, dpi=130)
        plt.close(fig)

    bar(rows_delta, "delta_mag_bin", ["gap_share"], "Share of total oracle gap by |oracle delta| magnitude", "gap_by_delta_magnitude.png")
    bar(rows_delta, "delta_mag_bin", [f"{n}_recovered_frac" for n in CKPTS], "Recovered fraction by |delta| bin", "recovered_by_delta_magnitude.png")
    bar(rows_mode, "failure_mode", ["gap_share"] + [f"{n}_recovered_frac" for n in CKPTS], "Gap share and recovery by failure mode", "gap_by_failure_mode.png", rotate=25)
    bar(rows_raw, "raw_err_bin", ["gap_share"] + [f"{n}_recovered_frac" for n in CKPTS], "Gap and recovery by raw-error bin", "gap_by_raw_error_bin.png")
    bar(rows_bound, "boundary_dist_bin", ["gap_share"] + [f"{n}_recovered_frac" for n in CKPTS], "Gap and recovery by boundary distance", "gap_by_boundary_distance.png")
    bar(rows_temp, "temporal_dt_bin", ["gap_share"] + [f"{n}_recovered_frac" for n in CKPTS], "Gap and recovery by temporal instability", "gap_by_temporal_instability.png")

    # magnitude calibration: mean |pred| per |delta| bin
    fig, ax = plt.subplots(figsize=(7, 5))
    centers = []
    for name in CKPTS:
        xs, ys = [], []
        for lo, hi in DELTA_BINS:
            m = (np.abs(F["delta"]) >= lo) & (np.abs(F["delta"]) < (hi if np.isfinite(hi) else 64)) & B
            if m.sum() > 100:
                xs.append(np.abs(F["delta"])[m].mean())
                ys.append(np.abs(P[name])[m].mean())
        ax.plot(xs, ys, marker="o", label=name)
        centers = xs
    if centers:
        ax.plot([0, max(centers)], [0, max(centers)], "k--", lw=1, label="perfect calibration")
    ax.set_xlabel("|oracle delta| (px, bin mean)")
    ax.set_ylabel("|model correction| (px, mean)")
    ax.set_title("Correction magnitude calibration on oracle-beneficial pixels")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "diagnostics" / "magnitude_calibration.png", dpi=130)
    plt.close(fig)

    (OUT / "README.md").write_text(
        "# Oracle-Gap Audit (pre-architecture)\n\n"
        "Empirical decomposition of the ~80% unrecovered oracle gap on the 502 selected frames,\n"
        "plus support/sign/magnitude failure analysis of EGBM-v1/v2/v3(window) and counterfactual\n"
        "gains from fixing each failure class in isolation. See `oracle_gap_decomposition.json`\n"
        "for headline numbers and `diagnostics/` for plots. Produced by\n"
        "`scripts/temporal_refinement/eval_scripts/audit_oracle_gap_next_architecture.py`.\n"
    )
    print(json.dumps(decomposition, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
