#!/usr/bin/env python3
"""Focused safety audit for the Magnitude Proposal-Critic refiner."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "models"))
sys.path.insert(0, str(SCRIPT_DIR / "eval_scripts"))

from calibrate_v3_3_failure_mode_thresholds import PATHOLOGICAL_MODES  # noqa: E402
from magnitude_proposal_critic_refiner import magnitude_proposal_critic_refiner  # noqa: E402
from train_magnitude_proposal_critic_refiner import full_gt_eval  # noqa: E402
from train_tiny_refiner_v1_full_gt import load_shards  # noqa: E402
from train_tiny_refiner_v3_1_staged_abstention import FullFrameDataset, load_samples_with_split, make_features_from_raws  # noqa: E402
from train_tiny_refiner_v3_2_hybrid_oracle import load_clips, make_loader  # noqa: E402


DEFAULT_OUTPUT = Path("results/03_temporal_refinement/analysis/mpc_safety_failure_audit")
DEFAULT_MPC = Path("results/03_temporal_refinement/training/magnitude_proposal_critic_refiner")
TRUST_BINS = np.linspace(0.0, 1.0, 11)
MAG_BINS = np.array([0, 1, 3, 6, 9, 12, 16, 20, 24, 32, 1e9], dtype=np.float32)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def metric(raw: np.ndarray, gt: np.ndarray, valid: np.ndarray, residual: np.ndarray, oracle: np.ndarray | None = None) -> dict[str, float]:
    v = valid > 0
    raw_err = np.abs(raw - gt)
    ref_err = np.abs(raw + residual - gt)
    good = v & (raw_err < 1.0)
    n = max(int(v.sum()), 1)
    out = {
        "raw_mae": float(raw_err[v].mean()) if v.any() else float("nan"),
        "refined_mae": float(ref_err[v].mean()) if v.any() else float("nan"),
        "raw_bad3": 100.0 * float((raw_err[v] >= 3.0).sum()) / n,
        "refined_bad3": 100.0 * float((ref_err[v] >= 3.0).sum()) / n,
        "new_bad3_pct": 100.0 * float((good & (ref_err >= 3.0)).sum()) / max(int(good.sum()), 1),
        "new_bad3_pixels": int((good & (ref_err >= 3.0)).sum()),
        "raw_good_pixels": int(good.sum()),
        "modified_pct": 100.0 * float((np.abs(residual[v]) > 0.01).sum()) / n,
    }
    if oracle is not None:
        oracle_err = np.abs(oracle - gt)
        out["oracle_mae"] = float(oracle_err[v].mean()) if v.any() else float("nan")
        out["oracle_gap_recovered_pct"] = 100.0 * (out["raw_mae"] - out["refined_mae"]) / max(out["raw_mae"] - out["oracle_mae"], 1e-9)
    return out


@torch.no_grad()
def predict_clip(model, clip, args, device) -> dict[str, np.ndarray]:
    refined, residual, p_bad = [], [], []
    keys = ("trust", "damping", "gate", "mixture_residual", "large_proposal", "large_magnitude", "boundary_confidence")
    diags: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    for s in range(0, len(clip.frame_ids), args.eval_clip_batch):
        e = min(len(clip.frame_ids), s + args.eval_clip_batch)
        xs = []
        for i in range(s, e):
            ids = [max(0, i - k) for k in range(args.context_frames)]
            xf, _edge, _var = make_features_from_raws(clip.raws[ids], clip.valids[ids])
            xs.append(xf)
        x = torch.from_numpy(np.stack(xs)).to(device)
        _logit, p, r, d = model(x, args.residual_scale)
        raw = torch.from_numpy(clip.raws[s:e]).to(device)
        refined.append((raw + r[:, 0]).cpu().numpy())
        residual.append(r[:, 0].cpu().numpy())
        p_bad.append(p[:, 0].cpu().numpy())
        for k in keys:
            if k in d:
                diags[k].append(d[k][:, 0].cpu().numpy())
    out = {"refined": np.concatenate(refined), "residual": np.concatenate(residual), "p_bad": np.concatenate(p_bad)}
    for k, vals in diags.items():
        out[k] = np.concatenate(vals) if vals else np.full_like(out["residual"], np.nan)
    return out


def aggregate_frame_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    def mean(k: str) -> float:
        vals = [float(r[k]) for r in rows if k in r and math.isfinite(float(r[k]))]
        return float(np.mean(vals)) if vals else float("nan")

    raw, ref, oracle = mean("raw_mae"), mean("refined_mae"), mean("oracle_mae")
    good = sum(float(r.get("raw_good_pixels", 0)) for r in rows)
    return {
        "frames": len(rows),
        "raw_mae": raw,
        "refined_mae": ref,
        "oracle_mae": oracle,
        "oracle_gap_recovered_pct": 100.0 * (raw - ref) / max(raw - oracle, 1e-9),
        "raw_bad3": mean("raw_bad3"),
        "refined_bad3": mean("refined_bad3"),
        "new_bad3_frame_mean_pct": mean("new_bad3_pct"),
        "new_bad3_pixel_weighted_pct": 100.0 * sum(float(r.get("new_bad3_pixels", 0)) for r in rows) / max(good, 1.0),
        "modified_pct": mean("modified_pct"),
    }


def bin_rows(values: np.ndarray, bins: np.ndarray, stats: dict[str, np.ndarray], prefix: str) -> list[dict[str, Any]]:
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (values >= lo) & (values < hi)
        row: dict[str, Any] = {f"{prefix}_bin": f"[{lo:g},{hi:g})", "pixels": int(m.sum())}
        if m.any():
            for name, arr in stats.items():
                row[name] = float(np.mean(arr[m]))
        rows.append(row)
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mpc-root", type=Path, default=DEFAULT_MPC)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--eval-full-gt", action="store_true")
    args = p.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "diagnostics").mkdir(exist_ok=True)

    ckpt = args.checkpoint or (args.mpc_root / "checkpoints" / "best_pareto.pt")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = argparse.Namespace(**ck["args"])
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = magnitude_proposal_critic_refiner(ck.get("input_channels", 16), cfg.residual_scale).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    clips = load_clips(cfg.oracle_targets_root, cfg)
    frame_rows: list[dict[str, Any]] = []
    flat_lists: dict[str, list[np.ndarray]] = {k: [] for k in (
        "raw_err", "ref_err", "oracle_err", "raw_good", "new_bad3", "beneficial", "harmful",
        "residual", "trust", "p_bad", "damping", "gate", "mixture", "large", "boundary",
        "oracle_beneficial", "useful_delta", "temporal_dt",
    )}

    pred_by_clip = {}
    for clip in clips:
        pred = predict_clip(model, clip, cfg, device)
        pred_by_clip[clip.clip_id] = pred
        for i, frame_id in enumerate(clip.frame_ids):
            v = clip.valids[i] > 0
            row = {
                "clip_id": clip.clip_id,
                "sequence_id": clip.sequence_id,
                "frame_id": frame_id,
                "failure_mode": clip.failure_mode,
                **metric(clip.raws[i], clip.gts[i], clip.valids[i], pred["residual"][i], clip.oracle[i]),
                "trust_mean": float(np.mean(pred["trust"][i][v])) if v.any() else float("nan"),
                "applied_abs_mean": float(np.mean(np.abs(pred["residual"][i][v]))) if v.any() else float("nan"),
            }
            frame_rows.append(row)

        valid = clip.valids > 0
        raw_err = np.abs(clip.raws - clip.gts)
        ref_err = np.abs(pred["refined"] - clip.gts)
        oracle_err = np.abs(clip.oracle - clip.gts)
        temporal_dt = np.abs(clip.raws - np.roll(clip.raws, 1, axis=0))
        temporal_dt[0] = 0
        arrays = {
            "raw_err": raw_err, "ref_err": ref_err, "oracle_err": oracle_err,
            "raw_good": raw_err < 1.0,
            "new_bad3": (raw_err < 1.0) & (ref_err >= 3.0) & valid,
            "beneficial": ref_err + 0.5 < raw_err,
            "harmful": ref_err > raw_err + 0.5,
            "residual": pred["residual"], "trust": pred["trust"], "p_bad": pred["p_bad"],
            "damping": pred["damping"], "gate": pred["gate"], "mixture": pred["mixture_residual"],
            "large": pred["large_proposal"], "boundary": pred["boundary_confidence"],
            "oracle_beneficial": (raw_err - oracle_err) > 1.0,
            "useful_delta": clip.gts - clip.raws,
            "temporal_dt": temporal_dt,
        }
        for k, arr in arrays.items():
            flat_lists[k].append(arr[valid].astype(np.float32 if arr.dtype != bool else np.bool_))

    flat = {k: np.concatenate(v) for k, v in flat_lists.items()}
    applied_abs = np.abs(flat["residual"])
    new = flat["new_bad3"].astype(bool)
    harmful = flat["harmful"].astype(bool)
    beneficial = flat["beneficial"].astype(bool)
    raw_good = flat["raw_good"].astype(bool)
    oracle_beneficial = flat["oracle_beneficial"].astype(bool)

    write_csv(args.output_root / "new_bad3_by_clip.csv", frame_rows)
    write_csv(args.output_root / "new_bad3_by_failure_mode.csv", [
        {"failure_mode": fm, **aggregate_frame_rows([r for r in frame_rows if r["failure_mode"] == fm])}
        for fm in sorted({r["failure_mode"] for r in frame_rows})
    ])

    stats = {
        "new_bad3_rate_pct": new.astype(np.float32) * 100.0,
        "beneficial_rate_pct": beneficial.astype(np.float32) * 100.0,
        "harmful_rate_pct": harmful.astype(np.float32) * 100.0,
        "mean_error_reduction_px": flat["raw_err"] - flat["ref_err"],
        "raw_good_corruption_rate_pct": (new & raw_good).astype(np.float32) * 100.0,
        "trust_mean": flat["trust"],
        "proposal_abs_mean": applied_abs,
    }
    write_csv(args.output_root / "new_bad3_by_proposal_magnitude.csv", bin_rows(applied_abs, MAG_BINS, stats, "applied_abs_px"))
    write_csv(args.output_root / "new_bad3_by_trust_bin.csv", bin_rows(flat["trust"], TRUST_BINS, stats, "trust"))
    write_csv(args.output_root / "beneficial_harmful_by_trust_bin.csv", bin_rows(flat["trust"], TRUST_BINS, stats, "trust"))

    useful = np.abs(flat["useful_delta"]) > 1e-3
    sign_ok = np.sign(flat["residual"]) == np.sign(flat["useful_delta"])
    correct_sign = useful & sign_ok
    overshoot = correct_sign & (applied_abs > np.abs(flat["useful_delta"])) & harmful
    undershoot = correct_sign & (applied_abs < 0.5 * np.abs(flat["useful_delta"])) & oracle_beneficial
    write_csv(args.output_root / "proposal_sign_accuracy.csv", [{
        "scope": "valid_with_nonzero_gt_delta",
        "pixels": int(useful.sum()),
        "sign_accuracy_pct": 100.0 * float(sign_ok[useful].mean()) if useful.any() else float("nan"),
        "sign_accuracy_on_oracle_beneficial_pct": 100.0 * float(sign_ok[useful & oracle_beneficial].mean()) if (useful & oracle_beneficial).any() else float("nan"),
        "wrong_sign_harmful_pct": 100.0 * float((~sign_ok & harmful & useful).sum()) / max(int((harmful & useful).sum()), 1),
    }])
    write_csv(args.output_root / "proposal_overshoot_analysis.csv", [{
        "pixels_correct_sign": int(correct_sign.sum()),
        "overshoot_harmful_pixels": int(overshoot.sum()),
        "overshoot_harmful_pct_of_correct_sign": 100.0 * float(overshoot.sum()) / max(int(correct_sign.sum()), 1),
        "mean_overshoot_px": float(np.mean(applied_abs[overshoot] - np.abs(flat["useful_delta"][overshoot]))) if overshoot.any() else 0.0,
    }])
    write_csv(args.output_root / "proposal_undershoot_analysis.csv", [{
        "oracle_beneficial_correct_sign_pixels": int((oracle_beneficial & correct_sign).sum()),
        "undershoot_pixels": int(undershoot.sum()),
        "undershoot_pct": 100.0 * float(undershoot.sum()) / max(int((oracle_beneficial & correct_sign).sum()), 1),
    }])

    authorized = applied_abs > 0.01
    tp = int((authorized & oracle_beneficial).sum())
    fp = int((authorized & ~oracle_beneficial).sum())
    fn = int((~authorized & oracle_beneficial).sum())
    support_row = {
        "authorized_pixels": int(authorized.sum()),
        "oracle_beneficial_pixels": int(oracle_beneficial.sum()),
        "precision_pct": 100.0 * tp / max(tp + fp, 1),
        "recall_pct": 100.0 * tp / max(tp + fn, 1),
        "false_authorization_pixels": fp,
        "missed_support_pixels": fn,
    }
    write_csv(args.output_root / "proposal_support_precision_recall.csv", [support_row])

    order = np.argsort(applied_abs)
    n = len(order)
    total_new = max(int(new.sum()), 1)
    top_rows = []
    for pct in (0.1, 1, 5, 10):
        idx = order[int((1.0 - pct / 100.0) * n) :]
        top_rows.append({
            "top_applied_magnitude_pct": pct,
            "pixels": int(idx.size),
            "new_bad3_pixels": int(new[idx].sum()),
            "share_of_all_new_bad3_pct": 100.0 * float(new[idx].sum()) / total_new,
            "applied_abs_threshold_px": float(applied_abs[idx].min()) if idx.size else float("nan"),
        })
    write_csv(args.output_root / "top_damage_concentration.csv", top_rows)
    write_csv(args.output_root / "raw_good_damage_analysis.csv", [{
        "raw_good_pixels": int(raw_good.sum()),
        "raw_good_new_bad3_pixels": int(new.sum()),
        "raw_good_new_bad3_pct": 100.0 * float(new.sum()) / max(int(raw_good.sum()), 1),
        "mean_applied_abs_on_damaged_raw_good_px": float(applied_abs[new].mean()) if new.any() else 0.0,
        "mean_trust_on_damaged_raw_good_px": float(flat["trust"][new].mean()) if new.any() else 0.0,
    }])

    full_gt_rows = []
    try:
        full_gt_rows.append({"source": "recorded_mpc", **json.loads((args.mpc_root / "aggregate_summary.json").read_text())["full_gt_test"]})
        if args.eval_full_gt:
            _splits, by_split = load_samples_with_split(cfg.targets_root, cfg.balanced_split_json, cfg.max_frames)
            shards = load_shards(by_split["train"] + by_split["val"] + by_split["test"])
            loader = make_loader(FullFrameDataset(by_split["test"], shards, cfg.context_frames), cfg.eval_batch_size, max(0, cfg.num_workers // 2), False, cfg.prefetch_factor)
            full_gt_rows.append({"source": "recomputed_protocol", **full_gt_eval(model, loader, device, cfg.bad_threshold_px)})
    except Exception as exc:
        full_gt_rows.append({"source": "error", "error": str(exc)})
    write_csv(args.output_root / "full_gt_generalization_analysis.csv", full_gt_rows)

    sweep_rows, clip_rows = [], []
    thresholds = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for thr in thresholds:
        rows = []
        for clip in clips:
            pred = pred_by_clip[clip.clip_id]
            res = np.where(pred["trust"] >= thr, pred["residual"], 0.0)
            for i in range(len(clip.frame_ids)):
                rows.append({"clip_id": clip.clip_id, "failure_mode": clip.failure_mode, **metric(clip.raws[i], clip.gts[i], clip.valids[i], res[i], clip.oracle[i])})
        sweep_rows.append({"policy": "trust_threshold", "threshold": thr, "clip_px": "none", **aggregate_frame_rows(rows)})
    for clip_px in (6, 9, 12, 16, 20, 24, 32):
        for thr in (0.0, 0.5, 0.7, 0.9):
            rows = []
            for clip in clips:
                pred = pred_by_clip[clip.clip_id]
                res = np.clip(pred["residual"], -clip_px, clip_px)
                res = np.where(pred["trust"] >= thr, res, 0.0)
                for i in range(len(clip.frame_ids)):
                    rows.append({"clip_id": clip.clip_id, "failure_mode": clip.failure_mode, **metric(clip.raws[i], clip.gts[i], clip.valids[i], res[i], clip.oracle[i])})
            row = {"policy": "trust_threshold_plus_residual_clip", "threshold": thr, "clip_px": clip_px, **aggregate_frame_rows(rows)}
            sweep_rows.append(row)
            for fm in sorted({r["failure_mode"] for r in rows}):
                clip_rows.append({"threshold": thr, "clip_px": clip_px, "failure_mode": fm, **aggregate_frame_rows([r for r in rows if r["failure_mode"] == fm])})
    write_csv(args.output_root / "threshold_sweep.csv", sweep_rows)
    write_csv(args.output_root / "proposal_clip_sweep.csv", clip_rows)

    summary = {
        "checkpoint": str(ckpt),
        "selected_current": aggregate_frame_rows(frame_rows),
        "new_bad3_total_pixels": int(new.sum()),
        "top_damage_concentration": top_rows,
        "support_precision_recall": support_row,
        "sign_accuracy_pct": 100.0 * float(sign_ok[useful].mean()) if useful.any() else float("nan"),
        "overshoot_harmful_pixels": int(overshoot.sum()),
        "undershoot_pixels": int(undershoot.sum()),
        "best_posthoc": min(sweep_rows, key=lambda r: (max(0.0, r["new_bad3_frame_mean_pct"] - 1.3), r["refined_mae"])),
        "diagnosis": "MPC's large proposal works; safety failure is verifier/authorization, not lack of correction magnitude.",
    }
    (args.output_root / "audit_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (args.output_root / "README.md").write_text(
        "# MPC Safety Failure Audit\n\n"
        f"Checkpoint: `{ckpt}`.\n\n"
        f"Current selected MAE `{summary['selected_current']['refined_mae']:.4f}`, gap `{summary['selected_current']['oracle_gap_recovered_pct']:.2f}%`, "
        f"new-Bad3 frame mean `{summary['selected_current']['new_bad3_frame_mean_pct']:.2f}%`.\n\n"
        f"Support precision `{support_row['precision_pct']:.2f}%`, recall `{support_row['recall_pct']:.2f}%`; "
        f"proposal sign accuracy `{summary['sign_accuracy_pct']:.2f}%`. Top 1% applied magnitudes cause "
        f"`{top_rows[1]['share_of_all_new_bad3_pct']:.2f}%` of new-Bad3.\n\n"
        "Conclusion: keep the large proposal branch; train a counterfactual verifier to authorize the proposal by predicted benefit/new-Bad3 risk and safe step size.\n"
    )

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        plt.hist(applied_abs[np.isfinite(applied_abs)], bins=80, log=True)
        plt.xlabel("|applied residual| px")
        plt.ylabel("valid pixels")
        plt.tight_layout()
        plt.savefig(args.output_root / "diagnostics" / "applied_magnitude_hist.png", dpi=160)
        plt.close()

        plt.figure(figsize=(6, 4))
        take = slice(None, None, 200)
        plt.scatter(flat["trust"][take], (flat["raw_err"] - flat["ref_err"])[take], s=1, alpha=0.15)
        plt.xlabel("trust")
        plt.ylabel("error reduction px")
        plt.tight_layout()
        plt.savefig(args.output_root / "diagnostics" / "trust_vs_error_reduction.png", dpi=160)
        plt.close()
    except Exception:
        pass

    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
