#!/usr/bin/env python3
"""v3.3: failure-mode-conditioned abstention threshold calibration on top of v3.2c.

Evaluation-only: loads the frozen v3.2c checkpoint, runs one forward pass per selected
clip frame, then sweeps per-failure-mode threshold policies in numpy. No training, no
S2M2/SAV/RAFT/DINO inference. Also evaluates full-GT val/test at the candidate default
thresholds via the existing evaluate() path.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "eval_scripts"))

from train_tiny_refiner_v1_full_gt import DEFAULT_TARGETS_ROOT, load_shards, parse_bool, write_csv  # noqa: E402
from train_tiny_refiner_v3_1_staged_abstention import (  # noqa: E402
    DEFAULT_BALANCED_SPLIT,
    AbstentionCropRefiner,
    FullFrameDataset,
    evaluate,
    load_samples_with_split,
)
from train_tiny_refiner_v3_2_hybrid_oracle import Clip, load_clips, make_loader  # noqa: E402
from evaluate_v3_1_on_selected_oracle_clips import predict_clip  # noqa: E402


DEFAULT_BASE_CHECKPOINT = Path("results/03_temporal_refinement/training/tiny_refiner_v3_2c_hybrid_oracle_freeze_detector_long/checkpoints/best.pt")
DEFAULT_ORACLE_TARGETS = Path("results/03_temporal_refinement/evaluation/distillation_targets_selected_clips")
DEFAULT_OUTPUT = Path("results/03_temporal_refinement/training/tiny_refiner_v3_3_failure_mode_threshold_calibration")

DEFAULT_CANDIDATES = (0.5, 0.7)
STRICT_CANDIDATES = (0.7, 0.8, 0.9, 0.95, 1.01)
PATHOLOGICAL_MODES = ("high_temporal_flicker", "high_boundary_error")
V32C_BASELINE = {"selected_mae": 11.0054, "oracle_gap_pct": 7.03, "frame_mean_new_bad3": 5.36, "patho_new_bad3": 15.75}


def frame_metrics_for_threshold(clip: Clip, p_bad: np.ndarray, residual: np.ndarray, threshold: float) -> list[dict[str, float]]:
    hard = (p_bad >= threshold).astype(np.float32)
    refined = clip.raws + hard * residual
    rows = []
    for i in range(len(clip.frame_ids)):
        valid = clip.valids[i] > 0
        n = max(int(valid.sum()), 1)
        raw_err = np.abs(clip.raws[i] - clip.gts[i])
        ref_err = np.abs(refined[i] - clip.gts[i])
        oracle_err = np.abs(clip.oracle[i] - clip.gts[i])
        good = valid & (raw_err < 1.0)
        n_good = max(int(good.sum()), 1)
        rows.append({
            "raw_mae": float(raw_err[valid].mean()) if valid.any() else float("nan"),
            "refined_mae": float(ref_err[valid].mean()) if valid.any() else float("nan"),
            "oracle_mae": float(oracle_err[valid].mean()) if valid.any() else float("nan"),
            "raw_bad3": 100.0 * float((raw_err[valid] > 3.0).sum()) / n,
            "refined_bad3": 100.0 * float((ref_err[valid] > 3.0).sum()) / n,
            "new_bad3_pct": 100.0 * float((ref_err[good] >= 3.0).sum()) / n_good,
            "new_bad3_pixels": float((ref_err[good] >= 3.0).sum()),
            "raw_good_pixels": float(good.sum()),
            "modified_pct": 100.0 * float(hard[i][valid].mean()) if valid.any() else 0.0,
        })
    return rows


def aggregate(frames: list[dict[str, float]]) -> dict[str, float]:
    def fmean(key: str) -> float:
        vals = [f[key] for f in frames if math.isfinite(f[key])]
        return float(np.mean(vals)) if vals else float("nan")

    raw, ref, oracle = fmean("raw_mae"), fmean("refined_mae"), fmean("oracle_mae")
    good = sum(f["raw_good_pixels"] for f in frames)
    return {
        "frames": len(frames),
        "raw_mae": raw,
        "refined_mae": ref,
        "oracle_mae": oracle,
        "oracle_gap_recovered_pct": 100.0 * (raw - ref) / (raw - oracle) if raw > oracle else float("nan"),
        "raw_bad3": fmean("raw_bad3"),
        "refined_bad3": fmean("refined_bad3"),
        "new_bad3_frame_mean_pct": fmean("new_bad3_pct"),
        "new_bad3_pixel_weighted_pct": 100.0 * sum(f["new_bad3_pixels"] for f in frames) / max(good, 1.0),
        "modified_pct": fmean("modified_pct"),
    }


def policy_threshold(clip: Clip, policy: dict[str, float]) -> float:
    if clip.failure_mode == "high_temporal_flicker":
        return policy["flicker"]
    if clip.failure_mode == "high_boundary_error":
        return policy["boundary"]
    return policy["default"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    p.add_argument("--targets-root", type=Path, default=DEFAULT_TARGETS_ROOT)
    p.add_argument("--oracle-targets-root", type=Path, default=DEFAULT_ORACLE_TARGETS)
    p.add_argument("--balanced-split-json", type=Path, default=DEFAULT_BALANCED_SPLIT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--context-frames", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--residual-scale", type=float, default=3.0)
    p.add_argument("--bad-threshold-px", type=float, default=3.0)
    p.add_argument("--good-threshold-px", type=float, default=1.0)
    p.add_argument("--oracle-min-improvement-px", type=float, default=1.0)
    p.add_argument("--oracle-hard-only", nargs="?", const=True, default=False, type=parse_bool)
    p.add_argument("--max-auc-pixels", type=int, default=200000)
    p.add_argument("--risk-cutoffs", type=float, nargs="*", default=[0.25, 0.35])
    p.add_argument("--skip-full-gt", nargs="?", const=True, default=False, type=parse_bool)
    p.add_argument("--overwrite", nargs="?", const=True, default=True, type=parse_bool)
    args = p.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    log: list[str] = [f"base_checkpoint={args.base_checkpoint}"]

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    model = AbstentionCropRefiner(int(ckpt.get("input_channels", 16))).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.append(f"device={device} ckpt_epoch={ckpt.get('epoch')} ckpt_threshold={ckpt.get('threshold')}")

    clips = load_clips(args.oracle_targets_root, args)
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for clip in clips:
        _, p_bad, residual, _ = predict_clip(model, clip.raws, clip.valids, args, device, args.residual_scale, 0.7)
        predictions[clip.clip_id] = (p_bad, residual)
    log.append(f"clips={len(clips)} frames={sum(len(c.frame_ids) for c in clips)}")

    # per-clip per-threshold frame metric cache
    unique_thresholds = sorted(set(DEFAULT_CANDIDATES) | set(STRICT_CANDIDATES))
    cache: dict[tuple[str, float], list[dict[str, float]]] = {}
    for clip in clips:
        p_bad, residual = predictions[clip.clip_id]
        for t in unique_thresholds:
            cache[(clip.clip_id, t)] = frame_metrics_for_threshold(clip, p_bad, residual, t)

    patho_clips = [c for c in clips if c.failure_mode in PATHOLOGICAL_MODES]
    clean_clips = [c for c in clips if c.failure_mode not in PATHOLOGICAL_MODES]

    def policy_rows(policy: dict[str, float]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        all_frames: list[dict[str, float]] = []
        patho_frames: list[dict[str, float]] = []
        clean_frames: list[dict[str, float]] = []
        for clip in clips:
            frames = cache[(clip.clip_id, policy_threshold(clip, policy))]
            all_frames.extend(frames)
            (patho_frames if clip in patho_clips else clean_frames).extend(frames)
        return aggregate(all_frames), aggregate(patho_frames), aggregate(clean_frames)

    # --- failure-mode policy grid ---
    grid_rows: list[dict[str, Any]] = []
    for default, flicker, boundary in itertools.product(DEFAULT_CANDIDATES, STRICT_CANDIDATES, STRICT_CANDIDATES):
        policy = {"default": default, "flicker": flicker, "boundary": boundary}
        agg, patho, clean = policy_rows(policy)
        grid_rows.append({
            "policy_type": "failure_mode",
            "default_threshold": default,
            "flicker_threshold": flicker,
            "boundary_threshold": boundary,
            **{f"all_{k}": v for k, v in agg.items()},
            **{f"patho_{k}": v for k, v in patho.items()},
            **{f"clean_{k}": v for k, v in clean.items()},
        })

    # --- frame-risk policy variant (deployable without failure-mode labels) ---
    # risk = predicted modified fraction at threshold 0.7; risky frames get stricter threshold
    risk_rows: list[dict[str, Any]] = []
    for cutoff, strict in itertools.product(args.risk_cutoffs, (0.9, 0.95, 1.01)):
        all_frames: list[dict[str, float]] = []
        patho_frames: list[dict[str, float]] = []
        clean_frames: list[dict[str, float]] = []
        for clip in clips:
            p_bad, _res = predictions[clip.clip_id]
            base = cache[(clip.clip_id, 0.7)]
            strict_frames = cache[(clip.clip_id, strict)]
            for i in range(len(clip.frame_ids)):
                valid = clip.valids[i] > 0
                risk = float((p_bad[i][valid] >= 0.7).mean()) if valid.any() else 0.0
                frame = strict_frames[i] if risk > cutoff else base[i]
                all_frames.append(frame)
                (patho_frames if clip in patho_clips else clean_frames).append(frame)
        agg, patho, clean = aggregate(all_frames), aggregate(patho_frames), aggregate(clean_frames)
        risk_rows.append({
            "policy_type": "frame_risk",
            "default_threshold": 0.7,
            "risk_cutoff": cutoff,
            "strict_threshold": strict,
            **{f"all_{k}": v for k, v in agg.items()},
            **{f"patho_{k}": v for k, v in patho.items()},
            **{f"clean_{k}": v for k, v in clean.items()},
        })

    write_csv_rows = grid_rows + risk_rows
    fields: list[str] = []
    for r in write_csv_rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with (args.output_root / "threshold_grid_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(write_csv_rows)

    # --- baseline (v3.2c behavior = uniform 0.7) ---
    baseline = next(r for r in grid_rows if r["default_threshold"] == 0.7 and r["flicker_threshold"] == 0.7 and r["boundary_threshold"] == 0.7)

    # --- constrained selection over failure-mode grid ---
    def feasible(r: dict[str, Any]) -> bool:
        # default pinned at 0.7 so the four well-behaved clips keep exact v3.2c behavior
        # (success criterion 5); a looser default (0.5) trades clean-clip safety for MAE.
        return (
            r["default_threshold"] == 0.7
            and r["all_new_bad3_frame_mean_pct"] < baseline["all_new_bad3_frame_mean_pct"]
            and r["patho_new_bad3_frame_mean_pct"] <= 0.5 * baseline["patho_new_bad3_frame_mean_pct"]
            and r["clean_modified_pct"] > 0.0
        )

    candidates = [r for r in grid_rows if feasible(r)]
    log.append(f"grid_policies={len(grid_rows)} risk_policies={len(risk_rows)} feasible={len(candidates)}")
    gap_ok = [r for r in candidates if r["all_oracle_gap_recovered_pct"] >= V32C_BASELINE["oracle_gap_pct"]]
    pool = gap_ok or candidates
    best = min(pool, key=lambda r: r["all_refined_mae"]) if pool else baseline
    best_is_baseline = best is baseline
    alt_safest = min(candidates, key=lambda r: (r["patho_new_bad3_frame_mean_pct"], r["all_refined_mae"])) if candidates else None

    # --- full-GT val/test at candidate default thresholds ---
    full_gt: dict[str, dict[float, dict[str, Any]]] = {}
    if not args.skip_full_gt:
        splits, by_split = load_samples_with_split(args.targets_root, args.balanced_split_json, args.max_frames)
        shards = load_shards(by_split["val"] + by_split["test"])
        for split in ("val", "test"):
            ds = FullFrameDataset(by_split[split], shards, args.context_frames)
            loader = make_loader(ds, args.eval_batch_size, args.num_workers, False, args.prefetch_factor)
            rows, _ = evaluate(model, loader, args, device, split, per_sequence=False)
            full_gt[split] = {round(float(r["threshold"]), 4): r for r in rows}
        log.append("full_gt_eval=done")

    best_policy = {
        "policy_type": "failure_mode",
        "default_threshold": best["default_threshold"],
        "high_temporal_flicker_threshold": best["flicker_threshold"],
        "high_boundary_error_threshold": best["boundary_threshold"],
        "selected_from_feasible": not best_is_baseline,
        "constraints": {
            "frame_mean_new_bad3_below_v32c": True,
            "patho_new_bad3_at_most_half_of_v32c": True,
            "no_identity_on_clean_clips": True,
            "oracle_gap_target_pct": V32C_BASELINE["oracle_gap_pct"],
            "oracle_gap_met": bool(gap_ok),
        },
        "v3_2c_baseline_policy_metrics": {k: v for k, v in baseline.items() if isinstance(v, (int, float))},
        "chosen_policy_metrics": {k: v for k, v in best.items() if isinstance(v, (int, float))},
        "alt_safety_max_policy": {k: v for k, v in alt_safest.items() if isinstance(v, (int, float))} if alt_safest else None,
    }
    if full_gt:
        thr = float(best["default_threshold"])
        best_policy["full_gt_val_at_default"] = full_gt["val"].get(round(thr, 4))
        best_policy["full_gt_test_at_default"] = full_gt["test"].get(round(thr, 4))
    (args.output_root / "best_threshold_policy.json").write_text(json.dumps(best_policy, indent=2, default=str) + "\n")

    # --- per-policy selected metrics (baseline vs chosen vs best risk policy) ---
    best_risk = min(risk_rows, key=lambda r: (r["patho_new_bad3_frame_mean_pct"], r["all_refined_mae"])) if risk_rows else None
    policy_compare = [
        {"policy": "v3.2c_uniform_0.7", **baseline},
        {"policy": "v3.3_failure_mode_chosen", **best},
    ]
    if best_risk:
        policy_compare.append({"policy": "v3.3_frame_risk_best", **best_risk})
    fields2: list[str] = []
    for r in policy_compare:
        for k in r:
            if k not in fields2:
                fields2.append(k)
    with (args.output_root / "selected_metrics_by_policy.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields2)
        w.writeheader()
        w.writerows(policy_compare)

    # --- per-failure-mode metrics under chosen policy ---
    chosen_policy = {"default": best["default_threshold"], "flicker": best["flicker_threshold"], "boundary": best["boundary_threshold"]}
    mode_rows = []
    for mode in sorted({c.failure_mode for c in clips}):
        frames: list[dict[str, float]] = []
        thr_used = None
        for clip in clips:
            if clip.failure_mode != mode:
                continue
            thr_used = policy_threshold(clip, chosen_policy)
            frames.extend(cache[(clip.clip_id, thr_used)])
        before = []
        for clip in clips:
            if clip.failure_mode == mode:
                before.extend(cache[(clip.clip_id, 0.7)])
        agg_after, agg_before = aggregate(frames), aggregate(before)
        mode_rows.append({
            "failure_mode": mode,
            "threshold_used": thr_used,
            **{f"after_{k}": v for k, v in agg_after.items()},
            **{f"before_{k}": v for k, v in agg_before.items()},
        })
    write_csv(args.output_root / "failure_mode_metrics.csv", mode_rows)

    # --- pathological vs clean summary ---
    patho_before = aggregate([f for c in patho_clips for f in cache[(c.clip_id, 0.7)]])
    patho_after = aggregate([f for c in patho_clips for f in cache[(c.clip_id, policy_threshold(c, chosen_policy))]])
    clean_before = aggregate([f for c in clean_clips for f in cache[(c.clip_id, 0.7)]])
    clean_after = aggregate([f for c in clean_clips for f in cache[(c.clip_id, policy_threshold(c, chosen_policy))]])
    write_csv(args.output_root / "pathological_vs_clean_summary.csv", [
        {"group": "pathological_2_clips", "when": "before_v3.2c_0.7", **patho_before},
        {"group": "pathological_2_clips", "when": "after_v3.3_policy", **patho_after},
        {"group": "clean_4_clips", "when": "before_v3.2c_0.7", **clean_before},
        {"group": "clean_4_clips", "when": "after_v3.3_policy", **clean_after},
    ])

    # --- plots ---
    fig, ax = plt.subplots(figsize=(8, 6))
    xs = [r["all_refined_mae"] for r in grid_rows]
    ys = [r["all_new_bad3_frame_mean_pct"] for r in grid_rows]
    ax.scatter(xs, ys, s=14, alpha=0.5, label="failure-mode policies")
    ax.scatter([r["all_refined_mae"] for r in risk_rows], [r["all_new_bad3_frame_mean_pct"] for r in risk_rows], s=20, marker="^", alpha=0.7, label="frame-risk policies")
    ax.scatter([baseline["all_refined_mae"]], [baseline["all_new_bad3_frame_mean_pct"]], s=90, marker="*", color="red", label="v3.2c (0.7 uniform)")
    ax.scatter([best["all_refined_mae"]], [best["all_new_bad3_frame_mean_pct"]], s=90, marker="*", color="green", label="chosen v3.3 policy")
    ax.set_xlabel("selected frame-mean refined MAE")
    ax.set_ylabel("frame-mean new-Bad3 from raw-good (%)")
    ax.set_title("v3.3 threshold policies: accuracy vs safety")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_root / "threshold_policy_mae_vs_newbad3.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    labels = [c.clip_id[:26] for c in patho_clips]
    before_vals = [aggregate(cache[(c.clip_id, 0.7)])["new_bad3_frame_mean_pct"] for c in patho_clips]
    after_vals = [aggregate(cache[(c.clip_id, policy_threshold(c, chosen_policy))])["new_bad3_frame_mean_pct"] for c in patho_clips]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, before_vals, width=0.4, label="v3.2c (0.7)")
    ax.bar(x + 0.2, after_vals, width=0.4, label="v3.3 policy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("new-Bad3 frame-mean (%)")
    ax.set_title("Pathological clips: new-Bad3 before/after v3.3 policy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_root / "pathological_newbad3_before_after.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [c.clip_id[:26] for c in clips]
    before_vals = [aggregate(cache[(c.clip_id, 0.7)])["modified_pct"] for c in clips]
    after_vals = [aggregate(cache[(c.clip_id, policy_threshold(c, chosen_policy))])["modified_pct"] for c in clips]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, before_vals, width=0.4, label="v3.2c (0.7)")
    ax.bar(x + 0.2, after_vals, width=0.4, label="v3.3 policy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("modified pixels (%)")
    ax.set_title("Modified pixels per clip before/after v3.3 policy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_root / "modified_pixels_before_after.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter([r["all_oracle_gap_recovered_pct"] for r in grid_rows], ys, s=14, alpha=0.5, label="failure-mode policies")
    ax.scatter([baseline["all_oracle_gap_recovered_pct"]], [baseline["all_new_bad3_frame_mean_pct"]], s=90, marker="*", color="red", label="v3.2c")
    ax.scatter([best["all_oracle_gap_recovered_pct"]], [best["all_new_bad3_frame_mean_pct"]], s=90, marker="*", color="green", label="chosen v3.3")
    ax.set_xlabel("oracle gap recovered (%)")
    ax.set_ylabel("frame-mean new-Bad3 (%)")
    ax.set_title("Oracle gain vs safety trade-off across policies")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_root / "oracle_gap_vs_safety_tradeoff.png", dpi=120)
    plt.close(fig)

    summary = {
        "base_checkpoint": str(args.base_checkpoint),
        "output_root": str(args.output_root),
        "elapsed_seconds": time.perf_counter() - start,
        "policies_swept": {"failure_mode": len(grid_rows), "frame_risk": len(risk_rows)},
        "v3_2c_baseline": {k: baseline[k] for k in ("all_refined_mae", "all_oracle_gap_recovered_pct", "all_new_bad3_frame_mean_pct", "all_new_bad3_pixel_weighted_pct", "all_modified_pct", "patho_new_bad3_frame_mean_pct", "clean_new_bad3_frame_mean_pct")},
        "chosen_policy": best_policy,
        "best_frame_risk_policy": best_risk,
        "no_teacher_inference": True,
        "no_training": True,
    }
    (args.output_root / "aggregate_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")

    improved_patho = baseline["patho_new_bad3_frame_mean_pct"] - best["patho_new_bad3_frame_mean_pct"]
    mae_cost = best["all_refined_mae"] - baseline["all_refined_mae"]
    gap_after = best["all_oracle_gap_recovered_pct"]
    verdict = (
        "Threshold-only v3.3 works: pathological new-Bad3 substantially reduced at small accuracy cost. Stop here; v3.3b training not needed."
        if not best_is_baseline and gap_after >= V32C_BASELINE["oracle_gap_pct"] - 1.0
        else (
            "Threshold-only v3.3 reduces pathological risk but costs meaningful oracle gain; consider v3.3b "
            "(hard-negative residual-only fine-tune on the two pathological clips, detector frozen) if the gain matters."
            if not best_is_baseline
            else "No feasible policy improved on v3.2c; keep v3.2c and consider v3.3b hard-negative fine-tuning."
        )
    )

    (args.output_root / "README.md").write_text(f"""# v3.3 Failure-Mode-Conditioned Threshold Calibration

Evaluation-only calibration on top of the frozen v3.2c checkpoint
(`{args.base_checkpoint}`). One forward pass per selected-clip frame, then a
{len(grid_rows)}-policy failure-mode threshold grid plus {len(risk_rows)} label-free frame-risk
variants. No training, no S2M2/SAV/RAFT/DINO inference.

## Chosen policy

- Default threshold (4 well-behaved failure modes): `{best['default_threshold']}`
- `high_temporal_flicker` threshold: `{best['flicker_threshold']}`
- `high_boundary_error` threshold: `{best['boundary_threshold']}`

## Selected clips: v3.2c (uniform 0.7) vs v3.3 policy (frame-mean, 502 frames)

| Metric | v3.2c | v3.3 chosen |
|---|---:|---:|
| Refined MAE | `{baseline['all_refined_mae']:.4f}` | `{best['all_refined_mae']:.4f}` |
| Oracle gap recovered | `{baseline['all_oracle_gap_recovered_pct']:.2f}%` | `{best['all_oracle_gap_recovered_pct']:.2f}%` |
| Refined Bad-3 | `{baseline['all_refined_bad3']:.3f}` | `{best['all_refined_bad3']:.3f}` |
| New-Bad3 frame-mean | `{baseline['all_new_bad3_frame_mean_pct']:.2f}%` | `{best['all_new_bad3_frame_mean_pct']:.2f}%` |
| New-Bad3 pixel-weighted | `{baseline['all_new_bad3_pixel_weighted_pct']:.2f}%` | `{best['all_new_bad3_pixel_weighted_pct']:.2f}%` |
| Modified pixels | `{baseline['all_modified_pct']:.2f}%` | `{best['all_modified_pct']:.2f}%` |
| Pathological 2-clip new-Bad3 | `{baseline['patho_new_bad3_frame_mean_pct']:.2f}%` | `{best['patho_new_bad3_frame_mean_pct']:.2f}%` |
| Clean 4-clip new-Bad3 | `{baseline['clean_new_bad3_frame_mean_pct']:.2f}%` | `{best['clean_new_bad3_frame_mean_pct']:.2f}%` |

Pathological new-Bad3 reduced by `{improved_patho:.2f}` points; selected MAE cost `{mae_cost:+.4f}`.
The four well-behaved clips keep their v3.2c behavior whenever the chosen default is `0.7`
(their thresholds are untouched by construction).

## Verdict

{verdict}

## Files

- `threshold_grid_results.csv`: all failure-mode and frame-risk policies with all/patho/clean aggregates.
- `best_threshold_policy.json`: chosen policy + constraints + full-GT val/test rows at the default threshold.
- `selected_metrics_by_policy.csv`: v3.2c baseline vs chosen policy vs best frame-risk policy.
- `failure_mode_metrics.csv`: before/after per failure mode under the chosen policy.
- `pathological_vs_clean_summary.csv`: 2 pathological vs 4 clean clips, before/after.
- `aggregate_summary.json`: machine-readable summary.
- Plots: `threshold_policy_mae_vs_newbad3.png`, `pathological_newbad3_before_after.png`,
  `modified_pixels_before_after.png`, `oracle_gap_vs_safety_tradeoff.png`.
""")
    log.append(f"elapsed={time.perf_counter() - start:.1f}s verdict={verdict}")
    (args.output_root / "run.log").write_text("\n".join(log) + "\n")
    print(json.dumps(summary["chosen_policy"], indent=2, default=str))
    print(f"\nverdict: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
