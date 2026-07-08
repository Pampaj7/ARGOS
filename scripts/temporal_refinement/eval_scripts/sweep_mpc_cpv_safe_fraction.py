#!/usr/bin/env python3
"""Post-hoc safe-fraction sweep for MPC and CPV."""

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
from counterfactual_proposal_verifier_refiner import counterfactual_proposal_verifier_refiner  # noqa: E402
from magnitude_proposal_critic_refiner import magnitude_proposal_critic_refiner  # noqa: E402
from train_tiny_refiner_v1_full_gt import load_shards  # noqa: E402
from train_tiny_refiner_v3_1_staged_abstention import FullFrameDataset, load_samples_with_split, make_features_from_raws  # noqa: E402
from train_tiny_refiner_v3_2_hybrid_oracle import load_clips, make_loader  # noqa: E402


OUT = Path("results/03_temporal_refinement/analysis/mpc_cpv_safe_fraction_sweep")
MPC = Path("results/03_temporal_refinement/training/magnitude_proposal_critic_refiner/checkpoints/best_pareto.pt")
CPV = Path("results/03_temporal_refinement/training/counterfactual_proposal_verifier_refiner/checkpoints/best_pareto.pt")
MULTS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.75, 0.90, 1.00)
CLIPS = (3.0, 6.0, 9.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0)
THRESHOLDS = (0.0, 0.5, 0.7, 0.8, 0.9)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def load_model(path: Path, name: str, device: torch.device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = argparse.Namespace(**ck["args"])
    model = (magnitude_proposal_critic_refiner if name == "mpc" else counterfactual_proposal_verifier_refiner)(ck.get("input_channels", 16), cfg.residual_scale).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, cfg


@torch.no_grad()
def predict_clip(model, clip, cfg, device, score_key: str):
    refined, residual, score = [], [], []
    for s in range(0, len(clip.frame_ids), cfg.eval_clip_batch):
        e = min(len(clip.frame_ids), s + cfg.eval_clip_batch)
        xs = []
        for i in range(s, e):
            ids = [max(0, i - k) for k in range(cfg.context_frames)]
            xs.append(make_features_from_raws(clip.raws[ids], clip.valids[ids])[0])
        x = torch.from_numpy(np.stack(xs)).to(device)
        _logit, _p, r, diag = model(x, cfg.residual_scale)
        raw = torch.from_numpy(clip.raws[s:e]).to(device)
        refined.append((raw + r[:, 0]).cpu().numpy())
        residual.append(r[:, 0].cpu().numpy())
        score.append(diag.get(score_key, diag["trust"])[:, 0].cpu().numpy())
    return np.concatenate(residual), np.concatenate(score)


def metrics_from_frames(rows: list[dict[str, float]]) -> dict[str, float]:
    def mean(k: str) -> float:
        vals = [r[k] for r in rows if math.isfinite(r[k])]
        return float(np.mean(vals)) if vals else float("nan")

    raw, ref, oracle = mean("raw_mae"), mean("refined_mae"), mean("oracle_mae")
    good = sum(r["raw_good_pixels"] for r in rows)
    return {
        "frames": len(rows),
        "raw_mae": raw,
        "refined_mae": ref,
        "oracle_mae": oracle,
        "oracle_gap_recovered_pct": 100.0 * (raw - ref) / max(raw - oracle, 1e-9),
        "raw_bad3": mean("raw_bad3"),
        "refined_bad3": mean("refined_bad3"),
        "new_bad3_frame_mean_pct": mean("new_bad3_pct"),
        "new_bad3_pixel_weighted_pct": 100.0 * sum(r["new_bad3_pixels"] for r in rows) / max(good, 1.0),
        "modified_pct": mean("modified_pct"),
        "beneficial_px_pct": mean("beneficial_px_pct"),
        "harmful_px_pct": mean("harmful_px_pct"),
    }


def frame_metric(raw, gt, valid, residual, oracle):
    v = valid > 0
    raw_err = np.abs(raw - gt)
    ref_err = np.abs(raw + residual - gt)
    oracle_err = np.abs(oracle - gt)
    good = v & (raw_err < 1.0)
    n = max(int(v.sum()), 1)
    return {
        "raw_mae": float(raw_err[v].mean()) if v.any() else float("nan"),
        "refined_mae": float(ref_err[v].mean()) if v.any() else float("nan"),
        "oracle_mae": float(oracle_err[v].mean()) if v.any() else float("nan"),
        "raw_bad3": 100.0 * float((raw_err[v] >= 3.0).sum()) / n,
        "refined_bad3": 100.0 * float((ref_err[v] >= 3.0).sum()) / n,
        "new_bad3_pct": 100.0 * float((good & (ref_err >= 3.0)).sum()) / max(int(good.sum()), 1),
        "new_bad3_pixels": float((good & (ref_err >= 3.0)).sum()),
        "raw_good_pixels": float(good.sum()),
        "modified_pct": 100.0 * float((np.abs(residual[v]) > 0.01).sum()) / n,
        "beneficial_px_pct": 100.0 * float(((raw_err - ref_err) > 0.5)[v].sum()) / n,
        "harmful_px_pct": 100.0 * float(((ref_err - raw_err) > 0.5)[v].sum()) / n,
    }


def policy_rows(model_name: str, score_key: str) -> list[dict[str, Any]]:
    return [
        {"model": model_name, "score_key": score_key, "multiplier": m, "clip_px": c, "threshold": t}
        for m in MULTS for c in CLIPS for t in THRESHOLDS
    ]


def empty_acc(n: int) -> dict[str, np.ndarray]:
    return {k: np.zeros(n, dtype=np.float64) for k in (
        "frames", "raw_mae_sum", "ref_mae_sum", "oracle_mae_sum", "raw_bad3_sum", "ref_bad3_sum",
        "new_bad3_pct_sum", "modified_pct_sum", "beneficial_pct_sum", "harmful_pct_sum",
        "good_pixels", "new_bad3_pixels",
    )}


def add_to_acc(acc: dict[str, np.ndarray], idx: np.ndarray, vals: dict[str, np.ndarray]) -> None:
    for k, v in vals.items():
        acc[k][idx] += v


def finalize_acc(prefix: str, rows: list[dict[str, Any]], acc: dict[str, np.ndarray]) -> None:
    frames = np.maximum(acc["frames"], 1.0)
    raw = acc["raw_mae_sum"] / frames
    ref = acc["ref_mae_sum"] / frames
    oracle = acc["oracle_mae_sum"] / frames
    vals = {
        f"{prefix}frames": acc["frames"],
        f"{prefix}raw_mae": raw,
        f"{prefix}refined_mae": ref,
        f"{prefix}oracle_mae": oracle,
        f"{prefix}oracle_gap_recovered_pct": 100.0 * (raw - ref) / np.maximum(raw - oracle, 1e-9),
        f"{prefix}raw_bad3": acc["raw_bad3_sum"] / frames,
        f"{prefix}refined_bad3": acc["ref_bad3_sum"] / frames,
        f"{prefix}new_bad3_frame_mean_pct": acc["new_bad3_pct_sum"] / frames,
        f"{prefix}new_bad3_pixel_weighted_pct": 100.0 * acc["new_bad3_pixels"] / np.maximum(acc["good_pixels"], 1.0),
        f"{prefix}modified_pct": acc["modified_pct_sum"] / frames,
        f"{prefix}beneficial_px_pct": acc["beneficial_pct_sum"] / frames,
        f"{prefix}harmful_px_pct": acc["harmful_pct_sum"] / frames,
    }
    for i, row in enumerate(rows):
        for k, v in vals.items():
            row[k] = float(v[i])


def policy_tensors(rows: list[dict[str, Any]], sl: slice, device: torch.device, ndims: int = 4):
    part = rows[sl]
    shape = (len(part),) + (1,) * (ndims - 1)
    return (
        torch.tensor([r["multiplier"] for r in part], device=device).view(shape),
        torch.tensor([r["clip_px"] for r in part], device=device).view(shape),
        torch.tensor([r["threshold"] for r in part], device=device).view(shape),
        np.arange(sl.start or 0, sl.stop or len(rows)),
    )


@torch.no_grad()
def accumulate_policy_chunk(raw, gt, valid, oracle, residual, score, rows, acc, sl, device):
    mult, clip_px, thr, idx = policy_tensors(rows, sl, device, 4)
    raw = torch.as_tensor(raw, device=device)
    gt = torch.as_tensor(gt, device=device)
    valid = torch.as_tensor(valid > 0, device=device)
    oracle = torch.as_tensor(oracle, device=device)
    residual = torch.as_tensor(residual, device=device)
    score = torch.as_tensor(score, device=device)

    r = torch.clamp(residual.unsqueeze(0) * mult, -clip_px, clip_px)
    r = torch.where(score.unsqueeze(0) >= thr, r, torch.zeros_like(r))
    raw_err = torch.abs(raw - gt)
    ref_err = torch.abs(raw.unsqueeze(0) + r - gt.unsqueeze(0))
    oracle_err = torch.abs(oracle - gt)
    v = valid
    good = v & (raw_err < 1.0)
    n = v.flatten(1).sum(dim=1).clamp_min(1).float()
    g = good.flatten(1).sum(dim=1).clamp_min(1).float()
    p = r.shape[0]
    vf = v.unsqueeze(0)
    goodf = good.unsqueeze(0)
    raw_mae = ((raw_err * v).flatten(1).sum(dim=1) / n).expand(p, -1)
    oracle_mae = ((oracle_err * v).flatten(1).sum(dim=1) / n).expand(p, -1)
    raw_bad3 = (((raw_err >= 3.0) & v).flatten(1).sum(dim=1).float() * (100.0 / n)).expand(p, -1)
    ref_bad3 = (((ref_err >= 3.0) & vf).flatten(2).sum(dim=2).float() * (100.0 / n.unsqueeze(0)))
    new_bad3 = (((ref_err >= 3.0) & goodf).flatten(2).sum(dim=2).float())
    vals = {
        "frames": np.full(p, raw.shape[0], dtype=np.float64),
        "raw_mae_sum": raw_mae.sum(dim=1).cpu().numpy(),
        "ref_mae_sum": ((ref_err * vf).flatten(2).sum(dim=2) / n.unsqueeze(0)).sum(dim=1).cpu().numpy(),
        "oracle_mae_sum": oracle_mae.sum(dim=1).cpu().numpy(),
        "raw_bad3_sum": raw_bad3.sum(dim=1).cpu().numpy(),
        "ref_bad3_sum": ref_bad3.sum(dim=1).cpu().numpy(),
        "new_bad3_pct_sum": (new_bad3 * (100.0 / g.unsqueeze(0))).sum(dim=1).cpu().numpy(),
        "modified_pct_sum": (((torch.abs(r) > 0.01) & vf).flatten(2).sum(dim=2).float() * (100.0 / n.unsqueeze(0))).sum(dim=1).cpu().numpy(),
        "beneficial_pct_sum": ((((raw_err.unsqueeze(0) - ref_err) > 0.5) & vf).flatten(2).sum(dim=2).float() * (100.0 / n.unsqueeze(0))).sum(dim=1).cpu().numpy(),
        "harmful_pct_sum": ((((ref_err - raw_err.unsqueeze(0)) > 0.5) & vf).flatten(2).sum(dim=2).float() * (100.0 / n.unsqueeze(0))).sum(dim=1).cpu().numpy(),
        "good_pixels": good.flatten(1).sum(dim=1).sum().repeat(p).cpu().numpy(),
        "new_bad3_pixels": new_bad3.sum(dim=1).cpu().numpy(),
    }
    add_to_acc(acc, idx, vals)


def selected_sweep(model, cfg, model_name: str, clips, device, score_key: str) -> list[dict[str, Any]]:
    rows = policy_rows(model_name, score_key)
    acc_all, acc_patho, acc_clean = empty_acc(len(rows)), empty_acc(len(rows)), empty_acc(len(rows))
    preds = {c.clip_id: predict_clip(model, c, cfg, device, score_key) for c in clips}
    for clip in clips:
        residual, score = preds[clip.clip_id]
        target = acc_patho if clip.failure_mode in PATHOLOGICAL_MODES else acc_clean
        for start in range(0, len(rows), cfg.policy_chunk):
            sl = slice(start, min(len(rows), start + cfg.policy_chunk))
            accumulate_policy_chunk(clip.raws, clip.gts, clip.valids, clip.oracle, residual, score, rows, acc_all, sl, device)
            accumulate_policy_chunk(clip.raws, clip.gts, clip.valids, clip.oracle, residual, score, rows, target, sl, device)
    finalize_acc("selected_", rows, acc_all)
    finalize_acc("patho_", rows, acc_patho)
    finalize_acc("clean_", rows, acc_clean)
    for row in rows:
        row["patho_new_bad3"] = row.pop("patho_new_bad3_frame_mean_pct")
        row["clean_new_bad3"] = row.pop("clean_new_bad3_frame_mean_pct")
    return rows


@torch.no_grad()
def val_sweep(model, cfg, rows: list[dict[str, Any]], device: torch.device):
    _splits, by_split = load_samples_with_split(cfg.targets_root, cfg.balanced_split_json, cfg.max_frames)
    shards = load_shards(by_split["train"] + by_split["val"] + by_split["test"])
    loader = make_loader(FullFrameDataset(by_split["val"], shards, cfg.context_frames), cfg.eval_batch_size, max(0, cfg.num_workers // 2), False, cfg.prefetch_factor)
    acc = {k: np.zeros(len(rows), dtype=np.float64) for k in ("n", "raw_abs", "ref_abs", "raw_b3", "ref_b3", "new_b3", "good", "mod")}
    for batch in loader:
        x = batch["x"].to(device)
        raw = batch["raw"].to(device)
        gt = batch["gt"].to(device)
        valid = batch["valid"].to(device)
        _logit, _p, residual, diag = model(x, cfg.residual_scale)
        score_cache = {"trust": diag["trust"], "verifier_safe": diag.get("verifier_safe", diag["trust"])}
        raw_err = torch.abs(raw - gt)
        v = valid > 0
        good = v & (raw_err < 1.0)
        raw_abs = float((raw_err * v).sum())
        raw_b3 = float((v & (raw_err >= 3.0)).sum())
        n = float(v.sum())
        g = float(good.sum())
        for start in range(0, len(rows), cfg.policy_chunk):
            sl = slice(start, min(len(rows), start + cfg.policy_chunk))
            mult, clip_px, thr, idx = policy_tensors(rows, sl, device, 5)
            score = score_cache[rows[start]["score_key"]]
            r = torch.clamp(residual.unsqueeze(0) * mult, -clip_px, clip_px)
            r = torch.where(score.unsqueeze(0) >= thr, r, torch.zeros_like(r))
            ref_err = torch.abs(raw + r - gt)
            vf = v.unsqueeze(0)
            acc["n"][idx] += n
            acc["raw_abs"][idx] += raw_abs
            acc["ref_abs"][idx] += (ref_err * vf).flatten(1).sum(dim=1).cpu().numpy()
            acc["raw_b3"][idx] += raw_b3
            acc["ref_b3"][idx] += ((vf & (ref_err >= 3.0)).flatten(1).sum(dim=1).float().cpu().numpy())
            acc["new_b3"][idx] += ((good.unsqueeze(0) & (ref_err >= 3.0)).flatten(1).sum(dim=1).float().cpu().numpy())
            acc["good"][idx] += g
            acc["mod"][idx] += ((vf & (torch.abs(r) > 0.01)).flatten(1).sum(dim=1).float().cpu().numpy())
    for i, row in enumerate(rows):
        n = max(acc["n"][i], 1.0)
        row.update({
            "val_raw_mae": acc["raw_abs"][i] / n,
            "val_refined_mae": acc["ref_abs"][i] / n,
            "val_raw_bad3": 100.0 * acc["raw_b3"][i] / n,
            "val_refined_bad3": 100.0 * acc["ref_b3"][i] / n,
            "val_new_bad3": 100.0 * acc["new_b3"][i] / max(acc["good"][i], 1.0),
            "val_modified_pct": 100.0 * acc["mod"][i] / n,
        })


def pareto(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        dominated = False
        for s in rows:
            if s is r:
                continue
            better_or_equal = (
                s["selected_refined_mae"] <= r["selected_refined_mae"]
                and s["patho_new_bad3"] <= r["patho_new_bad3"]
                and s["clean_new_bad3"] <= r["clean_new_bad3"]
                and s["val_refined_mae"] <= r["val_refined_mae"]
            )
            strictly = (
                s["selected_refined_mae"] < r["selected_refined_mae"]
                or s["patho_new_bad3"] < r["patho_new_bad3"]
                or s["clean_new_bad3"] < r["clean_new_bad3"]
                or s["val_refined_mae"] < r["val_refined_mae"]
            )
            if better_or_equal and strictly:
                dominated = True
                break
        if not dominated:
            out.append(r)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=OUT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--policy-chunk", type=int, default=32)
    args = p.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "diagnostics").mkdir(exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    all_rows: list[dict[str, Any]] = []
    for name, ckpt, score_key in (("mpc", MPC, "trust"), ("cpv", CPV, "verifier_safe")):
        model, cfg = load_model(ckpt, name, device)
        cfg.policy_chunk = args.policy_chunk
        clips = load_clips(cfg.oracle_targets_root, cfg)
        rows = selected_sweep(model, cfg, name, clips, device, score_key)
        val_sweep(model, cfg, rows, device)
        all_rows.extend(rows)
    front = pareto(all_rows)
    best_accuracy = min(all_rows, key=lambda r: (r["selected_refined_mae"], r["patho_new_bad3"], r["clean_new_bad3"]))
    safe = [r for r in all_rows if r["patho_new_bad3"] <= 2.0 and r["clean_new_bad3"] <= 1.0 and r["val_refined_mae"] < 5.20]
    best_safety = min(all_rows, key=lambda r: (max(0.0, r["patho_new_bad3"] - 2.0), max(0.0, r["clean_new_bad3"] - 1.0), r["selected_refined_mae"]))
    pool = safe or all_rows
    best_pareto = min(pool, key=lambda r: (max(0.0, 30.0 - r["selected_oracle_gap_recovered_pct"]), r["patho_new_bad3"] + r["clean_new_bad3"], r["val_refined_mae"]))
    write_csv(args.output_root / "sweep_results.csv", all_rows)
    write_csv(args.output_root / "pareto_front.csv", front)
    for name, row in (("best_accuracy_policy.json", best_accuracy), ("best_safety_policy.json", best_safety), ("best_pareto_policy.json", best_pareto)):
        (args.output_root / name).write_text(json.dumps(row, indent=2, default=str) + "\n")
    (args.output_root / "README.md").write_text(
        "# MPC/CPV Safe-Fraction Sweep\n\n"
        f"Policies evaluated: `{len(all_rows)}`. Best accuracy: `{best_accuracy['model']}` m={best_accuracy['multiplier']} clip={best_accuracy['clip_px']} thr={best_accuracy['threshold']} "
        f"MAE `{best_accuracy['selected_refined_mae']:.4f}`, gap `{best_accuracy['selected_oracle_gap_recovered_pct']:.2f}%`, "
        f"patho new-Bad3 `{best_accuracy['patho_new_bad3']:.2f}%`, clean `{best_accuracy['clean_new_bad3']:.2f}%`.\n\n"
        f"Best Pareto: `{best_pareto['model']}` m={best_pareto['multiplier']} clip={best_pareto['clip_px']} thr={best_pareto['threshold']} "
        f"MAE `{best_pareto['selected_refined_mae']:.4f}`, gap `{best_pareto['selected_oracle_gap_recovered_pct']:.2f}%`, "
        f"patho new-Bad3 `{best_pareto['patho_new_bad3']:.2f}%`, clean `{best_pareto['clean_new_bad3']:.2f}%`, val `{best_pareto['val_raw_mae']:.4f}->{best_pareto['val_refined_mae']:.4f}`.\n"
    )
    try:
        import matplotlib.pyplot as plt

        xs = [r["selected_oracle_gap_recovered_pct"] for r in all_rows]
        ys = [r["patho_new_bad3"] for r in all_rows]
        plt.figure(figsize=(6, 4))
        plt.scatter(xs, ys, s=8, alpha=0.5)
        plt.xlabel("selected oracle gap recovered %")
        plt.ylabel("patho new-Bad3 %")
        plt.tight_layout()
        plt.savefig(args.output_root / "diagnostics" / "gap_vs_newbad3.png", dpi=160)
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.scatter([r["selected_refined_mae"] for r in all_rows], [r["patho_new_bad3"] + r["clean_new_bad3"] for r in all_rows], s=8, alpha=0.5)
        plt.xlabel("selected MAE")
        plt.ylabel("patho+clean new-Bad3 %")
        plt.tight_layout()
        plt.savefig(args.output_root / "diagnostics" / "accuracy_safety_pareto.png", dpi=160)
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.scatter([r["multiplier"] for r in all_rows], [r["patho_new_bad3"] for r in all_rows], s=8, alpha=0.35)
        plt.xlabel("proposal multiplier")
        plt.ylabel("patho new-Bad3 %")
        plt.tight_layout()
        plt.savefig(args.output_root / "diagnostics" / "proposal_scale_vs_safety.png", dpi=160)
        plt.close()

        heat = np.full((len(CLIPS), len(THRESHOLDS)), np.nan)
        for i, c in enumerate(CLIPS):
            for j, t in enumerate(THRESHOLDS):
                vals = [r["patho_new_bad3"] for r in all_rows if r["model"] == "mpc" and r["multiplier"] == 1.0 and r["clip_px"] == c and r["threshold"] == t]
                if vals:
                    heat[i, j] = vals[0]
        plt.figure(figsize=(6, 4))
        plt.imshow(heat, aspect="auto", origin="lower")
        plt.xticks(range(len(THRESHOLDS)), THRESHOLDS)
        plt.yticks(range(len(CLIPS)), CLIPS)
        plt.xlabel("threshold")
        plt.ylabel("clip px")
        plt.colorbar(label="patho new-Bad3 %")
        plt.tight_layout()
        plt.savefig(args.output_root / "diagnostics" / "clip_threshold_heatmap.png", dpi=160)
        plt.close()
    except Exception:
        pass
    print(json.dumps({"rows": len(all_rows), "best_accuracy": best_accuracy, "best_safety": best_safety, "best_pareto": best_pareto}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
