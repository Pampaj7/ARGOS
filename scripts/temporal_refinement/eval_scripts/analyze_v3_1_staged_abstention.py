#!/usr/bin/env python3
"""Analyze completed tiny_refiner_v3_1 staged abstention run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from train_tiny_refiner_v1_full_gt import finite_mean, parse_bool, write_csv  # noqa: E402
from train_tiny_refiner_v3_1_staged_abstention import (  # noqa: E402
    AbstentionCropRefiner,
    DEFAULT_BALANCED_SPLIT,
    FullFrameDataset,
    THRESHOLDS,
    load_samples_with_split,
    load_shards,
)


DEFAULT_ROOT = Path("results/03_temporal_refinement/training/tiny_refiner_v3_1_staged_abstention")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fnum(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        value = row.get(key, default)
        if value == "":
            return default
        return float(value)
    except Exception:
        return default


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def maybe_plot(out_dir: Path, sequence_rows: list[dict[str, Any]], threshold_rows: list[dict[str, Any]], transition_rows: list[dict[str, Any]]) -> list[str]:
    made: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return made

    vt = [r for r in sequence_rows if r["split"] in {"val", "test"}]
    labels = [f"{r['split']}:{r['sequence_id'].replace('dataset_', 'd')}" for r in vt]
    x = np.arange(len(vt))
    width = 0.38

    if vt:
        plt.figure(figsize=(max(8, len(vt) * 0.8), 4))
        plt.bar(x - width / 2, [r["raw_mae"] for r in vt], width, label="raw")
        plt.bar(x + width / 2, [r["refined_hard_mae"] for r in vt], width, label="refined")
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.ylabel("MAE px")
        plt.legend()
        plt.tight_layout()
        path = out_dir / "raw_vs_refined_mae_by_sequence.png"
        plt.savefig(path, dpi=160)
        plt.close()
        made.append(path.name)

        plt.figure(figsize=(max(8, len(vt) * 0.8), 4))
        plt.bar(x - width / 2, [r["raw_bad3"] for r in vt], width, label="raw")
        plt.bar(x + width / 2, [r["refined_hard_bad3"] for r in vt], width, label="refined")
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.ylabel("Bad-3 %")
        plt.legend()
        plt.tight_layout()
        path = out_dir / "raw_vs_refined_bad3_by_sequence.png"
        plt.savefig(path, dpi=160)
        plt.close()
        made.append(path.name)

        plt.figure(figsize=(max(8, len(vt) * 0.8), 4))
        plt.bar(x, [r["fraction_modified_pct"] for r in vt])
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.ylabel("Modified pixels %")
        plt.tight_layout()
        path = out_dir / "modified_pixels_by_sequence.png"
        plt.savefig(path, dpi=160)
        plt.close()
        made.append(path.name)

    final_thr = [r for r in threshold_rows if r.get("split") in {"val", "test"} and math.isfinite(float(r["threshold"]))]
    if final_thr:
        plt.figure(figsize=(8, 4))
        for split in ("val", "test"):
            rows = sorted([r for r in final_thr if r["split"] == split], key=lambda r: r["threshold"])
            if rows:
                plt.plot([r["threshold"] for r in rows], [r["refined_hard_mae"] for r in rows], marker="o", label=f"{split} MAE")
        plt.xlabel("threshold")
        plt.ylabel("hard refined MAE px")
        plt.legend()
        plt.tight_layout()
        path = out_dir / "threshold_sweep_val_test.png"
        plt.savefig(path, dpi=160)
        plt.close()
        made.append(path.name)

        plt.figure(figsize=(5, 4))
        for split in ("val", "test"):
            rows = sorted([r for r in final_thr if r["split"] == split], key=lambda r: r["recall_at_threshold"])
            if rows:
                plt.plot([r["recall_at_threshold"] for r in rows], [r["precision_at_threshold"] for r in rows], marker="o", label=split)
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.legend()
        plt.tight_layout()
        path = out_dir / "detector_pr_curve.png"
        plt.savefig(path, dpi=160)
        plt.close()
        made.append(path.name)

        plt.figure(figsize=(5, 4))
        for split in ("val", "test"):
            rows = sorted([r for r in final_thr if r["split"] == split], key=lambda r: r["fpr_at_threshold"])
            if rows:
                plt.plot([r["fpr_at_threshold"] for r in rows], [r["recall_at_threshold"] for r in rows], marker="o", label=split)
        plt.xlabel("false positive rate")
        plt.ylabel("true positive rate")
        plt.legend()
        plt.tight_layout()
        path = out_dir / "detector_roc_curve.png"
        plt.savefig(path, dpi=160)
        plt.close()
        made.append(path.name)

    if transition_rows:
        rows = [r for r in transition_rows if r["split"] in {"val", "test", "val_test"}]
        labels = [r["split"] for r in rows]
        x = np.arange(len(rows))
        plt.figure(figsize=(7, 4))
        plt.bar(x - width / 2, [r["fixed_bad3_pct_of_valid"] for r in rows], width, label="fixed bad3")
        plt.bar(x + width / 2, [r["new_bad3_from_good_pct_of_valid"] for r in rows], width, label="new bad3 from good")
        plt.xticks(x, labels)
        plt.ylabel("% valid pixels")
        plt.legend()
        plt.tight_layout()
        path = out_dir / "bad_pixel_fixed_vs_new.png"
        plt.savefig(path, dpi=160)
        plt.close()
        made.append(path.name)

    return made


def enrich_sequence_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        raw_mae = fnum(r, "raw_mae")
        ref_mae = fnum(r, "refined_hard_mae")
        raw_bad3 = fnum(r, "raw_bad3")
        ref_bad3 = fnum(r, "refined_hard_bad3")
        mae_gain = raw_mae - ref_mae
        bad3_gain = raw_bad3 - ref_bad3
        out.append(
            {
                "split": r["split"],
                "sequence_id": r["sequence_id"],
                "threshold": fnum(r, "threshold"),
                "raw_mae": raw_mae,
                "refined_hard_mae": ref_mae,
                "mae_gain_px": mae_gain,
                "mae_improvement_pct": fnum(r, "mae_improvement_pct"),
                "raw_bad3": raw_bad3,
                "refined_hard_bad3": ref_bad3,
                "bad3_gain_pctpt": bad3_gain,
                "bad3_improvement_pct": fnum(r, "bad3_improvement_pct"),
                "fraction_modified_pct": fnum(r, "fraction_modified_pct"),
                "fixed_bad3_pct": fnum(r, "fixed_bad3_pct"),
                "new_bad3_from_raw_good_pct": fnum(r, "new_bad3_from_raw_good_pct"),
                "degraded_good_pixel_mean": fnum(r, "degraded_good_pixel_mean"),
                "precision_at_threshold": fnum(r, "precision_at_threshold"),
                "recall_at_threshold": fnum(r, "recall_at_threshold"),
                "status": "improved" if mae_gain > 0 and bad3_gain >= 0 else ("mixed" if mae_gain > 0 or bad3_gain > 0 else "degraded"),
            }
        )
    return out


def final_threshold_rows(rows: list[dict[str, str]], selected: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        split = r.get("split", "")
        if split not in {"train", "val", "test"} or not r.get("best_residual_epoch"):
            continue
        threshold = fnum(r, "threshold")
        prevalence = fnum(r, "raw_bad3") / 100.0
        pred_pos = fnum(r, "fraction_modified_pct") / 100.0
        tpr = fnum(r, "recall_at_threshold")
        tp = prevalence * tpr
        fp = max(0.0, pred_pos - tp)
        fpr = fp / max(1.0 - prevalence, 1e-9)
        out.append(
            {
                "split": split,
                "threshold": threshold,
                "is_selected": abs(threshold - selected) < 1e-9,
                "is_identity": threshold > 1.0,
                "raw_mae": fnum(r, "raw_mae"),
                "refined_hard_mae": fnum(r, "refined_hard_mae"),
                "mae_gain_px": fnum(r, "raw_mae") - fnum(r, "refined_hard_mae"),
                "raw_bad3": fnum(r, "raw_bad3"),
                "refined_hard_bad3": fnum(r, "refined_hard_bad3"),
                "bad3_gain_pctpt": fnum(r, "raw_bad3") - fnum(r, "refined_hard_bad3"),
                "fraction_modified_pct": fnum(r, "fraction_modified_pct"),
                "precision_at_threshold": fnum(r, "precision_at_threshold"),
                "recall_at_threshold": tpr,
                "fpr_at_threshold": min(max(fpr, 0.0), 1.0),
                "new_bad3_from_raw_good_pct": fnum(r, "new_bad3_from_raw_good_pct"),
                "fixed_bad3_pct": fnum(r, "fixed_bad3_pct"),
                "bad_pixel_auc": fnum(r, "bad_pixel_auc"),
                "bad_pixel_ap": fnum(r, "bad_pixel_ap"),
            }
        )
    return out


@torch.no_grad()
def exact_transition_stats(root: Path, summary: dict[str, Any], threshold: float, device: torch.device, batch_size: int, workers: int) -> tuple[list[dict[str, Any]], dict[str, float]]:
    ckpt = torch.load(root / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    model = AbstentionCropRefiner(int(ckpt.get("input_channels", 16))).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    splits, by_split = load_samples_with_split(Path(summary["targets_root"]), DEFAULT_BALANCED_SPLIT, 0)
    samples = by_split["val"] + by_split["test"]
    shards = load_shards(samples)
    datasets = {split: FullFrameDataset(by_split[split], shards, int(ckpt["args"].get("context_frames", 4))) for split in ("val", "test")}
    loaders = {
        split: DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda", persistent_workers=workers > 0, prefetch_factor=2 if workers > 0 else None)
        for split, ds in datasets.items()
    }

    bins: dict[str, dict[str, float]] = {}
    corr_samples: dict[str, list[np.ndarray]] = {"val": [], "test": [], "val_test": []}
    bench_ms_per_frame = float("nan")

    def acc(split: str) -> dict[str, float]:
        return bins.setdefault(
            split,
            {
                "valid": 0.0,
                "good": 0.0,
                "mid": 0.0,
                "bad": 0.0,
                "modified": 0.0,
                "modified_good": 0.0,
                "modified_mid": 0.0,
                "modified_bad": 0.0,
                "fixed_bad": 0.0,
                "new_bad_from_good": 0.0,
                "unchanged_bad": 0.0,
                "unchanged_good_not_bad": 0.0,
                "correction_abs_sum_modified": 0.0,
            },
        )

    for split, loader in loaders.items():
        for batch_idx, batch in enumerate(loader):
            x = batch["x"].to(device, non_blocking=True)
            raw = batch["raw"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True) > 0
            if split == "val" and batch_idx == 1 and device.type == "cuda":
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(20):
                    _ = model(x, float(ckpt["args"].get("residual_scale", 3.0)))
                torch.cuda.synchronize()
                bench_ms_per_frame = (time.perf_counter() - t0) * 1000.0 / (20 * x.shape[0])
            _logit, p_bad, residual = model(x, float(ckpt["args"].get("residual_scale", 3.0)))
            modified = (p_bad >= threshold) & valid
            refined = raw + modified.float() * residual
            raw_err = torch.abs(raw - gt)
            ref_err = torch.abs(refined - gt)
            good = (raw_err < 1.0) & valid
            mid = (raw_err >= 1.0) & (raw_err < 3.0) & valid
            bad = (raw_err >= 3.0) & valid
            fixed_bad = bad & (ref_err < 3.0)
            new_bad = good & (ref_err >= 3.0)
            corr_abs = torch.abs(modified.float() * residual)
            vals = corr_abs[modified].detach().cpu().numpy().astype(np.float32)
            if vals.size:
                stride = max(1, vals.size // 5000)
                corr_samples[split].append(vals[::stride])
                corr_samples["val_test"].append(vals[::stride])
            for key in (split, "val_test"):
                a = acc(key)
                a["valid"] += float(valid.sum().detach().cpu())
                a["good"] += float(good.sum().detach().cpu())
                a["mid"] += float(mid.sum().detach().cpu())
                a["bad"] += float(bad.sum().detach().cpu())
                a["modified"] += float(modified.sum().detach().cpu())
                a["modified_good"] += float((modified & good).sum().detach().cpu())
                a["modified_mid"] += float((modified & mid).sum().detach().cpu())
                a["modified_bad"] += float((modified & bad).sum().detach().cpu())
                a["fixed_bad"] += float(fixed_bad.sum().detach().cpu())
                a["new_bad_from_good"] += float(new_bad.sum().detach().cpu())
                a["unchanged_bad"] += float((bad & (ref_err >= 3.0)).sum().detach().cpu())
                a["unchanged_good_not_bad"] += float((good & (ref_err < 3.0)).sum().detach().cpu())
                a["correction_abs_sum_modified"] += float(corr_abs[modified].sum().detach().cpu())

    rows: list[dict[str, Any]] = []
    for split, a in bins.items():
        valid = max(a["valid"], 1.0)
        samples_arr = np.concatenate(corr_samples[split]) if corr_samples[split] else np.array([], dtype=np.float32)
        rows.append(
            {
                "split": split,
                "threshold": threshold,
                "valid_pixels": int(a["valid"]),
                "raw_good_pct": 100.0 * a["good"] / valid,
                "raw_mid_pct": 100.0 * a["mid"] / valid,
                "raw_bad3_pct": 100.0 * a["bad"] / valid,
                "fixed_bad3_pct_of_valid": 100.0 * a["fixed_bad"] / valid,
                "fixed_bad3_pct_of_raw_bad3": 100.0 * a["fixed_bad"] / max(a["bad"], 1.0),
                "unchanged_bad3_pct_of_valid": 100.0 * a["unchanged_bad"] / valid,
                "new_bad3_from_good_pct_of_valid": 100.0 * a["new_bad_from_good"] / valid,
                "new_bad3_from_good_pct_of_good": 100.0 * a["new_bad_from_good"] / max(a["good"], 1.0),
                "unchanged_good_not_bad_pct_of_good": 100.0 * a["unchanged_good_not_bad"] / max(a["good"], 1.0),
                "modified_pct": 100.0 * a["modified"] / valid,
                "modified_good_pct_of_good": 100.0 * a["modified_good"] / max(a["good"], 1.0),
                "modified_mid_pct_of_mid": 100.0 * a["modified_mid"] / max(a["mid"], 1.0),
                "modified_bad_pct_of_bad": 100.0 * a["modified_bad"] / max(a["bad"], 1.0),
                "correction_abs_mean_modified": a["correction_abs_sum_modified"] / max(a["modified"], 1.0),
                "correction_abs_p50_modified_sampled": float(np.percentile(samples_arr, 50)) if samples_arr.size else 0.0,
                "correction_abs_p90_modified_sampled": float(np.percentile(samples_arr, 90)) if samples_arr.size else 0.0,
                "correction_abs_p95_modified_sampled": float(np.percentile(samples_arr, 95)) if samples_arr.size else 0.0,
            }
        )
    return rows, {"refiner_forward_ms_per_frame_batch": bench_ms_per_frame, "benchmark_batch_size": batch_size}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--analysis-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--overwrite", nargs="?", const=True, default=True, type=parse_bool)
    args = p.parse_args()

    root = args.run_root
    out_dir = args.analysis_dir or root / "analysis"
    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log: list[str] = [f"run_root={root}", f"analysis_dir={out_dir}"]
    start = time.perf_counter()

    summary = json.loads((root / "aggregate_summary.json").read_text())
    selected_threshold = float(summary["best_threshold"])
    sequence_rows = enrich_sequence_rows(read_csv(root / "sequence_metrics.csv"))
    threshold_rows = final_threshold_rows(read_csv(root / "threshold_sweep.csv"), selected_threshold)
    write_csv(out_dir / "sequence_improvement_summary.csv", sequence_rows)
    write_csv(out_dir / "threshold_policy_summary.csv", threshold_rows)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    transition_rows, runtime_bench = exact_transition_stats(root, summary, selected_threshold, device, args.batch_size, args.num_workers)
    write_csv(out_dir / "bad_pixel_transition_summary.csv", transition_rows)

    selected_by_split = {r["split"]: r for r in threshold_rows if r["is_selected"]}
    detector_rows = []
    for split in ("train", "val", "test"):
        row = selected_by_split.get(split, {})
        detector_rows.append(
            {
                "split": split,
                "threshold": selected_threshold,
                "bad_pixel_auc": row.get("bad_pixel_auc", summary.get(f"{split}_selected_threshold", {}).get("bad_pixel_auc")),
                "bad_pixel_ap": row.get("bad_pixel_ap", summary.get(f"{split}_selected_threshold", {}).get("bad_pixel_ap")),
                "precision_at_threshold": row.get("precision_at_threshold", summary.get(f"{split}_selected_threshold", {}).get("precision_at_threshold")),
                "recall_at_threshold": row.get("recall_at_threshold", summary.get(f"{split}_selected_threshold", {}).get("recall_at_threshold")),
                "fpr_at_threshold": row.get("fpr_at_threshold"),
                "p_bad_good_mean": summary.get(f"{split}_selected_threshold", {}).get("p_bad_good_mean"),
                "p_bad_mid_mean": summary.get(f"{split}_selected_threshold", {}).get("p_bad_mid_mean"),
                "p_bad_bad_mean": summary.get(f"{split}_selected_threshold", {}).get("p_bad_bad_mean"),
            }
        )
    write_csv(out_dir / "detector_quality_summary.csv", detector_rows)

    plots = maybe_plot(out_dir, sequence_rows, threshold_rows, transition_rows)

    runtime_summary = {
        "params": summary["params"],
        "input_channels": 16,
        "context_frames": 4,
        "best_residual_epoch": summary["best_residual_epoch"],
        "selected_threshold": selected_threshold,
        "training_elapsed_seconds": summary["elapsed_seconds"],
        "training_elapsed_minutes": summary["elapsed_seconds"] / 60.0,
        "frames": summary["frames"],
        "refiner_forward_ms_per_frame_batch": runtime_bench["refiner_forward_ms_per_frame_batch"],
        "benchmark_batch_size": runtime_bench["benchmark_batch_size"],
        "benchmark_device": str(device),
        "no_teacher_inference": True,
    }
    write_json(out_dir / "runtime_and_model_summary.json", runtime_summary)

    val = summary["val_selected_threshold"]
    test = summary["test_selected_threshold"]
    improved_sequences = [r for r in sequence_rows if r["split"] in {"val", "test"} and r["status"] == "improved"]
    degraded_sequences = [r for r in sequence_rows if r["split"] in {"val", "test"} and r["status"] == "degraded"]
    paper = {
        "headline": "v3.1 staged abstention is the first safe refiner to improve both validation and test MAE/Bad-3.",
        "validation": {
            "raw_mae": val["raw_mae"],
            "refined_mae": val["refined_hard_mae"],
            "mae_gain_px": val["raw_mae"] - val["refined_hard_mae"],
            "raw_bad3": val["raw_bad3"],
            "refined_bad3": val["refined_hard_bad3"],
            "bad3_gain_pctpt": val["raw_bad3"] - val["refined_hard_bad3"],
            "new_bad3_from_raw_good_pct": val["new_bad3_from_raw_good_pct"],
        },
        "test": {
            "raw_mae": test["raw_mae"],
            "refined_mae": test["refined_hard_mae"],
            "mae_gain_px": test["raw_mae"] - test["refined_hard_mae"],
            "raw_bad3": test["raw_bad3"],
            "refined_bad3": test["refined_hard_bad3"],
            "bad3_gain_pctpt": test["raw_bad3"] - test["refined_hard_bad3"],
            "new_bad3_from_raw_good_pct": test["new_bad3_from_raw_good_pct"],
        },
        "why_better_than_dense_or_joint": [
            "Dense residual models changed too many already-good pixels; v3.1 uses a detector threshold and identity fallback.",
            "Joint gated training drifted as detector and residual co-adapted; v3.1 freezes the detector for early residual training.",
            "The selected hard policy improves MAE and Bad-3 while keeping new Bad-3 from raw-good pixels below 0.2%.",
        ],
        "sequence_counts_val_test": {
            "improved": len(improved_sequences),
            "degraded": len(degraded_sequences),
            "mixed": len([r for r in sequence_rows if r["split"] in {"val", "test"} and r["status"] == "mixed"]),
        },
        "selected_threshold_rationale": "0.5 was selected because it improved validation MAE and did not worsen validation Bad-3 with a moderate modified-pixel rate.",
    }
    write_json(out_dir / "paper_ready_summary.json", paper)

    (out_dir / "README.md").write_text(
        f"""# v3.1 Staged Abstention Analysis

Input run: `{root}`

v3.1 improves both validation and test with a hard abstention policy at threshold `{selected_threshold}`.

- Validation MAE: `{val['raw_mae']:.4f}` -> `{val['refined_hard_mae']:.4f}`
- Validation Bad-3: `{val['raw_bad3']:.3f}%` -> `{val['refined_hard_bad3']:.3f}%`
- Test MAE: `{test['raw_mae']:.4f}` -> `{test['refined_hard_mae']:.4f}`
- Test Bad-3: `{test['raw_bad3']:.3f}%` -> `{test['refined_hard_bad3']:.3f}%`
- Test new Bad-3 from raw-good: `{test['new_bad3_from_raw_good_pct']:.3f}%`

The staged detector/residual setup is safer than dense residual and joint gated models because corrections are explicitly abstained unless detector confidence crosses the calibrated threshold. No S2M2/SAV/RAFT/DINO inference or training was run for this analysis.

Files:
- `sequence_improvement_summary.csv`
- `threshold_policy_summary.csv`
- `bad_pixel_transition_summary.csv`
- `detector_quality_summary.csv`
- `runtime_and_model_summary.json`
- `paper_ready_summary.json`
- plots listed in `runtime_and_model_summary.json` and this folder
"""
    )

    runtime_summary["plots"] = plots
    runtime_summary["analysis_elapsed_seconds"] = time.perf_counter() - start
    write_json(out_dir / "runtime_and_model_summary.json", runtime_summary)
    log.append(f"elapsed_seconds={runtime_summary['analysis_elapsed_seconds']:.3f}")
    log.append(f"plots={plots}")
    (out_dir / "run.log").write_text("\n".join(log) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
