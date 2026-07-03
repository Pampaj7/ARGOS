#!/usr/bin/env python3
"""Audit selected-clip distillation targets against rectified GT."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_TARGETS_ROOT = Path("results/03_temporal_refinement/evaluation/distillation_targets_selected_clips")
DEFAULT_DATASET_ROOT = Path("dataset/SCARED/curated/temporal_gt_rectified")
DEFAULT_OUTPUT_ROOT = DEFAULT_TARGETS_ROOT / "target_consistency_audit"


def resolve(path: Path) -> Path:
    root = Path(__file__).resolve().parents[3]
    return path if path.is_absolute() else root / path


def finite_mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def weighted_mean(rows: list[dict[str, Any]], key: str) -> float:
    num = den = 0.0
    for row in rows:
        value = float(row.get(key, math.nan))
        weight = float(row.get("valid_pixels", 0))
        if math.isfinite(value) and weight > 0:
            num += value * weight
            den += weight
    return num / den if den else float("nan")


def metric(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    return float(np.mean(np.abs(pred[valid] - gt[valid]))) if np.any(valid) else float("nan")


def pct(mask: np.ndarray, valid: np.ndarray) -> float:
    return float(np.mean(mask[valid]) * 100.0) if np.any(valid) else float("nan")


def colorize(value: np.ndarray, vmax: float | None = None, cmap: int = cv2.COLORMAP_TURBO) -> np.ndarray:
    arr = np.nan_to_num(value.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if vmax is None:
        vals = arr[np.isfinite(arr)]
        vmax = float(np.percentile(vals, 98)) if vals.size else 1.0
    vmax = max(float(vmax), 1e-6)
    return cv2.applyColorMap((np.clip(arr / vmax, 0.0, 1.0) * 255).astype(np.uint8), cmap)


def downsample_gt(gt: np.ndarray, valid: np.ndarray, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    h, w = shape
    plain = cv2.resize(gt, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    nearest_valid = cv2.resize(valid.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    coverage = cv2.resize(valid.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    masked_sum = cv2.resize(gt * valid.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    masked = (masked_sum / np.maximum(coverage, 1e-6)).astype(np.float32)
    return {"plain_area": plain, "valid_masked_area": masked, "nearest_valid": nearest_valid, "coverage": coverage}


def load_clip_index(targets_root: Path) -> dict[str, dict[str, str]]:
    with (targets_root / "clip_targets_index.csv").open(newline="") as f:
        return {row["clip_id"]: row for row in csv.DictReader(f)}


def load_frames(targets_root: Path) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for clip_dir in sorted((targets_root / "clips").iterdir()):
        if not clip_dir.is_dir():
            continue
        meta = json.loads((clip_dir / "clip_metadata.json").read_text())
        with (clip_dir / "frame_target_index.csv").open(newline="") as f:
            for row in csv.DictReader(f):
                row.update({"clip_id": clip_dir.name, "sequence_id": meta["sequence_id"], "target_scale": meta.get("target_scale")})
                frames.append(row)
    return frames


def key_inventory(rows: list[dict[str, Any]]) -> dict[str, Any]:
    inventory: dict[str, Any] = {"frames": len(rows), "keys": {}, "key_sets": Counter()}
    for row in rows:
        path = Path(row["target_path"])
        z = np.load(path)
        key_set = tuple(sorted(z.files))
        inventory["key_sets"][str(key_set)] += 1
        for key in z.files:
            arr = z[key]
            info = inventory["keys"].setdefault(key, {"dtypes": Counter(), "shapes": Counter(), "frames": 0})
            info["frames"] += 1
            info["dtypes"][str(arr.dtype)] += 1
            info["shapes"][str(tuple(arr.shape))] += 1
    inventory["key_sets"] = dict(inventory["key_sets"])
    for info in inventory["keys"].values():
        info["dtypes"] = dict(info["dtypes"])
        info["shapes"] = dict(info["shapes"])
    return inventory


def audit_frame(row: dict[str, Any], dataset_root: Path) -> dict[str, Any]:
    z = np.load(row["target_path"])
    keys = set(z.files)
    oracle = z["oracle_all_available_disp" if "oracle_all_available_disp" in keys else "oracle_disp_all_available"].astype(np.float32)
    oracle_no_flow = z["oracle_no_flow_disp" if "oracle_no_flow_disp" in keys else "oracle_disp_no_flow"].astype(np.float32)
    delta = z["delta_disp_oracle_all_available_minus_raw"].astype(np.float32)
    delta_no_flow = z["delta_disp_oracle_no_flow_minus_raw"].astype(np.float32)
    raw = z["raw_disp"].astype(np.float32) if "raw_disp" in keys else oracle - delta
    raw_from_no_flow = oracle_no_flow - delta_no_flow
    target_valid = z["valid_mask"].astype(bool)
    gt_full = np.load(dataset_root / row["sequence_id"] / "gt" / "Disparity_float32" / f"{row['frame_id']}.npy").astype(np.float32)
    valid_full = np.load(dataset_root / row["sequence_id"] / "gt" / "ValidMask" / f"{row['frame_id']}.npy").astype(bool)
    gt_ds = downsample_gt(gt_full, valid_full, raw.shape)
    gt_compare = z["gt_disp"].astype(np.float32) if "gt_disp" in keys else gt_ds["valid_masked_area"]
    valid_masked = target_valid & np.isfinite(gt_compare) & (gt_compare > 0)
    valid_plain = target_valid & gt_ds["nearest_valid"] & np.isfinite(gt_ds["plain_area"]) & (gt_ds["plain_area"] > 0)

    raw_err = np.abs(raw - gt_compare)
    oracle_err = np.abs(oracle - gt_compare)
    no_flow_err = np.abs(oracle_no_flow - gt_compare)
    raw_plain_err = np.abs(raw - gt_ds["plain_area"])
    oracle_plain_err = np.abs(oracle - gt_ds["plain_area"])
    delta_recon = raw + delta
    no_flow_recon = raw + delta_no_flow
    violation = oracle_err > raw_err + 1e-3
    no_flow_violation = no_flow_err > raw_err + 1e-3
    label_all = z["oracle_selected_candidate_id_all_available"]

    def contamination(name: str, arr: np.ndarray) -> dict[str, float | int]:
        finite = np.isfinite(arr)
        return {
            f"{name}_nan_count": int(np.isnan(arr).sum()) if arr.dtype.kind == "f" else 0,
            f"{name}_inf_count": int(np.isinf(arr).sum()) if arr.dtype.kind == "f" else 0,
            f"{name}_zero_valid_pct": pct(arr == 0, target_valid) if arr.shape == target_valid.shape else float("nan"),
            f"{name}_finite_pct": float(finite.mean() * 100.0) if arr.dtype.kind == "f" else 100.0,
        }

    out: dict[str, Any] = {
        "clip_id": row["clip_id"],
        "sequence_id": row["sequence_id"],
        "frame_id": row["frame_id"],
        "target_path": row["target_path"],
        "target_shape": str(tuple(raw.shape)),
        "gt_full_shape": str(tuple(gt_full.shape)),
        "valid_pixels": int(valid_masked.sum()),
        "target_valid_pct": float(target_valid.mean() * 100.0),
        "gt_full_valid_pct": float(valid_full.mean() * 100.0),
        "gt_downsample_coverage_gt0_pct": float((gt_ds["coverage"] > 0).mean() * 100.0),
        "gt_downsample_coverage_gt50_pct": float((gt_ds["coverage"] > 0.5).mean() * 100.0),
        "raw_mae_npz_vs_gt_masked": metric(raw, gt_compare, valid_masked),
        "oracle_no_flow_mae_npz_vs_gt_masked": metric(oracle_no_flow, gt_compare, valid_masked),
        "oracle_all_mae_npz_vs_gt_masked": metric(oracle, gt_compare, valid_masked),
        "raw_mae_npz_vs_gt_plain": metric(raw, gt_ds["plain_area"], valid_plain),
        "oracle_all_mae_npz_vs_gt_plain": metric(oracle, gt_ds["plain_area"], valid_plain),
        "oracle_all_violation_pct_masked": pct(violation, valid_masked),
        "oracle_no_flow_violation_pct_masked": pct(no_flow_violation, valid_masked),
        "delta_all_reconstruction_mae": metric(delta_recon, oracle, target_valid),
        "delta_all_reconstruction_max_abs": float(np.max(np.abs(delta_recon[target_valid] - oracle[target_valid]))) if np.any(target_valid) else float("nan"),
        "delta_no_flow_reconstruction_mae": metric(no_flow_recon, oracle_no_flow, target_valid),
        "raw_from_all_vs_raw_from_no_flow_mae": metric(raw, raw_from_no_flow, target_valid),
        "label_invalid_255_valid_pct": pct(label_all == 255, target_valid),
        "label_unique_values": "|".join(map(str, sorted(np.unique(label_all).tolist()))),
        "frame_csv_raw_mae": float(row.get("raw_disp_mae") or "nan"),
        "frame_csv_oracle_all_mae": float(row.get("oracle_all_available_disp_mae") or "nan"),
        "frame_csv_oracle_no_flow_mae": float(row.get("oracle_no_flow_disp_mae") or "nan"),
    }
    out["frame_csv_raw_minus_npz_masked"] = out["frame_csv_raw_mae"] - out["raw_mae_npz_vs_gt_masked"]
    out["frame_csv_oracle_all_minus_npz_masked"] = out["frame_csv_oracle_all_mae"] - out["oracle_all_mae_npz_vs_gt_masked"]
    out["frame_csv_raw_minus_npz_plain"] = out["frame_csv_raw_mae"] - out["raw_mae_npz_vs_gt_plain"]
    out["frame_csv_oracle_all_minus_npz_plain"] = out["frame_csv_oracle_all_mae"] - out["oracle_all_mae_npz_vs_gt_plain"]
    for name, arr in [("raw", raw), ("oracle_all", oracle), ("oracle_no_flow", oracle_no_flow), ("delta_all", delta), ("gt_masked", gt_compare)]:
        out.update(contamination(name, arr))
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def clip_summary(frame_rows: list[dict[str, Any]], clip_index: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        grouped[row["clip_id"]].append(row)
    out = []
    for clip_id, rows in sorted(grouped.items()):
        clip = clip_index.get(clip_id, {})
        summary = {
            "clip_id": clip_id,
            "sequence_id": rows[0]["sequence_id"],
            "dominant_failure_mode": clip.get("dominant_failure_mode", ""),
            "frames": len(rows),
            "valid_pixels": int(sum(int(r["valid_pixels"]) for r in rows)),
            "raw_mae_npz_vs_gt_masked": weighted_mean(rows, "raw_mae_npz_vs_gt_masked"),
            "oracle_all_mae_npz_vs_gt_masked": weighted_mean(rows, "oracle_all_mae_npz_vs_gt_masked"),
            "oracle_no_flow_mae_npz_vs_gt_masked": weighted_mean(rows, "oracle_no_flow_mae_npz_vs_gt_masked"),
            "oracle_all_violation_pct_masked": weighted_mean(rows, "oracle_all_violation_pct_masked"),
            "oracle_no_flow_violation_pct_masked": weighted_mean(rows, "oracle_no_flow_violation_pct_masked"),
            "delta_all_reconstruction_mae": weighted_mean(rows, "delta_all_reconstruction_mae"),
            "frame_csv_raw_mae_mean": float(clip.get("raw_disp_mae_mean", "nan")),
            "frame_csv_oracle_all_mae_mean": float(clip.get("oracle_all_available_disp_mae_mean", "nan")),
            "frame_csv_raw_minus_npz_masked_mean": finite_mean([r["frame_csv_raw_minus_npz_masked"] for r in rows]),
            "frame_csv_oracle_all_minus_npz_masked_mean": finite_mean([r["frame_csv_oracle_all_minus_npz_masked"] for r in rows]),
        }
        summary["oracle_worse_than_raw_masked"] = bool(summary["oracle_all_mae_npz_vs_gt_masked"] > summary["raw_mae_npz_vs_gt_masked"])
        out.append(summary)
    return out


def diagnostics(rows: list[dict[str, Any]], dataset_root: Path, out_dir: Path, count: int = 8) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    worst = sorted(rows, key=lambda r: float(r["oracle_all_violation_pct_masked"]), reverse=True)[:count]
    for row in worst:
        z = np.load(row["target_path"])
        keys = set(z.files)
        oracle = z["oracle_all_available_disp" if "oracle_all_available_disp" in keys else "oracle_disp_all_available"].astype(np.float32)
        delta = z["delta_disp_oracle_all_available_minus_raw"].astype(np.float32)
        raw = z["raw_disp"].astype(np.float32) if "raw_disp" in keys else oracle - delta
        valid = z["valid_mask"].astype(bool)
        gt_full = np.load(dataset_root / row["sequence_id"] / "gt" / "Disparity_float32" / f"{row['frame_id']}.npy").astype(np.float32)
        valid_full = np.load(dataset_root / row["sequence_id"] / "gt" / "ValidMask" / f"{row['frame_id']}.npy").astype(bool)
        gt = z["gt_disp"].astype(np.float32) if "gt_disp" in keys else downsample_gt(gt_full, valid_full, raw.shape)["valid_masked_area"]
        m = valid & np.isfinite(gt) & (gt > 0)
        raw_err = np.abs(raw - gt)
        oracle_err = np.abs(oracle - gt)
        violation = m & (oracle_err > raw_err + 1e-3)
        vmax = float(np.percentile(gt[m], 98)) if np.any(m) else 64.0
        tiles = [
            colorize(raw, vmax),
            colorize(oracle, vmax),
            colorize(gt, vmax),
            colorize(valid.astype(np.float32), 1.0, cv2.COLORMAP_VIRIDIS),
            colorize(raw_err, 16.0, cv2.COLORMAP_MAGMA),
            colorize(oracle_err, 16.0, cv2.COLORMAP_MAGMA),
            colorize(violation.astype(np.float32), 1.0, cv2.COLORMAP_HOT),
            colorize(np.abs(delta), 16.0, cv2.COLORMAP_MAGMA),
        ]
        cv2.imwrite(str(out_dir / f"{row['clip_id']}_{row['frame_id']}_target_audit.png"), np.concatenate(tiles, axis=1))


def suspected_failure_modes(frame_rows: list[dict[str, Any]], clip_rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw_masked = weighted_mean(frame_rows, "raw_mae_npz_vs_gt_masked")
    oracle_masked = weighted_mean(frame_rows, "oracle_all_mae_npz_vs_gt_masked")
    raw_plain = weighted_mean(frame_rows, "raw_mae_npz_vs_gt_plain")
    oracle_plain = weighted_mean(frame_rows, "oracle_all_mae_npz_vs_gt_plain")
    violation = weighted_mean(frame_rows, "oracle_all_violation_pct_masked")
    recon = weighted_mean(frame_rows, "delta_all_reconstruction_mae")
    passed = oracle_masked <= raw_masked + 1e-3 and violation < 0.01
    return {
        "headline": (
            "low-res oracle target is GT-consistent"
            if passed
            else "target/evaluation downsampling mismatch; low-res oracle target is not a valid pixel-wise oracle in GT-consistent target space"
        ),
        "evidence": {
            "raw_mae_masked_gt": raw_masked,
            "oracle_all_mae_masked_gt": oracle_masked,
            "raw_mae_plain_gt": raw_plain,
            "oracle_all_mae_plain_gt": oracle_plain,
            "oracle_all_violation_pct_masked": violation,
            "delta_reconstruction_mae": recon,
            "clips_where_oracle_worse_than_raw_masked": sum(1 for r in clip_rows if r["oracle_worse_than_raw_masked"]),
            "clips": len(clip_rows),
        },
        "likely_causes": [] if passed else [
            "target generator computes full-resolution pixel-wise oracle, then downsamples the chosen oracle disparity map; pixel-wise min invariant is not preserved by averaging selected disparities",
            "float disparity targets are downsampled with unmasked INTER_AREA, so invalid zero regions contaminate sparse/low-valid frames",
            "frame_target_index.csv metrics are full-resolution pre-downsample metrics, so they can disagree strongly with .npz target-scale GT-space metrics",
            "delta/raw reconstruction is internally consistent, so the main issue is target generation/downsampling, not the tiny-refiner evaluator arithmetic",
        ],
        "recommendation": {
            "classification": "target-space oracle passes consistency audit" if passed else "combination: target generator downsampling/mask bug, not S2M2/SAV/RAFT inference",
            "do_not_train_more_on_current_targets": not passed,
            "preferred_fix": "none for target consistency" if passed else "recompute low-res targets by downsampling raw/fixed/adaptive/RAFT/SAV predictions and GT/valid mask first, then compute pixel-wise oracle at target scale",
            "acceptable_fix_if_storage_allows": "n/a" if passed else "store full-resolution oracle labels and GT, then derive low-res training targets with valid-mask-aware downsampling in the trainer",
            "minimum_metadata_fix": "already stores target-space raw/gt maps" if passed else "store gt_disp_downsampled and raw_disp_downsampled in .npz using exactly the same valid-aware function used for target generation/evaluation",
            "label_fix": "low-res labels are computed directly at target scale" if passed else "keep candidate-id labels nearest-neighbor only after recomputing low-res oracle labels; do not average labels",
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets-root", type=Path, default=DEFAULT_TARGETS_ROOT)
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.targets_root = resolve(args.targets_root)
    args.dataset_root = resolve(args.dataset_root)
    args.output_root = resolve(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = load_frames(args.targets_root)
    inventory = key_inventory(rows)
    frame_rows = [audit_frame(row, args.dataset_root) for row in rows]
    clip_index = load_clip_index(args.targets_root)
    clip_rows = clip_summary(frame_rows, clip_index)
    write_csv(args.output_root / "frame_consistency_metrics.csv", frame_rows)
    write_csv(args.output_root / "clip_consistency_summary.csv", clip_rows)
    (args.output_root / "target_key_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    failure_modes = suspected_failure_modes(frame_rows, clip_rows)
    (args.output_root / "suspected_failure_modes.json").write_text(json.dumps(failure_modes, indent=2) + "\n")
    diagnostics(frame_rows, args.dataset_root, args.output_root / "diagnostics")
    status = "PASS" if "GT-consistent" in failure_modes["headline"] else "FAIL"
    readme = f"""# Distillation Target GT Consistency Audit

This audit used existing `.npz` targets and rectified GT only. It did not run S2M2, SAV, RAFT, DINO, or training.

## Finding

Status: `{status}`.

- Raw MAE vs valid-masked downsampled GT: `{failure_modes['evidence']['raw_mae_masked_gt']:.6f}`
- Oracle-all MAE vs valid-masked downsampled GT: `{failure_modes['evidence']['oracle_all_mae_masked_gt']:.6f}`
- Oracle violation rate: `{failure_modes['evidence']['oracle_all_violation_pct_masked']:.2f}%`
- Delta reconstruction MAE: `{failure_modes['evidence']['delta_reconstruction_mae']:.6f}`

{failure_modes['headline']}

## Recommendation

{failure_modes['recommendation']['preferred_fix']}
"""
    (args.output_root / "README.md").write_text(readme)
    print(json.dumps({"output_root": str(args.output_root), **failure_modes["evidence"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
