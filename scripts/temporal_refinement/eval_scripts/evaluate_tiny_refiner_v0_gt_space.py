#!/usr/bin/env python3
"""Evaluate tiny_refiner_v0 against rectified GT at target scale."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.temporal_refinement.train_tiny_refiner_v0 import DISP_SCALE, TinyRefinerV0, colorize  # noqa: E402


DEFAULT_TARGETS_ROOT = ROOT / "results/03_temporal_refinement/evaluation/distillation_targets_selected_clips"
DEFAULT_CHECKPOINT = ROOT / "results/03_temporal_refinement/training/tiny_refiner_v0_smoke/checkpoints/best.pt"
DEFAULT_DATASET_ROOT = ROOT / "dataset/SCARED/curated/temporal_gt_rectified"
DEFAULT_OUTPUT_ROOT = ROOT / "results/03_temporal_refinement/training/tiny_refiner_v0_smoke/gt_space_eval"


@dataclass(frozen=True)
class FrameSample:
    split: str
    clip_id: str
    sequence_id: str
    frame_id: str
    target_path: Path
    prev_target_path: Path | None


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def finite_mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def weighted_mean(rows: list[dict[str, Any]], key: str) -> float:
    num = den = 0.0
    for row in rows:
        v = float(row.get(key, math.nan))
        w = float(row.get("valid_pixels", 0))
        if math.isfinite(v) and w > 0:
            num += v * w
            den += w
    return num / den if den else float("nan")


def load_splits(checkpoint: dict[str, Any], targets_root: Path) -> dict[str, list[str]]:
    splits = checkpoint.get("splits")
    if splits:
        return {str(k): list(map(str, v)) for k, v in splits.items()}
    with (targets_root / "clip_targets_index.csv").open(newline="") as f:
        ids = [row["clip_id"] for row in csv.DictReader(f)]
    return {"train": ids[:4], "val": ids[4:5], "test": ids[5:6]}


def load_samples(targets_root: Path, splits: dict[str, list[str]]) -> list[FrameSample]:
    samples: list[FrameSample] = []
    for split, clip_ids in splits.items():
        if split == "train":
            continue
        for clip_id in clip_ids:
            clip_dir = targets_root / "clips" / clip_id
            meta = json.loads((clip_dir / "clip_metadata.json").read_text())
            sequence_id = str(meta["sequence_id"])
            with (clip_dir / "frame_target_index.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            prev: Path | None = None
            for row in rows:
                path = Path(row["target_path"])
                samples.append(FrameSample(split, clip_id, sequence_id, row["frame_id"], path, prev))
                prev = path
    return samples


def load_target(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path)
    oracle_key = "oracle_all_available_disp" if "oracle_all_available_disp" in z.files else "oracle_disp_all_available"
    oracle = z[oracle_key].astype(np.float32)
    delta = z["delta_disp_oracle_all_available_minus_raw"].astype(np.float32)
    raw = z["raw_disp"].astype(np.float32) if "raw_disp" in z.files else oracle - delta
    gt = z["gt_disp"].astype(np.float32) if "gt_disp" in z.files else None
    return {
        "raw": raw,
        "oracle": oracle,
        "delta": delta,
        "valid": z["valid_mask"].astype(bool),
        "gt": gt,
        "confidence": z["raw_confidence_binary"].astype(np.float32),
        "high_boundary": z["high_boundary_error_mask"].astype(bool) if "high_boundary_error_mask" in z.files else np.zeros_like(raw, dtype=bool),
    }


def model_input(cur: dict[str, np.ndarray], prev: dict[str, np.ndarray] | None) -> torch.Tensor:
    prev_raw = cur["raw"] if prev is None else prev["raw"]
    abs_dt = np.abs(cur["raw"] - prev_raw)
    x = np.stack([cur["raw"] / DISP_SCALE, prev_raw / DISP_SCALE, abs_dt / DISP_SCALE, cur["valid"].astype(np.float32)], axis=0)
    return torch.from_numpy(x[None].astype(np.float32))


def read_gt(dataset_root: Path, sequence_id: str, frame_id: str, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    seq = dataset_root / sequence_id
    gt = np.load(seq / "gt" / "Disparity_float32" / f"{frame_id}.npy").astype(np.float32)
    valid_path = seq / "gt" / "ValidMask" / f"{frame_id}.npy"
    valid_full = np.load(valid_path).astype(bool)
    h, w = shape
    cover = cv2.resize(valid_full.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    gt_sum = cv2.resize(gt * valid_full.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    gt_small = (gt_sum / np.maximum(cover, 1e-6)).astype(np.float32)
    valid_small = cover > 0.5
    return gt_small, valid_small


def gt_boundary_mask(gt: np.ndarray, valid: np.ndarray, percentile: float = 80.0) -> np.ndarray:
    gx = np.zeros_like(gt, dtype=np.float32)
    gy = np.zeros_like(gt, dtype=np.float32)
    gx[:, 1:] = np.abs(gt[:, 1:] - gt[:, :-1])
    gy[1:, :] = np.abs(gt[1:, :] - gt[:-1, :])
    grad = np.maximum(gx, gy)
    m = valid & np.isfinite(grad)
    if not np.any(m):
        return np.zeros_like(valid)
    return m & (grad >= float(np.percentile(grad[m], percentile)))


def mae(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    return float(np.mean(np.abs(pred[valid] - gt[valid]))) if np.any(valid) else float("nan")


def bad(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, tau: float) -> float:
    return float(np.mean(np.abs(pred[valid] - gt[valid]) > tau) * 100.0) if np.any(valid) else float("nan")


@torch.no_grad()
def eval_samples(model: TinyRefinerV0, samples: list[FrameSample], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous_by_clip: dict[str, dict[str, np.ndarray]] = {}
    model.eval()
    for sample in samples:
        cur = load_target(sample.target_path)
        prev = previous_by_clip.get(sample.clip_id)
        x = model_input(cur, prev).to(args.device_obj)
        pred_delta, _conf_logits = model(x)
        pred_delta_np = pred_delta[0, 0].detach().float().cpu().numpy()
        refined = cur["raw"] + pred_delta_np
        gt, gt_valid = (cur["gt"], cur["valid"]) if cur["gt"] is not None else read_gt(args.dataset_root, sample.sequence_id, sample.frame_id, cur["raw"].shape)
        valid = cur["valid"] & gt_valid & np.isfinite(gt) & (gt > 0)
        boundary = gt_boundary_mask(gt, valid)
        temporal_pair = prev is not None
        row: dict[str, Any] = {
            "split": sample.split,
            "clip_id": sample.clip_id,
            "sequence_id": sample.sequence_id,
            "frame_id": sample.frame_id,
            "valid_pixels": int(valid.sum()),
            "raw_disp_mae_vs_gt": mae(cur["raw"], gt, valid),
            "refined_disp_mae_vs_gt": mae(refined, gt, valid),
            "oracle_all_available_disp_mae_vs_gt": mae(cur["oracle"], gt, valid),
            "raw_bad_1px": bad(cur["raw"], gt, valid, 1.0),
            "refined_bad_1px": bad(refined, gt, valid, 1.0),
            "oracle_bad_1px": bad(cur["oracle"], gt, valid, 1.0),
            "raw_bad_3px": bad(cur["raw"], gt, valid, 3.0),
            "refined_bad_3px": bad(refined, gt, valid, 3.0),
            "oracle_bad_3px": bad(cur["oracle"], gt, valid, 3.0),
            "residual_l1_to_oracle": mae(pred_delta_np, cur["delta"], valid),
            "boundary_raw_mae": mae(cur["raw"], gt, boundary),
            "boundary_refined_mae": mae(refined, gt, boundary),
            "boundary_oracle_mae": mae(cur["oracle"], gt, boundary),
            "temporal_pair": temporal_pair,
            "raw_temporal_diff": float("nan"),
            "refined_temporal_diff": float("nan"),
            "oracle_temporal_diff": float("nan"),
            "gt_temporal_diff": float("nan"),
        }
        den = row["raw_disp_mae_vs_gt"] - row["oracle_all_available_disp_mae_vs_gt"]
        row["oracle_gap_recovered"] = (
            (row["raw_disp_mae_vs_gt"] - row["refined_disp_mae_vs_gt"]) / den if math.isfinite(den) and abs(den) > 1e-8 else float("nan")
        )
        if temporal_pair:
            prev_valid = prev["valid"] & gt_valid
            tv = valid & prev_valid
            row["raw_temporal_diff"] = float(np.mean(np.abs(cur["raw"][tv] - prev["raw"][tv]))) if np.any(tv) else float("nan")
            row["refined_temporal_diff"] = float(np.mean(np.abs(refined[tv] - prev["refined"][tv]))) if np.any(tv) else float("nan")
            row["oracle_temporal_diff"] = float(np.mean(np.abs(cur["oracle"][tv] - prev["oracle"][tv]))) if np.any(tv) else float("nan")
            row["gt_temporal_diff"] = float(np.mean(np.abs(gt[tv] - prev["gt"][tv]))) if np.any(tv) else float("nan")
        rows.append(row)
        previous_by_clip[sample.clip_id] = {**cur, "refined": refined, "gt": gt}
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, Any]], include_by_split: bool = True) -> dict[str, Any]:
    out: dict[str, Any] = {"frames": len(rows), "valid_pixels": int(sum(int(r["valid_pixels"]) for r in rows))}
    keys = [
        "raw_disp_mae_vs_gt", "refined_disp_mae_vs_gt", "oracle_all_available_disp_mae_vs_gt",
        "raw_bad_1px", "refined_bad_1px", "oracle_bad_1px",
        "raw_bad_3px", "refined_bad_3px", "oracle_bad_3px",
        "residual_l1_to_oracle", "boundary_raw_mae", "boundary_refined_mae", "boundary_oracle_mae",
        "raw_temporal_diff", "refined_temporal_diff", "oracle_temporal_diff", "gt_temporal_diff",
    ]
    for key in keys:
        out[key] = weighted_mean(rows, key)
    den = out["raw_disp_mae_vs_gt"] - out["oracle_all_available_disp_mae_vs_gt"]
    out["oracle_gap_recovered"] = (
        (out["raw_disp_mae_vs_gt"] - out["refined_disp_mae_vs_gt"]) / den if math.isfinite(den) and abs(den) > 1e-8 else float("nan")
    )
    out["oracle_better_than_raw_gt_space"] = bool(out["oracle_all_available_disp_mae_vs_gt"] < out["raw_disp_mae_vs_gt"])
    out["oracle_gap_note"] = (
        "standard gap recovered is meaningful"
        if out["oracle_better_than_raw_gt_space"]
        else "oracle target is worse than raw in GT-space here; gap recovered has a negative denominator"
    )
    if include_by_split:
        out["by_split"] = {
            split: aggregate([r for r in rows if r["split"] == split], include_by_split=False)
            for split in sorted({r["split"] for r in rows})
        }
    return out


def write_diagnostics(model: TinyRefinerV0, samples: list[FrameSample], args: argparse.Namespace, count: int = 4) -> None:
    args.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for sample in samples[:count]:
            cur = load_target(sample.target_path)
            prev = load_target(sample.prev_target_path) if sample.prev_target_path else None
            pred_delta, _ = model(model_input(cur, prev).to(args.device_obj))
            pred_delta_np = pred_delta[0, 0].detach().float().cpu().numpy()
            refined = cur["raw"] + pred_delta_np
            gt, gt_valid = (cur["gt"], cur["valid"]) if cur["gt"] is not None else read_gt(args.dataset_root, sample.sequence_id, sample.frame_id, cur["raw"].shape)
            valid = cur["valid"] & gt_valid & np.isfinite(gt) & (gt > 0)
            vmax = float(np.percentile(gt[valid], 98)) if np.any(valid) else 64.0
            tiles = [
                colorize(cur["raw"], vmax),
                colorize(refined, vmax),
                colorize(gt, vmax),
                colorize(cur["oracle"], vmax),
                colorize(np.abs(cur["raw"] - gt), 16.0, cv2.COLORMAP_MAGMA),
                colorize(np.abs(refined - gt), 16.0, cv2.COLORMAP_MAGMA),
                colorize(np.abs(cur["oracle"] - gt), 16.0, cv2.COLORMAP_MAGMA),
                colorize(np.abs(pred_delta_np), 16.0, cv2.COLORMAP_MAGMA),
                colorize(np.abs(cur["delta"]), 16.0, cv2.COLORMAP_MAGMA),
            ]
            cv2.imwrite(str(args.diagnostics_dir / f"{sample.split}_{sample.clip_id}_{sample.frame_id}.png"), np.concatenate(tiles, axis=1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets-root", type=Path, default=DEFAULT_TARGETS_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.targets_root = resolve(args.targets_root)
    args.checkpoint = resolve(args.checkpoint)
    args.dataset_root = resolve(args.dataset_root)
    args.output_root = resolve(args.output_root)
    args.diagnostics_dir = args.output_root / "diagnostics"
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.device_obj = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=args.device_obj, weights_only=False)
    model = TinyRefinerV0(int(checkpoint.get("input_channels", 4))).to(args.device_obj)
    model.load_state_dict(checkpoint["model_state_dict"])
    splits = load_splits(checkpoint, args.targets_root)
    samples = load_samples(args.targets_root, splits)
    rows = eval_samples(model, samples, args)
    write_csv(args.output_root / "frame_metrics.csv", rows)
    summary = {
        "targets_root": str(args.targets_root),
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "splits_evaluated": sorted({s.split for s in samples}),
        "gt_alignment": "saved gt_disp and valid_mask from corrected low-resolution .npz targets; dataset GT resize is fallback only for older targets",
        **aggregate(rows),
    }
    (args.output_root / "aggregate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_diagnostics(model, [s for s in samples if s.split == "val"] + [s for s in samples if s.split == "test"], args)
    readme = f"""# Tiny Refiner v0 GT-Space Evaluation

This evaluation loads existing compressed targets, the trained tiny refiner checkpoint, and rectified GT disparity. It does not run S2M2, SAV, RAFT, DINO, or teacher inference.

- Targets: `{args.targets_root}`
- Checkpoint: `{args.checkpoint}`
- Dataset: `{args.dataset_root}`
- GT alignment: saved `gt_disp` and `valid_mask` from corrected low-resolution `.npz` targets; dataset GT resize is fallback only for older targets.
- Evaluated splits: `{summary['splits_evaluated']}`

Key aggregate:
- raw MAE vs GT: `{summary['raw_disp_mae_vs_gt']:.6f}`
- refined MAE vs GT: `{summary['refined_disp_mae_vs_gt']:.6f}`
- oracle MAE vs GT: `{summary['oracle_all_available_disp_mae_vs_gt']:.6f}`
- oracle gap recovered: `{summary['oracle_gap_recovered']:.6f}`

Note: `{summary['oracle_gap_note']}`.
"""
    (args.output_root / "README.md").write_text(readme)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
