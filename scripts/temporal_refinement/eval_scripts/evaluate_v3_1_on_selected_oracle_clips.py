#!/usr/bin/env python3
"""Evaluate v3.1 staged abstention on selected oracle distillation clips."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from train_tiny_refiner_v1_full_gt import colorize, parse_bool, write_csv  # noqa: E402
from train_tiny_refiner_v3_1_staged_abstention import AbstentionCropRefiner, make_features_from_raws  # noqa: E402


DEFAULT_MODEL = Path("results/03_temporal_refinement/training/tiny_refiner_v3_1_staged_abstention/checkpoints/best.pt")
DEFAULT_TARGETS = Path("results/03_temporal_refinement/evaluation/distillation_targets_selected_clips")
DEFAULT_OUTPUT = Path("results/03_temporal_refinement/training/tiny_refiner_v3_1_staged_abstention/selected_oracle_eval")
METHOD_KEYS = {
    "raw": "raw_disp",
    "refined": "refined_disp",
    "oracle_all_available": "oracle_all_available_disp",
    "sav": "sav_disp",
    "raftsmall": "raftsmall_disp",
    "oracle_no_flow": "oracle_no_flow_disp",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def safe_mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def metric(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    mask = valid > 0
    n = int(mask.sum())
    if n == 0 or pred is None:
        return {"mae": float("nan"), "bad1": float("nan"), "bad3": float("nan"), "valid_pixels": 0}
    err = np.abs(pred.astype(np.float32) - gt.astype(np.float32))[mask]
    return {
        "mae": float(err.mean()),
        "bad1": float((err > 1.0).mean() * 100.0),
        "bad3": float((err > 3.0).mean() * 100.0),
        "valid_pixels": n,
    }


def auc_ap(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        if len(np.unique(labels)) < 2:
            return float("nan"), float("nan")
        return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))
    except Exception:
        return float("nan"), float("nan")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def make_x(raws: np.ndarray, valids: np.ndarray, idx: int, context_frames: int) -> np.ndarray:
    ids = [max(0, idx - i) for i in range(context_frames)]
    x, _edge, _var = make_features_from_raws(raws[ids].astype(np.float32), valids[ids].astype(np.float32))
    return x


@torch.no_grad()
def predict_clip(model: torch.nn.Module, raws: np.ndarray, valids: np.ndarray, args: argparse.Namespace, device: torch.device, residual_scale: float, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    refined: list[np.ndarray] = []
    p_bad: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for start in range(0, len(raws), args.batch_size):
        end = min(len(raws), start + args.batch_size)
        x = np.stack([make_x(raws, valids, i, args.context_frames) for i in range(start, end)], axis=0)
        xt = torch.from_numpy(x).to(device)
        _logit, prob, residual = model(xt, residual_scale)
        prob_np = prob[:, 0].detach().cpu().numpy().astype(np.float32)
        res_np = residual[:, 0].detach().cpu().numpy().astype(np.float32)
        hard = (prob_np >= threshold).astype(np.float32)
        refined_np = raws[start:end].astype(np.float32) + hard * res_np
        refined.append(refined_np)
        p_bad.append(prob_np)
        residuals.append(res_np)
        masks.append(hard)
    return np.concatenate(refined), np.concatenate(p_bad), np.concatenate(residuals), np.concatenate(masks)


def diagnostic(path: Path, raw: np.ndarray, refined: np.ndarray, gt: np.ndarray, oracle: np.ndarray, sav: np.ndarray | None, valid: np.ndarray, p_bad: np.ndarray, mask: np.ndarray) -> None:
    v = valid > 0
    vmax = float(np.percentile(gt[v], 98)) if v.any() else 64.0
    raw_err = np.abs(raw - gt)
    ref_err = np.abs(refined - gt)
    oracle_err = np.abs(oracle - gt)
    tiles = [
        colorize(raw, vmax),
        colorize(refined, vmax),
        colorize(gt, vmax),
        colorize(oracle, vmax),
        colorize(sav if sav is not None else np.zeros_like(raw), vmax),
        colorize(raw_err, 16.0, cv2.COLORMAP_MAGMA),
        colorize(ref_err, 16.0, cv2.COLORMAP_MAGMA),
        colorize(oracle_err, 16.0, cv2.COLORMAP_MAGMA),
        colorize(p_bad, 1.0, cv2.COLORMAP_VIRIDIS),
        colorize(mask, 1.0, cv2.COLORMAP_VIRIDIS),
        colorize(np.clip(raw_err - ref_err, -8.0, 8.0), 8.0, cv2.COLORMAP_TURBO),
    ]
    cv2.imwrite(str(path), np.concatenate(tiles, axis=1))


def summarize(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key in sorted({r[group_key] for r in rows}):
        sub = [r for r in rows if r[group_key] == key]
        row: dict[str, Any] = {group_key: key, "frames": len(sub)}
        for method in ("raw", "refined", "oracle_all_available", "sav", "raftsmall", "oracle_no_flow"):
            row[f"{method}_mae"] = safe_mean([r.get(f"{method}_mae", float("nan")) for r in sub])
            row[f"{method}_bad1"] = safe_mean([r.get(f"{method}_bad1", float("nan")) for r in sub])
            row[f"{method}_bad3"] = safe_mean([r.get(f"{method}_bad3", float("nan")) for r in sub])
        raw = row["raw_mae"]
        ref = row["refined_mae"]
        oracle = row["oracle_all_available_mae"]
        sav = row["sav_mae"]
        row["oracle_gap_recovered"] = (raw - ref) / (raw - oracle) if math.isfinite(oracle) and raw > oracle else float("nan")
        row["sav_gap_recovered"] = (raw - ref) / (raw - sav) if math.isfinite(sav) and raw > sav else float("nan")
        row["modified_pixels_pct"] = safe_mean([r["modified_pixels_pct"] for r in sub])
        row["new_bad3_from_raw_good_pct"] = safe_mean([r["new_bad3_from_raw_good_pct"] for r in sub])
        row["fixed_bad3_pct"] = safe_mean([r["fixed_bad3_pct"] for r in sub])
        row["detector_auc"] = safe_mean([r["detector_auc"] for r in sub])
        row["detector_ap"] = safe_mean([r["detector_ap"] for r in sub])
        if group_key != "dominant_failure_mode":
            row["dominant_failure_mode"] = sub[0].get("dominant_failure_mode", "")
            row["sequence_id"] = sub[0].get("sequence_id", "")
        out.append(row)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-checkpoint", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--targets-root", type=Path, default=DEFAULT_TARGETS)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--context-frames", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-auc-pixels-per-clip", type=int, default=250000)
    p.add_argument("--diagnostics-per-clip", type=int, default=2)
    p.add_argument("--overwrite", nargs="?", const=True, default=True, type=parse_bool)
    args = p.parse_args()

    if args.output_root.exists() and not args.overwrite:
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "diagnostics").mkdir(exist_ok=True)
    log = [f"model_checkpoint={args.model_checkpoint}", f"targets_root={args.targets_root}"]
    start_time = time.perf_counter()

    ckpt = torch.load(args.model_checkpoint, map_location="cpu", weights_only=False)
    threshold = float(ckpt.get("threshold", 0.5))
    residual_scale = float(ckpt.get("args", {}).get("residual_scale", 3.0))
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = AbstentionCropRefiner(int(ckpt.get("input_channels", 16))).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    clip_index = {r["clip_id"]: r for r in read_csv(args.targets_root / "clip_targets_index.csv")}
    frame_rows: list[dict[str, Any]] = []
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for clip_dir in sorted((args.targets_root / "clips").iterdir()):
        if not clip_dir.is_dir():
            continue
        clip_id = clip_dir.name
        meta = json.loads((clip_dir / "clip_metadata.json").read_text())
        index = read_csv(clip_dir / "frame_target_index.csv")
        frames = []
        for row in index:
            z = np.load(row["target_path"])
            frames.append({k: z[k] for k in z.files} | {"frame_id": row["frame_id"]})
        raws = np.stack([f["raw_disp"].astype(np.float32) for f in frames])
        gts = np.stack([f["gt_disp"].astype(np.float32) for f in frames])
        valids = np.stack([f["valid_mask"].astype(np.float32) for f in frames])
        refined, p_bad, residual, correction_mask = predict_clip(model, raws, valids, args, device, residual_scale, threshold)
        labels = ((np.abs(raws - gts) >= 3.0) & (valids > 0)).reshape(-1).astype(np.uint8)
        scores = p_bad.reshape(-1)
        valid_flat = (valids > 0).reshape(-1)
        pick = np.flatnonzero(valid_flat)
        if pick.size:
            take = min(args.max_auc_pixels_per_clip, pick.size)
            pick = pick[np.linspace(0, pick.size - 1, take, dtype=np.int64)]
            all_labels.append(labels[pick])
            all_scores.append(scores[pick])
        clip_auc, clip_ap = auc_ap(scores[pick], labels[pick]) if pick.size else (float("nan"), float("nan"))

        diag_ids = set(np.linspace(0, len(frames) - 1, min(args.diagnostics_per_clip, len(frames)), dtype=int).tolist())
        for i, f in enumerate(frames):
            valid = valids[i] > 0
            raw_err = np.abs(raws[i] - gts[i])
            ref_err = np.abs(refined[i] - gts[i])
            good = valid & (raw_err < 1.0)
            bad = valid & (raw_err >= 3.0)
            row: dict[str, Any] = {
                "clip_id": clip_id,
                "sequence_id": meta["sequence_id"],
                "frame_id": f["frame_id"],
                "dominant_failure_mode": clip_index.get(clip_id, {}).get("dominant_failure_mode", ""),
                "threshold": threshold,
                "modified_pixels_pct": float(correction_mask[i][valid].mean() * 100.0) if valid.any() else float("nan"),
                "new_bad3_from_raw_good_pct": float(((good & (ref_err >= 3.0)).sum() / max(good.sum(), 1)) * 100.0),
                "fixed_bad3_pct": float(((bad & (ref_err < 3.0)).sum() / max(bad.sum(), 1)) * 100.0),
                "detector_auc": clip_auc,
                "detector_ap": clip_ap,
            }
            maps = {"raw_disp": raws[i], "refined_disp": refined[i], **f}
            for method, key in METHOD_KEYS.items():
                if key in maps:
                    m = metric(maps[key], gts[i], valid)
                    row[f"{method}_mae"] = m["mae"]
                    row[f"{method}_bad1"] = m["bad1"]
                    row[f"{method}_bad3"] = m["bad3"]
            raw = row["raw_mae"]
            ref = row["refined_mae"]
            oracle = row.get("oracle_all_available_mae", float("nan"))
            sav = row.get("sav_mae", float("nan"))
            row["oracle_gap_recovered"] = (raw - ref) / (raw - oracle) if math.isfinite(oracle) and raw > oracle else float("nan")
            row["sav_gap_recovered"] = (raw - ref) / (raw - sav) if math.isfinite(sav) and raw > sav else float("nan")
            frame_rows.append(row)
            if i in diag_ids:
                diagnostic(
                    args.output_root / "diagnostics" / f"{clip_id}_{f['frame_id']}.png",
                    raws[i],
                    refined[i],
                    gts[i],
                    f["oracle_all_available_disp"].astype(np.float32),
                    f["sav_disp"].astype(np.float32) if "sav_disp" in f else None,
                    valids[i],
                    p_bad[i],
                    correction_mask[i],
                )

    clip_rows = summarize(frame_rows, "clip_id")
    failure_rows = summarize(frame_rows, "dominant_failure_mode")
    write_csv(args.output_root / "frame_metrics.csv", frame_rows)
    write_csv(args.output_root / "clip_metrics.csv", clip_rows)
    write_csv(args.output_root / "failure_mode_summary.csv", failure_rows)
    overall_auc, overall_ap = auc_ap(np.concatenate(all_scores), np.concatenate(all_labels)) if all_scores else (float("nan"), float("nan"))
    aggregate = summarize(frame_rows, "dominant_failure_mode")
    all_row = summarize([{**r, "all": "all"} for r in frame_rows], "all")[0]
    all_row["detector_auc"] = overall_auc
    all_row["detector_ap"] = overall_ap
    summary = {
        "model_checkpoint": str(args.model_checkpoint),
        "targets_root": str(args.targets_root),
        "output_root": str(args.output_root),
        "threshold": threshold,
        "residual_scale": residual_scale,
        "frames": len(frame_rows),
        "clips": len(clip_rows),
        "aggregate": all_row,
        "by_failure_mode": aggregate,
        "elapsed_seconds": time.perf_counter() - start_time,
        "no_teacher_inference": True,
    }
    write_json(args.output_root / "aggregate_summary.json", summary)
    (args.output_root / "README.md").write_text(
        f"""# v3.1 on Selected Oracle Clips

Evaluates `{args.model_checkpoint}` on selected teacher/oracle target clips in `{args.targets_root}`.
No S2M2, SAV, RAFT, DINO, or training was run.

Key aggregate:
- Raw MAE: `{all_row['raw_mae']:.4f}`
- Refined MAE: `{all_row['refined_mae']:.4f}`
- Oracle-all MAE: `{all_row['oracle_all_available_mae']:.4f}`
- SAV MAE: `{all_row['sav_mae']:.4f}`
- Oracle gap recovered: `{all_row['oracle_gap_recovered']:.4f}`
- SAV gap recovered: `{all_row['sav_gap_recovered']:.4f}`
- Modified pixels: `{all_row['modified_pixels_pct']:.4f}%`
- New Bad-3 from raw-good: `{all_row['new_bad3_from_raw_good_pct']:.4f}%`
- Detector AUC/AP: `{overall_auc:.4f}` / `{overall_ap:.4f}`

Interpretation: compare `failure_mode_summary.csv` and `clip_metrics.csv` to decide whether selected-clip oracle/SAV fine-tuning is justified.
"""
    )
    log.append(f"elapsed_seconds={summary['elapsed_seconds']:.3f}")
    (args.output_root / "run.log").write_text("\n".join(log) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
