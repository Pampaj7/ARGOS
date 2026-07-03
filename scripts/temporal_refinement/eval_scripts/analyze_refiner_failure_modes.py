#!/usr/bin/env python3
"""Analyze why tiny refiners overcorrect on full S2M2-GT targets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
TRAIN_DIR = ROOT / "scripts/temporal_refinement"
sys.path.insert(0, str(TRAIN_DIR))

import train_tiny_refiner_v1_full_gt as v1  # noqa: E402
import train_tiny_refiner_v1_1_gated as v11  # noqa: E402
import train_tiny_refiner_v1_2_safe_gated as v12  # noqa: E402
import train_tiny_refiner_v2_temporal_context as v2  # noqa: E402


DEFAULT_TARGETS_ROOT = ROOT / "results/03_temporal_refinement/evaluation/s2m2_gt_refiner_targets_full"
DEFAULT_OUTPUT_ROOT = ROOT / "results/03_temporal_refinement/training/refiner_failure_analysis"
TRAINING_ROOTS = {
    "v1_full_gt": ROOT / "results/03_temporal_refinement/training/tiny_refiner_v1_full_gt",
    "v1_1_gated": ROOT / "results/03_temporal_refinement/training/tiny_refiner_v1_1_gated",
    "v1_2_safe_gated": ROOT / "results/03_temporal_refinement/training/tiny_refiner_v1_2_safe_gated",
    "v2_temporal_context": ROOT / "results/03_temporal_refinement/training/tiny_refiner_v2_temporal_context",
}
BINS = [(0, 1), (1, 3), (3, 5), (5, 10), (10, 20), (20, math.inf)]


def finite_mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def split_sequences(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    return v1.split_sequences(sorted({r["sequence_id"] for r in rows}))


def seq_to_split(splits: dict[str, list[str]]) -> dict[str, str]:
    return {seq: split for split, seqs in splits.items() for seq in seqs}


def load_shard(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path)
    return {k: z[k] for k in ["raw_disp", "gt_disp", "valid_mask", "delta_disp_gt_minus_raw"]}


def sequence_distribution(rows: list[dict[str, str]], splits: dict[str, list[str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_seq: dict[str, list[dict[str, str]]] = {}
    split_of = seq_to_split(splits)
    for row in rows:
        by_seq.setdefault(row["sequence_id"], []).append(row)
    seq_rows: list[dict[str, Any]] = []
    bin_acc = {
        f"[{lo},{hi})": {"pixels": 0, "err_sum": 0.0, "seqs": set(), "split_counts": {"train": 0, "val": 0, "test": 0}}
        for lo, hi in BINS
    }
    for seq, seq_frame_rows in sorted(by_seq.items()):
        shard = load_shard(Path(seq_frame_rows[0]["target_path"]))
        raw = shard["raw_disp"].astype(np.float32)
        gt = shard["gt_disp"].astype(np.float32)
        valid = shard["valid_mask"].astype(bool)
        err = np.abs(raw - gt)
        residual = (gt - raw)[valid]
        e = err[valid]
        total = int(e.size)
        err_sum = float(e.sum())
        for lo, hi in BINS:
            name = f"[{lo},{hi})"
            m = (e >= lo) & (e < hi)
            c = int(m.sum())
            bin_acc[name]["pixels"] += c
            bin_acc[name]["err_sum"] += float(e[m].sum())
            if c:
                bin_acc[name]["seqs"].add(seq)
                bin_acc[name]["split_counts"][split_of[seq]] += c
        seq_rows.append(
            {
                "sequence_id": seq,
                "split": split_of[seq],
                "frames": len(seq_frame_rows),
                "valid_ratio": float(valid.mean()),
                "raw_mae": float(e.mean()) if total else float("nan"),
                "raw_bad1": float((e > 1).mean() * 100.0) if total else float("nan"),
                "raw_bad3": float((e > 3).mean() * 100.0) if total else float("nan"),
                "residual_mean": float(residual.mean()) if total else float("nan"),
                "residual_std": float(residual.std()) if total else float("nan"),
                "residual_p05": float(np.percentile(residual, 5)) if total else float("nan"),
                "residual_p50": float(np.percentile(residual, 50)) if total else float("nan"),
                "residual_p95": float(np.percentile(residual, 95)) if total else float("nan"),
                "raw_good_frac": float((e < 1).mean()) if total else float("nan"),
                "raw_bad_frac": float((e >= 3).mean()) if total else float("nan"),
            }
        )
    split_rows = []
    for split in ["train", "val", "test"]:
        rs = [r for r in seq_rows if r["split"] == split]
        split_rows.append(
            {
                "split": split,
                "sequences": len(rs),
                "frames": sum(int(r["frames"]) for r in rs),
                "raw_mae_mean_seq": finite_mean([r["raw_mae"] for r in rs]),
                "raw_bad3_mean_seq": finite_mean([r["raw_bad3"] for r in rs]),
                "valid_ratio_mean_seq": finite_mean([r["valid_ratio"] for r in rs]),
                "raw_good_frac_mean_seq": finite_mean([r["raw_good_frac"] for r in rs]),
                "raw_bad_frac_mean_seq": finite_mean([r["raw_bad_frac"] for r in rs]),
            }
        )
    total_pixels = sum(v["pixels"] for v in bin_acc.values())
    total_err = sum(v["err_sum"] for v in bin_acc.values())
    bin_rows = []
    for name, v in bin_acc.items():
        bin_rows.append(
            {
                "raw_error_bin": name,
                "pixel_count": int(v["pixels"]),
                "pixel_fraction": v["pixels"] / total_pixels if total_pixels else float("nan"),
                "mae_contribution_fraction": v["err_sum"] / total_err if total_err else float("nan"),
                "sequence_count": len(v["seqs"]),
                "train_pixel_count": v["split_counts"]["train"],
                "val_pixel_count": v["split_counts"]["val"],
                "test_pixel_count": v["split_counts"]["test"],
            }
        )
    return seq_rows, split_rows, bin_rows


def make_inputs(kind: str, shard: dict[str, np.ndarray], offset: int, context_frames: int = 4) -> np.ndarray:
    raw = shard["raw_disp"][offset].astype(np.float32)
    valid = shard["valid_mask"][offset].astype(np.float32)
    if kind == "v1":
        prev = shard["raw_disp"][max(0, offset - 1)].astype(np.float32)
        return np.stack([raw / v1.DISP_SCALE, prev / v1.DISP_SCALE, np.abs(raw - prev) / v1.DISP_SCALE, valid], axis=0)
    if kind in {"v11", "v12"}:
        gx = np.zeros_like(raw)
        gy = np.zeros_like(raw)
        gx[:, 1:] = raw[:, 1:] - raw[:, :-1]
        gy[1:, :] = raw[1:, :] - raw[:-1, :]
        return np.stack([raw / v1.DISP_SCALE, valid, gx / v1.DISP_SCALE, gy / v1.DISP_SCALE], axis=0)
    indices = [max(0, offset - i) for i in range(context_frames)]
    raws = np.stack([shard["raw_disp"][i].astype(np.float32) for i in indices], axis=0)
    valids = np.stack([shard["valid_mask"][i].astype(np.float32) for i in indices], axis=0)
    median = np.median(raws, axis=0).astype(np.float32)
    mean = np.mean(raws, axis=0).astype(np.float32)
    var = np.var(raws, axis=0).astype(np.float32)
    gx = np.zeros_like(raw)
    gy = np.zeros_like(raw)
    gx[:, 1:] = raw[:, 1:] - raw[:, :-1]
    gy[1:, :] = raw[1:, :] - raw[:-1, :]
    edge = np.sqrt(gx * gx + gy * gy)
    return np.stack(
        [*(raws / v1.DISP_SCALE), *valids, np.abs(raws[0] - raws[1]) / v1.DISP_SCALE, mean / v1.DISP_SCALE, median / v1.DISP_SCALE,
         var / (v1.DISP_SCALE * v1.DISP_SCALE), np.abs(raw - median) / v1.DISP_SCALE, gx / v1.DISP_SCALE, gy / v1.DISP_SCALE, edge / v1.DISP_SCALE],
        axis=0,
    )


def load_model(name: str, root: Path, device: torch.device):
    ckpt_path = root / "checkpoints/best.pt"
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    if name == "v1_full_gt":
        model = v1.TinyRefinerV1(int(ckpt.get("input_channels", 4))).to(device)
        kind = "v1"
    elif name == "v1_1_gated":
        model = v11.TinyRefinerV11Gated(gate_bias_init=float(args.get("gate_bias_init", -2.0))).to(device)
        kind = "v11"
    elif name == "v1_2_safe_gated":
        model = v12.TinyRefinerV11Gated(gate_bias_init=float(args.get("gate_bias_init", -4.0))).to(device)
        kind = "v12"
    else:
        model = v2.TinyRefinerV2Temporal(int(ckpt.get("input_channels", 16)), gate_bias_init=float(args.get("gate_bias_init", -4.0))).to(device)
        kind = "v2"
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return {"model": model, "kind": kind, "args": args, "epoch": ckpt.get("epoch")}


@torch.no_grad()
def model_predict(bundle: dict[str, Any], x: torch.Tensor, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    model = bundle["model"]
    kind = bundle["kind"]
    if kind == "v1":
        applied = model(x)
        return raw + applied, None
    out = model(x, float(bundle["args"].get("residual_scale", 3.0)))
    applied, gate = out[0], out[2]
    return raw + applied, gate


def summarize_prediction(raw: np.ndarray, refined: np.ndarray, gt: np.ndarray, valid: np.ndarray, gate: np.ndarray | None) -> dict[str, float]:
    err_raw = np.abs(raw - gt)
    err_ref = np.abs(refined - gt)
    v = valid.astype(bool)
    good = v & (err_raw < 1)
    mid = v & (err_raw >= 1) & (err_raw < 3)
    bad = v & (err_raw >= 3)
    introduced = good & (err_ref >= 3)
    fixed = bad & (err_ref < 3)
    improved = v & (err_ref < err_raw)
    degraded = v & (err_ref > err_raw)
    out = {
        "valid_pixels": int(v.sum()),
        "raw_mae": float(err_raw[v].mean()),
        "refined_mae": float(err_ref[v].mean()),
        "raw_bad3": float((err_raw[v] >= 3).mean() * 100.0),
        "refined_bad3": float((err_ref[v] >= 3).mean() * 100.0),
        "pixels_improved_frac": float(improved.sum() / v.sum()),
        "pixels_degraded_frac": float(degraded.sum() / v.sum()),
        "raw_good_degradation_mean": float((err_ref[good] - err_raw[good]).mean()) if good.any() else float("nan"),
        "raw_bad_improvement_mean": float((err_raw[bad] - err_ref[bad]).mean()) if bad.any() else float("nan"),
        "new_bad3_from_raw_good_frac": float(introduced.sum() / max(1, good.sum())),
        "fixed_bad3_frac": float(fixed.sum() / max(1, bad.sum())),
        "correction_abs_mean": float(np.abs(refined[v] - raw[v]).mean()),
        "correction_abs_p95": float(np.percentile(np.abs(refined[v] - raw[v]), 95)),
    }
    if gate is not None:
        out.update(
            {
                "gate_mean": float(gate[v].mean()),
                "gate_good_mean": float(gate[good].mean()) if good.any() else float("nan"),
                "gate_mid_mean": float(gate[mid].mean()) if mid.any() else float("nan"),
                "gate_bad_mean": float(gate[bad].mean()) if bad.any() else float("nan"),
            }
        )
        try:
            from sklearn.metrics import roc_auc_score

            m = good | bad
            y = bad[m].astype(np.uint8)
            p = gate[m]
            if y.size > 500_000:
                idx = np.linspace(0, y.size - 1, 500_000, dtype=np.int64)
                y, p = y[idx], p[idx]
            out["gate_auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
        except Exception:
            out["gate_auc"] = float("nan")
    return out


def evaluate_models(rows: list[dict[str, str]], splits: dict[str, list[str]], device: torch.device, batch_size: int) -> list[dict[str, Any]]:
    split_of = seq_to_split(splits)
    eval_rows = [r for r in rows if split_of[r["sequence_id"]] in {"val", "test"}]
    model_rows = []
    for name, root in TRAINING_ROOTS.items():
        bundle = load_model(name, root, device)
        if bundle is None:
            continue
        for split in ["val", "test"]:
            selected = [r for r in eval_rows if split_of[r["sequence_id"]] == split]
            acc: list[dict[str, float]] = []
            batch_x: list[np.ndarray] = []
            batch_raw: list[np.ndarray] = []
            batch_gt: list[np.ndarray] = []
            batch_valid: list[np.ndarray] = []
            current_shards: dict[Path, dict[str, np.ndarray]] = {}
            for row in selected:
                path = Path(row["target_path"])
                shard = current_shards.get(path)
                if shard is None:
                    shard = load_shard(path)
                    current_shards[path] = shard
                offset = int(row["frame_offset"])
                batch_x.append(make_inputs(bundle["kind"], shard, offset, int(bundle["args"].get("context_frames", 4))))
                batch_raw.append(shard["raw_disp"][offset].astype(np.float32))
                batch_gt.append(shard["gt_disp"][offset].astype(np.float32))
                batch_valid.append(shard["valid_mask"][offset].astype(bool))
                if len(batch_x) == batch_size:
                    acc.extend(run_model_batch(bundle, batch_x, batch_raw, batch_gt, batch_valid, device))
                    batch_x, batch_raw, batch_gt, batch_valid = [], [], [], []
            if batch_x:
                acc.extend(run_model_batch(bundle, batch_x, batch_raw, batch_gt, batch_valid, device))
            total_pixels = sum(a["valid_pixels"] for a in acc)
            out = {"model": name, "best_epoch": bundle["epoch"], "split": split, "frames": len(acc), "valid_pixels": total_pixels}
            for key in [k for k in acc[0] if k != "valid_pixels"]:
                vals = [a[key] for a in acc if math.isfinite(float(a.get(key, math.nan)))]
                if key.endswith("_frac") or key in {"raw_mae", "refined_mae", "raw_bad3", "refined_bad3", "correction_abs_mean"}:
                    out[key] = float(np.average(vals, weights=[a["valid_pixels"] for a in acc if math.isfinite(float(a.get(key, math.nan))) ])) if vals else float("nan")
                else:
                    out[key] = finite_mean(vals)
            model_rows.append(out)
    return model_rows


def run_model_batch(bundle, batch_x, batch_raw, batch_gt, batch_valid, device):
    x = torch.from_numpy(np.stack(batch_x).astype(np.float32)).to(device)
    raw_t = torch.from_numpy(np.stack(batch_raw)[:, None].astype(np.float32)).to(device)
    refined, gate = model_predict(bundle, x, raw_t)
    refined_np = refined[:, 0].detach().float().cpu().numpy()
    gate_np = gate[:, 0].detach().float().cpu().numpy() if gate is not None else None
    return [
        summarize_prediction(batch_raw[i], refined_np[i], batch_gt[i], batch_valid[i], None if gate_np is None else gate_np[i])
        for i in range(len(batch_x))
    ]


def temporal_feature_auc(rows: list[dict[str, str]], splits: dict[str, list[str]], max_samples: int) -> list[dict[str, Any]]:
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        roc_auc_score = None
    split_of = seq_to_split(splits)
    rng = np.random.default_rng(0)
    features = {"abs_raw_t_minus_1": [], "abs_raw_minus_temporal_median": [], "temporal_variance": [], "spatial_gradient_magnitude": []}
    labels: dict[str, list[np.ndarray]] = {k: [] for k in features}
    by_path: dict[Path, list[dict[str, str]]] = {}
    for row in rows:
        by_path.setdefault(Path(row["target_path"]), []).append(row)
    for path, path_rows in by_path.items():
        shard = load_shard(path)
        raw = shard["raw_disp"].astype(np.float32)
        gt = shard["gt_disp"].astype(np.float32)
        valid = shard["valid_mask"].astype(bool)
        for row in path_rows[:: max(1, len(path_rows) // 80)]:
            i = int(row["frame_offset"])
            indices = [max(0, i - j) for j in range(4)]
            raws = raw[indices]
            err = np.abs(raw[i] - gt[i])
            label = (err >= 3) & valid[i]
            v = valid[i]
            if not v.any():
                continue
            gx = np.zeros_like(raw[i])
            gy = np.zeros_like(raw[i])
            gx[:, 1:] = raw[i, :, 1:] - raw[i, :, :-1]
            gy[1:, :] = raw[i, 1:, :] - raw[i, :-1, :]
            vals = {
                "abs_raw_t_minus_1": np.abs(raws[0] - raws[1]),
                "abs_raw_minus_temporal_median": np.abs(raw[i] - np.median(raws, axis=0)),
                "temporal_variance": np.var(raws, axis=0),
                "spatial_gradient_magnitude": np.sqrt(gx * gx + gy * gy),
            }
            idx = np.flatnonzero(v.reshape(-1))
            if idx.size > 2000:
                idx = rng.choice(idx, size=2000, replace=False)
            for name, arr in vals.items():
                features[name].append(arr.reshape(-1)[idx])
                labels[name].append(label.reshape(-1)[idx].astype(np.uint8))
    rows_out = []
    for name, chunks in features.items():
        x = np.concatenate(chunks) if chunks else np.array([])
        y = np.concatenate(labels[name]) if labels[name] else np.array([])
        if x.size > max_samples:
            idx = rng.choice(np.arange(x.size), size=max_samples, replace=False)
            x, y = x[idx], y[idx]
        auc = float(roc_auc_score(y, x)) if roc_auc_score and len(np.unique(y)) > 1 else float("nan")
        corr = float(np.corrcoef(x, y)[0, 1]) if x.size and np.std(x) > 0 and np.std(y) > 0 else float("nan")
        rows_out.append({"feature": name, "samples": int(x.size), "bad_pixel_fraction": float(y.mean()) if y.size else float("nan"), "auc": auc, "correlation": corr})
    return rows_out


def proposed_split(seq_rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    rows = sorted(seq_rows, key=lambda r: (r["raw_mae"], r["raw_bad3"], -r["valid_ratio"]), reverse=True)
    buckets = {"train": [], "val": [], "test": []}
    for i, row in enumerate(rows):
        split = ["train", "train", "train", "train", "val", "test"][i % 6]
        buckets[split].append(row["sequence_id"])
    return buckets


def make_plots(out: Path, seq_rows, bin_rows, model_rows, feat_rows) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    colors = {"train": "tab:blue", "val": "tab:orange", "test": "tab:green"}
    seq_rows = sorted(seq_rows, key=lambda r: r["raw_mae"])
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(seq_rows)), [r["raw_mae"] for r in seq_rows], color=[colors[r["split"]] for r in seq_rows])
    plt.xticks(range(len(seq_rows)), [r["sequence_id"] for r in seq_rows], rotation=90, fontsize=6)
    plt.tight_layout()
    plt.savefig(out / "raw_mae_by_sequence_split.png", dpi=160)
    plt.close()
    plt.figure(figsize=(7, 4))
    plt.bar([r["raw_error_bin"] for r in bin_rows], [r["mae_contribution_fraction"] for r in bin_rows])
    plt.tight_layout()
    plt.savefig(out / "error_bin_contribution.png", dpi=160)
    plt.close()
    plt.figure(figsize=(8, 4))
    labels = [f"{r['model']}-{r['split']}" for r in model_rows]
    x = np.arange(len(labels))
    plt.bar(x - 0.2, [r["raw_mae"] for r in model_rows], width=0.4, label="raw")
    plt.bar(x + 0.2, [r["refined_mae"] for r in model_rows], width=0.4, label="refined")
    plt.xticks(x, labels, rotation=45, ha="right", fontsize=7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "model_raw_vs_refined_mae.png", dpi=160)
    plt.close()
    gated = [r for r in model_rows if "gate_good_mean" in r]
    if gated:
        plt.figure(figsize=(8, 4))
        labels = [f"{r['model']}-{r['split']}" for r in gated]
        x = np.arange(len(labels))
        plt.bar(x - 0.2, [r["gate_good_mean"] for r in gated], width=0.4, label="good")
        plt.bar(x + 0.2, [r["gate_bad_mean"] for r in gated], width=0.4, label="bad")
        plt.xticks(x, labels, rotation=45, ha="right", fontsize=7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "gate_good_vs_bad.png", dpi=160)
        plt.close()
    plt.figure(figsize=(7, 4))
    plt.bar([r["feature"] for r in feat_rows], [r["auc"] for r in feat_rows])
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0.4, 1.0)
    plt.tight_layout()
    plt.savefig(out / "temporal_feature_auc.png", dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets-root", type=Path, default=DEFAULT_TARGETS_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--temporal-feature-max-samples", type=int, default=1_000_000)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(args.output_root)
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True)
    (args.output_root / "run.log").write_text("starting analysis\n")
    rows = read_csv(args.targets_root / "frame_targets_index.csv")
    splits = split_sequences(rows)
    seq_rows, split_rows, bin_rows = sequence_distribution(rows, splits)
    write_csv(args.output_root / "sequence_raw_difficulty.csv", seq_rows)
    write_csv(args.output_root / "split_distribution_summary.csv", split_rows)
    write_csv(args.output_root / "error_bin_summary.csv", bin_rows)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model_rows = evaluate_models(rows, splits, device, args.batch_size)
    write_csv(args.output_root / "model_overcorrection_summary.csv", model_rows)
    feat_rows = temporal_feature_auc(rows, splits, args.temporal_feature_max_samples)
    write_csv(args.output_root / "temporal_feature_auc.csv", feat_rows)
    balanced = proposed_split(seq_rows)
    (args.output_root / "proposed_balanced_split.json").write_text(json.dumps(balanced, indent=2) + "\n")
    make_plots(args.output_root, seq_rows, bin_rows, model_rows, feat_rows)
    shift = {
        "train_raw_mae": next(r["raw_mae_mean_seq"] for r in split_rows if r["split"] == "train"),
        "val_raw_mae": next(r["raw_mae_mean_seq"] for r in split_rows if r["split"] == "val"),
        "test_raw_mae": next(r["raw_mae_mean_seq"] for r in split_rows if r["split"] == "test"),
    }
    high_tail = next(r for r in bin_rows if r["raw_error_bin"] == "[20,inf)")
    best_auc = max((r["auc"] for r in feat_rows if math.isfinite(float(r["auc"]))), default=float("nan"))
    summary = {
        "targets_root": str(args.targets_root),
        "output_root": str(args.output_root),
        "split_distribution_shift": shift,
        "high_error_tail_mae_contribution_fraction": high_tail["mae_contribution_fraction"],
        "best_temporal_feature_auc": best_auc,
        "models_analyzed": sorted(set(r["model"] for r in model_rows)),
        "recommendation": "next safest change: predict an abstention/confidence map first and train a refiner only on balanced hard-pixel crops; keep identity fallback for full frames.",
    }
    (args.output_root / "aggregate_failure_analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    readme = f"""# Refiner Failure Analysis

Inputs: `{args.targets_root}` and existing training outputs for v1/v1.1/v1.2/v2.

## Answers

- Train/val/test are distribution shifted: train mean sequence MAE `{shift['train_raw_mae']:.3f}`, val `{shift['val_raw_mae']:.3f}`, test `{shift['test_raw_mae']:.3f}`.
- High-error outliers dominate loss: pixels in `[20,+inf)` contribute `{high_tail['mae_contribution_fraction']:.3f}` of total raw MAE.
- Temporal raw context alone is weak: best simple temporal/spatial feature AUC is `{best_auc:.3f}`.
- Models mainly fail by degrading raw-good pixels and introducing/expanding bad-3 regions; see `model_overcorrection_summary.csv`.
- Next safest change: train confidence/abstention and hard-pixel crop curriculum before applying residuals to full frames.

## Files

- `split_distribution_summary.csv`
- `sequence_raw_difficulty.csv`
- `error_bin_summary.csv`
- `model_overcorrection_summary.csv`
- `temporal_feature_auc.csv`
- `proposed_balanced_split.json`
- plots `*.png`
"""
    (args.output_root / "README.md").write_text(readme)
    (args.output_root / "run.log").write_text("analysis complete\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
