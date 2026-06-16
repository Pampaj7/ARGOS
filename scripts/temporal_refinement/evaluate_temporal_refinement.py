#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from scripts.temporal_refinement.lib.models import ConvGRURefiner, TinyUNetRefiner
from scripts.temporal_refinement.lib.training import colorize
from scripts.temporal_refinement.train_temporal_refiner_fastcache import split_fast_by_sequence


NAN = float("nan")


def read_rows(cache_root: Path, index_file: str, sample_ids: list[int] | None = None) -> list[dict]:
    with (cache_root / index_file).open() as f:
        rows = list(csv.DictReader(f))
    if sample_ids is not None:
        wanted = {int(x) for x in sample_ids}
        rows = [r for r in rows if int(r["sample_id"]) in wanted]
    return sorted(rows, key=lambda r: (r["sequence_id"], int(r["center_frame_id"])))


def group_rows(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["sequence_id"], []).append(row)
    return {k: sorted(v, key=lambda r: int(r["center_frame_id"])) for k, v in grouped.items()}


def path_for(row: dict, prefix: str, slot: str = "t") -> str:
    key = f"{prefix}_{slot}_path"
    if key in row and row[key]:
        return row[key]
    frame_key = f"frame_{slot}"
    frame_id = row[frame_key]
    disp_dir = "sav_disp" if prefix == "sav" else f"{prefix}_disp"
    return f"{row['sequence_id']}/{disp_dir}/{frame_id}.npy"


def load_disp(cache_root: Path, rel_path: str) -> np.ndarray:
    return np.load(cache_root / rel_path).astype(np.float32)


def load_rgb(cache_root: Path, row: dict) -> np.ndarray:
    img = cv2.imread(str(cache_root / row["rgb_center_path"]), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read RGB: {cache_root / row['rgb_center_path']}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def edge_mask(rgb: np.ndarray, low: int = 60, high: int = 120, dilate: int = 1) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high) > 0
    if dilate > 0:
        kernel = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
        edges = cv2.dilate(edges.astype(np.uint8), kernel) > 0
    return edges


def disp_to_depth_mm(disp: np.ndarray, fx: float, baseline_mm: float) -> np.ndarray:
    return fx * baseline_mm / np.maximum(disp.astype(np.float32), 1e-6)


def ema_baseline(disps: list[np.ndarray], alpha: float) -> list[np.ndarray]:
    out = []
    prev = None
    for cur in disps:
        refined = cur.copy() if prev is None else alpha * cur + (1.0 - alpha) * prev
        out.append(refined.astype(np.float32))
        prev = refined
    return out


def previous_blend(disps: list[np.ndarray], alpha: float) -> list[np.ndarray]:
    out = []
    prev_raw = None
    for cur in disps:
        refined = cur.copy() if prev_raw is None else alpha * cur + (1.0 - alpha) * prev_raw
        out.append(refined.astype(np.float32))
        prev_raw = cur
    return out


def temporal_median(disps: list[np.ndarray], window: int, causal: bool) -> list[np.ndarray]:
    radius = window // 2
    out = []
    for idx in range(len(disps)):
        if causal:
            lo, hi = max(0, idx - window + 1), idx + 1
        else:
            lo, hi = max(0, idx - radius), min(len(disps), idx + radius + 1)
        out.append(np.median(np.stack(disps[lo:hi], axis=0), axis=0).astype(np.float32))
    return out


def enumerate_checkpoints(run_dir: Path, selection: str) -> list[Path]:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return []
    if selection == "best":
        return [ckpt_dir / "best.pt"] if (ckpt_dir / "best.pt").exists() else []
    if selection == "latest":
        return [ckpt_dir / "latest.pt"] if (ckpt_dir / "latest.pt").exists() else []
    if selection == "periodic":
        return sorted(ckpt_dir.glob("epoch_*.pt"))
    if selection == "all":
        paths = []
        for name in ["best.pt", "latest.pt"]:
            p = ckpt_dir / name
            if p.exists():
                paths.append(p)
        paths.extend(sorted(ckpt_dir.glob("epoch_*.pt")))
        seen = set()
        unique = []
        for p in paths:
            if p.name not in seen:
                unique.append(p)
                seen.add(p.name)
        return unique
    explicit = ckpt_dir / selection
    return [explicit] if explicit.exists() else []


def pareto_front(rows: list[dict], x_key: str, y_key: str) -> list[dict]:
    pts = [r for r in rows if np.isfinite(r.get(x_key, NAN)) and np.isfinite(r.get(y_key, NAN))]
    front = []
    for row in pts:
        x, y = row[x_key], row[y_key]
        dominated = any(other is not row and other[x_key] <= x and other[y_key] <= y and (other[x_key] < x or other[y_key] < y) for other in pts)
        if not dominated:
            front.append(row)
    return sorted(front, key=lambda r: (r[x_key], r[y_key]))


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if vals.size == 0:
        return NAN, NAN, NAN
    mean = float(vals.mean())
    if vals.size < 2:
        return mean, NAN, NAN
    se = vals.std(ddof=1) / math.sqrt(vals.size)
    return mean, float(mean - 1.96 * se), float(mean + 1.96 * se)


def load_model_checkpoint(path: Path, model_type: str, device: torch.device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    args = ck.get("args", {})
    base = int(args.get("base_channels", 16))
    clamp = float(args.get("residual_clamp_px", 2.0))
    if model_type == "tiny_unet":
        model = TinyUNetRefiner(in_channels=8, base_channels=base, residual_clamp_px=clamp)
    elif model_type == "convgru":
        hidden = int(args.get("hidden_channels", 64))
        model = ConvGRURefiner(in_channels=4, base_channels=base, hidden_channels=hidden, residual_clamp_px=clamp)
    else:
        raise ValueError(model_type)
    model.load_state_dict(ck["model_state_dict"])
    return model.to(device).eval(), ck


def infer_tiny(model, cache_root: Path, rows_by_seq: dict[str, list[dict]], device: torch.device, amp: bool, disp_norm: float) -> tuple[list[np.ndarray], list[np.ndarray], float, float]:
    preds, residuals = [], []
    t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for rows in rows_by_seq.values():
            for row in rows:
                rgb = load_rgb(cache_root, row).astype(np.float32) / 255.0
                window = np.stack([load_disp(cache_root, path_for(row, "s2m2_l736", slot)) for slot in ["tminus2", "tminus1", "t", "tplus1", "tplus2"]])
                x = np.concatenate([rgb.transpose(2, 0, 1), window / disp_norm], axis=0)
                x_t = torch.from_numpy(x).unsqueeze(0).float().to(device)
                center = torch.from_numpy(window[2]).unsqueeze(0).unsqueeze(0).float().to(device)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda" and amp):
                    delta = model(x_t)
                    refined = torch.clamp(center + delta, min=0.0)
                preds.append(refined[0, 0].float().cpu().numpy())
                residuals.append(delta[0, 0].float().cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
    return preds, residuals, elapsed * 1000.0 / max(len(preds), 1), float(peak)


def infer_convgru(model, cache_root: Path, rows_by_seq: dict[str, list[dict]], device: torch.device, amp: bool, disp_norm: float) -> tuple[list[np.ndarray], list[np.ndarray], float, float]:
    preds, residuals = [], []
    t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for rows in rows_by_seq.values():
            hidden = None
            for row in rows:
                rgb = load_rgb(cache_root, row).astype(np.float32) / 255.0
                center_np = load_disp(cache_root, path_for(row, "s2m2_l736", "t"))
                x = np.concatenate([rgb.transpose(2, 0, 1), center_np[None] / disp_norm], axis=0)
                x_t = torch.from_numpy(x).unsqueeze(0).float().to(device)
                center = torch.from_numpy(center_np).unsqueeze(0).unsqueeze(0).float().to(device)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda" and amp):
                    delta, hidden = model(x_t, hidden)
                    refined = torch.clamp(center + delta, min=0.0)
                preds.append(refined[0, 0].float().cpu().numpy())
                residuals.append(delta[0, 0].float().cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
    return preds, residuals, elapsed * 1000.0 / max(len(preds), 1), float(peak)


def compute_metrics(method: str, preds: list[np.ndarray], raw: list[np.ndarray], sav: list[np.ndarray], rgbs: list[np.ndarray], rows: list[dict], common_masks: list[np.ndarray], runtime_ms: float, peak_vram: float, checkpoint: str = "", causal: bool = True) -> tuple[dict, list[dict], list[dict]]:
    per_frame = []
    edge_temporal, nonedge_temporal, temporal, valid_temporal, teacher_delta = [], [], [], [], []
    residual_abs = [float(np.mean(np.abs(p - b))) for p, b in zip(preds, raw)]
    residual_std = [float(np.std(p - b)) for p, b in zip(preds, raw)]
    to_sav = [float(np.mean(np.abs(p - t))) for p, t in zip(preds, sav)]
    to_backbone = [float(np.mean(np.abs(p - b))) for p, b in zip(preds, raw)]
    valid_masks = [m & np.isfinite(p) for m, p in zip(common_masks, preds)]
    edges = [edge_mask(rgb) for rgb in rgbs]
    for i, row in enumerate(rows):
        pf = {
            "method": method,
            "checkpoint": checkpoint,
            "sequence_id": row["sequence_id"],
            "frame_id": row["center_frame_id"],
            "sample_id": int(row["sample_id"]),
            "refined_to_backbone_mae": to_backbone[i],
            "refined_to_sav_mae": to_sav[i],
            "residual_abs_mean": residual_abs[i],
            "residual_std": residual_std[i],
            "valid_pixel_count": int(valid_masks[i].sum()),
            "disparity_mae_px": NAN,
            "disparity_rmse": NAN,
            "depth_mae_mm": NAN,
            "depth_rmse_mm": NAN,
            "depth_median_error_mm": NAN,
            "depth_p95_error_mm": NAN,
            "bad_2mm": NAN,
            "bad_4mm": NAN,
            "edge_depth_mae": NAN,
            "nonedge_depth_mae": NAN,
        }
        per_frame.append(pf)
    for i in range(1, len(preds)):
        same_seq = rows[i]["sequence_id"] == rows[i - 1]["sequence_id"]
        if not same_seq:
            continue
        both = valid_masks[i] & valid_masks[i - 1]
        if not both.any():
            continue
        diff = np.abs(preds[i] - preds[i - 1])
        sdiff = np.abs((preds[i] - preds[i - 1]) - (sav[i] - sav[i - 1]))
        temporal.append(float(diff[both].mean()))
        teacher_delta.append(float(sdiff[both].mean()))
        valid_temporal.append(float(diff[both].mean()))
        em = both & (edges[i] | edges[i - 1])
        nem = both & ~em
        edge_temporal.append(float(diff[em].mean()) if em.any() else NAN)
        nonedge_temporal.append(float(diff[nem].mean()) if nem.any() else NAN)
    stack = np.stack(preds, axis=0)
    summary = {
        "method": method,
        "checkpoint": checkpoint,
        "causal": causal,
        "disparity_mae_px": NAN,
        "disparity_rmse": NAN,
        "depth_mae_mm": NAN,
        "depth_rmse_mm": NAN,
        "depth_median_error_mm": NAN,
        "depth_p95_error_mm": NAN,
        "bad_2mm": NAN,
        "bad_4mm": NAN,
        "edge_depth_mae": NAN,
        "nonedge_depth_mae": NAN,
        "temporal_diff": float(np.nanmean(temporal)) if temporal else NAN,
        "motion_compensated_temporal_diff": NAN,
        "teacher_delta_mae": float(np.nanmean(teacher_delta)) if teacher_delta else NAN,
        "temporal_std": float(np.nanmean(np.std(stack, axis=0))),
        "temporal_diff_valid_gt": NAN,
        "temporal_diff_edge": float(np.nanmean(edge_temporal)) if edge_temporal else NAN,
        "temporal_diff_nonedge": float(np.nanmean(nonedge_temporal)) if nonedge_temporal else NAN,
        "refined_to_backbone_mae": float(np.mean(to_backbone)),
        "refined_to_sav_mae": float(np.mean(to_sav)),
        "residual_abs_mean": float(np.mean(residual_abs)),
        "residual_std": float(np.mean(residual_std)),
        "runtime_ms_per_frame": float(runtime_ms),
        "peak_vram_mb": float(peak_vram),
        "valid_pixel_count": int(sum(m.sum() for m in valid_masks)),
        "frames": len(preds),
    }
    per_seq = []
    for seq, seq_rows in group_rows(rows).items():
        idxs = [i for i, r in enumerate(rows) if r["sequence_id"] == seq]
        seq_temporal = []
        for a, b in zip(idxs[:-1], idxs[1:]):
            both = valid_masks[a] & valid_masks[b]
            if both.any():
                seq_temporal.append(float(np.abs(preds[b] - preds[a])[both].mean()))
        vals = [to_sav[i] for i in idxs]
        per_seq.append({"method": method, "checkpoint": checkpoint, "sequence_id": seq, "frames": len(idxs), "temporal_diff": float(np.mean(seq_temporal)) if seq_temporal else NAN, "refined_to_sav_mae": float(np.mean(vals))})
    return summary, per_frame, per_seq


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_plots(out_dir: Path, summary_rows: list[dict]) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception:
        (plot_dir / "README.txt").write_text("matplotlib unavailable; plots were not generated.\n")
        return
    finite = [r for r in summary_rows if np.isfinite(r.get("temporal_diff", NAN))]
    for x_key, y_key, name in [
        ("refined_to_backbone_mae", "temporal_diff", "temporal_vs_backbone.png"),
        ("refined_to_sav_mae", "teacher_delta_mae", "teacher_delta_vs_sav.png"),
        ("runtime_ms_per_frame", "temporal_diff", "runtime_vs_temporal.png"),
    ]:
        pts = [r for r in finite if np.isfinite(r.get(x_key, NAN)) and np.isfinite(r.get(y_key, NAN))]
        if not pts:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter([r[x_key] for r in pts], [r[y_key] for r in pts], s=24)
        for r in pts:
            ax.annotate(r["method"][:24], (r[x_key], r[y_key]), fontsize=6, alpha=0.75)
        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / name, dpi=160)
        plt.close(fig)


def save_qualitative(out_dir: Path, rows: list[dict], rgbs: list[np.ndarray], raw: list[np.ndarray], sav: list[np.ndarray], methods: dict[str, list[np.ndarray]]) -> None:
    qdir = out_dir / "qualitative"
    qdir.mkdir(parents=True, exist_ok=True)
    candidates = [0, len(rows) // 2, len(rows) - 1]
    for idx in sorted(set(candidates)):
        vmax = float(np.nanpercentile(np.concatenate([raw[idx].ravel(), sav[idx].ravel()]), 99))
        tiles = [cv2.cvtColor(rgbs[idx], cv2.COLOR_RGB2BGR), colorize(raw[idx], vmax), colorize(sav[idx], vmax)]
        labels = ["RGB", "raw S2M2-L", "SAV"]
        for name, preds in methods.items():
            tiles.append(colorize(preds[idx], vmax))
            labels.append(name[:18])
            tiles.append(colorize(np.abs(preds[idx] - raw[idx]), 5.0, cv2.COLORMAP_MAGMA))
            labels.append("|res|")
        small = []
        for tile, label in zip(tiles, labels):
            tile = cv2.resize(tile, (180, 120), interpolation=cv2.INTER_AREA)
            cv2.putText(tile, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            small.append(tile)
        cv2.imwrite(str(qdir / f"sample_{int(rows[idx]['sample_id']):06d}.png"), np.concatenate(small, axis=1))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-root", type=Path, default=Path("results/temporal_refinement_cache/large_v3_s2m2s512_fast"))
    p.add_argument("--index-file", default="index_s2m2l736.csv")
    p.add_argument("--out-dir", type=Path, default=Path("results/temporal_refinement_evaluation_l736_v1"))
    p.add_argument("--unet-run", type=Path, default=Path("results/temporal_refinement_train_unet_s2m2l736_fastcache_v2_conservative"))
    p.add_argument("--convgru-v1-run", type=Path, default=Path("results/temporal_refinement_train_convgru_l736_v1_100ep_b13"))
    p.add_argument("--convgru-v2-run", type=Path, default=Path("results/temporal_refinement_train_convgru_l736_v2_scheduled"))
    p.add_argument("--checkpoint-selection", choices=["best", "latest", "periodic", "all"], default="best")
    p.add_argument("--checkpoint-sweep", action="store_true")
    p.add_argument("--ema-alpha", type=float, nargs="*", default=[0.3, 0.5, 0.7, 0.9])
    p.add_argument("--median-windows", type=int, nargs="*", default=[3, 5])
    p.add_argument("--val-sequences", type=int, default=1)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--disp-norm", type=float, default=128.0)
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    train_ids, val_ids, val_sequences = split_fast_by_sequence(args.cache_root, args.index_file, args.val_sequences)
    if args.max_samples:
        val_ids = val_ids[: args.max_samples]
    rows = read_rows(args.cache_root, args.index_file, val_ids)
    rows_by_seq = group_rows(rows)
    print(f"eval_rows={len(rows)} val_sequences={val_sequences} device={device}", flush=True)
    if not any(r.get("has_gt") == "True" for r in rows):
        print("WARNING: selected cache rows have no GT/calibration; geometric depth metrics will be NaN.", flush=True)

    rgbs = [load_rgb(args.cache_root, r) for r in rows]
    raw = [load_disp(args.cache_root, path_for(r, "s2m2_l736", "t")) for r in rows]
    sav = [load_disp(args.cache_root, path_for(r, "sav", "t")) for r in rows]
    common_masks = [np.isfinite(b) & np.isfinite(t) & (b > 0.1) & (t > 0.1) for b, t in zip(raw, sav)]

    summary_rows, per_frame_rows, per_seq_rows, checkpoint_rows, baseline_rows = [], [], [], [], []
    qualitative_methods: dict[str, list[np.ndarray]] = {}

    def add_method(name, preds, residuals=None, runtime=0.0, peak=0.0, checkpoint="", causal=True, is_baseline=False):
        summary, pf, ps = compute_metrics(name, preds, raw, sav, rgbs, rows, common_masks, runtime, peak, checkpoint, causal)
        summary_rows.append(summary)
        per_frame_rows.extend(pf)
        per_seq_rows.extend(ps)
        if is_baseline:
            baseline_rows.append(summary)
        else:
            checkpoint_rows.append(summary)
        if name in {"raw_s2m2_l736", "stereoanyvideo", "tiny_unet_conservative", "convgru_v1_conservative", "convgru_v2_scheduled", "ema_alpha_0.7", "median5_noncausal"}:
            qualitative_methods[name] = preds

    add_method("raw_s2m2_l736", raw, runtime=0.0, peak=0.0, causal=True)
    add_method("stereoanyvideo", sav, runtime=0.0, peak=0.0, causal=False)
    for alpha in args.ema_alpha:
        add_method(f"ema_alpha_{alpha}", ema_baseline(raw, alpha), causal=True, is_baseline=True)
        add_method(f"prev_blend_alpha_{alpha}", previous_blend(raw, alpha), causal=True, is_baseline=True)
    for w in args.median_windows:
        add_method(f"median{w}_noncausal", temporal_median(raw, w, causal=False), causal=False, is_baseline=True)
        add_method(f"median{w}_causal", temporal_median(raw, w, causal=True), causal=True, is_baseline=True)

    model_specs = [
        ("tiny_unet_conservative", "tiny_unet", args.unet_run),
        ("convgru_v1_conservative", "convgru", args.convgru_v1_run),
        ("convgru_v2_scheduled", "convgru", args.convgru_v2_run),
    ]
    for name, model_type, run_dir in model_specs:
        selection = "all" if args.checkpoint_sweep else args.checkpoint_selection
        for ckpt_path in enumerate_checkpoints(run_dir, selection):
            print(f"evaluating {name} {ckpt_path.name}", flush=True)
            model, ck = load_model_checkpoint(ckpt_path, model_type, device)
            if model_type == "tiny_unet":
                preds, residuals, runtime, peak = infer_tiny(model, args.cache_root, rows_by_seq, device, args.amp, args.disp_norm)
            else:
                preds, residuals, runtime, peak = infer_convgru(model, args.cache_root, rows_by_seq, device, args.amp, args.disp_norm)
            method_name = name if ckpt_path.name == "best.pt" else f"{name}:{ckpt_path.stem}"
            add_method(method_name, preds, residuals, runtime, peak, checkpoint=str(ckpt_path), causal=(model_type == "convgru"))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    pareto_rows = []
    for x, y in [
        ("temporal_diff", "depth_mae_mm"),
        ("temporal_diff", "refined_to_backbone_mae"),
        ("teacher_delta_mae", "depth_mae_mm"),
        ("runtime_ms_per_frame", "temporal_diff"),
    ]:
        for row in pareto_front(checkpoint_rows + baseline_rows, x, y):
            pareto_rows.append({"pareto_x": x, "pareto_y": y, **row})

    save_qualitative(args.out_dir, rows, rgbs, raw, sav, qualitative_methods)
    save_plots(args.out_dir, summary_rows)
    write_csv(args.out_dir / "summary.csv", summary_rows)
    write_csv(args.out_dir / "per_frame_metrics.csv", per_frame_rows)
    write_csv(args.out_dir / "per_sequence_metrics.csv", per_seq_rows)
    write_csv(args.out_dir / "checkpoint_metrics.csv", checkpoint_rows)
    write_csv(args.out_dir / "baseline_metrics.csv", baseline_rows)
    write_csv(args.out_dir / "pareto_points.csv", pareto_rows)

    payload = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "val_sequences": val_sequences,
        "frames": len(rows),
        "gt_available": any(r.get("has_gt") == "True" for r in rows),
        "summary": summary_rows,
        "pareto_points": pareto_rows,
        "limitations": [
            "Selected long SCARED fast-cache rows have has_gt=False, so disparity/depth GT metrics are NaN.",
            "Motion-compensated temporal difference is NaN because no existing optical-flow support was found in this pipeline.",
            "Runtime for raw cached predictions and simple baselines excludes upstream S2M2/StereoAnyVideo inference cost.",
        ],
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    report_lines = [
        "# Temporal Refinement Evaluation L736 V1",
        "",
        f"Frames: `{len(rows)}`",
        f"Sequences: `{', '.join(val_sequences)}`",
        f"GT available: `{payload['gt_available']}`",
        "",
        "GT note: selected fast-cache validation rows have no SCARED GT/calibration; geometric depth metrics are reported as `NaN` rather than approximated.",
        "",
        "## Summary",
        "",
    ]
    for row in summary_rows:
        report_lines.append(
            f"- `{row['method']}`: temporal_diff={row['temporal_diff']:.4f}, "
            f"teacher_delta={row['teacher_delta_mae']:.4f}, "
            f"to_backbone={row['refined_to_backbone_mae']:.4f}, "
            f"to_SAV={row['refined_to_sav_mae']:.4f}, runtime={row['runtime_ms_per_frame']:.2f} ms"
        )
    (args.out_dir / "report.md").write_text("\n".join(report_lines) + "\n")
    print(json.dumps({"frames": len(rows), "summary_csv": str(args.out_dir / "summary.csv"), "gt_available": payload["gt_available"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
