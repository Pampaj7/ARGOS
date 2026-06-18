#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Method:
    name: str
    pred_dir: Path | None
    source: str
    causal: str
    notes: str = ""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def rel(path: Path) -> str:
    return str(path).replace("\\", "/")


def mean(values: list[float]) -> float:
    values = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(values)) if values else float("nan")


def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_disp_for_method(method: Method, frame_id: str, frame_name: str) -> np.ndarray:
    if method.pred_dir is None:
        raise RuntimeError("Simple baselines are generated separately")
    candidates = [
        method.pred_dir / f"{frame_id}.npy",
        method.pred_dir / f"{frame_name}_disp.npy",
        method.pred_dir / f"test_dataset_9_keyframe_3_frame_{frame_id}_disp.npy",
    ]
    for path in candidates:
        if path.exists():
            return np.load(path).astype(np.float32)
    matches = list(method.pred_dir.glob(f"*{frame_id}*disp.npy"))
    if matches:
        return np.load(matches[0]).astype(np.float32)
    raise FileNotFoundError(f"No prediction for {method.name} frame {frame_id} in {method.pred_dir}")


def load_metadata(metadata_csv: Path, min_valid_ratio: float) -> list[dict[str, object]]:
    rows = []
    for row in read_csv(metadata_csv):
        valid_ratio = float(row["valid_pixel_ratio"])
        if valid_ratio < min_valid_ratio:
            continue
        frame_id = row["frame_id"]
        rows.append(
            {
                "frame_id": frame_id,
                "frame_name": f"{row['sequence_id']}_frame_{frame_id}",
                "left_path": Path(row["left_path"]),
                "gt_disp_path": Path(row["disparity_float32_path"]),
                "gt_depth_path": Path(row["depth_float32_path"]),
                "valid_mask_path": Path(row["valid_mask_path"]),
                "calib_path": Path(row["calibration_path"]),
            }
        )
    return rows


def calibration_fx_baseline(path: Path) -> tuple[float, float]:
    calib = json.loads(path.read_text())
    if "fx" in calib and "baseline_mm" in calib:
        return float(calib["fx"]), float(calib["baseline_mm"])
    p1 = np.array(calib["P1"]["data"], dtype=np.float64).reshape(calib["P1"]["rows"], calib["P1"]["cols"])
    p2 = np.array(calib["P2"]["data"], dtype=np.float64).reshape(calib["P2"]["rows"], calib["P2"]["cols"])
    return float(p1[0, 0]), float(abs(p2[0, 3] / p2[0, 0]))


def depth_from_disp(disp: np.ndarray, fx: float, baseline_mm: float) -> np.ndarray:
    return (fx * baseline_mm) / np.maximum(disp.astype(np.float32), 1e-6)


def compute_geom(pred: np.ndarray, gt_disp: np.ndarray, gt_depth: np.ndarray, mask: np.ndarray, fx: float, baseline_mm: float) -> dict[str, float]:
    valid = mask & np.isfinite(pred) & (pred > 0.1)
    if not valid.any():
        return {"disp_mae": math.nan, "depth_mae": math.nan, "bad_2mm": math.nan, "evaluated_pixels": 0}
    disp_err = np.abs(pred[valid] - gt_disp[valid])
    pred_depth = depth_from_disp(pred, fx, baseline_mm)
    depth_err = np.abs(pred_depth[valid] - gt_depth[valid])
    return {
        "disp_mae": float(np.mean(disp_err)),
        "depth_mae": float(np.mean(depth_err)),
        "bad_2mm": float(np.mean(depth_err > 2.0) * 100.0),
        "evaluated_pixels": int(valid.sum()),
    }


def compute_flow(prev_rgb: np.ndarray, cur_rgb: np.ndarray) -> np.ndarray:
    prev_gray = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
    cur_gray = cv2.cvtColor(cur_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(
        prev_gray,
        cur_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=25,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    ).astype(np.float32)


def warp_prev_to_current(prev: np.ndarray, flow_prev_to_cur: np.ndarray) -> np.ndarray:
    h, w = prev.shape
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = xx - flow_prev_to_cur[..., 0]
    map_y = yy - flow_prev_to_cur[..., 1]
    return cv2.remap(prev.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)


def temporal_metrics(preds: dict[str, list[np.ndarray]], frames: list[dict[str, object]], masks: list[np.ndarray]) -> dict[str, dict[str, float]]:
    rgbs = [load_rgb(Path(frame["left_path"])) for frame in frames]
    flows = [None] + [compute_flow(rgbs[i - 1], rgbs[i]) for i in range(1, len(rgbs))]
    out: dict[str, dict[str, float]] = {}
    for name, seq in preds.items():
        raw_vals: list[float] = []
        mc_vals: list[float] = []
        for i in range(1, len(seq)):
            prev = seq[i - 1]
            cur = seq[i]
            valid_raw = masks[i - 1] & masks[i] & np.isfinite(prev) & np.isfinite(cur) & (prev > 0.1) & (cur > 0.1)
            if valid_raw.any():
                raw_vals.append(float(np.mean(np.abs(cur[valid_raw] - prev[valid_raw]))))
            warped = warp_prev_to_current(prev, flows[i])
            valid_mc = masks[i] & np.isfinite(warped) & np.isfinite(cur) & (warped > 0.1) & (cur > 0.1)
            if valid_mc.any():
                mc_vals.append(float(np.mean(np.abs(cur[valid_mc] - warped[valid_mc]))))
        out[name] = {
            "raw_temporal_diff": mean(raw_vals),
            "motion_compensated_temporal_mae": mean(mc_vals),
            "temporal_pairs": len(raw_vals),
            "flow": "OpenCV Farneback local fallback",
        }
    return out


def ema_sequence(seq: list[np.ndarray], alpha: float) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    prev = None
    for disp in seq:
        if prev is None:
            cur = disp.astype(np.float32).copy()
        else:
            cur = alpha * disp.astype(np.float32) + (1.0 - alpha) * prev
        out.append(cur)
        prev = cur
    return out


def median5_sequence(seq: list[np.ndarray]) -> list[np.ndarray]:
    out = []
    for i in range(len(seq)):
        lo = max(0, i - 2)
        hi = min(len(seq), i + 3)
        out.append(np.median(np.stack(seq[lo:hi], axis=0), axis=0).astype(np.float32))
    return out


def method_catalog(pred_root: Path, frame_root: Path) -> list[Method]:
    methods = [
        Method("S2M2-L@736", pred_root / "S2M2-L_736", "temporal_gt_existing", "yes"),
        Method("StereoAnyVideo@384x640", pred_root / "StereoAnyVideo_384x640", "temporal_gt_existing", "no"),
        Method("Tiny U-Net e100", pred_root / "TinyUNet-L736_temporal_refinement_train_unet_s2m2l736_fastcache_v2_conservative:epoch_0100", "temporal_gt_existing", "no"),
        Method("ConvGRU V2 e30", pred_root / "ConvGRU-L736_temporal_refinement_train_convgru_l736_v2_scheduled:epoch_0030", "temporal_gt_existing", "yes"),
        Method("ConvGRU V2 e40", pred_root / "ConvGRU-L736_temporal_refinement_train_convgru_l736_v2_scheduled:epoch_0040", "temporal_gt_existing", "yes"),
        Method("ConvGRU V2 e50", pred_root / "ConvGRU-L736_temporal_refinement_train_convgru_l736_v2_scheduled:epoch_0050", "temporal_gt_existing", "yes"),
        Method("ConvGRU V2 latest", pred_root / "ConvGRU-L736_temporal_refinement_train_convgru_l736_v2_scheduled:latest", "temporal_gt_existing", "yes"),
        Method("S2M2-S@512", pred_root / "S2M2-S_512", "temporal_gt_existing", "yes"),
        Method("Fast-FoundationStereo ONNX", frame_root / "Fast-FoundationStereo_ONNX", "frame_based_gt", "yes"),
        Method("DEFOM-Stereo ViT-L ETH3D", frame_root / "defom_vitl_eth3d", "frame_based_gt", "yes"),
        Method("CREStereo", frame_root / "crestereo", "frame_based_gt", "yes"),
        Method("S2M2-L full", frame_root / "S2M2-L_full", "frame_based_gt", "yes"),
        Method("S2M2-XL", frame_root / "S2M2-XL", "frame_based_gt", "yes"),
        Method("MonSter++ MixAll", frame_root / "monster_mixall", "frame_based_gt", "yes"),
        Method("RT-MonSter++ zero-shot", frame_root / "rtmonster_zeroshot", "frame_based_gt", "yes"),
        Method("RAFT-Stereo Middlebury", frame_root / "raft_middlebury", "frame_based_gt", "yes"),
        Method("StereoAnywhere", frame_root / "stereoanywhere", "frame_based_gt", "yes"),
        Method("SGBM", frame_root / "SGBM", "frame_based_gt", "yes", "fragile classical baseline"),
    ]
    return [m for m in methods if m.pred_dir is not None and m.pred_dir.exists()]


def runtime_lookup(summary_csv: Path) -> dict[str, dict[str, float | str]]:
    rows = read_csv(summary_csv)
    out = {}
    for row in rows:
        out[row["Method"]] = {
            "runtime": float(row["Runtime ↓"]) if row.get("Runtime ↓") else math.nan,
            "vram": float(row["VRAM ↓"]) if row.get("VRAM ↓") else math.nan,
        }
    return out


def metadata_runtime(method: Method) -> tuple[float, float]:
    if method.pred_dir is None:
        return math.nan, math.nan
    meta_path = method.pred_dir / "metadata.json"
    if not meta_path.exists():
        meta_path = method.pred_dir / "summary.json"
    if not meta_path.exists():
        return math.nan, math.nan
    meta = json.loads(meta_path.read_text())
    runtime = meta.get("avg_runtime_ms", meta.get("runtime_ms", math.nan))
    vram = meta.get("peak_vram_mb", meta.get("peak_gpu_memory_mb", math.nan))
    return float(runtime), float(vram)


def plot_scatter(rows: list[dict[str, object]], x: str, y: str, path: Path, title: str) -> None:
    plt.figure(figsize=(8, 5))
    for row in rows:
        try:
            xv = float(row[x])
            yv = float(row[y])
        except (ValueError, TypeError):
            continue
        if not math.isfinite(xv) or not math.isfinite(yv):
            continue
        label = str(row["method"])
        marker = "o"
        if "ConvGRU" in label:
            marker = "s"
        elif "StereoAnyVideo" in label:
            marker = "*"
        plt.scatter(xv, yv, marker=marker, s=70)
        plt.annotate(label, (xv, yv), fontsize=7, xytext=(4, 3), textcoords="offset points")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_bar(rows: list[dict[str, object]], path: Path) -> None:
    labels = [str(r["method"]) for r in rows]
    raw = [float(r["raw_temporal_diff"]) for r in rows]
    mc = [float(r["motion_compensated_temporal_mae"]) for r in rows]
    x = np.arange(len(labels))
    plt.figure(figsize=(12, 5))
    plt.bar(x - 0.2, raw, width=0.4, label="raw")
    plt.bar(x + 0.2, mc, width=0.4, label="motion compensated")
    plt.xticks(x, labels, rotation=75, ha="right", fontsize=7)
    plt.ylabel("temporal disparity MAE px")
    plt.title("Raw vs motion-compensated temporal ranking")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def make_qualitative(out_dir: Path, frames: list[dict[str, object]], preds: dict[str, list[np.ndarray]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = [0, len(frames) // 2, len(frames) - 1]
    methods = [m for m in ["S2M2-L@736", "StereoAnyVideo@384x640", "ConvGRU V2 e40", "EMA alpha=0.7", "median5 non-causal"] if m in preds]
    vmax = np.nanpercentile(np.stack([preds[m][i] for m in methods for i in selected]), 98)
    for i in selected:
        rgb = load_rgb(Path(frames[i]["left_path"]))
        tiles = [rgb]
        titles = ["RGB"]
        for m in methods:
            disp = preds[m][i]
            color = cv2.applyColorMap(np.clip(disp / max(vmax, 1e-6) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            tiles.append(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
            titles.append(m)
        h, w = 220, 275
        canvas = np.full((h, w * len(tiles), 3), 255, np.uint8)
        for j, tile in enumerate(tiles):
            resized = cv2.resize(tile, (w, h - 28), interpolation=cv2.INTER_AREA)
            canvas[28:, j * w : (j + 1) * w] = resized
            cv2.putText(canvas, titles[j][:28], (j * w + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"frame_{frames[i]['frame_id']}.png"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        vals = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                vals.append("" if not math.isfinite(val) else f"{val:.4f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", type=Path, default=Path("dataset/SCARED/curated/temporal_gt/test_dataset_9_keyframe_3/metadata.csv"))
    parser.add_argument("--base-dir", type=Path, default=Path("results/temporal evaluation/gt_temporal_test_dataset_9_keyframe_3"))
    parser.add_argument("--frame-methods-dir", type=Path, default=Path("results/temporal evaluation/frame_based_gt/native_frame_methods"))
    parser.add_argument("--summary-csv", type=Path, default=Path("results/temporal evaluation/temporal_evaluation.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/temporal evaluation/fair_interpretability_checks"))
    parser.add_argument("--min-valid-ratio", type=float, default=0.20)
    args = parser.parse_args()

    out = args.out_dir
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "qualitative").mkdir(parents=True, exist_ok=True)
    log = []
    frames = load_metadata(args.metadata_csv, args.min_valid_ratio)
    log.append(f"frames_with_gt={len(frames)}")
    methods = method_catalog(args.base_dir / "predictions", args.frame_methods_dir)
    log.append(f"methods={len(methods)}")

    gt_disps = [np.load(Path(f["gt_disp_path"])).astype(np.float32) for f in frames]
    gt_depths = [np.load(Path(f["gt_depth_path"])).astype(np.float32) for f in frames]
    masks = [np.load(Path(f["valid_mask_path"])).astype(bool) for f in frames]
    fxb = [calibration_fx_baseline(Path(f["calib_path"])) for f in frames]

    preds: dict[str, list[np.ndarray]] = {}
    for method in methods:
        preds[method.name] = [
            load_disp_for_method(method, str(frame["frame_id"]), str(frame["frame_name"])) for frame in frames
        ]

    base = preds["S2M2-L@736"]
    preds["EMA alpha=0.7"] = ema_sequence(base, alpha=0.7)
    preds["median5 non-causal"] = median5_sequence(base)
    methods.extend(
        [
            Method("EMA alpha=0.7", None, "simple_baseline_from_s2m2_l736", "yes"),
            Method("median5 non-causal", None, "simple_baseline_from_s2m2_l736", "no", "uses future frames"),
        ]
    )

    gt_valid_pixels = int(sum(mask.sum() for mask in masks))
    summary_rows = []
    coverage_rows = []
    for method in methods:
        geom = []
        eval_px = 0
        for i, frame in enumerate(frames):
            fx, baseline = fxb[i]
            g = compute_geom(preds[method.name][i], gt_disps[i], gt_depths[i], masks[i], fx, baseline)
            geom.append(g)
            eval_px += int(g["evaluated_pixels"])
        coverage = 100.0 * eval_px / gt_valid_pixels if gt_valid_pixels else math.nan
        rt, vram = metadata_runtime(method)
        summary_rows.append(
            {
                "method": method.name,
                "source": method.source,
                "causal": method.causal,
                "depth_mae_mm": mean([g["depth_mae"] for g in geom]),
                "bad_2mm_pct": mean([g["bad_2mm"] for g in geom]),
                "disp_mae_px": mean([g["disp_mae"] for g in geom]),
                "runtime_ms": rt,
                "peak_vram_mb": vram,
                "end_to_end_runtime_ms": rt,
                "frames": len(frames),
                "notes": method.notes,
            }
        )
        coverage_rows.append(
            {
                "method": method.name,
                "gt_valid_pixels": gt_valid_pixels,
                "evaluated_pixels": eval_px,
                "coverage_pct": coverage,
                "excluded_pct": 100.0 - coverage,
            }
        )

    common_masks = []
    for i, gt_mask in enumerate(masks):
        cm = gt_mask.copy()
        for name, seq in preds.items():
            pred = seq[i]
            cm &= np.isfinite(pred) & (pred > 0.1)
        common_masks.append(cm)

    common_rows = []
    common_valid_pixels = int(sum(cm.sum() for cm in common_masks))
    for method in methods:
        geom = []
        for i in range(len(frames)):
            fx, baseline = fxb[i]
            geom.append(compute_geom(preds[method.name][i], gt_disps[i], gt_depths[i], common_masks[i], fx, baseline))
        common_rows.append(
            {
                "method": method.name,
                "depth_mae_mm": mean([g["depth_mae"] for g in geom]),
                "bad_2mm_pct": mean([g["bad_2mm"] for g in geom]),
                "disp_mae_px": mean([g["disp_mae"] for g in geom]),
                "common_valid_pixels": common_valid_pixels,
                "common_coverage_pct": 100.0 * common_valid_pixels / gt_valid_pixels,
                "causal": method.causal,
                "notes": method.notes,
            }
        )

    temporal = temporal_metrics(preds, frames, masks)
    motion_rows = []
    for method in methods:
        motion_rows.append({"method": method.name, **temporal[method.name], "causal": method.causal, "notes": method.notes})

    temporal_by_method = {row["method"]: row for row in motion_rows}
    coverage_by_method = {row["method"]: row for row in coverage_rows}
    for row in summary_rows:
        row.update(temporal_by_method[row["method"]])
        row["coverage_pct"] = coverage_by_method[row["method"]]["coverage_pct"]

    backbone_rt, backbone_vram = metadata_runtime(Method("S2M2-L@736", args.base_dir / "predictions" / "S2M2-L_736", "", "yes"))
    runtime_rows = []
    for method in methods:
        rt, vram = metadata_runtime(method)
        if method.name.startswith("ConvGRU") or method.name.startswith("Tiny U-Net"):
            runtime_rows.append(
                {
                    "method": method.name,
                    "s2m2_l_backbone_runtime_ms": backbone_rt,
                    "refiner_overhead_ms": rt,
                    "end_to_end_runtime_ms": backbone_rt + rt,
                    "s2m2_l_backbone_peak_vram_mb": backbone_vram,
                    "refiner_peak_vram_mb": vram,
                    "total_peak_vram_mb_est": backbone_vram + vram,
                    "notes": "end-to-end estimate = backbone + cached-refiner timing",
                }
            )
            next(row for row in summary_rows if row["method"] == method.name)["end_to_end_runtime_ms"] = backbone_rt + rt
            next(row for row in summary_rows if row["method"] == method.name)["peak_vram_mb"] = backbone_vram + vram
        elif method.name == "S2M2-L@736":
            runtime_rows.append(
                {
                    "method": method.name,
                    "s2m2_l_backbone_runtime_ms": backbone_rt,
                    "refiner_overhead_ms": 0.0,
                    "end_to_end_runtime_ms": backbone_rt,
                    "s2m2_l_backbone_peak_vram_mb": backbone_vram,
                    "refiner_peak_vram_mb": 0.0,
                    "total_peak_vram_mb_est": backbone_vram,
                    "notes": "raw backbone",
                }
            )
            next(row for row in summary_rows if row["method"] == method.name)["end_to_end_runtime_ms"] = backbone_rt
        elif method.name in {"EMA alpha=0.7", "median5 non-causal"}:
            runtime_rows.append(
                {
                    "method": method.name,
                    "s2m2_l_backbone_runtime_ms": backbone_rt,
                    "refiner_overhead_ms": 0.0,
                    "end_to_end_runtime_ms": backbone_rt,
                    "s2m2_l_backbone_peak_vram_mb": backbone_vram,
                    "refiner_peak_vram_mb": 0.0,
                    "total_peak_vram_mb_est": backbone_vram,
                    "notes": "CPU post-processing overhead not timed; median5 uses future frames",
                }
            )
            next(row for row in summary_rows if row["method"] == method.name)["end_to_end_runtime_ms"] = backbone_rt
            next(row for row in summary_rows if row["method"] == method.name)["peak_vram_mb"] = backbone_vram
        else:
            runtime_rows.append(
                {
                    "method": method.name,
                    "s2m2_l_backbone_runtime_ms": "",
                    "refiner_overhead_ms": "",
                    "end_to_end_runtime_ms": rt,
                    "s2m2_l_backbone_peak_vram_mb": "",
                    "refiner_peak_vram_mb": "",
                    "total_peak_vram_mb_est": vram,
                    "notes": "native method runtime when metadata exists",
                }
            )
            next(row for row in summary_rows if row["method"] == method.name)["end_to_end_runtime_ms"] = rt

    convgru_names = {"S2M2-L@736", "ConvGRU V2 e30", "ConvGRU V2 e40", "ConvGRU V2 e50", "ConvGRU V2 latest"}
    convgru_rows = []
    for row in summary_rows:
        if row["method"] in convgru_names:
            runtime = next(r for r in runtime_rows if r["method"] == row["method"])
            convgru_rows.append(
                {
                    "method": row["method"],
                    "depth_mae_mm": row["depth_mae_mm"],
                    "disp_mae_px": row["disp_mae_px"],
                    "raw_temporal_diff": row["raw_temporal_diff"],
                    "motion_compensated_temporal_mae": row["motion_compensated_temporal_mae"],
                    "coverage_pct": row["coverage_pct"],
                    "total_runtime_ms": runtime["end_to_end_runtime_ms"],
                }
            )

    summary_rows = sorted(summary_rows, key=lambda r: (float(r["depth_mae_mm"]), float(r["motion_compensated_temporal_mae"])))
    common_rows = sorted(common_rows, key=lambda r: float(r["depth_mae_mm"]))
    motion_rows = sorted(motion_rows, key=lambda r: float(r["motion_compensated_temporal_mae"]))
    convgru_rows = sorted(convgru_rows, key=lambda r: str(r["method"]))

    summary_cols = ["method", "source", "causal", "depth_mae_mm", "bad_2mm_pct", "disp_mae_px", "raw_temporal_diff", "motion_compensated_temporal_mae", "coverage_pct", "runtime_ms", "end_to_end_runtime_ms", "peak_vram_mb", "frames", "notes"]
    common_cols = ["method", "depth_mae_mm", "bad_2mm_pct", "disp_mae_px", "common_valid_pixels", "common_coverage_pct", "causal", "notes"]
    coverage_cols = ["method", "gt_valid_pixels", "evaluated_pixels", "coverage_pct", "excluded_pct"]
    motion_cols = ["method", "raw_temporal_diff", "motion_compensated_temporal_mae", "temporal_pairs", "flow", "causal", "notes"]
    runtime_cols = ["method", "s2m2_l_backbone_runtime_ms", "refiner_overhead_ms", "end_to_end_runtime_ms", "s2m2_l_backbone_peak_vram_mb", "refiner_peak_vram_mb", "total_peak_vram_mb_est", "notes"]
    convgru_cols = ["method", "depth_mae_mm", "disp_mae_px", "raw_temporal_diff", "motion_compensated_temporal_mae", "coverage_pct", "total_runtime_ms"]
    write_csv(out / "summary.csv", summary_rows, summary_cols)
    write_csv(out / "summary_common_mask.csv", common_rows, common_cols)
    write_csv(out / "coverage.csv", coverage_rows, coverage_cols)
    write_csv(out / "motion_compensated_metrics.csv", motion_rows, motion_cols)
    write_csv(out / "runtime_pipeline.csv", runtime_rows, runtime_cols)
    write_csv(out / "convgru_checkpoints.csv", convgru_rows, convgru_cols)

    plot_scatter(summary_rows, "motion_compensated_temporal_mae", "depth_mae_mm", out / "plots/depth_mae_vs_motion_comp_temporal.png", "Depth MAE vs motion-compensated temporal error")
    plot_bar(motion_rows, out / "plots/raw_vs_motion_comp_temporal_ranking.png")
    plot_scatter(convgru_rows, "motion_compensated_temporal_mae", "depth_mae_mm", out / "plots/convgru_checkpoint_tradeoff.png", "ConvGRU checkpoint trade-off")
    plot_scatter(summary_rows, "end_to_end_runtime_ms", "motion_compensated_temporal_mae", out / "plots/runtime_vs_temporal_performance.png", "Runtime vs temporal performance")
    make_qualitative(out / "qualitative", frames, preds)

    raw = next(r for r in summary_rows if r["method"] == "S2M2-L@736")
    conv = [r for r in summary_rows if str(r["method"]).startswith("ConvGRU")]
    best_temporal = min(conv, key=lambda r: float(r["motion_compensated_temporal_mae"]))
    best_trade = min(conv, key=lambda r: float(r["depth_mae_mm"]) + 0.1 * float(r["motion_compensated_temporal_mae"]))
    report = (
        "# SCARED temporal GT fairness checks\n\n"
        f"Frames: `{len(frames)}` with GT valid-pixel ratio >= `{args.min_valid_ratio}`.\n\n"
        "Optical flow: OpenCV Farneback local fallback. The motion-compensated metric warps previous disparity toward the current frame before computing temporal MAE. This is an interpretability check, not a learned flow benchmark.\n\n"
        "## Summary\n\n"
        + markdown_table(summary_rows, summary_cols)
        + "\n\n## Common Valid-Pixel Intersection\n\n"
        + markdown_table(common_rows, common_cols)
        + "\n\n## ConvGRU Checkpoints\n\n"
        + markdown_table(convgru_rows, convgru_cols)
        + "\n\n## Interpretation\n\n"
        f"- Raw S2M2-L motion-compensated temporal MAE: `{raw['motion_compensated_temporal_mae']:.4f}` px.\n"
        f"- Best ConvGRU motion-compensated temporal MAE: `{best_temporal['method']}` at `{best_temporal['motion_compensated_temporal_mae']:.4f}` px.\n"
        f"- Best simple trade-off among ConvGRU checkpoints by depth+temporal score: `{best_trade['method']}`.\n"
        "- Median-5 is explicitly non-causal and uses future frames.\n"
        "- Runtime for refiners is reported both as refiner overhead and estimated full S2M2-L+refiner pipeline runtime.\n"
    )
    (out / "report.md").write_text(report)
    (out / "run.log").write_text("\n".join(log) + "\n")
    print(f"Wrote {out}")
    print(f"Best ConvGRU temporal: {best_temporal['method']} {best_temporal['motion_compensated_temporal_mae']:.4f}")
    print(f"Raw S2M2-L motion-comp temporal: {raw['motion_compensated_temporal_mae']:.4f}")


if __name__ == "__main__":
    main()
