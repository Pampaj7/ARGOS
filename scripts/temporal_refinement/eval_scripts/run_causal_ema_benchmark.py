#!/usr/bin/env python3
"""Focused causal EMA benchmark on the SCARED temporal-GT sequence."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.temporal_refinement.extend_temporal_gt_fairness_checks import (
    Method,
    calibration_fx_baseline,
    compute_geom,
    ema_sequence,
    load_disp_for_method,
    load_metadata,
    load_rgb,
    make_qualitative,
    mean,
    metadata_runtime,
    temporal_metrics,
    write_csv,
)


ALPHAS = (0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)


def load_sequence(method: Method, frames: list[dict[str, object]]) -> list[np.ndarray]:
    return [
        load_disp_for_method(method, str(frame["frame_id"]), str(frame["frame_name"]))
        for frame in frames
    ]


def build_common_masks(preds: dict[str, list[np.ndarray]], masks: list[np.ndarray]) -> list[np.ndarray]:
    common = []
    for i, gt_mask in enumerate(masks):
        mask = gt_mask.copy()
        for seq in preds.values():
            pred = seq[i]
            mask &= np.isfinite(pred) & (pred > 0.1)
        common.append(mask)
    return common


def evaluate_sequence(
    name: str,
    seq: list[np.ndarray],
    frames: list[dict[str, object]],
    gt_disps: list[np.ndarray],
    gt_depths: list[np.ndarray],
    common_masks: list[np.ndarray],
    temporal: dict[str, dict[str, float]],
    runtime_ms: float,
    peak_vram_mb: float,
    causal: str,
    notes: str = "",
) -> dict[str, object]:
    geom = []
    eval_px = 0
    gt_px = 0
    for i, frame in enumerate(frames):
        fx, baseline = calibration_fx_baseline(Path(frame["calib_path"]))
        result = compute_geom(seq[i], gt_disps[i], gt_depths[i], common_masks[i], fx, baseline)
        geom.append(result)
        eval_px += int(result["evaluated_pixels"])
        gt_px += int(common_masks[i].sum())
    coverage = 100.0 * eval_px / gt_px if gt_px else math.nan
    return {
        "method": name,
        "depth_mae_mm": mean([g["depth_mae"] for g in geom]),
        "bad_2mm_pct": mean([g["bad_2mm"] for g in geom]),
        "disp_mae_px": mean([g["disp_mae"] for g in geom]),
        "raw_temporal_diff": temporal[name]["raw_temporal_diff"],
        "motion_compensated_temporal_mae": temporal[name]["motion_compensated_temporal_mae"],
        "coverage_pct": coverage,
        "runtime_ms": runtime_ms,
        "peak_vram_mb": peak_vram_mb,
        "causal": causal,
        "notes": notes,
    }


def plot_depth_vs_temporal(rows: list[dict[str, object]], out: Path) -> None:
    plt.figure(figsize=(8, 5))
    for row in rows:
        label = str(row["method"])
        marker = "s" if "EMA" in label else ("*" if "StereoAnyVideo" in label else "o")
        plt.scatter(float(row["motion_compensated_temporal_mae"]), float(row["depth_mae_mm"]), s=70, marker=marker)
        plt.annotate(label, (float(row["motion_compensated_temporal_mae"]), float(row["depth_mae_mm"])), fontsize=7, xytext=(4, 3), textcoords="offset points")
    plt.xlabel("motion_compensated_temporal_mae")
    plt.ylabel("depth_mae_mm")
    plt.title("Depth MAE vs motion-compensated temporal MAE")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def plot_alpha(rows: list[dict[str, object]], y_col: str, out: Path, title: str) -> None:
    plt.figure(figsize=(7, 4.5))
    for prefix in ["S2M2-S@512 + EMA", "S2M2-L@736 + EMA"]:
        vals = [r for r in rows if str(r["method"]).startswith(prefix)]
        xs = [float(r["alpha"]) for r in vals]
        ys = [float(r[y_col]) for r in vals]
        plt.plot(xs, ys, marker="o", label=prefix)
    plt.xlabel("alpha")
    plt.ylabel(y_col)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def plot_model_comparison(rows: list[dict[str, object]], out: Path) -> None:
    keep = [
        r
        for r in rows
        if r["method"] in {"ConvGRU V2 e40", "StereoAnyVideo@384x640", "S2M2-S@512", "S2M2-L@736"}
        or str(r["method"]).endswith("best")
    ]
    labels = [str(r["method"]) for r in keep]
    temporal = [float(r["motion_compensated_temporal_mae"]) for r in keep]
    depth = [float(r["depth_mae_mm"]) for r in keep]
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    ax1.bar(x - 0.18, temporal, width=0.36, label="motion temporal px")
    ax1.set_ylabel("motion temporal px")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, depth, width=0.36, color="tab:orange", label="depth MAE mm")
    ax2.set_ylabel("depth MAE mm")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right")
    ax1.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        vals = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                vals.append("" if not math.isfinite(value) else f"{value:.4f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", type=Path, default=Path("dataset/SCARED/curated/temporal_gt/test_dataset_9_keyframe_3/metadata.csv"))
    parser.add_argument("--base-dir", type=Path, default=Path("results/03_temporal_refinement/evaluation/gt_temporal_test_dataset_9_keyframe_3"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/03_temporal_refinement/evaluation/causal_ema_benchmark"))
    parser.add_argument("--min-valid-ratio", type=float, default=0.20)
    args = parser.parse_args()

    out = args.out_dir
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "qualitative").mkdir(parents=True, exist_ok=True)

    frames = load_metadata(args.metadata_csv, args.min_valid_ratio)
    gt_disps = [np.load(Path(f["gt_disp_path"])).astype(np.float32) for f in frames]
    gt_depths = [np.load(Path(f["gt_depth_path"])).astype(np.float32) for f in frames]
    masks = [np.load(Path(f["valid_mask_path"])).astype(bool) for f in frames]
    pred_root = args.base_dir / "predictions"

    raw_methods = {
        "S2M2-S@512": Method("S2M2-S@512", pred_root / "S2M2-S_512", "temporal_gt_existing", "yes"),
        "S2M2-L@736": Method("S2M2-L@736", pred_root / "S2M2-L_736", "temporal_gt_existing", "yes"),
        "ConvGRU V2 e40": Method("ConvGRU V2 e40", pred_root / "ConvGRU-L736_temporal_refinement_train_convgru_l736_v2_scheduled:epoch_0040", "temporal_gt_existing", "yes"),
        "StereoAnyVideo@384x640": Method("StereoAnyVideo@384x640", pred_root / "StereoAnyVideo_384x640", "temporal_gt_existing", "no"),
    }

    preds: dict[str, list[np.ndarray]] = {name: load_sequence(method, frames) for name, method in raw_methods.items()}
    for alpha in ALPHAS:
        preds[f"S2M2-S@512 + EMA a={alpha:.2f}"] = ema_sequence(preds["S2M2-S@512"], alpha)
        preds[f"S2M2-L@736 + EMA a={alpha:.2f}"] = ema_sequence(preds["S2M2-L@736"], alpha)

    common_masks = build_common_masks(preds, masks)
    temporal = temporal_metrics(preds, frames, masks)

    s_rt, s_vram = metadata_runtime(raw_methods["S2M2-S@512"])
    l_rt, l_vram = metadata_runtime(raw_methods["S2M2-L@736"])
    conv_overhead, conv_vram = metadata_runtime(raw_methods["ConvGRU V2 e40"])
    sav_rt, sav_vram = metadata_runtime(raw_methods["StereoAnyVideo@384x640"])

    rows = []
    rows.append(evaluate_sequence("S2M2-S@512", preds["S2M2-S@512"], frames, gt_disps, gt_depths, common_masks, temporal, s_rt, s_vram, "yes"))
    rows.append(evaluate_sequence("S2M2-L@736", preds["S2M2-L@736"], frames, gt_disps, gt_depths, common_masks, temporal, l_rt, l_vram, "yes"))
    rows.append(evaluate_sequence("ConvGRU V2 e40", preds["ConvGRU V2 e40"], frames, gt_disps, gt_depths, common_masks, temporal, l_rt + conv_overhead, l_vram + conv_vram, "yes", "S2M2-L backbone + refiner overhead"))
    rows.append(evaluate_sequence("StereoAnyVideo@384x640", preds["StereoAnyVideo@384x640"], frames, gt_disps, gt_depths, common_masks, temporal, sav_rt, sav_vram, "no"))

    s_sweep = []
    l_sweep = []
    for alpha in ALPHAS:
        for backbone, runtime, vram, target in [
            ("S2M2-S@512", s_rt, s_vram, s_sweep),
            ("S2M2-L@736", l_rt, l_vram, l_sweep),
        ]:
            name = f"{backbone} + EMA a={alpha:.2f}"
            row = evaluate_sequence(name, preds[name], frames, gt_disps, gt_depths, common_masks, temporal, runtime, vram, "yes")
            row["alpha"] = alpha
            target.append(row)
            rows.append(row)

    def trade_score(row: dict[str, object]) -> float:
        return float(row["depth_mae_mm"]) + 0.2 * float(row["motion_compensated_temporal_mae"])

    best_s_depth = min(s_sweep, key=lambda r: float(r["depth_mae_mm"]))
    best_s_temporal = min(s_sweep, key=lambda r: float(r["motion_compensated_temporal_mae"]))
    best_s_trade = min(s_sweep, key=trade_score)
    best_l_depth = min(l_sweep, key=lambda r: float(r["depth_mae_mm"]))
    best_l_temporal = min(l_sweep, key=lambda r: float(r["motion_compensated_temporal_mae"]))
    best_l_trade = min(l_sweep, key=trade_score)

    best_rows = []
    for label, row in [
        ("best_s2m2s_depth", best_s_depth),
        ("best_s2m2s_temporal", best_s_temporal),
        ("best_s2m2s_tradeoff", best_s_trade),
        ("best_s2m2l_depth", best_l_depth),
        ("best_s2m2l_temporal", best_l_temporal),
        ("best_s2m2l_tradeoff", best_l_trade),
    ]:
        best_rows.append({"selection": label, **row})

    # Friendly aliases for the comparison plot.
    rows_for_plot = rows + [
        {**best_s_trade, "method": "S2M2-S EMA best"},
        {**best_l_trade, "method": "S2M2-L EMA best"},
    ]

    columns = [
        "method",
        "depth_mae_mm",
        "bad_2mm_pct",
        "disp_mae_px",
        "raw_temporal_diff",
        "motion_compensated_temporal_mae",
        "coverage_pct",
        "runtime_ms",
        "peak_vram_mb",
        "causal",
        "notes",
    ]
    sweep_cols = ["method", "alpha", *columns[1:]]
    best_cols = ["selection", "method", "alpha", *columns[1:]]
    rows = sorted(rows, key=lambda r: (float(r["motion_compensated_temporal_mae"]), float(r["depth_mae_mm"])))
    write_csv(out / "summary.csv", rows, columns)
    write_csv(out / "ema_sweep_s2m2s.csv", s_sweep, sweep_cols)
    write_csv(out / "ema_sweep_s2m2l.csv", l_sweep, sweep_cols)
    write_csv(out / "best_tradeoffs.csv", best_rows, best_cols)

    plot_depth_vs_temporal(rows, out / "plots/depth_mae_vs_motion_comp_temporal_mae.png")
    plot_alpha(s_sweep + l_sweep, "depth_mae_mm", out / "plots/alpha_vs_depth_mae.png", "Alpha vs depth MAE")
    plot_alpha(s_sweep + l_sweep, "motion_compensated_temporal_mae", out / "plots/alpha_vs_temporal_mae.png", "Alpha vs motion-compensated temporal MAE")
    plot_model_comparison(rows_for_plot, out / "plots/s2m2s_ema_vs_s2m2l_ema_vs_convgru.png")
    make_qualitative(
        out / "qualitative",
        frames,
        {
            "S2M2-L@736": preds["S2M2-L@736"],
            "S2M2-S@512": preds["S2M2-S@512"],
            "ConvGRU V2 e40": preds["ConvGRU V2 e40"],
            "StereoAnyVideo@384x640": preds["StereoAnyVideo@384x640"],
            f"S2M2-S@512 + EMA a={best_s_trade['alpha']:.2f}": preds[best_s_trade["method"]],
            f"S2M2-L@736 + EMA a={best_l_trade['alpha']:.2f}": preds[best_l_trade["method"]],
        },
    )

    conv = next(r for r in rows if r["method"] == "ConvGRU V2 e40")
    answer = (
        "yes" if float(best_s_trade["motion_compensated_temporal_mae"]) <= float(conv["motion_compensated_temporal_mae"])
        and float(best_s_trade["depth_mae_mm"]) <= float(conv["depth_mae_mm"])
        else "no"
    )
    report = (
        "# Causal EMA Benchmark on SCARED Temporal GT\n\n"
        f"Frames: `{len(frames)}` GT-valid frames using the same common-mask and Farneback motion-compensated protocol as the fairness check.\n\n"
        "## Summary\n\n"
        + markdown_table(rows, columns)
        + "\n\n## Best Tradeoffs\n\n"
        + markdown_table(best_rows, best_cols)
        + "\n\n## Answer\n\n"
        f"- Best S2M2-S EMA alpha by tradeoff: `{best_s_trade['alpha']:.2f}`.\n"
        f"- Best S2M2-L EMA alpha by tradeoff: `{best_l_trade['alpha']:.2f}`.\n"
        f"- Can S2M2-S@512 + causal EMA match or beat S2M2-L@736 + ConvGRU e40 on both depth and motion-compensated temporal MAE? `{answer}`.\n"
        f"- S2M2-S EMA runtime/VRAM: `{best_s_trade['runtime_ms']:.2f} ms`, `{best_s_trade['peak_vram_mb']:.1f} MB`.\n"
        f"- ConvGRU e40 end-to-end runtime/VRAM: `{conv['runtime_ms']:.2f} ms`, `{conv['peak_vram_mb']:.1f} MB`.\n"
        "- EMA is causal and has negligible compute beyond the selected backbone.\n"
    )
    (out / "report.md").write_text(report)
    (out / "run.log").write_text(
        f"frames={len(frames)}\n"
        f"alphas={','.join(f'{a:.2f}' for a in ALPHAS)}\n"
        "flow=OpenCV Farneback local fallback\n"
        "common_mask=GT valid mask intersected with positive finite predictions for all compared methods\n"
    )
    print(f"Wrote {out}")
    print(f"best_s2m2s_alpha={best_s_trade['alpha']:.2f}")
    print(f"best_s2m2l_alpha={best_l_trade['alpha']:.2f}")
    print(f"answer={answer}")


if __name__ == "__main__":
    main()
