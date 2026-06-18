#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import shutil
import textwrap
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.temporal_refinement.lib.training import colorize
from scripts.temporal_refinement.evaluate_temporal_refinement import (
    group_rows,
    infer_convgru,
    infer_tiny,
    load_disp,
    load_model_checkpoint,
    load_rgb,
    path_for,
    read_rows,
    temporal_median,
)
from scripts.temporal_refinement.train_temporal_refiner_fastcache import split_fast_by_sequence


OUT = Path("presentation/argos_progress")
CACHE = Path("results/03_temporal_refinement/cache/large_v3_s2m2s512_fast")
INDEX = "index_s2m2l736.csv"
UNIFIED = Path("results/temporal_refinement_evaluation_l736_v1")


def ensure_dirs():
    for name in ["tables", "plots", "diagrams", "qualitative", "videos", "gifs", "thumbnails", "source_data"]:
        (OUT / name).mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], columns: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[dict], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join(lines) + "\n"


def copy_source(path: Path) -> str:
    if not path.exists():
        return ""
    dst = OUT / "source_data" / path.as_posix().replace("/", "__")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dst)
    return dst.as_posix()


evidence_rows: list[dict] = []


def evidence(asset, asset_type, claim, source_file, source_row="", protocol="", notes=""):
    evidence_rows.append(
        {
            "asset": asset,
            "asset_type": asset_type,
            "claim_or_metric": claim,
            "source_file": source_file,
            "source_row_or_checkpoint": source_row,
            "evaluation_protocol": protocol,
            "notes": notes,
        }
    )


def safe_float(v):
    try:
        if v in ("", None):
            return math.nan
        return float(v)
    except Exception:
        return math.nan


def build_model_audit():
    unified = read_csv(UNIFIED / "summary.csv")
    stereo_repo = read_csv(Path("results/video_stereo_repos/report.csv"))
    servct = read_csv(Path("results/metrics/servct_scoreboard.csv"))
    sav32 = read_csv(Path("results/stereoanyvideo_temporal_eval/consecutive32/summary.csv"))

    rows = [
        {
            "model": "S2M2-S",
            "model_family": "frame stereo transformer",
            "repository_or_source": "stereo/s2m2",
            "frame_or_video": "frame",
            "general_or_surgical": "general pretrained, surgical evaluated",
            "causal": "yes",
            "input_type": "rectified stereo pair",
            "tested_resolution": "512 and full",
            "dataset_tested": "SCARED dataset_8, SERV-CT",
            "status": "quantitatively_evaluated",
            "main_advantage": "fastest S2M2 deployment candidate",
            "main_limitation": "lower accuracy than L/XL",
            "evidence_source": "results/s2m2_resolution_tradeoff/s2m2_resolution_tradeoff.csv",
        },
        {
            "model": "S2M2-L",
            "model_family": "frame stereo transformer",
            "repository_or_source": "stereo/s2m2",
            "frame_or_video": "frame",
            "general_or_surgical": "general pretrained, surgical evaluated",
            "causal": "yes",
            "input_type": "rectified stereo pair",
            "tested_resolution": "736 and full",
            "dataset_tested": "SCARED, temporal cache",
            "status": "used_in_final_pipeline",
            "main_advantage": "best practical frame backbone for temporal refinement",
            "main_limitation": "frame-wise temporal flicker remains",
            "evidence_source": "results/temporal_refinement_evaluation_l736_v1/summary.csv",
        },
        {
            "model": "S2M2-XL",
            "model_family": "frame stereo transformer",
            "repository_or_source": "stereo/s2m2",
            "frame_or_video": "frame",
            "general_or_surgical": "general pretrained, surgical evaluated",
            "causal": "yes",
            "input_type": "rectified stereo pair",
            "tested_resolution": "1024/full",
            "dataset_tested": "SCARED size tradeoff",
            "status": "quantitatively_evaluated",
            "main_advantage": "teacher-sized frame model",
            "main_limitation": "marginal benefit over L in current evidence",
            "evidence_source": "results/s2m2_size_tradeoff/s2m2_size_tradeoff.csv",
        },
        {
            "model": "StereoAnyVideo",
            "model_family": "video stereo",
            "repository_or_source": "stereo/stereoanyvideo",
            "frame_or_video": "video",
            "general_or_surgical": "general video stereo, surgical evaluated",
            "causal": "no",
            "input_type": "stereo sequence",
            "tested_resolution": "384x640",
            "dataset_tested": "SCARED consecutive32 and long cache",
            "status": "used_in_final_pipeline",
            "main_advantage": "strong temporal teacher/reference",
            "main_limitation": "heavy, non-causal upper-bound style teacher",
            "evidence_source": "results/stereoanyvideo_temporal_eval/consecutive32/summary.csv",
        },
        {
            "model": "Tiny U-Net refiner",
            "model_family": "ARGOS residual temporal refiner",
            "repository_or_source": "ARGOS",
            "frame_or_video": "window",
            "general_or_surgical": "surgical trained/refined",
            "causal": "no",
            "input_type": "RGB center + 5-frame disparity window",
            "tested_resolution": "crop training, full-frame eval",
            "dataset_tested": "SCARED temporal cache",
            "status": "quantitatively_evaluated",
            "main_advantage": "simple residual teacher-student prototype",
            "main_limitation": "bidirectional/non-causal and modest temporal gains",
            "evidence_source": "results/temporal_refinement_evaluation_l736_v1/summary.csv",
        },
        {
            "model": "ConvGRU refiner V2 scheduled",
            "model_family": "ARGOS causal recurrent refiner",
            "repository_or_source": "ARGOS",
            "frame_or_video": "video/online recurrent",
            "general_or_surgical": "surgical trained/refined",
            "causal": "yes",
            "input_type": "RGB_t + S2M2-L disparity_t",
            "tested_resolution": "crop training, full-frame eval",
            "dataset_tested": "SCARED temporal cache",
            "status": "used_in_final_pipeline",
            "main_advantage": "causal learned temporal stabilization; epoch 30-50 Pareto candidates",
            "main_limitation": "temporal improvement does not prove geometry without GT",
            "evidence_source": "results/temporal_refinement_evaluation_l736_v1/checkpoint_metrics.csv",
        },
    ]
    for model in [
        "Fast-FoundationStereo",
        "FoundationStereo",
        "StereoAnywhere",
        "RAFT-Stereo",
        "CREStereo",
        "DEFOM-Stereo",
        "MonSter++",
        "IGEV++ / Selective-Stereo",
        "TC-Stereo",
        "TemporalStereo",
        "PPMStereo",
        "TemporallyConsistentDepth",
        "DynamicStereo",
        "BiDAStereo",
    ]:
        rows.append(
            {
                "model": model,
                "model_family": "stereo/depth baseline",
                "repository_or_source": "downloaded local repo or scouting report",
                "frame_or_video": "mixed",
                "general_or_surgical": "general",
                "causal": "",
                "input_type": "",
                "tested_resolution": "",
                "dataset_tested": "SERV-CT/SCARED smoke where available",
                "status": "downloaded" if model in {"TC-Stereo", "TemporalStereo", "PPMStereo", "TemporallyConsistentDepth"} else "inference_attempted",
                "main_advantage": "SOTA exploration candidate",
                "main_limitation": "not selected for current final temporal pipeline or incomplete compatibility evidence",
                "evidence_source": "results/video_stereo_repos/report.csv",
            }
        )
    # Fill key quantitative values from unified summary.
    by_method = {r["method"]: r for r in unified}
    for row in rows:
        if row["model"] == "S2M2-L" and "raw_s2m2_l736" in by_method:
            src = by_method["raw_s2m2_l736"]
            row.update({"temporal_diff": src.get("temporal_diff"), "runtime_ms": src.get("runtime_ms_per_frame"), "peak_vram_mb": src.get("peak_vram_mb")})
        if row["model"] == "StereoAnyVideo" and "stereoanyvideo" in by_method:
            src = by_method["stereoanyvideo"]
            row.update({"temporal_diff": src.get("temporal_diff"), "runtime_ms": src.get("runtime_ms_per_frame"), "peak_vram_mb": src.get("peak_vram_mb")})
        if row["model"] == "Tiny U-Net refiner" and "tiny_unet_conservative" in by_method:
            src = by_method["tiny_unet_conservative"]
            row.update({"temporal_diff": src.get("temporal_diff"), "runtime_ms": src.get("runtime_ms_per_frame"), "peak_vram_mb": src.get("peak_vram_mb")})
    cols = [
        "model",
        "model_family",
        "repository_or_source",
        "frame_or_video",
        "general_or_surgical",
        "causal",
        "input_type",
        "tested_resolution",
        "dataset_tested",
        "status",
        "disparity_mae_px",
        "depth_mae_mm",
        "temporal_diff",
        "runtime_ms",
        "peak_vram_mb",
        "main_advantage",
        "main_limitation",
        "evidence_source",
    ]
    write_csv(OUT / "tables/model_audit_complete.csv", rows, cols)
    (OUT / "tables/model_audit_complete.md").write_text(md_table(rows, cols))
    slide_cols = ["model", "model_family", "frame_or_video", "status", "main_advantage", "main_limitation"]
    write_csv(OUT / "tables/model_audit_slide_ready.csv", rows, slide_cols)
    evidence("tables/model_audit_complete.csv", "table", "repository-wide model status audit", "results/video_stereo_repos/report.csv; results/temporal_refinement_evaluation_l736_v1/summary.csv")


def build_dataset_tables():
    meta = json.loads(Path("results/03_temporal_refinement/cache/large_v3_s2m2s512_fast/metadata.json").read_text())
    long_meta = json.loads(Path("results/04_dataset_derivatives/SCARED/scared_long_sequences/metadata.json").read_text())
    rows = [
        {
            "dataset": "SCARED long temporal cache",
            "domain": "robotic/surgical stereo",
            "real_or_synthetic": "real",
            "stereo_or_video_or_rgbd": "stereo video",
            "number_of_sequences": meta["sequence_count"],
            "number_of_frames_used": 1040,
            "ground_truth_available": "no for current long validation rows",
            "ground_truth_type": "",
            "current_project_use": "temporal refinement training/evaluation",
            "limitations": "current unified full-frame validation sequence has has_gt=False",
            "status": "prepared and evaluated",
            "evidence_source": "results/03_temporal_refinement/cache/large_v3_s2m2s512_fast/metadata.json",
        },
        {
            "dataset": "SCARED dataset_8 keyframes",
            "domain": "surgical stereo",
            "real_or_synthetic": "real",
            "stereo_or_video_or_rgbd": "stereo keyframes + depth",
            "number_of_sequences": 1,
            "number_of_frames_used": 5,
            "ground_truth_available": "yes",
            "ground_truth_type": "depth TIFF / derived disparity",
            "current_project_use": "frame benchmark and transfer checks",
            "limitations": "small GT subset",
            "status": "prepared and benchmarked",
            "evidence_source": "dataset/SCARED/curated/keyframes_gt_dataset8/",
        },
        {
            "dataset": "SERV-CT ARGOS",
            "domain": "surgical/lab stereo",
            "real_or_synthetic": "real/lab",
            "stereo_or_video_or_rgbd": "stereo + CT-derived GT",
            "number_of_sequences": "",
            "number_of_frames_used": "",
            "ground_truth_available": "yes",
            "ground_truth_type": "disp_gt.npy, depth_gt_mm.npy, valid_mask.npy",
            "current_project_use": "baseline scoring and fine-tuning evidence",
            "limitations": "domain differs from SCARED temporal videos",
            "status": "prepared",
            "evidence_source": "dataset/SERVCT/argos/servct_argos/",
        },
        {
            "dataset": "Future RGB-D / 3D-printed phantom plans",
            "domain": "open surgery-like acquisition",
            "real_or_synthetic": "planned",
            "stereo_or_video_or_rgbd": "RGB-D / stereo planned",
            "number_of_sequences": "",
            "number_of_frames_used": 0,
            "ground_truth_available": "planned",
            "ground_truth_type": "not yet available",
            "current_project_use": "future ARGOS validation",
            "limitations": "not yet measured in repo",
            "status": "planned",
            "evidence_source": "README.md; docs/STATUS.md",
        },
    ]
    cols = [
        "dataset",
        "domain",
        "real_or_synthetic",
        "stereo_or_video_or_rgbd",
        "number_of_sequences",
        "number_of_frames_used",
        "ground_truth_available",
        "ground_truth_type",
        "current_project_use",
        "limitations",
        "status",
        "evidence_source",
    ]
    write_csv(OUT / "tables/datasets_complete.csv", rows, cols)
    write_csv(OUT / "tables/datasets_slide_ready.csv", rows, ["dataset", "domain", "number_of_frames_used", "ground_truth_available", "current_project_use", "limitations"])
    evidence("tables/datasets_complete.csv", "table", "dataset recap", "results/03_temporal_refinement/cache/large_v3_s2m2s512_fast/metadata.json")


def table_subsets():
    # Copy source data.
    for p in [
        "results/s2m2_resolution_tradeoff/s2m2_resolution_tradeoff.csv",
        "results/s2m2_size_tradeoff/s2m2_size_tradeoff.csv",
        "results/stereoanyvideo_temporal_eval/consecutive32/summary.csv",
        "results/temporal_refinement_evaluation_l736_v1/summary.csv",
        "results/temporal_refinement_evaluation_l736_v1/checkpoint_metrics.csv",
        "results/temporal_refinement_evaluation_l736_v1/baseline_metrics.csv",
        "results/metrics/servct_scoreboard.csv",
        "results/video_stereo_repos/report.csv",
    ]:
        copy_source(Path(p))
    # Frame stereo benchmark.
    frame = read_csv(Path("results/s2m2_resolution_tradeoff/s2m2_resolution_tradeoff.csv"))
    frame_cols = ["model", "resize_label", "valid_disp_mae", "valid_depth_mae", "avg_inference_time_ms", "peak_gpu_memory_mb", "image_resolution_used", "frames", "checkpoint"]
    write_csv(OUT / "tables/frame_stereo_benchmark.csv", frame, frame_cols)
    slide = [r for r in frame if r.get("model") in {"S", "L", "XL"} and r.get("resize_label") in {"full", "736", "512", "1024"}]
    write_csv(OUT / "tables/frame_stereo_benchmark_slide_ready.csv", slide, frame_cols[:-1])
    evidence("tables/frame_stereo_benchmark.csv", "table", "S2M2 frame benchmark", "results/s2m2_resolution_tradeoff/s2m2_resolution_tradeoff.csv")
    # Temporal models.
    unified = read_csv(UNIFIED / "summary.csv")
    ckpt = read_csv(UNIFIED / "checkpoint_metrics.csv")
    selected = []
    selected_names = {
        "raw_s2m2_l736",
        "stereoanyvideo",
        "tiny_unet_conservative",
        "convgru_v1_conservative",
        "convgru_v2_scheduled",
        "convgru_v2_scheduled:epoch_0030",
        "convgru_v2_scheduled:epoch_0040",
        "convgru_v2_scheduled:epoch_0050",
        "convgru_v2_scheduled:latest",
        "ema_alpha_0.7",
        "prev_blend_alpha_0.7",
        "median5_noncausal",
        "median5_causal",
    }
    for r in unified:
        if r["method"] in selected_names:
            selected.append(r)
    temporal_cols = [
        "method",
        "causal",
        "checkpoint",
        "temporal_diff",
        "teacher_delta_mae",
        "refined_to_backbone_mae",
        "refined_to_sav_mae",
        "runtime_ms_per_frame",
        "peak_vram_mb",
        "depth_mae_mm",
        "bad_2mm",
    ]
    write_csv(OUT / "tables/unified_fullframe_evaluation.csv", unified, temporal_cols)
    write_csv(OUT / "tables/unified_fullframe_evaluation_slide_ready.csv", selected, temporal_cols)
    write_csv(OUT / "tables/temporal_models_complete.csv", ckpt, temporal_cols)
    write_csv(OUT / "tables/temporal_models_slide_ready.csv", selected, temporal_cols)
    evidence("tables/unified_fullframe_evaluation.csv", "table", "unified full-frame temporal evaluation", "results/temporal_refinement_evaluation_l736_v1/summary.csv")


def simple_box_diagram(path_base: Path, title: str, boxes: list[str], arrows: bool = True):
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.axis("off")
    n = len(boxes)
    xs = np.linspace(0.08, 0.92, n)
    for i, (x, txt) in enumerate(zip(xs, boxes)):
        ax.text(
            x,
            0.5,
            txt,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", fc="#eef5ff", ec="#2f5f9f", lw=1.5),
            transform=ax.transAxes,
        )
        if arrows and i < n - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.08, 0.5), xytext=(x + 0.08, 0.5), arrowprops=dict(arrowstyle="->", lw=1.6), xycoords=ax.transAxes)
    ax.set_title(title, fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=180)
    fig.savefig(path_base.with_suffix(".svg"))
    plt.close(fig)


def build_diagrams():
    d = OUT / "diagrams"
    simple_box_diagram(d / "frame_based_stereo_benchmark", "Frame-Based Stereo Benchmark", ["Stereo\nleft/right", "S2M2-S\nS2M2-L\nS2M2-XL", "Disparity", "Accuracy\nRuntime\nVRAM"])
    simple_box_diagram(d / "teacher_student_temporal_refinement", "Teacher-Student Temporal Refinement", ["Frozen\nS2M2 backbone", "Frozen\nStereoAnyVideo teacher", "Trainable\nresidual refiner", "Temporally refined\ndisparity"])
    simple_box_diagram(d / "tiny_unet_refinement", "Tiny U-Net Refinement", ["RGB center", "5-frame disparity\nwindow", "Tiny 2D U-Net\nnon-causal context", "Residual disparity", "Refined output"])
    simple_box_diagram(d / "convgru_refinement", "Causal ConvGRU Refinement", ["RGB_t + disp_t", "Encoder", "ConvGRU\nhidden state", "Decoder", "Residual correction", "Online refined\ndisparity"])
    simple_box_diagram(d / "scheduled_loss_training", "Scheduled-Loss Training", ["Epochs 1-10\n0.40/0.35/0.20/0.20/0.05", "Epochs 11-30\nlinear transition", "Epochs 31-100\n0.25/0.25/0.40/0.10/0.05"])
    simple_box_diagram(d / "dataset_overview", "ARGOS Dataset / Cache Overview", ["SCARED videos\n8 streams", "1040 stereo frames", "1008 valid\n5-frame windows", "Cached streams\nS2M2-S/L + SAV", "Validation\n126 frames no GT"])
    for name in ["frame_based_stereo_benchmark", "teacher_student_temporal_refinement", "tiny_unet_refinement", "convgru_refinement", "scheduled_loss_training", "dataset_overview"]:
        evidence(f"diagrams/{name}.png", "diagram", name.replace("_", " "), "results/03_temporal_refinement/cache/large_v3_s2m2s512_fast/metadata.json")


def scatter_plot(rows, x, y, label, out_name, highlight=None):
    pts = [r for r in rows if np.isfinite(safe_float(r.get(x))) and np.isfinite(safe_float(r.get(y)))]
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in pts:
        name = r.get("method") or r.get("model") or ""
        is_hi = highlight and highlight in name
        ax.scatter(safe_float(r[x]), safe_float(r[y]), s=65 if is_hi else 30, color="#d62728" if is_hi else "#1f77b4", alpha=0.9)
        ax.annotate(name[:22], (safe_float(r[x]), safe_float(r[y])), fontsize=7, alpha=0.75)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(label)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    png = OUT / "plots" / f"{out_name}.png"
    pdf = OUT / "plots" / f"{out_name}.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    write_csv(OUT / "source_data" / f"{out_name}.csv", pts)
    evidence(f"plots/{out_name}.png", "plot", label, f"source_data/{out_name}.csv")


def build_plots():
    frame = read_csv(Path("results/s2m2_resolution_tradeoff/s2m2_resolution_tradeoff.csv"))
    unified = read_csv(UNIFIED / "summary.csv")
    ckpt = read_csv(UNIFIED / "checkpoint_metrics.csv")
    baselines = read_csv(UNIFIED / "baseline_metrics.csv")
    scatter_plot(frame, "avg_inference_time_ms", "valid_depth_mae", "Frame stereo: accuracy vs runtime (SCARED GT keyframes)", "frame_accuracy_vs_runtime")
    scatter_plot(frame, "peak_gpu_memory_mb", "avg_inference_time_ms", "Frame stereo: runtime vs VRAM", "runtime_vs_vram")
    scatter_plot(unified, "refined_to_backbone_mae", "temporal_diff", "Full-frame temporal diff vs backbone deviation (no GT)", "temporal_vs_backbone", "convgru_v2_scheduled:epoch_0030")
    scatter_plot(unified, "refined_to_backbone_mae", "teacher_delta_mae", "Teacher-delta vs backbone deviation", "teacher_delta_vs_backbone", "convgru_v2_scheduled:epoch_0030")
    scatter_plot(unified, "runtime_ms_per_frame", "temporal_diff", "Temporal diff vs runtime", "temporal_vs_runtime", "convgru_v2_scheduled:epoch_0030")
    # Temporal improvement relative to raw.
    raw = next((safe_float(r["temporal_diff"]) for r in unified if r["method"] == "raw_s2m2_l736"), math.nan)
    imp = []
    for r in unified:
        td = safe_float(r.get("temporal_diff"))
        if np.isfinite(raw) and np.isfinite(td):
            row = dict(r)
            row["temporal_improvement_vs_raw"] = raw - td
            imp.append(row)
    scatter_plot(imp, "refined_to_backbone_mae", "temporal_improvement_vs_raw", "Temporal improvement vs raw S2M2-L", "temporal_improvement_vs_raw", "convgru_v2_scheduled:epoch_0030")
    # Checkpoint evolution V2.
    v2 = [r for r in ckpt if r["method"].startswith("convgru_v2_scheduled")]
    xs, ys, td = [], [], []
    for r in v2:
        ck = Path(r.get("checkpoint", "")).stem
        if ck.startswith("epoch_"):
            xs.append(int(ck.split("_")[1]))
            ys.append(safe_float(r["teacher_delta_mae"]))
            td.append(safe_float(r["temporal_diff"]))
    if xs:
        order = np.argsort(xs)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(np.array(xs)[order], np.array(td)[order], marker="o", label="temporal_diff")
        ax.plot(np.array(xs)[order], np.array(ys)[order], marker="s", label="teacher_delta")
        ax.axvline(30, color="#d62728", linestyle="--", label="epoch 30 candidate")
        ax.set_title("ConvGRU V2 checkpoint evolution")
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / "plots/checkpoint_evolution_convgru_v2.png", dpi=180)
        fig.savefig(OUT / "plots/checkpoint_evolution_convgru_v2.pdf")
        plt.close(fig)
        evidence("plots/checkpoint_evolution_convgru_v2.png", "plot", "ConvGRU V2 checkpoint evolution", "results/temporal_refinement_evaluation_l736_v1/checkpoint_metrics.csv")
    # Schedule weights.
    epochs = np.arange(1, 101)
    weights = []
    for e in epochs:
        if e <= 10:
            w = (0.40, 0.35, 0.20, 0.20, 0.05)
        elif e <= 30:
            a = (e - 10) / 20.0
            w = tuple(i + a * (f - i) for i, f in zip((0.40, 0.35, 0.20, 0.20, 0.05), (0.25, 0.25, 0.40, 0.10, 0.05)))
        else:
            w = (0.25, 0.25, 0.40, 0.10, 0.05)
        weights.append(w)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, name in enumerate(["spatial", "abs_sav", "delta_sav", "res", "edge"]):
        ax.plot(epochs, [w[i] for w in weights], label=name)
    ax.set_title("Scheduled loss weights")
    ax.set_xlabel("epoch")
    ax.set_ylabel("weight")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "plots/scheduled_loss_weight_evolution.png", dpi=180)
    fig.savefig(OUT / "plots/scheduled_loss_weight_evolution.pdf")
    plt.close(fig)
    evidence("plots/scheduled_loss_weight_evolution.png", "plot", "scheduled loss weights", "results/temporal_refinement_train_convgru_l736_v2_scheduled/train_log.csv")
    scatter_plot(baselines, "refined_to_backbone_mae", "temporal_diff", "Causal vs non-causal classical baselines", "causal_vs_noncausal_baselines")


def load_eval_sequence():
    _, val_ids, _ = split_fast_by_sequence(CACHE, INDEX, 1)
    rows = read_rows(CACHE, INDEX, val_ids)
    rows_by_seq = group_rows(rows)
    rgbs = [load_rgb(CACHE, r) for r in rows]
    raw = [load_disp(CACHE, path_for(r, "s2m2_l736", "t")) for r in rows]
    sav = [load_disp(CACHE, path_for(r, "sav", "t")) for r in rows]
    return rows, rows_by_seq, rgbs, raw, sav


def infer_selected_methods(rows_by_seq):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = {}
    specs = [
        ("tiny_unet", "tiny_unet", Path("results/temporal_refinement_train_unet_s2m2l736_fastcache_v2_conservative/checkpoints/best.pt")),
        ("convgru_v1", "convgru", Path("results/temporal_refinement_train_convgru_l736_v1_100ep_b13/checkpoints/best.pt")),
        ("convgru_v2_epoch30", "convgru", Path("results/temporal_refinement_train_convgru_l736_v2_scheduled/checkpoints/epoch_0030.pt")),
        ("convgru_v2_final", "convgru", Path("results/temporal_refinement_train_convgru_l736_v2_scheduled/checkpoints/epoch_0100.pt")),
    ]
    for name, kind, ckpt in specs:
        model, _ = load_model_checkpoint(ckpt, kind, device)
        if kind == "tiny_unet":
            preds, residuals, _, _ = infer_tiny(model, CACHE, rows_by_seq, device, True, 128.0)
        else:
            preds, residuals, _, _ = infer_convgru(model, CACHE, rows_by_seq, device, True, 128.0)
        selected[name] = {"preds": preds, "residuals": residuals}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        evidence(f"videos/{name}", "inference", f"{name} predictions used for assets", str(ckpt), "full-frame validation sequence")
    return selected


def make_tile(img, label, size=(320, 240), vmax=None, cmap=cv2.COLORMAP_TURBO):
    if img.ndim == 2:
        img = colorize(img, vmax=vmax, cmap=cmap)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    tile = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (size[0], 24), (0, 0, 0), -1)
    cv2.putText(tile, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def save_video_and_gif(name, frames, fps=12):
    video_path = OUT / "videos" / f"{name}.mp4"
    gif_path = OUT / "gifs" / f"{name}.gif"
    thumb_path = OUT / "thumbnails" / f"{name}.png"
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fr in frames:
        writer.write(fr)
    writer.release()
    cv2.imwrite(str(thumb_path), frames[len(frames) // 2])
    try:
        import imageio.v2 as imageio
        gif_frames = [cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) for fr in frames[::2]]
        imageio.mimsave(gif_path, gif_frames, fps=max(1, fps // 2))
    except Exception:
        gif_path.write_text("GIF generation unavailable.\n")
    evidence(f"videos/{name}.mp4", "video", name, "results/temporal_refinement_evaluation_l736_v1/summary.csv", "validation sequence synchronized frames")


def build_qualitative_and_videos():
    rows, rows_by_seq, rgbs, raw, sav = load_eval_sequence()
    methods = infer_selected_methods(rows_by_seq)
    ema = []
    prev = None
    for cur in raw:
        ref = cur.copy() if prev is None else 0.7 * cur + 0.3 * prev
        ema.append(ref.astype(np.float32)); prev = ref
    med5 = temporal_median(raw, 5, causal=False)
    vmax = float(np.nanpercentile(np.concatenate([x.ravel() for x in raw + sav]), 99))
    idxs = [0, len(rows)//2, len(rows)-1, 30, 60, 90]
    for idx in sorted(set(i for i in idxs if 0 <= i < len(rows))):
        tiles = [
            make_tile(rgbs[idx], "RGB"),
            make_tile(raw[idx], "raw S2M2-L", vmax=vmax),
            make_tile(sav[idx], "StereoAnyVideo", vmax=vmax),
            make_tile(methods["tiny_unet"]["preds"][idx], "Tiny U-Net", vmax=vmax),
            make_tile(methods["convgru_v1"]["preds"][idx], "ConvGRU V1", vmax=vmax),
            make_tile(methods["convgru_v2_epoch30"]["preds"][idx], "ConvGRU V2 e30", vmax=vmax),
            make_tile(methods["convgru_v2_final"]["preds"][idx], "ConvGRU V2 final", vmax=vmax),
            make_tile(np.abs(methods["convgru_v2_epoch30"]["preds"][idx] - raw[idx]), "V2 e30 residual", vmax=5.0, cmap=cv2.COLORMAP_MAGMA),
        ]
        grid = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])
        cv2.imwrite(str(OUT / "qualitative" / f"comparison_{int(rows[idx]['sample_id']):06d}.png"), grid)
        evidence(f"qualitative/comparison_{int(rows[idx]['sample_id']):06d}.png", "qualitative", "synchronized method comparison", "results/temporal_refinement_evaluation_l736_v1/summary.csv", f"sample_id={rows[idx]['sample_id']}")
    # Videos over all 126 frames, resized compact.
    videos = {
        "01_rgb_raw_s2m2": lambda i: np.hstack([make_tile(rgbs[i], f"RGB {i:03d}"), make_tile(raw[i], "raw S2M2-L", vmax=vmax)]),
        "02_main_temporal_raw_v2e30_sav": lambda i: np.hstack([make_tile(raw[i], f"raw {i:03d}", vmax=vmax), make_tile(methods["convgru_v2_epoch30"]["preds"][i], "ConvGRU V2 e30", vmax=vmax), make_tile(sav[i], "StereoAnyVideo", vmax=vmax)]),
        "03_learned_refiners": lambda i: np.hstack([make_tile(methods["tiny_unet"]["preds"][i], f"Tiny U-Net {i:03d}", vmax=vmax), make_tile(methods["convgru_v1"]["preds"][i], "ConvGRU V1", vmax=vmax), make_tile(methods["convgru_v2_epoch30"]["preds"][i], "ConvGRU V2 e30", vmax=vmax)]),
        "04_residual_temporal_change": lambda i: np.hstack([make_tile(rgbs[i], f"RGB {i:03d}"), make_tile(methods["convgru_v2_epoch30"]["preds"][i], "V2 e30 disp", vmax=vmax), make_tile(np.abs(methods["convgru_v2_epoch30"]["preds"][i] - raw[i]), "residual", vmax=5.0, cmap=cv2.COLORMAP_MAGMA), make_tile(np.zeros_like(raw[i]) if i == 0 else np.abs(methods["convgru_v2_epoch30"]["preds"][i] - methods["convgru_v2_epoch30"]["preds"][i-1]), "temporal diff", vmax=5.0, cmap=cv2.COLORMAP_MAGMA)]),
        "05_classical_baselines": lambda i: np.hstack([make_tile(raw[i], f"raw {i:03d}", vmax=vmax), make_tile(ema[i], "EMA a=0.7", vmax=vmax), make_tile(med5[i], "median5 non-causal", vmax=vmax), make_tile(methods["convgru_v2_epoch30"]["preds"][i], "ConvGRU V2 e30", vmax=vmax)]),
    }
    # 96 frames at 12 fps gives 8 seconds.
    frame_ids = list(range(0, min(len(rows), 96)))
    for name, fn in videos.items():
        save_video_and_gif(name, [fn(i) for i in frame_ids], fps=12)


def build_docs():
    summary = read_csv(UNIFIED / "summary.csv")
    def get(name):
        return next((r for r in summary if r["method"] == name), {})
    raw, sav = get("raw_s2m2_l736"), get("stereoanyvideo")
    v2e30 = next((r for r in read_csv(UNIFIED / "checkpoint_metrics.csv") if r["method"] == "convgru_v2_scheduled:epoch_0030"), {})
    presentation_summary = f"""# ARGOS Progress Presentation Summary

## Reliable headline

ARGOS now has a reproducible surgical stereo evaluation stack, SCARED temporal cache, S2M2/StereoAnyVideo cached predictions, and two learned temporal refinement families.

## Most presentation-worthy result

Unified full-frame evaluation on `test_dataset_9_keyframe_3` has no GT, so it measures temporal behavior rather than geometry.

- Raw S2M2-L temporal diff: `{safe_float(raw.get('temporal_diff')):.4f}`
- StereoAnyVideo temporal diff: `{safe_float(sav.get('temporal_diff')):.4f}`
- ConvGRU V2 epoch 30 temporal diff: `{safe_float(v2e30.get('temporal_diff')):.4f}`
- ConvGRU V2 epoch 30 teacher-delta MAE: `{safe_float(v2e30.get('teacher_delta_mae')):.4f}`
- ConvGRU V2 epoch 30 refined-to-backbone MAE: `{safe_float(v2e30.get('refined_to_backbone_mae')):.4f}`

## Mandatory caveat

The current 126-frame full-frame validation sequence has `has_gt=False`. Temporal smoothness does not prove geometric correctness.
"""
    (OUT / "presentation_summary.md").write_text(presentation_summary)
    slides = []
    titles = [
        "ARGOS motivation and surgical stereo problem",
        "Why surgery-like stereo is hard",
        "Datasets and current data pipeline",
        "Repository-wide SOTA exploration",
        "Frame-based S2M2 benchmark",
        "Why frame-wise predictions flicker",
        "StereoAnyVideo as temporal teacher",
        "Teacher-student residual refinement",
        "Tiny U-Net prototype",
        "Causal ConvGRU architecture",
        "Conservative vs scheduled training",
        "Unified full-frame evaluation",
        "Causal ConvGRU vs classical smoothing",
        "Limitations",
        "Next steps",
    ]
    for i, title in enumerate(titles, 1):
        slides.append(
            f"""## Slide {i}: {title}

**Core message:** {title}.

**Recommended asset:** see `tables/`, `plots/`, `diagrams/`, or `videos/` matching this topic.

**Bullets:**
- Use traceable numbers only.
- Distinguish crop-training metrics from full-frame validation.
- Keep causal/non-causal labels visible.

**Speaker notes:** Explain what evidence exists and what remains unproven.

**Important caveat:** Current unified full-frame validation lacks GT; do not claim geometric improvement there.
"""
        )
    (OUT / "slide_outline.md").write_text("\n".join(slides))
    readme = f"""# ARGOS Presentation Assets

Generated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`

This directory contains slide-ready tables, plots, diagrams, qualitative figures, videos, GIF previews, and source data for the internal ARGOS research presentation.

Important: every geometric GT metric for the unified full-frame validation is `NaN` because the evaluated long-cache rows have `has_gt=False`.

Recommended opening assets:

- `tables/model_audit_slide_ready.csv`
- `tables/datasets_slide_ready.csv`
- `tables/unified_fullframe_evaluation_slide_ready.csv`
- `diagrams/convgru_refinement.png`
- `plots/checkpoint_evolution_convgru_v2.png`
- `videos/02_main_temporal_raw_v2e30_sav.mp4`
"""
    (OUT / "README.md").write_text(readme)


def main():
    ensure_dirs()
    build_model_audit()
    build_dataset_tables()
    table_subsets()
    build_diagrams()
    build_plots()
    build_qualitative_and_videos()
    build_docs()
    write_csv(OUT / "evidence_manifest.csv", evidence_rows, ["asset", "asset_type", "claim_or_metric", "source_file", "source_row_or_checkpoint", "evaluation_protocol", "notes"])
    print(json.dumps({"out_dir": str(OUT), "evidence_items": len(evidence_rows)}, indent=2))


if __name__ == "__main__":
    main()
