#!/usr/bin/env python3
"""Build a traceable SERV-CT frame-stereo benchmark package from local evidence.

This script intentionally does not retrain models and does not invent metrics.
It aggregates existing per-frame SERV-CT metric files when they contain the
common holdout samples Experiment_2 frames 009-016.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path.cwd()
OUT = Path("results/servct_unified_frame_benchmark_v1")
DATASET = Path("dataset/SERVCT/argos/servct_argos")
STEREO_ROOT = ROOT.parent / "stereo"
HOLDOUT_FRAMES = set(range(9, 17))
NAN = float("nan")


@dataclass
class MethodEvidence:
    method: str
    model_family: str
    checkpoint: str
    source_dir: Path
    metrics_path: Path
    summary_path: Path | None
    input_resolution: str = "native/legacy_eval"
    precision: str = ""
    notes: str = ""


METHOD_HINTS = [
    ("S2M2-S all-surgical finetuned", "S2M2", "S all-surgical checkpoint", "stereo/s2m2/output_servct_eval_s2m2_S_all_surgical_checkpoint/metrics.csv", "fine-tuned on all surgical SERV-CT/SCARED evidence; not an honest held-out training protocol"),
    ("S2M2-S honest finetuned", "S2M2", "S honest holdout checkpoint", "stereo/s2m2/output_servct_finetune_honest_S/eval/metrics.csv", "trained on Experiment_1 only; evaluated on Experiment_2 holdout"),
    ("S2M2-S pretrained", "S2M2", "S pretrained", "stereo/s2m2/output_servct_eval_s2m2_S/metrics.csv", "zero-shot/pretrained baseline"),
    ("S2M2-M pretrained", "S2M2", "M pretrained", "stereo/s2m2/output_servct_eval_s2m2_M/metrics.csv", "zero-shot/pretrained baseline; local repo names this variant M"),
    ("S2M2-L pretrained", "S2M2", "L pretrained", "stereo/s2m2/output_servct_eval_s2m2_L/metrics.csv", "zero-shot/pretrained baseline; run for this unified audit"),
    ("S2M2-XL pretrained", "S2M2", "XL pretrained", "stereo/s2m2/output_servct_eval_s2m2_XL/metrics.csv", "zero-shot/pretrained baseline; run for this unified audit, collapsed on SERV-CT"),
    ("Fast-FoundationStereo ONNX", "Fast-FoundationStereo", "ONNX foundation baseline", "stereo/Fast-FoundationStereo/output_servct_eval/metrics.csv", "legacy evaluator uses fewer valid pixels; compare with caution"),
    ("SGBM", "Classical", "OpenCV SGBM", "stereo/Fast-FoundationStereo/output_servct_eval_sgbm/metrics.csv", "classical baseline; legacy evaluator uses model-valid pixels"),
    ("StereoAnywhere ViT-L", "StereoAnywhere", "ViT-L", "stereo/stereoanywhere/output_servct_eval_stereoanywhere_vitl/metrics.csv", "zero-shot local SERV-CT run"),
    ("StereoAnywhere", "StereoAnywhere", "default", "stereo/stereoanywhere/output_servct_eval_stereoanywhere/metrics.csv", "zero-shot local SERV-CT run"),
    ("RAFT-Stereo RVC", "RAFT-Stereo", "RVC", "stereo/RAFT-Stereo/output_servct_eval_raft_rvc/metrics.csv", "zero-shot local SERV-CT run"),
    ("RAFT-Stereo Middlebury", "RAFT-Stereo", "Middlebury", "stereo/RAFT-Stereo/output_servct_eval_raft_middlebury/metrics.csv", "zero-shot local SERV-CT run"),
    ("RAFT-Stereo SceneFlow", "RAFT-Stereo", "SceneFlow", "stereo/RAFT-Stereo/output_servct_eval_raft_sceneflow/metrics.csv", "zero-shot SERV-CT run launched during unified audit"),
    ("RAFT-Stereo ETH3D", "RAFT-Stereo", "ETH3D", "stereo/RAFT-Stereo/output_servct_eval_raft_eth3d/metrics.csv", "zero-shot SERV-CT run launched during unified audit"),
    ("CREStereo", "CREStereo", "local checkpoint", "stereo/stereo_matching_crestereo/output_servct_eval_crestereo/metrics.csv", "zero-shot local SERV-CT run"),
    ("DEFOM-Stereo ViT-L Middlebury", "DEFOM-Stereo", "ViT-L Middlebury", "stereo/DEFOM-Stereo/output_servct_eval_defom_vitl_middlebury/metrics.csv", "zero-shot local SERV-CT run"),
    ("DEFOM-Stereo ViT-L ETH3D", "DEFOM-Stereo", "ViT-L ETH3D", "stereo/DEFOM-Stereo/output_servct_eval_defom_vitl_eth3d/metrics.csv", "zero-shot SERV-CT run launched during unified audit"),
    ("DEFOM-Stereo ViT-L KITTI", "DEFOM-Stereo", "ViT-L KITTI", "stereo/DEFOM-Stereo/output_servct_eval_defom_vitl_kitti/metrics.csv", "zero-shot SERV-CT run launched during unified audit"),
    ("DEFOM-Stereo ViT-S RVC", "DEFOM-Stereo", "ViT-S RVC", "stereo/DEFOM-Stereo/output_servct_eval_defom_vits_rvc/metrics.csv", "zero-shot local SERV-CT run"),
    ("DEFOM-Stereo ViT-S SceneFlow", "DEFOM-Stereo", "ViT-S SceneFlow", "stereo/DEFOM-Stereo/output_servct_eval_defom_vits_sceneflow/metrics.csv", "zero-shot local SERV-CT run"),
    ("DEFOM-Stereo ViT-L SceneFlow", "DEFOM-Stereo", "ViT-L SceneFlow", "stereo/DEFOM-Stereo/output_servct_eval_defom_vitl_sceneflow/metrics.csv", "numerically unstable depth in existing summary; retained as evidence"),
    ("MonSter++ MixAll i16", "MonSter++", "MixAll i16", "stereo/MonSter-plusplus/MonSter++/output_servct_eval_monsterpp_mixall_i16/metrics.csv", "zero-shot local SERV-CT run"),
    ("MonSter++ MixAll", "MonSter++", "MixAll", "stereo/MonSter-plusplus/MonSter++/output_servct_eval_monsterpp_mixall/metrics.csv", "zero-shot local SERV-CT run"),
    ("RT-MonSter++ zero-shot", "RT-MonSter++", "zero-shot", "stereo/MonSter-plusplus/RT-MonSter++/output_servct_eval_rtmonster_zeroshot/metrics.csv", "real-time MonSter++ local SERV-CT run"),
]


REPO_AUDIT = [
    ("S2M2-S", "S2M2", "frame", "stereo/s2m2", "existing_comparable_SERVCT_result"),
    ("S2M2-L", "S2M2", "frame", "stereo/s2m2", "not_evaluated"),
    ("S2M2-XL", "S2M2", "frame", "stereo/s2m2", "not_evaluated"),
    ("Fast-FoundationStereo", "foundation stereo", "frame", "stereo/Fast-FoundationStereo", "existing_comparable_SERVCT_result"),
    ("FoundationStereo", "foundation stereo", "frame", "stereo/FoundationStereo", "not_evaluated"),
    ("StereoAnywhere", "foundation stereo", "frame", "stereo/stereoanywhere", "existing_comparable_SERVCT_result"),
    ("RAFT-Stereo", "RAFT stereo", "frame", "stereo/RAFT-Stereo", "existing_comparable_SERVCT_result"),
    ("CREStereo", "stereo matching", "frame", "stereo/stereo_matching_crestereo", "existing_comparable_SERVCT_result"),
    ("DEFOM-Stereo", "foundation stereo", "frame", "stereo/DEFOM-Stereo", "existing_comparable_SERVCT_result"),
    ("MonSter++", "stereo matching", "frame", "stereo/MonSter-plusplus/MonSter++", "existing_comparable_SERVCT_result"),
    ("RT-MonSter++", "stereo matching", "frame", "stereo/MonSter-plusplus/RT-MonSter++", "existing_comparable_SERVCT_result"),
    ("IGEV++", "stereo matching", "frame", "stereo/IGEV-plusplus", "requires_minor_adapter"),
    ("Selective-Stereo", "stereo matching", "frame", "stereo/Selective-Stereo", "requires_minor_adapter"),
    ("PPMStereo", "video/dynamic stereo", "video", "stereo/PPMStereo", "not_frame_based"),
    ("StereoAnyVideo", "video stereo", "video", "stereo/stereoanyvideo", "not_frame_based"),
    ("TemporalStereo", "video stereo", "video", "stereo/TemporalStereo", "not_frame_based"),
    ("TC-Stereo", "video stereo", "video", "stereo/Temporally-Consistent-Stereo-Matching", "not_frame_based"),
]

ATTEMPTED_FAILURES = [
    {
        "method": "RAFT-Stereo Realtime checkpoint",
        "failure_stage": "checkpoint_load",
        "error_summary": "Checkpoint state_dict does not match the RAFTStereo architecture configured by the existing SERV-CT adapter.",
        "attempted_fix": "Retried with local Fast-FoundationStereo conda environment to fix missing opt_einsum; environment fixed, architecture mismatch remained.",
        "final_status": "runtime_failure",
    }
]


def ensure_dirs() -> None:
    for sub in ["plots", "qualitative", "predictions"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def existing_path(path: str | Path) -> Path:
    """Resolve paths that may live inside ARGOS or the sibling stereo folder."""
    p = Path(path)
    if p.exists():
        return p
    if p.parts and p.parts[0] == "stereo":
        sibling = ROOT.parent / p
        if sibling.exists():
            return sibling
    return p


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for k in row:
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def dataset_manifest() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_dir in sorted(DATASET.glob("*")):
        if not split_dir.is_dir():
            continue
        for sample in sorted(split_dir.glob("Experiment_*")):
            if not sample.is_dir():
                continue
            left = sample / "left.png"
            right = sample / "right.png"
            disp = sample / "disp_gt.npy"
            depth = sample / "depth_gt_mm.npy"
            mask = sample / "valid_mask.npy"
            calib = load_json(sample / "calib.json")
            meta = load_json(sample / "metadata.json")
            image_size = ""
            if left.exists():
                with Image.open(left) as im:
                    image_size = f"{im.width}x{im.height}"
            valid_ratio = NAN
            if mask.exists():
                arr = np.load(mask)
                valid_ratio = float(np.mean(arr > 0))
            frame = meta.get("frame") or meta.get("frame_id") or sample.name.split("_")[-1]
            rows.append(
                {
                    "split": split_dir.name,
                    "sample_id": sample.name,
                    "frame": frame,
                    "left_exists": left.exists(),
                    "right_exists": right.exists(),
                    "disp_gt_exists": disp.exists(),
                    "depth_gt_mm_exists": depth.exists(),
                    "valid_mask_exists": mask.exists(),
                    "image_size": image_size,
                    "valid_mask_ratio": valid_ratio,
                    "fx_px": calib.get("fx") or calib.get("focal_length_px") or calib.get("focal_length"),
                    "baseline_mm": calib.get("baseline_mm") or calib.get("baseline"),
                    "calib_keys": ",".join(sorted(calib.keys())),
                    "source_dir": rel(sample),
                }
            )
    return rows


def discover_evidence() -> tuple[list[MethodEvidence], list[dict[str, Any]]]:
    evidences: list[MethodEvidence] = []
    audit_rows: list[dict[str, Any]] = []
    for method, family, checkpoint, metrics, notes in METHOD_HINTS:
        mp = existing_path(metrics)
        exists = mp.exists()
        summary = mp.with_name("summary.json")
        if not summary.exists() and mp.parent.name == "eval":
            summary = mp.parent / "summary.json"
        if exists:
            evidences.append(MethodEvidence(method, family, checkpoint, mp.parent, mp, summary if summary.exists() else None, notes=notes))
        audit_rows.append(
            {
                "method": method,
                "model_family": family,
                "repository_or_source": rel(mp.parent),
                "checkpoint": checkpoint,
                "status": "existing_comparable_SERVCT_result" if exists else "not_evaluated",
                "evidence_source": rel(mp) if exists else "",
                "notes": notes if exists else "Expected metrics file not found.",
            }
        )
    return evidences, audit_rows


def normalize_frame_id(row: pd.Series) -> int | None:
    try:
        return int(row.get("frame"))
    except Exception:
        return None


def summarize_method(ev: MethodEvidence) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    df = pd.read_csv(ev.metrics_path)
    df["frame_int"] = df.apply(normalize_frame_id, axis=1)
    holdout = df[df["frame_int"].isin(HOLDOUT_FRAMES)].copy()
    status = "existing_comparable_SERVCT_result" if len(holdout) == 8 else "not_evaluated"
    notes = ev.notes
    if len(holdout) != 8:
        notes += f"; holdout rows found={len(holdout)} not 8"

    per_frame = []
    for _, r in holdout.iterrows():
        per_frame.append(
            {
                "dataset": "SERV-CT honest_test",
                "method": ev.method,
                "model_family": ev.model_family,
                "checkpoint": ev.checkpoint,
                "sample_id": f"Experiment_2_{int(r['frame_int']):03d}",
                "frame": int(r["frame_int"]),
                "main_evaluation_mask": "SERV-CT valid mask; legacy metrics may also exclude invalid model predictions",
                "valid_px": r.get("valid_px", NAN),
                "disparity_mae_px": r.get("mae_px", NAN),
                "disparity_rmse_px": r.get("rmse_px", NAN),
                "bad_1px_percent": r.get("bad1_pct", NAN),
                "bad_2px_percent": r.get("bad2_pct", NAN),
                "bad_3px_percent": NAN,
                "bad_5px_percent": r.get("bad5_pct", NAN),
                "depth_mae_mm": r.get("depth_mae_mm", NAN),
                "depth_rmse_mm": r.get("depth_rmse_mm", NAN),
                "bad_1mm_percent": r.get("depth_bad1mm_pct", NAN),
                "bad_2mm_percent": r.get("depth_bad2mm_pct", NAN),
                "bad_4mm_percent": NAN,
                "bad_5mm_percent": r.get("depth_bad5mm_pct", NAN),
                "evidence_source": rel(ev.metrics_path),
            }
        )

    def mean(col: str) -> float:
        return float(pd.to_numeric(holdout[col], errors="coerce").mean()) if col in holdout else NAN

    def med(col: str) -> float:
        return float(pd.to_numeric(holdout[col], errors="coerce").median()) if col in holdout else NAN

    def p95(col: str) -> float:
        return float(pd.to_numeric(holdout[col], errors="coerce").quantile(0.95)) if col in holdout else NAN

    full = {
        "dataset": "SERV-CT honest_test",
        "method": ev.method,
        "model_family": ev.model_family,
        "checkpoint": ev.checkpoint,
        "input_resolution": ev.input_resolution,
        "evaluation_resolution": "original SERV-CT GT resolution",
        "number_of_frames": len(holdout),
        "main_evaluation_mask": "SERV-CT valid mask; legacy metrics may also exclude invalid model predictions",
        "disparity_mae_px": mean("mae_px"),
        "disparity_rmse_px": mean("rmse_px"),
        "disparity_median_error_px": med("mae_px"),
        "disparity_p95_error_px": p95("mae_px"),
        "bad_1px_percent": mean("bad1_pct"),
        "bad_2px_percent": mean("bad2_pct"),
        "bad_3px_percent": NAN,
        "depth_mae_mm": mean("depth_mae_mm"),
        "depth_rmse_mm": mean("depth_rmse_mm"),
        "depth_median_error_mm": med("depth_mae_mm"),
        "depth_p95_error_mm": p95("depth_mae_mm"),
        "bad_1mm_percent": mean("depth_bad1mm_pct"),
        "bad_2mm_percent": mean("depth_bad2mm_pct"),
        "bad_4mm_percent": NAN,
        "runtime_mean_ms": NAN,
        "runtime_median_ms": NAN,
        "runtime_std_ms": NAN,
        "fps": NAN,
        "peak_vram_mb": NAN,
        "precision": ev.precision,
        "status": status,
        "notes": notes + "; runtime/VRAM not present in existing SERV-CT per-frame metrics; median/p95 are across-frame summaries, not pixel-level p50/p95.",
        "evidence_source": rel(ev.metrics_path),
    }
    audit = {
        "result_path": rel(ev.metrics_path),
        "method": ev.method,
        "frames_total": int(len(df)),
        "holdout_frames_found": int(len(holdout)),
        "has_required_holdout_frames": len(holdout) == 8,
        "has_runtime": "runtime_ms" in df.columns,
        "has_pixel_median_or_p95": False,
        "reused": len(holdout) == 8,
        "compatibility_notes": full["notes"],
    }
    return per_frame, full, audit


def build_region_summary(per_frame: pd.DataFrame) -> pd.DataFrame:
    if per_frame.empty:
        return pd.DataFrame()
    rows = []
    for method, grp in per_frame.groupby("method"):
        rows.append(
            {
                "dataset": "SERV-CT honest_test",
                "method": method,
                "region": "all_valid",
                "frames": len(grp),
                "depth_mae_mm": pd.to_numeric(grp["depth_mae_mm"], errors="coerce").mean(),
                "disparity_mae_px": pd.to_numeric(grp["disparity_mae_px"], errors="coerce").mean(),
                "notes": "No occlusion/edge region masks were found in the ARGOS SERV-CT converted samples; only all_valid is reported.",
            }
        )
    return pd.DataFrame(rows)


def model_audit_rows(existing_audit: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    evidence_map = {
        "S2M2-S": [r for r in existing_audit if r["method"].startswith("S2M2-S")],
        "S2M2-L": [r for r in existing_audit if r["method"].startswith("S2M2-L")],
        "S2M2-XL": [r for r in existing_audit if r["method"].startswith("S2M2-XL")],
        "Fast-FoundationStereo": [r for r in existing_audit if r["method"].startswith("Fast-FoundationStereo")],
        "StereoAnywhere": [r for r in existing_audit if r["method"].startswith("StereoAnywhere")],
        "RAFT-Stereo": [r for r in existing_audit if r["method"].startswith("RAFT-Stereo")],
        "CREStereo": [r for r in existing_audit if r["method"].startswith("CREStereo")],
        "DEFOM-Stereo": [r for r in existing_audit if r["method"].startswith("DEFOM-Stereo")],
        "MonSter++": [r for r in existing_audit if r["method"].startswith("MonSter++")],
        "RT-MonSter++": [r for r in existing_audit if r["method"].startswith("RT-MonSter++")],
    }
    for model, family, kind, repo, status in REPO_AUDIT:
        found = existing_path(repo).exists()
        evidence = ""
        notes = ""
        matching = evidence_map.get(model, [])
        if matching:
            status = "existing_comparable_SERVCT_result"
            evidence = "; ".join(r["evidence_source"] for r in matching if r.get("evidence_source"))
            notes = f"{len(matching)} local SERV-CT metric file(s) found."
        if not found:
            status = "not_evaluated"
            notes = "Repository not found at expected local path."
        if model == "FoundationStereo" and found and not matching:
            status = "missing_checkpoint"
            notes = "Repository present, but pretrained_models directory contains no usable .pth/.pt checkpoint."
        if model in {"IGEV++", "Selective-Stereo"} and found and not matching:
            status = "missing_checkpoint"
            notes = "Repository present, but no local .pth/.pt/.ckpt checkpoint was found; adapter/demo exists but cannot be run fairly."
        if kind == "video":
            status = "not_frame_based"
            notes = "Excluded from raw frame-by-frame SERV-CT table."
        rows.append(
            {
                "method": model,
                "model_family": family,
                "frame_or_video": kind,
                "repository_or_source": repo,
                "status": status,
                "evidence_source": evidence,
                "notes": notes,
            }
        )
    return rows


def failures_from_audit(audit: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(ATTEMPTED_FAILURES)
    for row in audit:
        if row["status"] not in {"existing_comparable_SERVCT_result", "successfully_evaluated"}:
            rows.append(
                {
                    "method": row["method"],
                    "failure_stage": "audit",
                    "error_summary": row["notes"],
                    "attempted_fix": "No inference launched in this aggregation pass; use existing adapters/checkpoints first.",
                    "final_status": row["status"],
                }
            )
    return rows


def save_markdown_table(df: pd.DataFrame, path: Path, max_rows: int | None = None) -> None:
    out = df if max_rows is None else df.head(max_rows)
    cols = list(out.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in out.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append("" if math.isnan(v) else f"{v:.4g}")
            else:
                vals.append(str(v).replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n")


def plot_scatter(df: pd.DataFrame, x: str, y: str, path: Path, title: str) -> None:
    clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[x, y])
    fig, ax = plt.subplots(figsize=(9, 6), dpi=160)
    if clean.empty:
        ax.text(0.5, 0.5, f"No traceable {x} data in local SERV-CT evidence", ha="center", va="center", wrap=True)
        ax.set_axis_off()
    else:
        ax.scatter(clean[x], clean[y], s=60)
        for _, r in clean.iterrows():
            ax.annotate(str(r["method"])[:24], (r[x], r[y]), fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(True, alpha=0.25)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_distribution(per_frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6), dpi=160)
    if per_frame.empty:
        ax.text(0.5, 0.5, "No per-frame metrics found", ha="center", va="center")
        ax.set_axis_off()
    else:
        ordered = (
            per_frame.groupby("method")["depth_mae_mm"]
            .mean()
            .sort_values()
            .index.tolist()
        )
        data = [per_frame.loc[per_frame["method"] == m, "depth_mae_mm"].astype(float).values for m in ordered]
        ax.boxplot(data, labels=[m[:22] for m in ordered], vert=False, showfliers=True)
        ax.set_xlabel("Depth MAE [mm], per-frame")
        ax.grid(True, axis="x", alpha=0.25)
    ax.set_title("SERV-CT holdout per-frame depth MAE distribution")
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_ranking(df: pd.DataFrame, path: Path) -> None:
    clean = df.dropna(subset=["depth_mae_mm"]).sort_values("depth_mae_mm").copy()
    fig, ax = plt.subplots(figsize=(10, 7), dpi=160)
    if clean.empty:
        ax.text(0.5, 0.5, "No depth MAE data", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.barh(clean["method"], clean["depth_mae_mm"], color="#4676b8")
        ax.invert_yaxis()
        ax.set_xlabel("Depth MAE [mm], lower is better")
        ax.grid(True, axis="x", alpha=0.25)
    ax.set_title("SERV-CT holdout method ranking")
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def pareto_plot(df: pd.DataFrame, path: Path) -> None:
    # Runtime is mostly unavailable in legacy evidence. Fall back to disparity MAE vs depth MAE
    # so the file is useful while clearly labelled.
    fig, ax = plt.subplots(figsize=(9, 6), dpi=160)
    clean = df.dropna(subset=["disparity_mae_px", "depth_mae_mm"])
    ax.scatter(clean["disparity_mae_px"], clean["depth_mae_mm"], s=60)
    for _, r in clean.iterrows():
        ax.annotate(str(r["method"])[:24], (r["disparity_mae_px"], r["depth_mae_mm"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Disparity MAE [px]")
    ax.set_ylabel("Depth MAE [mm]")
    ax.grid(True, alpha=0.25)
    ax.set_title("Accuracy Pareto proxy: depth MAE vs disparity MAE (runtime unavailable)")
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def copy_qualitative(evidences: list[MethodEvidence], summary_df: pd.DataFrame) -> list[dict[str, Any]]:
    copied = []
    selected = summary_df.dropna(subset=["depth_mae_mm"]).sort_values("depth_mae_mm").head(8)["method"].tolist()
    for ev in evidences:
        if ev.method not in selected:
            continue
        montage = ev.source_dir / "montage_left_pred_gt_err.png"
        if montage.exists():
            dst = OUT / "qualitative" / f"{ev.method.lower().replace(' ', '_').replace('+','plus').replace('/','_')}_montage.png"
            shutil.copy2(montage, dst)
            copied.append({"method": ev.method, "asset": rel(dst), "source": rel(montage)})
    return copied


def main() -> None:
    ensure_dirs()
    manifest = dataset_manifest()
    write_csv(OUT / "servct_dataset_manifest.csv", manifest)

    evidences, existing_audit_seed = discover_evidence()
    per_frame_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    result_audit_rows: list[dict[str, Any]] = []
    for ev in evidences:
        pf, full, audit = summarize_method(ev)
        per_frame_rows.extend(pf)
        summary_rows.append(full)
        result_audit_rows.append(audit)

    per_frame = pd.DataFrame(per_frame_rows)
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(["depth_mae_mm", "disparity_mae_px"], na_position="last")

    # Full required benchmark table.
    full_cols = [
        "dataset", "method", "model_family", "checkpoint", "input_resolution", "evaluation_resolution",
        "number_of_frames", "main_evaluation_mask", "disparity_mae_px", "disparity_rmse_px",
        "disparity_median_error_px", "disparity_p95_error_px", "bad_1px_percent", "bad_2px_percent",
        "bad_3px_percent", "depth_mae_mm", "depth_rmse_mm", "depth_median_error_mm", "depth_p95_error_mm",
        "bad_1mm_percent", "bad_2mm_percent", "bad_4mm_percent", "runtime_mean_ms", "runtime_median_ms",
        "runtime_std_ms", "fps", "peak_vram_mb", "precision", "status", "notes", "evidence_source",
    ]
    summary.to_csv(OUT / "servct_benchmark_full.csv", index=False, columns=full_cols)
    summary.to_csv(OUT / "servct_per_method_summary.csv", index=False)
    per_frame.to_csv(OUT / "servct_per_frame_metrics.csv", index=False)

    region = build_region_summary(per_frame)
    region.to_csv(OUT / "servct_per_region_summary.csv", index=False)

    runtime = summary[["method", "runtime_mean_ms", "runtime_median_ms", "runtime_std_ms", "fps", "peak_vram_mb", "precision", "evidence_source"]].copy()
    runtime["notes"] = "Runtime/VRAM absent from legacy SERV-CT metric files; left empty rather than mixing unknown hardware numbers."
    runtime.to_csv(OUT / "servct_runtime_summary.csv", index=False)

    slide = summary.copy()
    slide["Method"] = slide["method"]
    slide["Input resolution"] = slide["input_resolution"]
    slide["Depth MAE [mm]"] = slide["depth_mae_mm"]
    slide["Bad-2 mm [%]"] = slide["bad_2mm_percent"]
    slide["Disparity MAE [px]"] = slide["disparity_mae_px"]
    slide["Runtime [ms]"] = slide["runtime_mean_ms"]
    slide["FPS"] = slide["fps"]
    slide["VRAM [GB]"] = slide["peak_vram_mb"] / 1024.0
    slide_cols = ["Method", "Input resolution", "Depth MAE [mm]", "Bad-2 mm [%]", "Disparity MAE [px]", "Runtime [ms]", "FPS", "VRAM [GB]"]
    slide[slide_cols].to_csv(OUT / "servct_benchmark_slide_ready.csv", index=False)
    save_markdown_table(slide[slide_cols], OUT / "servct_benchmark_slide_ready.md")

    model_audit = model_audit_rows(existing_audit_seed)
    write_csv(OUT / "servct_model_audit.csv", model_audit)
    write_csv(OUT / "servct_model_failures.csv", failures_from_audit(model_audit))
    write_csv(OUT / "servct_existing_results_audit.csv", result_audit_rows)

    plot_scatter(summary, "runtime_mean_ms", "depth_mae_mm", OUT / "plots/depth_mae_vs_runtime.png", "SERV-CT depth MAE vs runtime")
    plot_scatter(summary, "peak_vram_mb", "depth_mae_mm", OUT / "plots/depth_mae_vs_vram.png", "SERV-CT depth MAE vs VRAM")
    plot_scatter(summary, "runtime_mean_ms", "disparity_mae_px", OUT / "plots/disparity_mae_vs_runtime.png", "SERV-CT disparity MAE vs runtime")
    plot_distribution(per_frame, OUT / "plots/per_frame_depth_mae_distribution.png")
    plot_ranking(summary, OUT / "plots/method_ranking_depth_mae.png")
    pareto_plot(summary, OUT / "plots/accuracy_runtime_pareto_front.png")

    qualitative = copy_qualitative(evidences, summary)

    evidence_rows = []
    for _, row in summary.iterrows():
        evidence_rows.append(
            {
                "asset": "servct_benchmark_full.csv",
                "asset_type": "table",
                "claim_or_metric": f"{row['method']} holdout depth/disparity summary",
                "source_file": row["evidence_source"],
                "source_row_or_checkpoint": "Experiment_2 frames 009-016",
                "evaluation_protocol": "SERV-CT honest_test holdout, legacy per-frame metrics aggregated by this script",
                "notes": row["notes"],
            }
        )
    for q in qualitative:
        evidence_rows.append(
            {
                "asset": q["asset"],
                "asset_type": "qualitative",
                "claim_or_metric": f"{q['method']} qualitative montage",
                "source_file": q["source"],
                "source_row_or_checkpoint": "legacy montage",
                "evaluation_protocol": "Existing SERV-CT montage copied, not regenerated from predictions",
                "notes": "Prediction arrays were not available, so best/median/worst grids could not be regenerated.",
            }
        )
    write_csv(OUT / "evidence_manifest.csv", evidence_rows)

    payload = {
        "output_dir": rel(OUT),
        "dataset_samples_found": len(manifest),
        "holdout_samples_evaluated": 8,
        "methods_with_holdout_metrics": int((summary["number_of_frames"] == 8).sum()) if not summary.empty else 0,
        "limitations": [
            "Aggregated from existing per-frame SERV-CT metric CSVs; no new inference was launched.",
            "Prediction arrays are not present, so pixel-level median/p95, common-mask recomputation, and new qualitative best/median/worst grids could not be generated.",
            "Runtime and peak VRAM are absent from existing SERV-CT metric files; left as NaN.",
            "Most methods share the 8 Experiment_2 holdout frames, but some legacy evaluators use model-valid pixels in addition to the SERV-CT valid mask.",
        ],
        "benchmark": summary.replace({np.nan: None}).to_dict(orient="records"),
        "qualitative_assets": qualitative,
    }
    (OUT / "servct_benchmark.json").write_text(json.dumps(payload, indent=2))

    readme = f"""# SERV-CT Unified Frame Benchmark v1

This package aggregates traceable local SERV-CT evidence for raw frame-based stereo methods.

Main protocol:
- Dataset: `dataset/SERVCT/argos/servct_argos/honest_test`
- Samples: 8 holdout frames, `Experiment_2_009` through `Experiment_2_016`
- Evaluation resolution: original SERV-CT GT resolution
- Main mask: SERV-CT valid mask as used by the existing evaluators. Some legacy runs also exclude invalid/non-positive model predictions; see `valid_px` and notes.
- No retraining and no temporal refiners.

Important limitation:
Prediction arrays were not saved for these legacy runs, so this package does not recompute a stricter common mask or new pixel-level p50/p95. Runtime/VRAM are also not present in the SERV-CT metric files and remain empty.

Files:
- `servct_benchmark_full.csv`: complete benchmark table requested for the presentation.
- `servct_benchmark_slide_ready.csv` and `.md`: compact slide table.
- `servct_per_frame_metrics.csv`: holdout per-frame evidence.
- `servct_model_audit.csv`: every discovered/requested method and status.
- `servct_existing_results_audit.csv`: compatibility audit for reused metric files.
- `plots/`: presentation plots, with runtime plots explicitly marking missing runtime evidence.
- `qualitative/`: copied legacy montages for top methods where available.
"""
    (OUT / "README.md").write_text(readme)

    print(f"Wrote {rel(OUT)}")
    print(f"SERV-CT samples found: {len(manifest)}")
    print(f"Holdout methods summarized: {len(summary)}")
    if not summary.empty:
        print("Top slide rows:")
        print(slide[slide_cols].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
