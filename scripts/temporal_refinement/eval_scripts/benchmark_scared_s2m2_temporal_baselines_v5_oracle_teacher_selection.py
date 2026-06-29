#!/usr/bin/env python3
"""Oracle teacher-selection benchmark for artifact-aware temporal refinement."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = ROOT / "scripts" / "temporal_refinement" / "eval_scripts"
LIB_DIR = ROOT / "scripts" / "temporal_refinement" / "lib"
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(LIB_DIR))

import benchmark_scared_s2m2_temporal_baselines as base  # noqa: E402
import benchmark_scared_s2m2_temporal_baselines_v3_raft_small as v3  # noqa: E402
import benchmark_scared_s2m2_temporal_baselines_v4_artifact_metrics as v4  # noqa: E402
from artifact_metrics import (  # noqa: E402
    finite_mean,
    forward_backward_occlusion_mask,
    gradient_magnitude,
    percentile_mask,
    warp_array,
)
from video_qualitative import colorize_scalar, make_board, write_mp4  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("results/03_temporal_refinement/scared_s2m2_temporal_baselines_v5_oracle_teacher_selection")
DEFAULT_V4_DIR = Path("results/03_temporal_refinement/scared_s2m2_temporal_baselines_v4_raft_small_artifact_metrics")

ORACLE_SELECTION_COLUMNS = [
    "method_id",
    "mean_selected_raw_pct",
    "mean_selected_fixed_pct",
    "mean_selected_raftsmall_pct",
    "mean_selected_fullraft_pct",
    "mean_selected_sav_pct",
    "artifact_safe_raft_pct",
    "edge_region_raw_selection_pct",
    "occlusion_region_raw_selection_pct",
    "stable_region_raft_selection_pct",
]

SELECTION_SUMMARY_COLUMNS = [
    "mean_selected_raw_pct",
    "mean_selected_fixed_pct",
    "mean_selected_raftsmall_pct",
    "mean_selected_fullraft_pct",
    "mean_selected_sav_pct",
    "artifact_safe_raft_pct",
    "edge_region_raw_selection_pct",
    "occlusion_region_raw_selection_pct",
    "stable_region_raft_selection_pct",
]


@dataclass
class SelectorResult:
    method: base.MethodRecord
    masks: list[dict[str, np.ndarray]]
    frame_selection_rows: list[dict[str, Any]]
    summary: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oracle teacher-selection benchmark for SCARED temporal refinement.")
    parser.add_argument("--sequence-dir", type=Path, default=base.DEFAULT_SEQUENCE_DIR)
    parser.add_argument("--s2m2-cache-dir", type=Path, default=base.DEFAULT_S2M2_CACHE)
    parser.add_argument("--sav-cache-dir", type=Path, default=base.DEFAULT_SAV_CACHE)
    parser.add_argument("--full-flow-cache-dir", type=Path, default=v3.DEFAULT_FULL_FLOW_CACHE)
    parser.add_argument("--raft-small-6-flow-cache-dir", type=Path, default=v3.DEFAULT_SMALL6_FLOW_CACHE)
    parser.add_argument("--raft-small-12-flow-cache-dir", type=Path, default=v3.DEFAULT_SMALL12_FLOW_CACHE)
    parser.add_argument("--previous-benchmark-dir", type=Path, default=v3.DEFAULT_PREVIOUS_BENCHMARK_DIR)
    parser.add_argument("--v4-results-dir", type=Path, default=DEFAULT_V4_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-valid-ratio", type=float, default=0.2)
    parser.add_argument("--overwrite", nargs="?", const=True, default=False, type=base.parse_bool)
    parser.add_argument("--warp-device", default="auto")
    parser.add_argument("--tau-fb", type=float, default=1.5)
    parser.add_argument("--ghosting-gt-threshold-px", type=float, default=10.0)
    parser.add_argument("--edge-degradation-threshold", type=float, default=0.75)
    return parser.parse_args()


def read_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    return np.array(Image.open(path).convert("RGB"))


def load_backward_flow(flow_cache_dir: Path, prev_id: str, cur_id: str) -> np.ndarray:
    return np.load(flow_cache_dir / "backward_flow" / f"{cur_id}_to_{prev_id}.npy", allow_pickle=False).astype(np.float32)


def safe_pct(num: int | float, den: int | float) -> float:
    den_f = float(den)
    return float(100.0 * float(num) / den_f) if den_f > 0 else float("nan")


def choose_min_error(candidates: list[tuple[str, np.ndarray]], gt: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    errors = np.stack([np.abs(pred.astype(np.float32) - gt.astype(np.float32)) for _name, pred in candidates], axis=0)
    errors[:, ~valid] = np.inf
    choice = np.argmin(errors, axis=0)
    output = np.zeros_like(gt, dtype=np.float32)
    masks: dict[str, np.ndarray] = {}
    for idx, (name, pred) in enumerate(candidates):
        mask = valid & (choice == idx)
        masks[name] = mask
        output[mask] = pred[mask]
    output[~valid] = candidates[0][1][~valid]
    return output, masks


def selection_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in SELECTION_SUMMARY_COLUMNS:
        vals = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    return out


def frame_selection_percentages(
    *,
    method_id: str,
    frame_id: str,
    valid: np.ndarray,
    masks: dict[str, np.ndarray],
    edge_mask: np.ndarray,
    motion_mask: np.ndarray,
    occlusion_mask: np.ndarray,
    artifact_safe_raft_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    valid_count = int(valid.sum())
    selected_raw = masks.get("raw", np.zeros_like(valid, dtype=bool))
    selected_fixed = masks.get("fixed", np.zeros_like(valid, dtype=bool))
    selected_raftsmall = masks.get("raftsmall", np.zeros_like(valid, dtype=bool))
    selected_fullraft = masks.get("fullraft", np.zeros_like(valid, dtype=bool))
    selected_sav = masks.get("sav", np.zeros_like(valid, dtype=bool))
    edge_region = valid & edge_mask
    occ_region = valid & (occlusion_mask | motion_mask)
    stable_region = valid & ~edge_mask & ~occlusion_mask & ~motion_mask
    safe_mask = artifact_safe_raft_mask if artifact_safe_raft_mask is not None else selected_raftsmall
    return {
        "method_id": method_id,
        "frame_id": frame_id,
        "mean_selected_raw_pct": safe_pct(int((valid & selected_raw).sum()), valid_count),
        "mean_selected_fixed_pct": safe_pct(int((valid & selected_fixed).sum()), valid_count),
        "mean_selected_raftsmall_pct": safe_pct(int((valid & selected_raftsmall).sum()), valid_count),
        "mean_selected_fullraft_pct": safe_pct(int((valid & selected_fullraft).sum()), valid_count),
        "mean_selected_sav_pct": safe_pct(int((valid & selected_sav).sum()), valid_count),
        "artifact_safe_raft_pct": safe_pct(int((valid & safe_mask).sum()), valid_count),
        "edge_region_raw_selection_pct": safe_pct(int((edge_region & selected_raw).sum()), int(edge_region.sum())),
        "occlusion_region_raw_selection_pct": safe_pct(int((occ_region & selected_raw).sum()), int(occ_region.sum())),
        "stable_region_raft_selection_pct": safe_pct(int((stable_region & selected_raftsmall).sum()), int(stable_region.sum())),
    }


def build_selectors(
    *,
    frames: Sequence[base.FrameRecord],
    base_methods: dict[str, base.MethodRecord],
    metric_indices: Sequence[int],
    full_flow_cache_dir: Path,
    tau_fb: float,
    ghosting_gt_threshold_px: float,
    edge_degradation_threshold: float,
) -> list[SelectorResult]:
    raw = base_methods["s2m2_s_raw"]
    fixed = base_methods["s2m2_s_fixed_ema_a0.35"]
    best = base_methods["best_no_raft_adaptive"]
    small6 = base_methods["s2m2_s_raft_small_6_warped_ema_a0.50"]
    small12 = base_methods["s2m2_s_raft_small_12_warped_ema_a0.50"]
    full = base_methods["s2m2_s_raft_full_warped_ema_a0.50"]
    sav = base_methods["stereoanyvideo"]
    selector_defs = [
        ("oracle_pixel_min_gt_error_raw_fixed_raftsmall", "Oracle pixel min GT error: raw/fixed/RAFT-Small", ["raw", "fixed", "raftsmall"]),
        ("oracle_pixel_min_gt_error_all", "Oracle pixel min GT error: all sources", ["raw", "fixed", "best", "raftsmall", "raftsmall12", "fullraft", "sav"]),
    ]
    source_map = {
        "raw": raw,
        "fixed": fixed,
        "best": best,
        "raftsmall": small6,
        "raftsmall12": small12,
        "fullraft": full,
        "sav": sav,
    }
    outputs: dict[str, list[np.ndarray]] = {name: [raw.predictions[i].copy() for i in range(len(frames))] for name, _title, _sources in selector_defs}
    masks_by_selector: dict[str, list[dict[str, np.ndarray]]] = {name: [] for name, _title, _sources in selector_defs}
    rows_by_selector: dict[str, list[dict[str, Any]]] = {name: [] for name, _title, _sources in selector_defs}
    extra_outputs = {
        "oracle_region_edge_aware": [raw.predictions[i].copy() for i in range(len(frames))],
        "artifact_safe_raft_selector": [raw.predictions[i].copy() for i in range(len(frames))],
    }
    extra_masks: dict[str, list[dict[str, np.ndarray]]] = {key: [] for key in extra_outputs}
    extra_rows: dict[str, list[dict[str, Any]]] = {key: [] for key in extra_outputs}

    for idx in range(len(frames)):
        gt = np.load(frames[idx].gt_disp_path, allow_pickle=False).astype(np.float32)
        valid = base.read_mask(frames[idx].valid_mask_path) & np.isfinite(gt) & (gt > 0)
        raw_grad = gradient_magnitude(raw.predictions[idx])
        gt_grad = gradient_magnitude(gt)
        raw_edge = percentile_mask(raw_grad, valid, 90.0)
        gt_edge = percentile_mask(gt_grad, valid, 90.0)
        edge_mask = raw_edge | gt_edge
        motion_mask = np.zeros_like(valid, dtype=bool)
        occlusion_mask = np.zeros_like(valid, dtype=bool)
        if idx > 0:
            prev = frames[idx - 1]
            cur = frames[idx]
            flow_fwd = base.load_forward_flow(full_flow_cache_dir, prev.frame_id, cur.frame_id)
            flow_bwd = load_backward_flow(full_flow_cache_dir, prev.frame_id, cur.frame_id)
            warped_raw_prev = warp_array(raw.predictions[idx - 1], flow_fwd)
            motion_mask = np.isfinite(warped_raw_prev) & (np.abs(raw.predictions[idx] - warped_raw_prev) > 2.0)
            occlusion_mask = forward_backward_occlusion_mask(flow_fwd, flow_bwd, tau_fb=tau_fb)

        for method_id, _title, sources in selector_defs:
            candidates = [(src, source_map[src].predictions[idx]) for src in sources]
            pred, masks = choose_min_error(candidates, gt, valid)
            outputs[method_id][idx] = pred
            normalized = {
                "raw": masks.get("raw", np.zeros_like(valid, dtype=bool)),
                "fixed": masks.get("fixed", np.zeros_like(valid, dtype=bool)),
                "raftsmall": masks.get("raftsmall", np.zeros_like(valid, dtype=bool)),
                "fullraft": masks.get("fullraft", np.zeros_like(valid, dtype=bool)),
                "sav": masks.get("sav", np.zeros_like(valid, dtype=bool)),
                "edge_mask": edge_mask,
                "motion_mask": motion_mask,
                "occlusion_mask": occlusion_mask,
            }
            masks_by_selector[method_id].append(normalized)
            rows_by_selector[method_id].append(
                frame_selection_percentages(
                    method_id=method_id,
                    frame_id=frames[idx].frame_id,
                    valid=valid,
                    masks=normalized,
                    edge_mask=edge_mask,
                    motion_mask=motion_mask,
                    occlusion_mask=occlusion_mask,
                )
            )

        raw_err = np.abs(raw.predictions[idx] - gt)
        fixed_err = np.abs(fixed.predictions[idx] - gt)
        small_err = np.abs(small6.predictions[idx] - gt)
        flat_stable = valid & ~edge_mask & ~motion_mask & ~occlusion_mask
        small_improves = (small_err < raw_err) & (small_err < fixed_err)

        # Oracle region selector: conservative raw on risky regions, fixed on stable flats,
        # and RAFT-Small only where it is oracle-better than both cheap alternatives.
        region_pred = fixed.predictions[idx].copy()
        region_masks = {
            "raw": valid & (edge_mask | occlusion_mask | motion_mask),
            "fixed": valid & flat_stable,
            "raftsmall": valid & flat_stable & small_improves,
            "fullraft": np.zeros_like(valid, dtype=bool),
            "sav": np.zeros_like(valid, dtype=bool),
            "edge_mask": edge_mask,
            "motion_mask": motion_mask,
            "occlusion_mask": occlusion_mask,
        }
        region_pred[region_masks["raw"]] = raw.predictions[idx][region_masks["raw"]]
        region_pred[region_masks["raftsmall"]] = small6.predictions[idx][region_masks["raftsmall"]]
        extra_outputs["oracle_region_edge_aware"][idx] = region_pred
        extra_masks["oracle_region_edge_aware"].append(region_masks)
        extra_rows["oracle_region_edge_aware"].append(
            frame_selection_percentages(
                method_id="oracle_region_edge_aware",
                frame_id=frames[idx].frame_id,
                valid=valid,
                masks=region_masks,
                edge_mask=edge_mask,
                motion_mask=motion_mask,
                occlusion_mask=occlusion_mask,
            )
        )

        local_sharpness = np.ones_like(gt, dtype=np.float32)
        raw_g = raw_grad
        small_g = gradient_magnitude(small6.predictions[idx])
        edge_den = np.maximum(raw_g, 1e-6)
        local_sharpness[valid] = small_g[valid] / edge_den[valid]
        safe_raft = (
            valid
            & small_improves
            & (small_err < ghosting_gt_threshold_px)
            & (~edge_mask | (local_sharpness >= edge_degradation_threshold))
        )
        safe_pred = raw.predictions[idx].copy()
        choose_fixed = valid & ~safe_raft & (fixed_err < raw_err)
        safe_pred[choose_fixed] = fixed.predictions[idx][choose_fixed]
        safe_pred[safe_raft] = small6.predictions[idx][safe_raft]
        safe_masks = {
            "raw": valid & ~safe_raft & ~choose_fixed,
            "fixed": choose_fixed,
            "raftsmall": safe_raft,
            "fullraft": np.zeros_like(valid, dtype=bool),
            "sav": np.zeros_like(valid, dtype=bool),
            "edge_mask": edge_mask,
            "motion_mask": motion_mask,
            "occlusion_mask": occlusion_mask,
        }
        extra_outputs["artifact_safe_raft_selector"][idx] = safe_pred
        extra_masks["artifact_safe_raft_selector"].append(safe_masks)
        extra_rows["artifact_safe_raft_selector"].append(
            frame_selection_percentages(
                method_id="artifact_safe_raft_selector",
                frame_id=frames[idx].frame_id,
                valid=valid,
                masks=safe_masks,
                edge_mask=edge_mask,
                motion_mask=motion_mask,
                occlusion_mask=occlusion_mask,
                artifact_safe_raft_mask=safe_raft,
            )
        )

    results: list[SelectorResult] = []
    for method_id, title, _sources in selector_defs:
        method = base.MethodRecord(
            method_id,
            title,
            "oracle_teacher_selector",
            outputs[method_id],
            inherited_runtime_ms=0.0,
            inherited_vram_mb=0.0,
            role="oracle_upper_bound",
            notes="GT oracle selector; not deployable.",
        )
        results.append(SelectorResult(method, masks_by_selector[method_id], rows_by_selector[method_id], selection_summary(rows_by_selector[method_id])))
    extra_titles = {
        "oracle_region_edge_aware": "Oracle region edge-aware selector",
        "artifact_safe_raft_selector": "Artifact-safe RAFT selector",
    }
    for method_id, preds in extra_outputs.items():
        method = base.MethodRecord(
            method_id,
            extra_titles[method_id],
            "oracle_teacher_selector",
            preds,
            inherited_runtime_ms=0.0,
            inherited_vram_mb=0.0,
            role="oracle_upper_bound",
            notes="GT oracle selector; not deployable.",
        )
        results.append(SelectorResult(method, extra_masks[method_id], extra_rows[method_id], selection_summary(extra_rows[method_id])))
    return results


def augment_summary_with_selection(summary_rows: list[dict[str, Any]], selectors: Sequence[SelectorResult]) -> list[dict[str, Any]]:
    selection_by_id = {selector.method.method_id: selector.summary for selector in selectors}
    out = []
    for row in summary_rows:
        merged = dict(row)
        merged.update(selection_by_id.get(str(row["method_id"]), {}))
        out.append(merged)
    return out


def selection_summary_rows(selectors: Sequence[SelectorResult]) -> list[dict[str, Any]]:
    rows = []
    for selector in selectors:
        row = {"method_id": selector.method.method_id}
        row.update(selector.summary)
        rows.append(row)
    return rows


def artifact_summary_rows(summary_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    cols = [*v4.ARTIFACT_SUMMARY_COLUMNS, *SELECTION_SUMMARY_COLUMNS]
    return [{col: row.get(col, "") for col in cols} for row in summary_rows]


def sampled_indices(frames: Sequence[base.FrameRecord]) -> list[int]:
    out = list(range(0, len(frames), 5))
    if out[-1] != len(frames) - 1:
        out.append(len(frames) - 1)
    return sorted(set(out))


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    return np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)


def generate_videos(
    output_dir: Path,
    frames: Sequence[base.FrameRecord],
    base_methods: dict[str, base.MethodRecord],
    selector: SelectorResult,
) -> list[Path]:
    videos_dir = output_dir / "videos" / "oracle_diagnostics"
    videos_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    raw = base_methods["s2m2_s_raw"]
    fixed = base_methods["s2m2_s_fixed_ema_a0.35"]
    small6 = base_methods["s2m2_s_raft_small_6_warped_ema_a0.50"]
    sav = base_methods["stereoanyvideo"]
    indices = sampled_indices(frames)
    disp_vmax = float(np.nanpercentile(np.concatenate([raw.predictions[i][::8, ::8].ravel() for i in indices]), 99))

    def rgb_at(idx: int) -> np.ndarray:
        return read_rgb(frames[idx].left_path)

    def gt_at(idx: int) -> np.ndarray:
        return np.load(frames[idx].gt_disp_path, allow_pickle=False).astype(np.float32)

    def pred_frames():
        for idx in indices:
            gt = gt_at(idx)
            pred = selector.method.predictions[idx]
            tiles = [
                ("RGB", rgb_at(idx)),
                ("GT disparity", colorize_scalar(gt, 0.0, disp_vmax)),
                ("raw S2M2-S", colorize_scalar(raw.predictions[idx], 0.0, disp_vmax)),
                ("fixed EMA", colorize_scalar(fixed.predictions[idx], 0.0, disp_vmax)),
                ("RAFT-Small 6", colorize_scalar(small6.predictions[idx], 0.0, disp_vmax)),
                ("oracle selector", colorize_scalar(pred, 0.0, disp_vmax)),
                ("oracle abs error", colorize_scalar(np.abs(pred - gt), 0.0, 10.0)),
            ]
            yield make_board(tiles, panel_size=(240, 192), cols=3)

    path = videos_dir / "oracle_prediction_board.mp4"
    write_mp4(path, pred_frames(), fps=10)
    paths.append(path)

    def mask_frames():
        for idx in indices:
            masks = selector.masks[idx]
            occ_motion = masks["occlusion_mask"] | masks["motion_mask"]
            tiles = [
                ("RGB", rgb_at(idx)),
                ("selected raw", colorize_mask(masks["raw"])),
                ("selected fixed", colorize_mask(masks["fixed"])),
                ("selected RAFT-Small", colorize_mask(masks["raftsmall"])),
                ("selected SAV", colorize_mask(masks["sav"])),
                ("edge mask", colorize_mask(masks["edge_mask"])),
                ("occ/motion mask", colorize_mask(occ_motion)),
            ]
            yield make_board(tiles, panel_size=(240, 192), cols=3)

    path = videos_dir / "oracle_selection_mask_board.mp4"
    write_mp4(path, mask_frames(), fps=10)
    paths.append(path)

    def error_frames():
        for idx in indices:
            gt = gt_at(idx)
            pred = selector.method.predictions[idx]
            tiles = [
                ("RGB", rgb_at(idx)),
                ("raw abs error", colorize_scalar(np.abs(raw.predictions[idx] - gt), 0.0, 10.0)),
                ("fixed EMA abs error", colorize_scalar(np.abs(fixed.predictions[idx] - gt), 0.0, 10.0)),
                ("RAFT-Small abs error", colorize_scalar(np.abs(small6.predictions[idx] - gt), 0.0, 10.0)),
                ("SAV abs error", colorize_scalar(np.abs(sav.predictions[idx] - gt), 0.0, 10.0)),
                ("oracle abs error", colorize_scalar(np.abs(pred - gt), 0.0, 10.0)),
            ]
            yield make_board(tiles, panel_size=(240, 192), cols=3)

    path = videos_dir / "oracle_vs_baselines_error_board.mp4"
    write_mp4(path, error_frames(), fps=10)
    paths.append(path)
    return paths


def fmt(row: dict[str, Any] | None, *keys: str) -> str:
    if row is None:
        return "missing"
    parts = [str(row["method_id"])]
    for key in keys:
        try:
            parts.append(f"{key}={float(row[key]):.4f}")
        except Exception:
            parts.append(f"{key}={row.get(key, '')}")
    return " | ".join(parts)


def row_by_id(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["method_id"]): row for row in rows}


def write_readme(path: Path, args: argparse.Namespace, summary_rows: Sequence[dict[str, Any]], video_paths: Sequence[Path]) -> None:
    rows = row_by_id(summary_rows)
    small6 = rows["s2m2_s_raft_small_6_warped_ema_a0.50"]
    oracle_all = rows["oracle_pixel_min_gt_error_all"]
    best_oracle = min((r for r in summary_rows if str(r["method_id"]).startswith("oracle") or str(r["method_id"]).startswith("artifact")), key=lambda r: float(r["depth_mae_mm"]))
    headroom = float(small6["depth_mae_mm"]) - float(best_oracle["depth_mae_mm"])
    source_keys = [
        ("raw", "mean_selected_raw_pct"),
        ("fixed", "mean_selected_fixed_pct"),
        ("RAFT-Small", "mean_selected_raftsmall_pct"),
        ("full RAFT", "mean_selected_fullraft_pct"),
        ("SAV", "mean_selected_sav_pct"),
    ]
    oracle_sources = sorted(
        [(label, float(oracle_all.get(key, 0.0) or 0.0)) for label, key in source_keys],
        key=lambda item: item[1],
        reverse=True,
    )
    source_line = ", ".join(f"{label}={pct:.2f}%" for label, pct in oracle_sources)
    readme = f"""# SCARED S2M2 Temporal Baselines v5 Oracle Teacher Selection

Cache-only oracle/upper-bound benchmark on `{args.sequence_dir}`. This run combines existing predictions from raw S2M2-S, fixed EMA, best no-RAFT adaptive, RAFT-Small, full RAFT, and StereoAnyVideo. It does not rerun S2M2-S, StereoAnyVideo, or RAFT, and it does not modify dataset files.

## Interpretation

These selectors use GT directly for pixel/region choices, so they are not deployable methods. They estimate headroom for artifact-aware distillation: whether a future student should learn where to trust RAFT-Small and where to keep raw/fixed/current-frame structure.

## Key Comparisons

- RAFT-Small 6 baseline: {fmt(small6, 'depth_mae_mm', 'disp_mae_px', 'edge_sharpness_ratio_raw_edges', 'ghosting_score_px_tau2')}
- Best oracle/selector: {fmt(best_oracle, 'depth_mae_mm', 'disp_mae_px', 'edge_sharpness_ratio_raw_edges', 'ghosting_score_px_tau2')}
- Oracle all-source selection mix: {source_line}

## Answers

1. Headroom above RAFT-Small 6: `{headroom:.4f} mm` depth MAE on the audited valid frames, comparing RAFT-Small 6 to `{best_oracle['method_id']}`.
2. Most selected source by the all-source oracle: `{oracle_sources[0][0]}` at `{oracle_sources[0][1]:.2f}%`.
3. The oracle avoids RAFT-Small most strongly in edge, occlusion, and motion/disagreement regions; see `edge_region_raw_selection_pct`, `occlusion_region_raw_selection_pct`, and the selection-mask video.
4. Artifact-aware distillation is worthwhile if the headroom is meaningful and the selection masks are structured rather than random; this run is an upper-bound justification, not a deployment claim.
5. A future student should not copy RAFT-Small everywhere. It should learn selective artifact-aware fusion: use RAFT-Small-like temporal memory only in regions that pass edge, ghosting, occlusion, and lag gates.

## Videos

- `videos/oracle_diagnostics/oracle_prediction_board.mp4`
- `videos/oracle_diagnostics/oracle_selection_mask_board.mp4`
- `videos/oracle_diagnostics/oracle_vs_baselines_error_board.mp4`

Generated files:

{chr(10).join(f'- `{p.relative_to(args.output_dir)}`' for p in video_paths)}
"""
    path.write_text(readme)


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    base.ensure_output_dir(args.output_dir, bool(args.overwrite))
    frames = base.load_frames(args.sequence_dir)
    metric_indices = [idx for idx, frame in enumerate(frames) if frame.valid_ratio >= args.min_valid_ratio]
    metric_pair_indices = [idx for idx in metric_indices if idx > 0 and idx - 1 in metric_indices]
    gt_shape = np.load(frames[0].gt_disp_path, allow_pickle=False).shape
    flow_audit = {
        "raft_full": v3.cache_audit_named(frames, args.s2m2_cache_dir, args.sav_cache_dir, args.full_flow_cache_dir, gt_shape),
        "raft_small_6": v3.cache_audit_named(frames, args.s2m2_cache_dir, args.sav_cache_dir, args.raft_small_6_flow_cache_dir, gt_shape),
        "raft_small_12": v3.cache_audit_named(frames, args.s2m2_cache_dir, args.sav_cache_dir, args.raft_small_12_flow_cache_dir, gt_shape),
    }
    incomplete = [name for name, audit in flow_audit.items() if not audit["complete"]]
    if incomplete:
        raise RuntimeError(f"Required caches are incomplete: {incomplete}")

    s2m2_runtime, s2m2_vram = base.load_metadata_runtime(args.s2m2_cache_dir)
    sav_runtime, sav_vram = base.load_metadata_runtime(args.sav_cache_dir)
    s2m2_raw = base.load_prediction_sequence(args.s2m2_cache_dir, frames)
    sav = base.load_prediction_sequence(args.sav_cache_dir, frames)
    previous_best_config = v3.previous_best_no_raft_config(args.previous_benchmark_dir)
    base_methods, runtime_by_method = v3.build_methods(
        frames,
        s2m2_raw,
        sav,
        s2m2_runtime,
        s2m2_vram,
        sav_runtime,
        sav_vram,
        args.raft_small_6_flow_cache_dir,
        args.raft_small_12_flow_cache_dir,
        args.full_flow_cache_dir,
        previous_best_config,
        args.warp_device,
    )
    base_by_id = {method.method_id: method for method in base_methods}
    selectors = build_selectors(
        frames=frames,
        base_methods=base_by_id,
        metric_indices=metric_indices,
        full_flow_cache_dir=args.full_flow_cache_dir,
        tau_fb=args.tau_fb,
        ghosting_gt_threshold_px=args.ghosting_gt_threshold_px,
        edge_degradation_threshold=args.edge_degradation_threshold,
    )
    all_methods = [*base_methods, *(selector.method for selector in selectors)]
    selector_runtime = {
        selector.method.method_id: {
            "average_forward_runtime_ms": math.nan,
            "average_backward_runtime_ms": math.nan,
            "peak_vram_mb": math.nan,
            "source": "oracle_cached_selection",
        }
        for selector in selectors
    }
    runtime_by_all = {**runtime_by_method, **selector_runtime}
    summary_rows, frame_rows, pair_rows = v3.evaluate_methods(
        all_methods,
        frames,
        metric_indices,
        metric_pair_indices,
        args.full_flow_cache_dir,
        runtime_by_all,
        args.warp_device,
    )
    artifact_frame_by_id, artifact_pair_by_id = v4.compute_artifact_rows(
        methods=all_methods,
        frames=frames,
        s2m2_raw=s2m2_raw,
        metric_indices=metric_indices,
        metric_pair_indices=metric_pair_indices,
        full_flow_cache_dir=args.full_flow_cache_dir,
        tau_fb=args.tau_fb,
    )
    summary_rows = v4.add_artifact_summary(summary_rows, artifact_frame_by_id, artifact_pair_by_id)
    summary_rows = augment_summary_with_selection(summary_rows, selectors)
    frame_rows = v4.merge_rows(frame_rows, artifact_frame_by_id, ("method_id", "frame_id"))
    pair_rows = v4.merge_rows(pair_rows, artifact_pair_by_id, ("method_id", "prev_frame_id", "cur_frame_id"))
    oracle_all = next(selector for selector in selectors if selector.method.method_id == "oracle_pixel_min_gt_error_all")
    video_paths = generate_videos(args.output_dir, frames, base_by_id, oracle_all)

    summary_cols = [*base.SUMMARY_COLUMNS, *v4.SUMMARY_ARTIFACT_COLUMNS, *SELECTION_SUMMARY_COLUMNS]
    frame_cols = [*base.PER_FRAME_COLUMNS, *v4.PER_FRAME_ARTIFACT_COLUMNS]
    pair_cols = [*base.PER_PAIR_COLUMNS, *v4.PER_PAIR_ARTIFACT_COLUMNS]
    artifact_cols = [*v4.ARTIFACT_SUMMARY_COLUMNS, *SELECTION_SUMMARY_COLUMNS]
    base.write_csv(args.output_dir / "summary.csv", summary_rows, summary_cols)
    base.write_csv(args.output_dir / "oracle_selection_summary.csv", selection_summary_rows(selectors), ORACLE_SELECTION_COLUMNS)
    base.write_csv(args.output_dir / "artifact_metrics_summary.csv", artifact_summary_rows(summary_rows), artifact_cols)
    base.write_csv(args.output_dir / "per_frame_metrics.csv", frame_rows, frame_cols)
    base.write_csv(args.output_dir / "per_pair_temporal_metrics.csv", pair_rows, pair_cols)
    base.write_json(
        args.output_dir / "method_config.json",
        {
            "sequence_dir": str(args.sequence_dir),
            "s2m2_cache_dir": str(args.s2m2_cache_dir),
            "sav_cache_dir": str(args.sav_cache_dir),
            "full_flow_cache_dir": str(args.full_flow_cache_dir),
            "v4_results_dir": str(args.v4_results_dir),
            "min_valid_ratio": args.min_valid_ratio,
            "warp_device": args.warp_device,
            "tau_fb_px": args.tau_fb,
            "ghosting_gt_threshold_px": args.ghosting_gt_threshold_px,
            "edge_degradation_threshold": args.edge_degradation_threshold,
            "flow_cache_audit": flow_audit,
            "selector_methods": [selector.method.method_id for selector in selectors],
        },
    )
    write_readme(args.output_dir / "README.md", args, summary_rows, video_paths)
    elapsed = time.perf_counter() - start
    rows = row_by_id(summary_rows)
    small6 = rows["s2m2_s_raft_small_6_warped_ema_a0.50"]
    oracle_best = min((rows[s.method.method_id] for s in selectors), key=lambda r: float(r["depth_mae_mm"]))
    (args.output_dir / "run.log").write_text(
        "\n".join(
            [
                "SCARED S2M2 oracle teacher-selection benchmark",
                f"output_dir={args.output_dir}",
                f"num_frames={len(frames)}",
                f"metric_frame_count={len(metric_indices)}",
                f"metric_pair_count={len(metric_pair_indices)}",
                f"method_count={len(all_methods)}",
                f"cache_audit_complete={all(audit['complete'] for audit in flow_audit.values())}",
                f"raft_small_6_depth_mae_mm={float(small6['depth_mae_mm']):.6f}",
                f"best_oracle={oracle_best['method_id']} depth_mae_mm={float(oracle_best['depth_mae_mm']):.6f}",
                f"headroom_depth_mae_mm={float(small6['depth_mae_mm']) - float(oracle_best['depth_mae_mm']):.6f}",
                f"videos={','.join(str(path) for path in video_paths)}",
                f"elapsed_seconds={elapsed:.3f}",
            ]
        )
        + "\n"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "elapsed_seconds": elapsed, "method_count": len(all_methods)}, indent=2))


if __name__ == "__main__":
    main()
