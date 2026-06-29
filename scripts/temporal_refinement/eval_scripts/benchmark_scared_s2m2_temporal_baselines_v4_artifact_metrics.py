#!/usr/bin/env python3
"""Artifact-aware RAFT-Small temporal-refinement benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
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
from artifact_metrics import (  # noqa: E402
    forward_backward_occlusion_mask,
    frame_artifact_metrics,
    gradient_magnitude,
    lag_rate,
    nanmean_rows,
    pair_artifact_metrics,
    percentile_mask,
    warp_array,
)
from video_qualitative import colorize_scalar, make_board, write_mp4  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("results/03_temporal_refinement/scared_s2m2_temporal_baselines_v4_raft_small_artifact_metrics")

SUMMARY_ARTIFACT_COLUMNS = [
    "edge_sharpness_ratio_raw",
    "edge_sharpness_ratio_raw_edges",
    "boundary_disp_mae_px",
    "boundary_disp_mae_px_p80",
    "ghosting_score_px_tau2",
    "ghosting_gt_error_px_tau2",
    "ghosting_score_px_tau5",
    "ghosting_gt_error_px_tau5",
    "occlusion_disp_mae_px",
    "lag_rate",
    "lag_error_margin_px",
    "rgb_disp_edge_corr",
    "rgb_disp_edge_corr_rgb_edges",
]

PER_FRAME_ARTIFACT_COLUMNS = [
    "edge_sharpness_ratio_raw",
    "edge_sharpness_ratio_raw_edges",
    "boundary_disp_mae_px",
    "boundary_disp_mae_px_p80",
    "rgb_disp_edge_corr",
    "rgb_disp_edge_corr_rgb_edges",
]

PER_PAIR_ARTIFACT_COLUMNS = [
    "ghosting_score_px_tau2",
    "ghosting_gt_error_px_tau2",
    "ghosting_score_px_tau5",
    "ghosting_gt_error_px_tau5",
    "occlusion_disp_mae_px",
    "lagged_frame",
    "lag_error_margin_px",
]

ARTIFACT_SUMMARY_COLUMNS = [
    "method_id",
    "depth_mae_mm",
    "disp_mae_px",
    "motion_compensated_temporal_mae_px",
    "edge_sharpness_ratio_raw",
    "edge_sharpness_ratio_raw_edges",
    "boundary_disp_mae_px",
    "ghosting_score_px_tau2",
    "ghosting_gt_error_px_tau2",
    "occlusion_disp_mae_px",
    "lag_rate",
    "rgb_disp_edge_corr",
    "online_runtime_estimated_ms",
    "online_peak_vram_estimated_mb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SCARED S2M2-S artifact-aware RAFT-Small benchmark.")
    parser.add_argument("--sequence-dir", type=Path, default=base.DEFAULT_SEQUENCE_DIR)
    parser.add_argument("--s2m2-cache-dir", type=Path, default=base.DEFAULT_S2M2_CACHE)
    parser.add_argument("--sav-cache-dir", type=Path, default=base.DEFAULT_SAV_CACHE)
    parser.add_argument("--full-flow-cache-dir", type=Path, default=v3.DEFAULT_FULL_FLOW_CACHE)
    parser.add_argument("--raft-small-6-flow-cache-dir", type=Path, default=v3.DEFAULT_SMALL6_FLOW_CACHE)
    parser.add_argument("--raft-small-12-flow-cache-dir", type=Path, default=v3.DEFAULT_SMALL12_FLOW_CACHE)
    parser.add_argument("--previous-benchmark-dir", type=Path, default=v3.DEFAULT_PREVIOUS_BENCHMARK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-valid-ratio", type=float, default=0.2)
    parser.add_argument("--overwrite", nargs="?", const=True, default=False, type=base.parse_bool)
    parser.add_argument("--warp-device", default="auto")
    parser.add_argument("--tau-fb", type=float, default=1.5)
    return parser.parse_args()


def read_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    return np.array(Image.open(path).convert("RGB"))


def load_backward_flow(flow_cache_dir: Path, prev_id: str, cur_id: str) -> np.ndarray:
    return np.load(flow_cache_dir / "backward_flow" / f"{cur_id}_to_{prev_id}.npy", allow_pickle=False).astype(np.float32)


def compute_artifact_rows(
    *,
    methods: Sequence[base.MethodRecord],
    frames: Sequence[base.FrameRecord],
    s2m2_raw: Sequence[np.ndarray],
    metric_indices: Sequence[int],
    metric_pair_indices: Sequence[int],
    full_flow_cache_dir: Path,
    tau_fb: float,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    frame_rows_by_id: dict[str, list[dict[str, Any]]] = {method.method_id: [] for method in methods}
    pair_rows_by_id: dict[str, list[dict[str, Any]]] = {method.method_id: [] for method in methods}
    rgb_cache: dict[int, np.ndarray] = {}
    gt_cache: dict[int, np.ndarray] = {}
    valid_cache: dict[int, np.ndarray] = {}

    def rgb(idx: int) -> np.ndarray:
        if idx not in rgb_cache:
            rgb_cache[idx] = read_rgb(frames[idx].left_path)
        return rgb_cache[idx]

    def gt(idx: int) -> np.ndarray:
        if idx not in gt_cache:
            gt_cache[idx] = np.load(frames[idx].gt_disp_path, allow_pickle=False).astype(np.float32)
        return gt_cache[idx]

    def valid(idx: int) -> np.ndarray:
        if idx not in valid_cache:
            valid_cache[idx] = base.read_mask(frames[idx].valid_mask_path)
        return valid_cache[idx]

    for idx in metric_indices:
        for method in methods:
            metrics = frame_artifact_metrics(
                pred=method.predictions[idx],
                raw=s2m2_raw[idx],
                gt=gt(idx),
                valid_mask=valid(idx),
                rgb=rgb(idx),
            )
            metrics.update({"method_id": method.method_id, "frame_id": frames[idx].frame_id})
            frame_rows_by_id[method.method_id].append(metrics)

    for idx in metric_pair_indices:
        prev = frames[idx - 1]
        cur = frames[idx]
        flow_fwd = base.load_forward_flow(full_flow_cache_dir, prev.frame_id, cur.frame_id)
        flow_bwd = load_backward_flow(full_flow_cache_dir, prev.frame_id, cur.frame_id)
        for method in methods:
            metrics = pair_artifact_metrics(
                prev_pred=method.predictions[idx - 1],
                cur_pred=method.predictions[idx],
                cur_raw=s2m2_raw[idx],
                prev_gt=gt(idx - 1),
                cur_gt=gt(idx),
                prev_valid_mask=valid(idx - 1),
                cur_valid_mask=valid(idx),
                flow_fwd=flow_fwd,
                flow_bwd=flow_bwd,
                tau_fb=tau_fb,
            )
            metrics.update({"method_id": method.method_id, "prev_frame_id": prev.frame_id, "cur_frame_id": cur.frame_id})
            pair_rows_by_id[method.method_id].append(metrics)

    return frame_rows_by_id, pair_rows_by_id


def merge_rows(
    base_rows: list[dict[str, Any]],
    artifact_rows_by_id: dict[str, list[dict[str, Any]]],
    key_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    lookup = {
        tuple(str(row[field]) for field in key_fields): row
        for rows in artifact_rows_by_id.values()
        for row in rows
    }
    merged: list[dict[str, Any]] = []
    for row in base_rows:
        out = dict(row)
        artifact = lookup.get(tuple(str(row[field]) for field in key_fields), {})
        out.update({k: v for k, v in artifact.items() if k not in key_fields})
        merged.append(out)
    return merged


def add_artifact_summary(
    summary_rows: list[dict[str, Any]],
    frame_rows_by_id: dict[str, list[dict[str, Any]]],
    pair_rows_by_id: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in summary_rows:
        method_id = str(row["method_id"])
        merged = dict(row)
        frame_rows = frame_rows_by_id[method_id]
        pair_rows = pair_rows_by_id[method_id]
        for key in PER_FRAME_ARTIFACT_COLUMNS:
            merged[key] = nanmean_rows(frame_rows, key)
        for key in [
            "ghosting_score_px_tau2",
            "ghosting_gt_error_px_tau2",
            "ghosting_score_px_tau5",
            "ghosting_gt_error_px_tau5",
            "occlusion_disp_mae_px",
            "lag_error_margin_px",
        ]:
            merged[key] = nanmean_rows(pair_rows, key)
        merged["lag_rate"] = lag_rate(pair_rows)
        out.append(merged)
    return out


def artifact_summary_rows(summary_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{col: row.get(col, "") for col in ARTIFACT_SUMMARY_COLUMNS} for row in summary_rows]


def by_id(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["method_id"]): row for row in rows}


def finite_best(rows: Sequence[dict[str, Any]], key: str, maximize: bool = False) -> dict[str, Any] | None:
    candidates = []
    for row in rows:
        try:
            value = float(row.get(key, math.nan))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            candidates.append((value, row))
    if not candidates:
        return None
    return (max if maximize else min)(candidates, key=lambda item: item[0])[1]


def fmt_metric(row: dict[str, Any] | None, *keys: str) -> str:
    if row is None:
        return "missing"
    parts = [str(row["method_id"])]
    for key in keys:
        value = row.get(key, "")
        try:
            parts.append(f"{key}={float(value):.4f}")
        except (TypeError, ValueError):
            parts.append(f"{key}={value}")
    return " | ".join(parts)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    return np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)


def sampled_indices(frames: Sequence[base.FrameRecord]) -> list[int]:
    out = list(range(1, len(frames), 5))
    if 0 not in out:
        out.insert(0, 0)
    if out[-1] != len(frames) - 1:
        out.append(len(frames) - 1)
    return sorted(set(out))


def generate_artifact_videos(
    output_dir: Path,
    frames: Sequence[base.FrameRecord],
    methods: Sequence[base.MethodRecord],
    s2m2_raw: Sequence[np.ndarray],
    full_flow_cache_dir: Path,
) -> list[Path]:
    videos_dir = output_dir / "videos" / "artifact_diagnostics"
    videos_dir.mkdir(parents=True, exist_ok=True)
    method_by_id = {method.method_id: method for method in methods}
    raw = method_by_id["s2m2_s_raw"]
    fixed = method_by_id["s2m2_s_fixed_ema_a0.35"]
    small6 = method_by_id["s2m2_s_raft_small_6_warped_ema_a0.50"]
    full = method_by_id["s2m2_s_raft_full_warped_ema_a0.50"]
    paths: list[Path] = []
    indices = sampled_indices(frames)
    grad_vmax = 8.0
    disp_vmax = float(np.nanpercentile(np.concatenate([s2m2_raw[i][::8, ::8].ravel() for i in indices]), 99))

    def rgb_at(idx: int) -> np.ndarray:
        return read_rgb(frames[idx].left_path)

    def gt_at(idx: int) -> np.ndarray:
        return np.load(frames[idx].gt_disp_path, allow_pickle=False).astype(np.float32)

    def valid_at(idx: int) -> np.ndarray:
        return base.read_mask(frames[idx].valid_mask_path)

    def edge_frames():
        for idx in indices:
            tiles = [
                ("RGB", rgb_at(idx)),
                ("raw S2M2 grad", colorize_scalar(gradient_magnitude(raw.predictions[idx]), 0.0, grad_vmax)),
                ("fixed EMA grad", colorize_scalar(gradient_magnitude(fixed.predictions[idx]), 0.0, grad_vmax)),
                ("RAFT-Small 6 grad", colorize_scalar(gradient_magnitude(small6.predictions[idx]), 0.0, grad_vmax)),
                ("full RAFT grad", colorize_scalar(gradient_magnitude(full.predictions[idx]), 0.0, grad_vmax)),
                ("GT disparity grad", colorize_scalar(gradient_magnitude(gt_at(idx)), 0.0, grad_vmax)),
            ]
            yield make_board(tiles, panel_size=(240, 192), cols=3)

    path = videos_dir / "edge_sharpness_board.mp4"
    write_mp4(path, edge_frames(), fps=10)
    paths.append(path)

    def ghosting_frames():
        for idx in [i for i in indices if i > 0]:
            prev = frames[idx - 1]
            cur = frames[idx]
            flow = base.load_forward_flow(full_flow_cache_dir, prev.frame_id, cur.frame_id)
            warped_prev = warp_array(small6.predictions[idx - 1], flow)
            motion = np.abs(s2m2_raw[idx] - warped_prev) > 2.0
            gt = gt_at(idx)
            tiles = [
                ("RGB", rgb_at(idx)),
                ("raw disparity", colorize_scalar(s2m2_raw[idx], 0.0, disp_vmax)),
                ("fixed EMA", colorize_scalar(fixed.predictions[idx], 0.0, disp_vmax)),
                ("RAFT-Small 6", colorize_scalar(small6.predictions[idx], 0.0, disp_vmax)),
                ("abs(Small6 - raw)", colorize_scalar(np.abs(small6.predictions[idx] - s2m2_raw[idx]), 0.0, 8.0)),
                ("motion mask", colorize_mask(motion)),
                ("abs(Small6 - GT)", colorize_scalar(np.abs(small6.predictions[idx] - gt), 0.0, 10.0)),
            ]
            yield make_board(tiles, panel_size=(240, 192), cols=3)

    path = videos_dir / "ghosting_diagnostic_board.mp4"
    write_mp4(path, ghosting_frames(), fps=10)
    paths.append(path)

    def boundary_frames():
        for idx in indices:
            gt = gt_at(idx)
            valid = valid_at(idx)
            gt_edge = percentile_mask(gradient_magnitude(gt), valid, 90.0)
            tiles = [
                ("RGB", rgb_at(idx)),
                ("GT edge mask", colorize_mask(gt_edge)),
                ("raw abs error", colorize_scalar(np.abs(raw.predictions[idx] - gt), 0.0, 10.0)),
                ("fixed EMA abs error", colorize_scalar(np.abs(fixed.predictions[idx] - gt), 0.0, 10.0)),
                ("RAFT-Small 6 abs error", colorize_scalar(np.abs(small6.predictions[idx] - gt), 0.0, 10.0)),
                ("full RAFT abs error", colorize_scalar(np.abs(full.predictions[idx] - gt), 0.0, 10.0)),
            ]
            yield make_board(tiles, panel_size=(240, 192), cols=3)

    path = videos_dir / "boundary_error_board.mp4"
    write_mp4(path, boundary_frames(), fps=10)
    paths.append(path)

    def lag_frames():
        for idx in [i for i in indices if i > 0]:
            gt_cur = gt_at(idx)
            gt_prev = gt_at(idx - 1)
            valid = valid_at(idx) & valid_at(idx - 1)
            err_cur = np.abs(small6.predictions[idx] - gt_cur)
            err_prev = np.abs(small6.predictions[idx] - gt_prev)
            lag_mask = valid & np.isfinite(err_cur) & np.isfinite(err_prev) & (err_prev < err_cur)
            tiles = [
                ("RGB", rgb_at(idx)),
                ("GT current", colorize_scalar(gt_cur, 0.0, disp_vmax)),
                ("GT previous", colorize_scalar(gt_prev, 0.0, disp_vmax)),
                ("fixed EMA", colorize_scalar(fixed.predictions[idx], 0.0, disp_vmax)),
                ("RAFT-Small 6", colorize_scalar(small6.predictions[idx], 0.0, disp_vmax)),
                ("Small6 lag mask", colorize_mask(lag_mask)),
            ]
            yield make_board(tiles, panel_size=(240, 192), cols=3)

    path = videos_dir / "lag_diagnostic_board.mp4"
    write_mp4(path, lag_frames(), fps=10)
    paths.append(path)
    return paths


def write_readme(
    path: Path,
    args: argparse.Namespace,
    summary_rows: Sequence[dict[str, Any]],
    flow_audit: dict[str, Any],
    video_paths: Sequence[Path],
) -> None:
    rows = by_id(summary_rows)
    best_global = finite_best(summary_rows, "depth_mae_mm")
    best_edges = finite_best(summary_rows, "edge_sharpness_ratio_raw_edges", maximize=True)
    worst_ghost = finite_best(summary_rows, "ghosting_score_px_tau2", maximize=True)
    small6 = rows["s2m2_s_raft_small_6_warped_ema_a0.50"]
    fixed = rows["s2m2_s_fixed_ema_a0.35"]
    artifact_prone = (
        float(small6["edge_sharpness_ratio_raw_edges"]) < 0.95
        or float(small6["ghosting_score_px_tau2"]) > float(fixed["ghosting_score_px_tau2"])
        or float(small6["lag_rate"]) > float(fixed["lag_rate"])
    )
    viability = (
        "RAFT-Small should be treated as a numerical upper bound / artifact-prone teacher, not a direct deployment method."
        if artifact_prone
        else "RAFT-Small remains plausible for deployment on this sequence, though artifact checks should gate future use."
    )
    rows_to_compare = [
        "s2m2_s_raw",
        "s2m2_s_fixed_ema_a0.35",
        "s2m2_s_raft_small_6_warped_ema_a0.50",
        "s2m2_s_raft_small_12_warped_ema_a0.50",
        "s2m2_s_raft_full_warped_ema_a0.50",
        "stereoanyvideo",
    ]
    comparison_lines = "\n".join(
        "- "
        + fmt_metric(
            rows[mid],
            "depth_mae_mm",
            "edge_sharpness_ratio_raw_edges",
            "ghosting_gt_error_px_tau2",
            "lag_rate",
            "online_runtime_estimated_ms",
        )
        for mid in rows_to_compare
    )
    readme = f"""# SCARED S2M2 Temporal Baselines v4 Artifact Metrics

Cache-only artifact-aware benchmark on `{args.sequence_dir}`. This run reuses cached S2M2-S, StereoAnyVideo, full RAFT, and RAFT-Small flow outputs; it does not rerun stereo/video inference and does not modify dataset files.

## Numerical vs Visual Quality

The v3 global metrics favored RAFT-Small warped EMA, but qualitative boards showed ghosting, blur, and loss of sharpness. This v4 run keeps the global MAE/runtime metrics and adds artifact-aware checks for disparity edge preservation, boundary error, motion-region ghosting, occlusion error, temporal lag, and RGB/disparity edge alignment. Lower global MAE alone is not enough to call a temporal method deployment-ready.

## Flow Cache Audit

{v3.summarize_flow_cache('full RAFT reference', flow_audit['raft_full'])}
{v3.summarize_flow_cache('RAFT-Small 6', flow_audit['raft_small_6'])}
{v3.summarize_flow_cache('RAFT-Small 12', flow_audit['raft_small_12'])}

## Required Comparisons

{comparison_lines}

## Answers

1. Best global numerical metrics: {fmt_metric(best_global, 'depth_mae_mm', 'disp_mae_px', 'motion_compensated_temporal_mae_px')}.
2. Best edge preservation by raw-edge sharpness ratio: {fmt_metric(best_edges, 'edge_sharpness_ratio_raw_edges', 'boundary_disp_mae_px')}.
3. Worst motion-region ghosting score: {fmt_metric(worst_ghost, 'ghosting_score_px_tau2', 'ghosting_gt_error_px_tau2')}.
4. RAFT-Small viability after artifact metrics: {viability}
5. Deployment suitability: use RAFT-Small as an offline teacher or upper-bound candidate unless artifact gates are added; raw/fixed/no-RAFT methods remain cheaper and may preserve visual sharpness better.
6. Future distillation should be selective and artifact-aware, not copy RAFT-Small everywhere. It should prefer RAFT-Small only where edge sharpness, ghosting, occlusion, and lag diagnostics agree with current-frame evidence.

## Videos

- `videos/artifact_diagnostics/edge_sharpness_board.mp4`
- `videos/artifact_diagnostics/ghosting_diagnostic_board.mp4`
- `videos/artifact_diagnostics/boundary_error_board.mp4`
- `videos/artifact_diagnostics/lag_diagnostic_board.mp4`

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
    base.write_json(args.output_dir / "flow_cache_audit.json", flow_audit)
    incomplete = [name for name, audit in flow_audit.items() if not audit["complete"]]
    if incomplete:
        raise RuntimeError(f"Required caches are incomplete: {incomplete}")

    s2m2_runtime, s2m2_vram = base.load_metadata_runtime(args.s2m2_cache_dir)
    sav_runtime, sav_vram = base.load_metadata_runtime(args.sav_cache_dir)
    s2m2_raw = base.load_prediction_sequence(args.s2m2_cache_dir, frames)
    sav = base.load_prediction_sequence(args.sav_cache_dir, frames)
    previous_best_config = v3.previous_best_no_raft_config(args.previous_benchmark_dir)
    methods, runtime_by_method = v3.build_methods(
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
    summary_rows, frame_rows, pair_rows = v3.evaluate_methods(
        methods,
        frames,
        metric_indices,
        metric_pair_indices,
        args.full_flow_cache_dir,
        runtime_by_method,
        args.warp_device,
    )
    artifact_frame_by_id, artifact_pair_by_id = compute_artifact_rows(
        methods=methods,
        frames=frames,
        s2m2_raw=s2m2_raw,
        metric_indices=metric_indices,
        metric_pair_indices=metric_pair_indices,
        full_flow_cache_dir=args.full_flow_cache_dir,
        tau_fb=args.tau_fb,
    )
    summary_rows = add_artifact_summary(summary_rows, artifact_frame_by_id, artifact_pair_by_id)
    frame_rows = merge_rows(frame_rows, artifact_frame_by_id, ("method_id", "frame_id"))
    pair_rows = merge_rows(pair_rows, artifact_pair_by_id, ("method_id", "prev_frame_id", "cur_frame_id"))
    video_paths = generate_artifact_videos(args.output_dir, frames, methods, s2m2_raw, args.full_flow_cache_dir)

    base.write_csv(args.output_dir / "summary.csv", summary_rows, [*base.SUMMARY_COLUMNS, *SUMMARY_ARTIFACT_COLUMNS])
    base.write_csv(args.output_dir / "per_frame_metrics.csv", frame_rows, [*base.PER_FRAME_COLUMNS, *PER_FRAME_ARTIFACT_COLUMNS])
    base.write_csv(args.output_dir / "per_pair_temporal_metrics.csv", pair_rows, [*base.PER_PAIR_COLUMNS, *PER_PAIR_ARTIFACT_COLUMNS])
    base.write_csv(args.output_dir / "artifact_metrics_summary.csv", artifact_summary_rows(summary_rows), ARTIFACT_SUMMARY_COLUMNS)
    base.write_json(
        args.output_dir / "method_config.json",
        {
            "sequence_dir": str(args.sequence_dir),
            "s2m2_cache_dir": str(args.s2m2_cache_dir),
            "sav_cache_dir": str(args.sav_cache_dir),
            "full_flow_cache_dir": str(args.full_flow_cache_dir),
            "raft_small_6_flow_cache_dir": str(args.raft_small_6_flow_cache_dir),
            "raft_small_12_flow_cache_dir": str(args.raft_small_12_flow_cache_dir),
            "metric_flow_source": str(args.full_flow_cache_dir),
            "min_valid_ratio": args.min_valid_ratio,
            "warp_device": args.warp_device,
            "tau_motion_values_px": [2.0, 5.0],
            "tau_fb_px": args.tau_fb,
            "previous_best_no_raft_config": previous_best_config,
            "runtime_by_method": runtime_by_method,
        },
    )
    write_readme(args.output_dir / "README.md", args, summary_rows, flow_audit, video_paths)
    elapsed = time.perf_counter() - start
    rows = by_id(summary_rows)
    (args.output_dir / "run.log").write_text(
        "\n".join(
            [
                "SCARED S2M2 RAFT-Small artifact-aware temporal benchmark",
                f"output_dir={args.output_dir}",
                f"num_frames={len(frames)}",
                f"metric_frame_count={len(metric_indices)}",
                f"metric_pair_count={len(metric_pair_indices)}",
                f"method_count={len(methods)}",
                f"cache_audit_complete={all(audit['complete'] for audit in flow_audit.values())}",
                f"best_global={fmt_metric(finite_best(summary_rows, 'depth_mae_mm'), 'depth_mae_mm')}",
                f"best_edges={fmt_metric(finite_best(summary_rows, 'edge_sharpness_ratio_raw_edges', maximize=True), 'edge_sharpness_ratio_raw_edges')}",
                f"worst_ghosting={fmt_metric(finite_best(summary_rows, 'ghosting_score_px_tau2', maximize=True), 'ghosting_score_px_tau2', 'ghosting_gt_error_px_tau2')}",
                f"raft_small_6={fmt_metric(rows.get('s2m2_s_raft_small_6_warped_ema_a0.50'), 'depth_mae_mm', 'edge_sharpness_ratio_raw_edges', 'ghosting_score_px_tau2', 'ghosting_gt_error_px_tau2', 'lag_rate')}",
                f"videos={','.join(str(path) for path in video_paths)}",
                f"elapsed_seconds={elapsed:.3f}",
            ]
        )
        + "\n"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "elapsed_seconds": elapsed, "method_count": len(methods)}, indent=2))


if __name__ == "__main__":
    main()
