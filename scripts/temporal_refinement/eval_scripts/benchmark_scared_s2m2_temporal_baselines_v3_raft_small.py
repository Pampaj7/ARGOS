#!/usr/bin/env python3
"""Evaluate RAFT-Small flow caches for S2M2-S temporal refinement.

This script is cache-only for stereo predictions: it never runs S2M2-S or
StereoAnyVideo. It compares RAFT-Small warped EMA against fixed EMA, the best
previous no-RAFT adaptive baseline, full RAFT warped EMA, and StereoAnyVideo on
the audited SCARED temporal-GT sequence.
"""

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
from temporal_baselines import (  # noqa: E402
    adaptive_no_raft_diff_grad_sequence,
    adaptive_no_raft_diff_sequence,
    fixed_ema_sequence,
    raft_warped_ema_sequence,
    warp_disparity_numpy,
)
from video_qualitative import colorize_scalar, make_board, write_mp4  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("results/03_temporal_refinement/scared_s2m2_temporal_baselines_v3_raft_small")
DEFAULT_FULL_FLOW_CACHE = Path("results/04_dataset_derivatives/SCARED/temporal_gt_flow_cache/test_dataset_9_keyframe_3/raft")
DEFAULT_SMALL6_FLOW_CACHE = Path("results/04_dataset_derivatives/SCARED/temporal_gt_flow_cache/test_dataset_9_keyframe_3/raft_small_6")
DEFAULT_SMALL12_FLOW_CACHE = Path("results/04_dataset_derivatives/SCARED/temporal_gt_flow_cache/test_dataset_9_keyframe_3/raft_small_12")
DEFAULT_PREVIOUS_BENCHMARK_DIR = Path("results/03_temporal_refinement/scared_s2m2_temporal_baselines_v2_no_raft_adaptive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SCARED S2M2-S RAFT-Small temporal baseline benchmark.")
    parser.add_argument("--sequence-dir", type=Path, default=base.DEFAULT_SEQUENCE_DIR)
    parser.add_argument("--s2m2-cache-dir", type=Path, default=base.DEFAULT_S2M2_CACHE)
    parser.add_argument("--sav-cache-dir", type=Path, default=base.DEFAULT_SAV_CACHE)
    parser.add_argument("--full-flow-cache-dir", type=Path, default=DEFAULT_FULL_FLOW_CACHE)
    parser.add_argument("--raft-small-6-flow-cache-dir", type=Path, default=DEFAULT_SMALL6_FLOW_CACHE)
    parser.add_argument("--raft-small-12-flow-cache-dir", type=Path, default=DEFAULT_SMALL12_FLOW_CACHE)
    parser.add_argument("--previous-benchmark-dir", type=Path, default=DEFAULT_PREVIOUS_BENCHMARK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-valid-ratio", type=float, default=0.2)
    parser.add_argument("--overwrite", nargs="?", const=True, default=False, type=base.parse_bool)
    parser.add_argument("--warp-device", default="auto")
    return parser.parse_args()


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def previous_best_no_raft_config(previous_dir: Path) -> dict[str, Any]:
    config_path = previous_dir / "method_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing previous benchmark method_config.json: {config_path}")
    data = json.loads(config_path.read_text())
    best = data.get("best_no_raft_adaptive_by_depth_mae")
    if not best:
        raise RuntimeError(f"Previous benchmark has no best_no_raft_adaptive_by_depth_mae: {config_path}")
    method_id = str(best["method_id"])
    for config in data.get("no_raft_adaptive_sweep_configs", []):
        if str(config.get("method_id")) == method_id:
            return config
    raise RuntimeError(f"Could not find previous best config {method_id} in {config_path}")


def build_previous_best_no_raft_method(
    config: dict[str, Any],
    raw: Sequence[np.ndarray],
    inherited_runtime_ms: float,
    inherited_vram_mb: float,
) -> base.MethodRecord:
    method_type = str(config["method_type"])
    if method_type == "adaptive_no_raft_diff":
        result = adaptive_no_raft_diff_sequence(
            raw,
            alpha_min=float(config["alpha_min"]),
            alpha_max=float(config["alpha_max"]),
            diff_scale_px=float(config["diff_scale_px"]),
        )
    elif method_type == "adaptive_no_raft_diff_grad":
        result = adaptive_no_raft_diff_grad_sequence(
            raw,
            alpha_min=float(config["alpha_min"]),
            alpha_max=float(config["alpha_max"]),
            diff_scale_px=float(config["diff_scale_px"]),
            grad_scale_px=float(config["grad_scale_px"]),
            w_diff=float(config["w_diff"]),
            w_grad=float(config["w_grad"]),
        )
    else:
        raise ValueError(method_type)
    params = {
        key: float(config[key])
        for key in ["alpha_min", "alpha_max", "diff_scale_px", "grad_scale_px", "w_diff", "w_grad"]
        if key in config and math.isfinite(float(config[key]))
    }
    return base.MethodRecord(
        method_id="best_no_raft_adaptive",
        method_name=f"Best previous no-RAFT adaptive ({config['method_id']})",
        method_type=method_type,
        predictions=result.predictions,
        postprocess_ms=result.postprocess_ms_per_frame,
        inherited_runtime_ms=inherited_runtime_ms,
        inherited_vram_mb=inherited_vram_mb,
        notes="Selected from v2 by depth MAE; prediction formula does not use optical flow.",
        params=params,
    )


def cache_audit_named(
    frames: Sequence[base.FrameRecord],
    s2m2_cache_dir: Path,
    sav_cache_dir: Path,
    flow_cache_dir: Path,
    gt_shape: tuple[int, int],
) -> dict[str, Any]:
    audit = base.cache_audit(frames, s2m2_cache_dir, sav_cache_dir, flow_cache_dir, gt_shape)
    summary_path = flow_cache_dir / "flow_cache_summary.json"
    manifest_path = flow_cache_dir / "flow_cache_manifest.csv"
    failed_pairs: list[dict[str, str]] = []
    if manifest_path.exists():
        failed_pairs = [row for row in load_csv_rows(manifest_path) if row.get("status") != "ok"]
    audit.update(
        {
            "flow_cache_summary_path": str(summary_path),
            "flow_cache_manifest_path": str(manifest_path),
            "flow_cache_summary_exists": summary_path.exists(),
            "flow_cache_manifest_exists": manifest_path.exists(),
            "failed_pairs": failed_pairs,
            "failed_pair_count": len(failed_pairs),
        }
    )
    if summary_path.exists():
        data = json.loads(summary_path.read_text())
        audit["builder_cache_complete"] = bool(data.get("cache_complete", False))
        audit["average_forward_runtime_ms"] = data.get("average_forward_runtime_ms")
        audit["average_backward_runtime_ms"] = data.get("average_backward_runtime_ms")
        audit["peak_vram_mb"] = data.get("peak_vram_mb")
        audit["raft_iters"] = data.get("raft_iters")
        audit["raft_small"] = data.get("raft_small")
    return audit


def summarize_flow_cache(label: str, audit: dict[str, Any]) -> str:
    return (
        f"- {label}: complete={audit.get('complete')} builder_complete={audit.get('builder_cache_complete')} "
        f"forward={audit.get('forward_flow_count')}/{audit.get('expected_pairs')} "
        f"failed_pairs={audit.get('failed_pair_count')} "
        f"fwd_ms={float(audit.get('average_forward_runtime_ms', math.nan)):.4f} "
        f"bwd_ms={float(audit.get('average_backward_runtime_ms', math.nan)):.4f} "
        f"peak_vram_mb={float(audit.get('peak_vram_mb', math.nan)):.2f}"
    )


def build_methods(
    frames: Sequence[base.FrameRecord],
    s2m2_raw: list[np.ndarray],
    sav: list[np.ndarray],
    s2m2_runtime: float,
    s2m2_vram: float,
    sav_runtime: float,
    sav_vram: float,
    small6_flow_cache: Path,
    small12_flow_cache: Path,
    full_flow_cache: Path,
    previous_best_config: dict[str, Any],
    warp_device: str,
) -> tuple[list[base.MethodRecord], dict[str, dict[str, float | str]]]:
    frame_ids = [frame.frame_id for frame in frames]
    runtimes = {
        "none": {
            "average_forward_runtime_ms": math.nan,
            "average_backward_runtime_ms": math.nan,
            "peak_vram_mb": math.nan,
            "source": "not_used",
        },
        "raft_small_6": base.load_flow_runtime_metadata(small6_flow_cache),
        "raft_small_12": base.load_flow_runtime_metadata(small12_flow_cache),
        "raft_full": base.load_flow_runtime_metadata(full_flow_cache),
    }
    methods: list[base.MethodRecord] = [
        base.MethodRecord(
            "s2m2_s_raw",
            "S2M2-S raw",
            "cached_prediction",
            s2m2_raw,
            inherited_runtime_ms=s2m2_runtime,
            inherited_vram_mb=s2m2_vram,
        )
    ]
    fixed = fixed_ema_sequence(s2m2_raw, 0.35)
    methods.append(
        base.MethodRecord(
            "s2m2_s_fixed_ema_a0.35",
            "S2M2-S fixed EMA alpha=0.35",
            "fixed_ema",
            fixed.predictions,
            alpha=0.35,
            postprocess_ms=fixed.postprocess_ms_per_frame,
            inherited_runtime_ms=s2m2_runtime,
            inherited_vram_mb=s2m2_vram,
        )
    )
    methods.append(build_previous_best_no_raft_method(previous_best_config, s2m2_raw, s2m2_runtime, s2m2_vram))
    for method_id, name, flow_dir in [
        ("s2m2_s_raft_small_6_warped_ema_a0.50", "S2M2-S RAFT-Small 6-iters warped EMA alpha=0.50", small6_flow_cache),
        ("s2m2_s_raft_small_12_warped_ema_a0.50", "S2M2-S RAFT-Small 12-iters warped EMA alpha=0.50", small12_flow_cache),
        ("s2m2_s_raft_full_warped_ema_a0.50", "S2M2-S full RAFT warped EMA alpha=0.50", full_flow_cache),
    ]:
        flow_loader = lambda prev_id, cur_id, flow_dir=flow_dir: base.load_forward_flow(flow_dir, prev_id, cur_id)
        result = raft_warped_ema_sequence(s2m2_raw, frame_ids, flow_loader, 0.50, warp_device=warp_device)
        methods.append(
            base.MethodRecord(
                method_id,
                name,
                "raft_warped_ema",
                result.predictions,
                alpha=0.50,
                postprocess_ms=result.postprocess_ms_per_frame,
                inherited_runtime_ms=s2m2_runtime,
                inherited_vram_mb=s2m2_vram,
            )
        )
    methods.append(
        base.MethodRecord(
            "stereoanyvideo",
            "StereoAnyVideo",
            "cached_prediction",
            sav,
            inherited_runtime_ms=sav_runtime,
            inherited_vram_mb=sav_vram,
            role="teacher_or_comparison_not_gt",
            notes="Cached comparison/teacher; not ground truth.",
        )
    )
    runtime_by_method = {
        "s2m2_s_raw": runtimes["none"],
        "s2m2_s_fixed_ema_a0.35": runtimes["none"],
        "best_no_raft_adaptive": runtimes["none"],
        "s2m2_s_raft_small_6_warped_ema_a0.50": runtimes["raft_small_6"],
        "s2m2_s_raft_small_12_warped_ema_a0.50": runtimes["raft_small_12"],
        "s2m2_s_raft_full_warped_ema_a0.50": runtimes["raft_full"],
        "stereoanyvideo": runtimes["none"],
    }
    return methods, runtime_by_method


def evaluate_methods(
    methods: Sequence[base.MethodRecord],
    frames: Sequence[base.FrameRecord],
    metric_indices: Sequence[int],
    metric_pair_indices: Sequence[int],
    metric_flow_cache: Path,
    runtime_by_method: dict[str, dict[str, float | str]],
    warp_device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for method in methods:
        summary, method_frames, method_pairs = base.evaluate_method(
            method,
            frames,
            metric_indices,
            metric_pair_indices,
            metric_flow_cache,
            warp_device,
            runtime_by_method[method.method_id],
        )
        summary_rows.append(summary)
        frame_rows.extend(method_frames)
        pair_rows.extend(method_pairs)
    return summary_rows, frame_rows, pair_rows


def read_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    return np.array(Image.open(path).convert("RGB"))


def sampled_disparity_vmax(frames: Sequence[base.FrameRecord], methods: Sequence[base.MethodRecord]) -> float:
    values: list[np.ndarray] = []
    for idx in range(0, len(frames), max(len(frames) // 12, 1)):
        values.append(np.load(frames[idx].gt_disp_path, allow_pickle=False).astype(np.float32)[::8, ::8].ravel())
        for method in methods:
            values.append(method.predictions[idx][::8, ::8].ravel())
    merged = np.concatenate([v[np.isfinite(v)] for v in values if v.size])
    return float(np.nanpercentile(merged, 99)) if merged.size else 1.0


def generate_videos(
    output_dir: Path,
    frames: Sequence[base.FrameRecord],
    methods: Sequence[base.MethodRecord],
) -> list[Path]:
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    disp_vmax = sampled_disparity_vmax(frames, methods)
    frame_indices = list(range(0, len(frames), 2))
    if frame_indices[-1] != len(frames) - 1:
        frame_indices.append(len(frames) - 1)

    def rgb_at(idx: int) -> np.ndarray:
        return read_rgb(frames[idx].left_path)

    def gt_at(idx: int) -> np.ndarray:
        return np.load(frames[idx].gt_disp_path, allow_pickle=False).astype(np.float32)

    def disparity_frames():
        for idx in frame_indices:
            tiles = [("RGB", rgb_at(idx)), ("GT disparity", colorize_scalar(gt_at(idx), 0.0, disp_vmax))]
            for method in methods:
                tiles.append((method.method_name, colorize_scalar(method.predictions[idx], 0.0, disp_vmax)))
            yield make_board(tiles, panel_size=(240, 192), cols=4)

    path = videos_dir / "disparity_comparison_board.mp4"
    write_mp4(path, disparity_frames(), fps=10)
    paths.append(path)

    def error_frames():
        for idx in frame_indices:
            frame = frames[idx]
            gt = gt_at(idx)
            valid = base.read_mask(frame.valid_mask_path)
            tiles = [("RGB", rgb_at(idx))]
            for method in methods:
                err = np.abs(method.predictions[idx] - gt)
                err[~valid] = np.nan
                tiles.append((f"{method.method_name} abs err", colorize_scalar(err, 0.0, 10.0)))
            yield make_board(tiles, panel_size=(240, 192), cols=4)

    path = videos_dir / "error_comparison_board.mp4"
    write_mp4(path, error_frames(), fps=10)
    paths.append(path)

    def temporal_frames():
        for idx in [i for i in frame_indices if i > 0]:
            tiles = [("RGB", rgb_at(idx))]
            for method in methods:
                diff = np.abs(method.predictions[idx] - method.predictions[idx - 1])
                tiles.append((f"{method.method_name} |dt|", colorize_scalar(diff, 0.0, 5.0)))
            yield make_board(tiles, panel_size=(240, 192), cols=4)

    path = videos_dir / "temporal_difference_board.mp4"
    write_mp4(path, temporal_frames(), fps=10)
    paths.append(path)
    return paths


def row_by_id(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["method_id"]): row for row in rows}


def fmt(row: dict[str, Any] | None) -> str:
    if row is None:
        return "missing"
    return (
        f"depth={float(row['depth_mae_mm']):.4f} mm, "
        f"disp={float(row['disp_mae_px']):.4f} px, "
        f"mc={float(row['motion_compensated_temporal_mae_px']):.4f} px, "
        f"online={float(row['online_runtime_estimated_ms']):.4f} ms, "
        f"vram={float(row['online_peak_vram_estimated_mb']):.2f} MB"
    )


def relative_gain(raw: dict[str, Any], target: dict[str, Any], full: dict[str, Any], key: str) -> float:
    raw_v = float(raw[key])
    target_v = float(target[key])
    full_v = float(full[key])
    denom = raw_v - full_v
    if not math.isfinite(denom) or abs(denom) < 1e-9:
        return math.nan
    return 100.0 * (raw_v - target_v) / denom


def write_readme(
    path: Path,
    args: argparse.Namespace,
    summary_rows: Sequence[dict[str, Any]],
    flow_audit: dict[str, Any],
    previous_best_config: dict[str, Any],
    video_paths: Sequence[Path],
) -> None:
    rows = row_by_id(summary_rows)
    fixed = rows.get("s2m2_s_fixed_ema_a0.35")
    small6 = rows.get("s2m2_s_raft_small_6_warped_ema_a0.50")
    small12 = rows.get("s2m2_s_raft_small_12_warped_ema_a0.50")
    full = rows.get("s2m2_s_raft_full_warped_ema_a0.50")
    sav = rows.get("stereoanyvideo")
    raw = rows.get("s2m2_s_raw")
    gain6 = relative_gain(raw, small6, full, "depth_mae_mm") if raw and small6 and full else math.nan
    gain12 = relative_gain(raw, small12, full, "depth_mae_mm") if raw and small12 and full else math.nan
    viable = bool(
        small12
        and full
        and fixed
        and float(small12["depth_mae_mm"]) < float(fixed["depth_mae_mm"])
        and float(small12["online_runtime_estimated_ms"]) < float(full["online_runtime_estimated_ms"])
    )
    conclusion = (
        "RAFT-Small is a viable lightweight motion-compensation backend in this run: "
        "the 12-iteration cache improves over fixed EMA while reducing estimated online runtime versus full RAFT."
        if viable
        else
        "RAFT-Small is not yet a clearly viable lightweight backend in this run: it does not simultaneously beat fixed EMA quality and full-RAFT estimated runtime."
    )
    readme = f"""# SCARED S2M2 Temporal Baselines v3 RAFT-Small

Cache-only benchmark on `{args.sequence_dir}`.

- S2M2-S cache: `{args.s2m2_cache_dir}`
- StereoAnyVideo cache: `{args.sav_cache_dir}`
- Full RAFT reference/metric flow cache: `{args.full_flow_cache_dir}`
- RAFT-Small 6-iters flow cache: `{args.raft_small_6_flow_cache_dir}`
- RAFT-Small 12-iters flow cache: `{args.raft_small_12_flow_cache_dir}`
- Previous best no-RAFT adaptive source: `{args.previous_benchmark_dir}`
- Previous best no-RAFT adaptive selected by depth MAE: `{previous_best_config['method_id']}`

No S2M2-S or StereoAnyVideo inference is rerun. Dataset files are not modified. Motion-compensated temporal metrics use the full RAFT flow cache as the common reference flow.

## Flow Cache Audit

{summarize_flow_cache('full RAFT reference', flow_audit['raft_full'])}
{summarize_flow_cache('RAFT-Small 6', flow_audit['raft_small_6'])}
{summarize_flow_cache('RAFT-Small 12', flow_audit['raft_small_12'])}

## Required Comparisons

- fixed EMA alpha=0.35: {fmt(fixed)}
- RAFT-Small 6 warped EMA alpha=0.50: {fmt(small6)}; depth gain recovered vs full RAFT={gain6:.2f}%
- RAFT-Small 12 warped EMA alpha=0.50: {fmt(small12)}; depth gain recovered vs full RAFT={gain12:.2f}%
- full RAFT warped EMA alpha=0.50: {fmt(full)}
- StereoAnyVideo: {fmt(sav)}

StereoAnyVideo remains the cached video-model comparison/teacher, not ground truth.

## Conclusion

{conclusion}

## Outputs

- `summary.csv`
- `per_frame_metrics.csv`
- `per_pair_temporal_metrics.csv`
- `flow_cache_audit.json`
- `method_config.json`
- `run.log`
- `videos/disparity_comparison_board.mp4`
- `videos/error_comparison_board.mp4`
- `videos/temporal_difference_board.mp4`

Generated videos:

{chr(10).join(f'- `{path.relative_to(args.output_dir)}`' for path in video_paths)}
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
        "raft_full": cache_audit_named(frames, args.s2m2_cache_dir, args.sav_cache_dir, args.full_flow_cache_dir, gt_shape),
        "raft_small_6": cache_audit_named(frames, args.s2m2_cache_dir, args.sav_cache_dir, args.raft_small_6_flow_cache_dir, gt_shape),
        "raft_small_12": cache_audit_named(frames, args.s2m2_cache_dir, args.sav_cache_dir, args.raft_small_12_flow_cache_dir, gt_shape),
    }
    base.write_json(args.output_dir / "flow_cache_audit.json", flow_audit)
    incomplete = [name for name, audit in flow_audit.items() if not audit["complete"]]
    if incomplete:
        raise RuntimeError(f"Flow/prediction cache audit failed for {incomplete}; see {args.output_dir / 'flow_cache_audit.json'}")

    s2m2_runtime, s2m2_vram = base.load_metadata_runtime(args.s2m2_cache_dir)
    sav_runtime, sav_vram = base.load_metadata_runtime(args.sav_cache_dir)
    s2m2_raw = base.load_prediction_sequence(args.s2m2_cache_dir, frames)
    sav = base.load_prediction_sequence(args.sav_cache_dir, frames)
    previous_best_config = previous_best_no_raft_config(args.previous_benchmark_dir)
    methods, runtime_by_method = build_methods(
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
    summary_rows, frame_rows, pair_rows = evaluate_methods(
        methods,
        frames,
        metric_indices,
        metric_pair_indices,
        args.full_flow_cache_dir,
        runtime_by_method,
        args.warp_device,
    )
    base.write_csv(args.output_dir / "summary.csv", summary_rows, base.SUMMARY_COLUMNS)
    base.write_csv(args.output_dir / "per_frame_metrics.csv", frame_rows, base.PER_FRAME_COLUMNS)
    base.write_csv(args.output_dir / "per_pair_temporal_metrics.csv", pair_rows, base.PER_PAIR_COLUMNS)
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
            "previous_benchmark_dir": str(args.previous_benchmark_dir),
            "previous_best_no_raft_config": previous_best_config,
            "runtime_by_method": runtime_by_method,
            "methods": [
                {
                    "method_id": method.method_id,
                    "method_name": method.method_name,
                    "method_type": method.method_type,
                    "alpha": method.alpha,
                    "role": method.role,
                    "notes": method.notes,
                    "params": method.params or {},
                }
                for method in methods
            ],
        },
    )
    video_paths = generate_videos(args.output_dir, frames, methods)
    write_readme(args.output_dir / "README.md", args, summary_rows, flow_audit, previous_best_config, video_paths)
    elapsed = time.perf_counter() - start
    rows = row_by_id(summary_rows)
    (args.output_dir / "run.log").write_text(
        "\n".join(
            [
                "Cache-only SCARED S2M2 RAFT-Small temporal baseline benchmark",
                f"output_dir={args.output_dir}",
                f"num_frames={len(frames)}",
                f"metric_frame_count={len(metric_indices)}",
                f"metric_pair_count={len(metric_pair_indices)}",
                f"method_count={len(methods)}",
                f"cache_audit_complete={all(audit['complete'] for audit in flow_audit.values())}",
                f"raft_small_6={fmt(rows.get('s2m2_s_raft_small_6_warped_ema_a0.50'))}",
                f"raft_small_12={fmt(rows.get('s2m2_s_raft_small_12_warped_ema_a0.50'))}",
                f"raft_full={fmt(rows.get('s2m2_s_raft_full_warped_ema_a0.50'))}",
                f"videos={','.join(str(path) for path in video_paths)}",
                f"elapsed_seconds={elapsed:.3f}",
            ]
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "summary_csv": str(args.output_dir / "summary.csv"),
                "flow_cache_audit_json": str(args.output_dir / "flow_cache_audit.json"),
                "method_count": len(methods),
                "metric_frame_count": len(metric_indices),
                "metric_pair_count": len(metric_pair_indices),
                "raft_small_6": rows.get("s2m2_s_raft_small_6_warped_ema_a0.50"),
                "raft_small_12": rows.get("s2m2_s_raft_small_12_warped_ema_a0.50"),
                "raft_full": rows.get("s2m2_s_raft_full_warped_ema_a0.50"),
                "video_files": [str(path) for path in video_paths],
                "elapsed_seconds": elapsed,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
