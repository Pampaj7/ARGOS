#!/usr/bin/env python3
"""Build the compact SCARED temporal-GT benchmark table.

This table intentionally mixes frame-based, video-native, and ARGOS refiner
methods only after applying the same GT-valid-frame filter.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


BASE_COLUMNS = [
    "Method",
    "Training / Checkpoint",
    "Input res.",
    "Depth MAE ↓",
    "Bad-2 mm ↓",
    "Disp. MAE ↓",
    "Temporal diff ↓",
    "Runtime ↓",
    "VRAM ↓",
    "Causal",
    "Frames with GT",
    "Notes",
]


METHOD_ALIASES = {
    "S2M2-S": ("S2M2-S@512", "official pretrained S", "512 px width", "yes", ""),
    "S2M2-L": ("S2M2-L@736", "official pretrained L", "736 px width", "yes", ""),
    "S2M2-L_full": ("S2M2-L full", "official pretrained L", "full resolution", "yes", ""),
    "S2M2-XL": ("S2M2-XL", "official pretrained XL", "full resolution", "yes", ""),
    "Fast-FoundationStereo_ONNX": (
        "Fast-FoundationStereo ONNX",
        "official ONNX checkpoint",
        "ONNX script default",
        "yes",
        "",
    ),
    "rtmonster_zeroshot": (
        "RT-MonSter++ zero-shot",
        "official zero-shot checkpoint",
        "native adapter",
        "yes",
        "",
    ),
    "stereoanywhere": (
        "StereoAnywhere",
        "official checkpoint",
        "native adapter",
        "yes",
        "",
    ),
    "monster_mixall": (
        "MonSter++ MixAll",
        "official MixAll checkpoint",
        "native adapter",
        "yes",
        "",
    ),
    "raft_middlebury": (
        "RAFT-Stereo Middlebury",
        "official Middlebury checkpoint",
        "native adapter",
        "yes",
        "",
    ),
    "defom_vitl_eth3d": (
        "DEFOM-Stereo ViT-L ETH3D",
        "official ETH3D checkpoint",
        "native adapter",
        "yes",
        "",
    ),
    "crestereo": (
        "CREStereo",
        "official checkpoint",
        "native adapter",
        "yes",
        "",
    ),
    "SGBM": (
        "SGBM",
        "OpenCV classical baseline",
        "full resolution",
        "yes",
        "Fragile classical baseline; about one quarter of pixels are excluded by the positive-disparity filter.",
    ),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BASE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in BASE_COLUMNS})


def parse_frame_id(frame: str) -> str:
    tail = frame.rsplit("_frame_", 1)[-1]
    return tail[:6]


def mean(values: list[float]) -> float:
    values = [v for v in values if math.isfinite(v)]
    return float(sum(values) / len(values)) if values else float("nan")


def fmt(value: object) -> str:
    if value == "" or value is None:
        return ""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(x):
        return ""
    if abs(x) >= 1000000:
        return f"{x:.3e}"
    if abs(x) >= 100:
        return f"{x:.2f}"
    return f"{x:.4f}"


def load_valid_metadata(metadata_csv: Path, min_valid_ratio: float) -> tuple[list[str], dict[str, Path]]:
    rows = read_csv(metadata_csv)
    valid_ids = [
        row["frame_id"]
        for row in rows
        if float(row.get("valid_pixel_ratio", "0") or 0) >= min_valid_ratio
    ]
    masks = {row["frame_id"]: Path(row["valid_mask_path"]) for row in rows}
    return valid_ids, masks


def temporal_diff_for_method(method_dir: Path, valid_ids: list[str], mask_paths: dict[str, Path]) -> float:
    diffs: list[float] = []
    for prev_id, cur_id in zip(valid_ids[:-1], valid_ids[1:]):
        prev_candidates = sorted(method_dir.glob(f"*_frame_{prev_id}_disp.npy"))
        cur_candidates = sorted(method_dir.glob(f"*_frame_{cur_id}_disp.npy"))
        if not prev_candidates or not cur_candidates:
            continue
        prev = np.load(prev_candidates[0]).astype(np.float32)
        cur = np.load(cur_candidates[0]).astype(np.float32)
        prev_mask = np.load(mask_paths[prev_id]).astype(bool)
        cur_mask = np.load(mask_paths[cur_id]).astype(bool)
        valid = prev_mask & cur_mask & np.isfinite(prev) & np.isfinite(cur) & (prev > 0.1) & (cur > 0.1)
        if valid.any():
            diffs.append(float(np.mean(np.abs(cur[valid] - prev[valid]))))
    return mean(diffs)


def aggregate_frame_method(
    method_dir: Path,
    valid_ids: set[str],
    ordered_valid_ids: list[str],
    mask_paths: dict[str, Path],
) -> dict[str, object] | None:
    metrics_path = method_dir / "metrics.csv"
    if not metrics_path.exists():
        return None
    rows = [row for row in read_csv(metrics_path) if parse_frame_id(row["frame"]) in valid_ids]
    if not rows:
        return None

    method_name, checkpoint, input_res, causal, note = METHOD_ALIASES.get(
        method_dir.name,
        (
            rows[0].get("method", method_dir.name),
            rows[0].get("checkpoint", ""),
            rows[0].get("input_resolution", ""),
            "yes",
            "",
        ),
    )
    temporal_diff = temporal_diff_for_method(method_dir, ordered_valid_ids, mask_paths)

    return {
        "Method": method_name,
        "Training / Checkpoint": checkpoint,
        "Input res.": input_res,
        "Depth MAE ↓": mean([float(row["valid_disp_depth_mae_mm"]) for row in rows]),
        "Bad-2 mm ↓": mean([float(row["valid_disp_depth_bad2mm_pct"]) for row in rows]),
        "Disp. MAE ↓": mean([float(row["valid_disp_mae_px"]) for row in rows]),
        "Temporal diff ↓": temporal_diff,
        "Runtime ↓": mean([float(row.get("runtime_ms", "nan") or "nan") for row in rows]),
        "VRAM ↓": "",
        "Causal": causal,
        "Frames with GT": len(rows),
        "Notes": note,
    }


def load_existing_temporal_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in read_csv(path):
        row = dict(row)
        row.setdefault("Notes", "")
        rows.append(row)
    return rows


def markdown_table(rows: list[dict[str, object]]) -> str:
    out = []
    out.append("| " + " | ".join(BASE_COLUMNS) + " |")
    out.append("| " + " | ".join(["---"] * len(BASE_COLUMNS)) + " |")
    for row in rows:
        out.append("| " + " | ".join(fmt(row.get(col, "")) for col in BASE_COLUMNS) + " |")
    return "\n".join(out)


def sort_key(row: dict[str, object]) -> tuple[int, float, float]:
    note = str(row.get("Notes", ""))
    failed = 1 if "Fragile classical baseline" in note else 0
    try:
        depth = float(row.get("Depth MAE ↓", "nan"))
    except (TypeError, ValueError):
        depth = float("nan")
    if not math.isfinite(depth):
        depth = float("inf")
    try:
        temporal = float(row.get("Temporal diff ↓", "nan"))
    except (TypeError, ValueError):
        temporal = float("nan")
    if not math.isfinite(temporal):
        temporal = float("inf")
    return failed, depth, temporal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", type=Path, default=Path("dataset/SCARED/curated/temporal_gt/test_dataset_9_keyframe_3/metadata.csv"))
    parser.add_argument("--existing-temporal-csv", type=Path, default=Path("results/03_temporal_refinement/evaluation/temporal_evaluation.csv"))
    parser.add_argument("--frame-methods-dir", type=Path, default=Path("results/03_temporal_refinement/evaluation/frame_based_gt/native_frame_methods"))
    parser.add_argument("--out-csv", type=Path, default=Path("results/03_temporal_refinement/evaluation/temporal_evaluation.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("results/03_temporal_refinement/evaluation/temporal_evaluation.md"))
    parser.add_argument("--min-valid-ratio", type=float, default=0.20)
    args = parser.parse_args()

    ordered_valid_ids, mask_paths = load_valid_metadata(args.metadata_csv, args.min_valid_ratio)
    valid_ids = set(ordered_valid_ids)

    existing = load_existing_temporal_rows(args.existing_temporal_csv)
    temporal_method_prefixes = ("StereoAnyVideo", "ConvGRU", "Tiny U-Net")
    keep_existing = [
        row
        for row in existing
        if str(row["Method"]).startswith(temporal_method_prefixes)
    ]

    frame_rows: list[dict[str, object]] = []
    for method_dir in sorted(args.frame_methods_dir.iterdir()):
        if not method_dir.is_dir():
            continue
        row = aggregate_frame_method(method_dir, valid_ids, ordered_valid_ids, mask_paths)
        if row is not None:
            frame_rows.append(row)

    # Prefer the newly recomputed frame-based rows for S2M2-S/L, then append video/refiner rows.
    all_rows = sorted(frame_rows + keep_existing, key=sort_key)
    write_csv(args.out_csv, all_rows)

    args.out_md.write_text(
        "# SCARED temporal GT evaluation\n\n"
        "Protocol: `dataset/SCARED/curated/temporal_gt/test_dataset_9_keyframe_3`, "
        f"frames with GT valid-pixel ratio >= {args.min_valid_ratio:.2f}. Metrics are averaged over "
        f"{len(ordered_valid_ids)} valid-GT frames. Temporal diff is mean consecutive absolute disparity "
        "difference on the intersection of adjacent GT-valid masks and positive predicted disparity.\n\n"
        "Frame-based methods are run independently per frame; StereoAnyVideo and ARGOS refiners use temporal context. "
        "Lower is better for all numeric metric columns.\n\n"
        + markdown_table(all_rows)
        + "\n\n"
        "Caveat: temporal smoothness is not geometric correctness. SGBM is retained only as a fragile classical baseline.\n",
    )

    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_md}")
    print(f"Rows: {len(all_rows)}")


if __name__ == "__main__":
    main()
