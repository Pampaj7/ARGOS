#!/usr/bin/env python3
"""Batch convert extracted SCARED raw keyframes into rectified temporal GT.

The heavy lifting is delegated to
`convert_scared_keyframe_to_temporal_gt_rectified.py`, so the batch wrapper keeps
one source of truth for rectification. It discovers valid extracted keyframe
folders, skips unsafe/non-convertible inputs, and writes an audit trail for the
batch run.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = Path("dataset/SCARED/raw/extracted")
DEFAULT_OUTPUT_ROOT = Path("dataset/SCARED/curated/temporal_gt_rectified")
SINGLE_CONVERTER = ROOT / "scripts/dataset/convert_scared_keyframe_to_temporal_gt_rectified.py"
DEFAULT_TEST_DATASETS = {"dataset_8", "dataset_9", "test_dataset_8", "test_dataset_9"}


@dataclass(frozen=True)
class Candidate:
    dataset_name: str
    keyframe_id: str
    source_keyframe_path: Path
    output_sequence_path: Path
    sequence_name: str


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch convert SCARED raw temporal keyframes into rectified temporal GT.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", nargs="?", const=True, default=False, type=parse_bool)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-test-datasets", nargs="?", const=True, default=False, type=parse_bool)
    parser.add_argument("--save-debug", nargs="?", const=True, default=False, type=parse_bool)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def discover_keyframes(input_root: Path, output_root: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    if not input_root.exists():
        raise FileNotFoundError(input_root)
    for dataset_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        nested = dataset_dir / dataset_dir.name
        if not nested.is_dir():
            continue
        for keyframe_dir in sorted(nested.glob("keyframe_*")):
            if not keyframe_dir.is_dir():
                continue
            sequence_name = f"{dataset_dir.name}_{keyframe_dir.name}"
            candidates.append(
                Candidate(
                    dataset_name=dataset_dir.name,
                    keyframe_id=keyframe_dir.name,
                    source_keyframe_path=keyframe_dir,
                    output_sequence_path=output_root / sequence_name,
                    sequence_name=sequence_name,
                )
            )
    return candidates


def missing_requirements(candidate: Candidate) -> list[str]:
    keyframe = candidate.source_keyframe_path
    required = [
        keyframe / "data" / "rgb.mp4",
        keyframe / "data" / "frame_data.tar.gz",
        keyframe / "data" / "scene_points.tar.gz",
        keyframe / "Left_Image.png",
        keyframe / "Right_Image.png",
    ]
    missing = [str(path.relative_to(keyframe)) for path in required if not path.exists()]
    if not (keyframe / "endoscope_calibration.yaml").exists() and not (keyframe / "data" / "frame_data.tar.gz").exists():
        missing.append("endoscope_calibration.yaml_or_per_frame_frame_data")
    return missing


def empty_stats_row(candidate: Candidate, status: str, error_message: str = "") -> dict[str, Any]:
    return {
        "dataset_name": candidate.dataset_name,
        "keyframe_id": candidate.keyframe_id,
        "source_keyframe_path": str(candidate.source_keyframe_path),
        "output_sequence_path": str(candidate.output_sequence_path),
        "status": status,
        "num_frames": "",
        "valid_pixel_pct_mean": "",
        "valid_pixel_pct_min": "",
        "valid_pixel_pct_max": "",
        "depth_median_mean": "",
        "disparity_median_mean": "",
        "baseline_mean": "",
        "fx_mean": "",
        "warning_count": "",
        "error_message": error_message,
    }


def load_summary_stats(candidate: Candidate, status: str, error_message: str = "") -> dict[str, Any]:
    row = empty_stats_row(candidate, status, error_message)
    summary_path = candidate.output_sequence_path / "summary.json"
    metadata_path = candidate.output_sequence_path / "metadata.csv"
    if not summary_path.exists():
        return row
    summary = json.loads(summary_path.read_text())
    valid_values: list[float] = []
    if metadata_path.exists():
        with metadata_path.open() as f:
            for metadata_row in csv.DictReader(f):
                value = metadata_row.get("valid_pixel_ratio")
                if value not in (None, ""):
                    valid_values.append(float(value) * 100.0)
    warnings = summary.get("warnings", [])
    row.update(
        {
            "num_frames": summary.get("num_processed_frames", ""),
            "valid_pixel_pct_mean": summary.get("valid_pixel_ratio_mean", "") * 100.0 if summary.get("valid_pixel_ratio_mean") is not None else "",
            "valid_pixel_pct_min": min(valid_values) if valid_values else "",
            "valid_pixel_pct_max": max(valid_values) if valid_values else "",
            "depth_median_mean": summary.get("depth_median_mean", ""),
            "disparity_median_mean": summary.get("disparity_median_mean", ""),
            "baseline_mean": summary.get("baseline_mean", ""),
            "fx_mean": summary.get("fx_mean", ""),
            "warning_count": len(warnings) if isinstance(warnings, list) else "",
        }
    )
    return row


def run_single_converter(candidate: Candidate, overwrite: bool, max_frames: int, save_debug: bool) -> dict[str, Any]:
    if candidate.output_sequence_path.exists() and not overwrite:
        return load_summary_stats(candidate, "already_exists", "output exists; pass --overwrite true to regenerate")
    cmd = [
        sys.executable,
        str(SINGLE_CONVERTER),
        "--keyframe-path",
        str(candidate.source_keyframe_path),
        "--output-dir",
        str(candidate.output_sequence_path),
        "--overwrite",
        "true" if overwrite else "false",
        "--save-debug",
        "true" if save_debug else "false",
        "--defer-scene-count",
        "true",
    ]
    if max_frames > 0:
        cmd.extend(["--max-frames", str(max_frames)])
    started = time.perf_counter()
    completed = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        message = (completed.stderr.strip() or completed.stdout.strip()).splitlines()[-1:] or ["converter failed"]
        row = empty_stats_row(candidate, "failed", message[0])
        row["runtime_sec"] = elapsed
        return row
    row = load_summary_stats(candidate, "converted")
    row["runtime_sec"] = elapsed
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "dataset_name",
        "keyframe_id",
        "source_keyframe_path",
        "output_sequence_path",
        "status",
        "num_frames",
        "valid_pixel_pct_mean",
        "valid_pixel_pct_min",
        "valid_pixel_pct_max",
        "depth_median_mean",
        "disparity_median_mean",
        "baseline_mean",
        "fx_mean",
        "warning_count",
        "error_message",
        "runtime_sec",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in columns} for row in rows])


def write_readme(path: Path, summary: dict[str, Any], available: list[str], skip_reasons: dict[str, int]) -> None:
    lines = [
        "# Batch Rectified SCARED Temporal-GT Conversion",
        "",
        f"Input root: `{summary['input_root']}`",
        f"Output root: `{summary['output_root']}`",
        "",
        f"1. Keyframes discovered: `{summary['discovered_keyframes']}`",
        f"2. Convertible keyframes: `{summary['convertible_keyframes']}`",
        f"3. Converted/available successfully: `{summary['successful_or_available']}` (`{summary['converted']}` newly converted in this run, `{summary['already_exists']}` already existed)",
        f"4. Skipped/non-converted: `{summary['non_converted']}`",
        "",
        "## Skip Reasons",
        "",
    ]
    if skip_reasons:
        lines.extend([f"- `{reason}`: `{count}`" for reason, count in sorted(skip_reasons.items())])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Available For Multi-Sequence Temporal-GT Evaluation",
            "",
        ]
    )
    lines.extend([f"- `{name}`" for name in available] or ["- none"])
    lines.extend(
        [
            "",
            "## Reminder",
            "",
            "Rectification is required for S2M2/RAFT-style stereo evaluation. Raw top/bottom stereo splits may have plausible-looking depth overlays, but their disparities are not in the rectified horizontal-disparity coordinate system expected by stereo models.",
            "",
            "This batch converter does not run S2M2, RAFT, RAFT-Small, StereoAnyVideo, or temporal benchmarks.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if args.max_frames < 0:
        raise ValueError("--max-frames must be >= 0")
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    args.output_root.mkdir(parents=True, exist_ok=True)
    discovered = discover_keyframes(args.input_root, args.output_root)
    selected = discovered[: args.limit] if args.limit > 0 else discovered

    rows: list[dict[str, Any]] = []
    convert_jobs: list[Candidate] = []
    for candidate in selected:
        if candidate.dataset_name in DEFAULT_TEST_DATASETS and not args.include_test_datasets:
            rows.append(empty_stats_row(candidate, "skipped", "test dataset skipped by default"))
            continue
        missing = missing_requirements(candidate)
        if missing:
            rows.append(empty_stats_row(candidate, "skipped", "missing required files: " + ", ".join(missing)))
            continue
        convert_jobs.append(candidate)

    if args.workers == 1:
        for candidate in convert_jobs:
            action = "already exists" if candidate.output_sequence_path.exists() and not args.overwrite else "converting"
            print(f"{action} {candidate.sequence_name}", flush=True)
            rows.append(run_single_converter(candidate, bool(args.overwrite), args.max_frames, bool(args.save_debug)))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_single_converter, candidate, bool(args.overwrite), args.max_frames, bool(args.save_debug)): candidate
                for candidate in convert_jobs
            }
            for future in as_completed(futures):
                candidate = futures[future]
                print(f"finished {candidate.sequence_name}", flush=True)
                rows.append(future.result())

    rows.sort(key=lambda row: (row["dataset_name"], row["keyframe_id"]))
    converted_rows = [row for row in rows if row["status"] == "converted"]
    non_converted_rows = [row for row in rows if row["status"] in {"skipped", "failed"}]
    available = [
        row["output_sequence_path"]
        for row in rows
        if row["status"] in {"converted", "already_exists"} and Path(row["output_sequence_path"], "summary.json").exists()
    ]
    skip_reasons: dict[str, int] = {}
    for row in non_converted_rows:
        reason = row.get("error_message") or row["status"]
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    summary = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "overwrite": bool(args.overwrite),
        "max_frames": args.max_frames,
        "limit": args.limit,
        "include_test_datasets": bool(args.include_test_datasets),
        "save_debug": bool(args.save_debug),
        "workers": args.workers,
        "discovered_keyframes": len(discovered),
        "selected_keyframes": len(selected),
        "convertible_keyframes": len(convert_jobs),
        "converted": len(converted_rows),
        "already_exists": sum(row["status"] == "already_exists" for row in rows),
        "skipped": sum(row["status"] == "skipped" for row in rows),
        "failed": sum(row["status"] == "failed" for row in rows),
        "non_converted": len(non_converted_rows),
        "successful_or_available": len(converted_rows) + sum(row["status"] == "already_exists" for row in rows),
        "available_sequences": available,
        "skip_reasons": skip_reasons,
    }
    write_csv(args.output_root / "converted_sequences.csv", [row for row in rows if row["status"] in {"converted", "already_exists"}])
    write_csv(args.output_root / "skipped_sequences.csv", non_converted_rows)
    (args.output_root / "batch_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_readme(args.output_root / "README.md", summary, available, skip_reasons)
    (args.output_root / "run.log").write_text(
        "\n".join(
            [
                "Batch rectified SCARED temporal-GT conversion",
                f"input_root={args.input_root}",
                f"output_root={args.output_root}",
                f"discovered={len(discovered)}",
                f"selected={len(selected)}",
                f"convertible={len(convert_jobs)}",
                f"converted={len(converted_rows)}",
                f"already_exists={summary['already_exists']}",
                f"skipped={summary['skipped']}",
                f"failed={summary['failed']}",
            ]
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
