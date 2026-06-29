#!/usr/bin/env python3
"""Build a RAFT optical-flow cache for the audited SCARED temporal-GT sequence.

The cache contains adjacent forward/backward optical flow, confidence maps, and
forward occlusion masks. It is intended for no-training temporal baselines and
motion-compensated temporal metrics. It never runs S2M2 or StereoAnyVideo.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
LIB_DIR = ROOT / "scripts" / "temporal_refinement" / "lib"
sys.path.insert(0, str(LIB_DIR))

from flow_cache import (
    FlowFrame,
    create_flow_cache_dirs,
    flatten_validation,
    load_temporal_gt_frames,
    paths_for_pair,
    save_npy,
    validate_pair_outputs,
    write_json,
    write_manifest,
)


DEFAULT_SEQUENCE_DIR = Path("dataset/SCARED/curated/temporal_gt/test_dataset_9_keyframe_3")
DEFAULT_OUTPUT_DIR = Path(
    "results/04_dataset_derivatives/SCARED/temporal_gt_flow_cache/test_dataset_9_keyframe_3/raft"
)
DEFAULT_FULL_CHECKPOINT = Path("external/frame_stereo_repos/RAFT/checkpoints/raft-things.pth")
DEFAULT_SMALL_CHECKPOINT = Path("external/frame_stereo_repos/RAFT/checkpoints/raft-small.pth")


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
    parser = argparse.ArgumentParser(description="Build RAFT flow cache for SCARED temporal-GT.")
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--backend",
        choices=["raft-full", "raft-small"],
        default="raft-full",
        help="Optical-flow backend. raft-small selects the RAFT-Small architecture and default checkpoint.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--raft-iters", type=int, default=12)
    parser.add_argument("--small", action="store_true", help="Use RAFT-Small architecture.")
    parser.add_argument("--magnitude-threshold", type=float, default=20.0)
    parser.add_argument("--consistency-threshold", type=float, default=1.0)
    parser.add_argument("--overwrite", nargs="?", const=True, default=False, type=parse_bool)
    parser.add_argument("--max-pairs", type=int, default=0, help="0 builds all adjacent pairs; N builds only the first N pairs.")
    return parser.parse_args()


def resolve_device(requested: str):
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def read_rgb_tensor(frame: FlowFrame, device):
    import cv2
    import torch

    image = cv2.imread(str(frame.left_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read RGB frame: {frame.left_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    # FrozenRAFT.forward expects RGB tensors in [0, 1] and scales them to
    # [0, 255] internally before calling the bundled RAFT implementation.
    return (tensor / 255.0).to(device, non_blocking=True)


def flow_to_numpy(flow):
    import numpy as np

    return flow[0].permute(1, 2, 0).detach().float().cpu().numpy().astype(np.float32)


def tensor_map_to_numpy(value):
    import numpy as np

    array = value.detach().float().cpu().numpy()
    if array.ndim == 4:
        if array.shape[0] != 1 or array.shape[1] != 1:
            raise ValueError(f"Expected [B,1,H,W] map with B=1, got shape {array.shape}")
        array = array[0, 0]
    elif array.ndim == 3:
        if array.shape[0] == 1:
            array = array[0]
        elif array.shape[0] == 1 or array.shape[1] == 1:
            array = np.squeeze(array)
        else:
            raise ValueError(f"Expected [B,H,W] or [1,H,W] map with B=1, got shape {array.shape}")
    elif array.ndim != 2:
        raise ValueError(f"Expected map tensor with 2, 3, or 4 dims, got shape {array.shape}")
    if array.ndim != 2:
        raise ValueError(f"Expected HxW map after squeeze, got shape {array.shape}")
    return array.astype(np.float32)


def tensor_bool_map_to_numpy(value):
    import numpy as np

    return (tensor_map_to_numpy(value) > 0.5).astype(np.bool_)


def timed_raft(model, image_a, image_b, device):
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    flow = model(image_a, image_b)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return flow, elapsed_ms


def mean_finite(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    import numpy as np

    return float(np.mean(finite)) if finite else float("nan")


def ensure_safe_output_dir(output_dir: Path, overwrite: bool) -> None:
    resolved = output_dir.resolve()
    dataset_root = (ROOT / "dataset").resolve()
    scared_root = dataset_root / "SCARED"
    if resolved == dataset_root or resolved == scared_root or resolved.is_relative_to(dataset_root):
        raise RuntimeError(f"Refusing suspicious output path under dataset/: {output_dir}")
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Pass --overwrite true for an explicit rerun."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    create_flow_cache_dirs(output_dir)


def validate_checkpoint(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing RAFT checkpoint: {path}. Pass --checkpoint with a valid RAFT optical-flow checkpoint."
        )
    return path


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.backend == "raft-small":
        args.small = True
    if args.checkpoint is not None:
        return validate_checkpoint(args.checkpoint)
    return validate_checkpoint(DEFAULT_SMALL_CHECKPOINT if args.small else DEFAULT_FULL_CHECKPOINT)


def build_summary(
    args: argparse.Namespace,
    frames: list[FlowFrame],
    manifest_rows: list[dict[str, Any]],
    num_expected_pairs_total: int,
    num_pairs_requested: int,
    forward_runtimes: list[float],
    backward_runtimes: list[float],
    peak_vram_mb: float,
    output_size_mb: float,
    elapsed_s: float,
) -> dict[str, Any]:
    complete = all(row["status"] == "ok" for row in manifest_rows) and len(manifest_rows) == num_pairs_requested
    failed = [row for row in manifest_rows if row["status"] != "ok"]
    return {
        "cache_complete": complete,
        "checkpoint": str(args.checkpoint),
        "confidence_formula": {
            "source": "scripts.temporal_refinement.lib.flow.flow_confidence",
            "forward_occlusion": "occlusion(p)=|flow_fwd(p)+sample(flow_bwd,p+flow_fwd(p))| > consistency_threshold",
            "magnitude_confidence": "max(0, 1 - ||flow_fwd(p)|| / magnitude_threshold)",
            "forward_backward_confidence": "max(0, 1 - fb_error(p) / consistency_threshold)",
            "confidence": "magnitude_confidence * forward_backward_confidence * (1 - occlusion)",
            "magnitude_threshold": args.magnitude_threshold,
            "consistency_threshold": args.consistency_threshold,
        },
        "device": str(args.device),
        "elapsed_seconds": elapsed_s,
        "failed_pairs": len(failed),
        "flow_definition": {
            "forward_flow": "flow from frame t to frame t+1 in pixels, saved as HxWx2 float32",
            "backward_flow": "flow from frame t+1 to frame t in pixels, saved as HxWx2 float32",
            "forward_confidence": "confidence for forward_flow from forward/backward consistency, saved as HxW float32",
            "backward_confidence": "confidence for backward_flow from backward/forward consistency, saved as HxW float32",
            "occlusion": "forward occlusion mask for t to t+1, saved as HxW bool",
        },
        "frozen_raft_input_convention": "Builder passes RGB float tensors in [0, 1]; FrozenRAFT.forward multiplies by 255 internally before RAFT.",
        "image_resolution": [frames[0].height, frames[0].width] if frames else [],
        "num_computed_pairs": sum(1 for row in manifest_rows if row["status"] == "ok"),
        "num_expected_pairs_total": num_expected_pairs_total,
        "num_pairs_requested": num_pairs_requested,
        "num_frames": len(frames),
        "output_dir": str(args.output_dir),
        "output_size_mb": output_size_mb,
        "peak_vram_mb": peak_vram_mb,
        "raft_iters": args.raft_iters,
        "raft_small": bool(args.small),
        "sequence_dir": str(args.sequence_dir),
        "average_forward_runtime_ms": mean_finite(forward_runtimes),
        "average_backward_runtime_ms": mean_finite(backward_runtimes),
        "next_suggested_command": (
            "Implement scripts/temporal_refinement/eval_scripts/"
            "benchmark_scared_s2m2_temporal_baselines.py using this flow cache."
        ),
    }


def output_size_mb(output_dir: Path) -> float:
    total = 0
    for path in output_dir.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total / (1024.0 * 1024.0)


def write_run_log(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "RAFT flow-cache builder summary",
        f"flow_cache_path={summary['output_dir']}",
        f"num_frames={summary['num_frames']}",
        f"num_expected_pairs_total={summary['num_expected_pairs_total']}",
        f"num_pairs_requested={summary['num_pairs_requested']}",
        f"num_computed_pairs={summary['num_computed_pairs']}",
        f"num_failed_pairs={summary['failed_pairs']}",
        f"average_forward_runtime_ms={summary['average_forward_runtime_ms']}",
        f"average_backward_runtime_ms={summary['average_backward_runtime_ms']}",
        f"peak_vram_mb={summary['peak_vram_mb']}",
        f"output_size_mb={summary['output_size_mb']}",
        f"cache_complete={summary['cache_complete']}",
        f"next_suggested_command={summary['next_suggested_command']}",
    ]
    (output_dir / "run.log").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    sequence_dir = args.sequence_dir
    output_dir = args.output_dir
    checkpoint = resolve_checkpoint(args)
    args.checkpoint = checkpoint
    if args.max_pairs < 0:
        raise ValueError("--max-pairs must be >= 0")

    ensure_safe_output_dir(output_dir, bool(args.overwrite))
    frames = load_temporal_gt_frames(sequence_dir)
    if args.max_pairs > 0:
        frames = frames[: args.max_pairs + 1]
    if len(frames) < 2:
        raise RuntimeError(f"Need at least 2 frames to build adjacent flow cache, got {len(frames)}")
    num_expected_pairs_total = len(frames) - 1
    num_pairs_requested = num_expected_pairs_total if args.max_pairs == 0 else min(args.max_pairs, num_expected_pairs_total)

    device = resolve_device(args.device)
    import torch
    from flow import FrozenRAFT, flow_confidence

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = FrozenRAFT(checkpoint=checkpoint, iters=args.raft_iters, small=args.small).to(device).eval()
    manifest_rows: list[dict[str, Any]] = []
    forward_runtimes: list[float] = []
    backward_runtimes: list[float] = []
    start_all = time.perf_counter()

    with torch.no_grad():
        for pair_index, (prev_frame, cur_frame) in enumerate(zip(frames[:-1], frames[1:])):
            if pair_index >= num_pairs_requested:
                break
            paths = paths_for_pair(output_dir, prev_frame.frame_id, cur_frame.frame_id)
            row: dict[str, Any] = {
                "pair_index": pair_index,
                "prev_frame_id": prev_frame.frame_id,
                "cur_frame_id": cur_frame.frame_id,
                "forward_flow_path": str(paths.forward_flow),
                "backward_flow_path": str(paths.backward_flow),
                "forward_confidence_path": str(paths.forward_confidence),
                "backward_confidence_path": str(paths.backward_confidence),
                "occlusion_path": str(paths.occlusion),
                "forward_runtime_ms": "",
                "backward_runtime_ms": "",
                "exception_type": "",
                "exception_message": "",
                "status": "failed",
                "error": "",
            }
            try:
                prev_tensor = read_rgb_tensor(prev_frame, device)
                cur_tensor = read_rgb_tensor(cur_frame, device)
                forward_flow, forward_ms = timed_raft(model, prev_tensor, cur_tensor, device)
                backward_flow, backward_ms = timed_raft(model, cur_tensor, prev_tensor, device)
                forward_conf, forward_occ = flow_confidence(
                    forward_flow,
                    backward_flow,
                    magnitude_threshold=args.magnitude_threshold,
                    consistency_threshold=args.consistency_threshold,
                )
                backward_conf, _backward_occ = flow_confidence(
                    backward_flow,
                    forward_flow,
                    magnitude_threshold=args.magnitude_threshold,
                    consistency_threshold=args.consistency_threshold,
                )

                save_npy(paths.forward_flow, flow_to_numpy(forward_flow))
                save_npy(paths.backward_flow, flow_to_numpy(backward_flow))
                save_npy(paths.forward_confidence, tensor_map_to_numpy(forward_conf))
                save_npy(paths.backward_confidence, tensor_map_to_numpy(backward_conf))
                save_npy(paths.occlusion, tensor_bool_map_to_numpy(forward_occ))

                validation = validate_pair_outputs(paths, prev_frame.height, prev_frame.width)
                for key, result in validation.items():
                    row.update(flatten_validation(key, result))
                valid = all(result["valid"] for result in validation.values())
                row["forward_runtime_ms"] = f"{forward_ms:.6f}"
                row["backward_runtime_ms"] = f"{backward_ms:.6f}"
                row["status"] = "ok" if valid else "failed"
                if not valid:
                    row["error"] = "validation_failed"
                forward_runtimes.append(forward_ms)
                backward_runtimes.append(backward_ms)
            except Exception as exc:  # pragma: no cover - runtime failure recording
                row["exception_type"] = type(exc).__name__
                row["exception_message"] = str(exc)
                row["error"] = f"{type(exc).__name__}: {exc}"
                manifest_rows.append(row)
                continue
            manifest_rows.append(row)

    elapsed_s = time.perf_counter() - start_all
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024.0**2) if device.type == "cuda" else 0.0
    size_mb = output_size_mb(output_dir)
    summary = build_summary(
        args=args,
        frames=frames,
        manifest_rows=manifest_rows,
        num_expected_pairs_total=num_expected_pairs_total,
        num_pairs_requested=num_pairs_requested,
        forward_runtimes=forward_runtimes,
        backward_runtimes=backward_runtimes,
        peak_vram_mb=float(peak_vram_mb),
        output_size_mb=float(size_mb),
        elapsed_s=float(elapsed_s),
    )
    write_manifest(output_dir / "flow_cache_manifest.csv", manifest_rows)
    write_json(output_dir / "flow_cache_summary.json", summary)
    write_run_log(output_dir, summary)

    print(json.dumps({
        "flow_cache_path": summary["output_dir"],
        "num_expected_pairs_total": summary["num_expected_pairs_total"],
        "num_pairs_requested": summary["num_pairs_requested"],
        "num_computed_pairs": summary["num_computed_pairs"],
        "num_failed_pairs": summary["failed_pairs"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "peak_vram_mb": summary["peak_vram_mb"],
        "cache_complete": summary["cache_complete"],
        "next_suggested_command": summary["next_suggested_command"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
