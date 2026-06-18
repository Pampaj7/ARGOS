#!/usr/bin/env python3
"""Run a coherent SCARED keyframe benchmark for locally supported methods.

All evaluated methods use the same curated SCARED dataset_8 keyframes, the same
rectification routine, the same GT masks, and the same metric code. Methods that
exist in the SERV-CT table but do not yet have a SCARED adapter are listed in
the status report instead of being mixed into the ranking.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Callable

import cv2
import numpy as np
import onnxruntime as ort
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
STEREO_ROOT = ROOT.parent / "stereo"
sys.path.insert(0, str(ROOT / "scripts/scared"))

from benchmark_s2m2_size_tradeoff import (  # noqa: E402
    MODEL_CONFIG,
    build_model,
    collect_samples,
    infer as infer_s2m2,
    metric_row,
)
from eval_metrics import failure_aware_metrics  # noqa: E402


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class MethodResult:
    method: str
    checkpoint: str
    input_resolution: str
    metrics_path: Path | None
    status: str
    notes: str = ""


def resolve_data_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute() or path.exists():
        return path
    return ROOT / path


def read_rgb_path(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def collect_samples_from_metadata(metadata_csv: Path) -> list[dict]:
    samples = []
    with metadata_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("notes"):
                continue
            left_path = resolve_data_path(row["left_path"])
            right_path = resolve_data_path(row["right_path"])
            depth_path = resolve_data_path(row["depth_float32_path"])
            disp_path = resolve_data_path(row["disparity_float32_path"])
            mask_path = resolve_data_path(row["valid_mask_path"])
            calib_path = resolve_data_path(row["calibration_path"])
            required = [left_path, right_path, depth_path, disp_path, mask_path, calib_path]
            if not all(p.exists() for p in required):
                missing = [str(p) for p in required if not p.exists()]
                raise FileNotFoundError(f"Missing metadata sample files for {row}: {missing}")
            calib = json.loads(calib_path.read_text())
            p1 = np.array(calib["P1"]["data"], dtype=np.float64).reshape(3, 4)
            p2 = np.array(calib["P2"]["data"], dtype=np.float64).reshape(3, 4)
            if row.get("dataset_id") and row.get("keyframe_id"):
                frame = f"{row['dataset_id']}_{row['keyframe_id']}_frame_{int(row['frame_id']):06d}"
            else:
                frame = f"{row.get('sequence_id', 'sequence')}_frame_{int(row['frame_id']):06d}"
            samples.append(
                {
                    "frame": frame,
                    "left": read_rgb_path(left_path),
                    "right": read_rgb_path(right_path),
                    "gt_disp": np.load(disp_path).astype(np.float32),
                    "gt_depth": np.load(depth_path).astype(np.float32),
                    "valid": np.load(mask_path).astype(bool),
                    "fx": float(p1[0, 0]),
                    "baseline_mm": float(abs(p2[0, 3] / p2[0, 0])),
                }
            )
    if not samples:
        raise RuntimeError(f"No usable SCARED samples found in {metadata_csv}")
    return samples


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_imagenet(img_rgb: np.ndarray) -> np.ndarray:
    return ((img_rgb.astype(np.float32) / 255.0) - IMAGENET_MEAN) / IMAGENET_STD


def make_fastfoundation_session(model_file: Path) -> tuple[ort.InferenceSession, int, int]:
    cfg = yaml.safe_load(model_file.with_suffix(".yaml").read_text())
    target_h, target_w = cfg["image_size"]
    providers = []
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    session = ort.InferenceSession(str(model_file), providers=providers)
    return session, int(target_h), int(target_w)


def infer_fastfoundation(
    session: ort.InferenceSession,
    left: np.ndarray,
    right: np.ndarray,
    target_h: int,
    target_w: int,
) -> tuple[np.ndarray, float]:
    orig_h, orig_w = left.shape[:2]
    scale_x = target_w / float(orig_w)
    left_res = cv2.resize(left, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    right_res = cv2.resize(right, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    left_t = normalize_imagenet(left_res).transpose(2, 0, 1)[None].astype(np.float32)
    right_t = normalize_imagenet(right_res).transpose(2, 0, 1)[None].astype(np.float32)
    t0 = time.perf_counter()
    outputs = session.run(None, {"left_image": left_t, "right_image": right_t})
    runtime_ms = (time.perf_counter() - t0) * 1000.0
    pred = outputs[0].reshape(target_h, target_w).astype(np.float32).clip(0, None)
    pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) / scale_x
    return pred.astype(np.float32), runtime_ms


def make_sgbm(max_disp: int, block_size: int):
    max_disp = int(np.ceil(max_disp / 16.0) * 16)
    p1 = 8 * block_size * block_size
    p2 = 32 * block_size * block_size
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=max_disp,
        blockSize=block_size,
        P1=p1,
        P2=p2,
        disp12MaxDiff=2,
        uniquenessRatio=4,
        speckleWindowSize=80,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def infer_sgbm(matcher, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, float]:
    left_g = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
    right_g = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
    t0 = time.perf_counter()
    pred = matcher.compute(left_g, right_g).astype(np.float32) / 16.0
    runtime_ms = (time.perf_counter() - t0) * 1000.0
    return np.clip(pred, 0, None).astype(np.float32), runtime_ms


def summarize_metric_rows(rows: list[dict]) -> dict:
    skip = {"method", "checkpoint", "input_resolution", "frame"}
    numeric = [k for k, v in rows[0].items() if k not in skip and isinstance(v, (int, float))]
    return {k: float(np.mean([r[k] for r in rows])) for k in numeric}


def reset_cuda_peak() -> None:
    try:
        torch.cuda.reset_peak_memory_stats()
    except RuntimeError:
        pass


def current_cuda_peak_mb() -> float | None:
    try:
        return float(torch.cuda.max_memory_allocated() / 1024**2)
    except RuntimeError:
        return None


def evaluate_method(
    samples: list[dict],
    method: str,
    checkpoint: str,
    input_resolution: str,
    out_dir: Path,
    predict: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, float]],
) -> MethodResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample in samples:
        pred, runtime_ms = predict(sample["left"], sample["right"])
        row = {
            "method": method,
            "checkpoint": checkpoint,
            "input_resolution": input_resolution,
            "frame": sample["frame"],
            "runtime_ms": runtime_ms,
        }
        row.update(metric_row(pred, sample))
        pred_depth = sample["fx"] * sample["baseline_mm"] / np.maximum(pred, 1e-6)
        raw_mask = (
            sample["valid"]
            & np.isfinite(pred)
            & np.isfinite(pred_depth)
            & (sample["gt_disp"] > 0)
            & (sample["gt_depth"] > 0)
        )
        row.update(
            failure_aware_metrics(
                pred,
                pred_depth,
                sample["gt_disp"],
                sample["gt_depth"],
                sample["valid"],
                raw_mask,
            )
        )
        rows.append(row)
        np.save(out_dir / f"{sample['frame']}_disp.npy", pred.astype(np.float32))
        print(f"{method} {sample['frame']}: depth_mae={row['valid_depth_mae']:.3f} runtime={runtime_ms:.1f}ms", flush=True)
    write_csv(out_dir / "metrics.csv", rows)
    summary = summarize_metric_rows(rows)
    summary.update({"method": method, "checkpoint": checkpoint, "input_resolution": input_resolution, "frames": len(rows)})
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return MethodResult(method, checkpoint, input_resolution, out_dir / "metrics.csv", "evaluated")


def run_s2m2_methods(samples: list[dict], args, device: torch.device, out_root: Path) -> list[MethodResult]:
    results = []
    jobs = [
        ("S2M2-S", "S pretrained", "512 width", "S", 512),
        ("S2M2-L", "L pretrained", "736 width", "L", 736),
        ("S2M2-L full", "L pretrained", "1024x1280 full", "L", 0),
        ("S2M2-XL", "XL pretrained", "1024x1280 full", "XL", 0),
    ]
    for method, checkpoint, input_resolution, model_type, width in jobs:
        if device.type == "cuda":
            reset_cuda_peak()
        model_args = argparse.Namespace(
            s2m2_src=args.s2m2_src,
            weights_dir=args.s2m2_weights_dir,
            refine_iter=args.s2m2_refine_iter,
        )
        model, ckpt_path = build_model(model_args, model_type, device)

        def predict(left, right, model=model, width=width):
            pred, runtime_ms, _shape, _scale = infer_s2m2(model, left, right, width, device, args.s2m2_src)
            return pred, runtime_ms

        result = evaluate_method(samples, method, ckpt_path.name, input_resolution, out_root / method.replace(" ", "_").replace("/", "_"), predict)
        summary = json.loads((result.metrics_path.parent / "summary.json").read_text())
        if device.type == "cuda":
            peak_mb = current_cuda_peak_mb()
            if peak_mb is not None:
                summary["peak_gpu_memory_mb"] = peak_mb
            (result.metrics_path.parent / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        results.append(result)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results


def run_supported_methods(samples: list[dict], args, out_root: Path) -> list[MethodResult]:
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    results = run_s2m2_methods(samples, args, device, out_root)

    fast_model = args.fastfoundation_onnx
    if fast_model.exists():
        session, target_h, target_w = make_fastfoundation_session(fast_model)

        def predict_fast(left, right):
            return infer_fastfoundation(session, left, right, target_h, target_w)

        results.append(
            evaluate_method(
                samples,
                "Fast-FoundationStereo ONNX",
                fast_model.parent.name,
                f"{target_h}x{target_w}",
                out_root / "Fast-FoundationStereo_ONNX",
                predict_fast,
            )
        )
    else:
        results.append(MethodResult("Fast-FoundationStereo ONNX", str(fast_model), "320x736", None, "missing_checkpoint"))

    matcher = make_sgbm(args.sgbm_max_disp, args.sgbm_block_size)
    results.append(
        evaluate_method(
            samples,
            "SGBM",
            f"OpenCV SGBM block={args.sgbm_block_size} max_disp={args.sgbm_max_disp}",
            "1024x1280 full",
            out_root / "SGBM",
            lambda left, right: infer_sgbm(matcher, left, right),
        )
    )
    return results


def build_table(eval_root: Path, out_dir: Path, status_rows: list[MethodResult], protocol_name: str, dataset_label: str) -> None:
    native_protocol_dirs = {
        "CREStereo_native",
        "DEFOM-Stereo_ViT-L_ETH3D_native",
        "Fast-FoundationStereo_ONNX",
        "MonSter++_MixAll_native",
        "RAFT-Stereo_Middlebury_native",
        "RT-MonSter++_ZeroShot_native",
        "S2M2-L",
        "S2M2-L_full",
        "S2M2-S",
        "S2M2-XL",
        "SGBM",
        "StereoAnywhere_native",
    }
    native_rows = []
    evidence = []
    for summary_path in sorted(eval_root.glob("*/summary.json")):
        if summary_path.parent.name not in native_protocol_dirs:
            continue
        protocol = protocol_name
        data = json.loads(summary_path.read_text())
        metrics_path = summary_path.parent / "metrics.csv"
        runtime_values = []
        if metrics_path.exists():
            with metrics_path.open(newline="") as f:
                runtime_values = [float(r["runtime_ms"]) for r in csv.DictReader(f) if r.get("runtime_ms")]
        runtime_ms = median(runtime_values) if runtime_values else data.get("runtime_ms")
        depth_mae = data.get("valid_disp_depth_mae_mm", data.get("valid_depth_mae", data.get("depth_mae_mm")))
        bad_2mm = data.get("valid_disp_depth_bad2mm_pct", data.get("bad_2mm", data.get("depth_bad2mm_pct")))
        disp_mae = data.get("valid_disp_mae_px", data.get("valid_disp_mae", data.get("mae_px")))
        row = {
            "Method": data["method"],
            "Training / Checkpoint": data["checkpoint"],
            "Input res.": data["input_resolution"],
            "Depth MAE ↓": f"{depth_mae:.3f}",
            "Bad-2 mm ↓": f"{bad_2mm:.2f}%",
            "Disp. MAE ↓": f"{disp_mae:.3f}",
            "Runtime ↓": f"{runtime_ms:.1f} ms" if runtime_ms is not None else "",
            "VRAM ↓": f"{data['peak_gpu_memory_mb'] / 1024:.2f} GB" if "peak_gpu_memory_mb" in data else "",
        }
        native_rows.append((depth_mae, row))
        evidence.append({**row, "source_file": str(summary_path), "status": "evaluated", "protocol": protocol})
    native_rows = [r for _score, r in sorted(native_rows, key=lambda x: x[0])]
    frame_counts = []
    for summary_path in sorted(eval_root.glob("*/summary.json")):
        try:
            frame_counts.append(int(json.loads(summary_path.read_text()).get("frames", 0)))
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    frame_label = str(max(frame_counts)) if frame_counts else "unknown"
    cols = ["Method", "Training / Checkpoint", "Input res.", "Depth MAE ↓", "Bad-2 mm ↓", "Disp. MAE ↓", "Runtime ↓", "VRAM ↓"]
    write_csv(out_dir / "scared_evaluation.csv", native_rows, cols)
    write_csv(out_dir / "evidence.csv", evidence, cols + ["source_file", "status", "protocol"])

    table = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in native_rows:
        table.append("| " + " | ".join(row[c] for c in cols) + " |")
    missing = [
        {"Method": r.method, "Status": r.status, "Notes": r.notes}
        for r in status_rows
        if r.status != "evaluated"
    ]
    missing_lines = ["| Method | Status | Notes |", "| --- | --- | --- |"]
    for row in missing:
        missing_lines.append(f"| {row['Method']} | {row['Status']} | {row['Notes']} |")
    md = f"""# SCARED Evaluation

Single GT-backed SCARED table. The main table below only includes
methods evaluated through the `{protocol_name}` protocol in
`scripts/scared/run_all_scared_baselines.py`.

{chr(10).join(table)}

## Protocol

- Dataset: `{dataset_label}`.
- Frames: {frame_label} samples with left/right images, calibration, and depth/disparity GT.
- Images are rectified inside the shared loader for keyframes, or loaded from pre-rectified warped metadata for warped samples.
- Resized predictions are rescaled back to original disparity coordinates.
- Metrics: disparity MAE and metric-depth MAE/Bad-2 mm over the shared valid GT mask.
- Important: this is a same-evaluator / same-GT table, not a same-input-resolution table. The `Input res.` column is part of the result.
- Fast-FoundationStereo currently uses the only local Fast-FoundationStereo ONNX artifact available: fixed `320x736`.
- S2M2-S/L rows intentionally use deployment-style widths (`512`, `736`, or full), while external models currently run on the native rectified resolution unless otherwise stated.

## SERV-CT Methods Not Yet In This SCARED Table

{chr(10).join(missing_lines) if missing else 'All tracked methods with ready adapters were evaluated.'}

## Evidence

- Per-method outputs: `{eval_root}`.
- Row evidence: `evidence.csv`.
"""
    (out_dir / "scared_evaluation.md").write_text(md)
    (out_dir / "README.md").write_text(
        "# SCARED Evaluation\n\n"
        "Run/update with:\n\n"
        "```bash\n"
        "python3 scripts/scared/run_all_scared_baselines.py\n"
        "```\n\n"
        "Main outputs: `scared_evaluation.md`, `scared_evaluation.csv`, `evidence.csv`.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scared-root", type=Path, default=ROOT / "dataset/SCARED/curated/keyframes_gt_dataset8/dataset_8")
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--protocol-name", default="native_curated_scared")
    parser.add_argument("--dataset-label", default="dataset/SCARED/curated/keyframes_gt_dataset8/dataset_8")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/01_frame_stereo/SCARED")
    parser.add_argument("--eval-subdir", default="unified_keyframes")
    parser.add_argument("--s2m2-src", type=Path, default=STEREO_ROOT / "s2m2/src")
    parser.add_argument("--s2m2-weights-dir", type=Path, default=STEREO_ROOT / "s2m2/weights/pretrain_weights")
    parser.add_argument("--s2m2-refine-iter", type=int, default=3)
    parser.add_argument(
        "--fastfoundation-onnx",
        type=Path,
        default=STEREO_ROOT / "Fast-FoundationStereo/weights/onnx/20_30_48/320x736/20_30_48_iters_4_res_320x736.onnx",
    )
    parser.add_argument("--sgbm-max-disp", type=int, default=320)
    parser.add_argument("--sgbm-block-size", type=int, default=3)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    eval_root = args.out_dir / args.eval_subdir
    eval_root.mkdir(parents=True, exist_ok=True)
    (eval_root / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2) + "\n")

    status_rows = [
        MethodResult("StereoAnyVideo", "official checkpoint", "384x640 video", None, "separate_video_adapter", "Use `scripts/scared/run_stereoanyvideo_temporal_eval.py`; not frame-adapter compatible yet."),
    ]

    if not args.aggregate_only:
        samples = collect_samples_from_metadata(args.metadata_csv) if args.metadata_csv else collect_samples(args.scared_root)
        evaluated = run_supported_methods(samples, args, eval_root)
        status_rows.extend(evaluated)
    build_table(eval_root, args.out_dir, status_rows, args.protocol_name, args.dataset_label)


if __name__ == "__main__":
    main()
