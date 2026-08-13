"""On-the-fly DRENDS pilot evaluation for the reusable definitive protocol.

Only RAFT-Stereo is currently available here.  Predictions stay in memory;
the output is metrics and provenance only.  DRENDS depth comes from temporally
smoothed ToF, so it is a metric reference, not independent stereo GT.
"""
from __future__ import annotations

import json
import math
import sys
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HAND = ROOT.parent
CURATED = HAND / "dataset/DRENDS/processed/temporal_stereo_gt_curated"
RAFT_BACKBONE = "RAFT-Stereo"
CANONICAL_SIZE = (180, 144)
RAFT_CHECKPOINT = ROOT.parent.parent / "external/frame_stereo_repos/RAFT-Stereo/models/raftstereo-middlebury.pth"
# Fixed conservative exclusion: a stereo pair or ToF reference farther than
# 100 ms from its mate is not a meaningful instantaneous geometry sample.
MAX_TIMING_OFFSET_MS = 100.0
DRENDS_DEPTH_RANGE_MM = (1.0, 1000.0)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _manifest_paths(recording: str) -> tuple[Path, Path]:
    if recording == "Vid14_Pancreas_High":
        return CURATED / "temporal_pilot_manifest.json", CURATED / "temporal_pilot_quality_report.json"
    return CURATED / "recordings" / recording / "manifest.json", CURATED / "recordings" / recording / "quality_report.json"


def load_drends_records(recording: str, max_frames: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate materialized chronological input before any model is loaded."""
    manifest_path, quality_path = _manifest_paths(recording)
    manifest, quality = _read_json(manifest_path), _read_json(quality_path)
    sequence = manifest.get("sequence", {})
    frames = sequence.get("frames", []) if isinstance(sequence, Mapping) else []
    if manifest.get("path_base") != "repo_root" or sequence.get("recording") != recording or not isinstance(frames, list):
        raise RuntimeError(f"invalid DRENDS manifest: {manifest_path}")
    focal_baseline = quality.get("rectified_projection", {}).get("focal_baseline")
    if not isinstance(focal_baseline, (int, float)) or not math.isfinite(focal_baseline) or focal_baseline <= 0:
        raise RuntimeError(f"missing rectified focal-baseline: {quality_path}")
    required_paths = ("rect_left", "rect_right", "depth_left", "mask_left")
    required_times = ("helios", "left", "right")
    ids: list[str] = []
    values: dict[str, list[float]] = {name: [] for name in required_times}
    resolved: list[dict[str, Any]] = []
    excluded_timing: list[str] = []
    for frame in frames:
        if not isinstance(frame, Mapping) or not isinstance(frame.get("frame_id"), str):
            raise RuntimeError(f"invalid DRENDS frame in {manifest_path}")
        frame_id = frame["frame_id"]
        ids.append(frame_id)
        item = dict(frame)
        for name in required_paths:
            relative = item.get(name)
            path = HAND / relative if isinstance(relative, str) else None
            if path is None or not path.is_file():
                raise FileNotFoundError(f"DRENDS {recording}/{frame_id}: missing {name}")
            item[f"_{name}"] = path
        for name in required_times:
            value = item.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise RuntimeError(f"DRENDS {recording}/{frame_id}: invalid {name} timestamp")
            values[name].append(float(value))
        if any(float(item.get(name, float("inf"))) > MAX_TIMING_OFFSET_MS for name in
               ("left_right_offset_ms", "left_helios_offset_ms", "right_helios_offset_ms")):
            excluded_timing.append(frame_id)
            continue
        resolved.append(item)
    if len(ids) < 6 or len(ids) != len(set(ids)) or ids != sorted(ids):
        raise RuntimeError(f"DRENDS {recording}: need at least six unique chronological frame IDs")
    if any(np.any(np.diff(series) <= 0) for series in values.values()):
        raise RuntimeError(f"DRENDS {recording}: timestamps must be strictly chronological")
    if max_frames is not None:
        if max_frames < 6:
            raise ValueError("DRENDS --max-frames must be at least six")
        resolved = resolved[:max_frames]
    if len(resolved) < 6:
        raise RuntimeError(f"DRENDS {recording}: fewer than six selected frames")
    return resolved, {
        "recording": recording, "manifest": str(manifest_path), "quality_report": str(quality_path),
        "manifest_sha256": __import__("hashlib").sha256(manifest_path.read_bytes()).hexdigest(),
        "quality_sha256": __import__("hashlib").sha256(quality_path.read_bytes()).hexdigest(),
        "focal_baseline_native_px_m": float(focal_baseline),
        "independent_disparity_ground_truth": bool(quality.get("rectified_projection", {}).get("independent_disparity_ground_truth", False)),
        "frame_count": len(resolved), "timing_validation": f"finite chronological timestamps; excluded offsets > {MAX_TIMING_OFFSET_MS:g} ms",
        "excluded_timing_frame_ids": excluded_timing,
        "truncated": max_frames is not None and len(resolved) < len(frames),
    }


def _load_raft(device: Any):
    source = ROOT.parent.parent / "scripts/scared"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from eval_scared_external_native import eval_raft
    _, checkpoint, _, predict = eval_raft([], SimpleNamespace(), device)
    return checkpoint, predict


def _rgb(path: Path, size: tuple[int, int] | None = CANONICAL_SIZE) -> np.ndarray:
    import cv2
    value = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if value is None:
        raise FileNotFoundError(path)
    value = cv2.cvtColor(value, cv2.COLOR_BGR2RGB)
    return cv2.resize(value, size, interpolation=cv2.INTER_AREA) if size else value


def _depth(path: Path, mask_path: Path, scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import cv2
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if depth is None or mask is None or depth.shape != mask.shape or depth.ndim != 2:
        raise RuntimeError(f"invalid DRENDS depth/mask pair: {path}")
    valid = (mask > 0) & np.isfinite(depth) & (depth > 0)
    coverage = cv2.resize(valid.astype(np.float32), CANONICAL_SIZE, interpolation=cv2.INTER_AREA)
    numerator = cv2.resize(np.where(valid, depth, 0.0).astype(np.float32), CANONICAL_SIZE, interpolation=cv2.INTER_AREA)
    depth_mm = numerator / np.maximum(coverage, 1e-6)
    return depth_mm, coverage > .5, coverage


def _canonical_disparity(native: np.ndarray, scale: float) -> np.ndarray:
    import cv2
    return cv2.resize(native.astype(np.float32), CANONICAL_SIZE, interpolation=cv2.INTER_LINEAR) * scale


def _prediction_depth_mm(disparity: np.ndarray, product_mm: float) -> np.ndarray:
    """DRENDS clips finite positive prediction depths; invalid disparities stay invalid."""
    values = np.asarray(disparity, dtype=float)
    depth = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values) & (values > 0)
    depth[valid] = product_mm / values[valid]
    return np.clip(depth, *DRENDS_DEPTH_RANGE_MM)


def _canonical_frame(index: int, left_native: np.ndarray, right_native: np.ndarray, disparity_native: np.ndarray, *, device: Any) -> dict[str, Any]:
    """One immutable 144x180 frame contract shared by stereo, flow and H4."""
    import cv2
    import torch
    if left_native.shape != (720, 1280, 3) or right_native.shape != left_native.shape or disparity_native.shape != left_native.shape[:2]:
        raise RuntimeError("DRENDS requires rectified native RGB/disparity at 1280x720")
    scale = CANONICAL_SIZE[0] / left_native.shape[1]
    left = cv2.resize(left_native, CANONICAL_SIZE, interpolation=cv2.INTER_AREA)
    right = cv2.resize(right_native, CANONICAL_SIZE, interpolation=cv2.INTER_AREA)
    raw = _canonical_disparity(disparity_native, scale)
    frame = {"index": index, "raw": torch.from_numpy(raw)[None, None].to(device),
             "raw_valid": torch.from_numpy(np.isfinite(raw) & (raw > 0))[None, None].to(device),
             "rgb": torch.from_numpy(np.ascontiguousarray(left)).permute(2, 0, 1).float().to(device)[None],
             "right_rgb": torch.from_numpy(np.ascontiguousarray(right)).permute(2, 0, 1).float().to(device)[None]}
    _validate_canonical_frame(frame)
    return frame


def _validate_canonical_frame(frame: Mapping[str, Any]) -> None:
    expected = (1, 1, CANONICAL_SIZE[1], CANONICAL_SIZE[0])
    image = (1, 3, CANONICAL_SIZE[1], CANONICAL_SIZE[0])
    if tuple(frame["raw"].shape) != expected or tuple(frame["raw_valid"].shape) != expected or tuple(frame["rgb"].shape) != image or tuple(frame["right_rgb"].shape) != image:
        raise RuntimeError("DRENDS temporal inputs must all be canonical 144x180")


def _validate_canonical_rgb_frame(frame: Mapping[str, Any]) -> None:
    if tuple(frame["rgb"].shape) != (1, 3, CANONICAL_SIZE[1], CANONICAL_SIZE[0]):
        raise RuntimeError("DRENDS temporal RGB inputs must be canonical 144x180")


def _drends_flow(flow_model: Any, current: Mapping[str, Any], past: Mapping[str, Any]):
    _validate_canonical_frame(current); _validate_canonical_rgb_frame(past)
    return flow_model.current_to_anchor(current["rgb"], past["rgb"]), flow_model.anchor_to_current(past["rgb"], current["rgb"])


def _canonical_frames(records: list[dict[str, Any]], predict: Any, *, device: Any) -> tuple[list[dict[str, Any]], tuple[int, int]]:
    """Load, infer and discard each native stereo pair before continuing."""
    frames: list[dict[str, Any]] = []
    shape = (720, 1280)
    for index, item in enumerate(records):
        left_native, right_native = _rgb(item["_rect_left"], None), _rgb(item["_rect_right"], None)
        raw_native = predict(left_native, right_native)[0]
        if left_native.shape[:2] != shape or right_native.shape != left_native.shape or left_native.shape[2:] != (3,) or raw_native.shape != shape or not np.isfinite(raw_native).all():
            raise RuntimeError("DRENDS requires rectified native 1280x720 RGB and finite RAFT-Stereo disparity")
        frames.append(_canonical_frame(index, left_native, right_native, raw_native, device=device))
        del left_native, right_native, raw_native
    return frames, shape


def evaluate_drends(*, output: Path, module_spec: str, device_name: str, recordings: list[str] | None, max_frames: int | None) -> None:
    """Run H4 causally, then write compact reports without any prediction cache."""
    import torch
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT.parent.parent / "ARGOS_FREEZED/src")]
    from argos_freezed.alignment.sea_raft_adapter import SEARAFTFlowAdapter
    from model_design.comparison.run_comparison import atomic_json, check_adapter, drive, load_factory, prepare_output, sha256, validate_cuda
    from model_design.metrics.unified_metrics import MetricConfig, evaluate_argos_prediction

    selected = recordings or ["Vid14_Pancreas_High"]
    if len(selected) != len(set(selected)):
        raise ValueError("duplicate DRENDS recording")
    gpu = validate_cuda(device_name)
    factory = load_factory(module_spec); adapter = factory(device=device_name); check_adapter(adapter)
    prepare_output(output)
    metrics_path = ROOT / "model_design/metrics/unified_metrics.py"
    manifest = {"project": "ARGOS v2", "status": "INCOMPLETE", "dataset": "drends", "backbones": [RAFT_BACKBONE],
                "module_provenance": adapter.describe(), "CUDA_VISIBLE_DEVICES": gpu, "no_gt_in_adapter": True,
                "dense_predictions_written": False, "unified_metrics": {"path": str(metrics_path), "sha256": sha256(metrics_path)},
                "evaluator": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
                "comparison_driver": {"path": str(ROOT / "model_design/comparison/run_comparison.py"), "sha256": sha256(ROOT / "model_design/comparison/run_comparison.py")}}
    atomic_json(output / "run_manifest.json", manifest)
    try:
        flow_model = SEARAFTFlowAdapter(device=torch.device(device_name))
        checkpoint, predict = _load_raft(torch.device(device_name))
        if not RAFT_CHECKPOINT.is_file():
            raise FileNotFoundError(f"RAFT-Stereo checkpoint not found: {RAFT_CHECKPOINT}")
        manifest = manifest | {"flow": {"checkpoint": str(flow_model.checkpoint), "sha256": sha256(flow_model.checkpoint)},
                               "backbone": {"name": RAFT_BACKBONE, "checkpoint": str(RAFT_CHECKPOINT), "sha256": hashlib.sha256(RAFT_CHECKPOINT.read_bytes()).hexdigest()}}
        atomic_json(output / "run_manifest.json", manifest)
        reports = []
        for recording in selected:
            records, info = load_drends_records(recording, max_frames)
            frames, (height, width) = _canonical_frames(records, predict, device=device_name)
            scale = CANONICAL_SIZE[0] / width
            raw = [frame["raw"][0, 0].cpu().numpy() for frame in frames]
            def flow(current: Mapping[str, Any], past: Mapping[str, Any]):
                return _drends_flow(flow_model, current, past)
            outputs = dict(drive(adapter, frames, flow))
            depth_values, valid_values, coverages = zip(*(_depth(item["_depth_left"], item["_mask_left"], scale) for item in records))
            product_mm = info["focal_baseline_native_px_m"] * 1000.0 * scale
            gt = [product_mm / np.maximum(value, 1e-6) for value in depth_values]
            refined = [outputs[index]["disparity"][0, 0].detach().cpu().numpy() for index in range(len(records))]
            raw_depth = [_prediction_depth_mm(value, product_mm) for value in raw]
            refined_depth = [_prediction_depth_mm(value, product_mm) for value in refined]
            report = evaluate_argos_prediction(raw_disparity=np.asarray(raw), refined_disparity=np.asarray(refined), gt_disparity=np.asarray(gt),
                raw_depth=np.asarray(raw_depth), refined_depth=np.asarray(refined_depth), gt_depth=np.asarray(depth_values), depth_input_unit="mm",
                gt_valid=np.asarray(valid_values), protocol_mask=np.asarray(valid_values), config=MetricConfig(),
                keyframe_mask=np.asarray([index == 0 or bool(outputs[index]["reset"]) for index in range(len(records))])[None],
                sequence_ids=[recording], frame_ids=[item["frame_id"] for item in records])
            report.update({"dataset": "DRENDS", "split": "pilot", "backbone": RAFT_BACKBONE,
                "protocol": "chronological rectified frames; fixed H=4 causal recurrence; reset at recording boundary",
                "applicability": {"depth_reference": "APPLICABLE_WITH_CAVEAT", "independent_stereo_disparity_gt": "NOT_APPLICABLE",
                                  "caveat": "depth reference is temporally smoothed ToF and is not independent disparity ground truth"},
                "diagnostics": {"adapter_support_coverage": float(np.mean([outputs[index]["support"].float().mean().item() for index in outputs])),
                                "backbone_checkpoint": checkpoint, "native_resolution": [width, height], "canonical_resolution": list(CANONICAL_SIZE),
                                "disparity_scale": scale, "coverage_area_mean": float(np.mean(coverages)), **info}})
            json.dumps(report, allow_nan=False)
            atomic_json(output / "reports" / RAFT_BACKBONE / f"{recording}.json", report); reports.append(report)
        atomic_json(output / "summary.json", {"dataset": "DRENDS", "backbone": RAFT_BACKBONE, "recordings": selected,
                                                "status": "pilot" if max_frames is not None else "complete_recordings", "reports": [str(report["sequence_ids"][0]) for report in reports]})
        atomic_json(output / "applicability.json", {"dataset": "DRENDS", "temporal_h4_evaluation": "APPLICABLE_WITH_CAVEAT",
                                                      "metric_scope": "tof_reference_nonindependent",
                                                      "metric_reference": "temporally smoothed ToF; not independent stereo disparity GT"})
        atomic_json(output / "run_manifest.json", manifest | {"status": "COMPLETE", "outputs": sorted(str(path.relative_to(output)) for path in output.rglob("*") if path.is_file())})
    except BaseException as error:
        atomic_json(output / "run_manifest.json", manifest | {"error": f"{type(error).__name__}: {error}"}); raise
