#!/usr/bin/env python3
"""Causal BiDA t-1 evidence validation on canonical SCARED-C backbone caches.

Flow is inferred once per sequence and shared across all stereo backbones.  Dense
flow is kept in RAM and is not saved by default.  The CSV is checkpointed after
every backbone/sequence block, making long runs resumable without touching the
validated stereo caches.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

V2_ROOT = Path(__file__).resolve().parents[1]
ROOT = V2_ROOT.parent
sys.path.insert(0, str(V2_ROOT / "scripts"))

from argos_v2.cache_io import load_sequence_cache  # noqa: E402
from argos_v2.metrics import resize_gt_to_cache_corrected  # noqa: E402
from argos_v2.scared_c_data import load_frame_gt, load_frame_lr, load_sequence_info  # noqa: E402
from argos_v2.sequences import representative_sequences  # noqa: E402


def _load_bida():
    path = V2_ROOT / "model_design/external_components/bidavideo.py"
    spec = importlib.util.spec_from_file_location("argos_v2_bidavideo", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bida = _load_bida()
CACHE_SIZE = (144, 180)
DEFAULT_BACKBONES = ["S2M2-S", "RAFT-Stereo", "StereoAnywhere"]
DEFAULT_THRESHOLDS = [0.05, 0.25, 0.50, 0.90]
METHODS = ["raw", "memory", "blend_0.1", "blend_0.25", "blend_0.5", "gated", "oracle"]


def tensor_image(rgb: np.ndarray, size: tuple[int, int], device: torch.device) -> torch.Tensor:
    resized = cv2.resize(rgb, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(resized).permute(2, 0, 1).float().to(device)


def boundary_mask(disparity: np.ndarray, valid: np.ndarray) -> np.ndarray:
    dx = cv2.Sobel(disparity, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(disparity, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.hypot(dx, dy) > 1.0
    edge |= cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    return cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool) & valid


def geometry(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, boundary: np.ndarray) -> dict[str, float | int]:
    count = int(mask.sum())
    if count == 0:
        return {"epe": math.nan, "bad1": math.nan, "bad3": math.nan, "absrel": math.nan, "boundary_epe": math.nan}
    error = np.abs(pred - gt)
    boundary_common = mask & boundary
    return {
        "epe": float(error[mask].mean()),
        "bad1": float((error[mask] > 1.0).mean()),
        "bad3": float((error[mask] > 3.0).mean()),
        "absrel": float((error[mask] / np.maximum(gt[mask], 1e-6)).mean()),
        "boundary_epe": float(error[boundary_common].mean()) if boundary_common.any() else math.nan,
    }


def binned_advantage(
    raw_error: np.ndarray,
    memory_error: np.ndarray,
    signal: np.ndarray,
    mask: np.ndarray,
    prefix: str,
) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    edges = [0.0, 0.25, 0.50, 0.75, 1.000001]
    for lo, hi in zip(edges[:-1], edges[1:]):
        selected = mask & (signal >= lo) & (signal < hi)
        label = f"{int(lo * 100):02d}_{int(min(hi, 1) * 100):03d}"
        out[f"{prefix}_{label}_count"] = int(selected.sum())
        out[f"{prefix}_{label}_memory_advantage"] = (
            float((raw_error[selected] - memory_error[selected]).mean()) if selected.any() else math.nan
        )
    return out


def safety(
    raw: np.ndarray,
    candidate: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    raw_error = np.abs(raw - gt)
    candidate_error = np.abs(candidate - gt)
    clean = mask & (raw_error <= 3.0)
    if not mask.any():
        return {"new_bad3": math.nan, "clean_degradation_ratio": math.nan, "clean_update_mean": math.nan, "frame_degradation": math.nan}
    return {
        "new_bad3": float((candidate_error[clean] > 3.0).mean()) if clean.any() else math.nan,
        "clean_degradation_ratio": float((candidate_error[clean] > raw_error[clean]).mean()) if clean.any() else math.nan,
        "clean_update_mean": float(np.abs(candidate[clean] - raw[clean]).mean()) if clean.any() else math.nan,
        "frame_degradation": float(candidate_error[mask].mean() - raw_error[mask].mean()),
    }


def infer_sequence_flows(
    adapter,
    images: list[torch.Tensor],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    forward: list[np.ndarray] = []
    backward: list[np.ndarray] = []
    elapsed = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for start in range(1, len(images), batch_size):
        stop = min(len(images), start + batch_size)
        current = torch.stack(images[start:stop])
        past = torch.stack(images[start - 1 : stop - 1])
        pairs_target = torch.cat((current, past), dim=0)
        pairs_source = torch.cat((past, current), dim=0)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        both = adapter.infer(pairs_target, pairs_source)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed += time.perf_counter() - t0
        n = stop - start
        forward.extend(both[:n].detach().cpu().numpy().astype(np.float32))
        backward.extend(both[n:].detach().cpu().numpy().astype(np.float32))
    peak_mb = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
    return np.stack(forward), np.stack(backward), elapsed * 1000.0 / (len(images) - 1), peak_mb


def evaluate_frame(
    *,
    raw: np.ndarray,
    raw_valid: np.ndarray,
    past: np.ndarray,
    past_valid: np.ndarray,
    gt: np.ndarray,
    gt_valid: np.ndarray,
    current_rgb: torch.Tensor,
    past_rgb: torch.Tensor,
    flow_current_past: np.ndarray,
    flow_past_current: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], float]:
    def t1(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array.astype(np.float32))[None, None].to(device)

    t0 = time.perf_counter()
    evidence = bida.temporal_disparity_evidence(
        t1(raw),
        t1(past),
        torch.from_numpy(flow_current_past)[None].to(device),
        torch.from_numpy(flow_past_current)[None].to(device),
        current_valid=t1(raw_valid) > 0,
        past_valid=t1(past_valid) > 0,
        current_rgb=current_rgb[None],
        past_rgb=past_rgb[None],
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    evidence_ms = (time.perf_counter() - t0) * 1000.0
    e = {key: value[0, 0].detach().cpu().numpy() for key, value in evidence.as_dict().items()}
    aligned = e["aligned_past_disparity"]
    gate = (
        0.5
        * e["forward_backward_confidence"]
        * (1.0 - e["photometric_residual"])
        * e["aligned_validity"].astype(np.float32)
    )
    predictions = {
        "raw": raw,
        "memory": aligned,
        "blend_0.1": 0.9 * raw + 0.1 * aligned,
        "blend_0.25": 0.75 * raw + 0.25 * aligned,
        "blend_0.5": 0.5 * raw + 0.5 * aligned,
        "gated": raw + gate * (aligned - raw),
    }
    return predictions, e, evidence_ms


def native_inputs(
    raw: np.ndarray,
    raw_valid: np.ndarray,
    past: np.ndarray,
    past_valid: np.ndarray,
    gt_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = gt_shape
    scale = w / CACHE_SIZE[1]
    raw_n = cv2.resize(raw, (w, h), interpolation=cv2.INTER_LINEAR) * scale
    past_n = cv2.resize(past, (w, h), interpolation=cv2.INTER_LINEAR) * scale
    raw_v = cv2.resize(raw_valid.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    past_v = cv2.resize(past_valid.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    return raw_n, raw_v, past_n, past_v


def make_rows(
    *,
    namespace: str,
    flow_model: str,
    backbone: str,
    sequence: str,
    frame_id: str,
    frame_index: int,
    threshold: float | None,
    predictions: dict[str, np.ndarray],
    evidence: dict[str, np.ndarray],
    gt: np.ndarray,
    gt_valid: np.ndarray,
    raw_valid: np.ndarray,
    flow_latency_ms: float,
    evidence_latency_ms: float,
    peak_gpu_memory_mb: float,
) -> list[dict]:
    aligned_valid = evidence["aligned_validity"].astype(bool)
    support = evidence["warp_support"].astype(bool)
    common = gt_valid & raw_valid & aligned_valid & support
    boundary = boundary_mask(gt, gt_valid)
    predictions = dict(predictions)
    raw_error = np.abs(predictions["raw"] - gt)
    memory_error = np.abs(predictions["memory"] - gt)
    predictions["oracle"] = np.where(memory_error < raw_error, predictions["memory"], predictions["raw"])
    base = {
        "namespace": namespace,
        "coverage_threshold": threshold if threshold is not None else "native",
        "flow_model": flow_model,
        "backbone": backbone,
        "sequence": sequence,
        "frame_id": frame_id,
        "frame_index": frame_index,
        "memory_age": 1,
        "common_valid_count": int(common.sum()),
        "common_valid_ratio": float(common.mean()),
        "common_over_gt_raw_ratio": float(common.sum() / max(1, (gt_valid & raw_valid).sum())),
        "support_ratio": float(support.mean()),
        "occlusion_invalid_warp_ratio": float(1.0 - aligned_valid.mean()),
        "fb_consistent_ratio": float((evidence["forward_backward_confidence"] >= math.exp(-1)).mean()),
        "photometric_residual_mean": float(evidence["photometric_residual"][aligned_valid].mean()) if aligned_valid.any() else math.nan,
        "memory_better_ratio": float((memory_error[common] < raw_error[common]).mean()) if common.any() else math.nan,
        "raw_better_ratio": float((raw_error[common] < memory_error[common]).mean()) if common.any() else math.nan,
        "oracle_gain": float((raw_error[common] - np.minimum(raw_error[common], memory_error[common])).mean()) if common.any() else math.nan,
        "flow_latency_ms": flow_latency_ms,
        "evidence_latency_ms": evidence_latency_ms,
        "total_baseline_latency_ms": flow_latency_ms + evidence_latency_ms,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
    }
    base.update(binned_advantage(raw_error, memory_error, evidence["forward_backward_confidence"], common, "fb_conf"))
    base.update(binned_advantage(raw_error, memory_error, evidence["photometric_residual"], common, "photo"))
    rows = []
    for method, prediction in predictions.items():
        row = dict(base)
        row["method"] = method
        row.update(geometry(prediction, gt, common, boundary))
        row.update(safety(predictions["raw"], prediction, gt, common))
        rows.append(row)
    return rows


def save_contact_sheet(
    path: Path,
    rgb: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> None:
    """Save one compact diagnostic sheet; metrics remain the decision evidence."""
    h, w = gt.shape
    rgb_small = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    panels = [cv2.cvtColor(rgb_small, cv2.COLOR_RGB2BGR)]
    labels = ["RGB"]
    vmax = max(3.0, float(np.percentile(np.abs(predictions["raw"][valid] - gt[valid]), 95))) if valid.any() else 3.0
    for name in ("raw", "memory", "gated"):
        error = np.abs(predictions[name] - gt)
        vis = np.clip(error / vmax * 255.0, 0, 255).astype(np.uint8)
        vis[~valid] = 0
        panels.append(cv2.applyColorMap(vis, cv2.COLORMAP_TURBO))
        labels.append(f"{name} abs error")
    canvas = np.concatenate(panels, axis=1)
    for i, label in enumerate(labels):
        cv2.putText(canvas, label, (i * w + 5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def aggregate_rows(rows: list[dict], output_dir: Path, command: str) -> None:
    numeric_exclude = {"namespace", "coverage_threshold", "flow_model", "backbone", "sequence", "frame_id", "method"}
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["namespace"], row["coverage_threshold"], row["flow_model"], row["backbone"], row["sequence"], row["method"])
        groups[key].append(row)
    sequence_rows = []
    for key, values in groups.items():
        out = dict(zip(["namespace", "coverage_threshold", "flow_model", "backbone", "sequence", "method"], key))
        out["n_frames"] = len(values)
        for field in values[0]:
            if field in numeric_exclude or field in {"frame_index", "memory_age"}:
                continue
            numbers = [float(v[field]) for v in values if v.get(field) not in (None, "") and math.isfinite(float(v[field]))]
            if numbers:
                out[field] = float(np.mean(numbers))
        degradations = [float(v["frame_degradation"]) for v in values if math.isfinite(float(v["frame_degradation"]))]
        if degradations:
            out["frames_worsened_ratio"] = float((np.asarray(degradations) > 0).mean())
            out["worst_frame_degradation"] = float(np.max(degradations))
            out["p95_frame_degradation"] = float(np.percentile(degradations, 95))
        sequence_rows.append(out)
    write_csv(output_dir / "sequence_metrics.csv", sequence_rows)

    aggregate_groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in sequence_rows:
        key = (row["namespace"], row["coverage_threshold"], row["flow_model"], row["backbone"], row["method"])
        aggregate_groups[key].append(row)
    summaries = []
    for key, values in aggregate_groups.items():
        out = dict(zip(["namespace", "coverage_threshold", "flow_model", "backbone", "method"], key))
        out["n_sequences"] = len(values)
        out["n_frames"] = int(sum(int(v["n_frames"]) for v in values))
        for field in ["epe", "bad1", "bad3", "absrel", "boundary_epe", "oracle_gain", "memory_better_ratio", "raw_better_ratio", "support_ratio", "common_valid_ratio", "frames_worsened_ratio", "worst_frame_degradation", "p95_frame_degradation", "flow_latency_ms", "evidence_latency_ms", "total_baseline_latency_ms", "peak_gpu_memory_mb"]:
            numbers = [float(v[field]) for v in values if field in v and math.isfinite(float(v[field]))]
            if numbers:
                out[field] = float(np.mean(numbers))
        summaries.append(out)
    payload = {
        "schema_version": 1,
        "command": command,
        "cache_disparity_convention": "positive left disparity, cache-width pixels",
        "flow_convention": "target-to-source, grid + flow, align_corners=True",
        "native_policy": "cache disparity upsampled to native grid and multiplied by W_native/180; cache-grid flow resized with x/y component scaling; native GT disparity evaluated directly",
        "summaries": summaries,
    }
    (output_dir / "aggregate_summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    for row in rows[1:]:
        for field in row:
            if field not in fields:
                fields.append(field)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def append_csv(path: Path, rows: list[dict]) -> None:
    """Checkpoint one completed evaluation block without rewriting prior blocks."""
    if not rows:
        return
    if not path.exists():
        write_csv(path, rows)
        return
    fields = next(csv.reader(path.open()))
    unknown = sorted(set().union(*(row.keys() for row in rows)) - set(fields))
    if unknown:
        raise ValueError(f"cannot append new CSV fields {unknown}; start a fresh output")
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def read_existing(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open())) if path.exists() else []


def write_readme(output_dir: Path, args, reps: dict[str, str]) -> None:
    (output_dir / "README.md").write_text(f"""# BiDAVideo causal alignment validation

Strictly causal t-1 evaluation of backbone-independent BiDA alignment. No future
frame, stereo feature, backbone identity, cost volume, or backbone confidence is
used. Sequences were selected by the repository's quality-gate statistics:
easy `{reps['easy']}`, difficult `{reps['difficult']}`, boundary-heavy
`{reps['boundary_heavy']}`. Each run uses up to {args.frames} contiguous frames.

Cache metrics are reported independently at coverage thresholds
{', '.join(map(str, args.thresholds))}. Every raw/memory/blend/oracle comparison
uses the identical `GT & raw-valid & aligned-memory-valid & warp-support` mask.
The oracle is a true per-pixel `min(raw error, memory error)` oracle.

Native rows upsample disparity to the native GT grid, multiply disparity by
`W_native/180`, resize both flow components geometrically, and compare directly
with native disparity GT. They are never scalar-rescaled cache EPE values.

The gated baseline uses weight
`0.5 * FB-confidence * (1 - robust-photometric-residual)` within valid support.
It is an untrained diagnostic, not the proposed ARGOS v2 learned fusion.
""")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=V2_ROOT / "results/bidavideo_validation")
    parser.add_argument("--backbones", nargs="+", default=DEFAULT_BACKBONES)
    parser.add_argument("--flow-models", nargs="+", choices=["sea_raft", "raft"], default=["sea_raft"])
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--native-frames", type=int, default=0, help="native-grid prefix length; 0 disables")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reps = representative_sequences()
    sequences = args.sequences or [reps["easy"], reps["difficult"], reps["boundary_heavy"]]
    device = torch.device(args.device)
    frame_csv = args.output / "frame_metrics.csv"
    if not args.resume:
        frame_csv.unlink(missing_ok=True)
    all_rows = read_existing(frame_csv) if args.resume else []
    completed = {
        (r["namespace"], str(r["coverage_threshold"]), r["flow_model"], r["backbone"], r["sequence"], r["frame_id"], r["method"])
        for r in all_rows
    }
    log = (args.output / "run.log").open("a", buffering=1)
    command = " ".join(sys.argv)
    print(f"COMMAND {command}", file=log)
    write_readme(args.output, args, reps)

    for flow_name in args.flow_models:
        adapter = bida.BiDAFlowInferenceAdapter(flow_name, device=device)
        for sequence in sequences:
            info = load_sequence_info(sequence)
            first = args.start
            last = min(len(info.frame_ids), first + args.frames)
            frame_ids = info.frame_ids[first:last]
            if len(frame_ids) < 2:
                raise ValueError(f"{sequence}: need at least two frames")
            def block_complete(backbone: str) -> bool:
                cache_done = all(
                    ("cache", str(threshold), flow_name, backbone, sequence, str(frame_id), "raw") in completed
                    for frame_id in frame_ids[1:]
                    for threshold in args.thresholds
                )
                native_done = (
                    not args.native_frames
                    or all(
                        ("native", "native", flow_name, backbone, sequence, str(frame_id), "raw") in completed
                        for frame_id in frame_ids[1 : args.native_frames]
                    )
                )
                return cache_done and native_done

            pending_backbones = [backbone for backbone in args.backbones if not block_complete(backbone)]
            if not pending_backbones:
                print(f"SKIP complete flow={flow_name} sequence={sequence}", file=log)
                continue
            print(f"START flow={flow_name} sequence={sequence} frames={len(frame_ids)}", file=log)
            rgbs_native = [load_frame_lr(info, frame_id)[0] for frame_id in frame_ids]
            gt_native_data = [load_frame_gt(info, frame_id) for frame_id in frame_ids]
            images_cache = [tensor_image(rgb, CACHE_SIZE, device) for rgb in rgbs_native]
            flows_cp, flows_pc, flow_ms, peak_mb = infer_sequence_flows(
                adapter, images_cache, batch_size=args.batch_size, device=device
            )
            print(f"FLOW flow={flow_name} sequence={sequence} latency_ms={flow_ms:.3f} peak_mb={peak_mb:.1f}", file=log)

            for backbone in pending_backbones:
                disparities, validity, cache_frame_ids, _metadata = load_sequence_cache(backbone, sequence)
                lookup = {str(frame_id): index for index, frame_id in enumerate(cache_frame_ids)}
                block_rows = []
                for local_index in range(1, len(frame_ids)):
                    frame_id = frame_ids[local_index]
                    past_frame_id = frame_ids[local_index - 1]
                    cache_index = lookup[str(frame_id)]
                    past_cache_index = lookup[str(past_frame_id)]
                    raw = np.asarray(disparities[cache_index], dtype=np.float32)
                    past = np.asarray(disparities[past_cache_index], dtype=np.float32)
                    raw_valid = np.asarray(validity[cache_index]) > 0
                    past_valid = np.asarray(validity[past_cache_index]) > 0
                    gt_native, gt_valid_native = gt_native_data[local_index]
                    predictions, evidence, evidence_ms = evaluate_frame(
                        raw=raw,
                        raw_valid=raw_valid,
                        past=past,
                        past_valid=past_valid,
                        gt=np.zeros_like(raw),
                        gt_valid=np.ones_like(raw_valid),
                        current_rgb=images_cache[local_index],
                        past_rgb=images_cache[local_index - 1],
                        flow_current_past=flows_cp[local_index - 1],
                        flow_past_current=flows_pc[local_index - 1],
                        device=device,
                    )
                    for threshold in args.thresholds:
                        key = ("cache", str(threshold), flow_name, backbone, sequence, str(frame_id), "raw")
                        if key in completed:
                            continue
                        gt_cache, gt_valid_cache = resize_gt_to_cache_corrected(
                            gt_native, gt_valid_native, gt_native.shape[1], threshold
                        )
                        block_rows.extend(make_rows(
                            namespace="cache", flow_model=flow_name, backbone=backbone,
                            sequence=sequence, frame_id=str(frame_id), frame_index=first + local_index,
                            threshold=threshold, predictions=predictions, evidence=evidence,
                            gt=gt_cache, gt_valid=gt_valid_cache, raw_valid=raw_valid,
                            flow_latency_ms=flow_ms, evidence_latency_ms=evidence_ms,
                            peak_gpu_memory_mb=peak_mb,
                        ))
                        if (
                            flow_name == "sea_raft"
                            and backbone == "S2M2-S"
                            and abs(local_index - len(frame_ids) // 2) == 0
                            and abs(threshold - 0.25) < 1e-9
                        ):
                            save_contact_sheet(
                                args.output / "contact_sheets" / f"{sequence}_{frame_id}.png",
                                rgbs_native[local_index], gt_cache, gt_valid_cache & raw_valid,
                                predictions,
                            )

                    if args.native_frames and local_index < args.native_frames:
                        key = ("native", "native", flow_name, backbone, sequence, str(frame_id), "raw")
                        if key not in completed:
                            h, w = gt_native.shape
                            raw_n, raw_vn, past_n, past_vn = native_inputs(raw, raw_valid, past, past_valid, (h, w))
                            flow_cp_n = bida.resize_flow(torch.from_numpy(flows_cp[local_index - 1])[None].to(device), (h, w))[0].cpu().numpy()
                            flow_pc_n = bida.resize_flow(torch.from_numpy(flows_pc[local_index - 1])[None].to(device), (h, w))[0].cpu().numpy()
                            current_native = tensor_image(rgbs_native[local_index], (h, w), device)
                            past_native = tensor_image(rgbs_native[local_index - 1], (h, w), device)
                            predictions_n, evidence_n, evidence_ms_n = evaluate_frame(
                                raw=raw_n, raw_valid=raw_vn, past=past_n, past_valid=past_vn,
                                gt=gt_native, gt_valid=gt_valid_native,
                                current_rgb=current_native, past_rgb=past_native,
                                flow_current_past=flow_cp_n, flow_past_current=flow_pc_n, device=device,
                            )
                            block_rows.extend(make_rows(
                                namespace="native", flow_model=flow_name, backbone=backbone,
                                sequence=sequence, frame_id=str(frame_id), frame_index=first + local_index,
                                threshold=None, predictions=predictions_n, evidence=evidence_n,
                                gt=gt_native, gt_valid=gt_valid_native, raw_valid=raw_vn,
                                flow_latency_ms=flow_ms, evidence_latency_ms=evidence_ms_n,
                                peak_gpu_memory_mb=peak_mb,
                            ))
                all_rows.extend(block_rows)
                append_csv(frame_csv, block_rows)
                print(f"DONE flow={flow_name} backbone={backbone} sequence={sequence} rows={len(block_rows)}", file=log)
        del adapter.model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    aggregate_rows(all_rows, args.output, command)
    print(f"COMPLETE rows={len(all_rows)}", file=log)
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
