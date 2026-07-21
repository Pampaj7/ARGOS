#!/usr/bin/env python3
"""Large-scale, training-free causal BiDA t-1 signal audit for ARGOS v2.

The runner consumes only validated frozen disparity caches, SCARED-C processed
geometric targets, frozen SEA-RAFT flow, and the canonical ARGOS v2 BiDA warp.
It intentionally contains no learned fusion, proposal, detector, or calibration.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

V2_ROOT = Path(__file__).resolve().parents[1]
ROOT = V2_ROOT.parent
sys.path.insert(0, str(V2_ROOT))
sys.path.insert(0, str(V2_ROOT / "scripts"))

from argos_v2.cache_io import load_sequence_cache  # noqa: E402
from argos_v2.scared_c_data import load_frame_gt, load_sequence_info, read_rgb  # noqa: E402
from argos_v2.sequences import accepted_sequences, load_quality_gate_rows, representative_sequences  # noqa: E402
from model_design.data.temporal_pair_dataset import resize_gt_to_cache_masked  # noqa: E402


def _load_bida():
    path = V2_ROOT / "model_design/external_components/bidavideo.py"
    spec = importlib.util.spec_from_file_location("argos_v2_large_scale_bida", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bida = _load_bida()
CACHE_SIZE = (144, 180)
SEEN_BACKBONES = ("S2M2-S", "RAFT-Stereo", "StereoAnywhere")
UNSEEN_BACKBONES = ("Fast-FoundationStereo", "CREStereo")
SUM_FIELDS = (
    "raw_error_sum", "memory_error_sum", "oracle_error_sum",
    "helpful_sum", "harmful_sum", "temporal_difference_sum",
    "gt_relative_temporal_consistency_sum",
)
COUNT_FIELDS = (
    "valid_pixel_count", "memory_better_count", "memory_worse_count", "tie_count",
    "raw_bad1_count", "raw_bad3_count", "raw_bad5_count",
    "memory_bad1_count", "memory_bad3_count", "memory_bad5_count",
    "oracle_bad1_count", "oracle_bad3_count", "oracle_bad5_count",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    if not path.exists():
        write_csv(path, rows)
        return
    fields = next(csv.reader(path.open()))
    unknown = set().union(*(row.keys() for row in rows)) - set(fields)
    if unknown:
        raise ValueError(f"new fields in resumed block: {sorted(unknown)}")
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writerows(rows)
        handle.flush()


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open())) if path.exists() else []


def boundary_mask(gt: np.ndarray, gt_valid: np.ndarray) -> np.ndarray:
    dx = cv2.Sobel(gt, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gt, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.hypot(dx, dy) > 1.0
    validity_edge = cv2.morphologyEx(
        gt_valid.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
    ) > 0
    expanded = cv2.dilate((edge | validity_edge).astype(np.uint8), np.ones((3, 3), np.uint8))
    return expanded.astype(bool) & gt_valid


def metric_sums(
    raw_error: np.ndarray,
    memory_error: np.ndarray,
    mask: np.ndarray,
    *,
    temporal_difference: np.ndarray | None = None,
) -> dict:
    raw = raw_error[mask].astype(np.float64)
    memory = memory_error[mask].astype(np.float64)
    oracle = np.minimum(raw, memory)
    better = memory < raw
    worse = memory > raw
    tie = ~(better | worse)
    temporal = np.abs(raw - memory) if temporal_difference is None else temporal_difference[mask].astype(np.float64)
    result = {
        "valid_pixel_count": int(raw.size),
        "memory_better_count": int(better.sum()),
        "memory_worse_count": int(worse.sum()),
        "tie_count": int(tie.sum()),
        "raw_error_sum": float(raw.sum()),
        "memory_error_sum": float(memory.sum()),
        "oracle_error_sum": float(oracle.sum()),
        "helpful_sum": float((raw[better] - memory[better]).sum()),
        "harmful_sum": float((memory[worse] - raw[worse]).sum()),
        "temporal_difference_sum": float(temporal.sum()),
        "gt_relative_temporal_consistency_sum": float(np.abs(raw - memory).sum()),
    }
    for threshold in (1, 3, 5):
        result[f"raw_bad{threshold}_count"] = int((raw > threshold).sum())
        result[f"memory_bad{threshold}_count"] = int((memory > threshold).sum())
        result[f"oracle_bad{threshold}_count"] = int((oracle > threshold).sum())
    if raw.size:
        if not np.all(oracle <= raw + 1e-7) or not np.all(oracle <= memory + 1e-7):
            raise AssertionError("oracle invariant violated")
    return result


def finalize_sums(row: dict) -> dict:
    count = int(float(row.get("valid_pixel_count", 0)))
    better = int(float(row.get("memory_better_count", 0)))
    worse = int(float(row.get("memory_worse_count", 0)))
    out = dict(row)
    if count == 0:
        for key in (
            "raw_epe", "memory_epe", "oracle_epe", "oracle_epe_gain",
            "relative_oracle_gain", "memory_better_fraction", "memory_worse_fraction",
            "tie_fraction", "mean_helpful_magnitude", "mean_harmful_magnitude",
            "raw_bad1", "raw_bad3", "raw_bad5", "memory_bad1", "memory_bad3",
            "memory_bad5", "oracle_bad1", "oracle_bad3", "oracle_bad5",
            "flow_warped_raw_temporal_difference", "gt_relative_temporal_error_consistency",
        ):
            out[key] = math.nan
        return out
    out["raw_epe"] = float(row["raw_error_sum"]) / count
    out["memory_epe"] = float(row["memory_error_sum"]) / count
    out["oracle_epe"] = float(row["oracle_error_sum"]) / count
    out["oracle_epe_gain"] = out["raw_epe"] - out["oracle_epe"]
    out["relative_oracle_gain"] = out["oracle_epe_gain"] / max(out["raw_epe"], 1e-12)
    out["memory_better_fraction"] = better / count
    out["memory_worse_fraction"] = worse / count
    out["tie_fraction"] = int(float(row["tie_count"])) / count
    out["mean_helpful_magnitude"] = float(row["helpful_sum"]) / max(better, 1)
    out["mean_harmful_magnitude"] = float(row["harmful_sum"]) / max(worse, 1)
    for method in ("raw", "memory", "oracle"):
        for threshold in (1, 3, 5):
            out[f"{method}_bad{threshold}"] = int(float(row[f"{method}_bad{threshold}_count"])) / count
    out["flow_warped_raw_temporal_difference"] = float(row["temporal_difference_sum"]) / count
    out["gt_relative_temporal_error_consistency"] = (
        float(row["gt_relative_temporal_consistency_sum"]) / count
    )
    return out


def sum_metric_rows(rows: list[dict], base: dict | None = None) -> dict:
    out = dict(base or {})
    for field in SUM_FIELDS:
        out[field] = sum(float(row.get(field, 0)) for row in rows)
    for field in COUNT_FIELDS:
        out[field] = sum(int(float(row.get(field, 0))) for row in rows)
    out["valid_causal_pair_count"] = sum(
        int(float(row.get("valid_causal_pair_count", row.get("valid_causal_pair", 0))))
        for row in rows
    )
    out["causal_pair_count"] = sum(int(float(row.get("causal_pair_count", 1))) for row in rows)
    return finalize_sums(out)


def region_masks(
    common: np.ndarray,
    boundary: np.ndarray,
    flow_magnitude: np.ndarray,
    fb_consistent: np.ndarray,
    raw_error: np.ndarray,
    gt: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "all": common,
        "boundary": common & boundary,
        "non_boundary": common & ~boundary,
        "motion_low_lt_1px": common & (flow_magnitude < 1.0),
        "motion_high_ge_1px": common & (flow_magnitude >= 1.0),
        "fb_consistent": common & fb_consistent,
        "fb_inconsistent_occlusion_like": common & ~fb_consistent,
        "raw_error_low_le_1px": common & (raw_error <= 1.0),
        "raw_error_mid_1_to_3px": common & (raw_error > 1.0) & (raw_error <= 3.0),
        "raw_error_high_gt_3px": common & (raw_error > 3.0),
        "gt_disparity_0_to_2px": common & (gt >= 0.0) & (gt < 2.0),
        "gt_disparity_2_to_4px": common & (gt >= 2.0) & (gt < 4.0),
        "gt_disparity_4_to_8px": common & (gt >= 4.0) & (gt < 8.0),
        "gt_disparity_ge_8px": common & (gt >= 8.0),
    }


def bootstrap_mean(values: np.ndarray, *, seed: int, replicates: int = 10000) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return math.nan, math.nan
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    block = 1000
    for start in range(0, replicates, block):
        stop = min(start + block, replicates)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[indices].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def load_sequence_arrays(sequence: str, workers: int) -> tuple[object, np.ndarray, np.ndarray, np.ndarray]:
    info = load_sequence_info(sequence)
    cv2.setNumThreads(1)

    def load_one(frame_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rgb = read_rgb(info.seq_dir / "left" / f"{frame_id}.png")
        rgb = cv2.resize(rgb, (CACHE_SIZE[1], CACHE_SIZE[0]), interpolation=cv2.INTER_AREA)
        gt_native, valid_native = load_frame_gt(info, frame_id)
        gt, coverage = resize_gt_to_cache_masked(gt_native, valid_native)
        return rgb.astype(np.uint8), gt, coverage

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        loaded = list(pool.map(load_one, info.frame_ids))
    rgbs = np.stack([item[0] for item in loaded])
    gt = np.stack([item[1] for item in loaded])
    coverage = np.stack([item[2] for item in loaded])
    return info, rgbs, gt, coverage


def infer_flow_pair_batch(adapter, rgb: np.ndarray, indices: np.ndarray, device: torch.device):
    current = torch.from_numpy(rgb[indices]).permute(0, 3, 1, 2).float().to(device)
    past = torch.from_numpy(rgb[indices - 1]).permute(0, 3, 1, 2).float().to(device)
    targets = torch.cat((current, past), dim=0)
    sources = torch.cat((past, current), dim=0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    both = adapter.infer(targets, sources)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    n = len(indices)
    return both[:n], both[n:], current, past, elapsed_ms / max(n, 1)


def save_contact_sheet(path: Path, rgb: np.ndarray, gt: np.ndarray, mask: np.ndarray,
                       raw: np.ndarray, memory: np.ndarray) -> None:
    raw_error = np.abs(raw - gt)
    memory_error = np.abs(memory - gt)
    oracle_choice = (memory_error < raw_error).astype(np.uint8) * 255
    values = np.concatenate((raw_error[mask], memory_error[mask])) if mask.any() else np.array([1.0])
    vmax = max(1.0, float(np.quantile(values, 0.95)))
    panels = [cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)]
    for error in (raw_error, memory_error):
        visual = np.clip(error / vmax * 255.0, 0, 255).astype(np.uint8)
        visual[~mask] = 0
        panels.append(cv2.applyColorMap(visual, cv2.COLORMAP_TURBO))
    choice = cv2.applyColorMap(oracle_choice, cv2.COLORMAP_VIRIDIS)
    choice[~mask] = 0
    panels.append(choice)
    canvas = np.concatenate(panels, axis=1)
    labels = ("RGB", "raw error", "memory error", "oracle uses memory")
    for index, label in enumerate(labels):
        cv2.putText(canvas, label, (index * CACHE_SIZE[1] + 4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 255, 255), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def metric_definitions() -> dict:
    return {
        "grid": "cache grid 144x180",
        "disparity_units": "positive left disparity, pixels at cache width 180",
        "gt_resize": "resize(disparity * valid) / resize(valid), then multiply by 180/native_width",
        "primary_gt_coverage_threshold": 0.50,
        "flow": "SEA-RAFT first-image to second-image; current->past for target-to-source warp",
        "warp": "source sampled at integer target grid + target-to-source flow; bilinear, zeros, align_corners=True",
        "common_mask": "GT coverage > 0.50 & current prediction valid & sampled past valid & in-bounds warp support",
        "occlusion": "out-of-bounds and invalid sampled-source pixels are excluded; FB-inconsistent but sampleable pixels are a reported region",
        "raw_error": "abs(raw_t - GT_t)",
        "memory_error": "abs(warp(disparity_t-1, flow_t->t-1) - GT_t)",
        "oracle_error": "min(raw_error, memory_error) per pixel",
        "ties": "exact float32 equality after error computation",
        "boundary": "Sobel GT disparity magnitude >1 cache px or GT-valid boundary, dilated by 3x3",
        "motion_bins": {"low": "flow magnitude <1 cache px", "high": "flow magnitude >=1 cache px"},
        "raw_error_bins": ["<=1 px", ">1 and <=3 px", ">3 px"],
        "gt_disparity_bins": ["[0,2) px", "[2,4) px", "[4,8) px", "[8,inf) px"],
        "stereo_confidence_bins": "not evaluated: canonical caches expose only prediction-valid masks, not backbone-independent confidence",
        "temporal_difference": "abs(raw_t - aligned_raw_t-1), reported separately from accuracy",
        "gt_relative_temporal_error_consistency": "abs(raw_error - memory_error); lower is not interpreted as higher accuracy",
        "aggregation": "pixel-pooled geometry plus equal-sequence mean/std and sequence-unit bootstrap confidence intervals",
        "bootstrap": "10,000 deterministic resamples of complete sequences with replacement",
    }


def split_audit(backbones: list[str], sequences: list[str], role: str) -> dict:
    cache_root = V2_ROOT / "cache_scaredc_backbones"
    inventory = []
    for backbone in backbones:
        for sequence in sequences:
            disparity, valid, frame_ids, metadata = load_sequence_cache(backbone, sequence)
            cache_dir = cache_root / backbone / sequence
            inventory.append({
                "role": role, "backbone": backbone, "sequence": sequence,
                "frame_count": int(len(frame_ids)), "causal_pair_count": int(len(frame_ids) - 1),
                "shape": list(disparity.shape), "disparity_dtype": str(disparity.dtype),
                "valid_dtype": str(valid.dtype), "complete_flag": (cache_dir / ".complete").exists(),
                "metadata_completion_status": bool(metadata.get("completion_status")),
                "frame_ids_sha256": sha256(cache_dir / "frame_ids.npy"),
            })
    quality_rows = load_quality_gate_rows()
    return {
        "schema_version": 1,
        "quality_gate_source": str(ROOT / "dataset/SCARED-C/curated/manifests/quality_gate.csv"),
        "accepted_sequences": accepted_sequences(),
        "rejected_sequences": [row["sequence_id"] for row in quality_rows if row["status"] != "pass"],
        "evaluated_sequences": sequences,
        "independent_sequence_count": len(sequences),
        "backbones": backbones,
        "role": role,
        "pairs_per_backbone": sum(item["causal_pair_count"] for item in inventory if item["backbone"] == backbones[0]),
        "inventory": inventory,
        "canonical_bida_sha256": sha256(V2_ROOT / "model_design/external_components/bidavideo.py"),
        "sea_raft_checkpoint": str(bida.SEA_RAFT_CHECKPOINT),
        "sea_raft_checkpoint_sha256": sha256(bida.SEA_RAFT_CHECKPOINT),
    }


def evaluate(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    sequences = list(args.sequences or accepted_sequences())
    definitions = metric_definitions()
    (args.output / "metric_definitions.json").write_text(json.dumps(definitions, indent=2))
    audit = split_audit(args.backbones, sequences, args.role)
    (args.output / "split_audit.json").write_text(json.dumps(audit, indent=2))
    frame_path = args.output / "frame_metrics.csv"
    regional_path = args.output / "regional_metrics.csv"
    existing = read_csv(frame_path) if args.resume else []
    completed = {(row["backbone"], row["sequence"], row["frame_id"]) for row in existing}
    if not args.resume:
        frame_path.unlink(missing_ok=True)
        regional_path.unlink(missing_ok=True)
    log = (args.output / "run.log").open("a", buffering=1)
    print("COMMAND " + " ".join(sys.argv), file=log)
    device = torch.device(args.device)
    adapter = bida.BiDAFlowInferenceAdapter("sea_raft", device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    representatives = set(representative_sequences().values())

    for sequence in sequences:
        if all(
            all((backbone, sequence, str(frame_id)) in completed for frame_id in load_sequence_info(sequence).frame_ids[1:])
            for backbone in args.backbones
        ):
            print(f"SKIP complete sequence={sequence}", file=log)
            continue
        sequence_start = time.perf_counter()
        info, rgb, gt, coverage = load_sequence_arrays(sequence, args.workers)
        pair_indices = np.arange(1, len(info.frame_ids), dtype=np.int64)
        if args.max_pairs is not None:
            pair_indices = pair_indices[: args.max_pairs]
        caches = {}
        reference_ids = np.asarray(info.frame_ids).astype(str)
        for backbone in args.backbones:
            disparity, validity, cache_ids, metadata = load_sequence_cache(backbone, sequence)
            if not np.array_equal(np.asarray(cache_ids).astype(str), reference_ids):
                raise AssertionError(f"frame-ID mismatch: {backbone}/{sequence}")
            if tuple(disparity.shape[1:]) != CACHE_SIZE or metadata.get("disparity_convention") != "positive_left_disparity":
                raise AssertionError(f"cache contract mismatch: {backbone}/{sequence}")
            caches[backbone] = (disparity, validity)

        frame_blocks: dict[str, list[dict]] = {backbone: [] for backbone in args.backbones}
        region_accumulators: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for batch_start in range(0, len(pair_indices), args.batch_size):
            indices = pair_indices[batch_start : batch_start + args.batch_size]
            flow_cp, flow_pc, current_rgb, past_rgb, flow_ms = infer_flow_pair_batch(
                adapter, rgb, indices, device
            )
            fb = bida.forward_backward_consistency(flow_cp, flow_pc)
            flow_magnitude = torch.linalg.vector_norm(flow_cp, dim=1).detach().cpu().numpy()
            fb_valid = fb.valid[:, 0].detach().cpu().numpy().astype(bool)

            for backbone in args.backbones:
                disparity, validity = caches[backbone]
                raw_np = np.asarray(disparity[indices], dtype=np.float32)
                past_np = np.asarray(disparity[indices - 1], dtype=np.float32)
                raw_valid_np = np.asarray(validity[indices]) > 0
                past_valid_np = np.asarray(validity[indices - 1]) > 0
                raw_t = torch.from_numpy(raw_np)[:, None].to(device)
                past_t = torch.from_numpy(past_np)[:, None].to(device)
                past_valid_t = torch.from_numpy(past_valid_np)[:, None].to(device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                evidence_start = time.perf_counter()
                aligned = bida.causal_warp(past_t, flow_cp, source_valid=past_valid_t)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                evidence_ms = (time.perf_counter() - evidence_start) * 1000.0 / max(len(indices), 1)
                memory_np = aligned.warped[:, 0].detach().cpu().numpy()
                support_np = aligned.support[:, 0].detach().cpu().numpy().astype(bool)
                aligned_valid_np = aligned.valid[:, 0].detach().cpu().numpy().astype(bool) & raw_valid_np

                for local, index in enumerate(indices):
                    frame_id = str(info.frame_ids[int(index)])
                    if (backbone, sequence, frame_id) in completed:
                        continue
                    gt_valid = coverage[index] > args.coverage_threshold
                    common = gt_valid & raw_valid_np[local] & aligned_valid_np[local] & support_np[local]
                    raw_error = np.abs(raw_np[local] - gt[index])
                    memory_error = np.abs(memory_np[local] - gt[index])
                    temporal_difference = np.abs(raw_np[local] - memory_np[local])
                    sums = metric_sums(
                        raw_error, memory_error, common, temporal_difference=temporal_difference
                    )
                    row = finalize_sums({
                        "role": args.role, "backbone": backbone, "sequence": sequence,
                        "frame_id": frame_id, "frame_index": int(index), "memory_age": 1,
                        "coverage_threshold": args.coverage_threshold,
                        "causal_pair_count": 1, "valid_causal_pair": int(common.any()),
                        "gt_valid_count": int(gt_valid.sum()),
                        "raw_valid_count": int(raw_valid_np[local].sum()),
                        "warp_support_count": int(support_np[local].sum()),
                        "aligned_valid_count": int(aligned_valid_np[local].sum()),
                        "fb_consistent_count": int((fb_valid[local] & common).sum()),
                        "fb_inconsistent_evaluable_count": int((~fb_valid[local] & common).sum()),
                        "out_of_bounds_count": int((~support_np[local]).sum()),
                        "flow_latency_ms": flow_ms, "warp_latency_ms": evidence_ms,
                        **sums,
                    })
                    frame_blocks[backbone].append(row)
                    boundary = boundary_mask(gt[index], gt_valid)
                    for region, region_mask in region_masks(
                        common, boundary, flow_magnitude[local], fb_valid[local], raw_error, gt[index]
                    ).items():
                        region_accumulators[(backbone, region)].append(
                            {**metric_sums(
                                raw_error, memory_error, region_mask,
                                temporal_difference=temporal_difference,
                            ),
                             "valid_causal_pair": int(region_mask.any()), "causal_pair_count": 1}
                        )
                    if (
                        args.contact_sheets and backbone == "S2M2-S" and sequence in representatives
                        and int(index) == int(pair_indices[len(pair_indices) // 2])
                    ):
                        save_contact_sheet(
                            args.output / "contact_sheets" / f"{sequence}_{frame_id}.png",
                            rgb[index], gt[index], common, raw_np[local], memory_np[local]
                        )

        block_rows = [row for backbone in args.backbones for row in frame_blocks[backbone]]
        append_csv(frame_path, block_rows)
        regional_rows = []
        for (backbone, region), values in region_accumulators.items():
            regional_rows.append(sum_metric_rows(values, {
                "role": args.role, "backbone": backbone, "sequence": sequence,
                "region": region, "coverage_threshold": args.coverage_threshold,
            }))
        append_csv(regional_path, regional_rows)
        print(
            f"DONE sequence={sequence} frames={len(pair_indices)} backbones={len(args.backbones)} "
            f"elapsed_s={time.perf_counter() - sequence_start:.1f}", file=log
        )
    peak_mb = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
    print(f"COMPLETE peak_gpu_memory_mb={peak_mb:.1f}", file=log)
    log.close()


def aggregate(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    frame_rows: list[dict] = []
    regional_rows: list[dict] = []
    audits = []
    seen_keys = set()
    for input_dir in args.inputs:
        audits.append(json.loads((input_dir / "split_audit.json").read_text()))
        for row in read_csv(input_dir / "frame_metrics.csv"):
            key = (row["backbone"], row["sequence"], row["frame_id"])
            if key in seen_keys:
                raise ValueError(f"duplicate frame row: {key}")
            seen_keys.add(key)
            frame_rows.append(row)
        regional_rows.extend(read_csv(input_dir / "regional_metrics.csv"))
        source_contacts = input_dir / "contact_sheets"
        if source_contacts.exists():
            target = args.output / "contact_sheets"
            target.mkdir(exist_ok=True)
            for image in source_contacts.glob("*.png"):
                shutil.copy2(image, target / image.name)
    frame_rows.sort(key=lambda row: (row["role"], row["backbone"], row["sequence"], int(row["frame_index"])))
    write_csv(args.output / "frame_metrics.csv", frame_rows)

    sequence_groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in frame_rows:
        sequence_groups[(row["role"], row["backbone"], row["sequence"])].append(row)
    sequence_rows = []
    for (role, backbone, sequence), values in sorted(sequence_groups.items()):
        row = sum_metric_rows(values, {
            "role": role, "backbone": backbone, "sequence": sequence,
            "coverage_threshold": args.coverage_threshold,
        })
        row["frame_count"] = len(values)
        row["mean_frame_oracle_gain"] = float(np.mean([float(value["oracle_epe_gain"]) for value in values]))
        sequence_rows.append(row)
    write_csv(args.output / "sequence_metrics.csv", sequence_rows)

    backbone_rows = []
    for (role, backbone), values in sorted(_group(sequence_rows, ("role", "backbone")).items()):
        pooled = sum_metric_rows(values, {
            "role": role, "backbone": backbone, "coverage_threshold": args.coverage_threshold,
            "independent_sequence_count": len(values),
        })
        gains = np.asarray([float(row["oracle_epe_gain"]) for row in values])
        lo, hi = bootstrap_mean(gains, seed=args.bootstrap_seed)
        pooled.update({
            "sequence_mean_oracle_gain": float(gains.mean()),
            "sequence_std_oracle_gain": float(gains.std(ddof=1)) if len(gains) > 1 else 0.0,
            "sequence_bootstrap_ci95_low": lo, "sequence_bootstrap_ci95_high": hi,
            "positive_sequence_fraction": float((gains > 0).mean()),
        })
        backbone_rows.append(pooled)
    write_csv(args.output / "backbone_metrics.csv", backbone_rows)

    combined_regional = []
    for key, values in sorted(_group(regional_rows, ("role", "backbone", "sequence", "region")).items()):
        role, backbone, sequence, region = key
        combined_regional.append(sum_metric_rows(values, {
            "role": role, "backbone": backbone, "sequence": sequence, "region": region,
            "aggregation_level": "sequence", "coverage_threshold": args.coverage_threshold,
        }))
    regional_sequence_rows = list(combined_regional)
    for (role, backbone, region), values in sorted(
        _group(regional_sequence_rows, ("role", "backbone", "region")).items()
    ):
        combined_regional.append(sum_metric_rows(values, {
            "role": role, "backbone": backbone, "sequence": "ALL", "region": region,
            "aggregation_level": "pooled_backbone", "coverage_threshold": args.coverage_threshold,
        }))
    for (role, region), values in sorted(_group(regional_sequence_rows, ("role", "region")).items()):
        combined_regional.append(sum_metric_rows(values, {
            "role": role, "backbone": "ALL", "sequence": "ALL", "region": region,
            "aggregation_level": "pooled_role", "coverage_threshold": args.coverage_threshold,
        }))
    write_csv(args.output / "regional_metrics.csv", combined_regional)

    role_summaries = {}
    for role, values in sorted(_group(sequence_rows, ("role",)).items()):
        by_sequence = _group(values, ("sequence",))
        sequence_gains = np.asarray([
            np.mean([float(row["oracle_epe_gain"]) for row in rows]) for rows in by_sequence.values()
        ])
        lo, hi = bootstrap_mean(sequence_gains, seed=args.bootstrap_seed)
        pooled = sum_metric_rows(values, {
            "independent_sequence_count": len(by_sequence),
            "backbone_sequence_group_count": len(values),
        })
        pooled.update({
            "sequence_mean_oracle_gain": float(sequence_gains.mean()),
            "sequence_std_oracle_gain": float(sequence_gains.std(ddof=1)),
            "sequence_bootstrap_ci95_low": lo, "sequence_bootstrap_ci95_high": hi,
            "positive_backbone_sequence_fraction": float(np.mean([float(row["oracle_epe_gain"]) > 0 for row in values])),
        })
        role_summaries[role[0] if isinstance(role, tuple) else role] = pooled

    seen = role_summaries.get("seen")
    unseen = role_summaries.get("unseen")
    if seen and unseen:
        strong = (
            seen["sequence_bootstrap_ci95_low"] > 0 and unseen["sequence_bootstrap_ci95_low"] > 0
            and seen["positive_backbone_sequence_fraction"] >= 0.90
            and unseen["positive_backbone_sequence_fraction"] >= 0.90
        )
        classification = "STRONG SIGNAL" if strong else "WEAK/CONDITIONAL SIGNAL"
        if seen["sequence_bootstrap_ci95_high"] <= 0 or unseen["sequence_bootstrap_ci95_high"] <= 0:
            classification = "NO RELIABLE SIGNAL"
    else:
        classification = "SEEN AUDIT COMPLETE; UNSEEN AUDIT PENDING"
    payload = {
        "schema_version": 1,
        "classification": classification,
        "role_summaries": role_summaries,
        "decision_rule": (
            "STRONG requires positive seen and unseen sequence-bootstrap CI lower bounds and "
            ">=90% positive backbone-sequence groups; no learned threshold is used."
        ),
        "metric_namespace": "cache-grid-from-cached-predictions",
        "coverage_threshold": args.coverage_threshold,
    }
    (args.output / "aggregate_summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False))
    (args.output / "metric_definitions.json").write_text(json.dumps(metric_definitions(), indent=2))
    merged_audit = {
        "schema_version": 1,
        "accepted_sequences": accepted_sequences(),
        "independent_sequence_count": len(accepted_sequences()),
        "pairs_per_backbone": sum(len(load_sequence_info(sequence).frame_ids) - 1 for sequence in accepted_sequences()),
        "shards": audits,
        "no_training": True,
        "learned_components_used": [],
    }
    (args.output / "split_audit.json").write_text(json.dumps(merged_audit, indent=2))
    make_plots(args.output, sequence_rows, backbone_rows)
    write_readme(args.output, payload, backbone_rows)
    log_lines = ["# ARGOS v2 large-scale causal BiDA signal audit", ""]
    for input_dir in args.inputs:
        log_lines.append(f"## SHARD {input_dir}")
        log_lines.append((input_dir / "run.log").read_text().rstrip())
        log_lines.append("")
    log_lines.extend([
        "## AGGREGATION",
        "AGGREGATE_COMMAND " + " ".join(sys.argv),
        f"COMPLETE rows={len(frame_rows)} classification={classification}",
        "",
    ])
    (args.output / "run.log").write_text("\n".join(log_lines))


def _group(rows: list[dict], fields: tuple[str, ...]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    return groups


def make_plots(output: Path, sequence_rows: list[dict], backbone_rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plot_dir = output / "plots"
    plot_dir.mkdir(exist_ok=True)
    backbones = sorted(set(row["backbone"] for row in sequence_rows))
    data = [[float(row["oracle_epe_gain"]) for row in sequence_rows if row["backbone"] == backbone]
            for backbone in backbones]
    fig, axis = plt.subplots(figsize=(max(7, len(backbones) * 1.5), 4))
    axis.boxplot(data, tick_labels=backbones, showmeans=True)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("Per-sequence oracle gain (cache px)")
    axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(plot_dir / "oracle_gain_by_backbone.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4))
    labels = [row["backbone"] for row in backbone_rows]
    fractions = [float(row["memory_better_fraction"]) for row in backbone_rows]
    axis.bar(labels, fractions)
    axis.set_ylabel("Memory-better pixel fraction")
    axis.set_ylim(0, 1)
    axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(plot_dir / "memory_better_fraction.png", dpi=150)
    plt.close(fig)


def write_readme(output: Path, summary: dict, backbone_rows: list[dict]) -> None:
    table = ["| Role | Backbone | Raw EPE | Memory EPE | Oracle EPE | Gain | Relative gain | Better pixels | Seq. CI95 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in backbone_rows:
        table.append(
            f"| {row['role']} | {row['backbone']} | {float(row['raw_epe']):.6f} | "
            f"{float(row['memory_epe']):.6f} | {float(row['oracle_epe']):.6f} | "
            f"{float(row['oracle_epe_gain']):.6f} | {100*float(row['relative_oracle_gain']):.2f}% | "
            f"{100*float(row['memory_better_fraction']):.2f}% | "
            f"[{float(row['sequence_bootstrap_ci95_low']):.6f}, {float(row['sequence_bootstrap_ci95_high']):.6f}] |"
        )
    output.joinpath("README.md").write_text(
        "# ARGOS v2 large-scale causal BiDA signal audit\n\n"
        "Training-free evaluation over every causal pair in all 17 accepted SCARED-C sequences. "
        "It uses only frozen disparity caches, frozen SEA-RAFT, and canonical causal BiDA warping.\n\n"
        f"**Classification: {summary['classification']}**\n\n" + "\n".join(table) +
        "\n\nThe unit of uncertainty is the complete sequence, not the pixel. Geometry is "
        "reported on the cache grid at GT coverage >0.50 using coverage-normalized GT resize. "
        "Temporal-difference metrics are reported separately and are not interpreted as accuracy. "
        "Exact shard commands, PIDs as recorded during execution, and aggregation command are in `run.log`.\n\n"
        "Audit trace: a first complete run was superseded before interpretation because its temporal-difference "
        "field accidentally used error difference. Geometry/oracle values were recomputed as well; this report is "
        "the corrected run, where temporal difference is `abs(raw_t - aligned_raw_t-1)`.\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("evaluate", "aggregate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+")
    parser.add_argument("--backbones", nargs="+", default=list(SEEN_BACKBONES))
    parser.add_argument("--sequences", nargs="+")
    parser.add_argument("--role", choices=("seen", "unseen"), default="seen")
    parser.add_argument("--coverage-threshold", type=float, default=0.50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--contact-sheets", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260717)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "evaluate":
        evaluate(args)
    else:
        if not args.inputs:
            raise ValueError("--inputs is required in aggregate mode")
        aggregate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
