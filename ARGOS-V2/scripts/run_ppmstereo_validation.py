#!/usr/bin/env python3
"""Stage-gated causal PPMStereo memory validation for ARGOS v2.

The initial ``oracle`` mode deliberately trains nothing. It measures whether
BiDA-aligned ages 2/4/8 add per-pixel evidence beyond aligned t-1. Only compact
CSV/JSON summaries are written; flow and disparity predictions remain in RAM.
"""
from __future__ import annotations

import argparse
import csv
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

V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))
sys.path.insert(0, str(V2_ROOT / "scripts"))

from argos_v2.cache_io import load_sequence_cache  # noqa: E402
from argos_v2.scared_c_data import load_frame_gt, load_frame_lr, load_sequence_info  # noqa: E402
from model_design.data.temporal_pair_dataset import (  # noqa: E402
    DEFAULT_VALIDATION_SEQUENCES,
    SEEN_BACKBONES,
    build_split_manifest,
    resize_gt_to_cache_masked,
)
from model_design.external_components.bidavideo import (  # noqa: E402
    BiDAFlowInferenceAdapter,
    temporal_disparity_evidence,
)

CACHE_HEIGHT = 144
CACHE_WIDTH = 180
DEFAULT_AGES = (1, 2, 4, 8)
DEFAULT_THRESHOLDS = (0.05, 0.25, 0.50, 0.90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("oracle",), default="oracle")
    parser.add_argument(
        "--output", type=Path, default=V2_ROOT / "results/ppmstereo_validation"
    )
    parser.add_argument("--backbones", nargs="+", default=list(SEEN_BACKBONES))
    parser.add_argument(
        "--sequences", nargs="+", default=list(DEFAULT_VALIDATION_SEQUENCES)
    )
    parser.add_argument("--ages", nargs="+", type=int, default=list(DEFAULT_AGES))
    parser.add_argument("--thresholds", nargs="+", type=float, default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def save_json(path: Path, value: object) -> None:
    def clean(item):
        if isinstance(item, dict):
            return {str(key): clean(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(child) for child in item]
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    if not path.exists():
        write_csv(path, rows)
        return
    with path.open() as handle:
        fields = next(csv.reader(handle))
    unknown = set().union(*(row.keys() for row in rows)) - set(fields)
    if unknown:
        raise RuntimeError(f"new CSV fields during resume: {sorted(unknown)}")
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def rgb_tensor(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    resized = cv2.resize(rgb, (CACHE_WIDTH, CACHE_HEIGHT), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(np.ascontiguousarray(resized)).permute(2, 0, 1).float().to(device)


def boundary_mask(gt: np.ndarray, gt_valid: np.ndarray) -> np.ndarray:
    dx = cv2.Sobel(gt, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gt, cv2.CV_32F, 0, 1, ksize=3)
    boundary = np.hypot(dx, dy) > 1.0
    boundary |= (
        cv2.morphologyEx(
            gt_valid.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
        )
        > 0
    )
    return cv2.dilate(
        boundary.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1
    ).astype(bool) & gt_valid


def infer_age_flows(
    adapter: BiDAFlowInferenceAdapter,
    images: list[torch.Tensor],
    current_indices: list[int],
    ages: list[int],
    batch_size: int,
    device: torch.device,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[int, float], float]:
    flows: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    latencies: dict[int, float] = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for age in ages:
        forward: list[np.ndarray] = []
        backward: list[np.ndarray] = []
        elapsed = 0.0
        for start in range(0, len(current_indices), batch_size):
            indices = current_indices[start : start + batch_size]
            current = torch.stack([images[index] for index in indices])
            past = torch.stack([images[index - age] for index in indices])
            target = torch.cat((current, past), dim=0)
            source = torch.cat((past, current), dim=0)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            tick = time.perf_counter()
            inferred = adapter.infer(target, source)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - tick
            count = len(indices)
            forward.extend(inferred[:count].detach().cpu().numpy().astype(np.float32))
            backward.extend(inferred[count:].detach().cpu().numpy().astype(np.float32))
        flows[age] = (np.stack(forward), np.stack(backward))
        latencies[age] = elapsed * 1000.0 / max(len(current_indices), 1)
    peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
    return flows, latencies, peak


def aligned_candidates(
    *,
    raw: np.ndarray,
    raw_valid: np.ndarray,
    past_disparities: list[np.ndarray],
    past_validity: list[np.ndarray],
    current_rgb: torch.Tensor,
    past_rgb: list[torch.Tensor],
    forward: list[np.ndarray],
    backward: list[np.ndarray],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], float]:
    count = len(past_disparities)
    raw_t = torch.from_numpy(raw)[None, None].float().to(device).expand(count, -1, -1, -1)
    raw_valid_t = (
        torch.from_numpy(raw_valid)[None, None].bool().to(device).expand(count, -1, -1, -1)
    )
    past_t = torch.from_numpy(np.stack(past_disparities))[:, None].float().to(device)
    past_valid_t = torch.from_numpy(np.stack(past_validity))[:, None].bool().to(device)
    flow_f = torch.from_numpy(np.stack(forward)).float().to(device)
    flow_b = torch.from_numpy(np.stack(backward)).float().to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    tick = time.perf_counter()
    evidence = temporal_disparity_evidence(
        raw_t,
        past_t,
        flow_f,
        flow_b,
        current_valid=raw_valid_t,
        past_valid=past_valid_t,
        current_rgb=current_rgb[None].expand(count, -1, -1, -1),
        past_rgb=torch.stack(past_rgb),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - tick) * 1000.0
    maps = {
        name: value[:, 0].detach().cpu().numpy()
        for name, value in evidence.as_dict().items()
    }
    return maps, elapsed_ms


def masked_sum(values: np.ndarray, mask: np.ndarray) -> float:
    return float(values[mask].sum()) if mask.any() else 0.0


def frame_oracle_row(
    *,
    backbone: str,
    sequence: str,
    frame_id: str,
    frame_index: int,
    threshold: float,
    ages: list[int],
    raw: np.ndarray,
    raw_valid: np.ndarray,
    past_disparities: list[np.ndarray],
    past_validity: list[np.ndarray],
    evidence: dict[str, np.ndarray],
    gt: np.ndarray,
    coverage: np.ndarray,
    flow_latency_by_age: dict[int, float],
    evidence_latency_ms: float,
    peak_gpu_memory_mb: float,
) -> dict:
    gt_valid = coverage > threshold
    aligned = evidence["aligned_past_disparity"]
    aligned_valid = evidence["aligned_validity"].astype(bool)
    support = evidence["warp_support"].astype(bool)
    base = gt_valid & raw_valid & aligned_valid[0] & support[0]
    count = int(base.sum())
    raw_error = np.abs(raw - gt)
    aligned_errors = np.abs(aligned - gt[None])
    aligned_candidate_valid = aligned_valid & support[:, None] if support.ndim == 2 else aligned_valid & support
    aligned_errors = np.where(aligned_candidate_valid, aligned_errors, np.inf)

    unaligned = np.stack(past_disparities)
    unaligned_valid = np.stack(past_validity).astype(bool) & raw_valid[None]
    unaligned_errors = np.where(unaligned_valid, np.abs(unaligned - gt[None]), np.inf)

    progressive = raw_error.copy()
    progressive_unaligned = raw_error.copy()
    aligned_progress: dict[int, np.ndarray] = {}
    unaligned_progress: dict[int, np.ndarray] = {}
    for candidate_index, age in enumerate(ages):
        progressive = np.minimum(progressive, aligned_errors[candidate_index])
        progressive_unaligned = np.minimum(
            progressive_unaligned, unaligned_errors[candidate_index]
        )
        aligned_progress[age] = progressive.copy()
        unaligned_progress[age] = progressive_unaligned.copy()

    boundary = boundary_mask(gt, gt_valid)
    boundary_common = base & boundary
    clean1 = base & (raw_error <= 1.0)
    clean3 = base & (raw_error <= 3.0)
    memory_stack = np.concatenate((raw_error[None], aligned_errors), axis=0)
    best_index = np.argmin(memory_stack, axis=0)
    best_ages = np.asarray([0] + ages, dtype=np.int16)[best_index]

    row: dict[str, object] = {
        "backbone": backbone,
        "sequence": sequence,
        "frame_id": frame_id,
        "frame_index": frame_index,
        "coverage_threshold": threshold,
        "common_valid_count": count,
        "common_valid_ratio": float(base.mean()),
        "gt_raw_valid_count": int((gt_valid & raw_valid).sum()),
        "raw_error_sum": masked_sum(raw_error, base),
        "aligned_t1_error_sum": masked_sum(aligned_errors[0], base),
        "oracle_t1_error_sum": masked_sum(aligned_progress[ages[0]], base),
        "oracle_multi_error_sum": masked_sum(aligned_progress[ages[-1]], base),
        "unaligned_oracle_t1_error_sum": masked_sum(unaligned_progress[ages[0]], base),
        "unaligned_oracle_multi_error_sum": masked_sum(unaligned_progress[ages[-1]], base),
        "boundary_count": int(boundary_common.sum()),
        "boundary_raw_error_sum": masked_sum(raw_error, boundary_common),
        "boundary_t1_oracle_error_sum": masked_sum(aligned_progress[ages[0]], boundary_common),
        "boundary_multi_oracle_error_sum": masked_sum(aligned_progress[ages[-1]], boundary_common),
        "clean1_count": int(clean1.sum()),
        "clean1_t1_oracle_error_sum": masked_sum(aligned_progress[ages[0]], clean1),
        "clean1_multi_oracle_error_sum": masked_sum(aligned_progress[ages[-1]], clean1),
        "clean3_count": int(clean3.sum()),
        "clean3_t1_oracle_error_sum": masked_sum(aligned_progress[ages[0]], clean3),
        "clean3_multi_oracle_error_sum": masked_sum(aligned_progress[ages[-1]], clean3),
        "evidence_latency_ms": evidence_latency_ms,
        "total_flow_latency_ms": float(sum(flow_latency_by_age.values())),
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
    }
    for age_index, age in enumerate(ages):
        row[f"aligned_age_{age}_valid_ratio"] = float((aligned_candidate_valid[age_index] & gt_valid & raw_valid).sum() / max(1, (gt_valid & raw_valid).sum()))
        row[f"flow_age_{age}_latency_ms"] = flow_latency_by_age[age]
        row[f"oracle_through_age_{age}_error_sum"] = masked_sum(aligned_progress[age], base)
        row[f"unaligned_oracle_through_age_{age}_error_sum"] = masked_sum(unaligned_progress[age], base)
        row[f"best_age_{age}_count"] = int((base & (best_ages == age)).sum())
    row["best_age_raw_count"] = int((base & (best_ages == 0)).sum())
    return row


def aggregate_group(rows: list[dict]) -> dict:
    count = sum(int(row["common_valid_count"]) for row in rows)

    def ratio(numerator: str, denominator: str = "common_valid_count") -> float:
        den = sum(float(row[denominator]) for row in rows)
        return sum(float(row[numerator]) for row in rows) / max(den, 1.0)

    raw = ratio("raw_error_sum")
    t1 = ratio("oracle_t1_error_sum")
    multi = ratio("oracle_multi_error_sum")
    unaligned_t1 = ratio("unaligned_oracle_t1_error_sum")
    unaligned_multi = ratio("unaligned_oracle_multi_error_sum")
    result = {
        "frames": len(rows),
        "common_valid_count": count,
        "common_valid_ratio": float(np.mean([float(row["common_valid_ratio"]) for row in rows])),
        "raw_epe": raw,
        "aligned_t1_memory_epe": ratio("aligned_t1_error_sum"),
        "t1_oracle_epe": t1,
        "multi_memory_oracle_epe": multi,
        "t1_oracle_gain": raw - t1,
        "multi_memory_oracle_gain": raw - multi,
        "incremental_long_memory_gain": t1 - multi,
        "incremental_fraction_of_t1_gain": (t1 - multi) / max(raw - t1, 1e-12),
        "unaligned_t1_oracle_epe": unaligned_t1,
        "unaligned_multi_oracle_epe": unaligned_multi,
        "bida_advantage_over_unaligned_multi": unaligned_multi - multi,
        "boundary_raw_epe": ratio("boundary_raw_error_sum", "boundary_count"),
        "boundary_t1_oracle_epe": ratio("boundary_t1_oracle_error_sum", "boundary_count"),
        "boundary_multi_oracle_epe": ratio("boundary_multi_oracle_error_sum", "boundary_count"),
        "clean1_t1_oracle_epe": ratio("clean1_t1_oracle_error_sum", "clean1_count"),
        "clean1_multi_oracle_epe": ratio("clean1_multi_oracle_error_sum", "clean1_count"),
        "clean3_t1_oracle_epe": ratio("clean3_t1_oracle_error_sum", "clean3_count"),
        "clean3_multi_oracle_epe": ratio("clean3_multi_oracle_error_sum", "clean3_count"),
        "mean_evidence_latency_ms": float(np.mean([float(row["evidence_latency_ms"]) for row in rows])),
        "mean_total_flow_latency_ms": float(np.mean([float(row["total_flow_latency_ms"]) for row in rows])),
        "peak_gpu_memory_mb": max(float(row["peak_gpu_memory_mb"]) for row in rows),
    }
    ages = sorted(
        int(key.split("_")[3])
        for key in rows[0]
        if key.startswith("oracle_through_age_") and key.endswith("_error_sum")
    )
    previous = raw
    for age in ages:
        epe = ratio(f"oracle_through_age_{age}_error_sum")
        result[f"oracle_through_age_{age}_epe"] = epe
        result[f"incremental_gain_adding_age_{age}"] = previous - epe
        result[f"best_age_{age}_ratio"] = sum(int(row[f"best_age_{age}_count"]) for row in rows) / max(count, 1)
        result[f"age_{age}_valid_ratio"] = float(np.mean([float(row[f"aligned_age_{age}_valid_ratio"]) for row in rows]))
        previous = epe
    result["best_age_raw_ratio"] = sum(int(row["best_age_raw_count"]) for row in rows) / max(count, 1)
    return result


def aggregate_outputs(rows: list[dict], output: Path, config: dict) -> dict:
    groups: dict[tuple[str, str, float], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(str(row["backbone"]), str(row["sequence"]), float(row["coverage_threshold"]))].append(row)
    sequence_rows = []
    for (backbone, sequence, threshold), values in sorted(groups.items()):
        sequence_rows.append(
            {"backbone": backbone, "sequence": sequence, "coverage_threshold": threshold}
            | aggregate_group(values)
        )
    write_csv(output / "sequence_metrics.csv", sequence_rows)

    backbone_groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    threshold_groups: dict[float, list[dict]] = defaultdict(list)
    for row in rows:
        backbone_groups[(str(row["backbone"]), float(row["coverage_threshold"]))].append(row)
        threshold_groups[float(row["coverage_threshold"])].append(row)
    by_backbone = {
        f"{backbone}@{threshold}": aggregate_group(values)
        for (backbone, threshold), values in sorted(backbone_groups.items())
    }
    aggregate = {
        str(threshold): aggregate_group(values)
        for threshold, values in sorted(threshold_groups.items())
    }

    primary = float(config["primary_coverage_threshold"])
    primary_total = aggregate[str(primary)]
    backbone_positive = sum(
        by_backbone[f"{backbone}@{primary}"]["incremental_long_memory_gain"]
        >= config["promotion_thresholds"]["per_group_min_gain_px"]
        for backbone in config["backbones"]
    )
    primary_sequences = [
        row for row in sequence_rows if float(row["coverage_threshold"]) == primary
    ]
    sequence_positive = sum(
        float(row["incremental_long_memory_gain"])
        >= config["promotion_thresholds"]["per_group_min_gain_px"]
        for row in primary_sequences
    )
    criteria = {
        "aggregate_incremental_gain": primary_total["incremental_long_memory_gain"]
        >= config["promotion_thresholds"]["aggregate_min_gain_px"],
        "positive_backbones": backbone_positive,
        "positive_sequences": sequence_positive,
        "required_backbones": 2,
        "required_sequences": 3,
    }
    go = (
        criteria["aggregate_incremental_gain"]
        and backbone_positive >= criteria["required_backbones"]
        and sequence_positive >= criteria["required_sequences"]
    )
    summary = {
        "schema_version": 1,
        "stage": "oracle-only memory horizon",
        "namespace": "cache-grid-from-cached-predictions",
        "mask_policy": "GT coverage & raw valid & aligned t-1 valid & t-1 warp support; older invalid candidates abstain to raw",
        "aggregation": "pixel-count weighted",
        "by_backbone": by_backbone,
        "aggregate": aggregate,
        "promotion_criteria": criteria,
        "oracle_horizon_decision": "GO" if go else "NO-GO",
        "next_stage_authorized": bool(go),
    }
    save_json(output / "memory_oracle_summary.json", summary)
    save_json(output / "aggregate_summary.json", summary)

    age_rows = []
    for row in sequence_rows:
        for age in config["ages"]:
            age_rows.append(
                {
                    "backbone": row["backbone"],
                    "sequence": row["sequence"],
                    "coverage_threshold": row["coverage_threshold"],
                    "age": age,
                    "best_age_ratio": row[f"best_age_{age}_ratio"],
                    "valid_ratio": row[f"age_{age}_valid_ratio"],
                    "incremental_gain_when_added": row[f"incremental_gain_adding_age_{age}"],
                }
            )
    write_csv(output / "memory_age_statistics.csv", age_rows)
    return summary


def write_readme(output: Path, config: dict, summary: dict | None = None) -> None:
    decision = "PENDING" if summary is None else summary["oracle_horizon_decision"]
    text = f"""# ARGOS v2 PPMStereo validation

Stage 1 is an oracle-only causal memory-horizon study. It uses SEA-RAFT and the
canonical BiDA warp to align exact ages {config['ages']} into the current cache
grid. It trains no selector and never reads future frames, stereo internals,
backbone identity, cost volumes or backbone confidence.

Decision: **{decision}** for proceeding beyond the oracle gate.

All primary comparisons use one identical mask: GT coverage, current raw-valid,
aligned t-1 validity and t-1 warp support. Older invalid candidates abstain to
raw rather than shrinking the common mask. Oracle selection is per pixel.
Metrics are pixel-count weighted. The namespace is explicitly
`cache-grid-from-cached-predictions`; no result here is native stereo inference.

The predeclared gate at coverage {config['primary_coverage_threshold']} requires
at least {config['promotion_thresholds']['aggregate_min_gain_px']} px aggregate
incremental oracle gain beyond t-1, at least
{config['promotion_thresholds']['per_group_min_gain_px']} px on two backbones and
three backbone/sequence groups. If it fails, learned long-memory selection is
not trained.
"""
    output.joinpath("README.md").write_text(text)


def main() -> int:
    args = parse_args()
    if sorted(args.ages) != args.ages or not args.ages or args.ages[0] != 1:
        raise ValueError("ages must be ascending and begin with t-1")
    if any(age <= 0 for age in args.ages):
        raise ValueError("ages must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        "mode": args.mode,
        "output": str(args.output),
        "backbones": args.backbones,
        "sequences": args.sequences,
        "ages": args.ages,
        "thresholds": args.thresholds,
        "primary_coverage_threshold": 0.50,
        "frames": args.frames,
        "start": args.start,
        "flow_model": "SEA-RAFT",
        "flow_cache_policy": "RAM only; no dense cache written",
        "primary_unseen_backbone": "Fast-FoundationStereo (not accessed)",
        "promotion_thresholds": {
            "aggregate_min_gain_px": 0.01,
            "per_group_min_gain_px": 0.005,
        },
        "command": " ".join(sys.argv),
    }
    save_json(args.output / "config.json", config)
    manifest = build_split_manifest(validation_sequences=args.sequences)
    manifest["actual_evaluation_sequences"] = args.sequences
    manifest["actual_oracle_backbones"] = args.backbones
    manifest["memory_ages"] = args.ages
    manifest["causal_clip"] = "only exact past indices; current index starts at max(memory ages)"
    save_json(args.output / "split_manifest.json", manifest)
    write_readme(args.output, config)

    frame_path = args.output / "frame_metrics.csv"
    if not args.resume:
        frame_path.unlink(missing_ok=True)
    existing = list(csv.DictReader(frame_path.open())) if args.resume and frame_path.exists() else []
    completed = {(row["backbone"], row["sequence"]) for row in existing}
    rows: list[dict] = list(existing)
    device = torch.device(args.device)
    log = (args.output / "run.log").open("a", buffering=1)
    print(f"COMMAND {config['command']}", file=log)
    adapter = BiDAFlowInferenceAdapter("sea_raft", device=device)

    for sequence in args.sequences:
        pending = [backbone for backbone in args.backbones if (backbone, sequence) not in completed]
        if not pending:
            print(f"SKIP {sequence}: all backbones complete", file=log)
            continue
        info = load_sequence_info(sequence)
        first = args.start
        last = min(len(info.frame_ids), first + args.frames)
        frame_ids = info.frame_ids[first:last]
        if len(frame_ids) <= max(args.ages):
            raise ValueError(f"{sequence}: insufficient frames for age {max(args.ages)}")
        current_indices = list(range(max(args.ages), len(frame_ids)))
        print(f"START sequence={sequence} frames={len(frame_ids)} queries={len(current_indices)}", file=log)
        rgbs = [load_frame_lr(info, frame_id)[0] for frame_id in frame_ids]
        images = [rgb_tensor(rgb, device) for rgb in rgbs]
        gt_data = [load_frame_gt(info, frame_id) for frame_id in frame_ids]
        flows, flow_latency, peak_mb = infer_age_flows(
            adapter, images, current_indices, args.ages, args.batch_size, device
        )
        print(f"FLOW sequence={sequence} latency={flow_latency} peak_mb={peak_mb:.1f}", file=log)

        for backbone in pending:
            disparities, validity, cache_ids, _metadata = load_sequence_cache(backbone, sequence)
            lookup = {str(frame_id): index for index, frame_id in enumerate(cache_ids)}
            block: list[dict] = []
            for query_offset, local_index in enumerate(current_indices):
                frame_id = frame_ids[local_index]
                cache_index = lookup[str(frame_id)]
                raw = np.asarray(disparities[cache_index], dtype=np.float32)
                raw_valid = np.asarray(validity[cache_index]) > 0
                past_disparities = []
                past_validity = []
                past_images = []
                forwards = []
                backwards = []
                for age in args.ages:
                    past_frame_id = frame_ids[local_index - age]
                    past_index = lookup[str(past_frame_id)]
                    past_disparities.append(np.asarray(disparities[past_index], dtype=np.float32))
                    past_validity.append(np.asarray(validity[past_index]) > 0)
                    past_images.append(images[local_index - age])
                    forwards.append(flows[age][0][query_offset])
                    backwards.append(flows[age][1][query_offset])
                evidence, evidence_ms = aligned_candidates(
                    raw=raw,
                    raw_valid=raw_valid,
                    past_disparities=past_disparities,
                    past_validity=past_validity,
                    current_rgb=images[local_index],
                    past_rgb=past_images,
                    forward=forwards,
                    backward=backwards,
                    device=device,
                )
                gt_native, gt_valid_native = gt_data[local_index]
                gt, coverage = resize_gt_to_cache_masked(gt_native, gt_valid_native)
                for threshold in args.thresholds:
                    block.append(
                        frame_oracle_row(
                            backbone=backbone,
                            sequence=sequence,
                            frame_id=str(frame_id),
                            frame_index=first + local_index,
                            threshold=threshold,
                            ages=args.ages,
                            raw=raw,
                            raw_valid=raw_valid,
                            past_disparities=past_disparities,
                            past_validity=past_validity,
                            evidence=evidence,
                            gt=gt,
                            coverage=coverage,
                            flow_latency_by_age=flow_latency,
                            evidence_latency_ms=evidence_ms,
                            peak_gpu_memory_mb=peak_mb,
                        )
                    )
            append_csv(frame_path, block)
            rows.extend(block)
            print(f"DONE backbone={backbone} sequence={sequence} rows={len(block)}", file=log)

    summary = aggregate_outputs(rows, args.output, config)
    write_readme(args.output, config, summary)
    save_json(
        args.output / "runtime_summary.json",
        {
            "flow_model": "SEA-RAFT",
            "aggregate": summary["aggregate"],
            "peak_gpu_memory_mb": max(
                float(row["peak_gpu_memory_mb"]) for row in rows
            ),
        },
    )
    # Stage-1 has no trained selector; keep explicit placeholders rather than
    # fabricating calibration or safety claims.
    save_json(args.output / "selector_calibration.json", {"status": "not run at oracle stage"})
    save_json(args.output / "safety_summary.json", {"status": "not applicable to per-pixel oracle ceiling"})
    print(
        f"COMPLETE decision={summary['oracle_horizon_decision']} rows={len(rows)}",
        file=log,
    )
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
