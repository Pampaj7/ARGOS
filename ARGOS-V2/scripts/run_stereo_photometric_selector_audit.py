#!/usr/bin/env python3
"""Frozen audit of current-stereo photometric selection for causal BiDA memory.

This script never trains a network and never writes prediction/flow caches.
It compares cached raw disparity against canonical causally aligned t-1 memory
using an externally observable current-frame left/right reprojection cost.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

V2_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(V2_ROOT), str(V2_ROOT / "scripts")]

from argos_v2.scared_c_data import load_sequence_info, read_rgb  # noqa: E402
from model_design.data.raw_error_dataset import CALIBRATION_SEQUENCES, TEST_SEQUENCES  # noqa: E402
from model_design.data.temporal_pair_dataset import SEEN_BACKBONES  # noqa: E402
from model_design.data.utility_memory_selector_dataset import UtilityMemorySelectorDataset, utility_targets  # noqa: E402
from model_design.external_components.bidavideo import BiDAFlowInferenceAdapter, SEA_RAFT_CHECKPOINT, temporal_disparity_evidence  # noqa: E402
from model_design.external_components.stereo_photometric import (  # noqa: E402
    select_lower_stereo_cost, stereo_photometric_evidence,
)
from run_raw_error_abstention import boundary_mask_tensor, map_metrics  # noqa: E402


UNSEEN_BACKBONES = ("Fast-FoundationStereo", "CREStereo")
WINDOWS = (15, 21, 31)
MARGINS = (0.0, 0.002, 0.005, 0.01, 0.02)
COSTS = ("local_rgb_l1", "zncc")
# Census is an ordinal cost in [0,1], so it uses a compact, separately
# preregistered support/margin grid rather than silently inheriting L1 units.
CENSUS_WINDOWS = (5, 7, 9)
CENSUS_MARGINS = (0.0, 0.01, 0.02, 0.05, 0.10)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "sweep", "evaluate", "unseen"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operating-point", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=min(48, os.cpu_count() or 8))
    parser.add_argument("--preload-workers", type=int, default=min(48, os.cpu_count() or 8))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--coverage-threshold", type=float, default=.50)
    parser.add_argument("--max-pairs-per-sequence", type=int, default=0)
    parser.add_argument("--backbones", nargs="+", default=None)
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--costs", nargs="+", choices=(*COSTS, "ternary_census"), default=list(COSTS),
                        help="Frozen candidate costs; census is run as a distinct signal audit before any selector training.")
    return parser.parse_args()


def clean(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path: Path, content) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(clean(content), indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        keys.extend(key for key in row if key not in keys)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def make_loader(dataset: UtilityMemorySelectorDataset, args: argparse.Namespace) -> DataLoader:
    kwargs = dict(batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                  pin_memory=True, persistent_workers=args.workers > 0)
    if args.workers:
        kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **kwargs)


class RightFrameCache:
    """Compact in-RAM current-right RGB cache; never persisted to disk."""

    def __init__(self, dataset: UtilityMemorySelectorDataset, workers: int) -> None:
        wanted = {(record.sequence, record.current_index, record.current_frame_id) for record in dataset.records}
        infos = {sequence: load_sequence_info(sequence) for sequence, _, _ in wanted}

        def load(item):
            sequence, index, frame_id = item
            image = read_rgb(infos[sequence].seq_dir / "right" / f"{frame_id}.png")
            image = cv2.resize(image, (180, 144), interpolation=cv2.INTER_AREA)
            image = np.ascontiguousarray(image).transpose(2, 0, 1)
            return (sequence, index), torch.from_numpy(image)

        self.values: dict[tuple[str, int], torch.Tensor] = {}
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for key, value in pool.map(load, sorted(wanted)):
                self.values[key] = value
        self.bytes = sum(value.numel() * value.element_size() for value in self.values.values())

    def batch(self, sequences: list[str], indices: torch.Tensor, device: torch.device) -> torch.Tensor:
        tensors = [self.values[(str(sequence), int(index))] for sequence, index in zip(sequences, indices.tolist())]
        return torch.stack(tensors).float().to(device, non_blocking=True)


@torch.no_grad()
def build_temporal(adapter: BiDAFlowInferenceAdapter, batch: dict) -> dict:
    current, past = batch["current_rgb"], batch["past_rgb"]
    forward = torch.cat((current, past), 0)
    backward = torch.cat((past, current), 0)
    flow = adapter.infer(forward, backward)
    count = current.shape[0]
    return temporal_disparity_evidence(
        batch["raw"], batch["past"], flow[:count], flow[count:],
        current_valid=batch["raw_valid"], past_valid=batch["past_valid"],
        current_rgb=current, past_rgb=past,
    ).as_dict()


def specs(costs: tuple[str, ...]) -> list[dict]:
    result = []
    for cost in costs:
        windows = CENSUS_WINDOWS if cost == "ternary_census" else WINDOWS
        margins = CENSUS_MARGINS if cost == "ternary_census" else MARGINS
        result.extend(
            {"cost": cost, "local_kernel": window, "minimum_improvement": margin,
             "method": f"photometric_{cost}_k{window}_m{margin:g}"}
            for window in windows for margin in margins
        )
    return result


def cost_map(evidence, cost: str) -> torch.Tensor:
    """Map the compact public cost label to its evidence tensor."""
    if cost == "zncc":
        return evidence.zncc_cost
    if cost == "ternary_census":
        return evidence.ternary_census_cost
    if cost == "local_rgb_l1":
        return evidence.local_rgb_l1
    raise ValueError(f"unknown stereo cost: {cost}")


def metric_row(*, method: str, prediction: torch.Tensor, raw: torch.Tensor, memory: torch.Tensor,
               gt: torch.Tensor, valid: torch.Tensor, boundary: torch.Tensor, authorization: torch.Tensor,
               backbone: str, sequence: str, frame_id: str, cost: str | None = None,
               local_kernel: int | None = None, minimum_improvement: float | None = None) -> dict:
    update = torch.where(authorization, memory - raw, torch.zeros_like(raw))
    metrics = map_metrics(prediction, raw, gt, valid, boundary, update)
    selected = valid.bool()
    error = (prediction - gt).abs()
    memory_error = (memory - gt).abs()
    metrics.update({
        "bad5": float((error[selected] > 5).float().mean()),
        "memory_epe": float(memory_error[selected].mean()),
        "oracle_epe": float(torch.minimum((raw - gt).abs(), memory_error)[selected].mean()),
        "flow_warped_temporal_disparity_difference": float((prediction - memory).abs()[selected].mean()),
        "gt_relative_temporal_error_consistency": float((error - memory_error).abs()[selected].mean()),
        "backbone": backbone, "sequence": sequence, "frame_id": frame_id, "method": method,
        "cost": cost, "local_kernel": local_kernel, "minimum_improvement": minimum_improvement,
        "coverage_threshold": .50,
        "metric_namespace": "cache-grid-from-cached-predictions",
        "units": "pixels at cache width 180",
    })
    return metrics


def aggregate(rows: list[dict]) -> tuple[list[dict], list[dict], dict]:
    metric_keys = ("epe", "raw_epe", "memory_epe", "oracle_epe", "bad1", "bad3", "bad5", "boundary_epe",
                   "intervention_coverage", "intervention_precision", "false_update_rate", "clean_pixel_degradation",
                   "new_bad3", "flow_warped_temporal_disparity_difference", "gt_relative_temporal_error_consistency")
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["backbone"], row["sequence"])].append(row)
    sequence_rows: list[dict] = []
    for (method, backbone, sequence), group in sorted(groups.items()):
        valid = [row for row in group if row["valid_count"]]
        if not valid:
            continue
        pixels = sum(int(row["valid_count"]) for row in valid)
        def weighted(key):
            finite = [row for row in valid if row.get(key) is not None and math.isfinite(float(row[key]))]
            den = sum(int(row["valid_count"]) for row in finite)
            return sum(float(row[key]) * int(row["valid_count"]) for row in finite) / den if den else math.nan
        changed = sum(int(row["changed_count"]) for row in valid)
        clean = sum(int(row["clean_count"]) for row in valid)
        helpful = sum(int(row["helpful_count"]) for row in valid)
        false = sum(int(row["false_update_count"]) for row in valid)
        clean_degrade = sum(int(row["clean_degradation_count"]) for row in valid)
        sequence_rows.append({"method": method, "backbone": backbone, "sequence": sequence, "frames": len(valid),
                              "valid_count": pixels, **{key: weighted(key) for key in metric_keys},
                              "intervention_coverage": changed / max(pixels, 1),
                              "intervention_precision": helpful / max(changed, 1),
                              "false_update_rate": false / max(clean, 1),
                              "clean_pixel_degradation": clean_degrade / max(clean, 1)})
    backbone_rows: list[dict] = []
    for (method, backbone), group in sorted(((key, [row for row in sequence_rows if (row["method"], row["backbone"]) == key])
                                              for key in {(r["method"], r["backbone"]) for r in sequence_rows}), key=lambda x: x[0]):
        pixels = sum(int(row["valid_count"]) for row in group)
        def weighted(key): return sum(float(row[key]) * int(row["valid_count"]) for row in group) / max(pixels, 1)
        backbone_rows.append({"method": method, "backbone": backbone, "valid_count": pixels,
                              **{key: weighted(key) for key in metric_keys}})
    overall: dict[str, dict] = {}
    for method in sorted({row["method"] for row in sequence_rows}):
        group = [row for row in sequence_rows if row["method"] == method]
        pixels = sum(int(row["valid_count"]) for row in group)
        def weighted(key): return sum(float(row[key]) * int(row["valid_count"]) for row in group) / max(pixels, 1)
        # Backbones are repeated predictions of each sequence, so collapse
        # them before sequence-unit bootstrap.
        by_sequence: dict[str, list[float]] = defaultdict(list)
        for row in group:
            by_sequence[row["sequence"]].append(float(row["raw_epe"]) - float(row["epe"]))
        gains = np.array([np.mean(values) for values in by_sequence.values()], dtype=np.float64)
        rng = np.random.default_rng(20260719)
        boot = np.array([rng.choice(gains, size=len(gains), replace=True).mean() for _ in range(10000)]) if len(gains) else np.array([math.nan])
        overall[method] = {"valid_count": pixels, "independent_sequence_count": len(gains),
                           **{key: weighted(key) for key in metric_keys},
                           "gain": weighted("raw_epe") - weighted("epe"),
                           "oracle_gain": weighted("raw_epe") - weighted("oracle_epe"),
                           "oracle_recovery": (weighted("raw_epe") - weighted("epe")) / max(weighted("raw_epe") - weighted("oracle_epe"), 1e-12),
                           "sequence_bootstrap_ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                           "positive_sequence_fraction": float((gains >= 0).mean()) if len(gains) else math.nan}
    return sequence_rows, backbone_rows, overall


def select_operating_point(overall: dict, candidates: list[dict]) -> dict:
    candidate_by_name = {item["method"]: item for item in candidates}
    scored = []
    for method, metrics in overall.items():
        if method not in candidate_by_name:
            continue
        item = candidate_by_name[method]
        scored.append(item | {"validation_gain": metrics["gain"], "validation_oracle_recovery": metrics["oracle_recovery"],
                              "validation_false_update_rate": metrics["false_update_rate"],
                              "validation_clean_pixel_degradation": metrics["clean_pixel_degradation"],
                              "validation_coverage": metrics["intervention_coverage"]})
    eligible = [item for item in scored if item["validation_gain"] > 0 and item["validation_false_update_rate"] <= .02
                and item["validation_clean_pixel_degradation"] <= .01 and item["validation_coverage"] >= .002]
    selected = max(eligible, key=lambda item: (item["validation_gain"], item["validation_oracle_recovery"]), default=None)
    return {"eligible": bool(selected), "selection_split": list(CALIBRATION_SEQUENCES),
            "constraint": "gain>0; false-update<=2%; clean-degradation<=1%; coverage>=0.2%",
            "selected": selected, "candidates": scored}


def dataset_for(args: argparse.Namespace) -> tuple[UtilityMemorySelectorDataset, tuple[str, ...]]:
    if args.mode == "smoke":
        sequences = tuple(args.sequences or (CALIBRATION_SEQUENCES[0],))
        backbones = tuple(args.backbones or (SEEN_BACKBONES[0],))
    elif args.mode == "sweep":
        sequences, backbones = tuple(CALIBRATION_SEQUENCES), tuple(args.backbones or SEEN_BACKBONES)
    elif args.mode == "evaluate":
        sequences, backbones = tuple(TEST_SEQUENCES), tuple(args.backbones or SEEN_BACKBONES)
    else:
        sequences, backbones = tuple(TEST_SEQUENCES), tuple(args.backbones or UNSEEN_BACKBONES)
    selection_only = set(backbones).issubset(set(SEEN_BACKBONES))
    return UtilityMemorySelectorDataset(backbones, sequences, coverage_threshold=args.coverage_threshold,
                                        max_pairs_per_sequence=args.max_pairs_per_sequence or None,
                                        selection_only=selection_only), backbones


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / "config.json", vars(args))
    save_json(args.output / "metric_definitions.json", {
        "raw": "current cached disparity", "aligned_memory": "canonical causal BiDA t-1 warp",
        "oracle_raw_memory": "per-pixel min absolute error between raw and aligned memory",
        "photometric selection": "choose aligned memory only when its current-frame stereo reprojection cost is lower by the frozen margin",
        "ternary census": "ordinal local left-versus-candidate-warped-right cost; census windows touching invalid right reconstruction are excluded",
        "common_mask": "GT coverage & raw valid & aligned-memory valid & BiDA support & raw right support & memory right support; census additionally requires both full census windows",
        "false_update_rate": "changed pixels with raw EPE <=0.5 px, divided by raw-clean pixels",
        "clean_pixel_degradation": "raw-clean changed pixels whose EPE rises by >0.02 px, divided by raw-clean pixels",
    })
    device = torch.device(args.device)
    dataset, backbones = dataset_for(args)
    if args.preload_workers:
        preload = dataset.base.preload_frame_data(args.preload_workers)
    else:
        preload = {"enabled": False}
    right_cache = RightFrameCache(dataset, args.preload_workers or 1)
    loader = make_loader(dataset, args)
    adapter = BiDAFlowInferenceAdapter(device=device)
    candidates = specs(tuple(args.costs))
    selected = None
    if args.mode in {"evaluate", "unseen"}:
        if not args.operating_point:
            raise ValueError("evaluate/unseen requires --operating-point frozen on validation")
        policy = json.loads(args.operating_point.read_text())
        if not policy.get("eligible"):
            raise RuntimeError("validation found no safe photometric operating point; final evaluation is prohibited")
        selected = policy["selected"]
        candidates = [selected]
    started = time.perf_counter()
    rows: list[dict] = []
    for number, cpu_batch in enumerate(loader):
        batch = to_device(cpu_batch, device)
        temporal = build_temporal(adapter, batch)
        right = right_cache.batch(list(batch["sequence"]), batch["current_index"], device)
        raw_photo: dict[int, object] = {}
        mem_photo: dict[int, object] = {}
        for window in {item["local_kernel"] for item in candidates}:
            raw_photo[window] = stereo_photometric_evidence(
                batch["current_rgb"], right, batch["raw"], local_kernel=window, census_kernel=window,
            )
            mem_photo[window] = stereo_photometric_evidence(
                batch["current_rgb"], right, temporal["aligned_past_disparity"], local_kernel=window, census_kernel=window,
            )
        target = utility_targets(batch, temporal["aligned_past_disparity"], temporal["aligned_validity"], temporal["warp_support"],
                                 coverage_threshold=args.coverage_threshold)
        boundary = boundary_mask_tensor(batch["gt"])
        for index in range(batch["raw"].shape[0]):
            # Candidate-specific stereo support is part of the common paired
            # mask, so the comparison cannot gain coverage by changing output.
            # All candidates in one run use a single strict reference support
            # so raw/oracle/selection EPE are paired.  For a census-only run
            # this is the largest census window; for L1/ZNCC it reduces to the
            # exact same right-image support used historically.
            support_window = max(raw_photo)
            raw_support = raw_photo[support_window].right_support[index:index + 1]
            mem_support = mem_photo[support_window].right_support[index:index + 1]
            if "ternary_census" in args.costs:
                raw_support = raw_support & raw_photo[support_window].census_support[index:index + 1]
                mem_support = mem_support & mem_photo[support_window].census_support[index:index + 1]
            valid = target.valid[index:index + 1] & raw_support & mem_support
            if not valid.any():
                continue
            raw = batch["raw"][index:index + 1]; memory = temporal["aligned_past_disparity"][index:index + 1]
            gt = batch["gt"][index:index + 1]; bound = boundary[index:index + 1]
            raw_error = (raw - gt).abs(); memory_error = (memory - gt).abs()
            oracle = memory_error < raw_error
            common = dict(backbone=str(batch["backbone"][index]), sequence=str(batch["sequence"][index]),
                          frame_id=str(batch["current_frame_id"][index]))
            rows.append(metric_row(method="raw", prediction=raw, raw=raw, memory=memory, gt=gt, valid=valid, boundary=bound,
                                   authorization=torch.zeros_like(valid), **common))
            rows.append(metric_row(method="aligned_memory", prediction=memory, raw=raw, memory=memory, gt=gt, valid=valid, boundary=bound,
                                   authorization=valid, **common))
            rows.append(metric_row(method="oracle_raw_memory", prediction=torch.where(oracle, memory, raw), raw=raw, memory=memory, gt=gt,
                                   valid=valid, boundary=bound, authorization=oracle & valid, **common))
            for item in candidates:
                window = item["local_kernel"]
                raw_cost = cost_map(raw_photo[window], item["cost"])[index:index + 1]
                memory_cost = cost_map(mem_photo[window], item["cost"])[index:index + 1]
                authorize = select_lower_stereo_cost(raw_cost, memory_cost, valid,
                                                     minimum_improvement=item["minimum_improvement"])
                rows.append(metric_row(method=item["method"], prediction=torch.where(authorize, memory, raw), raw=raw, memory=memory,
                                       gt=gt, valid=valid, boundary=bound, authorization=authorize, cost=item["cost"],
                                       local_kernel=window, minimum_improvement=item["minimum_improvement"], **common))
        if number % 50 == 0:
            print(json.dumps({"batch": number, "frames": len(rows), "elapsed_s": time.perf_counter() - started}), flush=True)
    sequence_rows, backbone_rows, overall = aggregate(rows)
    write_csv(args.output / "frame_metrics.csv", rows)
    write_csv(args.output / "sequence_metrics.csv", sequence_rows)
    write_csv(args.output / "backbone_metrics.csv", backbone_rows)
    split = {"mode": args.mode, "sequences": list(dataset.sequences), "backbones": list(backbones),
             "pairs": len(dataset), "causal_pair": "t-1 -> t only", "selection_only": dataset.selection_only,
             "frozen": {"sea_raft_checkpoint": str(SEA_RAFT_CHECKPOINT), "sea_raft_sha256": sha256(SEA_RAFT_CHECKPOINT),
                        "bida_source_sha256": sha256(V2_ROOT / "model_design/external_components/bidavideo.py")}}
    save_json(args.output / "split_audit.json", split)
    save_json(args.output / "runtime_summary.json", {"elapsed_s": time.perf_counter() - started, "right_rgb_ram_bytes": right_cache.bytes,
                                                       "left_gt_preload": preload, "batch_size": args.batch_size, "workers": args.workers})
    save_json(args.output / "aggregate_summary.json", {"metric_namespace": "cache-grid-from-cached-predictions", "coverage_threshold": .50,
                                                         "units": "pixels at cache width 180", "overall": overall})
    if args.mode == "sweep":
        operating = select_operating_point(overall, candidates)
        save_json(args.output / "operating_point.json", operating)
        write_csv(args.output / "threshold_selection.csv", operating["candidates"])
    if args.mode == "smoke":
        primary = overall.get(candidates[0]["method"], {})
        if not rows or not math.isfinite(primary.get("epe", math.nan)):
            raise RuntimeError("smoke produced no finite metrics")
        # Oracle inequalities are exact by construction, checked on every
        # frame by the aggregate values as a final guard.
        raw = overall["raw"]; oracle = overall["oracle_raw_memory"]
        if oracle["epe"] > raw["epe"] + 1e-7 or oracle["epe"] > oracle["memory_epe"] + 1e-7:
            raise RuntimeError("raw-or-memory oracle inequality failed")
        save_json(args.output / "smoke_summary.json", {"passed": True, "raw_epe": raw["epe"], "oracle_epe": oracle["epe"],
                                                        "candidate_epe": primary["epe"], "rows": len(rows)})


if __name__ == "__main__":
    run(arguments())
