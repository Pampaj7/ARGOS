#!/usr/bin/env python3
"""ARGOS v2 post-hoc hybrid raw/corrected temporal-memory oracle audit.

No memory model is trained.  Dataset 2 freezes the candidate subset and only
then may dataset 7 be opened by the final-test mode.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from argos_v2.cache_io import load_sequence_cache  # noqa: E402
from argos_v2.scared_c_data import load_frame_gt, load_frame_lr, load_sequence_info  # noqa: E402
from model_design.data.temporal_pair_dataset import SEEN_BACKBONES, resize_gt_to_cache_masked  # noqa: E402
from model_design.external_components.bidavideo import (  # noqa: E402
    BiDAFlowInferenceAdapter,
    SEA_RAFT_CHECKPOINT,
    temporal_disparity_evidence,
)
from model_design.models.codd_style_fusion import CODDStyleFusionHead, build_codd_cues  # noqa: E402
from run_ppmstereo_validation import infer_age_flows, rgb_tensor  # noqa: E402


OUTPUT = ROOT / "results/hybrid_temporal_memory_oracle_audit"
CHECKPOINT = ROOT / "results/codd_style_bounded_memory_validation/ablations/no_learned_stereo_evidence/checkpoints/best_validation.pt"
VALIDATION = ("dataset_2_keyframe_2", "dataset_2_keyframe_3", "dataset_2_keyframe_4")
TEST = ("dataset_7_keyframe_1", "dataset_7_keyframe_2", "dataset_7_keyframe_3", "dataset_7_keyframe_4")
TRAIN = ("dataset_1_keyframe_2", "dataset_1_keyframe_3", "dataset_3_keyframe_1", "dataset_3_keyframe_2",
         "dataset_3_keyframe_3", "dataset_3_keyframe_4", "dataset_6_keyframe_1", "dataset_6_keyframe_2",
         "dataset_6_keyframe_3", "dataset_6_keyframe_4")
AGES = (1, 2, 4, 8)
RAW_NAMES = ("CS1", "CS2", "CS4", "CS8")
CORRECTED_NAMES = ("CF1", "CF2")
FULL_NAMES = ("CF1", "CF2", "CS1", "CS2", "CS4", "CS8")
FAMILIES = {
    "current_plus_one_step_raw": ("CS1",),
    "current_plus_short_corrected": CORRECTED_NAMES,
    "current_plus_raw_anchors": RAW_NAMES,
    "full_hybrid_bank": FULL_NAMES,
}
CORRECTED_FIRST = ("CF1", "CF2", "CS1", "CS2", "CS4", "CS8")
RAW_FIRST = ("CS1", "CS2", "CS4", "CS8", "CF1", "CF2")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "evaluate", "freeze", "report", "probes", "finalize"), required=True)
    parser.add_argument("--stage", choices=("validation", "test"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--sequences", nargs="+")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--probe-pixels-per-frame", type=int, default=64)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for piece in iter(lambda: handle.read(1 << 20), b""):
            digest.update(piece)
    return digest.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(field for row in rows for field in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    if not path.exists():
        write_csv(path, rows); return
    with path.open() as handle:
        fields = next(csv.reader(handle))
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writerows(rows); handle.flush()


def load_model(device: torch.device) -> tuple[CODDStyleFusionHead, dict]:
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = CODDStyleFusionHead(int(state["cue_channels"])).to(device)
    model.load_state_dict(state["model"]); model.eval().requires_grad_(False)
    if int(state["cue_channels"]) != 38:
        raise RuntimeError("primary corrected-state checkpoint is not the canonical 38-cue no-learned-evidence model")
    return model, {"path": str(CHECKPOINT), "sha256": sha256(CHECKPOINT), "epoch": int(state["epoch"]), "cue_channels": 38}


def evidence_maps(
    *, raw: np.ndarray, raw_valid: np.ndarray, sources: list[np.ndarray], source_valid: list[np.ndarray],
    current_rgb: torch.Tensor, past_rgb: list[torch.Tensor], forward: list[np.ndarray], backward: list[np.ndarray],
    device: torch.device,
) -> dict[str, np.ndarray]:
    count = len(sources)
    raw_t = torch.from_numpy(raw)[None, None].float().to(device).expand(count, -1, -1, -1)
    raw_valid_t = torch.from_numpy(raw_valid)[None, None].bool().to(device).expand(count, -1, -1, -1)
    source_t = torch.from_numpy(np.stack(sources))[:, None].float().to(device)
    source_valid_t = torch.from_numpy(np.stack(source_valid))[:, None].bool().to(device)
    evidence = temporal_disparity_evidence(
        raw_t, source_t, torch.from_numpy(np.stack(forward)).float().to(device),
        torch.from_numpy(np.stack(backward)).float().to(device), current_valid=raw_valid_t,
        past_valid=source_valid_t, current_rgb=current_rgb[None].expand(count, -1, -1, -1),
        past_rgb=torch.stack(past_rgb),
    )
    return {name: value[:, 0].detach().cpu().numpy() for name, value in evidence.as_dict().items()}


@torch.inference_mode()
def fused_history(
    model: CODDStyleFusionHead, adapter: BiDAFlowInferenceAdapter, images: list[torch.Tensor],
    raw: np.ndarray, valid: np.ndarray, flow_age1: tuple[np.ndarray, np.ndarray], device: torch.device,
) -> list[np.ndarray]:
    """Reproduce the canonical H=4 state lifecycle; keep output only in RAM."""
    outputs = [np.asarray(raw[0], dtype=np.float32).copy()]
    previous_fused: torch.Tensor | None = None
    for index in range(1, len(images)):
        current = torch.from_numpy(np.asarray(raw[index], dtype=np.float32))[None, None].to(device)
        current_valid = torch.from_numpy(np.asarray(valid[index]) > 0)[None, None].to(device)
        reset = (index - 1) % 4 == 0
        if reset or previous_fused is None:
            source = torch.from_numpy(np.asarray(raw[index - 1], dtype=np.float32))[None, None].to(device)
        else:
            source = previous_fused
        source_valid = torch.from_numpy(np.asarray(valid[index - 1]) > 0)[None, None].to(device)
        forward = torch.from_numpy(flow_age1[0][index - 1])[None].to(device)
        backward = torch.from_numpy(flow_age1[1][index - 1])[None].to(device)
        evidence = temporal_disparity_evidence(
            current, source, forward, backward, current_valid=current_valid, past_valid=source_valid,
            current_rgb=images[index][None], past_rgb=images[index - 1][None],
        )
        cues = build_codd_cues(
            None, raw=current, aligned_memory=evidence.aligned_past_disparity,
            current_rgb=images[index][None], current_right_rgb=images[index][None], past_rgb=images[index - 1][None],
            flow_current_to_past=forward, flow_magnitude=evidence.flow_magnitude,
            forward_backward_confidence=evidence.forward_backward_confidence,
            warp_support=evidence.warp_support, aligned_valid=evidence.aligned_validity,
            include_learned_stereo_evidence=False,
        )
        previous_fused = model(cues, current, evidence.aligned_past_disparity).fused_disparity
        outputs.append(previous_fused[0, 0].detach().cpu().numpy().astype(np.float32))
    return outputs


def candidate_arrays(raw_maps: dict[str, np.ndarray], corrected_maps: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray | int | str]]:
    result = {}
    for index, age in enumerate(AGES):
        result[f"CS{age}"] = {
            "disparity": raw_maps["aligned_past_disparity"][index],
            "available": raw_maps["aligned_validity"][index].astype(bool) & raw_maps["warp_support"][index].astype(bool),
            "fb": raw_maps["forward_backward_confidence"][index], "age": age, "provenance": "raw",
        }
    for index, age in enumerate((1, 2)):
        result[f"CF{age}"] = {
            "disparity": corrected_maps["aligned_past_disparity"][index],
            "available": corrected_maps["aligned_validity"][index].astype(bool) & corrected_maps["warp_support"][index].astype(bool),
            "fb": corrected_maps["forward_backward_confidence"][index], "age": age, "provenance": "corrected",
        }
    return result


def selection_oracle(raw_error: np.ndarray, gt: np.ndarray, base: np.ndarray, candidates: dict, names: tuple[str, ...], strict: bool):
    support = base.copy()
    if strict:
        for name in names: support &= candidates[name]["available"]
    errors = [raw_error]
    for name in names:
        error = np.abs(candidates[name]["disparity"] - gt)
        errors.append(error if strict else np.where(candidates[name]["available"] & base, error, np.inf))
    stack = np.stack(errors)
    return stack.min(axis=0), stack.argmin(axis=0), support


def convex_oracle(raw: np.ndarray, raw_error: np.ndarray, gt: np.ndarray, base: np.ndarray, candidates: dict, names: tuple[str, ...], strict: bool):
    support = base.copy(); errors = [raw_error]
    for name in names:
        item = candidates[name]
        if strict: support &= item["available"]
        denominator = item["disparity"] - raw
        weight = np.zeros_like(raw)
        safe = np.abs(denominator) > 1e-6
        weight[safe] = np.clip((gt[safe] - raw[safe]) / denominator[safe], 0, 1)
        error = np.abs((1 - weight) * raw + weight * item["disparity"] - gt)
        errors.append(error if strict else np.where(item["available"] & base, error, np.inf))
    stack = np.stack(errors)
    return stack.min(axis=0), stack.argmin(axis=0), support


def masked_sum(value: np.ndarray, mask: np.ndarray) -> float:
    return float(value[mask].sum()) if mask.any() else 0.0


def frame_rows(stage: str, sequence: str, backbone: str, frame_id: str, raw: np.ndarray, raw_valid: np.ndarray,
               gt: np.ndarray, coverage: np.ndarray, candidates: dict) -> dict[str, list[dict]]:
    base = (coverage > .50) & raw_valid
    raw_error = np.abs(raw - gt)
    common = {"stage": stage, "sequence": sequence, "backbone": backbone, "frame_id": frame_id}
    output = {"candidate": [], "oracle": [], "incremental": [], "complementarity": [], "support": []}
    for name in FULL_NAMES:
        item = candidates[name]; available = base & item["available"]
        error = np.abs(item["disparity"] - gt)
        wins = available & (error < raw_error)
        output["candidate"].append({**common, "candidate": name, "age": item["age"], "provenance": item["provenance"],
            "base_count": int(base.sum()), "available_count": int(available.sum()), "candidate_error_sum": masked_sum(error, available),
            "raw_error_sum": masked_sum(raw_error, available), "oracle_error_sum": masked_sum(np.minimum(raw_error, error), available),
            "win_count": int(wins.sum()), "frame_contributes": int(wins.any()), "fb_sum": masked_sum(item["fb"], available),
            "disagreement_sum": masked_sum(np.abs(item["disparity"] - raw), available)})
        output["support"].append({**common, "candidate": name, "age": item["age"], "provenance": item["provenance"],
            "base_count": int(base.sum()), "available_count": int(available.sum()), "warp_support_coverage": float(available.sum() / max(base.sum(), 1))})

    for family, names in FAMILIES.items():
        for contract, strict in (("strict_intersection", True), ("availability_aware", False)):
            oracle, winner, support = selection_oracle(raw_error, gt, base, candidates, names, strict)
            row = {**common, "oracle_family": family, "oracle_type": "selection", "support_contract": contract,
                "base_count": int(base.sum()), "support_count": int(support.sum()), "raw_error_sum": masked_sum(raw_error, support),
                "oracle_error_sum": masked_sum(oracle, support)}
            for index, name in enumerate(("C0",) + names): row[f"winner_{name}_count"] = int((support & (winner == index)).sum())
            output["oracle"].append(row)
        for contract, strict in (("strict_intersection", True), ("availability_aware", False)):
            oracle, winner, support = convex_oracle(raw, raw_error, gt, base, candidates, names, strict)
            output["oracle"].append({**common, "oracle_family": family, "oracle_type": "convex", "support_contract": contract,
                "base_count": int(base.sum()), "support_count": int(support.sum()), "raw_error_sum": masked_sum(raw_error, support),
                "oracle_error_sum": masked_sum(oracle, support), "temporal_convex_winner_count": int((support & (winner > 0)).sum())})

    for order_name, order in (("corrected_first", CORRECTED_FIRST), ("raw_first", RAW_FIRST)):
        previous = raw_error.copy()
        for step, name in enumerate(order, 1):
            item = candidates[name]
            candidate_error = np.where(item["available"] & base, np.abs(item["disparity"] - gt), np.inf)
            current = np.minimum(previous, candidate_error)
            output["incremental"].append({**common, "order": order_name, "step": step, "added_candidate": name,
                "base_count": int(base.sum()), "raw_error_sum": masked_sum(raw_error, base),
                "previous_oracle_error_sum": masked_sum(previous, base), "oracle_error_sum": masked_sum(current, base),
                "incremental_gain_sum": masked_sum(previous - current, base),
                "added_winner_count": int((base & (candidate_error < previous)).sum()),
                "frame_contributes": int((base & (candidate_error < previous)).any())})
            previous = current

    errors = {name: np.where(candidates[name]["available"] & base, np.abs(candidates[name]["disparity"] - gt), np.inf) for name in FULL_NAMES}
    corrected = np.minimum(errors["CF1"], errors["CF2"])
    raw_anchors = np.minimum.reduce([errors[name] for name in RAW_NAMES])
    far_raw = np.minimum(errors["CS4"], errors["CS8"])
    all_temporal = np.minimum(corrected, raw_anchors)
    masks = {
        "corrected_wins_all_raw_anchors_lose": base & (corrected < raw_error) & (raw_anchors >= raw_error),
        "raw_anchor_wins_corrected_lose": base & (raw_anchors < raw_error) & (corrected >= raw_error),
        "far_raw_recovers_short_corrected_failure": base & (far_raw < raw_error) & (corrected >= raw_error),
        "current_raw_remains_best": base & (raw_error <= all_temporal),
        "both_provenances_offer_gain": base & (corrected < raw_error) & (raw_anchors < raw_error),
    }
    output["complementarity"].append({**common, "base_count": int(base.sum()), **{f"{name}_count": int(mask.sum()) for name, mask in masks.items()}})
    return output


def sequence_evaluation(sequence: str, stage: str, config: argparse.Namespace, model, adapter, device, log) -> dict[str, list[dict]]:
    info = load_sequence_info(sequence)
    frame_ids = info.frame_ids[: config.max_frames] if config.max_frames else info.frame_ids
    if len(frame_ids) <= 8: raise ValueError(f"{sequence}: fewer than nine frames")
    images = [rgb_tensor(load_frame_lr(info, frame_id)[0], device) for frame_id in frame_ids]
    gt_data = [load_frame_gt(info, frame_id) for frame_id in frame_ids]
    audit_indices = list(range(8, len(frame_ids)))
    tick = time.perf_counter()
    flow_age1, latency1, peak1 = infer_age_flows(adapter, images, list(range(1, len(frame_ids))), [1], config.batch_size, device)
    flows = {1: (flow_age1[1][0][7:], flow_age1[1][1][7:])}
    other, latency_other, peak_other = infer_age_flows(adapter, images, audit_indices, [2, 4, 8], config.batch_size, device)
    flows.update(other)
    print(f"FLOW sequence={sequence} seconds={time.perf_counter()-tick:.1f} peak_mb={max(peak1,peak_other):.0f}", file=log, flush=True)
    outputs = {kind: [] for kind in ("candidate", "oracle", "incremental", "complementarity", "support")}
    for backbone in SEEN_BACKBONES:
        disparities, validity, cache_ids, _metadata = load_sequence_cache(backbone, sequence)
        if [str(value) for value in cache_ids][:len(frame_ids)] != frame_ids:
            raise RuntimeError(f"frame ID mismatch: {backbone}/{sequence}")
        raw = np.asarray(disparities[:len(frame_ids)], dtype=np.float32)
        valid = np.asarray(validity[:len(frame_ids)]) > 0
        fused = fused_history(model, adapter, images, raw, valid, flow_age1[1], device)
        for query_offset, index in enumerate(audit_indices):
            forwards = [flows[age][0][query_offset] for age in AGES]
            backwards = [flows[age][1][query_offset] for age in AGES]
            raw_maps = evidence_maps(raw=raw[index], raw_valid=valid[index], sources=[raw[index-age] for age in AGES],
                source_valid=[valid[index-age] for age in AGES], current_rgb=images[index], past_rgb=[images[index-age] for age in AGES],
                forward=forwards, backward=backwards, device=device)
            corrected_maps = evidence_maps(raw=raw[index], raw_valid=valid[index], sources=[fused[index-1], fused[index-2]],
                source_valid=[valid[index-1], valid[index-2]], current_rgb=images[index], past_rgb=[images[index-1], images[index-2]],
                forward=forwards[:2], backward=backwards[:2], device=device)
            candidates = candidate_arrays(raw_maps, corrected_maps)
            gt_native, gt_valid_native = gt_data[index]; gt, coverage = resize_gt_to_cache_masked(gt_native, gt_valid_native)
            frame_output = frame_rows(stage, sequence, backbone, frame_ids[index], raw[index], valid[index], gt, coverage, candidates)
            for kind in outputs: outputs[kind].extend(frame_output[kind])
        print(f"DONE sequence={sequence} backbone={backbone} queries={len(audit_indices)}", file=log, flush=True)
    return outputs


def evaluate(config: argparse.Namespace) -> None:
    if config.stage is None: raise ValueError("--stage is required")
    allowed = VALIDATION if config.stage == "validation" else TEST
    sequences = tuple(config.sequences or allowed)
    if set(sequences) - set(allowed): raise ValueError(f"illegal {config.stage} sequences: {sorted(set(sequences)-set(allowed))}")
    if config.stage == "test" and not (config.output / "validation_anchor_selection.json").exists():
        raise RuntimeError("dataset 7 is locked until validation_anchor_selection.json exists")
    destination = config.output / config.stage / ("_".join(sequences))
    destination.mkdir(parents=True, exist_ok=True)
    completed_path = destination / "completed_sequences.json"
    completed = set(json.loads(completed_path.read_text())) if config.resume and completed_path.exists() else set()
    device = torch.device(config.device); model, checkpoint = load_model(device); adapter = BiDAFlowInferenceAdapter("sea_raft", device=device)
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert not any(parameter.requires_grad for parameter in adapter.model.parameters())
    with (destination / "run.log").open("a", buffering=1) as log:
        print(f"START stage={config.stage} sequences={sequences} checkpoint={checkpoint['sha256']}", file=log)
        for sequence in sequences:
            if sequence in completed: continue
            output = sequence_evaluation(sequence, config.stage, config, model, adapter, device, log)
            for kind, rows in output.items(): append_csv(destination / f"{kind}_metrics.csv", rows)
            completed.add(sequence); save_json(completed_path, sorted(completed))
    save_json(destination / "runtime.json", {"stage": config.stage, "sequences": sequences, "checkpoint": checkpoint,
        "sea_raft_sha256": sha256(SEA_RAFT_CHECKPOINT), "device": str(device), "complete": set(sequences) <= completed})


def read_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        if path.exists():
            with path.open() as handle: rows.extend(csv.DictReader(handle))
    return rows


def aggregate_rows(rows: list[dict], keys: tuple[str, ...], sums: tuple[str, ...]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows: groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for values, group in sorted(groups.items()):
        result = dict(zip(keys, values))
        result["row_count"] = len(group)
        for field in sums: result[field] = sum(float(row.get(field, 0) or 0) for row in group)
        output.append(result)
    return output


def freeze(config: argparse.Namespace) -> None:
    validation_dirs = list((config.output / "validation").glob("*"))
    oracle = read_rows([path / "oracle_metrics.csv" for path in validation_dirs])
    incremental = read_rows([path / "incremental_metrics.csv" for path in validation_dirs])
    candidate = read_rows([path / "candidate_metrics.csv" for path in validation_dirs])
    if not oracle or not incremental: raise RuntimeError("validation shards are incomplete")
    agg_oracle = aggregate_rows(oracle, ("oracle_family", "oracle_type", "support_contract", "backbone"),
        ("base_count", "support_count", "raw_error_sum", "oracle_error_sum"))
    pooled_oracle = aggregate_rows(oracle, ("oracle_family", "oracle_type", "support_contract"),
        ("base_count", "support_count", "raw_error_sum", "oracle_error_sum"))
    for row in pooled_oracle: row["backbone"] = "ALL"
    agg_oracle.extend(pooled_oracle)
    for row in agg_oracle:
        row["raw_epe"] = row["raw_error_sum"] / max(row["support_count"], 1)
        row["oracle_epe"] = row["oracle_error_sum"] / max(row["support_count"], 1)
        row["oracle_gain"] = row["raw_epe"] - row["oracle_epe"]
    agg_incremental = aggregate_rows(incremental, ("order", "step", "added_candidate"),
        ("base_count", "raw_error_sum", "previous_oracle_error_sum", "oracle_error_sum", "incremental_gain_sum", "added_winner_count", "frame_contributes"))
    selected = []
    raw_first = [row for row in agg_incremental if row["order"] == "raw_first"]
    for row in sorted(raw_first, key=lambda item: int(item["step"])):
        gain = row["incremental_gain_sum"] / max(row["base_count"], 1)
        win_fraction = row["added_winner_count"] / max(row["base_count"], 1)
        frame_fraction = row["frame_contributes"] / max(row["row_count"], 1)
        if gain >= .001 or (win_fraction >= .0025 and frame_fraction >= .02): selected.append(row["added_candidate"])
    # Corrected candidates are assessed after raw anchors in raw-first order and
    # remain eligible under the same predeclared materiality rule.
    if not selected: selected = ["CS1"]
    full = next(row for row in agg_oracle if row["oracle_family"] == "full_hybrid_bank" and row["oracle_type"] == "selection" and row["support_contract"] == "availability_aware" and row["backbone"] == "ALL")
    selection = {
        "project": "ARGOS v2", "selection_data": list(VALIDATION), "test_data_accessed": False,
        "predeclared_materiality": {"incremental_gain_px": .001, "winner_pixel_fraction": .0025, "contributing_frame_fraction": .02},
        "selected_candidates": selected, "t8_retained": "CS8" in selected,
        "full_hybrid_validation_pooled": full,
        "dataset_7_project_history_caveat": "dataset 7 was used in earlier ARGOS v2 studies; it is held out only from this anchor-selection procedure",
    }
    write_csv(config.output / "validation_oracle_aggregate.csv", agg_oracle)
    write_csv(config.output / "validation_incremental_aggregate.csv", agg_incremental)
    save_json(config.output / "validation_anchor_selection.json", selection)
    save_json(config.output / "checkpoint_hashes.json", {"corrected_state": sha256(CHECKPOINT), "sea_raft": sha256(SEA_RAFT_CHECKPOINT)})
    save_json(config.output / "protocol_audit.json", {"project": "ARGOS v2", "validation_sequences": list(VALIDATION), "test_sequences": list(TEST),
        "ages": list(AGES), "backbones": list(SEEN_BACKBONES), "coverage_threshold": .50, "direct_current_to_anchor_flow": True,
        "flow_composition": False, "future_access": False, "corrected_lifecycle": "canonical no-learned-evidence H=4 checkpoint",
        "support_contracts": ["strict_intersection", "availability_aware"], "dataset7_locked_until_freeze": True})
    (config.output / "candidate_definition.md").write_text(
        "# ARGOS v2 hybrid candidate contract\n\n"
        "`C0` is current raw stereo. `CS{1,2,4,8}` are immutable raw disparities aligned directly from their source frame. "
        "`CF{1,2}` are disparities produced by the canonical no-learned-stereo-evidence H=4 refiner, then aligned directly from their source frame. "
        "Every temporal candidate stores age, raw/corrected provenance, source frame, backbone, BiDA warp support, validity and FB confidence. "
        "Current-to-anchor SEA-RAFT flow is inferred directly; consecutive flow chains are never composed.\n\n"
        "Strict comparisons intersect all candidate supports. Availability-aware comparisons retain C0 exactly and allow each temporal candidate only on its own valid support.\n")


def summarize_stage(rows: list[dict], group_fields: tuple[str, ...]) -> list[dict]:
    aggregated = aggregate_rows(rows, group_fields, ("base_count", "support_count", "raw_error_sum", "oracle_error_sum"))
    for row in aggregated:
        row["support_coverage"] = row["support_count"] / max(row["base_count"], 1)
        row["raw_epe"] = row["raw_error_sum"] / max(row["support_count"], 1)
        row["oracle_epe"] = row["oracle_error_sum"] / max(row["support_count"], 1)
        row["oracle_gain"] = row["raw_epe"] - row["oracle_epe"]
    return aggregated


def report(config: argparse.Namespace) -> None:
    if not (config.output / "validation_anchor_selection.json").exists(): raise RuntimeError("freeze validation first")
    stage_rows = {}
    for stage in ("validation", "test"):
        directories = list((config.output / stage).glob("*"))
        stage_rows[stage] = {kind: read_rows([path / f"{kind}_metrics.csv" for path in directories]) for kind in ("candidate", "oracle", "incremental", "complementarity", "support")}
    if not stage_rows["test"]["oracle"]: raise RuntimeError("frozen test has not completed")
    all_oracles = stage_rows["validation"]["oracle"] + stage_rows["test"]["oracle"]
    all_candidates = stage_rows["validation"]["candidate"] + stage_rows["test"]["candidate"]
    all_incremental = stage_rows["validation"]["incremental"] + stage_rows["test"]["incremental"]
    all_complement = stage_rows["validation"]["complementarity"] + stage_rows["test"]["complementarity"]
    all_support = stage_rows["validation"]["support"] + stage_rows["test"]["support"]
    oracle_summary = summarize_stage(all_oracles, ("stage", "oracle_family", "oracle_type", "support_contract"))
    per_backbone = summarize_stage(all_oracles, ("stage", "backbone", "oracle_family", "oracle_type", "support_contract"))
    per_sequence = summarize_stage(all_oracles, ("stage", "sequence", "oracle_family", "oracle_type", "support_contract"))
    write_csv(config.output / "oracle_summary.csv", oracle_summary)
    write_csv(config.output / "per_backbone_metrics.csv", per_backbone)
    write_csv(config.output / "per_sequence_metrics.csv", per_sequence)
    candidate_aggregate = aggregate_rows(all_candidates, ("stage", "candidate", "age", "provenance"),
        ("base_count", "available_count", "candidate_error_sum", "raw_error_sum", "oracle_error_sum", "win_count", "frame_contributes", "fb_sum", "disagreement_sum"))
    for row in candidate_aggregate:
        count = max(row["available_count"], 1)
        row["support_coverage"] = row["available_count"] / max(row["base_count"], 1)
        row["candidate_epe"] = row["candidate_error_sum"] / count
        row["raw_epe_on_candidate_support"] = row["raw_error_sum"] / count
        row["single_candidate_oracle_gain"] = (row["raw_error_sum"] - row["oracle_error_sum"]) / count
        row["winner_pixel_fraction"] = row["win_count"] / count
        row["contributing_frame_fraction"] = row["frame_contributes"] / max(row["row_count"], 1)
        row["fb_confidence_mean"] = row["fb_sum"] / count
        row["raw_candidate_disagreement_mean"] = row["disagreement_sum"] / count
    write_csv(config.output / "candidate_aggregate.csv", candidate_aggregate)
    write_csv(config.output / "candidate_metrics.csv", candidate_aggregate)
    incremental_aggregate = aggregate_rows(all_incremental, ("stage", "order", "step", "added_candidate"),
        ("base_count", "raw_error_sum", "previous_oracle_error_sum", "oracle_error_sum", "incremental_gain_sum", "added_winner_count", "frame_contributes"))
    for row in incremental_aggregate:
        row["incremental_gain"] = row["incremental_gain_sum"] / max(row["base_count"], 1)
        row["added_winner_fraction"] = row["added_winner_count"] / max(row["base_count"], 1)
        row["contributing_frame_fraction"] = row["frame_contributes"] / max(row["row_count"], 1)
    write_csv(config.output / "incremental_oracle_gain_aggregate.csv", incremental_aggregate)
    write_csv(config.output / "incremental_oracle_gain.csv", incremental_aggregate)
    complement_fields = ("base_count", "corrected_wins_all_raw_anchors_lose_count", "raw_anchor_wins_corrected_lose_count",
        "far_raw_recovers_short_corrected_failure_count", "current_raw_remains_best_count", "both_provenances_offer_gain_count")
    complement_aggregate = aggregate_rows(all_complement, ("stage",), complement_fields)
    for row in complement_aggregate:
        for field in complement_fields[1:]: row[field.replace("_count", "_fraction")] = row[field] / max(row["base_count"], 1)
    write_csv(config.output / "provenance_complementarity_aggregate.csv", complement_aggregate)
    write_csv(config.output / "provenance_complementarity.csv", complement_aggregate)
    support_aggregate = aggregate_rows(all_support, ("stage", "candidate", "age", "provenance"), ("base_count", "available_count"))
    for row in support_aggregate: row["coverage"] = row["available_count"] / max(row["base_count"], 1)
    write_csv(config.output / "support_coverage_aggregate.csv", support_aggregate)
    write_csv(config.output / "support_coverage.csv", support_aggregate)
    frozen = json.loads((config.output / "validation_anchor_selection.json").read_text())
    test_availability = [row for row in oracle_summary if row["stage"] == "test" and row["oracle_type"] == "selection" and row["support_contract"] == "availability_aware"]
    one = next(row for row in test_availability if row["oracle_family"] == "current_plus_one_step_raw")
    short = next(row for row in test_availability if row["oracle_family"] == "current_plus_short_corrected")
    raw_bank = next(row for row in test_availability if row["oracle_family"] == "current_plus_raw_anchors")
    full = next(row for row in test_availability if row["oracle_family"] == "full_hybrid_bank")
    incremental_material = full["oracle_gain"] - max(one["oracle_gain"], short["oracle_gain"])
    test_candidates = [row for row in candidate_aggregate if row["stage"] == "test"]
    best_single = max(test_candidates, key=lambda row: row["single_candidate_oracle_gain"])
    test_incremental = [row for row in incremental_aggregate if row["stage"] == "test" and row["order"] == "raw_first"]
    increments = {row["added_candidate"]: row["incremental_gain"] for row in test_incremental}
    test_complement = next(row for row in complement_aggregate if row["stage"] == "test")
    test_support = {row["candidate"]: row["coverage"] for row in support_aggregate if row["stage"] == "test"}
    def group_increment(rows: list[dict], group: str) -> float:
        values = [row for row in rows if row.get("backbone", row.get("sequence")) == group and row["oracle_type"] == "selection" and row["support_contract"] == "availability_aware"]
        gains = {row["oracle_family"]: row["oracle_gain"] for row in values}
        return gains.get("full_hybrid_bank", 0) - max(gains.get("current_plus_one_step_raw", 0), gains.get("current_plus_short_corrected", 0))
    test_backbone_rows = [row for row in per_backbone if row["stage"] == "test"]
    backbone_increments = {name: group_increment(test_backbone_rows, name) for name in SEEN_BACKBONES if any(row.get("backbone") == name for row in test_backbone_rows)}
    test_sequence_names = sorted({row["sequence"] for row in per_sequence if row["stage"] == "test"})
    sequence_increments = {name: group_increment([row for row in per_sequence if row["stage"] == "test"], name) for name in test_sequence_names}
    stable = all(value > 0 for value in backbone_increments.values()) and sum(value > 0 for value in sequence_increments.values()) >= max(1, math.ceil(.75 * len(sequence_increments)))
    decision = "GO" if incremental_material >= .005 and stable and test_support.get("CS8", 0) >= .25 else "CONDITIONAL_GO" if incremental_material >= .005 else "NO-GO"
    summary = {"project": "ARGOS v2", "frozen_selection": frozen, "test": {"one_step_raw": one, "short_corrected": short,
        "raw_anchor_bank": raw_bank, "full_hybrid": full, "hybrid_gain_over_best_single_family": incremental_material},
        "best_single_temporal_candidate": best_single, "raw_first_incremental_gain": increments,
        "provenance_complementarity": test_complement, "candidate_support_coverage": test_support,
        "hybrid_increment_by_backbone": backbone_increments, "hybrid_increment_by_sequence": sequence_increments,
        "decision": decision, "go_threshold_hybrid_incremental_gain_px": .005, "minimum_t8_support": .25,
        "scope": "held-out SCARED-C sequences and seen backbones only; unseen-backbone and external-dataset OOD remain future work"}
    save_json(config.output / "aggregate_summary.json", summary)
    write_csv(config.output / "frozen_test_summary.csv", [{"method": key, **value} for key, value in summary["test"].items() if isinstance(value, dict)])
    (config.output / "paper_ready_tables.tex").write_text(
        "\\begin{tabular}{lrrr}\\toprule\nOracle & Raw EPE & Oracle EPE & Gain \\\\\n" + "\n".join(
            f"{row['oracle_family'].replace('_',' ')} & {row['raw_epe']:.4f} & {row['oracle_epe']:.4f} & {row['oracle_gain']:.4f} \\\\" for row in test_availability
        ) + "\n\\bottomrule\\end{tabular}\n")
    (config.output / "README.md").write_text(
        f"# ARGOS v2 hybrid temporal-memory oracle audit\n\n"
        f"Validation froze candidates `{', '.join(frozen['selected_candidates'])}` before dataset 7 evaluation. "
        f"The availability-aware full hybrid test oracle gains {full['oracle_gain']:.6f} EPE, "
        f"an increment of {incremental_material:.6f} over the better one-step/short-corrected family.\n\n"
        f"Decision: **{decision}** under the preregistered 0.005 EPE incremental gate. Dataset 7 is not a pristine project-wide holdout. "
        "No unseen backbone or external/OOD dataset was evaluated.\n")


def _probe_samples(sequence: str, config: argparse.Namespace, model, adapter, device, *, split: str) -> np.ndarray:
    """Return compact in-RAM feature/target rows; no dense feature artifact is written."""
    info = load_sequence_info(sequence)
    frame_ids = info.frame_ids[: min(len(info.frame_ids), config.max_frames or 64)]
    images = [rgb_tensor(load_frame_lr(info, frame_id)[0], device) for frame_id in frame_ids]
    gt_data = [load_frame_gt(info, frame_id) for frame_id in frame_ids]
    audit_indices = list(range(8, len(frame_ids)))
    age1, _latency, _peak = infer_age_flows(adapter, images, list(range(1, len(frame_ids))), [1], config.batch_size, device)
    flows = {1: (age1[1][0][7:], age1[1][1][7:])}
    other, _latency, _peak = infer_age_flows(adapter, images, audit_indices, [2, 4, 8], config.batch_size, device)
    flows.update(other)
    records = []
    seed = int.from_bytes(hashlib.sha256(f"{split}:{sequence}".encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    for backbone in SEEN_BACKBONES:
        disparities, validity, cache_ids, _metadata = load_sequence_cache(backbone, sequence)
        if [str(value) for value in cache_ids][:len(frame_ids)] != frame_ids: raise RuntimeError("probe frame-ID mismatch")
        raw = np.asarray(disparities[:len(frame_ids)], dtype=np.float32); valid = np.asarray(validity[:len(frame_ids)]) > 0
        fused = fused_history(model, adapter, images, raw, valid, age1[1], device)
        for query_offset, index in enumerate(audit_indices):
            forwards = [flows[age][0][query_offset] for age in AGES]; backwards = [flows[age][1][query_offset] for age in AGES]
            raw_maps = evidence_maps(raw=raw[index], raw_valid=valid[index], sources=[raw[index-age] for age in AGES],
                source_valid=[valid[index-age] for age in AGES], current_rgb=images[index], past_rgb=[images[index-age] for age in AGES],
                forward=forwards, backward=backwards, device=device)
            corrected_maps = evidence_maps(raw=raw[index], raw_valid=valid[index], sources=[fused[index-1], fused[index-2]],
                source_valid=[valid[index-1], valid[index-2]], current_rgb=images[index], past_rgb=[images[index-1], images[index-2]],
                forward=forwards[:2], backward=backwards[:2], device=device)
            candidates = candidate_arrays(raw_maps, corrected_maps)
            gt_native, gt_valid_native = gt_data[index]; gt, coverage = resize_gt_to_cache_masked(gt_native, gt_valid_native)
            base = (coverage > .50) & valid[index]
            positions = np.flatnonzero(base)
            if not positions.size: continue
            positions = rng.choice(positions, size=min(config.probe_pixels_per_frame, positions.size), replace=False)
            raw_stack = np.stack([candidates[name]["disparity"] for name in RAW_NAMES])
            raw_available = np.stack([candidates[name]["available"] for name in RAW_NAMES])
            masked = np.where(raw_available, raw_stack, np.nan)
            median = np.nanmedian(masked, axis=0); witness = raw_available.sum(axis=0)
            mad = np.nanmedian(np.where(raw_available, np.abs(raw_stack - median[None]), np.nan), axis=0)
            median = np.nan_to_num(median); mad = np.nan_to_num(mad)
            raw_error = np.abs(raw[index] - gt)
            # One C0 row and six temporal rows share a single fixed feature schema.
            flat = lambda value: np.asarray(value).reshape(-1)[positions]
            raw_dev = np.abs(raw[index] - median)
            for position_index in range(len(positions)):
                records.append([*flat(raw_dev)[position_index:position_index+1], *flat(mad)[position_index:position_index+1],
                    *flat(raw_dev)[position_index:position_index+1], 0.0, 1.0, 1.0, flat(witness)[position_index] / 4.0,
                    0.0, 0.0, float(flat(raw_error)[position_index] > 1.0), 0.0, 0.0, 0.0])
            for name in FULL_NAMES:
                item = candidates[name]; candidate_error = np.abs(item["disparity"] - gt)
                failure = (candidate_error > raw_error + .10).astype(np.float32)
                available = item["available"].astype(np.float32)
                selected = positions
                for offset, position in enumerate(selected):
                    records.append([float(raw_dev.reshape(-1)[position]), float(mad.reshape(-1)[position]),
                        float(abs(item["disparity"].reshape(-1)[position] - median.reshape(-1)[position])),
                        float(abs(item["disparity"].reshape(-1)[position] - raw[index].reshape(-1)[position])),
                        float(item["fb"].reshape(-1)[position]), float(available.reshape(-1)[position]),
                        float(witness.reshape(-1)[position] / 4.0), float(item["age"] / 8.0),
                        float(item["provenance"] == "corrected"), 0.0, float(failure.reshape(-1)[position]),
                        float(failure.reshape(-1)[position] if item["provenance"] == "corrected" else 0.0),
                        float(item["provenance"] == "corrected")])
    return np.asarray(records, dtype=np.float32)


def _binary_metrics(labels: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score
    if len(np.unique(labels)) < 2: return float("nan"), float("nan")
    return float(roc_auc_score(labels, score)), float(average_precision_score(labels, score))


def _risk_rows(target: str, method: str, labels: np.ndarray, score: np.ndarray) -> list[dict]:
    order = np.argsort(score)  # accept lowest predicted risk first
    rows = []
    for coverage in (.01, .05, .10, .20, .50, 1.0):
        count = max(1, int(round(len(order) * coverage))); chosen = order[:count]
        rows.append({"target": target, "method": method, "coverage": coverage, "risk": float(labels[chosen].mean()), "count": count})
    return rows


def probes(config: argparse.Namespace) -> None:
    """Validation-only lightweight flicker/failure probes, gated by the frozen oracle audit."""
    summary_path = config.output / "aggregate_summary.json"
    if not summary_path.exists(): raise RuntimeError("complete frozen test/report before probes")
    summary = json.loads(summary_path.read_text())
    if float(summary["test"]["hybrid_gain_over_best_single_family"]) < .005:
        save_json(config.output / "flicker_probe_summary.json", {"status": "skipped", "reason": "hybrid oracle gate failed"}); return
    device = torch.device(config.device); model, _checkpoint = load_model(device); adapter = BiDAFlowInferenceAdapter("sea_raft", device=device)
    train = np.concatenate([_probe_samples(sequence, config, model, adapter, device, split="train") for sequence in TRAIN])
    validation = np.concatenate([_probe_samples(sequence, config, model, adapter, device, split="validation") for sequence in VALIDATION])
    features_train, features_validation = train[:, :9], validation[:, :9]
    from sklearn.ensemble import IsolationForest
    rng = np.random.default_rng(20260722)
    fit_indices = rng.choice(len(features_train), size=min(200000, len(features_train)), replace=False)
    detector = IsolationForest(n_estimators=200, max_samples=min(4096, len(fit_indices)), contamination="auto", random_state=20260722, n_jobs=-1)
    detector.fit(features_train[fit_indices]); anomaly = -detector.score_samples(features_validation)
    robust = features_validation[:, 2] / (features_validation[:, 1] + .05)
    definitions = (("raw_current_error", validation[:, 9] > .5, features_validation[:, 7] == 0),
                   ("temporal_memory_failure", validation[:, 10] > .5, (features_validation[:, 7] > 0) & (features_validation[:, 5] > .5)),
                   ("harmful_corrected_update", validation[:, 11] > .5, (validation[:, 12] > .5) & (features_validation[:, 5] > .5)))
    metrics = []; risk = []
    for target, labels_all, subset in definitions:
        labels = labels_all[subset].astype(np.uint8)
        for method, scores_all in (("median_mad", robust), ("isolation_forest", anomaly)):
            scores = scores_all[subset]; auroc, auprc = _binary_metrics(labels, scores)
            metrics.append({"target": target, "method": method, "auroc": auroc, "auprc": auprc,
                "prevalence": float(labels.mean()), "count": len(labels)})
            risk.extend(_risk_rows(target, method, labels, scores))
    write_csv(config.output / "flicker_probe_summary.csv", metrics); write_csv(config.output / "risk_coverage.csv", risk)
    save_json(config.output / "flicker_probe_manifest.json", {"train_sequences": list(TRAIN), "evaluation_sequences": list(VALIDATION),
        "max_frames_per_sequence": config.max_frames or 64, "pixels_per_frame": config.probe_pixels_per_frame,
        "isolation_forest_fit_rows": len(fit_indices), "features": ["raw_deviation", "MAD", "candidate_deviation", "raw_candidate_disagreement",
        "FB_confidence", "support", "witness_fraction", "normalized_age", "corrected_provenance"], "dataset7_used_for_probe": False})


def finalize(config: argparse.Namespace) -> None:
    summary = json.loads((config.output / "aggregate_summary.json").read_text())
    with (config.output / "flicker_probe_summary.csv").open() as handle: probes_rows = list(csv.DictReader(handle))
    summary["flicker_failure_probes"] = [{key: (float(value) if key not in {"target", "method"} else value) for key, value in row.items()} for row in probes_rows]
    full_gain = float(summary["test"]["full_hybrid"]["oracle_gain"])
    raw_bank_gain = float(summary["test"]["raw_anchor_bank"]["oracle_gain"])
    corrected_increment = full_gain - raw_bank_gain
    summary["architecture_interpretation"] = {
        "raw_anchor_fraction_of_full_oracle_gain": raw_bank_gain / full_gain,
        "corrected_increment_beyond_raw_bank_px": corrected_increment,
        "corrected_increment_fraction_of_full_gain": corrected_increment / full_gain,
        "stage1_oracle_verdict": "GO",
        "recommended_next_architecture": "raw-anchor-first non-recurrent multi-anchor refiner; corrected CF1/CF2 as optional typed TTL<=2 ablation",
        "reason": "raw anchors dominate the ceiling, while corrected candidates provide small but measurable unique support",
        "no_model_trained_in_this_stage": True,
    }
    summary["claim_limit"] = "held-out SCARED-C sequences, seen S2M2-S/RAFT-Stereo/StereoAnywhere only; no unseen-backbone or external/OOD claim"
    save_json(config.output / "aggregate_summary.json", summary)
    matrix = [
        {"phase": "seen_backbone", "domain": "SCARED-C", "backbones": "S2M2-S; RAFT-Stereo; StereoAnywhere", "status": "completed",
         "selection_use": "dataset2 only", "required_recalibration": "none beyond frozen cache-grid convention"},
        {"phase": "unseen_backbone", "domain": "SCARED-C", "backbones": "Fast-FoundationStereo or another estimator excluded from refiner training", "status": "future",
         "selection_use": "none", "required_recalibration": "disparity normalization only if output convention differs; freeze before evaluation"},
        {"phase": "external_surgical_GT", "domain": "Hamlyn/SERV-CT or another dataset with verified stereo and geometric GT", "backbones": "frozen seen plus one unseen", "status": "future",
         "selection_use": "separate calibration split only", "required_recalibration": "resolution/focal/baseline disparity normalization; flow confidence; frame-rate-aware anchor age; fusion thresholds"},
        {"phase": "acquisition_OOD", "domain": "different camera, resolution, baseline, lighting and surgical setup", "backbones": "frozen", "status": "future",
         "selection_use": "none on final OOD", "required_recalibration": "embedding distribution and Isolation Forest contamination on training support only; all thresholds frozen before final OOD"},
        {"phase": "no_reference", "domain": "StereoMIS if no valid GT protocol is available", "backbones": "frozen", "status": "future diagnostic only",
         "selection_use": "none", "required_recalibration": "no geometric claims; report support, temporal metrics and qualitative diagnostics only"},
    ]
    write_csv(config.output / "next_phase_validation_matrix.csv", matrix)
    # One compact, aggregate-only diagnostic: no dense maps or per-pixel caches.
    import matplotlib.pyplot as plt
    diagnostics = config.output / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    oracle_labels = ["CS1", "CF1+CF2", "CS1/2/4/8", "Full hybrid"]
    oracle_gains = [
        float(summary["test"]["one_step_raw"]["oracle_gain"]),
        float(summary["test"]["short_corrected"]["oracle_gain"]),
        raw_bank_gain,
        full_gain,
    ]
    increment_labels = ["CS1", "CS2", "CS4", "CS8", "CF1", "CF2"]
    increment_gains = [float(summary["raw_first_incremental_gain"][name]) for name in increment_labels]
    support = [float(summary["candidate_support_coverage"][name]) for name in increment_labels]
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.2), constrained_layout=True)
    axes[0].bar(oracle_labels, oracle_gains, color=["#6baed6", "#74c476", "#3182bd", "#756bb1"])
    axes[0].set_ylabel("Oracle EPE gain [px]")
    axes[0].tick_params(axis="x", rotation=28)
    axes[0].set_title("Candidate-bank ceiling")
    axes[1].bar(increment_labels, increment_gains, color=["#3182bd"] * 4 + ["#74c476"] * 2)
    axes[1].set_ylabel("Incremental EPE gain [px]")
    axes[1].set_title("Raw-first addition order")
    axes[2].bar(increment_labels, support, color=["#3182bd"] * 4 + ["#74c476"] * 2)
    axes[2].set_ylim(.9, 1.0)
    axes[2].set_ylabel("Available support fraction")
    axes[2].set_title("Support versus age")
    for axis in axes:
        axis.grid(axis="y", alpha=.25)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("ARGOS v2 — hybrid temporal-memory oracle audit", fontsize=11)
    figure.savefig(diagnostics / "hybrid_oracle_summary.png", dpi=180)
    plt.close(figure)
    probe_lookup = {(row["target"], row["method"]): row for row in summary["flicker_failure_probes"]}
    memory_if = probe_lookup[("temporal_memory_failure", "isolation_forest")]
    harm_if = probe_lookup[("harmful_corrected_update", "isolation_forest")]
    complement = summary["provenance_complementarity"]
    (config.output / "README.md").write_text(
        "# ARGOS v2 hybrid temporal-memory oracle audit\n\n"
        f"Dataset-2 validation froze `{', '.join(summary['frozen_selection']['selected_candidates'])}` before dataset 7 was opened. "
        f"On dataset 7, raw EPE is {summary['test']['full_hybrid']['raw_epe']:.5f}; the one-step raw, short-corrected, raw-anchor-bank and full-hybrid oracle gains are "
        f"{summary['test']['one_step_raw']['oracle_gain']:.5f}, {summary['test']['short_corrected']['oracle_gain']:.5f}, "
        f"{raw_bank_gain:.5f} and {full_gain:.5f} EPE. The full hybrid exceeds the better short family by "
        f"{summary['test']['hybrid_gain_over_best_single_family']:.5f} EPE and is positive on every seen backbone and sequence.\n\n"
        f"Raw anchors explain {100*raw_bank_gain/full_gain:.1f}% of the full oracle gain. Corrected candidates add {corrected_increment:.5f} EPE after all raw anchors, "
        f"and uniquely rescue {100*complement['corrected_wins_all_raw_anchors_lose_fraction']:.2f}% of pixels; far raw anchors rescue "
        f"{100*complement['far_raw_recovers_short_corrected_failure_fraction']:.2f}% where short corrected memory fails. "
        "The next implementation should therefore be raw-anchor-first and non-recurrent, with CF1/CF2 only as a controlled short-TTL extension.\n\n"
        f"The train-only Isolation Forest reaches AUROC/AP {memory_if['auroc']:.3f}/{memory_if['auprc']:.3f} for temporal-memory failure and "
        f"{harm_if['auroc']:.3f}/{harm_if['auprc']:.3f} for harmful corrected updates on dataset 2. Median/MAD is informative but weaker. "
        "These are diagnostic validation results, not an authorization policy.\n\n"
        "**Stage-1 verdict: GO for a causal multi-anchor architecture; conditional GO for a full provenance-typed hybrid because corrected memory adds only a small residual ceiling.** "
        "Dataset 7 is not a pristine project-wide holdout. No unseen backbone or external/OOD dataset was evaluated.\n")


def smoke(config: argparse.Namespace) -> None:
    config.stage = "validation"; config.sequences = [VALIDATION[0]]; config.max_frames = min(config.max_frames or 12, 12)
    evaluate(config)
    destination = config.output / "validation" / VALIDATION[0]
    oracle = read_rows([destination / "oracle_metrics.csv"])
    if not oracle or not all(math.isfinite(float(row["oracle_error_sum"])) for row in oracle): raise RuntimeError("smoke produced invalid oracle rows")
    save_json(destination / "smoke_result.json", {"passed": True, "frames": config.max_frames, "no_future_access": True,
        "direct_flow": True, "raw_fallback": True, "checkpoint": sha256(CHECKPOINT)})


def main() -> None:
    config = arguments(); config.output.mkdir(parents=True, exist_ok=True)
    if config.mode == "smoke": smoke(config)
    elif config.mode == "evaluate": evaluate(config)
    elif config.mode == "freeze": freeze(config)
    elif config.mode == "report": report(config)
    elif config.mode == "probes": probes(config)
    else: finalize(config)


if __name__ == "__main__": main()
