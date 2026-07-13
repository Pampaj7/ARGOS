#!/usr/bin/env python3
"""Stage 2/3 PPMStereo controls: unaligned vs BiDA and fixed top-K policies."""
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
    resize_gt_to_cache_masked,
)
from model_design.external_components import ppmstereo as ppm  # noqa: E402
from model_design.external_components.bidavideo import BiDAFlowInferenceAdapter  # noqa: E402
from model_design.models.learned_t1_refiner import LearnedT1Refiner  # noqa: E402
from run_ppmstereo_validation import (  # noqa: E402
    aligned_candidates,
    boundary_mask,
    infer_age_flows,
    rgb_tensor,
    save_json,
    write_csv,
)


METHODS = (
    "A0_raw",
    "A2_existing_learned_t1",
    "A3_unaligned_latest",
    "A4_unaligned_ppm_topk_k3",
    "A4_control_unaligned_uniform_all",
    "A5_bida_aligned_uniform_all",
    "A6_bida_aligned_recent_k3",
    "A7_bida_faithful_score_k3",
    "A7_argos_deterministic_k3",
    "oracle_t1",
    "oracle_multi",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=V2_ROOT / "results/ppmstereo_validation")
    parser.add_argument("--backbones", nargs="+", default=list(SEEN_BACKBONES))
    parser.add_argument("--sequences", nargs="+", default=list(DEFAULT_VALIDATION_SEQUENCES))
    parser.add_argument("--ages", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.05, 0.25, 0.50, 0.90])
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--t1-checkpoint",
        type=Path,
        default=V2_ROOT / "results/learned_t1_refiner/ablations/A2/checkpoints/best_validation.pt",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    if not path.exists():
        write_csv(path, rows)
        return
    fields = next(csv.reader(path.open()))
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def global_scores(
    raw: torch.Tensor,
    memory: torch.Tensor,
    validity: torch.Tensor,
    *,
    fb_confidence: torch.Tensor | None = None,
    photometric: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """ARGOS deterministic quality/similarity/redundancy score components."""
    mask = validity.float()
    count = mask.sum(dim=(2, 3, 4)).clamp_min(1.0)
    similarity = torch.exp(
        -((raw[:, None] - memory).abs() * mask).sum(dim=(2, 3, 4)) / count / 4.0
    )
    support_ratio = mask.mean(dim=(2, 3, 4))
    if fb_confidence is None or photometric is None:
        quality = support_ratio
    else:
        reliable = fb_confidence.clamp(0, 1) * (1.0 - photometric.clamp(0, 1))
        quality = support_ratio * (reliable * mask).sum(dim=(2, 3, 4)) / count
    redundancy_matrix, pair_valid = ppm.spatial_redundancy_matrix(memory / 64.0, validity)
    redundancy = ppm.max_off_diagonal_redundancy(redundancy_matrix, pair_valid)
    candidate_valid = validity.any(dim=(2, 3, 4))
    return quality, similarity, redundancy, candidate_valid


def aggregate_policy(
    memory: torch.Tensor,
    validity: torch.Tensor,
    selection: ppm.TopKSelection,
    weights: torch.Tensor,
    raw: torch.Tensor,
) -> torch.Tensor:
    aggregated = ppm.aggregate_selected_memory(memory, validity, selection, weights)
    return torch.where(aggregated.valid, aggregated.value, raw)


def uniform_selection(
    batch: int, indices: list[int], candidate_valid: torch.Tensor, device: torch.device
) -> tuple[ppm.TopKSelection, torch.Tensor]:
    selected_indices = torch.tensor(indices, device=device)[None].expand(batch, -1)
    valid = candidate_valid.gather(1, selected_indices)
    scores = torch.zeros_like(selected_indices, dtype=torch.float32)
    selection = ppm.TopKSelection(selected_indices, scores, valid)
    weights = valid.float()
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    return selection, weights


@torch.no_grad()
def predictions_for_frame(
    model: LearnedT1Refiner,
    raw_np: np.ndarray,
    raw_valid_np: np.ndarray,
    past_np: list[np.ndarray],
    past_valid_np: list[np.ndarray],
    evidence_np: dict[str, np.ndarray],
    ages: list[int],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    raw = torch.from_numpy(raw_np)[None, None].float().to(device)
    raw_valid = torch.from_numpy(raw_valid_np)[None, None].bool().to(device)
    aligned = torch.from_numpy(evidence_np["aligned_past_disparity"])[None, :, None].float().to(device)
    aligned_valid = torch.from_numpy(evidence_np["aligned_validity"])[None, :, None].bool().to(device)
    unaligned = torch.from_numpy(np.stack(past_np))[None, :, None].float().to(device)
    unaligned_valid = torch.from_numpy(np.stack(past_valid_np))[None, :, None].bool().to(device) & raw_valid[:, None]
    fb = torch.from_numpy(evidence_np["forward_backward_confidence"])[None, :, None].float().to(device)
    photo = torch.from_numpy(evidence_np["photometric_residual"])[None, :, None].float().to(device)

    aq, asim, ared, avalid = global_scores(
        raw, aligned, aligned_valid, fb_confidence=fb, photometric=photo
    )
    uq, usim, ured, uvalid = global_scores(raw, unaligned, unaligned_valid)
    ages_tensor = torch.tensor(ages, device=device)

    unaligned_scores = ppm.deterministic_argos_scores(uq, usim, ured, uvalid)
    unaligned_selection = ppm.deterministic_topk(
        unaligned_scores, 3, ages=ages_tensor, candidate_valid=uvalid
    )
    unaligned_weights = ppm.normalized_play_weights(unaligned_selection)

    argos_scores = ppm.deterministic_argos_scores(aq, asim, ared, avalid)
    argos_selection = ppm.deterministic_topk(
        argos_scores, 3, ages=ages_tensor, candidate_valid=avalid
    )
    argos_weights = ppm.normalized_play_weights(argos_selection)

    faithful_scores, _ = ppm.quality_aware_scores_faithful(asim, aq)
    faithful_selection = ppm.deterministic_topk(
        faithful_scores, 3, ages=ages_tensor, candidate_valid=avalid
    )
    faithful_weights = ppm.normalized_play_weights(faithful_selection)

    all_aligned, all_aligned_weights = uniform_selection(1, list(range(len(ages))), avalid, device)
    recent, recent_weights = uniform_selection(1, list(range(min(3, len(ages)))), avalid, device)
    all_unaligned, all_unaligned_weights = uniform_selection(1, list(range(len(ages))), uvalid, device)

    t1_evidence = {
        "aligned_past_disparity": aligned[:, 0],
        "current_valid": raw_valid,
        "aligned_validity": aligned_valid[:, 0],
        "warp_support": torch.from_numpy(evidence_np["warp_support"][0])[None, None].bool().to(device),
    }
    learned = model(raw, t1_evidence).disparity

    aligned_errors_placeholder = None
    predictions = {
        "A0_raw": raw,
        "A2_existing_learned_t1": learned,
        "A3_unaligned_latest": torch.where(unaligned_valid[:, 0], unaligned[:, 0], raw),
        "A4_unaligned_ppm_topk_k3": aggregate_policy(
            unaligned, unaligned_valid, unaligned_selection, unaligned_weights, raw
        ),
        "A4_control_unaligned_uniform_all": aggregate_policy(
            unaligned, unaligned_valid, all_unaligned, all_unaligned_weights, raw
        ),
        "A5_bida_aligned_uniform_all": aggregate_policy(
            aligned, aligned_valid, all_aligned, all_aligned_weights, raw
        ),
        "A6_bida_aligned_recent_k3": aggregate_policy(
            aligned, aligned_valid, recent, recent_weights, raw
        ),
        "A7_bida_faithful_score_k3": aggregate_policy(
            aligned, aligned_valid, faithful_selection, faithful_weights, raw
        ),
        "A7_argos_deterministic_k3": aggregate_policy(
            aligned, aligned_valid, argos_selection, argos_weights, raw
        ),
    }
    diagnostics = {
        "argos_selected_ages": [ages[index] for index in argos_selection.indices[0].tolist()],
        "faithful_selected_ages": [ages[index] for index in faithful_selection.indices[0].tolist()],
        "unaligned_selected_ages": [ages[index] for index in unaligned_selection.indices[0].tolist()],
        "argos_weight_entropy": float(
            -(argos_weights * argos_weights.clamp_min(1e-12).log()).sum().cpu()
        ),
        "faithful_weight_entropy": float(
            -(faithful_weights * faithful_weights.clamp_min(1e-12).log()).sum().cpu()
        ),
    }
    return {name: value[0, 0].cpu().numpy() for name, value in predictions.items()}, diagnostics


def frame_rows(
    *,
    predictions: dict[str, np.ndarray],
    evidence: dict[str, np.ndarray],
    diagnostics: dict[str, object],
    raw: np.ndarray,
    raw_valid: np.ndarray,
    gt: np.ndarray,
    coverage: np.ndarray,
    threshold: float,
    backbone: str,
    sequence: str,
    frame_id: str,
    frame_index: int,
) -> list[dict]:
    gt_valid = coverage > threshold
    aligned_valid = evidence["aligned_validity"].astype(bool)
    support = evidence["warp_support"].astype(bool)
    common = gt_valid & raw_valid & aligned_valid[0] & support[0]
    boundary = boundary_mask(gt, gt_valid) & common
    raw_error = np.abs(raw - gt)
    aligned_error = np.where(aligned_valid & support, np.abs(evidence["aligned_past_disparity"] - gt[None]), np.inf)
    oracle_t1 = np.minimum(raw_error, aligned_error[0])
    oracle_multi = np.minimum(raw_error, aligned_error.min(axis=0))
    predictions = predictions | {
        "oracle_t1": np.where(aligned_error[0] < raw_error, evidence["aligned_past_disparity"][0], raw),
        "oracle_multi": raw - np.sign(raw - gt) * (raw_error - oracle_multi),
    }
    rows = []
    clean = common & (raw_error <= 1.0)
    for method, prediction in predictions.items():
        error = np.abs(prediction - gt)
        update = np.abs(prediction - raw)
        rows.append(
            {
                "backbone": backbone,
                "sequence": sequence,
                "frame_id": frame_id,
                "frame_index": frame_index,
                "coverage_threshold": threshold,
                "method": method,
                "common_valid_count": int(common.sum()),
                "error_sum": float(error[common].sum()),
                "bad1_count": int((common & (error > 1.0)).sum()),
                "bad3_count": int((common & (error > 3.0)).sum()),
                "absrel_sum": float((error[common] / np.maximum(gt[common], 1e-6)).sum()),
                "boundary_count": int(boundary.sum()),
                "boundary_error_sum": float(error[boundary].sum()),
                "false_update_count": int((clean & (update > 0.05)).sum()),
                "clean_count": int(clean.sum()),
                "clean_degraded_count": int((clean & (error > raw_error + 1e-6)).sum()),
                "clean_update_sum": float(update[clean].sum()),
                "frame_degradation": float(error[common].mean() - raw_error[common].mean()) if common.any() else math.nan,
                "selected_ages": json.dumps(diagnostics.get("argos_selected_ages")) if method == "A7_argos_deterministic_k3" else "",
                "weight_entropy": diagnostics.get("argos_weight_entropy", "") if method == "A7_argos_deterministic_k3" else "",
            }
        )
    return rows


def aggregate(rows: list[dict], output: Path) -> dict:
    groups: dict[tuple[str, str, float, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["backbone"], row["sequence"], float(row["coverage_threshold"]), row["method"])].append(row)
    sequence_rows = []
    for key, values in sorted(groups.items()):
        count = sum(int(row["common_valid_count"]) for row in values)
        boundary_count = sum(int(row["boundary_count"]) for row in values)
        clean_count = sum(int(row["clean_count"]) for row in values)
        degradations = np.asarray([float(row["frame_degradation"]) for row in values])
        sequence_rows.append(
            dict(zip(("backbone", "sequence", "coverage_threshold", "method"), key))
            | {
                "frames": len(values),
                "common_valid_count": count,
                "epe": sum(float(row["error_sum"]) for row in values) / max(count, 1),
                "bad1": sum(int(row["bad1_count"]) for row in values) / max(count, 1),
                "bad3": sum(int(row["bad3_count"]) for row in values) / max(count, 1),
                "absrel": sum(float(row["absrel_sum"]) for row in values) / max(count, 1),
                "boundary_epe": sum(float(row["boundary_error_sum"]) for row in values) / max(boundary_count, 1),
                "false_update_rate": sum(int(row["false_update_count"]) for row in values) / max(clean_count, 1),
                "clean_degradation_ratio": sum(int(row["clean_degraded_count"]) for row in values) / max(clean_count, 1),
                "mean_clean_update": sum(float(row["clean_update_sum"]) for row in values) / max(clean_count, 1),
                "frames_worsened_ratio": float((degradations > 0).mean()),
                "worst_frame_degradation": float(np.max(degradations)),
                "p95_frame_degradation": float(np.percentile(degradations, 95)),
            }
        )
    write_csv(output / "baseline_sequence_metrics.csv", sequence_rows)

    primary = [row for row in sequence_rows if float(row["coverage_threshold"]) == 0.5]
    summaries = {}
    for method in METHODS:
        values = [row for row in primary if row["method"] == method]
        count = sum(int(row["common_valid_count"]) for row in values)
        summaries[method] = {
            "epe": sum(float(row["epe"]) * int(row["common_valid_count"]) for row in values) / max(count, 1),
            "backbones_improved_vs_raw": None,
            "frames_worsened_ratio": float(np.mean([float(row["frames_worsened_ratio"]) for row in values])),
            "worst_frame_degradation": max(float(row["worst_frame_degradation"]) for row in values),
            "clean_degradation_ratio": sum(float(row["clean_degradation_ratio"]) * int(row["common_valid_count"]) for row in values) / max(count, 1),
            "false_update_rate": sum(float(row["false_update_rate"]) * int(row["common_valid_count"]) for row in values) / max(count, 1),
        }
    raw = summaries["A0_raw"]["epe"]
    for method in summaries:
        summaries[method]["delta_epe_vs_raw"] = summaries[method]["epe"] - raw
    payload = {
        "stage": "unaligned versus BiDA plus fixed/deterministic top-K",
        "namespace": "cache-grid-from-cached-predictions",
        "primary_coverage_threshold": 0.5,
        "summaries": summaries,
        "k_policy": "K=1 and K=3 evaluated; K=5 unavailable because exact age set contains four candidates",
    }
    save_json(output / "baseline_summary.json", payload)
    return payload


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    output_csv = args.output / "baseline_frame_metrics.csv"
    if not args.resume:
        output_csv.unlink(missing_ok=True)
    existing = list(csv.DictReader(output_csv.open())) if args.resume and output_csv.exists() else []
    completed = {(row["backbone"], row["sequence"]) for row in existing}
    rows: list[dict] = list(existing)
    device = torch.device(args.device)
    checkpoint = torch.load(args.t1_checkpoint, map_location="cpu")
    model = LearnedT1Refiner(variant=checkpoint["variant"], tau_px=checkpoint["tau_px"])
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    adapter = BiDAFlowInferenceAdapter("sea_raft", device=device)
    log = (args.output / "run.log").open("a", buffering=1)
    command = " ".join(sys.argv)
    print(f"BASELINE_COMMAND {command}", file=log)

    for sequence in args.sequences:
        pending = [backbone for backbone in args.backbones if (backbone, sequence) not in completed]
        if not pending:
            continue
        info = load_sequence_info(sequence)
        frame_ids = info.frame_ids[: min(len(info.frame_ids), args.frames)]
        current_indices = list(range(max(args.ages), len(frame_ids)))
        rgbs = [load_frame_lr(info, frame_id)[0] for frame_id in frame_ids]
        images = [rgb_tensor(rgb, device) for rgb in rgbs]
        gt_data = [load_frame_gt(info, frame_id) for frame_id in frame_ids]
        flows, flow_latency, peak = infer_age_flows(
            adapter, images, current_indices, args.ages, args.batch_size, device
        )
        print(f"BASELINE_FLOW sequence={sequence} latency={flow_latency} peak={peak:.1f}", file=log)
        for backbone in pending:
            disparities, validity, cache_ids, _ = load_sequence_cache(backbone, sequence)
            lookup = {str(frame_id): index for index, frame_id in enumerate(cache_ids)}
            block = []
            for query_offset, current_index in enumerate(current_indices):
                frame_id = frame_ids[current_index]
                cache_index = lookup[str(frame_id)]
                raw = np.asarray(disparities[cache_index], dtype=np.float32)
                raw_valid = np.asarray(validity[cache_index]) > 0
                past, past_valid, past_images, forward, backward = [], [], [], [], []
                for age in args.ages:
                    past_frame = frame_ids[current_index - age]
                    past_index = lookup[str(past_frame)]
                    past.append(np.asarray(disparities[past_index], dtype=np.float32))
                    past_valid.append(np.asarray(validity[past_index]) > 0)
                    past_images.append(images[current_index - age])
                    forward.append(flows[age][0][query_offset])
                    backward.append(flows[age][1][query_offset])
                evidence, _ = aligned_candidates(
                    raw=raw,
                    raw_valid=raw_valid,
                    past_disparities=past,
                    past_validity=past_valid,
                    current_rgb=images[current_index],
                    past_rgb=past_images,
                    forward=forward,
                    backward=backward,
                    device=device,
                )
                predictions, diagnostics = predictions_for_frame(
                    model, raw, raw_valid, past, past_valid, evidence, args.ages, device
                )
                gt_native, gt_valid_native = gt_data[current_index]
                gt, coverage = resize_gt_to_cache_masked(gt_native, gt_valid_native)
                for threshold in args.thresholds:
                    block.extend(
                        frame_rows(
                            predictions=predictions,
                            evidence=evidence,
                            diagnostics=diagnostics,
                            raw=raw,
                            raw_valid=raw_valid,
                            gt=gt,
                            coverage=coverage,
                            threshold=threshold,
                            backbone=backbone,
                            sequence=sequence,
                            frame_id=str(frame_id),
                            frame_index=current_index,
                        )
                    )
            append_csv(output_csv, block)
            rows.extend(block)
            print(f"BASELINE_DONE backbone={backbone} sequence={sequence} rows={len(block)}", file=log)
    payload = aggregate(rows, args.output)
    print(f"BASELINE_COMPLETE methods={len(payload['summaries'])} rows={len(rows)}", file=log)
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
