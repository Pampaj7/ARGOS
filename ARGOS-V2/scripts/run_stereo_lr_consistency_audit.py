#!/usr/bin/env python3
"""Frozen left--right-consistency confidence audit for ARGOS v2.

This runner evaluates only a cached current left disparity and a separately
cached, flip-swap right-reference disparity.  It never trains, aligns frames
temporally, selects memory, or writes a prediction cache.  The reverse cache
must have been made explicitly with ``run_backbone_cache.py --right-reference``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

V2_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(V2_ROOT), str(V2_ROOT / "scripts")]

from argos_v2.cache_io import load_sequence_cache  # noqa: E402
from model_design.data.temporal_pair_dataset import TemporalPairDataset  # noqa: E402
from model_design.external_components.bidavideo import (  # noqa: E402
    BiDAFlowInferenceAdapter,
    temporal_disparity_evidence,
)
from model_design.external_components.stereo_lr_consistency import (  # noqa: E402
    left_right_consistency,
    unflip_right_reference_disparity,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--right-reference-backbone", default=None,
                        help="default: _rightref_<backbone>; use _smoke__rightref_<backbone> for a temporary smoke cache")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--coverage-threshold", type=float, default=.50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--with-memory", action="store_true",
                        help="also compute frozen causal BiDA t-1 memory and whether LRC difference predicts raw-vs-memory utility")
    return parser.parse_args()


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def auroc(score: np.ndarray, positive: np.ndarray) -> float | None:
    """Mann--Whitney AUROC; diagnostic only and independent of any threshold."""
    count_positive = int(positive.sum()); count_negative = int((~positive).sum())
    if not count_positive or not count_negative:
        return None
    # Stable sorting makes this deterministic.  Values are floating residuals,
    # so exact ties are negligible; tie-average ranks are still handled below.
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    sorted_ranks = np.empty_like(sorted_score, dtype=np.float64)
    start = 0
    while start < len(sorted_score):
        end = start + 1
        while end < len(sorted_score) and sorted_score[end] == sorted_score[start]:
            end += 1
        sorted_ranks[start:end] = (start + 1 + end) / 2.0
        start = end
    ranks = np.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    return float((ranks[positive].sum() - count_positive * (count_positive + 1) / 2) /
                 (count_positive * count_negative))


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    reference_backbone = args.right_reference_backbone or f"_rightref_{args.backbone}"
    right_disparity, right_valid, right_frame_ids, right_meta = load_sequence_cache(reference_backbone, args.sequence)
    if not right_meta.get("right_reference_unflip_required", False):
        raise RuntimeError(f"{reference_backbone}/{args.sequence} is not a flip-swap right-reference cache")

    dataset = TemporalPairDataset([args.backbone], [args.sequence], coverage_threshold=args.coverage_threshold,
                                  max_pairs_per_sequence=args.max_pairs or None, random_clip_start=False)
    # The temporary reverse cache can intentionally be shorter than the
    # canonical left cache, but every evaluated current frame must be present
    # with the identical frame ID.
    reverse_ids = [str(value) for value in right_frame_ids]
    canonical_ids = dataset._infos[args.sequence].frame_ids  # validated dataset-owned source of frame ordering
    if reverse_ids != canonical_ids[:len(reverse_ids)]:
        raise RuntimeError("right-reference cache frame IDs are not the canonical sequence prefix")
    if dataset.records and max(record.current_index for record in dataset.records) >= len(reverse_ids):
        raise RuntimeError("right-reference cache does not cover all requested current frames")

    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    rows: list[dict] = []
    all_residual: list[np.ndarray] = []
    all_error: list[np.ndarray] = []
    all_bad: list[np.ndarray] = []
    all_lrc_difference: list[np.ndarray] = []
    all_utility: list[np.ndarray] = []
    adapter = BiDAFlowInferenceAdapter(device=device) if args.with_memory else None
    for batch in loader:
        indices = batch["current_index"].numpy().astype(np.int64)
        raw = batch["raw"].to(device)
        raw_valid = batch["raw_valid"].to(device).bool()
        gt = batch["gt"].to(device)
        gt_valid = batch["gt_valid"].to(device).bool()
        ref = torch.from_numpy(np.asarray(right_disparity[indices], dtype=np.float32).copy()).to(device)[:, None]
        ref_valid = torch.from_numpy((np.asarray(right_valid[indices]) > 0).copy()).to(device)[:, None]
        ref = unflip_right_reference_disparity(ref)
        ref_valid = unflip_right_reference_disparity(ref_valid.float()).bool()
        evidence = left_right_consistency(raw, ref, left_valid=raw_valid, right_valid=ref_valid)
        valid = gt_valid & evidence.valid & torch.isfinite(evidence.residual)
        memory = memory_valid = warp_support = None
        memory_lrc = None
        if adapter is not None:
            current, past = batch["current_rgb"].to(device), batch["past_rgb"].to(device)
            flow = adapter.infer(torch.cat((current, past), 0), torch.cat((past, current), 0))
            count = current.shape[0]
            temporal = temporal_disparity_evidence(
                raw, batch["past"].to(device), flow[:count], flow[count:],
                current_valid=raw_valid, past_valid=batch["past_valid"].to(device),
                current_rgb=current, past_rgb=past,
            ).as_dict()
            memory = temporal["aligned_past_disparity"]
            memory_valid = temporal["aligned_validity"].bool()
            warp_support = temporal["warp_support"].bool()
            memory_lrc = left_right_consistency(memory, ref, left_valid=memory_valid & warp_support, right_valid=ref_valid)
        raw_error = (raw - gt).abs()
        for index in range(raw.shape[0]):
            mask = valid[index, 0]
            if not mask.any():
                continue
            residual = evidence.residual[index, 0][mask].detach().cpu().numpy()
            error = raw_error[index, 0][mask].detach().cpu().numpy()
            all_residual.append(residual)
            all_error.append(error)
            all_bad.append(error > 1.0)
            row = {
                "backbone": str(batch["backbone"][index]), "sequence": str(batch["sequence"][index]),
                "frame_id": str(batch["current_frame_id"][index]), "valid_count": int(mask.sum()),
                "raw_epe": float(error.mean()), "lrc_mean": float(residual.mean()),
                "lrc_raw_error_correlation": correlation(residual, error),
                "raw_bad1_fraction": float((error > 1.0).mean()),
                "lrc_mean_raw_clean": float(residual[error <= 1.0].mean()) if np.any(error <= 1.0) else None,
                "lrc_mean_raw_bad1": float(residual[error > 1.0].mean()) if np.any(error > 1.0) else None,
                "lrc_support_ratio": float(evidence.valid[index, 0].float().mean().detach().cpu()),
            }
            if memory is not None and memory_lrc is not None and memory_valid is not None and warp_support is not None:
                pair_valid = valid[index, 0] & memory_lrc.valid[index, 0]
                if pair_valid.any():
                    mem_error = (memory[index, 0] - gt[index, 0]).abs()[pair_valid].detach().cpu().numpy()
                    raw_pair_error = raw_error[index, 0][pair_valid].detach().cpu().numpy()
                    utility = raw_pair_error - mem_error
                    lrc_difference = (evidence.residual[index, 0][pair_valid] - memory_lrc.residual[index, 0][pair_valid]).detach().cpu().numpy()
                    all_lrc_difference.append(lrc_difference)
                    all_utility.append(utility)
                    helpful = utility > .10
                    row.update({
                        "memory_pair_valid_count": int(pair_valid.sum()),
                        "memory_epe": float(mem_error.mean()), "raw_pair_epe": float(raw_pair_error.mean()),
                        "memory_oracle_epe": float(np.minimum(raw_pair_error, mem_error).mean()),
                        "lrc_difference_utility_correlation": correlation(lrc_difference, utility),
                        "lrc_memory_better_auroc": auroc(lrc_difference, helpful),
                        "lrc_raw_mean_pair": float(evidence.residual[index, 0][pair_valid].mean().detach().cpu()),
                        "lrc_memory_mean_pair": float(memory_lrc.residual[index, 0][pair_valid].mean().detach().cpu()),
                    })
            rows.append(row)
    if not rows:
        raise RuntimeError("LRC audit had no valid GT/LRC pixels")
    residual, error, bad = (np.concatenate(values) for values in (all_residual, all_error, all_bad))
    summary = {
        "metric_namespace": "cache-grid-from-cached-predictions", "units": "pixels at cache width 180",
        "coverage_threshold": args.coverage_threshold, "backbone": args.backbone, "sequence": args.sequence,
        "right_reference_backbone": reference_backbone, "frames": len(rows), "valid_pixels": int(error.size),
        "raw_epe": float(error.mean()), "lrc_mean": float(residual.mean()),
        "lrc_raw_error_pearson": correlation(residual, error), "lrc_raw_bad1_auroc": auroc(residual, bad),
        "lrc_mean_raw_clean": float(residual[~bad].mean()) if np.any(~bad) else None,
        "lrc_mean_raw_bad1": float(residual[bad].mean()) if np.any(bad) else None,
        "interpretation": "AUROC > .5 means larger LRC residual correlates with raw Bad1; this is a frozen confidence diagnostic, not a selector result.",
    }
    if all_lrc_difference:
        difference, utility = (np.concatenate(values) for values in (all_lrc_difference, all_utility))
        helpful = utility > .10
        summary.update({
            "memory_pair_valid_pixels": int(utility.size),
            "lrc_difference_utility_pearson": correlation(difference, utility),
            "lrc_memory_better_auroc": auroc(difference, helpful),
            "memory_better_fraction_margin_0p10": float(helpful.mean()),
            "lrc_difference_sign_convention": "positive means memory has lower LRC residual than raw",
        })
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "frame_metrics.csv", rows)
    save_json(args.output / "aggregate_summary.json", summary)
    save_json(args.output / "metric_definitions.json", {
        "lrc": "abs(d_left(x) - d_right(x-d_left(x))) after flip-swap right-reference inference and horizontal unflip",
        "valid": "GT-valid & current left prediction-valid & right-reference sampled-valid & in-bounds right support",
        "raw_bad1_auroc": "threshold-free ability of LRC residual to identify raw error > 1 cache-grid pixel",
        "lrc_difference": "LRC(raw) - LRC(aligned t-1 memory); positive values favor memory under stereo consistency",
        "lrc_memory_better_auroc": "threshold-free ability of LRC difference to identify true raw-minus-memory utility > .10 px",
    })


if __name__ == "__main__":
    run(arguments())
