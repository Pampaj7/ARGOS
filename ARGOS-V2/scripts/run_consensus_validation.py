#!/usr/bin/env python3
"""Stage-gated validation of Cross-Memory Consensus Correction (CMC).

Design, formulation and predeclared gates: ``model_design/CONSENSUS_AUDIT.md``.
Stage 1 sweeps the predeclared config grid on train-split sequences only.
Stage 2 evaluates the single frozen best config on held-out dataset_7.
Stage 3 (only if stage 2 passes) evaluates the unseen backbone.

Reuses the PPM validation machinery for flow inference and BiDA alignment so
the mask policy and namespace are identical to the committed oracle study.
"""
from __future__ import annotations

import argparse
import csv
import json
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
    PRIMARY_UNSEEN_BACKBONE,
    SEEN_BACKBONES,
    build_split_manifest,
    resize_gt_to_cache_masked,
)
from model_design.external_components.bidavideo import BiDAFlowInferenceAdapter  # noqa: E402
from model_design.external_components.temporal_consensus import (  # noqa: E402
    ConsensusConfig,
    consensus_correction,
    consensus_fields,
    sweep_grid,
)
from run_ppmstereo_validation import (  # noqa: E402
    aligned_candidates,
    append_csv,
    infer_age_flows,
    rgb_tensor,
    save_json,
    write_csv,
)

AGES = (1, 2, 4, 8)
THRESHOLDS = (0.05, 0.25, 0.50, 0.90)
PRIMARY = 0.50
DEFAULT_TRAIN_SWEEP_SEQUENCES = (
    "dataset_1_keyframe_2",   # easy/moderate
    "dataset_2_keyframe_4",   # difficult
    "dataset_6_keyframe_4",   # high error variance
)
GATES = {
    "stage1_min_gain_px": 0.005,
    "stage1_max_false_update_rate": 0.20,
    "stage1_max_clean1_degradation": 0.15,
    "stage2_min_gain_px": 0.005,
    "stage2_min_positive_backbones": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("sweep", "heldout", "unseen"), required=True)
    parser.add_argument("--output", type=Path, default=V2_ROOT / "results/consensus_validation")
    parser.add_argument("--backbones", nargs="+", default=list(SEEN_BACKBONES))
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def frame_config_row(
    *,
    stage: str,
    config_label: str,
    backbone: str,
    sequence: str,
    frame_id: str,
    threshold: float,
    raw_error: np.ndarray,
    refined_error: np.ndarray,
    oracle_error: np.ndarray,
    gate: np.ndarray,
    base: np.ndarray,
) -> dict:
    updated = base & gate
    clean1 = base & (raw_error <= 1.0)
    false_update = updated & (refined_error > raw_error + 1e-6)
    clean_degraded = clean1 & (refined_error > raw_error + 1e-6)
    new_bad3 = base & (raw_error <= 3.0) & (refined_error > 3.0)
    return {
        "stage": stage,
        "config": config_label,
        "backbone": backbone,
        "sequence": sequence,
        "frame_id": frame_id,
        "coverage_threshold": threshold,
        "common_valid_count": int(base.sum()),
        "raw_error_sum": float(raw_error[base].sum()) if base.any() else 0.0,
        "refined_error_sum": float(refined_error[base].sum()) if base.any() else 0.0,
        "mechanism_oracle_error_sum": float(oracle_error[base].sum()) if base.any() else 0.0,
        "updated_count": int(updated.sum()),
        "false_update_count": int(false_update.sum()),
        "clean1_count": int(clean1.sum()),
        "clean1_degraded_count": int(clean_degraded.sum()),
        "new_bad3_count": int(new_bad3.sum()),
        "frame_degraded": int(
            float(refined_error[base].sum()) > float(raw_error[base].sum()) + 1e-9
        )
        if base.any()
        else 0,
        "frame_count_one": 1,
    }


def aggregate(rows: list[dict]) -> dict:
    count = sum(int(r["common_valid_count"]) for r in rows)
    updated = sum(int(r["updated_count"]) for r in rows)
    clean1 = sum(int(r["clean1_count"]) for r in rows)
    raw = sum(float(r["raw_error_sum"]) for r in rows) / max(count, 1)
    refined = sum(float(r["refined_error_sum"]) for r in rows) / max(count, 1)
    oracle = sum(float(r["mechanism_oracle_error_sum"]) for r in rows) / max(count, 1)
    return {
        "frames": sum(int(r["frame_count_one"]) for r in rows),
        "common_valid_count": count,
        "raw_epe": raw,
        "refined_epe": refined,
        "gain_px": raw - refined,
        "mechanism_oracle_epe": oracle,
        "mechanism_oracle_gain_px": raw - oracle,
        "update_ratio": updated / max(count, 1),
        "false_update_rate": sum(int(r["false_update_count"]) for r in rows) / max(updated, 1),
        "clean1_degradation_ratio": sum(int(r["clean1_degraded_count"]) for r in rows) / max(clean1, 1),
        "new_bad3_ratio": sum(int(r["new_bad3_count"]) for r in rows) / max(count, 1),
        "frames_worsened_ratio": sum(int(r["frame_degraded"]) for r in rows)
        / max(sum(int(r["frame_count_one"]) for r in rows), 1),
    }


def run_stage(args: argparse.Namespace) -> None:
    stage = args.stage
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest = build_split_manifest()

    if stage == "sweep":
        sequences = args.sequences or list(DEFAULT_TRAIN_SWEEP_SEQUENCES)
        illegal = set(sequences) & set(DEFAULT_VALIDATION_SEQUENCES)
        if illegal:
            raise ValueError(f"sweep must not touch held-out sequences: {sorted(illegal)}")
        backbones = args.backbones
        configs = sweep_grid()
    elif stage == "heldout":
        sequences = args.sequences or list(DEFAULT_VALIDATION_SEQUENCES)
        backbones = args.backbones
        configs = [load_frozen_config(output)]
    else:  # unseen
        gate_check = json.loads((output / "stage2_summary.json").read_text())
        if not gate_check["gate_to_stage3"]["passed"]:
            raise RuntimeError("stage 2 gate failed; unseen stage is not authorized")
        sequences = args.sequences or list(DEFAULT_VALIDATION_SEQUENCES)
        backbones = [PRIMARY_UNSEEN_BACKBONE]
        configs = [load_frozen_config(output)]

    config_meta = {
        "stage": stage,
        "sequences": sequences,
        "backbones": backbones,
        "ages": list(AGES),
        "thresholds": list(THRESHOLDS),
        "primary_coverage_threshold": PRIMARY,
        "bound_px": 3.0,
        "gates": GATES,
        "configs": [c.label() for c in configs],
        "flow_model": "SEA-RAFT",
        "namespace": "cache-grid-from-cached-predictions",
        "command": " ".join(sys.argv),
    }
    save_json(output / f"{stage}_config.json", config_meta)
    save_json(output / "split_manifest.json", manifest | {"sweep_sequences": list(DEFAULT_TRAIN_SWEEP_SEQUENCES)})

    frame_path = output / f"{stage}_frame_metrics.csv"
    if not args.resume:
        frame_path.unlink(missing_ok=True)
    existing = list(csv.DictReader(frame_path.open())) if args.resume and frame_path.exists() else []
    completed = {(r["backbone"], r["sequence"]) for r in existing}
    rows: list[dict] = list(existing)

    device = torch.device(args.device)
    log = (output / "run.log").open("a", buffering=1)
    print(f"COMMAND {config_meta['command']}", file=log)
    adapter = BiDAFlowInferenceAdapter("sea_raft", device=device)
    ages = list(AGES)

    for sequence in sequences:
        pending = [b for b in backbones if (b, sequence) not in completed]
        if not pending:
            print(f"SKIP {sequence}", file=log)
            continue
        info = load_sequence_info(sequence)
        frame_ids = info.frame_ids[: args.frames]
        if len(frame_ids) <= max(ages):
            raise ValueError(f"{sequence}: insufficient frames")
        current_indices = list(range(max(ages), len(frame_ids)))
        print(f"START {sequence} queries={len(current_indices)}", file=log)
        tick = time.perf_counter()
        rgbs = [load_frame_lr(info, fid)[0] for fid in frame_ids]
        images = [rgb_tensor(rgb, device) for rgb in rgbs]
        gt_data = [load_frame_gt(info, fid) for fid in frame_ids]
        flows, flow_latency, peak_mb = infer_age_flows(
            adapter, images, current_indices, ages, args.batch_size, device
        )
        print(f"FLOW {sequence} {time.perf_counter()-tick:.1f}s peak={peak_mb:.0f}MB", file=log)

        for backbone in pending:
            disparities, validity, cache_ids, _meta = load_sequence_cache(backbone, sequence)
            lookup = {str(fid): i for i, fid in enumerate(cache_ids)}
            block: list[dict] = []
            for query_offset, local_index in enumerate(current_indices):
                frame_id = str(frame_ids[local_index])
                raw = np.asarray(disparities[lookup[frame_id]], dtype=np.float32)
                raw_valid = np.asarray(validity[lookup[frame_id]]) > 0
                past_disparities, past_validity, past_images = [], [], []
                forwards, backwards = [], []
                for age in ages:
                    past_id = str(frame_ids[local_index - age])
                    past_disparities.append(np.asarray(disparities[lookup[past_id]], dtype=np.float32))
                    past_validity.append(np.asarray(validity[lookup[past_id]]) > 0)
                    past_images.append(images[local_index - age])
                    forwards.append(flows[age][0][query_offset])
                    backwards.append(flows[age][1][query_offset])
                evidence, _ms = aligned_candidates(
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
                aligned = evidence["aligned_past_disparity"]
                aligned_valid = evidence["aligned_validity"].astype(bool) & evidence[
                    "warp_support"
                ].astype(bool)
                fields = consensus_fields(aligned, aligned_valid)
                gt_native, gt_valid_native = gt_data[local_index]
                gt, coverage = resize_gt_to_cache_masked(gt_native, gt_valid_native)
                raw_error = np.abs(raw - gt)
                delta = np.clip(np.nan_to_num(fields.median - raw), -3.0, 3.0)
                oracle_error = np.minimum(raw_error, np.abs(raw + delta - gt))
                bases = {
                    threshold: (coverage > threshold) & raw_valid & aligned_valid[0]
                    for threshold in THRESHOLDS
                }
                for config in configs:
                    refined, gate = consensus_correction(raw, fields, config)
                    refined_error = np.abs(refined - gt)
                    for threshold in THRESHOLDS:
                        block.append(
                            frame_config_row(
                                stage=stage,
                                config_label=config.label(),
                                backbone=backbone,
                                sequence=sequence,
                                frame_id=frame_id,
                                threshold=threshold,
                                raw_error=raw_error,
                                refined_error=refined_error,
                                oracle_error=oracle_error,
                                gate=gate,
                                base=bases[threshold],
                            )
                        )
            append_csv(frame_path, block)
            rows.extend(block)
            print(f"DONE {backbone}/{sequence} rows={len(block)}", file=log)

    summarize(stage, rows, output)
    log.close()


def load_frozen_config(output: Path) -> ConsensusConfig:
    frozen = json.loads((output / "frozen_config.json").read_text())
    return ConsensusConfig(**frozen["parameters"])


def summarize(stage: str, rows: list[dict], output: Path) -> None:
    groups: dict[tuple[str, str, float], list[dict]] = defaultdict(list)
    for row in rows:
        groups[
            (str(row["config"]), str(row["backbone"]), float(row["coverage_threshold"]))
        ].append(row)

    table = []
    for (config, backbone, threshold), values in sorted(groups.items()):
        table.append(
            {"config": config, "backbone": backbone, "coverage_threshold": threshold}
            | aggregate(values)
        )
    write_csv(output / f"{stage}_summary_by_backbone.csv", table)

    primary_by_config: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if float(row["coverage_threshold"]) == PRIMARY:
            primary_by_config[str(row["config"])].append(row)
    overall = {
        config: aggregate(values) for config, values in sorted(primary_by_config.items())
    }

    if stage == "sweep":
        passing = {
            config: stats
            for config, stats in overall.items()
            if stats["gain_px"] >= GATES["stage1_min_gain_px"]
            and stats["false_update_rate"] <= GATES["stage1_max_false_update_rate"]
            and stats["clean1_degradation_ratio"] <= GATES["stage1_max_clean1_degradation"]
        }
        best = max(passing, key=lambda c: passing[c]["gain_px"]) if passing else None
        summary = {
            "stage": "sweep (train split only)",
            "primary_coverage_threshold": PRIMARY,
            "configs_total": len(overall),
            "configs_passing_gate": len(passing),
            "gate": GATES,
            "best_config": best,
            "best_stats": passing.get(best) if best else None,
            "all_configs": overall,
            "gate_to_stage2": {"passed": best is not None},
        }
        save_json(output / "sweep_summary.json", summary)
        if best:
            params = {}
            for part in best.split("_"):
                key, value = part[0], part[1:]
                params[{"n": "min_count", "s": "spread_max", "d": "disagree_min", "k": "kappa"}[key]] = (
                    int(value) if key == "n" else float(value)
                )
            save_json(
                output / "frozen_config.json",
                {"selected_on": "train sweep, pixel-weighted gain at coverage 0.50", "label": best, "parameters": params},
            )
    elif stage == "heldout":
        per_backbone = {
            row["backbone"]: row
            for row in table
            if float(row["coverage_threshold"]) == PRIMARY and row["config"] in overall
        }
        positive = sum(
            r["gain_px"] >= GATES["stage2_min_gain_px"] for r in per_backbone.values()
        )
        config_label = next(iter(overall))
        stats = overall[config_label]
        safety_ok = (
            stats["false_update_rate"] <= GATES["stage1_max_false_update_rate"]
            and stats["clean1_degradation_ratio"] <= GATES["stage1_max_clean1_degradation"]
        )
        summary = {
            "stage": "heldout (dataset_7, frozen config)",
            "config": config_label,
            "aggregate": stats,
            "per_backbone": per_backbone,
            "positive_backbones": positive,
            "gate_to_stage3": {
                "passed": bool(positive >= GATES["stage2_min_positive_backbones"] and safety_ok),
                "positive_backbones": positive,
                "safety_ok": safety_ok,
            },
        }
        save_json(output / "stage2_summary.json", summary)
    else:
        summary = {
            "stage": "unseen (Fast-FoundationStereo, frozen config)",
            "aggregate": next(iter(overall.values())) if overall else None,
            "per_backbone_rows": table,
        }
        save_json(output / "stage3_summary.json", summary)


if __name__ == "__main__":
    run_stage(parse_args())
