#!/usr/bin/env python3
"""D2-only, no-GT diagnostic probe for lower_guard_h4:factory."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
HAND_ROOT = ROOT.parent
FROZEN_ROOT = HAND_ROOT.parent / "ARGOS_FREEZED"
ORIGINAL_H4 = HAND_ROOT / "original_h4"
RUNTIME_PATHS = (ROOT, ORIGINAL_H4, ORIGINAL_H4 / "scripts", FROZEN_ROOT / "src",
                 FROZEN_ROOT / "experiments/02_massive_training/scripts")
PROBE_ROOT = ROOT / "probes"
sys.path[:0] = [str(path) for path in RUNTIME_PATHS]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--backbones", nargs="+", default=["S2M2-S", "RAFT-Stereo", "StereoAnywhere"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--flow-batch-size", type=int, default=32)
    parser.add_argument("--static-check", action="store_true")
    return parser.parse_args()


def validate_output_path(output: Path) -> Path:
    """Refuse writes outside the dedicated candidate-probe subtree."""
    root, candidate = PROBE_ROOT.resolve(), output.resolve()
    if candidate == root or root not in candidate.parents:
        raise ValueError(f"output must be inside {PROBE_ROOT}")
    if candidate.exists():
        raise FileExistsError(f"refusing existing output: {candidate}")
    return candidate


def static_check() -> None:
    """Resolve the full runner dependency graph without loading a frame or writing output."""
    if any(not path.is_dir() for path in RUNTIME_PATHS):
        raise RuntimeError("required probe runtime path is missing")
    import cv2  # noqa: F401
    import torch  # noqa: F401
    from argos_freezed.alignment.sea_raft_adapter import SEARAFTFlowAdapter  # noqa: F401
    from argos_v2.cache_io import load_sequence_cache  # noqa: F401
    from argos_v2.scared_c_data import load_sequence_info, read_rgb  # noqa: F401
    from campaign_common import VALIDATION_SEQUENCES  # noqa: F401
    from lower_guard_h4 import factory  # noqa: F401
    from model_design.comparison.run_comparison import drive  # noqa: F401
    assert frame_stats(np.array([1.]), np.array([2.]), np.array([True]))["ratio_gt_two"] == 0
    try:
        validate_output_path(ROOT / "not-a-probe-output")
    except ValueError:
        pass
    else:
        raise AssertionError("outside output path was accepted")


def frame_stats(raw: np.ndarray, hard: np.ndarray, support: np.ndarray) -> dict[str, Any]:
    finite_raw, finite_hard = np.isfinite(raw), np.isfinite(hard)
    ratio_mask = finite_raw & (raw > 0) & finite_hard
    ratio = hard[ratio_mask] / raw[ratio_mask]
    changed_outside = (~support) & finite_raw & finite_hard & (hard != raw)
    return {"pixels": int(raw.size), "support_pixels": int(support.sum()), "support_coverage": float(support.mean()),
            "hard_change_outside_support": int(changed_outside.sum()), "raw_nonfinite": int((~finite_raw).sum()),
            "hard_nonfinite": int((~finite_hard).sum()), "raw_nonpositive": int((finite_raw & (raw <= 0)).sum()),
            "hard_nonpositive": int((finite_hard & (hard <= 0)).sum()), "ratio_valid": int(ratio.size),
            "ratio_lt_half": int((ratio < .5).sum()), "ratio_gt_two": int((ratio > 2.).sum()),
            "ratio_q01": float(np.quantile(ratio, .01)) if ratio.size else None,
            "ratio_q50": float(np.quantile(ratio, .50)) if ratio.size else None,
            "ratio_q99": float(np.quantile(ratio, .99)) if ratio.size else None, "_ratios": ratio}


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[int(row["state_age"])].append(row)
    result = []
    for age, values in sorted(groups.items()):
        keys = ("pixels", "support_pixels", "hard_change_outside_support", "raw_nonfinite", "hard_nonfinite", "raw_nonpositive", "hard_nonpositive", "ratio_valid", "ratio_lt_half", "ratio_gt_two")
        total = {key: sum(int(row[key]) for row in values) for key in keys}
        ratios = np.concatenate([row["_ratios"] for row in values])
        result.append({"state_age": age, "frames": len(values), **total,
                       "support_coverage": total["support_pixels"] / total["pixels"] if total["pixels"] else 0.0,
                       "ratio_lt_half_rate": total["ratio_lt_half"] / total["ratio_valid"] if total["ratio_valid"] else 0.0,
                       "ratio_gt_two_rate": total["ratio_gt_two"] / total["ratio_valid"] if total["ratio_valid"] else 0.0,
                       "ratio_q01": float(np.quantile(ratios, .01)) if ratios.size else None,
                       "ratio_q50": float(np.quantile(ratios, .50)) if ratios.size else None,
                       "ratio_q99": float(np.quantile(ratios, .99)) if ratios.size else None})
    return result


def run(config: argparse.Namespace) -> list[dict[str, Any]]:
    import cv2
    import torch
    from argos_freezed.alignment.sea_raft_adapter import SEARAFTFlowAdapter
    from argos_v2.cache_io import load_sequence_cache
    from argos_v2.scared_c_data import load_sequence_info, read_rgb
    from campaign_common import VALIDATION_SEQUENCES
    from lower_guard_h4 import factory
    from model_design.comparison.run_comparison import drive

    if config.device != "cuda:0":
        raise ValueError("probe requires logical cuda:0")
    device = torch.device(config.device)
    flow_model, adapter = SEARAFTFlowAdapter(device=device), factory(device=config.device)
    rows = []
    for sequence in VALIDATION_SEQUENCES:
        info = load_sequence_info(sequence); ids = info.frame_ids
        images = [torch.from_numpy(np.ascontiguousarray(cv2.resize(read_rgb(info.seq_dir / "left" / f"{item}.png"), (180, 144), interpolation=cv2.INTER_AREA))).permute(2, 0, 1).float().to(device)[None] for item in ids]
        right = [torch.from_numpy(np.ascontiguousarray(cv2.resize(read_rgb(info.seq_dir / "right" / f"{item}.png"), (180, 144), interpolation=cv2.INTER_AREA))).permute(2, 0, 1).float().to(device)[None] for item in ids]
        pairs = {}
        for start in range(1, len(ids), config.flow_batch_size):
            index = list(range(start, min(start + config.flow_batch_size, len(ids))))
            inferred = flow_model.infer(torch.cat([images[item] for item in index] + [images[item - 1] for item in index]), torch.cat([images[item - 1] for item in index] + [images[item] for item in index])).cpu().numpy()
            for offset, item in enumerate(index): pairs[item, item - 1] = inferred[offset:offset + 1], inferred[len(index) + offset:len(index) + offset + 1]
        def flow(current, past):
            forward, backward = pairs[current["index"], past["index"]]
            return torch.from_numpy(forward).float().to(device), torch.from_numpy(backward).float().to(device)
        for backbone in config.backbones:
            disparity, valid, cache_ids, _ = load_sequence_cache(backbone, sequence)
            if [str(item) for item in cache_ids[:len(ids)]] != ids: raise RuntimeError(f"frame-ID mismatch: {backbone}/{sequence}")
            frames = [{"index": index, "raw": torch.from_numpy(np.asarray(disparity[index:index + 1], np.float32))[:, None].to(device), "raw_valid": torch.from_numpy(np.asarray(valid[index:index + 1]) > 0)[:, None].to(device), "rgb": images[index], "right_rgb": right[index]} for index in range(len(ids))]
            for index, result in drive(adapter, frames, flow)[1:]:
                raw, hard, support = (value[0, 0].detach().cpu().numpy() for value in (frames[index]["raw"], result["hard_disparity"], result["support"]))
                rows.append({"trajectory": sequence, "backbone": backbone, "frame_id": ids[index], "frame_index": index, "reanchor": bool(result["reset"]), "state_age": int(result["state_age"]), **frame_stats(raw, hard, support)})
    return rows


def main() -> None:
    config = arguments()
    if config.static_check:
        if config.output is not None: raise ValueError("--static-check accepts no output path")
        static_check()
        return
    if config.output is None: raise ValueError("output path is required")
    config.output = validate_output_path(config.output)
    config.output.mkdir(parents=True)
    rows = run(config)
    with (config.output / "per_frame.csv").open("w", newline="") as stream:
        fields = [key for key in rows[0] if not key.startswith("_")]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerows({key: value for key, value in row.items() if not key.startswith("_")} for row in rows)
    summary = {"dataset": "SCARED-D2", "ground_truth": "NOT_READ", "rows": len(rows), "grouped_by_state_age": aggregate(rows)}
    (config.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
