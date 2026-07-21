#!/usr/bin/env python3
"""Validation-only spatial confidence audit for an ARGOS v2 utility selector.

This does not train or alter a selector.  It asks whether a fixed local mean
of its probability map can reduce harmful raw-versus-memory replacements on
the frozen SCARED-C calibration split.  Only compact aggregate rows are saved.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

V2_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(V2_ROOT), str(V2_ROOT / "scripts")]

from model_design.data.raw_error_dataset import CALIBRATION_SEQUENCES  # noqa: E402
from model_design.data.utility_memory_selector_dataset import (  # noqa: E402
    UtilityMemorySelectorDataset, utility_targets,
)
from model_design.external_components.bidavideo import BiDAFlowInferenceAdapter  # noqa: E402
from run_utility_memory_selector import (  # noqa: E402
    SEEN_BACKBONES, build_evidence, load_model, make_loader, to_device,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--objective", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--preload-workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--kernels", nargs="+", type=int, default=(1, 3, 5, 9),
                        help="Odd local-mean kernel sizes; fixed deterministic policy audit only.")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(rows[0]) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    cli = arguments()
    args = Namespace(
        output=cli.model_output, checkpoint=None, objective=cli.objective,
        device=cli.device, workers=cli.workers, batch_size=cli.batch_size,
        seed=0, coverage_threshold=.5, epsilon=.1, validation_pair_cap=0,
        preload_workers=cli.preload_workers,
    )
    device = torch.device(args.device)
    model, _, checkpoint = load_model(args, device)
    dataset = UtilityMemorySelectorDataset(
        SEEN_BACKBONES, CALIBRATION_SEQUENCES, coverage_threshold=.5,
        selection_only=True,
    )
    preload = dataset.base.preload_frame_data(cli.preload_workers)
    adapter = BiDAFlowInferenceAdapter("sea_raft", device=device)
    kernels = tuple(sorted(set(cli.kernels)))
    if not kernels or any(kernel < 1 or kernel % 2 == 0 for kernel in kernels):
        raise ValueError("kernels must be positive odd integers")
    values: dict[tuple[str, int], list[np.ndarray]] = {
        (kind, kernel): [] for kind in ("probability", "oracle_local_utility") for kernel in kernels
    }
    utilities: list[np.ndarray] = []
    with torch.no_grad():
        for cpu_batch in make_loader(dataset, args, training=False):
            batch = to_device(cpu_batch, device)
            evidence, _ = build_evidence(adapter, batch)
            target = utility_targets(
                batch, evidence.aligned_memory, evidence.aligned_valid,
                evidence.warp_support, epsilon_px=.1, coverage_threshold=.5,
            )
            probability = model(evidence).memory_better_probability
            valid = target.valid.float()
            for kernel in kernels:
                if kernel == 1:
                    probability_score = probability
                    oracle_score = target.utility
                else:
                    support = F.avg_pool2d(valid, kernel, 1, kernel // 2).clamp_min(1e-6)
                    probability_score = F.avg_pool2d(probability * valid, kernel, 1, kernel // 2) / support
                    oracle_score = F.avg_pool2d(target.utility * valid, kernel, 1, kernel // 2) / support
                values[("probability", kernel)].append(probability_score[target.valid].float().cpu().numpy())
                values[("oracle_local_utility", kernel)].append(oracle_score[target.valid].float().cpu().numpy())
            utilities.append(target.utility[target.valid].float().cpu().numpy())
    utility = np.concatenate(utilities)
    rows: list[dict] = []
    for (score_kind, kernel), chunks in values.items():
        score = np.concatenate(chunks)
        for coverage in (.001, .002, .005, .01, .02, .05, .10, .20):
            count = max(1, round(coverage * len(score)))
            selected = np.argpartition(score, -count)[-count:]
            chosen = utility[selected]
            gain = np.maximum(chosen, 0).sum()
            harm = np.maximum(-chosen, 0).sum()
            rows.append({
                "score_kind": score_kind, "kernel": kernel, "coverage": coverage, "selected_count": count,
                "mean_realized_utility_px": float(chosen.mean()),
                "harm_cost_fraction": float(harm / max(gain, 1e-12)),
                "helpful_fraction": float((chosen > .1).mean()),
                "harmful_fraction": float((chosen < -.1).mean()),
                "score_threshold": float(np.min(score[selected])),
            })
    write_csv(cli.output, rows)
    print(json.dumps({"checkpoint": str(checkpoint), "calibration_sequences": list(CALIBRATION_SEQUENCES),
                      "preload": preload, "rows": rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
