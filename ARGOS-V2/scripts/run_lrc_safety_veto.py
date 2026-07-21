#!/usr/bin/env python3
"""Frozen one-way LRC safety veto for the ARGOS v2 causal selector.

The program intentionally trains nothing.  It reads the already frozen
raw-versus-t-1-memory selector, recomputes its causal BiDA evidence, and
allows a frame-relative left--right-consistency (LRC) score only to *close*
an existing authorization.  It cannot open an update or modify the selected
raw/memory value.

``calibrate`` is restricted to dataset_7 keyframes 1--2.  ``evaluate`` is
restricted by the caller to a pre-frozen quantile and is meant to be run only
after calibration has written ``operating_point.json``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

V2_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(V2_ROOT), str(V2_ROOT / "scripts")]

from argos_v2.cache_io import load_sequence_cache  # noqa: E402
from model_design.data.raw_error_dataset import CALIBRATION_SEQUENCES, TEST_SEQUENCES  # noqa: E402
from model_design.data.utility_memory_selector_dataset import UtilityMemorySelectorDataset  # noqa: E402
from model_design.external_components.bidavideo import BiDAFlowInferenceAdapter  # noqa: E402
from model_design.external_components.stereo_lr_consistency import (  # noqa: E402
    left_right_consistency,
    unflip_right_reference_disparity,
)
from model_design.models.lrc_safety_veto import lrc_safety_veto  # noqa: E402
from model_design.models.utility_memory_selector import memory_authorization  # noqa: E402
from run_utility_memory_selector import (  # noqa: E402
    SEEN_BACKBONES,
    build_evidence,
    load_model,
    to_device,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("calibrate", "evaluate"), required=True)
    p.add_argument("--selector-output", type=Path, required=True,
                   help="Frozen utility-selector seed directory containing operating_point.json.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--quantiles", type=float, nargs="+", default=(.50, .75, .90, .95),
                   help="Predeclared calibration candidates.")
    p.add_argument("--quantile", type=float,
                   help="Previously frozen quantile; required by evaluate.")
    p.add_argument("--sequences", nargs="+")
    p.add_argument("--backbones", nargs="+", default=list(SEEN_BACKBONES))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--coverage-threshold", type=float, default=.50)
    p.add_argument("--epsilon", type=float, default=.10,
                   help="Strict margin for reporting a harmful replacement.")
    p.add_argument("--clean-error", type=float, default=.10)
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def make_loader(backbone: str, sequence: str, args: argparse.Namespace) -> DataLoader:
    dataset = UtilityMemorySelectorDataset(
        [backbone], [sequence], coverage_threshold=args.coverage_threshold, selection_only=True,
    )
    options = dict(batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                   pin_memory=True, persistent_workers=args.workers > 0)
    if args.workers > 0:
        options["prefetch_factor"] = 4
    return DataLoader(dataset, **options)


def empty_counts() -> defaultdict[str, float]:
    return defaultdict(float)


def add_batch(counts: defaultdict[str, float], *, common: torch.Tensor, base: torch.Tensor,
              veto: torch.Tensor, raw_error: torch.Tensor, memory_error: torch.Tensor,
              epsilon: float, clean_error: float) -> None:
    """Add counts on one identical paired LRC/GT/common support mask."""
    n = common.sum().item()
    if not n:
        return
    harmful = memory_error > raw_error + epsilon
    clean = raw_error <= clean_error
    base_out = torch.where(base, memory_error, raw_error)
    veto_out = torch.where(veto, memory_error, raw_error)
    counts["valid_pixels"] += n
    counts["raw_error_sum"] += raw_error[common].sum().item()
    counts["base_error_sum"] += base_out[common].sum().item()
    counts["veto_error_sum"] += veto_out[common].sum().item()
    counts["base_authorized"] += (base & common).sum().item()
    counts["veto_authorized"] += (veto & common).sum().item()
    counts["base_harmful"] += (base & common & harmful).sum().item()
    counts["veto_harmful"] += (veto & common & harmful).sum().item()
    counts["raw_clean"] += (common & clean).sum().item()
    counts["base_clean_harmful"] += (base & common & clean & harmful).sum().item()
    counts["veto_clean_harmful"] += (veto & common & clean & harmful).sum().item()


def finalize(counts: dict[str, float], *, quantile: float, backbone: str = "aggregate",
             sequence: str = "aggregate") -> dict:
    n = max(counts["valid_pixels"], 1.0)
    raw = counts["raw_error_sum"] / n
    base = counts["base_error_sum"] / n
    veto = counts["veto_error_sum"] / n
    base_gain = raw - base
    veto_gain = raw - veto
    base_auth = max(counts["base_authorized"], 1.0)
    veto_auth = max(counts["veto_authorized"], 1.0)
    clean = max(counts["raw_clean"], 1.0)
    return {
        "quantile": quantile, "backbone": backbone, "sequence": sequence,
        **dict(counts), "raw_epe": raw, "base_epe": base, "veto_epe": veto,
        "base_gain": base_gain, "veto_gain": veto_gain,
        "gain_retained": veto_gain / max(base_gain, 1e-12),
        "base_coverage": counts["base_authorized"] / n,
        "veto_coverage": counts["veto_authorized"] / n,
        "base_harmful_acceptance": counts["base_harmful"] / base_auth,
        "veto_harmful_acceptance": counts["veto_harmful"] / veto_auth,
        # These are rates over all raw-clean valid pixels, not conditional on an update.
        "base_clean_degradation": counts["base_clean_harmful"] / clean,
        "veto_clean_degradation": counts["veto_clean_harmful"] / clean,
        "harmful_updates_removed": 1.0 - counts["veto_harmful"] / max(counts["base_harmful"], 1.0),
        "authorized_updates_retained": counts["veto_authorized"] / base_auth,
    }


@torch.no_grad()
def run(args: argparse.Namespace, quantiles: tuple[float, ...], sequences: tuple[str, ...]) -> list[dict]:
    device = torch.device(args.device)
    # load_model's Namespace is deliberately minimal and matches the frozen selector.
    selector_args = argparse.Namespace(
        output=args.selector_output, checkpoint=None, objective="legacy", stereo_photometric=False,
        stereo_photometric_kernel=21, coverage_threshold=args.coverage_threshold, epsilon=args.epsilon,
    )
    model, _, checkpoint = load_model(selector_args, device)
    policy = json.loads((args.selector_output / "operating_point.json").read_text())["balanced"]
    adapter = BiDAFlowInferenceAdapter("sea_raft", device=device)
    grouped = {(q, b, s): empty_counts() for q in quantiles for b in args.backbones for s in sequences}
    for backbone in args.backbones:
        for sequence in sequences:
            right_disp, right_valid, _, _ = load_sequence_cache(f"_rightref_{backbone}", sequence)
            for cpu_batch in make_loader(backbone, sequence, args):
                batch = to_device(cpu_batch, device)
                evidence, _ = build_evidence(adapter, batch)
                output = model(evidence)
                base = memory_authorization(
                    output, evidence,
                    probability_threshold=policy["probability_threshold"],
                    utility_threshold_px=policy["utility_threshold_px"],
                    harm_threshold_px=policy["harm_threshold_px"],
                )
                indices = batch["current_index"].cpu().numpy()
                ref = torch.from_numpy(np.asarray(right_disp[indices], dtype=np.float32).copy()).to(device)[:, None]
                ref_valid = torch.from_numpy((np.asarray(right_valid[indices]) > 0).copy()).to(device)[:, None]
                lr = left_right_consistency(
                    batch["raw"], unflip_right_reference_disparity(ref),
                    left_valid=batch["raw_valid"].bool(),
                    right_valid=unflip_right_reference_disparity(ref_valid.float()).bool(),
                )
                common = ((batch["gt_coverage"] > args.coverage_threshold) & batch["raw_valid"].bool()
                          & evidence.aligned_valid.bool() & evidence.warp_support.bool() & lr.valid.bool())
                raw_error = (batch["raw"] - batch["gt"]).abs()
                memory_error = (evidence.aligned_memory - batch["gt"]).abs()
                for quantile in quantiles:
                    veto = lrc_safety_veto(base, lr.residual, lr.valid, quantile=quantile) & common
                    add_batch(grouped[(quantile, backbone, sequence)], common=common, base=base & common,
                              veto=veto, raw_error=raw_error, memory_error=memory_error,
                              epsilon=args.epsilon, clean_error=args.clean_error)
    rows = [finalize(v, quantile=q, backbone=b, sequence=s) for (q, b, s), v in grouped.items()]
    aggregates: list[dict] = []
    for quantile in quantiles:
        for backbone in args.backbones:
            per_backbone = empty_counts()
            for sequence in sequences:
                for key, value in grouped[(quantile, backbone, sequence)].items():
                    per_backbone[key] += value
            aggregates.append(finalize(per_backbone, quantile=quantile, backbone=backbone))
        total = empty_counts()
        for backbone in args.backbones:
            for sequence in sequences:
                for key, value in grouped[(quantile, backbone, sequence)].items():
                    total[key] += value
        aggregates.append(finalize(total, quantile=quantile))
    return rows + aggregates, checkpoint, policy


def choose_quantile(rows: list[dict]) -> dict:
    """Select only from validation aggregates, with conservative predeclared gates."""
    candidates = [r for r in rows if r["backbone"] == "aggregate" and r["sequence"] == "aggregate"]
    eligible = [r for r in candidates if r["gain_retained"] >= .70
                and r["veto_harmful_acceptance"] < r["base_harmful_acceptance"]
                and r["veto_clean_degradation"] < r["base_clean_degradation"]
                and r["authorized_updates_retained"] >= .25]
    # First minimize harmful acceptance, then prefer more retained geometric gain.
    pool = eligible or candidates
    selected = sorted(pool, key=lambda r: (r["veto_harmful_acceptance"], -r["gain_retained"], -r["veto_coverage"]))[0]
    return {"rule": "gain_retained>=0.70; safety strictly improves; >=25% authorized updates retained; then minimum harmful acceptance",
            "eligible_quantiles": [r["quantile"] for r in eligible], "selected": selected}


def main() -> None:
    args = parse_args()
    if args.mode == "calibrate":
        sequences = tuple(args.sequences or CALIBRATION_SEQUENCES)
        quantiles = tuple(args.quantiles)
        if tuple(sequences) != tuple(CALIBRATION_SEQUENCES):
            raise ValueError("calibration is intentionally restricted to held-out SCARED-C keyframes 1/2")
    else:
        if args.quantile is None:
            raise ValueError("--quantile must be a frozen calibration value for evaluate")
        sequences = tuple(args.sequences or TEST_SEQUENCES)
        quantiles = (float(args.quantile),)
    args.output.mkdir(parents=True, exist_ok=True)
    rows, checkpoint, policy = run(args, quantiles, sequences)
    write_csv(args.output / f"{args.mode}_metrics.csv", rows)
    frozen = {
        "selector_checkpoint": str(checkpoint), "selector_checkpoint_sha256": sha256(Path(checkpoint)),
        "selector_policy": policy,
        "bida_source_sha256": sha256(V2_ROOT / "model_design/external_components/bidavideo.py"),
        "lrc_source_sha256": sha256(V2_ROOT / "model_design/external_components/stereo_lr_consistency.py"),
        "veto_source_sha256": sha256(V2_ROOT / "model_design/models/lrc_safety_veto.py"),
        "sequences": list(sequences), "backbones": list(args.backbones),
        "coverage_threshold": args.coverage_threshold, "harm_margin_px": args.epsilon,
        "clean_error_px": args.clean_error,
        "no_ood_or_unseen_data_used": args.mode == "calibrate",
    }
    write_json(args.output / f"{args.mode}_frozen_manifest.json", frozen)
    if args.mode == "calibrate":
        selected = choose_quantile(rows)
        write_json(args.output / "operating_point.json", selected)
        print(json.dumps(selected, indent=2))
    else:
        print(json.dumps([r for r in rows if r["backbone"] == "aggregate"], indent=2))


if __name__ == "__main__":
    main()
