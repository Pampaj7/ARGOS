#!/usr/bin/env python3
"""Continuous-fusion oracle and selective-risk analysis of the frozen canonical H4 head.

Answers two open paper items in a single pass over the SCARED protocol support:

  (3) The published oracle selects per pixel between raw and aligned memory, i.e. it is
      the ceiling of w in {0,1}.  The head acts on w in [0,1], whose ceiling is the
      continuous oracle w* = clip((gt - raw) / (memory - raw), 0, 1).  Both are reported.

  (8) The effective temporal weight is recovered exactly from the bundle as
      w = (refined - raw) / (memory - raw), and scored as a risk signal for harmful
      intervention (AUROC/AUPRC, risk-coverage, AURC, excess AURC, reliability).

Nothing is trained and no threshold is tuned.  The evaluation support is the driver's
own prediction-independent protocol mask, so these numbers sit on the paper's contract.
"""
from __future__ import annotations

import argparse
import csv
import json
from argparse import Namespace
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]                      # ARGOS_hand/original_h4
COMPARISON = ROOT / "model_design/comparison"
OUT_ROOT = ROOT.parent / "results" / "oracle_risk_analysis"
BINS = 2000                                                     # score resolution for streaming AUROC/AURC
EPS = 1e-6
DISAGREEMENT_FLOOR = 1e-3                                       # |memory - raw| below this cannot be an intervention


def _import_driver():
    import sys
    for path in (str(COMPARISON), str(ROOT / "scripts")):
        if path not in sys.path:
            sys.path.insert(0, path)
    import canonical_h4
    import run_comparison
    return run_comparison, canonical_h4


class Accumulator:
    """Streaming per-(backbone, split) statistics; never holds a full pixel array."""

    def __init__(self) -> None:
        self.pixels = 0
        self.frames = 0
        self.sums = defaultdict(float)
        # score histograms: harmful / beneficial / neutral counts and signed error mass
        self.hist_count = {name: np.zeros((3, BINS)) for name in ("weight", "disagreement", "update")}
        self.hist_delta = {name: np.zeros(BINS) for name in ("weight", "disagreement", "update")}
        self.reliability = np.zeros((2, 20))                    # harmful count / total, 20 weight bins

    def add_scalar(self, name: str, value: float) -> None:
        self.sums[name] += float(value)

    def add_scores(self, name: str, score: np.ndarray, delta: np.ndarray, upper: float) -> None:
        index = np.clip((score / max(upper, EPS) * BINS).astype(np.int64), 0, BINS - 1)
        harmful = delta > 0
        beneficial = delta < 0
        neutral = ~(harmful | beneficial)
        for row, mask in enumerate((harmful, beneficial, neutral)):
            if mask.any():
                self.hist_count[name][row] += np.bincount(index[mask], minlength=BINS)
        self.hist_delta[name] += np.bincount(index, weights=delta, minlength=BINS)


def auroc_from_histogram(positive: np.ndarray, negative: np.ndarray) -> float | None:
    """Mann-Whitney AUC with exact mid-rank handling of ties inside a bin."""
    total_positive, total_negative = positive.sum(), negative.sum()
    if total_positive == 0 or total_negative == 0:
        return None
    below = np.concatenate(([0.0], np.cumsum(negative)[:-1]))     # negatives strictly in lower bins
    wins = float((positive * (below + 0.5 * negative)).sum())
    return wins / (total_positive * total_negative)


def auprc_from_histogram(positive: np.ndarray, negative: np.ndarray) -> float | None:
    """Precision-recall area, sweeping the threshold from the highest score downwards."""
    total_positive = positive.sum()
    if total_positive == 0:
        return None
    true_positive = np.cumsum(positive[::-1])
    predicted = true_positive + np.cumsum(negative[::-1])
    precision = np.where(predicted > 0, true_positive / np.maximum(predicted, 1), 1.0)
    recall = true_positive / total_positive
    return float(np.sum(np.diff(np.concatenate(([0.0], recall))) * precision))


def risk_coverage(count: np.ndarray, delta: np.ndarray) -> dict:
    """Reject the highest-scoring pixels first; risk is mean |error change| kept.

    Returns AURC, the oracle AURC obtained by rejecting the genuinely harmful pixels
    first, and their difference (excess AURC).
    """
    total = count.sum()
    if total == 0:
        return {}
    kept_count = np.cumsum(count)                                # ascending score = kept first
    kept_delta = np.cumsum(delta)
    coverage = kept_count / total
    risk = np.where(kept_count > 0, kept_delta / np.maximum(kept_count, 1), 0.0)
    aurc = float(np.trapezoid(risk, coverage)) if hasattr(np, "trapezoid") else float(np.trapz(risk, coverage))
    return {"aurc": aurc, "coverage": coverage, "risk": risk}


def summarize(accumulator: Accumulator) -> dict:
    row: dict[str, float | None] = {"frames": accumulator.frames, "pixels": accumulator.pixels}
    pixels = max(accumulator.pixels, 1)
    for key, value in accumulator.sums.items():
        row[key] = value / pixels if key.startswith("e_") or key.startswith("frac_") else value
    for name in ("weight", "disagreement", "update"):
        harmful, beneficial = accumulator.hist_count[name][0], accumulator.hist_count[name][1]
        row[f"auroc_{name}"] = auroc_from_histogram(harmful, beneficial)
        row[f"auprc_{name}"] = auprc_from_histogram(harmful, beneficial)
        curve = risk_coverage(accumulator.hist_count[name].sum(axis=0), accumulator.hist_delta[name])
        if curve:
            row[f"aurc_{name}"] = curve["aurc"]
    # oracle AURC: reject the harmful mass first, independent of any score
    harmful_mass = accumulator.sums.get("harm_sum", 0.0)
    row["excess_aurc_weight"] = None
    if "aurc_weight" in row and accumulator.pixels:
        row["oracle_aurc"] = float(accumulator.sums.get("delta_sum", 0.0) - harmful_mass) / pixels
        row["excess_aurc_weight"] = row["aurc_weight"] - row["oracle_aurc"]
    return row


def analyse(bundle: dict, accumulators: dict, reliability: dict) -> None:
    raw = bundle["raw_disparity"].astype(np.float64)
    refined = bundle["refined_disparity"].astype(np.float64)
    memory = bundle["aligned_memory"].astype(np.float64)
    gt = bundle["gt_disparity"].astype(np.float64)
    mask = bundle["protocol_mask"].astype(bool) & bundle["gt_valid"].astype(bool)
    mask &= np.isfinite(raw) & np.isfinite(refined) & np.isfinite(memory) & np.isfinite(gt) & (gt > 0)
    if not mask.any():
        return
    key = (bundle["dataset"], bundle["split"], bundle["backbone"])
    accumulator = accumulators.setdefault(key, Accumulator())
    accumulator.frames += int(bundle["raw_disparity"].shape[0])

    raw, refined, memory, gt = raw[mask], refined[mask], memory[mask], gt[mask]
    accumulator.pixels += raw.size

    e_raw = np.abs(raw - gt)
    e_memory = np.abs(memory - gt)
    e_steer = np.abs(refined - gt)
    accumulator.add_scalar("e_raw", e_raw.sum())
    accumulator.add_scalar("e_memory", e_memory.sum())
    accumulator.add_scalar("e_steer", e_steer.sum())

    # (3) selection oracle over {raw, memory} versus the continuous oracle over [0,1]
    accumulator.add_scalar("e_oracle_binary", np.minimum(e_raw, e_memory).sum())
    span = memory - raw
    usable = np.abs(span) > DISAGREEMENT_FLOOR
    star = np.zeros_like(raw)
    star[usable] = np.clip((gt[usable] - raw[usable]) / span[usable], 0.0, 1.0)
    continuous = raw + star * span
    accumulator.add_scalar("e_oracle_continuous", np.abs(continuous - gt).sum())
    bracketed = ((gt - raw) * (gt - memory)) < 0
    accumulator.add_scalar("frac_bracketed", bracketed.sum())
    accumulator.add_scalar("frac_usable_span", usable.sum())

    # (8) exact recovery of the effective temporal weight, then risk scoring
    weight = np.zeros_like(raw)
    weight[usable] = np.clip((refined[usable] - raw[usable]) / span[usable], 0.0, 1.0)
    delta = e_steer - e_raw                                      # positive = the head introduced error
    accumulator.add_scalar("delta_sum", delta.sum())
    accumulator.add_scalar("harm_sum", np.maximum(delta, 0.0).sum())
    accumulator.add_scalar("benefit_sum", np.maximum(-delta, 0.0).sum())
    accumulator.add_scalar("frac_harmful", (delta > 0).sum())
    accumulator.add_scalar("frac_intervened", (weight > 0).sum())
    accumulator.add_scalar("weight_sum", weight.sum())

    accumulator.add_scores("weight", weight, delta, 1.0)
    disagreement = np.abs(span)
    accumulator.add_scores("disagreement", np.minimum(disagreement, 8.0), delta, 8.0)
    accumulator.add_scores("update", np.minimum(np.abs(refined - raw), 8.0), delta, 8.0)

    bins = np.clip((weight * 20).astype(np.int64), 0, 19)
    accumulator.reliability[0] += np.bincount(bins[delta > 0], minlength=20)
    accumulator.reliability[1] += np.bincount(bins, minlength=20)
    reliability.setdefault(key, accumulator)


def run(args: argparse.Namespace) -> None:
    run_comparison, canonical_h4 = _import_driver()
    # The head was hardcoded, which is how the paper ended up quoting this analysis for a
    # model it was never run on. Selecting it names the output directory too, so the two
    # heads' numbers cannot land in the same place.
    from model_design.comparison.run_comparison import load_factory
    adapter = load_factory(args.module)(device=args.device)
    accumulators: dict = {}
    reliability: dict = {}

    def sink(bundle: dict) -> None:
        print(f"  bundle {bundle['backbone']}/{bundle['sequence_id']} "
              f"frames={bundle['raw_disparity'].shape[0]}", flush=True)
        analyse(bundle, accumulators, reliability)

    for dataset in args.datasets:
        config = Namespace(dataset=dataset, backbones=tuple(args.backbones), sequences=args.sequences,
                           max_frames=args.max_frames, smoke=args.smoke, device=args.device,
                           flow_batch_size=args.flow_batch_size, workers=0, preload_workers=0)
        print(f"[{dataset}] backbones={config.backbones}", flush=True)
        run_comparison._scared(config, adapter, sink)

    rows = []
    for (dataset, split, backbone), accumulator in sorted(accumulators.items()):
        row = {"dataset": dataset, "split": split, "backbone": backbone} | summarize(accumulator)
        pixels = max(accumulator.pixels, 1)
        gain_steer = row["e_raw"] - row["e_steer"]
        gain_binary = row["e_raw"] - row["e_oracle_binary"]
        gain_continuous = row["e_raw"] - row["e_oracle_continuous"]
        row["gain_steer_px"] = gain_steer
        row["gain_oracle_binary_px"] = gain_binary
        row["gain_oracle_continuous_px"] = gain_continuous
        row["recovery_vs_binary_oracle"] = gain_steer / gain_binary if gain_binary else None
        row["recovery_vs_continuous_oracle"] = gain_steer / gain_continuous if gain_continuous else None
        row["bracketed_fraction"] = accumulator.sums["frac_bracketed"] / pixels
        row["harmful_fraction"] = accumulator.sums["frac_harmful"] / pixels
        row["intervened_fraction"] = accumulator.sums["frac_intervened"] / pixels
        row["mean_weight"] = accumulator.sums["weight_sum"] / pixels
        row["H_plus"] = accumulator.sums["harm_sum"] / pixels
        row["B_plus"] = accumulator.sums["benefit_sum"] / pixels
        rows.append(row)

    # One directory per head, so a rerun cannot overwrite the other head's numbers and a
    # reader of the results tree can tell which model produced them.
    OUT = OUT_ROOT if args.module.endswith("canonical_h4:factory") else OUT_ROOT / "a2"
    OUT.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / "oracle_risk_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)

    curves = {}
    for (dataset, split, backbone), accumulator in sorted(accumulators.items()):
        name = f"{dataset}/{split}/{backbone}"
        total = accumulator.reliability[1]
        curves[name] = {
            "reliability_weight_bins": np.linspace(0.025, 0.975, 20).round(4).tolist(),
            "reliability_harm_rate": np.where(total > 0, accumulator.reliability[0] / np.maximum(total, 1), np.nan).round(6).tolist(),
            "reliability_support": total.astype(np.int64).tolist(),
        }
    (OUT / "risk_curves.json").write_text(json.dumps(curves, indent=2) + "\n")
    (OUT / "run_manifest.json").write_text(json.dumps({
        "module": args.module,
        "project": "ARGOS v2",
        "purpose": "continuous-fusion oracle ceiling and selective-risk analysis of the frozen canonical H4 head",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "module_provenance": adapter.describe(),
        "datasets": list(args.datasets),
        "backbones": list(args.backbones),
        "support": "driver protocol_mask (prediction-independent) & GT coverage & finite positive GT",
        "weight_recovery": "w = clip((refined - raw)/(memory - raw), 0, 1) on |memory - raw| > 1e-3",
        "training_performed": False,
        "threshold_tuning_performed": False,
        "score_bins": BINS,
    }, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "rows": len(rows), "csv": str(OUT / "oracle_risk_metrics.csv")}, indent=2))


def self_check() -> None:
    """Smallest runnable check of the two derivations this script adds."""
    raw = np.array([[[1.0, 4.0, 2.0, 5.0]]])
    memory = np.array([[[3.0, 2.0, 2.0, 9.0]]])
    gt = np.array([[[2.0, 5.0, 2.5, 1.0]]])
    refined = raw + 0.25 * (memory - raw)
    bundle = {"dataset": "unit", "split": "unit", "backbone": "unit",
              "raw_disparity": raw, "refined_disparity": refined, "aligned_memory": memory,
              "gt_disparity": gt, "gt_valid": np.ones_like(raw, bool),
              "protocol_mask": np.ones_like(raw, bool)}
    accumulators: dict = {}
    analyse(bundle, accumulators, {})
    a = accumulators[("unit", "unit", "unit")]
    # Only pixel 0 is bracketed: gt=2 lies strictly between raw=1 and memory=3.
    assert a.sums["frac_bracketed"] == 1.0, a.sums["frac_bracketed"]
    # Pixel 2 has memory == raw, so its span carries no intervention.
    assert a.sums["frac_usable_span"] == 3.0, a.sums["frac_usable_span"]
    # Binary oracle: min(1,1) + min(1,3) + min(.5,.5) + min(4,8) = 6.5
    assert abs(a.sums["e_oracle_binary"] - 6.5) < 1e-9, a.sums["e_oracle_binary"]
    # Continuous oracle: 0 + 1 + .5 + 4 = 5.5, exact on the bracketed pixel alone.
    assert abs(a.sums["e_oracle_continuous"] - 5.5) < 1e-9, a.sums["e_oracle_continuous"]
    # The whole point: w in [0,1] has a strictly lower ceiling than w in {0,1}.
    assert a.sums["e_oracle_continuous"] < a.sums["e_oracle_binary"]
    # the recovered weight must equal the 0.25 that generated `refined`
    assert abs(a.sums["weight_sum"] / 3.0 - 0.25) < 1e-9
    positive = np.array([0.0, 1.0, 2.0])
    negative = np.array([2.0, 1.0, 0.0])
    assert auroc_from_histogram(positive, negative) > 0.5
    assert auroc_from_histogram(negative, positive) < 0.5
    print(json.dumps({"status": "PASS", "check": "continuous oracle beats binary oracle; weight recovery exact"}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=["scared-d2", "scared-d7"])
    parser.add_argument("--backbones", nargs="+",
                        default=["S2M2-S", "RAFT-Stereo", "StereoAnywhere", "CREStereo", "Fast-FoundationStereo"])
    parser.add_argument("--sequences", nargs="+")
    parser.add_argument("--module", default="model_design.comparison.canonical_h4:factory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--flow-batch-size", type=int, default=32)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    run(args)


if __name__ == "__main__":
    main()
