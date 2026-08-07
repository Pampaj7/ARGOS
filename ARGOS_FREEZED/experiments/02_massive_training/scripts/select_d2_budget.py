#!/usr/bin/env python3
"""Select a D2 budget only from an externally validated strict-support frame table."""
from __future__ import annotations
import argparse, csv, json, math, random, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from campaign_common import *

METRICS = CAMPAIGN / "selection/d2_common_support_frame_metrics.csv"
RESULT = CAMPAIGN / "selection/validation_selection_results.json"
REQUIRED = {"budget", "seed", "backbone", "sequence", "frame_id", "valid_pixel_count", "raw_error_sum", "h4_error_sum", "multi_error_sum", "parity_pass"}

def rows_from(path: Path) -> list[dict]:
    if not path.is_file(): raise RuntimeError(f"D2 strict-support table missing: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not REQUIRED <= set(rows[0]): raise RuntimeError(f"D2 table lacks required columns: {sorted(REQUIRED)}")
    cleaned = []
    for row in rows:
        if row["budget"] not in {"1x", "3x", "6x"} or int(row["seed"]) not in SEEDS: raise RuntimeError(f"unregistered row: {row}")
        if str(row["parity_pass"]).lower() not in {"1", "true", "yes"}: raise RuntimeError(f"parity/cache failure: {row['budget']}/{row['seed']}/{row['frame_id']}")
        for key in ("valid_pixel_count", "raw_error_sum", "h4_error_sum", "multi_error_sum"):
            row[key] = float(row[key])
            if not math.isfinite(row[key]) or (key == "valid_pixel_count" and row[key] <= 0): raise RuntimeError(f"invalid D2 value: {key}")
        cleaned.append(row)
    expected = {(budget, str(seed)) for budget in ("1x", "3x", "6x") for seed in SEEDS}
    got = {(row["budget"], str(row["seed"])) for row in cleaned}
    if got != expected: raise RuntimeError(f"incomplete D2 rows: missing={sorted(expected-got)}")
    keys = defaultdict(set)
    for row in cleaned: keys[(row["budget"], row["seed"])] .add((row["backbone"], row["sequence"], row["frame_id"]))
    if len({frozenset(value) for value in keys.values()}) != 1: raise RuntimeError("strict common support differs across checkpoints")
    return cleaned

def summaries(rows: list[dict]) -> dict:
    result = {}
    for budget in ("1x", "3x", "6x"):
        per_seed = []
        for seed in SEEDS:
            group = [row for row in rows if row["budget"] == budget and int(row["seed"]) == seed]
            count = sum(row["valid_pixel_count"] for row in group)
            raw = sum(row["raw_error_sum"] for row in group) / count; h4 = sum(row["h4_error_sum"] for row in group) / count
            multi = sum(row["multi_error_sum"] for row in group) / count
            per_seed.append({"seed": seed, "valid_pixel_count": count, "raw_epe": raw, "h4_epe": h4, "multi_epe": multi,
                             "gain_vs_raw": raw-multi, "gain_vs_h4": h4-multi})
        result[budget] = {"per_seed": per_seed, "mean_multi_epe": sum(row["multi_epe"] for row in per_seed)/3,
                          "mean_gain_vs_raw": sum(row["gain_vs_raw"] for row in per_seed)/3,
                          "mean_gain_vs_h4": sum(row["gain_vs_h4"] for row in per_seed)/3}
    return result

def paired_bootstrap(rows: list[dict], smaller: str, larger: str) -> dict:
    grouped = defaultdict(dict)
    for row in rows: grouped[(row["backbone"], row["sequence"])][(row["budget"], int(row["seed"]), row["frame_id"])] = row
    sequences = sorted(grouped)
    rng = random.Random(20260725); samples = []
    for _ in range(10_000):
        numerator = denominator = 0.0
        for sequence in (rng.choice(sequences) for _ in sequences):
            frame_ids = sorted({key[2] for key in grouped[sequence]})
            for frame_id in (rng.choice(frame_ids) for _ in frame_ids):
                for seed in SEEDS:
                    low = grouped[sequence][(smaller, seed, frame_id)]; high = grouped[sequence][(larger, seed, frame_id)]
                    if low["valid_pixel_count"] != high["valid_pixel_count"]: raise RuntimeError("paired rows have unequal common support")
                    numerator += low["multi_error_sum"] - high["multi_error_sum"]; denominator += low["valid_pixel_count"]
        samples.append(numerator / denominator)
    samples.sort(); return {"smaller_budget": smaller, "larger_budget": larger, "replicates": 10_000, "seed": 20260725,
                            "resampling_unit": "paired sequence then frame within sequence", "mean_epe_improvement": sum(samples)/len(samples),
                            "ci95": [samples[249], samples[9749]], "positive_improvement_established": samples[249] > 0}

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--input", type=Path, default=METRICS); args = parser.parse_args(); verify_frozen_core()
    integrity = CAMPAIGN / "aggregate/run_integrity.json"
    if not integrity.is_file() or not json.loads(integrity.read_text()).get("integrity_pass"): raise RuntimeError("run integrity is required before selection")
    rows = rows_from(args.input); table = summaries(rows); eligible = []
    for budget, value in table.items():
        checks = {"mean_gain_vs_raw_positive": value["mean_gain_vs_raw"] > 0,
                  "all_three_seed_gains_vs_raw_positive": all(row["gain_vs_raw"] > 0 for row in value["per_seed"]),
                  "mean_gain_vs_h4_positive": value["mean_gain_vs_h4"] > 0,
                  "at_least_two_of_three_seeds_improve_h4": sum(row["gain_vs_h4"] > 0 for row in value["per_seed"]) >= 2,
                  "cache_support_convention_and_parity_failures": 0}
        value["eligibility"] = checks; value["eligible"] = all(value for key, value in checks.items() if key != "cache_support_convention_and_parity_failures")
        if value["eligible"]: eligible.append(budget)
    if not eligible:
        atomic_json(RESULT, {"project": "ARGOS v2", "verdict": "TRAINING-SCALE NO-GO", "eligible_budgets": [], "table": table,
                             "dataset_7_opened": False, "selection_input_sha256": sha256(args.input)})
        return
    choice = min(eligible, key=lambda budget: table[budget]["mean_multi_epe"]); comparisons = []
    for smaller in ("1x", "3x", "6x"):
        if int(smaller[0]) < int(choice[0]) and smaller in eligible:
            comparison = paired_bootstrap(rows, smaller, choice); comparisons.append(comparison)
            if not comparison["positive_improvement_established"]: choice = smaller
    atomic_json(RESULT, {"project": "ARGOS v2", "verdict": "ELIGIBLE BUDGET SELECTED", "selected_budget": choice, "eligible_budgets": eligible,
                         "table": table, "bootstrap": comparisons, "selection_input_sha256": sha256(args.input), "dataset_7_opened": False})

if __name__ == "__main__": main()
