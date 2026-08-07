#!/usr/bin/env python3
"""Fail-closed aggregation and artifact integrity for the nine-run campaign."""
from __future__ import annotations
import argparse, csv, json, math, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from campaign_common import *

REQUIRED = ("manifest.json", "state.json", "runtime_summary.json", "config.yaml", "train_metrics.csv", "validation_metrics.csv",
            "checkpoints/best_validation.pt", "checkpoints/final.pt")

def finite_csv(path: Path) -> tuple[bool, list[str]]:
    bad = []
    with path.open(newline="") as handle:
        for line, row in enumerate(csv.DictReader(handle), 2):
            for key, value in row.items():
                if value in (None, ""): continue
                try: numeric = float(value)
                except ValueError: continue
                if not math.isfinite(numeric): bad.append(f"{path.name}:{line}:{key}")
    return not bad, bad

def completed_runs() -> list[tuple[int, int, Path, dict]]:
    result = []
    for budget in (1, 3, 6):
        for seed in SEEDS:
            root = run_directory(budget, seed); manifest = root / "manifest.json"
            if not manifest.is_file(): raise RuntimeError(f"missing completed manifest: {root}")
            value = json.loads(manifest.read_text())
            if value.get("exit_status") != "complete": raise RuntimeError(f"run is not complete: {root}")
            result.append((budget, seed, root, value))
    return result

def integrity_rows(runs: list[tuple[int, int, Path, dict]]) -> list[dict]:
    rows = []
    for budget, seed, root, manifest in runs:
        artifacts, failures = {}, []
        for name in REQUIRED:
            path = root / name; exists = path.is_file()
            artifacts[name] = {"present": exists, "sha256": sha256(path) if exists else None}
            if not exists: failures.append(f"missing:{name}")
        for name in ("train_metrics.csv", "validation_metrics.csv"):
            path = root / name
            if path.is_file():
                valid, invalid = finite_csv(path); artifacts[name]["finite_numeric_values"] = valid
                failures.extend(f"non_finite:{value}" for value in invalid)
        for field, name in (("best_checkpoint_hash", "checkpoints/best_validation.pt"), ("final_checkpoint_hash", "checkpoints/final.pt")):
            if artifacts[name]["present"] and manifest.get(field) != artifacts[name]["sha256"]:
                failures.append(f"manifest_hash_mismatch:{field}")
        rows.append({"budget": f"{budget}x", "seed": seed, "run_directory": str(root), "complete": True,
                     "integrity_pass": not failures, "failures": failures, "artifacts": artifacts,
                     "manifest_sha256": sha256(root / "manifest.json")})
    return rows

def write_integrity() -> dict:
    runs = completed_runs(); rows = integrity_rows(runs)
    incident_log = CAMPAIGN / "budget_3x/seed_20260724/logs/train.log"
    incident = {"attempt": "budget_3x/seed_20260724", "classification": "historical_invalid_attempt_excluded",
                "cause": "CUDA out of memory while constructing RAM bank on a shared GPU", "evidence_log": str(incident_log),
                "evidence_sha256": sha256(incident_log), "completion_evidence": str(run_directory(3, 20260724) / "manifest.json")}
    report = {"project": "ARGOS v2", "expected_completed_runs": 9, "completed_runs": len(rows),
              "integrity_pass": len(rows) == 9 and all(row["integrity_pass"] for row in rows), "runs": rows,
              "historical_invalid_attempts": [incident], "dataset_7_opened": False}
    atomic_json(CAMPAIGN / "aggregate/run_integrity.json", report)
    flat = [{"budget": row["budget"], "seed": row["seed"], "complete": row["complete"], "integrity_pass": row["integrity_pass"],
             "manifest_sha256": row["manifest_sha256"], "failures": ";".join(row["failures"])} for row in rows]
    write_csv(CAMPAIGN / "aggregate/run_integrity.csv", flat)
    if not report["integrity_pass"]: raise RuntimeError("run integrity failed; selection remains closed")
    return report

def smoke(args: argparse.Namespace) -> None:
    rows = [json.loads(path.read_text()) for path in sorted(args.smoke_root.glob("budget_*_seed_*/manifest.json"))]
    if len(rows) != 3 or {row["budget"] for row in rows} != {"1x", "3x", "6x"}: raise RuntimeError("incomplete smoke aggregate")
    atomic_json(args.smoke_root / "smoke_summary.json", {"project": "ARGOS v2", "status": "PASS", "aggregation": "PASS", "dataset_7_opened": False})

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--smoke-root", type=Path); args = parser.parse_args()
    verify_frozen_core()
    if args.smoke_root: smoke(args); return
    report = write_integrity()
    atomic_json(CAMPAIGN / "aggregate/aggregate_summary.json", {"project": "ARGOS v2", "completed_runs": report["completed_runs"],
                "integrity_pass": report["integrity_pass"], "dataset_7_opened": False})

if __name__ == "__main__": main()
