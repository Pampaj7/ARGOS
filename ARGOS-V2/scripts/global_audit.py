#!/usr/bin/env python3
"""Post-run global integrity audit over all expected 17x5=85 sequence/backbone pairs.
Produces a final verdict: PASS, PASS WITH WARNINGS, or FAIL. Never declares PASS if any
expected cache is missing or invalid.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from argos_v2.backbones import BACKBONE_NAMES
from argos_v2.cache_io import validate_written_cache
from argos_v2.paths import CACHE_DIR, V2_ROOT
from argos_v2.scared_c_data import load_sequence_info
from argos_v2.sequences import accepted_sequences

EXPECTED_PAIRS = 85


def main() -> int:
    warnings = []
    failures = []

    seqs = accepted_sequences()
    expected_pairs = {(b, s) for b in BACKBONE_NAMES for s in seqs}
    if len(expected_pairs) != EXPECTED_PAIRS:
        failures.append(f"expected pair count mismatch: {len(expected_pairs)} != {EXPECTED_PAIRS}")

    # missing pairs / per-pair validation
    n_valid = 0
    for backbone, seq in sorted(expected_pairs):
        d = CACHE_DIR / backbone / seq
        if not d.exists():
            failures.append(f"MISSING: {backbone}/{seq}")
            continue
        info = load_sequence_info(seq)
        checks = validate_written_cache(backbone, seq, info.frame_ids)
        if not checks.get("passed"):
            failures.append(f"INVALID: {backbone}/{seq}: {json.dumps({k: v for k, v in checks.items() if k != 'passed'}, default=str)}")
        else:
            n_valid += 1

    # unexpected directories: smoke dirs or backbones outside the canonical 5
    actual_backbone_dirs = {p.name for p in CACHE_DIR.iterdir() if p.is_dir() and p.name != "reports_full"}
    unexpected_backbone_dirs = actual_backbone_dirs - set(BACKBONE_NAMES)
    if unexpected_backbone_dirs:
        failures.append(f"unexpected top-level dirs (smoke leftovers?): {sorted(unexpected_backbone_dirs)}")

    for backbone in BACKBONE_NAMES:
        bd = CACHE_DIR / backbone
        if not bd.exists():
            continue
        actual_seq_dirs = {p.name for p in bd.iterdir() if p.is_dir()}
        unexpected_seqs = {s for s in actual_seq_dirs if s.startswith(".tmp_") or s.startswith("_smoke")}
        if unexpected_seqs:
            failures.append(f"{backbone}: stale tmp/smoke dirs present: {sorted(unexpected_seqs)}")
        extra_seqs = actual_seq_dirs - set(seqs) - unexpected_seqs
        if extra_seqs:
            warnings.append(f"{backbone}: unexpected sequence dirs (not in accepted list): {sorted(extra_seqs)}")

    # storage sanity
    total_bytes = sum(f.stat().st_size for f in CACHE_DIR.rglob("*") if f.is_file() and "reports_full" not in f.parts)
    total_gb = total_bytes / 1e9
    if total_gb > 15.0:
        failures.append(f"storage {total_gb:.2f}GB exceeds the 15GB promotion threshold")
    elif total_gb > 10.0:
        warnings.append(f"storage {total_gb:.2f}GB is above the ~8.5GB plan estimate but under the 15GB threshold")

    # terminology check: no "native-equiv" anywhere in scripts, reports, or metadata
    grep = subprocess.run(
        ["grep", "-rli", "native-equiv\\|native_equiv\\|mae_equiv", str(V2_ROOT)],
        capture_output=True, text=True,
    )
    if grep.stdout.strip():
        failures.append(f"'native-equiv' terminology still present in: {grep.stdout.strip().splitlines()}")

    # cache-resolution metrics must not be mislabeled as native anywhere in the reports
    report_dir = CACHE_DIR / "reports_full"
    cache_report = report_dir / "backbone_metric_summary_cache.csv"
    native_report = report_dir / "native_resolution_sanity.csv"
    if cache_report.exists():
        header = cache_report.open().readline()
        if "native" in header.lower():
            failures.append(f"backbone_metric_summary_cache.csv header mentions 'native': {header.strip()}")
    if native_report.exists():
        header = native_report.open().readline()
        if "_cache" in header.lower():
            failures.append(f"native_resolution_sanity.csv header mentions '_cache': {header.strip()}")

    verdict = "PASS"
    if failures:
        verdict = "FAIL"
    elif warnings:
        verdict = "PASS WITH WARNINGS"

    report = {
        "expected_pairs": EXPECTED_PAIRS,
        "valid_pairs": n_valid,
        "total_storage_gb": total_gb,
        "failures": failures,
        "warnings": warnings,
        "verdict": verdict,
    }
    (report_dir / "global_audit_verdict.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if verdict != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
