#!/usr/bin/env python3
"""Phase-4 freeze for the ARGOS v2 spatial safety critic.

Sets the predeclared PRIMARY family (stereo) as the selected policy, records a
predeclared safety-oriented SECONDARY operating point, enriches the freeze
manifest with the full provenance the protocol requires, and writes a sidecar
SHA-256 over the final manifest bytes. Uses dataset-2 calibration only; dataset
7 stays closed. Run once, before opening dataset 7.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from run_raw_multi_anchor_spatial_safety_critic import OUTPUT  # noqa: E402
from run_raw_multi_anchor_selective_gate import EXPECTED_FROZEN_SHA256, FIXED_H4_EPE  # noqa: E402
from model_design.models.spatial_error_critic import feature_channels  # noqa: E402

PRIMARY = "stereo"  # predeclared primary (runner docstring: "stereo (E, primary)")
CAL = OUTPUT / "calibration"
MANIFEST = CAL / "freeze_manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def safety_oriented_point(family: str) -> dict:
    """Most conservative feasible-ish point: minimum harmful-update rate among
    grid policies with coverage >= 0.5% of valid pixels (a predeclared
    safety-first fallback; may still exceed the 10% target on these
    checkpoints)."""
    rows = [r for r in csv.DictReader((OUTPUT / "validation_summary.csv").open())
            if r.get("policy") == family and r.get("harmful_update_rate")
            and float(r["coverage"]) >= 0.005]
    best = min(rows, key=lambda r: (float(r["harmful_update_rate"]), -float(r["gain"])))
    keys = ("lambda_uncertainty", "tau_gain", "tau_harm", "gain", "coverage",
            "harmful_update_rate", "clean_degradation", "degraded_frame_fraction",
            "intervention_precision", "gain_over_fixed_h4")
    return {k: float(best[k]) for k in keys}


def detection_metrics(family: str) -> dict:
    rows = [r for r in csv.DictReader((OUTPUT / "harm_detection_metrics.csv").open())
            if r["model"] == family and abs(float(r["margin"]) - 0.10) < 1e-9]
    r = rows[0]
    out = {}
    for k in ("harm_auroc", "harm_auprc", "brier", "ece", "delta_mae", "delta_spearman",
              "raw_error_mae", "proposal_error_mae"):
        if r.get(k):
            out[k] = float(r[k])
    return out


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    manifest["selected_policy"] = PRIMARY
    primary_point = manifest["policies"][PRIMARY]
    secondary_point = safety_oriented_point(PRIMARY)

    try:
        commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                         text=True).strip()
    except Exception:
        commit = "unknown"

    manifest.update({
        "phase": "frozen (dataset-2 only); dataset 7 not yet accessed",
        "repository_commit": commit,
        "seed": 20260722,
        "primary_family": PRIMARY,
        "primary_policy": {k: primary_point[k] for k in (
            "lambda_uncertainty", "tau_gain", "tau_harm", "gain", "coverage",
            "harmful_update_rate", "clean_degradation", "degraded_frame_fraction",
            "intervention_precision", "feasible")},
        "predeclared_secondary_policies": {"safety_oriented": secondary_point},
        "primary_in_channels": feature_channels(PRIMARY),
        "parameter_count": {f: feature_channels(f) for f in ("geometry", "temporal", "stereo", "plane_sweep")},
        "harm_margin": manifest.get("harm_margin", 0.10),
        "help_margin": 0.10,
        "rank_margin": 0.05,
        "fixed_h4_epe_reference_constant": FIXED_H4_EPE,
        "frozen_refiner_sha256": EXPECTED_FROZEN_SHA256,
        "feature_schema_file": "feature_schema.json",
        "primary_validation_detection": detection_metrics(PRIMARY),
        "code_hashes": {
            "run_raw_multi_anchor_spatial_safety_critic.py":
                sha256_file(ROOT / "scripts/run_raw_multi_anchor_spatial_safety_critic.py"),
            "spatial_error_critic.py":
                sha256_file(ROOT / "model_design/models/spatial_error_critic.py"),
        },
        "selected_policy_rationale": (
            "Primary = stereo, the predeclared primary family (plane_sweep is a controlled "
            "ablation). On the full dataset-2 risk-coverage grid NO family reaches a feasible "
            "operating point (harm<=10% at coverage>=0.5% of valid pixels); the four families "
            "are statistically indistinguishable in net gain (0.0056-0.0065 EPE) and the "
            "plane_sweep gain edge over stereo (~0.0002) is within noise, so the predeclared "
            "stereo primary is retained rather than rewarding the heavier ablation family. The "
            "primary point is the net-utility maximiser; a safety-oriented secondary (minimum "
            "harm at coverage>=0.5%) is predeclared for completeness. Safety verdict is expected "
            "NO-GO from validation alone; dataset 7 is opened to confirm the geometry advantage "
            "and quantify D2->D7 transfer of the (infeasible) policy, not to re-select it."),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "test_dataset_id": 7,
        "dataset7_accessed": False,
    })

    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (CAL / "frozen_policy.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    digest = sha256_file(MANIFEST)
    (CAL / "freeze_manifest.sha256").write_text(digest + "\n")

    assert manifest["selected_policy"] == PRIMARY
    assert manifest["dataset7_accessed"] is False
    print(json.dumps({
        "selected_policy": manifest["selected_policy"],
        "primary_point": manifest["primary_policy"],
        "safety_oriented_secondary": secondary_point,
        "freeze_manifest_sha256": digest,
        "repository_commit": commit,
    }, indent=2))


if __name__ == "__main__":
    main()
