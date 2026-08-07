#!/usr/bin/env python3
"""Materialize the D2 no-go record and planning-only cross-dataset contract."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from campaign_common import CAMPAIGN, SEEDS, atomic_json, sha256, write_csv

ROOT = CAMPAIGN.parents[1]
AGG = CAMPAIGN / "aggregate"
SELECTION = CAMPAIGN / "selection/validation_selection_results.json"
INTEGRITY = AGG / "run_integrity.json"
CROSS = ROOT / "experiments/04_cross_dataset_scaling"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def run(script: str) -> None:
    subprocess.run([sys.executable, str(HERE / script)], check=True)


def write_d2_artifacts() -> None:
    run("aggregate_campaign.py")
    run("select_d2_budget.py")
    selection, integrity = load_json(SELECTION), load_json(INTEGRITY)
    if selection["verdict"] != "TRAINING-SCALE NO-GO" or selection["eligible_budgets"]:
        raise RuntimeError("D2 did not produce the required no-go")
    if not integrity.get("integrity_pass"):
        raise RuntimeError("integrity did not pass")

    per_seed, per_budget = [], []
    for budget, summary in selection["table"].items():
        for row in summary["per_seed"]:
            per_seed.append({"budget": budget, **row,
                             "strict_support_frame_rows": 11886,
                             "safety_diagnostics": None,
                             "safety_diagnostics_reason": "not present in the strict D2 selection table; prohibited selection input",
                             "provenance": "selection/d2_common_support_frame_metrics.csv"})
        per_budget.append({"budget": budget, "mean_multi_epe": summary["mean_multi_epe"],
                           "mean_gain_vs_raw": summary["mean_gain_vs_raw"], "mean_gain_vs_h4": summary["mean_gain_vs_h4"],
                           "all_three_seed_gains_vs_raw_positive": summary["eligibility"]["all_three_seed_gains_vs_raw_positive"],
                           "at_least_two_of_three_seeds_improve_h4": summary["eligibility"]["at_least_two_of_three_seeds_improve_h4"],
                           "eligible": summary["eligible"], "selection_metric": "strict-common-support D2 EPE",
                           "diagnostics": None,
                           "diagnostics_reason": "not present in the strict D2 selection table",
                           "provenance": "selection/validation_selection_results.json"})
    write_csv(AGG / "per_seed_metrics.csv", per_seed)
    write_csv(AGG / "per_budget_metrics.csv", per_budget)
    write_csv(AGG / "paired_budget_comparisons.csv", [{
        "status": "not_applicable", "comparison": None,
        "reason": "no budget is eligible; preregistered bootstrap parsimony comparison is not entered",
        "provenance": "selection/validation_selection_results.json"}])

    curves, runtimes = [], []
    for run_info in integrity["runs"]:
        root = Path(run_info["run_directory"])
        for source, name in (("train", "train_metrics.csv"), ("validation", "validation_metrics.csv")):
            with (root / name).open(newline="") as handle:
                for row in csv.DictReader(handle):
                    curves.append({"budget": run_info["budget"], "seed": run_info["seed"], "source": source, **row,
                                   "provenance": str(root / name)})
        runtime = load_json(root / "runtime_summary.json")
        runtimes.append({"budget": run_info["budget"], "seed": run_info["seed"], **runtime,
                         "provenance": str(root / "runtime_summary.json")})
    write_csv(AGG / "training_curves.csv", curves)
    write_csv(AGG / "runtime_summary.csv", runtimes)

    summary = {"project": "ARGOS v2", "verdict": selection["verdict"], "completed_runs": integrity["completed_runs"],
               "integrity_pass": True, "dataset_7_opened": False, "eligible_budgets": [],
               "selection_sha256": sha256(SELECTION), "selection_input_sha256": selection["selection_input_sha256"],
               "integrity_sha256": sha256(INTEGRITY), "selection_protocol_sha256": sha256(CAMPAIGN / "selection_protocol.json"),
               "artifacts": ["per_seed_metrics.csv", "per_budget_metrics.csv", "paired_budget_comparisons.csv", "training_curves.csv", "runtime_summary.csv"],
               "unavailable_diagnostics": "null fields are explicitly marked in CSV provenance/reason columns"}
    atomic_json(AGG / "aggregate_summary.json", summary)
    atomic_json(CAMPAIGN / "selection/D7_LOCKED_NO_GO.json", {
        "project": "ARGOS v2", "verdict": "TRAINING-SCALE NO-GO", "dataset_7_opened": False,
        "lock": "D7 remains closed because no D2 budget met the preregistered H4 gate",
        "selection_results_sha256": sha256(SELECTION), "selection_input_sha256": selection["selection_input_sha256"],
        "run_integrity_sha256": sha256(INTEGRITY), "selection_protocol_sha256": sha256(CAMPAIGN / "selection_protocol.json")})
    rows = "\n".join(f"{r['budget']} & {r['mean_multi_epe']:.6f} & {r['mean_gain_vs_h4']:.6f} & no \\\\" for r in per_budget)
    (AGG / "paper_ready_tables.tex").write_text(
        "% D2 strict-common-support selection; all budgets are ineligible.\n"
        "\\begin{tabular}{lrrc}\nBudget & multi EPE & gain vs H4 & eligible" + "\\\\" + "\n\\hline\n" + rows +
        "\n\\end{tabular}\n")
    (AGG / "README.md").write_text(
        "# D2 budget-selection aggregate\n\n"
        "**TRAINING-SCALE NO-GO.** All 1x/3x/6x means are worse than H4 on preregistered strict-common-support D2 EPE; D7 remains locked.\n\n"
        "`per_seed_metrics.csv` and `per_budget_metrics.csv` are derived only from the validated strict D2 table. "
        "`paired_budget_comparisons.csv` is explicitly not applicable because no budget is eligible. "
        "Missing strict-table diagnostics are `null` with a reason, not inferred.\n")


def source(path: str, note: str, hashed: bool = True) -> dict:
    item = {"path": path, "note": note}
    file = Path(path)
    if hashed and file.is_file(): item["sha256"] = sha256(file)
    elif hashed: item["sha256"] = None; item["hash_reason"] = "directory provenance; no synthetic directory hash"
    return item


def write_cross_dataset_contract() -> None:
    CROSS.mkdir(parents=True, exist_ok=True)
    scared_card = "/dtu/p1/leopam/ARGOS/dataset/SCARED-C/DATASET_CARD.md"
    scared_gate = "/dtu/p1/leopam/ARGOS/dataset/SCARED-C/curated/manifests/quality_gate.csv"
    stereo_card = "/dtu/p1/leopam/ARGOS/dataset/StereoMIS/DATASET_CARD.md"
    serv_card = "/dtu/p1/leopam/ARGOS/dataset/SERVCT/DATASET_CARD.md"
    d4d_card = "/dtu/p1/leopam/ARGOS/dataset/D4D/processed/keyframe_stereo_gt_curated/DATASET_CARD.md"
    sota = "/dtu/p1/leopam/ARGOS/SOTA/state.md"
    registry = {
        "project": "ARGOS v2", "phase": "planning-only; no launch", "dataset_7_opened": False,
        "datasets": [
            {"id": "SCARED-C", "status": "included_for_existing_supervised_protocol_only", "supervision": "processed temporal pseudo-GT after quality gate", "evidence_counts": {"gated_sequences": 17, "frames": 16921}, "allowed": "supervised D1/D3/D6; D2 validation only after warmup", "excluded": "any D7 use; future-frame anchors; un-gated sequences", "provenance": [source(scared_card, "quality-gate and frame-count authority"), source(scared_gate, "per-sequence quality-gate manifest"), source("/dtu/p1/leopam/ARGOS/dataset/SCARED-C/curated/geometric_gt/corrected_temporal_gt", "processed data root", False)]},
            {"id": "StereoMIS", "status": "no_reference_only", "supervision": "no dense depth/disparity reference", "evidence_counts": {"pilot_sequences": 3, "rectified_pairs": 38241, "fps": 60}, "pilot_sequences": ["P1", "P2_8", "P3"], "allowed": "no-reference temporal or qualitative diagnostic only after a separate protocol", "excluded": "supervised training, geometric-accuracy claim, zero-shot launch", "provenance": [source(stereo_card, "pilot count, FPS, and no-reference limitation"), source("/dtu/p1/leopam/ARGOS/dataset/StereoMIS/curated/geometric_gt/temporal_sequences", "rectified pilot root", False)]},
            {"id": "SERV-CT", "status": "static_CT_GT_only", "supervision": "CT-derived static geometry reference", "evidence_counts": {"samples": 16}, "allowed": "static OOD geometry/safety planning only", "excluded": "temporal supervision or temporal metric", "provenance": [source(serv_card, "sample count and static-only limitation"), source("/dtu/p1/leopam/ARGOS/dataset/SERVCT/argos/servct_argos/honest_train", "8 static samples", False), source("/dtu/p1/leopam/ARGOS/dataset/SERVCT/argos/servct_argos/honest_test", "8 static samples", False)]},
            {"id": "D4D", "status": "sparse_anchor_GT_zero_shot_only", "supervision": "Zivid structured-light sparse anchors", "evidence_counts": {"anchors_built": 362, "anchors_usable": 239}, "allowed": "future zero-shot sparse-anchor evaluation only, never temporal supervision", "excluded": "training, tuning, temporal supervision, launch in this no-go", "provenance": [source(d4d_card, "anchor counts and sparse/non-rigid restriction"), source("/dtu/p1/leopam/ARGOS/dataset/D4D/processed/keyframe_stereo_gt_curated/manifests/valid_and_warning_manifest.csv", "usable-anchor manifest"), source("/dtu/p1/leopam/ARGOS/dataset/D4D/processed/keyframe_stereo_gt_curated/splits", "deterministic split root", False)]},
            {"id": "Hamlyn", "status": "locally_unavailable_unknown", "supervision": None, "allowed": "none", "excluded": "all claims and launches", "provenance": [source(sota, "SOTA state lines 177-197, 276-279", True)]},
            {"id": "EndoSLAM", "status": "locally_unavailable_unknown", "supervision": None, "allowed": "none", "excluded": "all claims and launches", "provenance": [source(sota, "SOTA state lines 177-197, 276-279", True)]}
        ]}
    splits = {"project": "ARGOS v2", "dataset_7_opened": False,
              "SCARED-C": {"train_dataset_ids": [1, 3, 6], "validation_dataset_ids": [2], "temporal_anchor_ages_frames": [1, 2, 4, 8], "warmup_frames": 8, "rule": "target t is eligible only when all anchors are earlier than t; no future frame"},
              "StereoMIS": {"registered_pilot_sequences": ["P1", "P2_8", "P3"], "split_use": "no-reference only; no train/evaluation launch"},
              "SERV-CT": {"honest_train_samples": 8, "honest_test_samples": 8, "temporal_split": "invalid; static pairs"},
              "D4D": {"source": "/dtu/p1/leopam/ARGOS/dataset/D4D/processed/keyframe_stereo_gt_curated/splits", "rule": "use existing deterministic session/specimen split files; keep both clip anchors together"}}
    usage = {"project": "ARGOS v2", "current_authorization": "no training, no zero-shot, no evaluation", "dataset_7_opened": False,
             "rules": {"SCARED-C": "existing supervised protocol only", "StereoMIS": "no-reference diagnostic protocol required", "SERV-CT": "static geometry only", "D4D": "future zero-shot sparse-anchor only", "Hamlyn": "none", "EndoSLAM": "none"}}
    leakage = {"project": "ARGOS v2", "dataset_7_opened": False,
               "controls": ["D2 is the only budget-selection dataset", "D4D is never train/tune/threshold-select", "SERV-CT is never temporal", "SCARED-C anchors use only earlier frames after warmup", "no frozen recipe, test unlock, training, zero-shot, cache generation, or evaluation is authorized by this registry"]}
    protocols = {"project": "ARGOS v2", "status": "NO-GO planning contract", "dataset_7_opened": False,
                 "temporal": "only quality-gated SCARED-C processed pseudo-GT after warmup; never future frames", "static": "SERV-CT only where a static CT reference is valid", "sparse": "D4D zero-shot only after a separately frozen support protocol", "no_reference": "StereoMIS requires a separately preregistered diagnostic", "unknown": "Hamlyn and EndoSLAM require local inventory before inclusion"}
    audit = {"project": "ARGOS v2", "dataset_7_opened": False, "evidence": registry["datasets"],
             "SOTA_provenance": source(sota, "lines 177-197 and 276-279 constrain permitted uses"),
             "decision": "planning-only registry; no launch"}
    for name, data in (("dataset_registry.json", registry), ("evidence_audit.json", audit), ("splits.json", splits), ("usage.json", usage), ("leakage_controls.json", leakage), ("protocols.json", protocols), ("protocol.json", protocols)):
        atomic_json(CROSS / name, data)
    csv_rows = [{"dataset": d["id"], "status": d["status"], "supervision": d["supervision"], "allowed": d["allowed"], "excluded": d["excluded"]} for d in registry["datasets"]]
    write_csv(CROSS / "dataset_registry.csv", csv_rows)
    (CROSS / "README.md").write_text(
        "# ARGOS v2 cross-dataset scaling — planning-only\n\n"
        "No training, zero-shot run, cache generation, evaluation, frozen recipe, or test unlock is authorized. D7 remains closed.\n\n"
        "SCARED-C is the only registered supervised source: 17 quality-gated processed pseudo-GT sequences / 16,921 frames, causal ages 1/2/4/8 after an 8-frame warmup. "
        "StereoMIS is no-reference (3 pilot sequences / 38,241 pairs at 60fps); SERV-CT is 16 static CT-GT pairs; D4D is 362 sparse anchors / 239 usable and future zero-shot only. "
        "Hamlyn and EndoSLAM are locally unavailable/unknown. See machine-readable registry, audit, splits, usage, leakage controls, and protocols.\n")
    (CROSS / "audit.md").write_text("# Evidence audit\n\nPlanning-only evidence is recorded in `evidence_audit.json`; all inclusion/exclusion decisions are deterministic and preserve the D7 lock.\n")


def self_check() -> None:
    selection = load_json(SELECTION)
    assert selection["verdict"] == "TRAINING-SCALE NO-GO" and not selection["eligible_budgets"]
    assert all(row["mean_gain_vs_h4"] < 0 for row in selection["table"].values())
    lock = load_json(CAMPAIGN / "selection/D7_LOCKED_NO_GO.json")
    assert lock["selection_results_sha256"] == sha256(SELECTION)
    assert lock["run_integrity_sha256"] == sha256(INTEGRITY)
    for path in (AGG / "per_seed_metrics.csv", AGG / "per_budget_metrics.csv", AGG / "paired_budget_comparisons.csv", AGG / "training_curves.csv", AGG / "runtime_summary.csv", AGG / "paper_ready_tables.tex", CROSS / "dataset_registry.json", CROSS / "evidence_audit.json", CROSS / "splits.json", CROSS / "usage.json", CROSS / "leakage_controls.json", CROSS / "protocols.json", CROSS / "DATASET_REGISTRY.yaml", CROSS / "DATASET_AUDIT.md", CROSS / "DATASET_CAPABILITY_MATRIX.csv", CROSS / "DATASET_SPLITS.json", CROSS / "DATA_USAGE_MANIFEST.json", CROSS / "LEAKAGE_AUDIT.csv", CROSS / "CAMERA_GEOMETRY.csv", CROSS / "TEMPORAL_SUPPORT.csv", CROSS / "ZERO_SHOT_PROTOCOL.json", CROSS / "CROSS_TRAINING_PROTOCOL.json", CROSS / "POOLED_TRAINING_PROTOCOL.json", CROSS / "LODO_PROTOCOL.json", CROSS / "scripts/self_check.py"):
        assert path.is_file(), path
    assert load_json(CROSS / "usage.json")["current_authorization"] == "no training, no zero-shot, no evaluation"
    print("PASS materialize_no_go: D2 no-go locked; required artifacts present")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--self-check", action="store_true"); args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    write_d2_artifacts(); write_cross_dataset_contract(); self_check()


if __name__ == "__main__": main()
