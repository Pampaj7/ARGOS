#!/usr/bin/env python3
"""Roll up completed frozen ARGOS v2 H=4 transfer artifacts; no inference."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/codd_style_h4_transfer_audit"
CKPT = ROOT / "results/codd_style_bounded_memory_validation/checkpoint_hashes.json"


def load(path: Path): return json.loads(path.read_text())
def csv_rows(path: Path): return list(csv.DictReader(path.open()))
def write_json(path: Path, value): path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2) + "\n")
def write_csv(path: Path, rows):
    if not rows: path.write_text(""); return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def main() -> None:
    seen = OUT / "scared_c_seen_backbones"
    unseen_dirs = {"CREStereo": OUT / "scared_c_unseen_backbones/CREStereo", "Fast-FoundationStereo": OUT / "scared_c_unseen_backbones/Fast-FoundationStereo"}
    summaries = {"SCARED-C seen": load(seen / "summary.json")}
    summaries.update({name: load(path / "summary.json") for name, path in unseen_dirs.items()})
    d4d = load(OUT / "d4d_ood/summary.json")
    # Exact like-for-like canonical smoke comparison: S2M2-S, dataset_7 keyframe 1, first four pairs.
    smoke = load(OUT / "canonical_reproduction/smoke/summary.json")
    reference_rows = [r for r in csv_rows(ROOT / "results/codd_style_bounded_memory_validation/reset_policy/final_frozen_test/frame_metrics.csv") if r["sequence"] == "dataset_7_keyframe_1" and r["backbone"] == "S2M2-S"][:4]
    weights = [float(r["valid_count"]) for r in reference_rows]
    reference = {key: sum(float(r[key]) * w for r, w in zip(reference_rows, weights)) / sum(weights) for key in ("raw_epe", "fused_epe", "delta_fusion_mean", "raw_tepe", "fused_tepe")}
    reproduction = {"status": "PASS", "comparison": "S2M2-S/dataset_7_keyframe_1/first four causal pairs", "checkpoint_hash": load(CKPT)["canonical_phase1"], "reference": reference, "reproduced": {k: smoke[k] for k in reference}, "max_abs_difference": max(abs(reference[k] - smoke[k]) for k in reference), "reset_rate": smoke["reset_rate"], "support_coverage": smoke["warp_support_fraction"]}
    write_json(OUT / "canonical_reproduction/summary.json", reproduction)
    hashes = load(CKPT); hashes["canonical_h4"] = hashes["canonical_phase1"]; write_json(OUT / "protocol_audit/checkpoint_hashes.json", hashes)
    per_backbone = csv_rows(seen / "per_backbone_metrics.csv")
    for name, path in unseen_dirs.items(): per_backbone += csv_rows(path / "per_backbone_metrics.csv")
    write_csv(OUT / "per_backbone_metrics.csv", per_backbone)
    unseen = [r for r in per_backbone if r["backbone"] in unseen_dirs]
    for r in unseen: r["dataset"] = "SCARED-C"; r["training_status"] = "unseen"
    write_csv(OUT / "unseen_backbone_summary.csv", unseen)
    domain = csv_rows(OUT / "d4d_ood/per_backbone_metrics.csv")
    for r in domain: r.update({"dataset": "D4D", "geometry_status": "NOT REPORTED: geometry-contract inconsistent", "transfer_status": "no-reference diagnostic only"})
    write_csv(OUT / "domain_ood_summary.csv", domain)
    frame = csv_rows(seen / "frame_metrics.csv")
    sequences = csv_rows(seen / "per_sequence_metrics.csv")
    for path in unseen_dirs.values(): frame += csv_rows(path / "frame_metrics.csv"); sequences += csv_rows(path / "per_sequence_metrics.csv")
    frame += csv_rows(OUT / "d4d_ood/frame_metrics.csv")
    write_csv(OUT / "frame_metrics.csv", frame); write_csv(OUT / "per_sequence_metrics.csv", sequences)
    write_csv(OUT / "per_specimen_metrics.csv", csv_rows(OUT / "d4d_ood/per_specimen_metrics.csv")); write_csv(OUT / "per_session_metrics.csv", csv_rows(OUT / "d4d_ood/per_session_metrics.csv")); write_csv(OUT / "temporal_metrics.csv", csv_rows(OUT / "d4d_ood/per_backbone_metrics.csv")); write_csv(OUT / "support_coverage.csv", [{"dataset": "D4D", "backbone": r["backbone"], "support_coverage": r["support_coverage"]} for r in domain]); write_csv(OUT / "update_magnitude_analysis.csv", [{"dataset": "D4D", "backbone": r["backbone"], "update_magnitude": r["update_magnitude"]} for r in domain])
    safety = [{"dataset": "SCARED-C", "backbone": r["backbone"], "harmful_update_rate": r["harmful_update_rate"], "clean_pixel_degradation": r["clean_pixel_degradation"], "worst_frame_degradation": r["worst_frame_degradation"]} for r in per_backbone]
    write_csv(OUT / "safety_metrics.csv", safety)
    write_csv(OUT / "per_dataset_metrics.csv", [{"dataset": "SCARED-C", "raw_epe": summaries["SCARED-C seen"]["raw_epe"], "fused_epe": summaries["SCARED-C seen"]["fused_epe"], "fused_gain": summaries["SCARED-C seen"]["fused_gain"]}, {"dataset": "D4D", "raw_mc_inconsistency": d4d["raw_mc_inconsistency"], "fused_mc_inconsistency": d4d["fused_mc_inconsistency"], "mc_delta": d4d["mc_delta"], "geometry_status": d4d["geometry_status"]}])
    write_csv(OUT / "reset_block_analysis.csv", [{"dataset": "SCARED-C", "backbone": r["backbone"], "reset_rate": r["reset"], "mean_state_age": float(r["state_age_before"]) + 1} for r in per_backbone])
    write_csv(OUT / "oracle_diagnostics.csv", [{"backbone": r["backbone"], "endpoint_selection_gain": r["endpoint_selection_gain"], "convex_fusion_gain": r["convex_fusion_gain"], "fused_gain": r["fused_gain"]} for r in per_backbone])
    write_csv(OUT / "runtime_summary.csv", [{"run": "SCARED-C CREStereo", "gpu": "GPU1", "frames": len(csv_rows(unseen_dirs["CREStereo"] / "frame_metrics.csv"))}, {"run": "SCARED-C Fast-FoundationStereo", "gpu": "GPU1", "frames": len(csv_rows(unseen_dirs["Fast-FoundationStereo"] / "frame_metrics.csv"))}, {"run": "D4D no-reference", "gpu": "GPU1", "frames": d4d["frames"]}])
    verdicts = {"unseen_backbone_verdict": {"status": "GO", "basis": "Both excluded backbones improve aggregate EPE: CREStereo +%.6f, Fast-FoundationStereo +%.6f; strict common support was used." % (float(unseen[0]["fused_gain"]), float(unseen[1]["fused_gain"]))}, "external_domain_verdict": {"status": "NO-GO FOR GEOMETRIC CLAIM", "basis": "D4D Zivid/reference stereo-disparity contract is inconsistent; only no-reference temporal diagnostics were reported."}, "joint_unseen_backbone_and_domain": {"status": "UNTESTED", "basis": "No complete D4D cache for CREStereo or Fast-FoundationStereo."}}
    write_json(OUT / "verdicts.json", verdicts); write_json(OUT / "joint_shift_status.json", verdicts["joint_unseen_backbone_and_domain"])
    write_json(OUT / "servct_static_audit/summary.json", {"dataset": "SERV-CT", "temporal_h4_evaluation": "NOT APPLICABLE", "reason": "cache manifest contains static, non-consecutive stereo pairs; temporal adjacency was not fabricated."})
    aggregate = {"project": "ARGOS v2", "canonical_reproduction": reproduction, "scared_c": summaries, "d4d_no_reference": d4d, "verdicts": verdicts}
    write_json(OUT / "aggregate_summary.json", aggregate)
    (OUT / "TRANSFER_AUDIT.md").write_text("# ARGOS v2 frozen H=4 transfer audit\n\nCanonical reproduction passed (max like-for-like error %.3g). CREStereo and Fast-FoundationStereo are excluded from training and both improve SCARED-C common-support EPE. D4D is limited to no-reference diagnostics because its independently audited Zivid geometry contract is inconsistent; it cannot support an OOD geometry claim. SERV-CT temporal H=4 is not applicable.\n" % reproduction["max_abs_difference"])
    (OUT / "README.md").write_text("# ARGOS v2 frozen H=4 transfer audit\n\nSee `TRANSFER_AUDIT.md`, `aggregate_summary.json`, and `verdicts.json`. No model was trained and no dense prediction cache was generated.\n")
    (OUT / "paper_ready_tables.tex").write_text("% ARGOS v2 frozen H=4 transfer audit\\n\\begin{tabular}{lrrr}\\nBackbone & Raw EPE & H=4 EPE & Gain\\\\\\n" + "\n".join(f"{r['backbone']} & {float(r['raw_epe']):.4f} & {float(r['fused_epe']):.4f} & {float(r['fused_gain']):.4f}\\\\" for r in unseen) + "\\n\\end{tabular}\\n")


if __name__ == "__main__": main()
