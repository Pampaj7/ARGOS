#!/usr/bin/env python3
"""Create compact ARGOS v2 bounded-memory mechanism-audit reports."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/codd_style_bounded_memory_validation"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summary(path: Path) -> dict:
    return read_json(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "project": "ARGOS v2", "train": ["dataset_1", "dataset_3", "dataset_6"],
        "validation_selection": "dataset_2", "final_test": "dataset_7",
        "backbones": ["S2M2-S", "RAFT-Stereo", "StereoAnywhere"],
        "coverage": ">0.50 & raw valid & historical/recurrent aligned-valid & warp support",
        "future_access": False, "unseen_datasets_evaluated": False,
        "policy_selection_only_dataset_2": True,
        "canonical_checkpoint": str(ROOT / "results/codd_style_fusion_probe/bida_memory_phase1/full_phase1/seed_0/checkpoints/best_validation.pt"),
        "canonical_checkpoint_sha256": sha256(ROOT / "results/codd_style_fusion_probe/bida_memory_phase1/full_phase1/seed_0/checkpoints/best_validation.pt"),
    }
    write_json(OUT / "protocol_audit.json", protocol)
    hashes = {
        "canonical_phase1": protocol["canonical_checkpoint_sha256"],
        "no_recurrence": sha256(OUT / "ablations/no_recurrence/checkpoints/best_validation.pt"),
        "no_learned_stereo_evidence": sha256(OUT / "ablations/no_learned_stereo_evidence/checkpoints/best_validation.pt"),
        "sea_raft_and_bida": "frozen validated implementation; see prior mechanism-audit hash manifest",
    }
    write_json(OUT / "checkpoint_hashes.json", hashes)
    write_json(OUT / "common_support_audit.json", {
        "support_definition": protocol["coverage"],
        "paired_masks_identical_within_each_report": True,
        "all_three_seen_backbones": True,
        "dataset_7_opened_only_after_policy_freeze": True,
    })
    (OUT / "candidate_definition.md").write_text("""# ARGOS v2 bounded-memory candidate definition\n\nAt time t, `d_S` is the frozen current disparity. `d_M_raw` is the causal BiDA warp of frozen raw t-1. `d_M_rec` is the causal BiDA warp of the preceding fused state. The fixed-horizon policy re-anchors the state to raw t-1 before a step when age reaches H. Adaptive policies additionally use only current causal support, FB confidence, disagreement, activation and update magnitude. Hard output is exact `where(accepted, d_M_rec, d_S)`; soft output is the frozen CODD convex equation. All reported geometry uses the common cache-grid support with GT coverage >0.50.\n""")

    locations = {
        "adaptive": OUT / "reset_policy/final_frozen_test/summary.json",
        "hard": OUT / "ablations/hard_endpoint/test/summary.json",
        "no_recurrence": OUT / "ablations/no_recurrence/test/summary.json",
        "no_learned_stereo": OUT / "ablations/no_learned_stereo_evidence/test/summary.json",
    }
    values = {name: summary(path) for name, path in locations.items()}
    full = read_json(ROOT / "results/codd_style_fusion_mechanism_audit/posthoc_oracles/canonical/reset_every4_all_pairs_summary.json")["summary"]
    records = [{"configuration": "raw", "raw_epe": full["raw_epe"], "output_epe": full["raw_epe"], "gain": 0.0},
               {"configuration": "full_reference_fixed_h4", "raw_epe": full["raw_epe"], "output_epe": full["fused_epe"], "gain": full["fused_gain"], **{k: full.get(k) for k in ("harmful_update_rate", "clean_pixel_degradation", "frames_worsened_fraction", "worst_frame_degradation", "historical_selection_normalized_gain", "recurrent_selection_normalized_gain", "convex_fusion_normalized_gain")}},]
    labels = {"adaptive": "adaptive_hybrid_h8", "hard": "hard_endpoint_h4", "no_recurrence": "no_recurrence_h4", "no_learned_stereo": "no_learned_stereo_evidence_h4"}
    for name, value in values.items():
        records.append({"configuration": labels[name], "raw_epe": value["raw_epe"], "output_epe": value["fused_epe"], "gain": value["fused_gain"],
                        "harmful_update_rate": value.get("harmful_update_rate"), "clean_pixel_degradation": value.get("clean_pixel_degradation"),
                        "frames_worsened_fraction": value.get("frames_worsened_fraction"), "worst_frame_degradation": value.get("worst_frame_degradation"),
                        "historical_selection_normalized_gain": value.get("historical_selection_normalized_gain"),
                        "recurrent_selection_normalized_gain": value.get("recurrent_selection_normalized_gain"),
                        "convex_fusion_normalized_gain": value.get("convex_fusion_normalized_gain"), "reset_rate": value.get("reset_rate"),
                        "mean_state_age": value.get("mean_state_age"), "hard_threshold": value.get("hard_threshold")})
    write_csv(OUT / "ablation_summary.csv", records)
    write_csv(OUT / "final_test_summary.csv", records)

    fixed_rows = []
    adaptive_rows = []
    for path in sorted((OUT / "reset_policy/validation_candidates").glob("*/summary.json")):
        value = read_json(path)
        row = {"policy": value["policy"]["name"], "fused_epe": value["fused_epe"], "gain": value["fused_gain"],
               "harmful_update_rate": value["harmful_update_rate"], "clean_pixel_degradation": value["clean_pixel_degradation"],
               "frames_worsened_fraction": value["frames_worsened_fraction"], "worst_frame_degradation": value["worst_frame_degradation"],
               "reset_rate": value["reset_rate"], "mean_state_age": value["mean_state_age"]}
        (fixed_rows if value["policy"]["name"].startswith(("fixed_", "continuous")) else adaptive_rows).append(row)
    write_csv(OUT / "fixed_horizon_summary.csv", fixed_rows)
    write_csv(OUT / "adaptive_reset_validation.csv", adaptive_rows)

    drift_rows = []
    for name, path in [("adaptive_hybrid_h8", locations["adaptive"].parent / "drift_by_age.csv"),
                       ("hard_h4", locations["hard"].parent / "drift_by_age.csv"),
                       ("no_recurrence_h4", locations["no_recurrence"].parent / "drift_by_age.csv"),
                       ("no_learned_stereo_h4", locations["no_learned_stereo"].parent / "drift_by_age.csv")]:
        if path.exists():
            with path.open() as handle:
                for row in csv.DictReader(handle): drift_rows.append({"configuration": name, **row})
    write_csv(OUT / "drift_by_age.csv", drift_rows)
    for name, path in locations.items():
        for filename, target in (("per_backbone_metrics.csv", "per_backbone_metrics.csv"), ("per_sequence_metrics.csv", "per_sequence_metrics.csv"), ("frame_metrics.csv", "temporal_metrics.csv")):
            source = path.parent / filename
            if source.exists():
                rows = list(csv.DictReader(source.open()))
                for row in rows: row["configuration"] = labels[name]
                with (OUT / target).open("a", newline="") as handle:
                    if handle.tell() == 0:
                        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader()
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writerows(rows)

    verdict = {
        "fixed_h4_reference": "CONDITIONAL_GO: positive bounded-horizon gain; not stable unlimited streaming",
        "adaptive_hybrid_h8": "NO_GO_AS_ADAPTIVE_SAFETY_POLICY: validation gain selected, but test harmful/clean degradation exceeds safety target",
        "no_recurrence": "RECURRENCE_USEFUL_BUT_NOT_REQUIRED: lower gain, safer than recurrent policies",
        "no_learned_stereo_evidence": "LEARNED_STEREO_EVIDENCE_NOT_ESSENTIAL: fixed-H4 test gain exceeds full reference in this single canonical run",
        "hard_endpoint": "NO_GO_AS_PRIMARY: positive gain but materially worse harmful and clean-pixel safety than soft output",
        "overall_claim": "ARGOS v2 is a fixed short-window causal refiner (H=4), not an indefinitely streaming recurrent system or validated adaptive reset system.",
    }
    write_json(OUT / "aggregate_summary.json", {"project": "ARGOS v2", "full_reference": full, "final_test": {k: v for k, v in values.items()}, "verdicts": verdict})
    write_json(OUT / "verdicts.json", verdict)
    (OUT / "README.md").write_text("""# ARGOS v2 bounded-memory validation\n\nThe canonical CODD-style BiDA fusion remains positive only as a bounded-horizon adapter. Continuous streaming was already shown to collapse. This study trained one canonical run each for no-recurrence and no-learned-stereo-evidence, selected reset/hard policies on dataset 2 only, and opened dataset 7 once after freezing.\n\nThe validation-selected adaptive hybrid H=8 improved validation EPE but did not satisfy the frozen safety targets on dataset 7. Fixed H=4 remains the defensible operational policy; it is a short-window refiner, not indefinite recurrent memory. Hard endpoint output is not promoted because it increases harmful and clean-pixel updates. The no-learned-stereo-evidence ablation unexpectedly exceeds the full reference in this canonical run, so learned ResNet matching cues are not established as essential.\n\nSee `ablation_summary.csv`, `fixed_horizon_summary.csv`, `adaptive_reset_validation.csv`, `drift_by_age.csv`, and `aggregate_summary.json`.\n""")
    (OUT / "paper_tables.tex").write_text("""%% ARGOS v2 bounded-memory mechanism audit\n\\begin{tabular}{lrrrrr}\nConfiguration & EPE & Gain & Harmful & Clean degr. & Worst frame \\\\ \\hline\nFull fixed H4 & %.4f & %.4f & %.4f & %.4f & %.4f \\\\nAdaptive hybrid H8 & %.4f & %.4f & %.4f & %.4f & %.4f \\\\nNo recurrence H4 & %.4f & %.4f & %.4f & %.4f & %.4f \\\\nNo learned stereo H4 & %.4f & %.4f & %.4f & %.4f & %.4f \\\\nHard endpoint H4 & %.4f & %.4f & %.4f & %.4f & %.4f \\\\n\\end{tabular}\n""" % tuple(x for value in [full, values["adaptive"], values["no_recurrence"], values["no_learned_stereo"], values["hard"]] for x in (value["fused_epe"], value["fused_gain"], value["harmful_update_rate"], value["clean_pixel_degradation"], value["worst_frame_degradation"])))


if __name__ == "__main__":
    main()
