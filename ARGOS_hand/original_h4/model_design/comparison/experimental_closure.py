#!/usr/bin/env python3
"""Frozen D2 closure: trivial blends, canonical H4, and a GT-only endpoint oracle."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from model_design.comparison.canonical_horizons import CanonicalHorizon
from model_design.comparison.definitive_evaluation import evaluate_scared_bundle
from model_design.comparison.experimental_policies import POLICIES, ExperimentalPolicy
from model_design.comparison.run_comparison import ALL_BACKBONES, _scared, atomic_csv, atomic_json, prepare_output, sha256, validate_cuda


RESULTS = ROOT.parent / "results/definitive_evaluation/experimental_closure"
HISTORICAL_FREEZE = RESULTS / "protocol/freeze_manifest.json"
HISTORICAL_FREEZE_V2 = RESULTS / "protocol/freeze_manifest_v2.json"
FREEZE = RESULTS / "protocol/freeze_manifest_v3.json"
PROMOTION = RESULTS / "protocol/promotion_freeze_manifest_v3.json"
D2_DECISION = RESULTS / "protocol/d2_decision.json"
CHECKPOINT_SHA256 = "99c5745c164fd4903b8aa8acf8f57efccacd9cdcd0d4ed4305cd10609324d725"
CANONICAL_HORIZONS = {"canonical_h1": 1, "canonical_h2": 2, "canonical_h4": 4, "canonical_h6": 6,
                      "canonical_h8": 8, "canonical_continuous": None}
FULL_D2_METHODS = tuple((*CANONICAL_HORIZONS, *POLICIES))
D7_CONFIRMABLE = frozenset(name for name, horizon in CANONICAL_HORIZONS.items() if horizon == 4) | frozenset(
    name for name, spec in POLICIES.items() if spec.horizon == 4)


def endpoint_oracle(raw: np.ndarray, aligned_memory: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Posthoc diagnostic only: GT selects the better raw/aligned endpoint on official support."""
    choose_memory = mask & (np.abs(aligned_memory - gt) < np.abs(raw - gt))
    return np.where(choose_memory, aligned_memory, raw)


def _validation_sequences() -> tuple[str, ...]:
    source = ROOT.parents[1] / "ARGOS_FREEZED/experiments/02_massive_training/scripts"
    if str(source) not in sys.path: sys.path.insert(0, str(source))
    from campaign_common import VALIDATION_SEQUENCES
    return tuple(VALIDATION_SEQUENCES)


def _expected_report_count() -> int:
    return len(FULL_D2_METHODS) * len(ALL_BACKBONES) * len(_validation_sequences())


def _entry(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"immutable input unavailable: {path}")
    return {"path": str(path), "sha256": sha256(path)}


def _immutable_inputs() -> dict[str, Any]:
    manifest_path = ROOT / "model_design/checkpoints/inference_manifest.json"
    source = json.loads(manifest_path.read_text())
    entries = {"checkpoint": _entry(ROOT / "model_design/checkpoints/codd_style_h4_best_validation.pt"),
               "policy": _entry(ROOT / "model_design/checkpoints/codd_style_h4_policy.json"),
               "canonical_h4": _entry(Path(__file__).with_name("canonical_h4.py")),
               "canonical_horizons": _entry(Path(__file__).with_name("canonical_horizons.py")),
               "experimental_policies": _entry(Path(__file__).with_name("experimental_policies.py")),
               "experimental_closure": _entry(Path(__file__)), "run_comparison": _entry(Path(__file__).with_name("run_comparison.py")),
               "definitive_evaluation": _entry(Path(__file__).with_name("definitive_evaluation.py")),
               "unified_metrics": _entry(ROOT / "model_design/metrics/unified_metrics.py"), "inference_manifest": _entry(manifest_path),
               "sea_raft_checkpoint": _entry(Path("/dtu/p1/leopam/ARGOS/external/bidavideo/third_party/SEA-RAFT/models/Tartan-C-T-TSKH-spring540x960-S.pth"))}
    for name, value in source["artifacts"].items():
        path = Path(value["path"]); path = path if path.is_absolute() else ROOT / path
        entries[f"inference_{name}"] = _entry(path)
    entries["git_commit"] = subprocess.run(["git", "-C", str(ROOT.parent), "rev-parse", "HEAD"], text=True, capture_output=True, check=False).stdout.strip() or None
    return entries


def _freeze() -> dict[str, Any]:
    revision = subprocess.run(["git", "-C", str(ROOT.parent), "rev-parse", "HEAD"], text=True, capture_output=True, check=False).stdout.strip() or None
    return {"project": "ARGOS v2", "freeze_version": 3, "status": "FROZEN_D2_DEVELOPMENT", "device": "physical cuda:1 via CUDA_VISIBLE_DEVICES=1; logical --device cuda:0",
            "checkpoint_sha256": CHECKPOINT_SHA256, "checkpoint": str(ROOT / "model_design/checkpoints/codd_style_h4_best_validation.pt"),
            "development": {"allowed": "SCARED-C D2 only", "forbidden_until_promotion": ["SCARED-C D7", "DRENDS tuning", "training", "threshold calibration"]},
            "datasets": {"D2": "development", "D7": "confirmation after separate promotion freeze", "DRENDS": "external analysis only; no tuning"},
            "protocol": {"D2": "paper_d2_strict_all_anchors", "D7": "h4_only_common_support (one D2-selected H=4 method only)",
                         "aggregation": "macro_sequence primary; micro_pixel secondary", "future_access": False,
                         "official_mask": "prediction-independent; adapter support diagnostic only", "baseline_fallback": "raw outside adapter support; official mask is asserted a subset of that support", "dense_predictions_written": False},
            "methods": {**{name: {"horizon": horizon, "checkpoint": CHECKPOINT_SHA256} for name, horizon in CANONICAL_HORIZONS.items()},
                        **{name: {"horizon": spec.horizon, "weight": spec.weight, "memory": spec.memory} for name, spec in POLICIES.items()},
                        "raw_vs_aligned_memory_oracle": {"diagnostic": True, "inference": "GT forbidden; posthoc only"}},
            "horizon_sensitivity": [1, 2, 4, 6, 8, None],
            "decision_criteria": {"selection": "D2 only", "D7": "one decision-pinned H=4 confirmation after promotion freeze", "no_backbone_or_dataset_specific_tuning": True},
            "full_d2_method_grid": list(FULL_D2_METHODS),
            "required_d2_scope": {"smoke": False, "backbones": list(ALL_BACKBONES), "sequences": list(_validation_sequences()), "max_frames": None,
                                  "protocol": "paper_d2_strict_all_anchors", "method_report_count": _expected_report_count(),
                                  "summary_row_count": _expected_report_count() + len(ALL_BACKBONES) * len(_validation_sequences())},
            "immutable_inputs": _immutable_inputs(), "git_commit": revision}


def _verify_entries(value: Mapping[str, Any]) -> None:
    for name, entry in value["immutable_inputs"].items():
        if name == "git_commit": continue
        if not isinstance(entry, Mapping) or sha256(Path(entry["path"])) != entry["sha256"]:
            raise RuntimeError(f"freeze hash mismatch: {name}")


def load_freeze(path: Path | None = None) -> dict[str, Any]:
    path = FREEZE if path is None else path
    if not path.is_file(): raise RuntimeError(f"missing immutable freeze: {path}")
    value = json.loads(path.read_text())
    if value.get("project") != "ARGOS v2" or value.get("freeze_version") != 3:
        raise RuntimeError("closure requires freeze manifest v3")
    _verify_entries(value)
    if value.get("checkpoint_sha256") != CHECKPOINT_SHA256 or value.get("full_d2_method_grid") != list(FULL_D2_METHODS) or value.get("required_d2_scope") != _freeze()["required_d2_scope"]:
        raise RuntimeError("freeze method grid or canonical checkpoint mismatch")
    return value


def _load_decision(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    selected = value.get("selected_confirmation_method")
    if selected not in D7_CONFIRMABLE:
        raise ValueError(f"D7 selection must be one frozen H=4 method, got {selected!r}")
    return value


def write_freeze(*, promotion: bool = False, decision: Path = D2_DECISION) -> Path:
    path = PROMOTION if promotion else FREEZE
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable freeze: {path}")
    if not promotion:
        value = _freeze()
    else:
        freeze, d2_manifest = load_freeze(), RESULTS / "d2/run_manifest.json"
        if not d2_manifest.is_file(): raise RuntimeError("promotion requires a COMPLETE D2 closure run")
        d2 = json.loads(d2_manifest.read_text())
        expected = freeze["required_d2_scope"]
        if (d2.get("status") != "COMPLETE" or d2.get("methods_requested") != list(FULL_D2_METHODS) or d2.get("scope") != expected or
            d2.get("method_report_count") != expected["method_report_count"] or d2.get("summary_row_count") != expected["summary_row_count"]):
            raise RuntimeError("promotion requires the exact COMPLETE D2 full method grid")
        selected = _load_decision(decision)
        if selected.get("d2_manifest_sha256") != sha256(d2_manifest):
            raise RuntimeError("D2 decision is not pinned to the completed D2 manifest")
        value = freeze | {"status": "FROZEN_D7_CONFIRMATION", "requires": str(FREEZE), "d2_manifest": _entry(d2_manifest),
                          "d2_decision": _entry(decision), "selected_confirmation_method": selected["selected_confirmation_method"],
                          "allowed": "SCARED-C D7 selected method only"}
    atomic_json(path, value)
    return path


def _row(report: Mapping[str, Any], method: str, *, diagnostic: bool = False) -> dict[str, Any]:
    aggregate = report["aggregate"]["disparity_px"]
    value = lambda which: aggregate[which]["MAE"]["macro_sequence"]
    return {"dataset": report["dataset"], "split": report["split"], "backbone": report["backbone"], "sequence": report["sequence_ids"][0],
            "method": method, "diagnostic_gt_only": int(diagnostic), "raw_epe_macro_sequence": value("raw"),
            "candidate_epe_macro_sequence": value("refined"), "gain": value("raw") - value("refined"),
            "protocol": report["protocol"], "primary_aggregate": report["primary_aggregate"]}


def _oracle_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    return dict(bundle) | {"refined_disparity": endpoint_oracle(np.asarray(bundle["raw_disparity"]), np.asarray(bundle["aligned_memory"]),
                                                          np.asarray(bundle["gt_disparity"]),
                                                          np.asarray(bundle["gt_valid"], bool) & np.asarray(bundle["protocol_mask"], bool))}


def _adapter(name: str, device: str):
    return CanonicalHorizon(horizon=CANONICAL_HORIZONS[name], device=device) if name in CANONICAL_HORIZONS else ExperimentalPolicy(POLICIES[name], device=device)


def _output_hashes(output: Path) -> dict[str, str]:
    return {str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob("*")) if path.is_file() and path.name != "run_manifest.json"}


def _scope(config: argparse.Namespace) -> dict[str, Any]:
    return {"smoke": bool(config.smoke), "backbones": list(config.backbones), "sequences": list(config.sequences or _validation_sequences()),
            "max_frames": config.max_frames, "protocol": "paper_d2_strict_all_anchors"}


def run(config: argparse.Namespace) -> None:
    freeze = load_freeze()
    selected = None
    output = config.output or RESULTS / ("d2_smoke" if config.smoke and config.dataset == "scared-d2" else "d2" if config.dataset == "scared-d2" else "d7_confirmation")
    d2_root = (RESULTS / "d2").resolve()
    if config.output is not None and output.resolve() == d2_root:
        raise ValueError("smoke output is forced to RESULTS/d2_smoke" if config.smoke else "custom output may not collide with the canonical D2 result root")
    if config.dataset == "scared-d2":
        if not config.smoke and (tuple(config.methods) != FULL_D2_METHODS or tuple(config.backbones) != ALL_BACKBONES or config.sequences is not None or config.max_frames is not None):
            raise ValueError("complete D2 closure forbids smoke, custom scope, frame limits, and partial methods")
        if config.smoke and output.resolve() != (RESULTS / "d2_smoke").resolve():
            raise ValueError("smoke output is forced to RESULTS/d2_smoke")
    if config.dataset == "scared-d7":
        if not PROMOTION.is_file(): raise RuntimeError("D7 is locked until a promotion freeze exists")
        promotion = json.loads(PROMOTION.read_text()); _verify_entries(promotion)
        if promotion.get("requires") != str(FREEZE) or sha256(Path(promotion["d2_manifest"]["path"])) != promotion["d2_manifest"]["sha256"] or sha256(Path(promotion["d2_decision"]["path"])) != promotion["d2_decision"]["sha256"]:
            raise RuntimeError("promotion D2 manifest or decision hash mismatch")
        selected = promotion.get("selected_confirmation_method")
        if config.methods != [selected]: raise ValueError("D7 confirmation permits only the decision-pinned selected method")
    gpu = validate_cuda(config.device)
    prepare_output(output)
    manifest = {"project": "ARGOS v2", "status": "INCOMPLETE", "dataset": config.dataset, "CUDA_VISIBLE_DEVICES": gpu,
                "device": "physical cuda:1 remapped to logical cuda:0", "freeze": _entry(FREEZE), "methods_requested": config.methods,
                "full_method_grid": list(FULL_D2_METHODS), "selected_confirmation_method": selected, "output": str(output),
                "scope": freeze["required_d2_scope"] if config.dataset == "scared-d2" and not config.smoke else _scope(config),
                "dense_predictions_written": False, "module_provenance": {}}
    atomic_json(output / "run_manifest.json", manifest)
    rows: list[dict[str, Any]] = []
    try:
        for name in config.methods:
            adapter = _adapter(name, config.device); manifest["module_provenance"][name] = adapter.describe()
            reports: list[tuple[dict[str, Any], Mapping[str, Any]]] = []
            def save(bundle: Mapping[str, Any]) -> None:
                if name in POLICIES and np.any(np.asarray(bundle["protocol_mask"], bool) & ~np.asarray(bundle["adapter_support"], bool)):
                    raise RuntimeError("official protocol support must be a subset of baseline adapter support")
                report = evaluate_scared_bundle(bundle); reports.append((report, bundle))
                atomic_json(output / "reports" / name / bundle["backbone"] / f"{bundle['sequence_id']}.json", report | {"method": name})
            _scared(config, adapter, save)
            rows.extend(_row(report, name) for report, _ in reports)
            if name == "canonical_h4":
                for _report, bundle in reports:
                    report = evaluate_scared_bundle(_oracle_bundle(bundle))
                    report["diagnostic"] = "GT-only raw-vs-aligned-memory endpoint oracle; never adapter inference"
                    atomic_json(output / "reports" / "raw_vs_aligned_memory_oracle" / bundle["backbone"] / f"{bundle['sequence_id']}.json", report)
                    rows.append(_row(report, "raw_vs_aligned_memory_oracle", diagnostic=True))
        atomic_csv(output / "summary.csv", rows)
        load_freeze()  # TOCTOU guard before publishing a COMPLETE result.
        method_reports = sum(1 for name in _output_hashes(output) if name.startswith("reports/") and "/raw_vs_aligned_memory_oracle/" not in name)
        atomic_json(output / "run_manifest.json", manifest | {"status": "COMPLETE", "method_report_count": method_reports, "summary_row_count": len(rows), "output_hashes": _output_hashes(output),
                                                                 "outputs": sorted(_output_hashes(output))})
    except BaseException as error:
        atomic_json(output / "run_manifest.json", manifest | {"error": f"{type(error).__name__}: {error}"})
        raise


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-freeze", action="store_true"); parser.add_argument("--write-promotion-freeze", action="store_true")
    parser.add_argument("--decision", type=Path, default=D2_DECISION)
    parser.add_argument("--dataset", choices=("scared-d2", "scared-d7")); parser.add_argument("--methods", nargs="+", default=list(FULL_D2_METHODS))
    parser.add_argument("--backbones", nargs="+", default=ALL_BACKBONES); parser.add_argument("--sequences", nargs="+")
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--flow-batch-size", type=int, default=32); parser.add_argument("--max-frames", type=int); parser.add_argument("--output", type=Path); parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    config = arguments()
    if config.write_freeze or config.write_promotion_freeze:
        if config.dataset:
            raise ValueError("freeze creation does not evaluate a dataset")
        print(write_freeze(promotion=config.write_promotion_freeze, decision=config.decision)); return
    if not config.dataset:
        raise ValueError("--dataset is required for evaluation")
    unknown = set(config.methods) - (set(CANONICAL_HORIZONS) | set(POLICIES))
    if unknown: raise ValueError(f"unknown methods: {sorted(unknown)}")
    run(config)


if __name__ == "__main__":
    main()
