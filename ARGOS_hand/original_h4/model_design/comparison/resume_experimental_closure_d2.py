#!/usr/bin/env python3
"""Fail-closed recovery of the interrupted frozen experimental-closure D2 run."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from model_design.comparison import experimental_closure as closure
from model_design.comparison import run_comparison as comparison


D2 = closure.RESULTS / "d2"
ORACLE = "raw_vs_aligned_memory_oracle"
PROTOCOL = "paper_d2_strict_all_anchors"
ORACLE_DIAGNOSTIC = "GT-only raw-vs-aligned-memory endpoint oracle; never adapter inference"
SUMMARY_FIELDS = ("dataset", "split", "backbone", "sequence", "method", "diagnostic_gt_only",
                  "raw_epe_macro_sequence", "candidate_epe_macro_sequence", "gain", "protocol", "primary_aggregate")


def _target(method: str, backbone: str, sequence: str) -> tuple[str, str, str]:
    return method, backbone, sequence


def _targets() -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    normal = [_target(method, backbone, sequence) for method in closure.FULL_D2_METHODS
              for sequence in closure._validation_sequences() for backbone in closure.ALL_BACKBONES]
    oracle = [_target(ORACLE, backbone, sequence) for sequence in closure._validation_sequences() for backbone in closure.ALL_BACKBONES]
    return normal, oracle


def _path(output: Path, target: tuple[str, str, str]) -> Path:
    method, backbone, sequence = target
    return output / "reports" / method / backbone / f"{sequence}.json"


def _read(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid report JSON: {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise RuntimeError(f"report is not a JSON object: {path}")
    return value


def _validate_report(path: Path, target: tuple[str, str, str], *, oracle: bool) -> Mapping[str, Any]:
    method, backbone, sequence = target
    report = _read(path)
    expected = {"dataset": "SCARED-C", "split": "d2", "backbone": backbone, "sequence_ids": [sequence],
                "protocol": PROTOCOL, "primary_aggregate": "macro_sequence"}
    if any(report.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"report identity mismatch: {path}")
    if oracle:
        if "method" in report or report.get("diagnostic") != ORACLE_DIAGNOSTIC:
            raise RuntimeError(f"oracle report mismatch: {path}")
    elif report.get("method") != method:
        raise RuntimeError(f"report method mismatch: {path}")
    try:
        closure._row(report, method, diagnostic=oracle)
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"report metric schema mismatch: {path}: {error}") from error
    return report


def _validate_manifest(output: Path, freeze: Mapping[str, Any]) -> Mapping[str, Any]:
    path = output / "run_manifest.json"
    manifest = _read(path)
    expected_freeze = closure._entry(closure.FREEZE)
    if (manifest.get("project") != "ARGOS v2" or manifest.get("status") != "INCOMPLETE" or
        manifest.get("dataset") != "scared-d2" or manifest.get("freeze") != expected_freeze or
        manifest.get("methods_requested") != list(closure.FULL_D2_METHODS) or
        manifest.get("full_method_grid") != list(closure.FULL_D2_METHODS) or
        manifest.get("scope") != freeze["required_d2_scope"] or
        manifest.get("output") != str(output.resolve()) or
        manifest.get("CUDA_VISIBLE_DEVICES") != "1" or
        manifest.get("device") != "physical cuda:1 remapped to logical cuda:0" or
        manifest.get("dense_predictions_written") is not False):
        raise RuntimeError("D2 manifest is not the exact interrupted frozen full-scope run")
    return manifest


def _validate_summary(path: Path, normal: list[tuple[str, str, str]], oracle: list[tuple[str, str, str]]) -> None:
    try:
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
    except (OSError, csv.Error) as error:
        raise RuntimeError(f"invalid summary CSV: {path}: {error}") from error
    if tuple(reader.fieldnames or ()) != SUMMARY_FIELDS or len(rows) != len(normal) + len(oracle):
        raise RuntimeError(f"summary is not the exact frozen D2 grid: {path}")
    expected = []
    for method in closure.FULL_D2_METHODS:
        expected.extend(("SCARED-C", "d2", backbone, sequence, method, "0") for candidate_method, backbone, sequence in normal if candidate_method == method)
        if method == "canonical_h4":
            expected.extend(("SCARED-C", "d2", backbone, sequence, ORACLE, "1") for _, backbone, sequence in oracle)
    actual = [(row.get("dataset"), row.get("split"), row.get("backbone"), row.get("sequence"), row.get("method"), row.get("diagnostic_gt_only")) for row in rows]
    if actual != expected:
        raise RuntimeError(f"summary identity/order mismatch: {path}")


def audit(output: Path = D2) -> dict[str, Any]:
    """Validate the immutable scope and every reusable report without inference."""
    output = output.resolve()
    freeze = closure.load_freeze()
    manifest = _validate_manifest(output, freeze)
    normal, oracle = _targets()
    allowed = {Path("run_manifest.json"), Path("summary.csv"), *(_path(output, target).relative_to(output) for target in normal + oracle)}
    if not output.is_dir():
        raise RuntimeError(f"missing D2 recovery root: {output}")
    unexpected = sorted(str(path.relative_to(output)) for path in output.rglob("*") if path.is_file() and path.relative_to(output) not in allowed)
    if unexpected:
        raise RuntimeError(f"unexpected D2 recovery files: {unexpected}")
    if (summary := output / "summary.csv").exists():
        _validate_summary(summary, normal, oracle)
    known: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    missing_normal, missing_oracle = [], []
    for target in normal:
        path = _path(output, target)
        if path.exists(): known[target] = _validate_report(path, target, oracle=False)
        else: missing_normal.append(target)
    for target in oracle:
        path = _path(output, target)
        if path.exists(): known[target] = _validate_report(path, target, oracle=True)
        else: missing_oracle.append(target)
    return {"freeze": freeze, "manifest": manifest, "known": known, "missing_normal": missing_normal,
            "missing_oracle": missing_oracle, "reused_normal": len(normal) - len(missing_normal),
            "reused_oracle": len(oracle) - len(missing_oracle)}


def _batches(missing: list[tuple[str, str, str]]) -> list[tuple[str, tuple[str, ...], tuple[str, ...]]]:
    """Coalesce only complete missing Cartesian products, never re-evaluate a saved cell."""
    by_method_sequence: dict[tuple[str, str], set[str]] = defaultdict(set)
    for method, backbone, sequence in missing:
        by_method_sequence[method, sequence].add(backbone)
    grouped: dict[tuple[str, tuple[str, ...]], list[str]] = defaultdict(list)
    backbone_order = {name: index for index, name in enumerate(closure.ALL_BACKBONES)}
    sequence_order = {name: index for index, name in enumerate(closure._validation_sequences())}
    for (method, sequence), backbones in by_method_sequence.items():
        grouped[method, tuple(sorted(backbones, key=backbone_order.__getitem__))].append(sequence)
    return [(method, backbones, tuple(sorted(sequences, key=sequence_order.__getitem__)))
            for (method, backbones), sequences in sorted(grouped.items(), key=lambda item: closure.FULL_D2_METHODS.index(item[0][0]))]


def _summary(output: Path) -> list[dict[str, Any]]:
    rows = []
    for method in closure.FULL_D2_METHODS:
        for sequence in closure._validation_sequences():
            for backbone in closure.ALL_BACKBONES:
                rows.append(closure._row(_validate_report(_path(output, _target(method, backbone, sequence)), _target(method, backbone, sequence), oracle=False), method))
        if method == "canonical_h4":
            for sequence in closure._validation_sequences():
                for backbone in closure.ALL_BACKBONES:
                    target = _target(ORACLE, backbone, sequence)
                    rows.append(closure._row(_validate_report(_path(output, target), target, oracle=True), ORACLE, diagnostic=True))
    return rows


def resume(*, output: Path = D2, device: str = "cuda:0", flow_batch_size: int = 32) -> dict[str, Any]:
    state = audit(output)
    if comparison.validate_cuda(device) != "1":
        raise RuntimeError("D2 recovery requires physical CUDA_VISIBLE_DEVICES=1")
    output = output.resolve(); missing = set(state["missing_normal"]); missing_oracle = set(state["missing_oracle"])
    provenance: dict[str, Any] = {}
    for method, backbones, sequences in _batches(state["missing_normal"]):
        adapter = closure._adapter(method, device); provenance[method] = adapter.describe()
        expected = {_target(method, backbone, sequence) for backbone in backbones for sequence in sequences}
        if not expected <= missing:
            raise RuntimeError(f"recovery batch is not entirely missing: {method}")
        config = SimpleNamespace(dataset="scared-d2", backbones=backbones, sequences=sequences, smoke=False,
                                 device=device, flow_batch_size=flow_batch_size, max_frames=None)
        emitted: set[tuple[str, str, str]] = set()
        def save(bundle: Mapping[str, Any]) -> None:
            target = _target(method, str(bundle["backbone"]), str(bundle["sequence_id"]))
            if target not in expected or target in emitted:
                raise RuntimeError(f"unexpected recovery bundle: {target}")
            if method in closure.POLICIES and np.any(np.asarray(bundle["protocol_mask"], bool) & ~np.asarray(bundle["adapter_support"], bool)):
                raise RuntimeError("official protocol support must be a subset of baseline adapter support")
            report = closure.evaluate_scared_bundle(bundle)
            comparison.atomic_json(_path(output, target), report | {"method": method})
            emitted.add(target)
            oracle_target = _target(ORACLE, target[1], target[2])
            if method == "canonical_h4" and oracle_target in missing_oracle:
                oracle = closure.evaluate_scared_bundle(closure._oracle_bundle(bundle))
                oracle["diagnostic"] = ORACLE_DIAGNOSTIC
                comparison.atomic_json(_path(output, oracle_target), oracle)
                missing_oracle.remove(oracle_target)
        closure._scared(config, adapter, save)
        if emitted != expected:
            raise RuntimeError(f"recovery batch did not emit its exact missing scope: {method}")
        missing -= emitted
    # Reports intentionally omit dense raw/aligned/GT arrays, so an oracle-only gap needs this exact canonical bundle rerun.
    oracle_only = [_target("canonical_h4", target[1], target[2]) for target in missing_oracle]
    for method, backbones, sequences in _batches(oracle_only):
        if method != "canonical_h4":
            raise RuntimeError("only canonical H4 may regenerate the diagnostic oracle")
        adapter = closure._adapter(method, device); provenance[method] = adapter.describe()
        expected = {_target(ORACLE, backbone, sequence) for backbone in backbones for sequence in sequences}
        emitted: set[tuple[str, str, str]] = set()
        config = SimpleNamespace(dataset="scared-d2", backbones=backbones, sequences=sequences, smoke=False,
                                 device=device, flow_batch_size=flow_batch_size, max_frames=None)
        def save_oracle(bundle: Mapping[str, Any]) -> None:
            target = _target(ORACLE, str(bundle["backbone"]), str(bundle["sequence_id"]))
            if target not in expected or target in emitted:
                raise RuntimeError(f"unexpected oracle recovery bundle: {target}")
            report = closure.evaluate_scared_bundle(closure._oracle_bundle(bundle))
            report["diagnostic"] = ORACLE_DIAGNOSTIC
            comparison.atomic_json(_path(output, target), report)
            emitted.add(target)
        closure._scared(config, adapter, save_oracle)
        if emitted != expected:
            raise RuntimeError("oracle recovery batch did not emit its exact missing scope")
        missing_oracle -= emitted
    if missing or missing_oracle:
        raise RuntimeError(f"recovery incomplete: normal={sorted(missing)}, oracle={sorted(missing_oracle)}")
    rows = _summary(output)
    comparison.atomic_csv(output / "summary.csv", rows)
    closure.load_freeze()  # Re-hash every frozen v3 input before publishing.
    output_hashes = closure._output_hashes(output)
    manifest = state["manifest"] | {"status": "COMPLETE", "method_report_count": len(closure.FULL_D2_METHODS) * len(closure.ALL_BACKBONES) * len(closure._validation_sequences()),
                                      "summary_row_count": len(rows), "output_hashes": output_hashes, "outputs": sorted(output_hashes),
                                      "recovery": {"launcher": str(Path(__file__).resolve()), "launcher_sha256": comparison.sha256(Path(__file__)),
                                                   "reused_normal_reports": state["reused_normal"], "reused_oracle_reports": state["reused_oracle"],
                                                   "executed_normal_reports": len(state["missing_normal"]), "generated_oracle_reports": len(state["missing_oracle"]),
                                                   "rerun_canonical_for_oracle_reports": len(oracle_only),
                                                   "freeze_before": state["manifest"]["freeze"], "freeze_after": closure._entry(closure.FREEZE),
                                                   "module_provenance": provenance}}
    comparison.atomic_json(output / "run_manifest.json", manifest)
    return {"reused": state["reused_normal"] + state["reused_oracle"], "executed": len(state["missing_normal"]), "summary_rows": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--flow-batch-size", type=int, default=32)
    config = parser.parse_args()
    state = audit()
    if config.dry_run:
        print(json.dumps({"status": "AUDITED", "reused_reports": state["reused_normal"] + state["reused_oracle"],
                          "missing_normal_reports": ["/".join(target) for target in state["missing_normal"]],
                          "missing_oracle_reports": ["/".join(target) for target in state["missing_oracle"]]}, indent=2))
        return
    print(json.dumps(resume(device=config.device, flow_batch_size=config.flow_batch_size), sort_keys=True))


if __name__ == "__main__":
    main()
