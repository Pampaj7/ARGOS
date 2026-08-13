#!/usr/bin/env python3
"""Run or compile the frozen definitive evaluator into complete long-form CSVs.

The evaluator remains immutable: this wrapper only launches it or reads its
COMPLETE outputs.  It never writes dense predictions and never pools datasets.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model_design.comparison.build_paper_table import flatten_scared_reports
from model_design.comparison.run_comparison import ALL_BACKBONES, D4D_BACKBONES, DEFAULT_MODULE


DATASETS = ("scared-d2", "scared-d7", "d4d", "servct", "drends")
RESULTS = ROOT.parent / "results/definitive_temporal_evaluation_csv"
FIELDS = (
    "dataset", "split", "backbone", "sequence", "frame", "protocol", "record_level", "section", "metric_path",
    "method", "horizon", "metric_scope", "applicability", "diagnostic_tags", "statistic", "value", "source_id", "source_run",
    "source_manifest_sha256", "source_report_sha256", "specimen", "session", "window_index", "step_since_reset", "reset",
)

SEEN_BACKBONES = frozenset(("S2M2-S", "RAFT-Stereo", "StereoAnywhere"))
UNSEEN_BACKBONES = frozenset(("CREStereo", "Fast-FoundationStereo"))
WIDE_METADATA = (
    "dataset", "split", "temporal_model", "temporal_model_hash", "temporal_model_checkpoint_hash", "backbone",
    "backbone_status", "dataset_role", "domain_status", "dataset_domain_seen_in_training",
    "evaluation_split_used_in_training", "temporal_applicability", "metric_scope", "protocol",
)
DATASET_METADATA = {
    ("SCARED-C", "d2"): ("D2_validation", "in_domain", True, "APPLICABLE"),
    ("SCARED-C", "d7"): ("D7_heldout_test", "in_domain", True, "APPLICABLE"),
    ("D4D", ""): ("OOD_no_reference", "OOD", False, "NO_REFERENCE_DIAGNOSTIC"),
    ("SERV-CT", ""): ("static_OOD", "OOD", False, "NOT_APPLICABLE"),
    ("DRENDS", "pilot"): ("OOD_metric", "OOD", False, "APPLICABLE_WITH_CAVEAT"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, prefix=f".{path.name}.") as stream:
            temporary = Path(stream.name)
            stream.write(text)
        os.replace(temporary, path)
    except BaseException:
        if temporary:
            temporary.unlink(missing_ok=True)
        raise


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path}")
    import io
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="raise")
    writer.writeheader(); writer.writerows(rows)
    atomic_text(path, stream.getvalue())


def finalize_output_hashes(root: Path) -> None:
    """Hash the durable published bytes after any staging-path rewrite."""
    manifest_path = root / "run_manifest.json"
    manifest = read_json(manifest_path)
    outputs = sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file() and path != manifest_path)
    atomic_json(manifest_path, manifest | {"status": "COMPLETE", "outputs": outputs,
                                           "output_hashes": {name: sha256(root / name) for name in outputs}})


class CsvSink:
    """Streaming CSV writer; the enclosing staged directory is the atomic unit."""
    def __init__(self, path: Path) -> None:
        self.path, self.count = path, 0
        self.stream = path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.stream, fieldnames=FIELDS, extrasaction="raise")
        self.writer.writeheader()

    def write(self, rows: Iterable[dict[str, Any]]) -> None:
        for row in rows:
            self.writer.writerow(row); self.count += 1

    def close(self) -> None:
        self.stream.close()
        if not self.count:
            self.path.unlink(missing_ok=True)


def staged_directory(output: Path, writer: Any) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / f".{output.name}.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise FileExistsError(f"output publication already in progress: {output}") from error
    os.close(descriptor)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        writer(stage)
        if output.exists():
            raise FileExistsError(f"output appeared during build: {output}")
        os.rename(stage, output)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        lock.unlink(missing_ok=True)


def _declared_path(value: str, run: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else run / path).resolve(strict=True)


def _verify_declaration(manifest: Mapping[str, Any], run: Path, label: str) -> None:
    declaration = manifest.get(label)
    if not isinstance(declaration, Mapping) or not isinstance(declaration.get("path"), str) or not isinstance(declaration.get("sha256"), str):
        raise RuntimeError(f"missing hash declaration {label}: {run}")
    path = _declared_path(declaration["path"], run)
    if sha256(path) != declaration["sha256"]:
        raise RuntimeError(f"hash mismatch for {label}: {path}")


def verify_run(run: Path, dataset: str) -> dict[str, Any]:
    manifest_path = run / "run_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("project") != "ARGOS v2" or manifest.get("status") != "COMPLETE" or manifest.get("dataset") != dataset:
        raise RuntimeError(f"not a complete ARGOS v2 {dataset} run: {run}")
    for label in ("unified_metrics", "evaluator", "comparison_driver"):
        _verify_declaration(manifest, run, label)
    module = manifest.get("module_provenance")
    if not isinstance(module, Mapping) or not module:
        raise RuntimeError(f"missing module provenance: {run}")
    for stem in ("checkpoint", "policy", "code"):
        digest = module.get(f"{stem}_sha256")
        if digest is not None:
            path = module.get(stem)
            if not isinstance(path, str) or not isinstance(digest, str) or sha256(_declared_path(path, run)) != digest:
                raise RuntimeError(f"invalid module {stem} provenance: {run}")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or not outputs or any(not isinstance(item, str) or not (run / item).is_file() for item in outputs):
        raise RuntimeError(f"incomplete declared outputs: {run}")
    return manifest


def locate_run(source_root: Path, dataset: str) -> tuple[Path, dict[str, Any]]:
    parent = source_root / dataset
    candidates = sorted(path.parent for path in parent.glob("*/run_manifest.json"))
    complete = [(path, verify_run(path, dataset)) for path in candidates]
    if len(complete) != 1:
        raise RuntimeError(f"expected exactly one complete {dataset} run below {parent}, found {len(complete)}")
    return complete[0]


def _scalar(value: Any) -> float | bool:
    if isinstance(value, bool):
        return value
    value = float(value)
    if not math.isfinite(value):
        raise RuntimeError("non-finite metric value")
    return value


def _source_id(dataset: str, run: Path, manifest: Mapping[str, Any]) -> str:
    module = manifest["module_provenance"]
    name = str(module.get("module", run.name)).replace("/", "_")
    digest = str(module.get("code_sha256", sha256(run / "run_manifest.json")))[:12]
    return f"definitive/{dataset}/{name}@{digest}"


def _base_row(dataset: str, split: str, manifest: Mapping[str, Any], run: Path, report: Path | None = None, **changes: Any) -> dict[str, Any]:
    row = {key: "" for key in FIELDS}
    row.update({"dataset": dataset, "split": split, "metric_scope": "gt_referenced" if dataset in {"SCARED-C", "DRENDS"} else "no_reference_prediction_space",
                "applicability": "APPLICABLE", "diagnostic_tags": "{}", "source_id": _source_id(dataset, run, manifest),
                "source_run": str(run), "source_manifest_sha256": sha256(run / "run_manifest.json"),
                "source_report_sha256": sha256(report) if report else ""})
    row.update(changes)
    return row


def _method(path: Iterable[str]) -> str:
    return next((item for item in path if item in {"raw", "refined"}), "candidate_only")


def scalar_metric_rows(metrics: Mapping[str, Any], base: dict[str, Any]) -> list[dict[str, Any]]:
    """Lossless scalar serializer for the evaluator's metric report schema."""
    rows: list[dict[str, Any]] = []
    def visit(value: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                visit(item, path + (str(key),))
        elif isinstance(value, (float, int)) and not isinstance(value, bool):
            rows.append(base | {"metric_path": "/".join(path[:-1]), "method": _method(path), "statistic": path[-1], "value": _scalar(value)})
        elif isinstance(value, bool):
            rows.append(base | {"metric_path": "/".join(path[:-1]), "method": _method(path), "statistic": path[-1], "value": value})
    visit(metrics)
    return rows


def compile_scared(run: Path, manifest: Mapping[str, Any], dataset: str, aggregate_sink: CsvSink, sequence_sink: CsvSink, frame_sink: CsvSink) -> None:
    split = dataset.removeprefix("scared-")
    reports: list[dict[str, Any]] = []
    paths = sorted((run / "reports").glob("*/*.json"))
    if not paths:
        raise RuntimeError(f"missing SCARED reports: {run}")
    for path in paths:
        report = read_json(path)
        if report.get("dataset") != "SCARED-C" or report.get("split") != split:
            raise RuntimeError(f"invalid SCARED report: {path}")
        reports.append(report)
    # Keep the validated paper flattening as a coverage check, but serialize all
    # scalar leaf fields rather than silently discarding aggregate metadata.
    expected_primary = {(row["backbone"], row["sequence"], row["metric_path"], "macro_sequence" if row["macro_sequence"] is not None else "value")
                        for row in flatten_scared_reports(reports, split)}
    aggregate = []
    found_primary: set[tuple[str, str, str, str]] = set()
    for report, path in zip(reports, paths):
        common = {"backbone": report["backbone"], "sequence": report["sequence_ids"][0], "protocol": report["protocol"]}
        for section in ("aggregate", "spatial", "safety", "temporal"):
            rows = scalar_metric_rows({section: report.get(section, {})}, _base_row("SCARED-C", split, manifest, run, path, record_level="aggregate", section=section, **common))
            aggregate += rows
            found_primary |= {(report["backbone"], report["sequence_ids"][0], row["metric_path"], row["statistic"]) for row in rows}
    if not expected_primary <= found_primary:
        raise RuntimeError("lossless aggregate export diverges from validated paper flattening")
    if not aggregate:
        raise RuntimeError(f"incomplete SCARED aggregate CSV export: {run}")
    aggregate_sink.write(aggregate)
    sequence_count = frame_count = 0
    for report, path in zip(reports, paths):
        common = {"backbone": report["backbone"], "sequence": report["sequence_ids"][0], "protocol": report["protocol"]}
        for item in report.get("per_sequence", []):
            rows = scalar_metric_rows(item.get("metrics", {}), _base_row("SCARED-C", split, manifest, run, path, record_level="per_sequence", **common))
            sequence_sink.write(rows); sequence_count += len(rows)
        for item in report.get("per_frame", []):
            rows = scalar_metric_rows(item.get("metrics", {}), _base_row("SCARED-C", split, manifest, run, path, record_level="per_frame", frame=item.get("frame_id", ""), **common))
            frame_sink.write(rows); frame_count += len(rows)
    if not sequence_count or not frame_count:
        raise RuntimeError(f"incomplete SCARED CSV export: {run}")


def compile_drends(run: Path, manifest: Mapping[str, Any], aggregate_sink: CsvSink, sequence_sink: CsvSink, frame_sink: CsvSink) -> None:
    """DRENDS shares the unified report schema but is an OOD ToF-reference pilot."""
    paths = sorted((run / "reports").glob("*/*.json"))
    if not paths:
        raise RuntimeError(f"missing DRENDS reports: {run}")
    aggregate: list[dict[str, Any]] = []
    per_sequence = per_frame = 0
    for path in paths:
        report = read_json(path)
        if report.get("dataset") != "DRENDS" or report.get("split") != "pilot" or not isinstance(report.get("backbone"), str):
            raise RuntimeError(f"invalid DRENDS report: {path}")
        common = {"backbone": report["backbone"], "sequence": report["sequence_ids"][0], "protocol": report["protocol"]}
        for section in ("aggregate", "spatial", "safety", "temporal"):
            aggregate += scalar_metric_rows({section: report.get(section, {})}, _base_row("DRENDS", "pilot", manifest, run, path,
                                             record_level="aggregate", section=section, metric_scope="tof_reference_nonindependent",
                                             applicability="APPLICABLE_WITH_CAVEAT", **common))
        for item in report.get("per_sequence", []):
            values = scalar_metric_rows(item.get("metrics", {}), _base_row("DRENDS", "pilot", manifest, run, path,
                                        record_level="per_sequence", metric_scope="tof_reference_nonindependent",
                                        applicability="APPLICABLE_WITH_CAVEAT", **common))
            sequence_sink.write(values); per_sequence += len(values)
        for item in report.get("per_frame", []):
            values = scalar_metric_rows(item.get("metrics", {}), _base_row("DRENDS", "pilot", manifest, run, path,
                                        record_level="per_frame", frame=item.get("frame_id", ""), metric_scope="tof_reference_nonindependent",
                                        applicability="APPLICABLE_WITH_CAVEAT", **common))
            frame_sink.write(values); per_frame += len(values)
    if not aggregate or not per_sequence or not per_frame:
        raise RuntimeError(f"incomplete DRENDS CSV export: {run}")
    aggregate_sink.write(aggregate)


def read_d4d(run: Path) -> tuple[Path, list[dict[str, str]]]:
    path = run / "d4d_diagnostics.csv"
    if not path.is_file():
        raise RuntimeError(f"missing D4D diagnostics: {run}")
    with path.open(newline="", encoding="utf-8") as stream:
        source_rows = list(csv.DictReader(stream))
    required = {"dataset", "metric_scope", "backbone", "specimen", "session", "sequence", "window_index", "frame_id", "step_since_reset", "reset", "support_coverage", "raw_mc_inconsistency", "temporal_module_mc_inconsistency", "update_magnitude"}
    if not source_rows or set(source_rows[0]) != required or any(row.get("dataset") != "D4D" or row.get("metric_scope") != "no_reference_prediction_space" for row in source_rows):
        raise RuntimeError(f"invalid D4D no-reference diagnostics: {path}")
    return path, source_rows


def compile_d4d(run: Path, manifest: Mapping[str, Any], path: Path, source_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    definitions = (("support_coverage", "support", "support_coverage"), ("raw_mc_inconsistency", "raw", "motion_compensated_inconsistency"),
                   ("temporal_module_mc_inconsistency", "refined", "motion_compensated_inconsistency"), ("update_magnitude", "refined", "update_magnitude"))
    for item in source_rows:
        base = _base_row("D4D", "", manifest, run, path, backbone=item["backbone"], sequence=item["sequence"], frame=item["frame_id"],
                         protocol="four-frame curated causal windows, past-to-present", record_level="per_frame", section="d4d", applicability="NO_REFERENCE_DIAGNOSTIC")
        for key, method, metric in definitions:
            rows.append(base | {"metric_path": metric, "method": method, "statistic": "value", "value": _scalar(item[key]),
                                "specimen": item["specimen"], "session": item["session"], "window_index": item["window_index"], "step_since_reset": item["step_since_reset"], "reset": item["reset"]})
    return rows


def d4d_summary_rows(run: Path, manifest: Mapping[str, Any], path: Path, source_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Equal-frame no-reference summaries; never geometry or pooled score claims."""
    definitions = (("support_coverage", "support", "support_coverage"), ("raw_mc_inconsistency", "raw", "motion_compensated_inconsistency"),
                   ("temporal_module_mc_inconsistency", "refined", "motion_compensated_inconsistency"), ("update_magnitude", "refined", "update_magnitude"))
    groups = (("sequence", ("backbone", "specimen", "session", "sequence")), ("session", ("backbone", "specimen", "session")),
              ("specimen", ("backbone", "specimen")), ("backbone", ("backbone",)), ("dataset_equal_frame_diagnostic", ()))
    rows: list[dict[str, Any]] = []
    for level, keys in groups:
        partitions: dict[tuple[str, ...], list[dict[str, str]]] = {}
        for item in source_rows:
            key = tuple(item[name] for name in keys) if keys else ("ALL_BACKBONES_EQUAL_FRAME_DIAGNOSTIC",)
            partitions.setdefault(key, []).append(item)
        for group, items in sorted(partitions.items()):
            values = {name: group[index] for index, name in enumerate(keys)} if keys else {"backbone": group[0]}
            for key, method, metric in definitions:
                mean = sum(_scalar(item[key]) for item in items) / len(items)
                base = _base_row("D4D", "", manifest, run, path, backbone=values.get("backbone", ""), sequence=values.get("sequence", ""),
                    protocol="four-frame curated causal windows, past-to-present", record_level=level, section="d4d_no_reference_summary",
                    applicability="NO_REFERENCE_DIAGNOSTIC", specimen=values.get("specimen", ""), session=values.get("session", ""))
                rows += [base | {"metric_path": metric, "method": method, "statistic": "equal_frame_mean", "value": mean},
                         base | {"metric_path": metric, "method": method, "statistic": "frame_count", "value": len(items)}]
    return rows


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _wide_name(*values: str) -> str:
    """Stable collision-free CSV name for one scalar report leaf."""
    # Hex-escape every non-alphanumeric character (including underscore), so
    # distinct metric paths can never collapse to the same CSV header.
    parts = ["".join(char if char.isascii() and char.isalnum() else f"x{ord(char):x}x" for char in value) or "root" for value in values]
    return "__".join(parts)


def _backbone_status(backbone: str) -> str:
    if backbone in SEEN_BACKBONES:
        return "training_seen"
    if backbone in UNSEEN_BACKBONES:
        return "training_unseen"
    return "unsupported_or_unknown"


def _wide_metadata(dataset: str, split: str, backbone: str, provenance: Mapping[str, Any], *, protocol: str, scope: str, applicability: str | None=None) -> dict[str, Any]:
    try:
        role, domain, seen_domain, default_applicability = DATASET_METADATA[(dataset, split)]
    except KeyError as error:
        raise RuntimeError(f"missing definitive metadata for {dataset}/{split}") from error
    declared_roles = provenance.get("evaluation_split_roles", {})
    if not isinstance(declared_roles, Mapping):
        raise RuntimeError("adapter evaluation_split_roles must be a mapping")
    role = str(declared_roles.get(f"{dataset}/{split}", role))
    return {
        "dataset": dataset, "split": split, "temporal_model": str(provenance.get("module", "unknown")),
        "temporal_model_hash": str(provenance.get("code_sha256", "")),
        "temporal_model_checkpoint_hash": str(provenance.get("checkpoint_sha256", "")),
        "backbone": backbone, "backbone_status": _backbone_status(backbone), "dataset_role": role,
        "domain_status": domain, "dataset_domain_seen_in_training": bool(seen_domain),
        "evaluation_split_used_in_training": role == "mixed_train_validation_exposed", "temporal_applicability": applicability or default_applicability,
        "metric_scope": scope, "protocol": protocol,
    }


def _as_number(value: str | float | bool | int) -> float | bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return float(value)


def _reduce_scalars(rows: list[dict[str, str]]) -> dict[str, float | bool]:
    """Collapse one report leaf over SCARED sequences without pooling datasets.

    The report already contains sequence-primary aggregate values.  Counts are
    summed, macro values are equal-sequence means, and micro values use the
    matching sequence support.  Booleans are schema metadata and must agree.
    """
    by_statistic: dict[str, list[tuple[dict[str, str], float | bool]]] = {}
    for row in rows:
        by_statistic.setdefault(row["statistic"], []).append((row, _as_number(row["value"])))
    result: dict[str, float | bool] = {}
    for statistic, entries in by_statistic.items():
        numeric = [float(value) for _, value in entries]
        if statistic in {"support_count", "frame_count", "frame_pair_count", "sequence_count"}:
            result[statistic] = float(sum(numeric))
        elif statistic == "higher_is_better":
            flags = {bool(value) for _, value in entries}
            if len(flags) != 1:
                raise RuntimeError("incompatible higher_is_better metadata across sequences")
            result[statistic] = flags.pop()
        elif statistic == "micro_pixel":
            weighted: list[tuple[float, float]] = []
            for row, value in entries:
                support = None
                # `rows` contains all sequence leaves: find this leaf's own support.
                for candidate in rows:
                    if candidate["sequence"] == row["sequence"] and candidate["statistic"] == "support_count":
                        support = float(candidate["value"]); break
                if support and support > 0:
                    weighted.append((float(value), support))
            result[statistic] = (sum(value * weight for value, weight in weighted) / sum(weight for _, weight in weighted)
                                 if weighted else sum(numeric) / len(numeric))
        elif statistic in {"median_sequence", "P95_sequence", "worst_sequence"}:
            macros = [float(value) for row, value in by_statistic.get("macro_sequence", entries)]
            if statistic == "median_sequence":
                result[statistic] = float(sorted(macros)[len(macros) // 2]) if len(macros) % 2 else float((sorted(macros)[len(macros)//2-1] + sorted(macros)[len(macros)//2]) / 2)
            elif statistic == "P95_sequence":
                result[statistic] = float(__import__("numpy").quantile(macros, .95))
            else:
                flags = {bool(value) for _, value in by_statistic.get("higher_is_better", [])}
                if len(flags) > 1:
                    raise RuntimeError("incompatible higher_is_better metadata across sequences")
                higher = flags.pop() if flags else False
                result[statistic] = min(macros) if higher else max(macros)
        else:
            result[statistic] = sum(numeric) / len(numeric)
    return result


def _pivot_gt(rows: list[dict[str, str]], provenance: Mapping[str, Any]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = {}
    descriptors: dict[tuple[str, str, str], tuple[str, str]] = {}
    for row in rows:
        if row["record_level"] != "aggregate":
            continue
        key = (row["dataset"], row["split"], row["backbone"], row["section"], row["metric_path"] + "\x1f" + row["method"])
        groups.setdefault(key, []).append(row)
        descriptors[(row["dataset"], row["split"], row["backbone"])] = (row["protocol"], row["metric_scope"])
    output: dict[tuple[str, str, str], dict[str, Any]] = {}
    for (dataset, split, backbone, section, encoded), leaves in groups.items():
        protocol, scope = descriptors[(dataset, split, backbone)]
        target = output.setdefault((dataset, split, backbone), _wide_metadata(dataset, split, backbone, provenance, protocol=protocol, scope=scope))
        metric_path, method = encoded.split("\x1f", 1)
        for statistic, value in _reduce_scalars(leaves).items():
            column = _wide_name(section, metric_path, method, statistic)
            if column in target and target[column] != value:
                raise RuntimeError(f"wide metric collision: {column}")
            target[column] = value
    return [output[key] for key in sorted(output)]


def _pivot_d4d(rows: list[dict[str, str]], provenance: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["record_level"] != "backbone":
            continue
        backbone = row["backbone"]
        target = output.setdefault(backbone, _wide_metadata("D4D", "", backbone, provenance, protocol=row["protocol"], scope=row["metric_scope"], applicability="NO_REFERENCE_DIAGNOSTIC"))
        column = _wide_name(row["section"], row["metric_path"], row["method"], row["statistic"])
        value = _as_number(row["value"])
        if column in target and target[column] != value:
            raise RuntimeError(f"wide D4D metric collision: {column}")
        target[column] = value
    return [output[key] for key in sorted(output)]


def build_wide_rows(stage: Path, provenance: Mapping[str, Any], selected: Iterable[str], source_manifests: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One dataset/split × module × backbone row, with all aggregate scalars."""
    rows: list[dict[str, Any]] = []
    selected = tuple(selected)
    if any(dataset.startswith("scared-") for dataset in selected):
        rows += _pivot_gt(_csv_rows(stage / "scared_aggregate_metrics.csv"), provenance)
    if "drends" in selected:
        rows += _pivot_gt(_csv_rows(stage / "drends_aggregate_metrics.csv"), provenance)
    if "d4d" in selected:
        rows += _pivot_d4d(_csv_rows(stage / "d4d_no_reference_summary.csv"), provenance)
    if "servct" in selected:
        for backbone in source_manifests["servct"].get("backbones", []):
            rows.append(_wide_metadata("SERV-CT", "", str(backbone), provenance,
                                       protocol="static stereo pairs; mandatory reset identity only", scope="not_applicable", applicability="NOT_APPLICABLE"))
    return sorted(rows, key=lambda row: tuple(str(row.get(key, "")) for key in ("dataset", "split", "temporal_model", "backbone")))


def write_wide_tables(stage: Path, provenance: Mapping[str, Any], selected: Iterable[str], source_manifests: Mapping[str, Mapping[str, Any]]) -> None:
    rows = build_wide_rows(stage, provenance, selected, source_manifests)
    if not rows:
        raise RuntimeError("definitive wide table has no rows")
    fields = list(WIDE_METADATA) + sorted({key for row in rows for key in row} - set(WIDE_METADATA))
    def write(path: Path, items: list[dict[str, Any]]) -> None:
        import io
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader(); writer.writerows(items)
        atomic_text(path, stream.getvalue())
    write(stage / "definitive_table.csv", rows)
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_dataset.setdefault(str(row["dataset"]), []).append(row)
    for dataset, items in by_dataset.items():
        write(stage / "tables" / f"{dataset.lower().replace('-', '_')}.csv", items)


def compile_servct(run: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    applicability = read_json(run / "applicability.json")
    if applicability.get("unified_gt_metric_families") != "NOT_APPLICABLE":
        raise RuntimeError(f"SERV-CT temporal status must be NOT_APPLICABLE: {run}")
    return [_base_row("SERV-CT", "", manifest, run, run / "applicability.json", record_level="applicability", section="temporal_h4",
                      metric_path="temporal_h4_evaluation", method="", metric_scope="not_applicable", applicability="NOT_APPLICABLE",
                      diagnostic_tags=json.dumps(applicability, sort_keys=True), value=None)]


def applicability_row(dataset: str, run: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    if dataset.startswith("scared-"):
        return _base_row("SCARED-C", dataset.removeprefix("scared-"), manifest, run, record_level="applicability", section="unified_metrics",
                         metric_path="gt_referenced_evaluation", metric_scope="gt_referenced", applicability="APPLICABLE")
    if dataset == "d4d":
        return _base_row("D4D", "", manifest, run, record_level="applicability", section="unified_metrics", metric_path="gt_referenced_evaluation",
                         metric_scope="no_reference_prediction_space", applicability="NOT_APPLICABLE")
    if dataset == "drends":
        return _base_row("DRENDS", "pilot", manifest, run, record_level="applicability", section="unified_metrics",
                         metric_path="tof_reference_evaluation", metric_scope="tof_reference_nonindependent", applicability="APPLICABLE_WITH_CAVEAT")
    return compile_servct(run, manifest)[0]


def consumed_paths(runs: Mapping[str, tuple[Path, Mapping[str, Any]]]) -> list[Path]:
    paths: list[Path] = []
    for dataset, (run, _) in runs.items():
        paths.append(run / "run_manifest.json")
        if dataset.startswith("scared-"):
            paths += sorted((run / "reports").glob("*/*.json"))
        elif dataset == "d4d":
            paths.append(run / "d4d_diagnostics.csv")
        elif dataset == "drends":
            paths += sorted((run / "reports").glob("*/*.json"))
        else:
            paths.append(run / "applicability.json")
    if any(not path.is_file() for path in paths):
        raise RuntimeError("missing consumed source artifact")
    return paths


def hash_map(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path.resolve()): sha256(path) for path in sorted(paths)}


def compile_results(source_root: Path, output: Path, datasets: Iterable[str]) -> None:
    selected = tuple(datasets)
    runs = {dataset: locate_run(source_root, dataset) for dataset in selected}
    module_values = [manifest["module_provenance"] for _, manifest in runs.values()]
    if any(value != module_values[0] for value in module_values[1:]):
        raise RuntimeError("source runs use different module provenance")
    source_paths = consumed_paths(runs)
    hashes_before = hash_map(source_paths)
    def write(stage: Path) -> None:
        wrapper = Path(__file__).resolve(); evaluator = wrapper.with_name("definitive_evaluation.py"); metrics = ROOT / "model_design/metrics/unified_metrics.py"; driver = wrapper.with_name("run_comparison.py")
        manifest: dict[str, Any] = {"project": "ARGOS v2", "status": "INCOMPLETE", "mode": "compile", "datasets": list(selected), "source_root": str(source_root.resolve()),
            "module_provenance": module_values[0], "wrapper": {"path": str(wrapper), "sha256": sha256(wrapper)}, "evaluator": {"path": str(evaluator), "sha256": sha256(evaluator)},
            "unified_metrics": {"path": str(metrics), "sha256": sha256(metrics)}, "comparison_driver": {"path": str(driver), "sha256": sha256(driver)},
            "source_runs": {dataset: {"path": str(run), "manifest_sha256": sha256(run / "run_manifest.json")} for dataset, (run, _) in runs.items()},
            "consumed_input_hashes": hashes_before, "consumed_input_hashes_note": "wrapper post-read integrity snapshot; source manifests do not predeclare every report/diagnostic/applicability artifact", "dense_predictions_written": False}
        atomic_json(stage / "run_manifest.json", manifest)
        aggregate_sink = CsvSink(stage / "scared_aggregate_metrics.csv") if any(dataset.startswith("scared-") for dataset in selected) else None
        sequence_sink = CsvSink(stage / "scared_per_sequence_metrics.csv") if aggregate_sink else None
        frame_sink = CsvSink(stage / "scared_per_frame_metrics.csv") if aggregate_sink else None
        drends_aggregate_sink = CsvSink(stage / "drends_aggregate_metrics.csv") if "drends" in selected else None
        drends_sequence_sink = CsvSink(stage / "drends_per_sequence_metrics.csv") if "drends" in selected else None
        drends_frame_sink = CsvSink(stage / "drends_per_frame_metrics.csv") if "drends" in selected else None
        d4d_sink = CsvSink(stage / "d4d_no_reference_metrics.csv") if "d4d" in selected else None
        d4d_summary_sink = CsvSink(stage / "d4d_no_reference_summary.csv") if "d4d" in selected else None
        applicability_sink = CsvSink(stage / "applicability.csv")
        try:
            for dataset, (run, source_manifest) in runs.items():
                if dataset.startswith("scared-"):
                    compile_scared(run, source_manifest, dataset, aggregate_sink, sequence_sink, frame_sink)
                elif dataset == "d4d":
                    path, source_rows = read_d4d(run)
                    d4d_sink.write(compile_d4d(run, source_manifest, path, source_rows))
                    d4d_summary_sink.write(d4d_summary_rows(run, source_manifest, path, source_rows))
                elif dataset == "drends":
                    compile_drends(run, source_manifest, drends_aggregate_sink, drends_sequence_sink, drends_frame_sink)
                applicability_sink.write([applicability_row(dataset, run, source_manifest)])
        finally:
            for sink in (aggregate_sink, sequence_sink, frame_sink, drends_aggregate_sink, drends_sequence_sink, drends_frame_sink,
                         d4d_sink, d4d_summary_sink, applicability_sink):
                if sink: sink.close()
        write_wide_tables(stage, module_values[0], selected, {dataset: manifest for dataset, (_, manifest) in runs.items()})
        hashes_after = hash_map(source_paths)
        if hashes_after != hashes_before:
            raise RuntimeError("consumed source artifact changed during compile")
        outputs = sorted(str(path.relative_to(stage)) for path in stage.rglob("*") if path.is_file() and path.name != "run_manifest.json")
        output_hashes = {name: sha256(stage / name) for name in outputs}
        atomic_json(stage / "run_manifest.json", manifest | {"status": "COMPLETE", "outputs": outputs, "output_hashes": output_hashes})
    staged_directory(output, write)


def run_evaluator(stage_root: Path, config: argparse.Namespace) -> Path:
    source_root = stage_root / "runs"
    evaluator = Path(__file__).with_name("definitive_evaluation.py")
    for dataset in config.datasets:
        if dataset == "drends":
            from model_design.comparison.drends_evaluation import evaluate_drends
            name = config.module.replace(":", "__").replace(".", "_")
            evaluate_drends(output=source_root / dataset / name, module_spec=config.module, device_name=config.device,
                            recordings=config.drends_recordings, max_frames=config.max_frames)
            continue
        backbones = (config.external_backbones or D4D_BACKBONES) if dataset in {"d4d", "servct"} else (config.scared_backbones or config.backbones or ALL_BACKBONES)
        if dataset in {"d4d", "servct"} and set(backbones) - set(D4D_BACKBONES):
            raise ValueError(f"{dataset} only has cached {', '.join(D4D_BACKBONES)}")
        name = config.module.replace(":", "__").replace(".", "_")
        command = [sys.executable, str(evaluator), "--dataset", dataset, "--module", config.module, "--output", str(source_root / dataset / name), "--device", config.device, "--backbones", *backbones, "--flow-batch-size", str(config.flow_batch_size)]
        if config.max_frames is not None: command += ["--max-frames", str(config.max_frames)]
        if config.smoke: command.append("--smoke")
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"frozen evaluator failed for {dataset}; the adapter contract requires a finite diagnostics.update_magnitude from start() and step()") from error
    return source_root


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--module", default=DEFAULT_MODULE)
    parser.add_argument("--scared-backbones", nargs="+", help="SCARED-C cache backbones; default is all five")
    parser.add_argument("--external-backbones", nargs="+", help="D4D/SERV-CT cache backbones; default is both compatible backbones")
    parser.add_argument("--backbones", nargs="+", help="legacy shorthand for --scared-backbones")
    parser.add_argument("--output", type=Path); parser.add_argument("--compile-from", type=Path)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--flow-batch-size", type=int, default=32)
    parser.add_argument("--max-frames", type=int); parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--drends-recordings", nargs="+", help="curated chronological DRENDS recordings; default Vid14 pilot")
    return parser.parse_args()


def main() -> None:
    config = arguments()
    output = config.output or RESULTS / config.module.replace(":", "__").replace(".", "_")
    if config.compile_from:
        compile_results(config.compile_from, output, config.datasets)
        return
    def run_and_compile(stage: Path) -> None:
        source_root = run_evaluator(stage, config)
        compiled = stage / "compiled"
        compile_results(source_root, compiled, config.datasets)
        for path in compiled.iterdir():
            os.rename(path, stage / path.name)
        compiled.rmdir()
        # The temporary staging path is not a durable provenance location.
        # Runs remain under the published output, so replace it before publish.
        for path in [stage / "run_manifest.json", *stage.glob("*.csv"), *stage.glob("tables/*.csv")]:
            atomic_text(path, path.read_text(encoding="utf-8").replace(str(stage), str(output.resolve())))
        finalize_output_hashes(stage)
    staged_directory(output, run_and_compile)


if __name__ == "__main__":
    main()
