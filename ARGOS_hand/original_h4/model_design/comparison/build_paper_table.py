#!/usr/bin/env python3
"""Compile the frozen ARGOS v2 evidence into a deterministic paper table.

This is deliberately a reader of completed artifacts.  It never loads the H4
adapter, starts CUDA, or writes predictions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from model_design.metrics.unified_metrics import MetricConfig, compute_spatial_metrics

RESULTS = ROOT.parent / "results/definitive_temporal_evaluation"
ARGOS_ROOT = ROOT.parents[1]
V2_ROOT = ARGOS_ROOT / "ARGOS-V2"
DEFAULT_OUTPUT = RESULTS / "paper/canonical_h4_v2"
EXPECTED_CHECKPOINT = "99c5745c164fd4903b8aa8acf8f57efccacd9cdcd0d4ed4305cd10609324d725"
EXPECTED_UNIFIED = "1142e53f5f6865343ca2723f125789e1c592b68278751e177517b33add10139a"
BACKBONES = ("S2M2-S", "RAFT-Stereo", "StereoAnywhere", "CREStereo", "Fast-FoundationStereo")
SEEN = ("S2M2-S", "RAFT-Stereo", "StereoAnywhere")
UNSEEN = ("CREStereo", "Fast-FoundationStereo")
SCARED_REPORT_FIELDS = {"aggregate", "applicability", "backbone", "dataset", "diagnostics", "frame_ids", "gate", "per_frame", "per_sequence", "primary_aggregate", "protocol", "safety", "selective", "sequence_ids", "spatial", "split", "temporal"}
D4D_FIELDS = {"dataset", "metric_scope", "backbone", "specimen", "session", "sequence", "window_index", "frame_id", "step_since_reset", "reset", "support_coverage", "raw_mc_inconsistency", "temporal_module_mc_inconsistency", "update_magnitude"}
CSV_FIELDS = ("panel", "row_level", "dataset", "split", "split_role", "sequence", "protocol", "domain_scope", "backbone_scope", "backbone", "baseline_method", "candidate_method", "metric_family", "metric", "unit", "metric_source", "aggregate", "baseline_value", "candidate_value", "delta", "relative_change_pct", "support_count", "frame_count", "sequence_count", "applicability", "verdict", "limitation", "source_id")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    def duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"duplicate JSON key {key!r}: {path}")
            out[key] = value
        return out
    def nonfinite(value: str) -> None:
        raise ValueError(f"non-finite JSON token {value}: {path}")
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=duplicate, parse_constant=nonfinite)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _atomic(path: Path, text: str) -> None:
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


def _staged_directory(output: Path, writer: Any) -> None:
    """Publish a complete directory once; the lock prevents cooperative races."""
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


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str] = CSV_FIELDS) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path}")
    import io
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=list(fields), extrasaction="raise")
    writer.writeheader(); writer.writerows(rows)
    _atomic(path, out.getvalue())


def finite(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite metric")
    return value


def scope(backbone: str) -> str:
    return "seen_training_backbone" if backbone in SEEN else "unseen_backbone"


def scared_context(split: str) -> tuple[str, str, str]:
    return ("validation", "in_domain", "separate validation split") if split == "d2" else ("heldout_test", "in_domain", "separate H4 support/test split")


def empty_row(**changes: Any) -> dict[str, Any]:
    row = {key: "" for key in CSV_FIELDS}
    row.update({"baseline_value": None, "candidate_value": None, "delta": None, "relative_change_pct": None,
                "support_count": 0, "frame_count": 0, "sequence_count": 0})
    row.update(changes)
    return row


def compare(baseline: float | None, candidate: float | None) -> tuple[float | None, float | None]:
    if baseline is None or candidate is None:
        return None, None
    delta = finite(candidate - baseline)
    return delta, finite(100.0 * delta / baseline) if baseline else None


def aggregate_sequence_metrics(entries: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Equal-sequence macro with separately retained support-weighted micro."""
    if not entries:
        raise ValueError("empty metric aggregation")
    values = [finite(entry["macro_sequence"]) for entry in entries]
    if any(value is None for value in values):
        raise ValueError("missing sequence metric")
    supports = [int(entry["support_count"]) for entry in entries]
    if any(value < 0 for value in supports):
        raise ValueError("negative support")
    total = sum(supports)
    micro = sum(value * support for value, support in zip(values, supports)) / total if total else None
    return {"macro_sequence": finite(sum(values) / len(values)), "micro_pixel": finite(micro),
            "support_count": total, "frame_count": sum(int(entry["frame_count"]) for entry in entries),
            "sequence_count": len(entries), "higher_is_better": bool(entries[0].get("higher_is_better", False))}


def _declared_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve(strict=True)


def validate_declared_artifact(declaration: Mapping[str, Any], label: str, expected: str | None = None) -> tuple[Path, str]:
    if not isinstance(declaration.get("path"), str) or not isinstance(declaration.get("sha256"), str):
        raise RuntimeError(f"invalid {label} declaration")
    try:
        path = _declared_path(declaration["path"])
    except FileNotFoundError as error:
        raise RuntimeError(f"missing declared {label}: {declaration['path']}") from error
    actual = sha256(path)
    declared = declaration["sha256"]
    if actual != declared or (expected is not None and actual != expected):
        raise RuntimeError(f"{label} hash mismatch: {path}")
    return path, actual


def validate_manifests() -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    manifests: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    invariant: dict[str, Any] | None = None
    for dataset in ("scared-d2", "scared-d7", "d4d", "servct"):
        path = RESULTS / dataset / "canonical_h4/run_manifest.json"
        manifest = read_json(path)
        if manifest.get("status") != "COMPLETE" or manifest.get("smoke") or manifest.get("max_frames"):
            raise RuntimeError(f"not a complete non-smoke manifest: {path}")
        module = manifest.get("module_provenance", {})
        if module.get("checkpoint_sha256") != EXPECTED_CHECKPOINT:
            raise RuntimeError(f"checkpoint hash mismatch: {path}")
        if manifest.get("unified_metrics", {}).get("sha256") != EXPECTED_UNIFIED:
            raise RuntimeError(f"unified-metrics hash mismatch: {path}")
        declarations = {
            "checkpoint": {"path": module.get("checkpoint"), "sha256": module.get("checkpoint_sha256")},
            "policy": {"path": module.get("policy"), "sha256": module.get("policy_sha256")},
            "module code": {"path": module.get("code"), "sha256": module.get("code_sha256")},
            "comparison driver": manifest.get("comparison_driver", {}),
            "evaluator": manifest.get("evaluator", {}),
            "unified metrics": manifest.get("unified_metrics", {}),
        }
        for label, declaration in declarations.items():
            declared_path, digest = validate_declared_artifact(declaration, label, EXPECTED_CHECKPOINT if label == "checkpoint" else EXPECTED_UNIFIED if label == "unified metrics" else None)
            hashes[str(declared_path)] = digest
        current = {key: module.get(key) for key in ("checkpoint_sha256", "policy_sha256", "code_sha256", "reset_protocol")}
        current |= {key: manifest.get(key, {}).get("sha256") for key in ("unified_metrics", "comparison_driver", "evaluator")}
        if invariant is None:
            invariant = current
        elif current != invariant:
            raise RuntimeError("canonical policy/code/driver/evaluator provenance differs across sources")
        manifests[dataset] = manifest; hashes[str(path)] = sha256(path)
    training = ROOT / "model_design/checkpoints/training_provenance.json"
    provenance = read_json(training)
    if provenance.get("canonical_checkpoint", {}).get("sha256") != EXPECTED_CHECKPOINT:
        raise RuntimeError("frozen training provenance checkpoint mismatch")
    hashes[str(training)] = sha256(training)
    return manifests, hashes


def scared_reports(split: str, input_hashes: dict[str, str]) -> list[dict[str, Any]]:
    expected_sequences = ("dataset_2_keyframe_2", "dataset_2_keyframe_3", "dataset_2_keyframe_4") if split == "d2" else tuple(f"dataset_7_keyframe_{i}" for i in range(1, 5))
    expected_protocol = "paper_d2_strict_all_anchors" if split == "d2" else "h4_only_common_support"
    root = RESULTS / f"scared-{split}/canonical_h4/reports"
    reports: list[dict[str, Any]] = []
    for backbone in BACKBONES:
        paths = sorted((root / backbone).glob("*.json"))
        if [path.stem for path in paths] != list(expected_sequences):
            raise RuntimeError(f"SCARED-{split} coverage mismatch for {backbone}")
        for path in paths:
            report = read_json(path)
            if set(report) != SCARED_REPORT_FIELDS:
                raise RuntimeError(f"unexpected SCARED report schema: {path}")
            if report.get("dataset") != "SCARED-C" or report.get("backbone") != backbone or report.get("split") != split or report.get("protocol") != expected_protocol or report.get("sequence_ids") != [path.stem]:
                raise RuntimeError(f"invalid frozen SCARED report: {path}")
            if report.get("primary_aggregate") != "macro_sequence" or not isinstance(report.get("aggregate"), Mapping):
                raise RuntimeError(f"invalid SCARED aggregate schema: {path}")
            if report.get("applicability", {}).get("official_support", "").startswith("gt_valid AND protocol_mask") is False:
                raise RuntimeError(f"non-official support: {path}")
            reports.append(report); input_hashes[str(path)] = sha256(path)
    if len(reports) != len(BACKBONES) * len(expected_sequences):
        raise RuntimeError("incomplete SCARED report coverage")
    return reports


def _scared_entries(reports: list[dict[str, Any]], split: str, family: str, metric: str, source_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output, flat = [], []
    unit = "px" if family == "disparity_px" else "mm"
    split_role, domain_scope, limitation = scared_context(split)
    for backbone in BACKBONES:
        selected = [report for report in reports if report["backbone"] == backbone]
        raw = [report["aggregate"][family]["raw"][metric] for report in selected]
        refined = [report["aggregate"][family]["refined"][metric] for report in selected]
        if [entry["support_count"] for entry in raw] != [entry["support_count"] for entry in refined]:
            raise RuntimeError(f"raw/refined support mismatch: {split}/{backbone}/{family}/{metric}")
        a, b = aggregate_sequence_metrics(raw), aggregate_sequence_metrics(refined)
        delta, relative = compare(a["macro_sequence"], b["macro_sequence"])
        output.append(empty_row(panel="SCARED primary", row_level="split", dataset="SCARED-C", split=split, split_role=split_role,
            sequence="ALL", protocol=selected[0]["protocol"], domain_scope=domain_scope, backbone_scope=scope(backbone), backbone=backbone,
            baseline_method="raw", candidate_method="canonical_h4", metric_family=family, metric=metric, unit=unit, metric_source="official_unified_metrics",
            aggregate="macro_sequence", baseline_value=a["macro_sequence"], candidate_value=b["macro_sequence"], delta=delta, relative_change_pct=relative,
            support_count=a["support_count"], frame_count=a["frame_count"], sequence_count=a["sequence_count"], applicability="APPLICABLE", verdict="OBSERVED", limitation=limitation, source_id=source_id))
        for method, aggregate in (("raw", a), ("canonical_h4", b)):
            flat.append({"dataset": "SCARED-C", "split": split, "backbone": backbone, "sequence": "ALL", "protocol": selected[0]["protocol"], "metric_family": family, "metric": metric,
                         "method": method, "primary_aggregate": "macro_sequence", **aggregate, "source_id": source_id})
    return output, flat


def scared_rows(reports: list[dict[str, Any]], split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, flat = [], []
    source = f"definitive/scared-{split}/canonical_h4"
    split_role, domain_scope, limitation = scared_context(split)
    for family, metrics in (("disparity_px", ("EPE", "RMSE", "Bad1", "Bad3", "Bad5")), ("depth_mm", ("MAE", "RMSE"))):
        for metric in metrics:
            a, b = _scared_entries(reports, split, family, metric, source); rows += a; flat += b
    for metric in ("HUR", "BUR"):
        # Safety is separately labelled so it cannot be pooled with spatial error.
        for backbone in BACKBONES:
            selected = [report for report in reports if report["backbone"] == backbone]
            entries = [report["safety"]["disparity_px"]["aggregate"][metric] for report in selected]
            aggregate = aggregate_sequence_metrics(entries)
            rows.append(empty_row(panel="SCARED safety", row_level="split", dataset="SCARED-C", split=split, split_role=split_role, sequence="ALL",
                protocol=selected[0]["protocol"], domain_scope=domain_scope, backbone_scope=scope(backbone), backbone=backbone,
                baseline_method="raw", candidate_method="canonical_h4_vs_raw", metric_family="safety_disparity_px", metric=metric, unit="fraction", metric_source="official_unified_metrics",
                aggregate="macro_sequence", baseline_value=None, candidate_value=aggregate["macro_sequence"], delta=None, relative_change_pct=None,
                support_count=aggregate["support_count"], frame_count=aggregate["frame_count"], sequence_count=aggregate["sequence_count"], applicability="APPLICABLE", verdict="DIAGNOSTIC", limitation=f"{limitation}; refinement safety, not spatial error", source_id=source))
            flat.append({"dataset": "SCARED-C", "split": split, "backbone": backbone, "sequence": "ALL", "protocol": selected[0]["protocol"], "metric_family": "safety_disparity_px", "metric": metric,
                         "method": "canonical_h4_vs_raw", "primary_aggregate": "macro_sequence", **aggregate, "source_id": source})
    # Temporal reports are diagnostics (not paper spatial scores), but retaining
    # their labels here makes the flat source auditably complete.
    temporal: dict[tuple[str, str, str, str, str], list[Mapping[str, Any]]] = {}
    for report in reports:
        for unit, horizons in report.get("temporal", {}).items():
            for horizon, payload in horizons.items():
                for method, measures in payload.get("methods", {}).items():
                    for measure, metrics in measures.items():
                        if measure == "aggregate":
                            continue
                        for metric, value in metrics.items():
                            if isinstance(value, Mapping) and "value" in value:
                                key = (report["backbone"], unit, str(horizon), method, f"{measure}/{metric}")
                                temporal.setdefault(key, []).append({"macro_sequence": value["value"], "support_count": value["support_count"], "frame_count": value["frame_count"]})
    for (backbone, unit, horizon, method, metric), entries in sorted(temporal.items()):
        aggregate = aggregate_sequence_metrics(entries)
        flat.append({"dataset": "SCARED-C", "split": split, "backbone": backbone, "sequence": "ALL", "protocol": "diagnostic_grid_based", "metric_family": f"temporal_diagnostic_{unit}", "metric": f"h{horizon}/{metric}", "method": method, "primary_aggregate": "macro_sequence", **aggregate, "source_id": source})
    return rows, flat


def d4d_rows(input_hashes: dict[str, str]) -> list[dict[str, Any]]:
    path = RESULTS / "d4d/canonical_h4/d4d_diagnostics.csv"
    rows = read_csv(path)
    input_hashes[str(path)] = sha256(path)
    if not rows or set(rows[0]) != D4D_FIELDS or any(set(row) != D4D_FIELDS for row in rows):
        raise RuntimeError("D4D diagnostics schema mismatch")
    if any(row["dataset"] != "D4D" or row["metric_scope"] != "no_reference_prediction_space" for row in rows):
        raise RuntimeError("D4D dataset or metric scope mismatch")
    for row in rows:
        for key in ("support_coverage", "raw_mc_inconsistency", "temporal_module_mc_inconsistency", "update_magnitude"):
            if finite(row[key]) is None:
                raise RuntimeError(f"D4D missing value: {key}")
    if len(rows) != 504 or {row["backbone"] for row in rows} != {"RAFT-Stereo", "StereoAnywhere"}:
        raise RuntimeError("D4D must contain exactly 504 two-backbone diagnostics")
    if {row["specimen"] for row in rows} != {"specimen_2", "specimen_3"}:
        raise RuntimeError("D4D must use specimens 2 and 3 only")
    out: list[dict[str, Any]] = []
    for backbone in ("RAFT-Stereo", "StereoAnywhere"):
        part = [row for row in rows if row["backbone"] == backbone]
        if len(part) != 252:
            raise RuntimeError(f"D4D incomplete backbone: {backbone}")
        for metric, baseline_key, candidate_key, unit in (("MC inconsistency", "raw_mc_inconsistency", "temporal_module_mc_inconsistency", "px"), ("support coverage", "support_coverage", "support_coverage", "fraction"), ("update magnitude", None, "update_magnitude", "px")):
            base = finite(sum(float(row[baseline_key]) for row in part) / len(part)) if baseline_key else None
            candidate = finite(sum(float(row[candidate_key]) for row in part) / len(part))
            delta, relative = compare(base, candidate)
            out.append(empty_row(panel="D4D no-reference", row_level="dataset", dataset="D4D", split="OOD", split_role="OOD", sequence="ALL", protocol="four-frame curated causal windows, past-to-present",
                domain_scope="clinical_OOD", backbone_scope="seen_training_backbone", backbone=backbone, baseline_method="raw", candidate_method="canonical_h4", metric_family="no_reference_prediction_space", metric=metric,
                unit=unit, metric_source="official_d4d_diagnostics", aggregate="mean_frame", baseline_value=base, candidate_value=candidate, delta=delta, relative_change_pct=relative,
                support_count=len(part), frame_count=len(part), sequence_count=len({row["sequence"] for row in part}), applicability="NOT_APPLICABLE" if metric == "update magnitude" else "APPLICABLE",
                verdict="DIAGNOSTIC", limitation="no ground truth or geometric metric", source_id="definitive/d4d/canonical_h4"))
    out.append(empty_row(panel="D4D applicability", row_level="dataset", dataset="D4D", split="OOD", split_role="OOD", sequence="ALL", protocol="four-frame curated causal windows, past-to-present", domain_scope="clinical_OOD", backbone_scope="seen_training_backbone", backbone="RAFT-Stereo,StereoAnywhere", baseline_method="raw", candidate_method="canonical_h4", metric_family="applicability", metric="unavailable specimen_1 windows", unit="windows", metric_source="official_d4d_diagnostics", aggregate="count", baseline_value=None, candidate_value=72.0, support_count=0, frame_count=0, sequence_count=0, applicability="NOT_APPLICABLE", verdict="EXCLUDED", limitation="source pairs unavailable", source_id="definitive/d4d/canonical_h4"))
    return out


def _serv_contract(input_hashes: dict[str, str]) -> tuple[list[dict[str, str]], dict[str, tuple[np.ndarray, np.ndarray]]]:
    prepared = ARGOS_ROOT / "results/03_temporal_refinement/ood/prepared/servct"
    sequence_path, index_path = prepared / "sequence_manifest.csv", prepared / "frame_targets_index.csv"
    sequence_rows, index = read_csv(sequence_path), read_csv(index_path)
    for path in (sequence_path, index_path): input_hashes[str(path)] = sha256(path)
    expected = [(row["sequence_id"], row["frame_id"]) for row in sorted(sequence_rows, key=lambda row: (row["sequence_id"], int(row["order_index"])))]
    indexed = sorted(index, key=lambda row: (row["sequence_id"], int(row["frame_index"])))
    if len(expected) != 16 or len(set(expected)) != 16 or [(row["sequence_id"], row["frame_id"]) for row in indexed] != expected:
        raise RuntimeError("SERV frame-ID join is not exact")
    if any((row["disp_convention"] != "positive_px_left_reference" or row["source_units"] != "disparity_px / depth_mm" or row["converted_units"] != "disparity_px / depth_mm" or float(row["disp_scale"]) != 1.0 or int(row["width"]) != 720 or int(row["height"]) != 576 or row["continuity_flag"] != "weak_sparse") for row in sequence_rows):
        raise RuntimeError("SERV GT convention or native grid is not proven")
    shards: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sequence in sorted({row["sequence_id"] for row in sequence_rows}):
        path = prepared / "shards" / f"{sequence}.npz"; z = np.load(path)
        if z["gt_disp"].shape != (8, 144, 180) or z["valid_mask"].shape != (8, 144, 180):
            raise RuntimeError("SERV GT grid mismatch")
        gt, valid = np.asarray(z["gt_disp"], np.float32), np.asarray(z["valid_mask"], bool)
        if not np.all(np.isfinite(gt[valid]) & (gt[valid] > 0)):
            raise RuntimeError("SERV GT invalid on valid mask")
        shards[sequence] = (gt, valid); input_hashes[str(path)] = sha256(path)
    return sorted(sequence_rows, key=lambda row: (row["sequence_id"], int(row["order_index"]))), shards


def serv_cache_factor(rows: list[dict[str, str]], metadata: Mapping[str, Any]) -> float:
    widths = {int(row["width"]) for row in rows}
    if len(widths) != 1 or metadata.get("disparity_units") != "pixels_at_cache_resolution" or metadata.get("disparity_convention") != "positive_left_disparity":
        raise RuntimeError("SERV native/cache disparity contract mismatch")
    try:
        factor = finite(float(metadata["cache_width"]) / widths.pop())
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("SERV cache width is invalid") from error
    if factor is None or factor <= 0:
        raise RuntimeError("SERV disparity scale must be finite and positive")
    return factor


def serv_rows(input_hashes: dict[str, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, shards = _serv_contract(input_hashes)
    by_sequence = {sequence: [row for row in rows if row["sequence_id"] == sequence] for sequence in sorted({row["sequence_id"] for row in rows})}
    output, flat = [], []
    for backbone in ("RAFT-Stereo", "StereoAnywhere"):
        base = V2_ROOT / "cache_multidomain_backbones" / backbone / "SERV-CT"
        metadata, manifest = read_json(base / "metadata.json"), read_csv(base / "frame_manifest.csv")
        needed = (base / ".complete", base / "disparity.npy", base / "valid_mask.npy", base / "frame_ids.npy", base / "metadata.json", base / "frame_manifest.csv")
        if not all(path.exists() for path in needed) or metadata.get("completion_status") is not True or metadata.get("integrity_checks", {}).get("passed") is not True or metadata.get("frame_count") != 16 or metadata.get("cache_height") != 144 or metadata.get("cache_width") != 180 or metadata.get("disparity_convention") != "positive_left_disparity":
            raise RuntimeError(f"SERV cache contract failed: {backbone}")
        factor = serv_cache_factor(rows, metadata)
        for path in needed: input_hashes[str(path)] = sha256(path)
        cache_ids = [str(value) for value in np.load(base / "frame_ids.npy", allow_pickle=True).tolist()]
        manifest_ids = [row["frame_id"] for row in manifest]
        expected_ids = [f"{row['sequence_id']}__{row['frame_id']}" for row in rows]
        if cache_ids != manifest_ids or cache_ids != expected_ids or len(cache_ids) != 16:
            raise RuntimeError(f"SERV cache IDs mismatch: {backbone}")
        disparity, valid = np.load(base / "disparity.npy", mmap_mode="r"), np.load(base / "valid_mask.npy", mmap_mode="r")
        valid_bool = valid.astype(bool)
        if disparity.shape != (16, 144, 180) or valid.shape != disparity.shape or disparity.dtype != np.float16 or valid.dtype != np.uint8 or not np.all(np.isfinite(disparity[valid_bool]) & (disparity[valid_bool] > 0)):
            raise RuntimeError(f"SERV cache arrays invalid: {backbone}")
        lookup = {frame_id: i for i, frame_id in enumerate(cache_ids)}
        reports = []
        for sequence, seq_rows in by_sequence.items():
            pred = np.asarray([disparity[lookup[f"{sequence}__{row['frame_id']}"]] for row in seq_rows], np.float32)
            mask = np.asarray([valid[lookup[f"{sequence}__{row['frame_id']}"]] for row in seq_rows], bool)
            gt_native, gt_valid = shards[sequence]
            gt = gt_native * factor
            if pred.shape != (8, 144, 180): raise RuntimeError("SERV prediction grid invalid")
            reset_identity = pred.copy()
            if not np.array_equal(pred, reset_identity): raise RuntimeError("SERV reset identity is not bit-exact")
            calibration_values = {(row["fx_px"], row["baseline_mm"], row["width"]) for row in seq_rows}
            if len(calibration_values) != 1: raise RuntimeError("SERV calibration varies within sequence")
            fx_native, baseline, width = next(iter(calibration_values))
            if float(width) != 720:
                raise RuntimeError("SERV native width differs from prepared-grid contract")
            calibration = MetricConfig(fx_px=float(fx_native) * factor, baseline_mm=float(baseline))
            reports.append(compute_spatial_metrics(pred, gt, gt_valid, mask, calibration))
        for family, metrics, unit in (("disparity_px", ("EPE", "RMSE", "Bad1", "Bad3", "Bad5"), "px"), ("depth_mm", ("MAE", "RMSE"), "mm")):
            for metric in metrics:
                aggregate = aggregate_sequence_metrics([report["aggregate"][family][metric] for report in reports])
                output.append(empty_row(panel="SERV-CT static identity", row_level="split", dataset="SERV-CT", split="honest_train+honest_test", split_role="context_only", sequence="ALL", protocol="STATIC-ONLY/reset_identity; gt_cache=gt_native*180/720; non-consecutive", domain_scope="clinical_OOD", backbone_scope="seen_training_backbone", backbone=backbone, baseline_method="raw", candidate_method="reset_identity", metric_family=family, metric=metric, unit=unit, metric_source="compute_spatial_metrics", aggregate="macro_sequence", baseline_value=aggregate["macro_sequence"], candidate_value=aggregate["macro_sequence"], delta=0.0, relative_change_pct=0.0, support_count=aggregate["support_count"], frame_count=aggregate["frame_count"], sequence_count=aggregate["sequence_count"], applicability="STATIC_ONLY", verdict="IDENTITY", limitation="temporal_h4 NOT_APPLICABLE; weak_sparse non-consecutive; cached prediction unscaled", source_id="servct_static_cache"))
                for method in ("raw", "reset_identity"):
                    flat.append({"dataset": "SERV-CT", "split": "honest_train+honest_test", "backbone": backbone, "sequence": "ALL", "protocol": "STATIC-ONLY/reset_identity; non-consecutive", "metric_family": family, "metric": metric, "method": method, "primary_aggregate": "macro_sequence", **aggregate, "source_id": "servct_static_cache"})
    return output, flat


def applicability_rows() -> list[dict[str, Any]]:
    common = dict(panel="Applicability", row_level="dataset", protocol="canonical_h4", domain_scope="clinical_OOD", baseline_method="raw", candidate_method="canonical_h4", metric_family="applicability", metric="canonical H4 prediction cache", unit="", metric_source="compiler", aggregate="none", applicability="NOT_APPLICABLE", verdict="UNTESTED", limitation="no canonical H4 prediction caches; no inference performed", source_id="compiler")
    return [empty_row(dataset="StereoMIS", split="OOD", split_role="OOD", sequence="ALL", backbone_scope="unknown", backbone="ALL", **common),
            empty_row(dataset="joint unseen-backbone+OOD", split="OOD", split_role="OOD", sequence="ALL", backbone_scope="unseen_backbone", backbone="CREStereo,Fast-FoundationStereo", **common)]


def rounded(value: Any) -> str:
    return "N/A" if value is None or value == "" else f"{float(value):.4f}" if isinstance(value, (float, int)) else str(value)


def compact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = [row for row in rows if (row["dataset"] == "SCARED-C" and row["metric"] == "EPE") or (row["dataset"] == "D4D" and row["metric"] == "MC inconsistency") or (row["dataset"] == "SERV-CT" and row["metric"] == "EPE") or row["dataset"] in {"StereoMIS", "joint unseen-backbone+OOD"}]
    return wanted


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["# ARGOS v2 definitive paper table", "", "No global pooled score; SCARED-D2/D7 and metric families remain separate. Values below are `macro_sequence`; support-weighted `micro_pixel` values live in `all_unified_metrics.csv`.", "", "| Dataset | Split/OOD role | Scope | Backbone | Protocol | Methods | Metric | Raw | Candidate | Δ | Verdict |", "|---|---|---|---|---|---|---|---:|---:|---:|---|"]
    for row in compact_rows(rows):
        values = ("dataset", "split_role", "backbone_scope", "backbone", "protocol", "baseline_method", "candidate_method", "metric", "baseline_value", "candidate_value", "delta", "verdict")
        rendered = [str(row[key]) if key not in {"baseline_value", "candidate_value", "delta"} else rounded(row[key]) for key in values]
        rendered[5] = f"{rendered[5]} → {rendered[6]}"; del rendered[6]
        rendered[6] = f"{rendered[6]} [macro_sequence]"
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines) + "\n"


def render_tex(rows: list[dict[str, Any]]) -> str:
    def esc(value: Any) -> str: return str(value).replace("_", "\\_").replace("+", "{+}")
    br = "\\\\"
    lines = ["\\begin{tabular}{lllllllrrrl}", f"Dataset & Role & Scope & Backbone & Protocol & Methods & Metric & Raw & Candidate & $\\Delta$ & Verdict {br}", "\\hline"]
    for row in compact_rows(rows):
        values = ("dataset", "split_role", "backbone_scope", "backbone", "protocol", "metric", "baseline_value", "candidate_value", "delta", "verdict")
        rendered = [esc(row[key]) if key not in {"baseline_value", "candidate_value", "delta"} else rounded(row[key]) for key in values]
        rendered.insert(5, f"{esc(row['baseline_method'])} $\\to$ {esc(row['candidate_method'])}")
        rendered[6] = f"{rendered[6]} [macro\\_sequence]"
        lines.append(" & ".join(rendered) + f" {br}")
    return "\n".join(lines + ["\\end{tabular}", ""])


def _metric_leaves(value: Any, path: tuple[str, ...] = (), tags: Mapping[str, Any] | None = None):
    tags = dict(tags or {})
    if not isinstance(value, Mapping):
        return
    tags.update({key: item for key, item in value.items() if key in {"diagnostic_grid_based", "flow_conditioned_proxy"}})
    if "support_count" in value and ("macro_sequence" in value or "value" in value):
        yield path, value, tags
        return
    for key, item in value.items():
        if key not in {"diagnostic_grid_based", "flow_conditioned_proxy"}:
            yield from _metric_leaves(item, path + (str(key),), tags)


def flatten_scared_reports(reports: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    """Flatten all report aggregate/spatial/safety/temporal leaves, never frames."""
    rows: list[dict[str, Any]] = []
    for report in reports:
        for section in ("aggregate", "spatial", "safety", "temporal"):
            for path, leaf, tags in _metric_leaves(report.get(section, {}), (section,)):
                value = leaf.get("macro_sequence", leaf.get("value"))
                if value is None:
                    continue
                method = next((part for part in path if part in {"raw", "refined"}), "candidate_only")
                horizon = next((part for part in path if part.isdigit()), "")
                rows.append({"dataset": report["dataset"], "split": split, "backbone": report["backbone"], "sequence": report["sequence_ids"][0], "protocol": report["protocol"], "section": section,
                             "metric_path": "/".join(path), "method": method, "horizon": horizon, "diagnostic_tags": json.dumps(tags, sort_keys=True), "value": finite(value),
                             "primary_aggregate": leaf.get("primary_aggregate", "per_sequence"), "macro_sequence": finite(leaf.get("macro_sequence")), "micro_pixel": finite(leaf.get("micro_pixel")),
                             "support_count": int(leaf["support_count"]), "frame_count": int(leaf["frame_count"]), "sequence_count": int(leaf["sequence_count"]), "source_id": f"definitive/scared-{split}/canonical_h4"})
    if not rows:
        raise RuntimeError("empty SCARED unified flattening")
    return rows


def build(output: Path) -> None:
    _staged_directory(output, _build_into)


def _build_into(output: Path) -> None:
    manifests, input_hashes = validate_manifests()
    reports_d2, reports_d7 = scared_reports("d2", input_hashes), scared_reports("d7", input_hashes)
    rows_d2, _ = scared_rows(reports_d2, "d2"); rows_d7, _ = scared_rows(reports_d7, "d7")
    d4d = d4d_rows(input_hashes); serv, flat_serv = serv_rows(input_hashes)
    rows = rows_d2 + rows_d7 + d4d + serv + applicability_rows()
    if any(row["dataset"] == "SCARED-C" and row["aggregate"] != "macro_sequence" for row in rows):
        raise RuntimeError("SCARED primary must be macro_sequence")
    atomic_csv(output / "paper_table.csv", rows)
    all_flat = flatten_scared_reports(reports_d2, "d2") + flatten_scared_reports(reports_d7, "d7") + flat_serv
    atomic_csv(output / "all_unified_metrics.csv", all_flat, fields=tuple(dict.fromkeys(key for row in all_flat for key in row)))
    _atomic(output / "paper_table.md", render_markdown(rows)); _atomic(output / "paper_table.tex", render_tex(rows))
    output_hashes = {str(path.relative_to(output)): sha256(path) for path in sorted(output.iterdir()) if path.is_file()}
    compiler = Path(__file__).resolve()
    provenance = {"project": "ARGOS v2", "status": "COMPLETE", "compiler": {"path": str(compiler), "sha256": sha256(compiler)}, "inputs": dict(sorted(input_hashes.items())), "outputs": output_hashes,
        "output_hash_note": "provenance.json is excluded from its own hash map because a self-referential SHA-256 cannot be represented in the file it hashes.", "source_manifests": manifests,
        "expected_hashes": {"checkpoint": EXPECTED_CHECKPOINT, "unified_metrics": EXPECTED_UNIFIED}, "tuning": "none", "aggregation_rules": {"SCARED_primary": "equal-sequence macro_sequence", "micro": "support-weighted micro_pixel retained only in all_unified_metrics.csv", "pooling": "never pool D2+D7 or metric families"},
        "dataset_roles": {"SCARED-D2": "validation", "SCARED-D7": "held-out in-domain SCARED-C test", "D4D": "no-reference clinical OOD", "SERV-CT": "static-only clinical OOD context", "StereoMIS": "UNTESTED clinical OOD"}, "backbone_status": {"seen": list(SEEN), "unseen": list(UNSEEN)},
        "exclusions": ["A2", "critic", "geometry-v1", "older 936-row D4D replay", "smoke outputs", "canonical_h4_v1 superseded: SERV native/cache disparity unit mismatch"], "serv_disparity_conversion": "gt_cache=gt_native*180/720; fx_cache=fx_native*180/720; cached predictions are not rescaled", "serv_temporal_h4": "NOT_APPLICABLE/static non-consecutive"}
    atomic_json(output / "provenance.json", provenance)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    build(arguments().output)
