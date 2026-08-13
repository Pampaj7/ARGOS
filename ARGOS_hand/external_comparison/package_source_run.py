"""Package validated competitor output through the frozen SCARED metric implementation."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from bridge import read_input, read_output_snapshot

ROOT = Path(__file__).resolve().parent
ARGOS = ROOT.parent
H4 = ARGOS / "original_h4"
sys.path.insert(0, str(H4))
from model_design.comparison.definitive_evaluation import evaluate_scared_bundle  # noqa: E402

METHODS = {item["method"]: item for path in (ROOT / "protocols").glob("*.json") for item in [json.loads(path.read_text())]}
EVALUATION_LOCK = ROOT / "evaluation_artifacts.lock.json"
EXECUTION_MANIFEST_LOCK = ROOT / "execution_manifests.lock.json"
UPSTREAM_LOCK = ROOT / "upstreams.lock.json"
CHECKPOINT_LOCK = ROOT / "checkpoints.lock.json"
EVALUATION_METADATA = ("dataset", "split", "backbone", "sequence_id")
METHOD_IDENTITIES = {
    "nvds_plus_forward_clip4": ("nvds", "nvds_plus"),
    "nvds_plus_bidirectional_offline": ("nvds", "nvds_plus"),
    "bidastabilizer_bidirectional_offline": ("bidavideo", "bidastabilizer_raftstereo_robust"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _evaluation_artifact(artifact_id: str, input_sha256: str) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    artifacts = _json(EVALUATION_LOCK).get("artifacts")
    if not isinstance(artifacts, dict) or not isinstance(artifacts.get(artifact_id), dict):
        raise ValueError(f"unknown evaluation artifact: {artifact_id}")
    locked = artifacts[artifact_id]
    path_value = locked.get("path")
    if not isinstance(path_value, str):
        raise ValueError(f"evaluation artifact path missing: {artifact_id}")
    path = (ROOT / path_value).resolve()
    if ROOT not in path.parents or path.suffix != ".npz":
        raise ValueError(f"invalid evaluation artifact path: {artifact_id}")
    sidecar = path.with_suffix(".json")
    if not path.is_file() or not sidecar.is_file():
        raise ValueError(f"evaluation artifact hash mismatch: {artifact_id}")
    artifact, sidecar_data = path.read_bytes(), sidecar.read_bytes()
    for label, data in (("npz_sha256", artifact), ("json_sha256", sidecar_data)):
        if not isinstance(locked.get(label), str) or hashlib.sha256(data).hexdigest() != locked[label]:
            raise ValueError(f"evaluation artifact hash mismatch: {artifact_id}")
    metadata = json.loads(sidecar_data)
    expected_metadata = locked.get("metadata")
    if locked.get("input_sha256") != input_sha256 or not isinstance(expected_metadata, dict):
        raise ValueError(f"evaluation artifact input mismatch: {artifact_id}")
    if any(metadata.get(key) != expected_metadata.get(key) or not isinstance(metadata.get(key), str) or not metadata[key]
           for key in EVALUATION_METADATA):
        raise ValueError(f"evaluation artifact metadata mismatch: {artifact_id}")
    return artifact, metadata, locked


def _diagnostic_evaluation(path: Path, input_sha256: str) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    """Accept only a self-hashed local smoke artifact; it can never publish."""
    path = path.resolve()
    results = (ROOT / "results").resolve()
    if results not in path.parents or path.suffix != ".npz" or not path.is_file() or not path.with_suffix(".json").is_file():
        raise ValueError("invalid diagnostic evaluation artifact path")
    artifact, sidecar = path.read_bytes(), path.with_suffix(".json").read_bytes()
    metadata = json.loads(sidecar)
    if (metadata.get("purpose") not in {"SMOKE_DIAGNOSTIC", "D2_FULL_DIAGNOSTIC"} or metadata.get("publication") != "TEST_ONLY"
            or metadata.get("input_sha256") != input_sha256 or metadata.get("evaluation_npz_sha256") != hashlib.sha256(artifact).hexdigest()):
        raise ValueError("diagnostic evaluation artifact mismatch")
    bridge_id = metadata.get("bridge_artifact_id")
    bridge = path.parent / str(bridge_id)
    if (not isinstance(bridge_id, str) or Path(bridge_id).name != bridge_id or not bridge.is_file()
            or metadata.get("bridge_npz_sha256") != sha256(bridge)
            or metadata.get("bridge_json_sha256") != sha256(bridge.with_suffix(".json"))):
        raise ValueError("diagnostic evaluation bridge binding mismatch")
    return artifact, metadata, {"publication": "TEST_ONLY", "npz_sha256": hashlib.sha256(artifact).hexdigest(),
                                "json_sha256": hashlib.sha256(sidecar).hexdigest()}


def _execution_manifest(execution_id: str, method: str, input_sha256: str, prediction_sha256: str) -> dict[str, str]:
    manifests = _json(EXECUTION_MANIFEST_LOCK).get("manifests")
    if not isinstance(manifests, dict) or not isinstance(manifests.get(execution_id), dict):
        raise ValueError(f"unknown execution manifest: {execution_id}")
    locked = manifests[execution_id]
    path_value, locked_sha256 = locked.get("path"), locked.get("sha256")
    if not isinstance(path_value, str) or not isinstance(locked_sha256, str):
        raise ValueError(f"execution manifest lock incomplete: {execution_id}")
    path = (ROOT / path_value).resolve()
    if ROOT not in path.parents or path.suffix != ".json" or not path.is_file():
        raise ValueError(f"execution manifest hash mismatch: {execution_id}")
    manifest_data = path.read_bytes()
    if hashlib.sha256(manifest_data).hexdigest() != locked_sha256:
        raise ValueError(f"execution manifest hash mismatch: {execution_id}")
    manifest = json.loads(manifest_data)
    protocol = ROOT / "protocols" / f"{method}.json"
    if manifest.get("method") != method or not protocol.is_file() or manifest.get("protocol_sha256") != sha256(protocol):
        raise ValueError(f"execution manifest method/protocol mismatch: {execution_id}")
    expected_upstream, expected_checkpoint = METHOD_IDENTITIES[method]
    upstream = manifest.get("upstream")
    upstreams = _json(UPSTREAM_LOCK).get("upstreams")
    if not isinstance(upstream, dict) or not isinstance(upstreams, dict):
        raise ValueError(f"execution manifest upstream mismatch: {execution_id}")
    upstream_lock = upstreams.get(expected_upstream)
    if (upstream.get("name") != expected_upstream or not isinstance(upstream_lock, dict)
            or upstream.get("commit") != upstream_lock.get("commit")):
        raise ValueError(f"execution manifest upstream mismatch: {execution_id}")
    checkpoint = manifest.get("checkpoint")
    checkpoints = _json(CHECKPOINT_LOCK).get("checkpoints")
    if not isinstance(checkpoint, dict) or not isinstance(checkpoints, dict):
        raise ValueError(f"execution manifest checkpoint mismatch: {execution_id}")
    checkpoint_lock = checkpoints.get(expected_checkpoint)
    if (checkpoint.get("id") != expected_checkpoint or not isinstance(checkpoint_lock, dict) or checkpoint_lock.get("status") != "READY"
            or not isinstance(checkpoint_lock.get("sha256"), str) or checkpoint.get("sha256") != checkpoint_lock["sha256"]):
        raise ValueError(f"execution manifest checkpoint mismatch: {execution_id}")
    if manifest.get("input_sha256") != input_sha256 or manifest.get("output_prediction_sha256") != prediction_sha256:
        raise ValueError(f"execution manifest input/output mismatch: {execution_id}")
    return {"id": execution_id, "sha256": locked_sha256}


def _bundle(data: bytes, values: dict[str, np.ndarray], prediction: np.ndarray, metadata: dict[str, Any]) -> dict[str, Any]:
    required = {"gt_disparity", "gt_valid", "protocol_mask", "keyframe_mask", "adapter_support"}
    with np.load(io.BytesIO(data), allow_pickle=False) as loaded:
        if set(loaded.files) != required:
            raise ValueError(f"evaluation NPZ keys must be {sorted(required)}")
        arrays = {key: loaded[key] for key in required}
    expected = values["raw_disparity"].shape
    for key in ("gt_disparity", "gt_valid", "protocol_mask", "adapter_support"):
        if arrays[key].shape != expected:
            raise ValueError(f"{key} must match {expected}")
    if arrays["keyframe_mask"].shape != (expected[0],):
        raise ValueError("keyframe_mask must be [T]")
    if arrays["gt_disparity"].dtype != np.float32 or not np.isfinite(arrays["gt_disparity"]).all() or np.any(arrays["gt_disparity"][arrays["gt_valid"].astype(bool)] <= 0):
        raise ValueError("gt_disparity must be finite positive float32 on gt_valid")
    if arrays["gt_valid"].dtype != np.bool_ or arrays["protocol_mask"].dtype != np.bool_ or arrays["adapter_support"].dtype != np.bool_ or arrays["keyframe_mask"].dtype != np.bool_:
        raise ValueError("evaluation masks must be bool")
    for key in ("dataset", "split", "backbone", "sequence_id", "protocol"):
        if not isinstance(metadata.get(key), str) or not metadata[key]:
            raise ValueError(f"evaluation JSON requires non-empty {key}")
    return {"raw_disparity": values["raw_disparity"][:, 0], "refined_disparity": prediction[:, 0],
            "gt_disparity": arrays["gt_disparity"][:, 0], "gt_valid": arrays["gt_valid"][:, 0],
            "protocol_mask": arrays["protocol_mask"][:, 0], "keyframe_mask": arrays["keyframe_mask"],
            "adapter_support": arrays["adapter_support"], "frame_ids": values["frame_ids"].tolist(),
            **metadata}


def package(source_root: Path, method: str, input_path: Path, prediction_path: Path, evaluation_id: str | None = None,
            execution_manifest_id: str | None = None, diagnostic_evaluation: Path | None = None, append: bool = False) -> Path:
    protocol = METHODS.get(method)
    if not protocol:
        raise ValueError(f"unknown method: {method}")
    values, input_meta = read_input(input_path)
    prediction, prediction_sha256 = read_output_snapshot(prediction_path, values, input_meta, method)
    prediction_meta = _json(prediction_path.with_suffix(".json"))
    if not isinstance(input_meta.get("rgb_input_sha256"), str) or not input_meta["rgb_input_sha256"]:
        raise ValueError("bridge input must declare an RGB snapshot hash")
    if (prediction_meta.get("source_input_sha256") != input_meta["input_sha256"]
            or prediction_meta.get("source_rgb_input_sha256") != input_meta["rgb_input_sha256"]
            or prediction_meta.get("frame_ids") != values["frame_ids"].tolist()):
        raise ValueError("prediction provenance does not bind the canonical bridge input")
    if (evaluation_id is None) == (diagnostic_evaluation is None):
        raise ValueError("provide exactly one allowlisted or diagnostic evaluation artifact")
    evaluation_data, evaluation_meta, evaluation_lock = (_evaluation_artifact(evaluation_id, input_meta["input_sha256"])
                                                           if evaluation_id else _diagnostic_evaluation(diagnostic_evaluation, input_meta["input_sha256"]))
    publication = evaluation_lock.get("publication")
    if publication not in {"TEST_ONLY", "PUBLISHABLE"}:
        raise ValueError(f"evaluation artifact publication invalid: {evaluation_id}")
    if publication == "PUBLISHABLE" and execution_manifest_id is None:
        raise ValueError(f"execution manifest required for publishable artifact: {evaluation_id}")
    execution = (_execution_manifest(execution_manifest_id, method, input_meta["input_sha256"], prediction_sha256)
                 if execution_manifest_id is not None else {"id": None, "sha256": None})
    if evaluation_meta.get("frame_ids") != values["frame_ids"].tolist():
        raise ValueError("evaluation JSON frame_ids must exactly match bridge input")
    bundle = _bundle(evaluation_data, values, prediction, evaluation_meta)
    if bundle["dataset"] != "SCARED-C" or bundle["split"] not in {"d2", "d7"}:
        raise ValueError("only SCARED-C D2/D7 source runs are packageable")
    dataset = f"scared-{bundle['split']}"
    run = source_root / dataset / method
    if run.exists() != append:
        raise FileExistsError(f"refusing existing run: {run}")
    report = evaluate_scared_bundle(bundle)
    frozen = H4 / "model_design"
    declarations = {name: {"path": str(path), "sha256": sha256(path)} for name, path in {
        "unified_metrics": frozen / "metrics/unified_metrics.py", "evaluator": frozen / "comparison/definitive_evaluation.py", "comparison_driver": frozen / "comparison/run_comparison.py"}.items()}
    if not append:
        run.mkdir(parents=True)
    report_path = run / "reports" / str(bundle["backbone"]) / f"{bundle['sequence_id']}.json"
    if report_path.exists():
        raise FileExistsError(f"refusing existing sequence report: {report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True); report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    applicability = {"dataset": "SCARED-C", "unified_gt_metric_families": "APPLICABLE", "adapter_support": "DIAGNOSTIC_ONLY", "flow_referenced_temporal_metrics": "NOT_APPLICABLE", "temporal_access": protocol["causality"]}
    (run / "applicability.json").write_text(json.dumps(applicability, indent=2, sort_keys=True) + "\n")
    binding = {"input_sha256": input_meta["input_sha256"], "source_input_sha256": prediction_meta["source_input_sha256"],
               "rgb_input_sha256": input_meta["rgb_input_sha256"], "source_rgb_input_sha256": prediction_meta["source_rgb_input_sha256"],
               "frame_ids": values["frame_ids"].tolist(), "prediction_sha256": prediction_sha256,
               "evaluation_artifact_id": evaluation_id or f"sha256:{evaluation_lock['npz_sha256']}", "evaluation_npz_sha256": evaluation_lock["npz_sha256"],
               "evaluation_json_sha256": evaluation_lock["json_sha256"], "prediction_provenance": prediction_meta}
    method_manifest = {"method": method, "causality": protocol["causality"], "future_frames_required": protocol["future_frames_required"], "online_or_h4": False,
                       "input_sha256": input_meta["input_sha256"], "source_input_sha256": prediction_meta["source_input_sha256"],
                       "rgb_input_sha256": input_meta["rgb_input_sha256"], "source_rgb_input_sha256": prediction_meta["source_rgb_input_sha256"],
                       "frame_ids": values["frame_ids"].tolist(), "prediction_sha256": prediction_sha256,
                       "publication": publication, "execution_manifest_id": execution["id"], "execution_manifest_sha256": execution["sha256"],
                       "evaluation_artifact_id": evaluation_id or f"sha256:{evaluation_lock['npz_sha256']}", "evaluation_npz_sha256": evaluation_lock["npz_sha256"],
                       "evaluation_json_sha256": evaluation_lock["json_sha256"], "prediction_provenance": prediction_meta}
    if append:
        existing = _json(run / "external_method.json")
        if any(existing.get(key) != method_manifest[key] for key in ("method", "causality", "future_frames_required", "online_or_h4", "publication")):
            raise ValueError("cannot append incompatible source run")
        bindings = dict(existing.get("sequence_bindings", {}))
        bindings[bundle["sequence_id"]] = binding
        method_manifest = existing | {"sequence_bindings": bindings}
    else:
        method_manifest["sequence_bindings"] = {bundle["sequence_id"]: binding}
    (run / "external_method.json").write_text(json.dumps(method_manifest, indent=2, sort_keys=True) + "\n")
    source_code = Path(__file__).resolve()
    manifest = {"project": "ARGOS v2", "status": "COMPLETE", "dataset": dataset, "backbones": [bundle["backbone"]], "dense_predictions_written": False,
                "no_gt_in_adapter": True, "module_provenance": {"module": method, "code": str(source_code), "code_sha256": sha256(source_code), "external_method": "external_method.json"},
                "publication": publication, "execution_manifest_id": execution["id"], "execution_manifest_sha256": execution["sha256"],
                **declarations, "external_method": method_manifest,
                "outputs": sorted(str(path.relative_to(run)) for path in run.rglob("*") if path.is_file())}
    (run / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True); parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--input", type=Path, required=True); parser.add_argument("--prediction", type=Path, required=True); parser.add_argument("--evaluation", help="allowlisted evaluation artifact ID")
    parser.add_argument("--diagnostic-evaluation", type=Path, help="self-hashed TEST_ONLY smoke artifact under external_comparison/results")
    parser.add_argument("--execution-manifest", help="allowlisted execution attestation ID (required for PUBLISHABLE evaluation artifacts)")
    parser.add_argument("--append", action="store_true", help="append a distinct sequence report to a TEST_ONLY aggregate source run")
    args = parser.parse_args()
    print(package(args.source_root, args.method, args.input, args.prediction, args.evaluation, args.execution_manifest, args.diagnostic_evaluation, args.append))


if __name__ == "__main__":
    main()
