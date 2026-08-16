#!/usr/bin/env python3
"""Write-once frozen launcher for the canonical-H4 external OOD evaluation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ARGOS = ROOT.parents[1]
RESULTS = ROOT.parent / "results/definitive_temporal_evaluation_csv"
PROTOCOL = RESULTS / "protocol"
FREEZE = PROTOCOL / "canonical_h4_ood_v7_freeze.json"
INVENTORY = PROTOCOL / "canonical_h4_ood_v7_input_inventory.json"
OUTPUT = RESULTS / "canonical_h4_ood_v7"
ATTESTATION = OUTPUT / "external_ood_attestation.json"
MODULE = "model_design.comparison.canonical_h4:factory"
D4D_BACKBONES = ("RAFT-Stereo", "StereoAnywhere")
DRENDS_RECORDING = "Vid14_Pancreas_High"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def entry(path: Path) -> dict[str, str]:
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": sha256(path)}


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, prefix=f".{path.name}.") as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        if temporary:
            temporary.unlink(missing_ok=True)
        raise


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def verify_entries(values: Mapping[str, Any], *, label: str) -> None:
    if not values:
        raise RuntimeError(f"empty {label}")
    for name, item in values.items():
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str):
            raise RuntimeError(f"invalid {label} entry: {name}")
        path = Path(item["path"])
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise RuntimeError(f"{label} hash mismatch: {name}")


def _source_inputs() -> dict[str, dict[str, str]]:
    frozen = ARGOS / "ARGOS_FREEZED"
    return {name: entry(path) for name, path in {
        "launcher": Path(__file__),
        "run_definitive_evaluation": ROOT / "model_design/comparison/run_definitive_evaluation.py",
        "definitive_evaluation": ROOT / "model_design/comparison/definitive_evaluation.py",
        "comparison_driver": ROOT / "model_design/comparison/run_comparison.py",
        "paper_table_compiler": ROOT / "model_design/comparison/build_paper_table.py",
        "drends_evaluation": ROOT / "model_design/comparison/drends_evaluation.py",
        "canonical_h4": ROOT / "model_design/comparison/canonical_h4.py",
        "unified_metrics": ROOT / "model_design/metrics/unified_metrics.py",
        "canonical_provenance": ROOT / "scripts/canonical_h4_provenance.py",
        "frozen_codd_provenance": frozen / "experiments/02_massive_training/scripts/provenance/codd_style_fusion.py",
        "bida_pull_warp": frozen / "src/argos_freezed/alignment/bida_pull_warp.py",
        "sea_raft_adapter": frozen / "src/argos_freezed/alignment/sea_raft_adapter.py",
        "d4d_source_reader": ARGOS / "ARGOS-V2/scripts/build_multidomain_backbone_cache.py",
        "d4d_keyframe_gt": ARGOS / "scripts/temporal_refinement/ood/d4d/d4d_keyframe_gt.py",
        "d4d_flow_adapter": ROOT / "model_design/external_components/bidavideo.py",
        "raft_stereo_loader": ARGOS / "scripts/scared/eval_scared_external_native.py",
        "canonical_checkpoint": ROOT / "model_design/checkpoints/codd_style_h4_best_validation.pt",
        "canonical_policy": ROOT / "model_design/checkpoints/codd_style_h4_policy.json",
        "sea_raft_checkpoint": ARGOS / "external/bidavideo/third_party/SEA-RAFT/models/Tartan-C-T-TSKH-spring540x960-S.pth",
        "raft_stereo_checkpoint": ARGOS / "external/frame_stereo_repos/RAFT-Stereo/models/raftstereo-middlebury.pth",
    }.items()}


def _tree_entry(root: Path) -> dict[str, Any]:
    files = sorted(path for path in root.rglob("*.py") if path.is_file())
    entries = {str(path.relative_to(root)): entry(path) for path in files}
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"root": str(root.resolve()), "sha256": digest, "files": entries}


def _d4d_inventory() -> dict[str, Any]:
    cache_root = ARGOS / "ARGOS-V2/cache_multidomain_backbones"
    context_root = ARGOS / "results/03_temporal_refinement/ood/d4d_s2m2_zero_shot"
    context_path, index_path = context_root / "context_manifest.csv", context_root / "d4d_index.csv"
    gt_manifest = ARGOS / "dataset/D4D/processed/keyframe_stereo_gt_curated/manifests/valid_and_warning_manifest.csv"
    contexts = {row["anchor_id"]: row for row in csv.DictReader(context_path.open(encoding="utf-8"))}
    windows = [row for row in csv.DictReader(index_path.open(encoding="utf-8")) if row["sequence_id"] in contexts]
    manifests: dict[str, dict[str, dict[str, str]]] = {}
    caches: dict[str, dict[str, Any]] = {}
    for backbone in D4D_BACKBONES:
        base = cache_root / backbone / "D4D"
        manifest = {row["frame_id"]: row for row in csv.DictReader((base / "frame_manifest.csv").open(encoding="utf-8"))}
        metadata = read_json(base / "metadata.json")
        if not metadata.get("completion_status") or metadata.get("disparity_convention") != "positive_left_disparity":
            raise RuntimeError(f"incompatible D4D cache: {base}")
        manifests[backbone] = manifest
        caches[backbone] = {name: entry(base / name) for name in (".complete", "metadata.json", "frame_manifest.csv", "disparity.npy", "valid_mask.npy", "frame_ids.npy")}
    selected: list[dict[str, Any]] = []
    images: dict[str, dict[str, str]] = {}
    cameras: dict[str, dict[str, str]] = {}
    for window in windows:
        context = contexts[window["sequence_id"]]
        stems = context["context_stems"].split(";")
        ids = [f"{window['specimen']}__{window['session']}__{stem}" for stem in stems]
        if len(ids) != 4:
            continue
        pairs: list[dict[str, str]] = []
        complete = True
        for frame_id in ids:
            per_backbone = [manifests[backbone].get(frame_id) for backbone in D4D_BACKBONES]
            if any(item is None for item in per_backbone):
                complete = False; break
            first = per_backbone[0]
            if any(item["left_path"] != first["left_path"] or item["right_path"] != first["right_path"] for item in per_backbone[1:]):
                raise RuntimeError(f"D4D source-pair disagreement: {frame_id}")
            left, right = Path(first["left_path"]), Path(first["right_path"])
            if not left.is_file() or not right.is_file():
                complete = False; break
            for image in (left, right):
                key = str(image.resolve())
                images.setdefault(key, entry(image))
            pairs.append({"frame_id": frame_id, "left_path": str(left.resolve()), "right_path": str(right.resolve())})
        if complete:
            camera_dir = ARGOS / "dataset/D4D/raw/extracted" / window["specimen"] / window["session"] / "camera_info"
            for name in ("left.yaml", "right.yaml"):
                cameras.setdefault(str((camera_dir / name).resolve()), entry(camera_dir / name))
            selected.append({"sequence_id": window["sequence_id"], "specimen": window["specimen"], "session": window["session"], "frames": pairs})
    if len(contexts) != 156 or len(windows) != 156 or len(selected) != 84:
        raise RuntimeError(f"unexpected D4D scope: contexts={len(contexts)} windows={len(windows)} eligible={len(selected)}")
    return {"context_manifest": entry(context_path), "index": entry(index_path), "gt_manifest": entry(gt_manifest), "camera_yamls": cameras, "backbones": list(D4D_BACKBONES), "caches": caches,
            "input_context_count": len(windows), "eligible_context_count": len(selected), "contexts": selected, "images": images,
            "expected_diagnostics": 504, "metric_scope": "no_reference_prediction_space"}


def _drends_inventory() -> dict[str, Any]:
    from model_design.comparison.drends_evaluation import load_drends_records

    records, info = load_drends_records(DRENDS_RECORDING)
    manifest = read_json(Path(info["manifest"]))
    input_frames = manifest.get("sequence", {}).get("frames", [])
    excluded = info["excluded_timing_frame_ids"]
    if len(input_frames) != 1500 or len(records) != 1499 or excluded != ["00000"]:
        raise RuntimeError(f"unexpected DRENDS scope: input={len(input_frames)} eligible={len(records)} excluded={excluded}")
    files: dict[str, dict[str, str]] = {}
    frames: list[dict[str, Any]] = []
    for record in records:
        paths = {}
        for name in ("rect_left", "rect_right", "depth_left", "mask_left"):
            path = record[f"_{name}"]
            key = str(path.resolve())
            files.setdefault(key, entry(path)); paths[name] = key
        frames.append({"frame_id": record["frame_id"], "files": paths})
    excluded_frame = next(frame for frame in input_frames if frame.get("frame_id") == "00000")
    excluded_files = {name: entry(ARGOS / path) for name, path in ((name, excluded_frame[name]) for name in ("rect_left", "rect_right", "depth_left", "mask_left"))}
    return {"recording": DRENDS_RECORDING, "manifest": entry(Path(info["manifest"])), "quality_report": entry(Path(info["quality_report"])),
            "input_frame_count": len(input_frames), "eligible_frame_count": len(records), "excluded_timing_frame_ids": excluded,
            "frames": frames, "files": files, "excluded_timing_availability_files": excluded_files, "metric_scope": "tof_reference_nonindependent"}


def inventory_payload() -> dict[str, Any]:
    sea_root = ARGOS / "external/bidavideo/third_party/SEA-RAFT/core"
    return {"project": "ARGOS v2", "inventory_version": 7, "module": MODULE, "sea_raft_import_tree": _tree_entry(sea_root),
            "d4d": _d4d_inventory(), "drends": _drends_inventory(),
            "servct": {"backbones": list(D4D_BACKBONES), "temporal_h4_evaluation": "NOT_APPLICABLE",
                       "reason": "static stereo pairs have no temporal adjacency", "consumed_dataset_files": 0}}


def validate_inventory_scope(value: Mapping[str, Any]) -> None:
    d4d, drends, servct = value.get("d4d"), value.get("drends"), value.get("servct")
    if not isinstance(d4d, Mapping) or d4d.get("backbones") != list(D4D_BACKBONES) or d4d.get("input_context_count") != 156 or d4d.get("eligible_context_count") != 84 or d4d.get("expected_diagnostics") != 504 or not isinstance(d4d.get("contexts"), list) or len(d4d["contexts"]) != 84:
        raise RuntimeError("invalid D4D frozen scope")
    if not isinstance(drends, Mapping) or drends.get("recording") != DRENDS_RECORDING or drends.get("input_frame_count") != 1500 or drends.get("eligible_frame_count") != 1499 or drends.get("excluded_timing_frame_ids") != ["00000"] or not isinstance(drends.get("frames"), list) or len(drends["frames"]) != 1499:
        raise RuntimeError("invalid DRENDS frozen scope")
    if not isinstance(servct, Mapping) or servct.get("backbones") != list(D4D_BACKBONES) or servct.get("temporal_h4_evaluation") != "NOT_APPLICABLE" or servct.get("consumed_dataset_files") != 0:
        raise RuntimeError("invalid SERV-CT frozen scope")


def verify_inventory(value: Mapping[str, Any]) -> None:
    if value.get("project") != "ARGOS v2" or value.get("inventory_version") != 7 or value.get("module") != MODULE:
        raise RuntimeError("invalid v7 input inventory")
    validate_inventory_scope(value)
    d4d, drends = value["d4d"], value["drends"]
    verify_entries({"context_manifest": d4d["context_manifest"], "index": d4d["index"], "gt_manifest": d4d["gt_manifest"]}, label="D4D manifest")
    verify_entries(d4d.get("camera_yamls", {}), label="D4D camera YAML")
    for backbone in D4D_BACKBONES:
        verify_entries(d4d["caches"].get(backbone, {}), label=f"D4D {backbone} cache")
    verify_entries(d4d.get("images", {}), label="D4D source image")
    verify_entries({"manifest": drends["manifest"], "quality_report": drends["quality_report"]}, label="DRENDS manifest")
    verify_entries(drends.get("files", {}), label="DRENDS consumed file")
    verify_entries(drends.get("excluded_timing_availability_files", {}), label="DRENDS excluded-frame availability")
    tree = value.get("sea_raft_import_tree")
    if not isinstance(tree, Mapping) or not isinstance(tree.get("root"), str) or not isinstance(tree.get("files"), Mapping):
        raise RuntimeError("invalid SEA-RAFT source tree")
    verify_entries(tree["files"], label="SEA-RAFT source")
    if tree.get("sha256") != hashlib.sha256(json.dumps(tree["files"], sort_keys=True, separators=(",", ":")).encode()).hexdigest():
        raise RuntimeError("SEA-RAFT source tree digest mismatch")


def freeze_payload(inventory_sha256: str | None = None) -> dict[str, Any]:
    return {"project": "ARGOS v2", "freeze_version": 7, "freeze_id": "canonical_h4_ood_v7", "status": "FROZEN_PRE_RUN", "write_once": True,
            "module": MODULE, "datasets": ["d4d", "servct", "drends"], "immutable_sources_and_checkpoints": _source_inputs(),
            "input_inventory": {"path": str(INVENTORY.resolve()), "sha256": inventory_sha256 or sha256(INVENTORY)}, "output": str(OUTPUT.resolve()),
            "scope": {"seen_backbones": list(D4D_BACKBONES), "joint_unseen_backbone_and_ood": "UNAVAILABLE", "max_frames": None, "smoke": False,
                      "dense_predictions_written": False, "d4d_diagnostics": 504, "drends_eligible_frames": 1499, "servct": "NOT_APPLICABLE"}}


def _freeze_scope() -> dict[str, Any]:
    return {"seen_backbones": list(D4D_BACKBONES), "joint_unseen_backbone_and_ood": "UNAVAILABLE", "max_frames": None, "smoke": False,
            "dense_predictions_written": False, "d4d_diagnostics": 504, "drends_eligible_frames": 1499, "servct": "NOT_APPLICABLE"}


def write_freeze() -> tuple[Path, Path]:
    stage = PROTOCOL / ".canonical_h4_ood_v7_pair"
    if FREEZE.exists() and INVENTORY.exists():
        verify_frozen_inputs(); return FREEZE, INVENTORY
    if FREEZE.exists():
        raise RuntimeError("unrecoverable freeze-without-inventory publication")
    if not INVENTORY.exists():
        stage.mkdir(parents=True, exist_ok=True)
        staged_inventory, staged_freeze = stage / INVENTORY.name, stage / FREEZE.name
        if not staged_inventory.exists():
            if any(stage.iterdir()):
                raise RuntimeError("unexpected staged v3 freeze artifact")
            payload = inventory_payload(); verify_inventory(payload); atomic_json(staged_inventory, payload)
        else:
            verify_inventory(read_json(staged_inventory))
        if not staged_freeze.exists():
            atomic_json(staged_freeze, freeze_payload(sha256(staged_inventory)))
        if not staged_inventory.is_file() or not staged_freeze.is_file():
            raise RuntimeError("incomplete staged v3 freeze pair")
        os.replace(staged_inventory, INVENTORY)
    staged_freeze = stage / FREEZE.name
    if not staged_freeze.is_file():
        raise RuntimeError("missing staged v3 freeze for inventory orphan recovery")
    os.replace(staged_freeze, FREEZE)
    stage.rmdir()
    verify_frozen_inputs()
    return FREEZE, INVENTORY


def verify_frozen_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    freeze = read_json(FREEZE)
    if freeze.get("project") != "ARGOS v2" or freeze.get("freeze_version") != 7 or freeze.get("freeze_id") != "canonical_h4_ood_v7" or freeze.get("status") != "FROZEN_PRE_RUN" or freeze.get("module") != MODULE:
        raise RuntimeError("invalid v7 OOD freeze")
    if freeze.get("datasets") != ["d4d", "servct", "drends"] or freeze.get("scope") != _freeze_scope():
        raise RuntimeError("v3 OOD freeze scope changed")
    verify_entries(freeze.get("immutable_sources_and_checkpoints", {}), label="frozen source")
    inventory_entry = freeze.get("input_inventory")
    if not isinstance(inventory_entry, Mapping) or inventory_entry.get("path") != str(INVENTORY.resolve()) or inventory_entry.get("sha256") != sha256(INVENTORY):
        raise RuntimeError("v3 input inventory is not pinned by freeze")
    inventory = read_json(INVENTORY); verify_inventory(inventory)
    return freeze, inventory


def _run_path(output: Path, dataset: str) -> Path:
    return output / "runs" / dataset / MODULE.replace(":", "__").replace(".", "_")


def _verify_output_hashes(root: Path, manifest: Mapping[str, Any]) -> None:
    outputs = manifest.get("outputs"); hashes = manifest.get("output_hashes")
    if not isinstance(outputs, list) or not isinstance(hashes, Mapping) or sorted(outputs) != sorted(hashes):
        raise RuntimeError(f"invalid compiled output hashes: {root}")
    for name in outputs:
        path = root / name
        if not path.is_file() or hashes[name] != sha256(path):
            raise RuntimeError(f"compiled output hash mismatch: {path}")


def _reject_undeclared_files(root: Path, manifest: Mapping[str, Any]) -> None:
    declared = set(manifest.get("outputs", [])) | {"run_manifest.json"}
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    if actual != declared:
        raise RuntimeError("compiled output has undeclared or missing files")


def validate_completed_output(output: Path) -> dict[str, Any]:
    from model_design.comparison.run_definitive_evaluation import verify_run

    compiled = read_json(output / "run_manifest.json")
    if compiled.get("project") != "ARGOS v2" or compiled.get("status") != "COMPLETE" or compiled.get("datasets") != ["d4d", "servct", "drends"] or compiled.get("dense_predictions_written") is not False:
        raise RuntimeError("compiled OOD output is incomplete or wrong scope")
    _verify_output_hashes(output, compiled)
    _reject_undeclared_files(output, compiled)
    runs = {dataset: (_run_path(output, dataset), verify_run(_run_path(output, dataset), dataset)) for dataset in ("d4d", "servct", "drends")}
    if any(manifest.get("dense_predictions_written") is not False for _, manifest in runs.values()):
        raise RuntimeError("dense predictions were written")
    d4d_path, d4d_manifest = runs["d4d"]
    with (d4d_path / "d4d_diagnostics.csv").open(newline="", encoding="utf-8") as stream:
        d4d_rows = list(csv.DictReader(stream))
    if len(d4d_rows) != 504 or {row.get("backbone") for row in d4d_rows} != set(D4D_BACKBONES):
        raise RuntimeError("D4D diagnostics do not cover the frozen 504-row scope")
    serv_path, serv_manifest = runs["servct"]
    serv = read_json(serv_path / "applicability.json")
    if serv.get("unified_gt_metric_families") != "NOT_APPLICABLE" or serv.get("numeric_gt_metrics") is not None:
        raise RuntimeError("SERV-CT must remain not applicable")
    drends_path, drends_manifest = runs["drends"]
    report = read_json(drends_path / "reports" / "RAFT-Stereo" / f"{DRENDS_RECORDING}.json")
    info = report.get("diagnostics", {})
    if info.get("frame_count") != 1499 or info.get("excluded_timing_frame_ids") != ["00000"]:
        raise RuntimeError("DRENDS report does not match frozen eligible frames")
    if any(path.suffix.lower() in {".npy", ".npz", ".pt", ".pth", ".pkl", ".pickle"} for path in output.rglob("*") if path.is_file()):
        raise RuntimeError("dense output artifact detected")
    return {"compiled": entry(output / "run_manifest.json"), "source_runs": {dataset: entry(path / "run_manifest.json") for dataset, (path, _) in runs.items()},
            "checks": {"d4d_diagnostics": len(d4d_rows), "servct_temporal_h4": "NOT_APPLICABLE", "drends_eligible_frames": info["frame_count"], "dense_predictions_written": False},
            "compiled_output_hashes": compiled["output_hashes"], "source_backbones": {dataset: manifest.get("backbones") for dataset, (_, manifest) in runs.items()}}


def finalize_child_output(child: Path, output: Path) -> None:
    """Remove the transient child path before its directory becomes durable."""
    from model_design.comparison.run_definitive_evaluation import atomic_text, finalize_output_hashes

    old, new = str(child.resolve()), str(output.resolve())
    for path in child.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".csv"}:
            text = path.read_text(encoding="utf-8")
            if old in text:
                atomic_text(path, text.replace(old, new))
    manifest_path = child / "run_manifest.json"
    manifest = read_json(manifest_path)
    consumed = manifest.get("consumed_input_hashes")
    if isinstance(consumed, Mapping):
        rewritten: dict[str, str] = {}
        for name in consumed:
            final_name = str(name)
            local_name = Path(str(name).replace(new, old))
            if not local_name.is_file():
                raise RuntimeError(f"vanished consumed source after child publication: {local_name}")
            rewritten[final_name] = sha256(local_name)
        atomic_json(manifest_path, manifest | {"consumed_input_hashes": rewritten})
    finalize_output_hashes(child)
    if any(old in path.read_text(encoding="utf-8") for path in child.rglob("*") if path.is_file() and path.suffix in {".json", ".csv"}):
        raise RuntimeError("transient child path remains in durable provenance")


def _final_paths(value: Any, child: Path, output: Path) -> Any:
    if isinstance(value, str):
        return value.replace(str(child.resolve()), str(output.resolve()))
    if isinstance(value, list):
        return [_final_paths(item, child, output) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _final_paths(item, child, output) for key, item in value.items()}
    return value


def write_child_attestation(child: Path, output: Path, evidence: Mapping[str, Any]) -> Path:
    path = child / ATTESTATION.name
    manifest = read_json(child / "run_manifest.json")
    if path.exists() or ATTESTATION.name in manifest.get("outputs", []) or ATTESTATION.name in manifest.get("output_hashes", {}):
        raise RuntimeError("attestation must not pre-exist")
    bound = {key: value for key, value in evidence.items() if key not in {"compiled", "compiled_output_hashes"}}
    value = {"project": "ARGOS v2", "attestation_version": 5, "status": "COMPLETE_EXTERNAL_OOD", "freeze": entry(FREEZE), "inventory": entry(INVENTORY),
             "output": str(output.resolve()), **_final_paths(bound, child, output)}
    atomic_json(path, value)
    if not path.is_file() or str(child.resolve()) in path.read_text(encoding="utf-8"):
        raise RuntimeError("invalid staged external attestation")
    from model_design.comparison.run_definitive_evaluation import finalize_output_hashes
    finalize_output_hashes(child)
    return path


def run(config: argparse.Namespace) -> Path:
    if config.output.exists():
        raise FileExistsError("refusing existing OOD output")
    freeze_before, inventory_before = verify_frozen_inputs()
    if str(config.output.resolve()) != freeze_before.get("output"):
        raise RuntimeError("OOD output differs from frozen output path")
    stage = Path(tempfile.mkdtemp(prefix=f".{config.output.name}.external-stage-", dir=config.output.parent))
    child = stage / "child_output"  # deliberately absent: child owns staged_directory publication.
    command = [sys.executable, str(ROOT / "model_design/comparison/run_definitive_evaluation.py"), "--datasets", "d4d", "servct", "drends", "--module", MODULE,
               "--output", str(child), "--device", config.device, "--external-backbones", *D4D_BACKBONES]
    subprocess.run(command, check=True)
    if not child.is_dir():
        raise RuntimeError("child did not publish its staged output")
    evidence = validate_completed_output(child)
    freeze_after, inventory_after = verify_frozen_inputs()
    if freeze_before != freeze_after or inventory_before != inventory_after:
        raise RuntimeError("frozen inputs changed during OOD evaluation")
    finalize_child_output(child, config.output)
    evidence = validate_completed_output(child)
    freeze_after, inventory_after = verify_frozen_inputs()
    if freeze_before != freeze_after or inventory_before != inventory_after:
        raise RuntimeError("frozen inputs changed during OOD finalization")
    write_child_attestation(child, config.output, evidence)
    validate_completed_output(child)
    freeze_after, inventory_after = verify_frozen_inputs()
    if freeze_before != freeze_after or inventory_before != inventory_after:
        raise RuntimeError("frozen inputs changed after staged attestation")
    if config.output.exists():
        raise FileExistsError("OOD output appeared during staged evaluation")
    os.rename(child, config.output)
    stage.rmdir()
    return config.output / ATTESTATION.name


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-freeze", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    config = arguments()
    if config.write_freeze:
        if config.output != OUTPUT:
            raise ValueError("freeze creation does not accept an output override")
        freeze, inventory = write_freeze(); print(f"{freeze}\n{inventory}")
    else:
        print(run(config))


if __name__ == "__main__":
    main()
