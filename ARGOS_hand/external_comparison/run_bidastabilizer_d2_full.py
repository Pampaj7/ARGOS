"""Resumable, TEST_ONLY full-D2 BiDAStabilizer execution on physical GPU 1."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "bidastabilizer_raftstereo_robust"
FINAL = RESULTS / "d2_full"
INCOMPLETE = RESULTS / "d2_full.incomplete"
SEQUENCES = ("dataset_2_keyframe_2", "dataset_2_keyframe_3", "dataset_2_keyframe_4")
PYTHON = Path("/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python")

sys.path.insert(0, str(ROOT))
from bridge import read_input, read_output_snapshot  # noqa: E402
from package_source_run import _diagnostic_evaluation, sha256  # noqa: E402


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _run(command: list[str], env: dict[str, str]) -> None:
    subprocess.run(command, check=True, cwd=ROOT, env=env)


def _sequence_complete(path: Path) -> bool:
    required = ("seed.npz", "raw.npz", "refined.npz", "evaluation.npz", "sequence_manifest.json")
    if not all((path / name).is_file() for name in required):
        return False
    values, meta = read_input(path / "raw.npz")
    read_output_snapshot(path / "refined.npz", values, meta, "bidastabilizer_bidirectional_offline")
    _diagnostic_evaluation(path / "evaluation.npz", meta["input_sha256"])
    return True


def _sequence_manifest(path: Path, sequence: str, seconds: float) -> dict[str, Any]:
    values, meta = read_input(path / "raw.npz")
    _, prediction_sha256 = read_output_snapshot(path / "refined.npz", values, meta, "bidastabilizer_bidirectional_offline")
    evaluation = path / "evaluation.npz"
    return {"sequence_id": sequence, "frames": int(values["frame_ids"].shape[0]), "seconds": seconds,
            "input_sha256": meta["input_sha256"], "rgb_input_sha256": meta["rgb_input_sha256"],
            "frame_ids_sha256": hashlib.sha256("\n".join(values["frame_ids"].tolist()).encode()).hexdigest(),
            "raw_npz_sha256": sha256(path / "raw.npz"), "refined_prediction_sha256": prediction_sha256,
            "evaluation_npz_sha256": sha256(evaluation), "runtime": _json(path / "raw.json")["runtime"]}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def validate_compiled_manifest(compiled: Path) -> None:
    """Reject a compiled result retaining staging paths after publication."""
    manifest = _json(compiled / "run_manifest.json")
    encoded = json.dumps(manifest, sort_keys=True)
    if ".incomplete" in encoded:
        raise ValueError("compiled manifest retains an incomplete path")

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(key, str) and key.startswith("/") and not Path(key).exists():
                    raise ValueError(f"compiled manifest path does not resolve: {key}")
                if key == "path" and isinstance(item, str):
                    path = Path(item)
                    path = path if path.is_absolute() else ROOT / path
                    if not path.exists():
                        raise ValueError(f"compiled manifest path does not resolve: {item}")
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, str) and value.startswith("/") and not Path(value).exists():
            raise ValueError(f"compiled manifest path does not resolve: {value}")

    visit(manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=PYTHON)
    args = parser.parse_args()
    if FINAL.exists():
        raise FileExistsError(f"refusing to overwrite final full-D2 result: {FINAL}")
    INCOMPLETE.mkdir(parents=True, exist_ok=True)
    env = os.environ | {"ARGOS_EXTERNAL_CUDA_VISIBLE_DEVICES": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    state = {"status": "RUNNING", "publication": "TEST_ONLY", "purpose": "D2_FULL_DIAGNOSTIC",
             "physical_gpu": 1, "logical_device": "cuda:0", "sequences": {}}
    _atomic_json(INCOMPLETE / "state.json", state)

    equivalence = INCOMPLETE / "official_wrapper_equivalence.json"
    if not equivalence.exists():
        smoke_seed = RESULTS / "d2_kf2_smoke64" / "seed.npz"
        _run([str(args.python), str(ROOT / "workers/bidastabilizer.py"), "--input", str(smoke_seed),
              "--equivalence", str(equivalence)], env)
    equivalence_data = _json(equivalence)
    if not (equivalence_data["raw"]["allclose"] and equivalence_data["refined"]["allclose"]):
        raise RuntimeError("official wrapper equivalence is not proven")

    for sequence in SEQUENCES:
        destination = INCOMPLETE / sequence
        if _sequence_complete(destination):
            state["sequences"][sequence] = _json(destination / "sequence_manifest.json") | {"status": "RESUMED"}
            _atomic_json(INCOMPLETE / "state.json", state)
            continue
        stage = INCOMPLETE / f".{sequence}.stage"
        if stage.exists():
            raise FileExistsError(f"incomplete sequence staging requires inspection: {stage}")
        stage.mkdir()
        started = time.monotonic()
        _run([sys.executable, str(ROOT / "export_scared_d2_smoke.py"), "--full", "--sequence", sequence,
              "--seed", str(stage / "seed.npz")], env)
        _run([sys.executable, str(ROOT / "run_external_evaluation.py"), "--method", "bidastabilizer_bidirectional_offline",
              "--input", str(stage / "seed.npz"), "--raw-output", str(stage / "raw.npz"), "--output", str(stage / "refined.npz"),
              "--python", str(args.python), "--purpose", "D2_FULL_DIAGNOSTIC"], env)
        _run([sys.executable, str(ROOT / "export_scared_d2_smoke.py"), "--full", "--sequence", sequence,
              "--bridge", str(stage / "raw.npz"), "--evaluation", str(stage / "evaluation.npz")], env)
        manifest = _sequence_manifest(stage, sequence, time.monotonic() - started)
        _atomic_json(stage / "sequence_manifest.json", manifest)
        os.replace(stage, destination)
        state["sequences"][sequence] = manifest | {"status": "COMPLETE"}
        _atomic_json(INCOMPLETE / "state.json", state)

    source = INCOMPLETE / "source_runs"
    run = source / "scared-d2" / "bidastabilizer_bidirectional_offline"
    existing = set(_json(run / "external_method.json").get("sequence_bindings", {})) if run.exists() else set()
    for sequence in SEQUENCES:
        if sequence in existing:
            continue
        path = INCOMPLETE / sequence
        command = [sys.executable, str(ROOT / "package_source_run.py"), "--source-root", str(source),
                   "--method", "bidastabilizer_bidirectional_offline", "--input", str(path / "raw.npz"),
                   "--prediction", str(path / "refined.npz"), "--diagnostic-evaluation", str(path / "evaluation.npz")]
        if run.exists():
            command.append("--append")
        _run(command, env)

    reports = sorted((run / "reports" / "RAFTStereo robust").glob("*.json"))
    if len(reports) != len(SEQUENCES):
        raise RuntimeError("aggregate source run is incomplete")
    manifest = {"status": "COMPLETE", "publication": "TEST_ONLY", "purpose": "D2_FULL_DIAGNOSTIC",
                "method": "bidastabilizer_bidirectional_offline", "physical_gpu": 1, "logical_device": "cuda:0",
                "sequences": [_json(INCOMPLETE / sequence / "sequence_manifest.json") for sequence in SEQUENCES],
                "official_wrapper_equivalence": {"artifact_id": equivalence.name, "sha256": sha256(equivalence),
                                                  "worker_code_sha256": equivalence_data["worker_code_sha256"],
                                                  "raw_max_abs": equivalence_data["raw"]["max_abs"], "refined_max_abs": equivalence_data["refined"]["max_abs"]},
                "source_run": {"artifact_id": "source_runs/scared-d2/bidastabilizer_bidirectional_offline",
                               "run_manifest_sha256": sha256(run / "run_manifest.json"), "report_sha256": {path.name: sha256(path) for path in reports}}}
    _atomic_json(INCOMPLETE / "state.json", state | {"status": "COMPLETE"})
    os.replace(INCOMPLETE, FINAL)
    compiled = FINAL / "compiled_test_only"
    _run([sys.executable, str(ROOT.parent / "original_h4/model_design/comparison/run_definitive_evaluation.py"),
          "--compile-from", str(FINAL / "source_runs"), "--datasets", "scared-d2", "--output", str(compiled)], env)
    validate_compiled_manifest(compiled)
    manifest["frozen_compile"] = {"artifact_id": "compiled_test_only", "run_manifest_sha256": sha256(compiled / "run_manifest.json")}
    _atomic_json(FINAL / "full_manifest.json", manifest)
    print(FINAL)


if __name__ == "__main__":
    main()
