"""Resumable full-D2 direct official NVDS DPT diagnostic on physical GPU 1."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from run_nvds_dpt_d2_smoke import _validate_published

ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "results" / "nvds_dpt_bidirectional_offline" / "d2_full"
RUNNER = ROOT / "run_nvds_dpt_d2_smoke.py"
PYTHON = Path("/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python")
EXPECTED = {"dataset_2_keyframe_2": 1033, "dataset_2_keyframe_3": 1102, "dataset_2_keyframe_4": 2114}


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", mode="w", delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True); handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _manifest(path: Path, sequence: str) -> dict[str, object]:
    _validate_published(path)
    manifest = json.loads((path / "run_manifest.json").read_text())
    if (manifest.get("method") != "nvds_dpt_bidirectional_offline" or manifest["input"].get("sequence") != sequence
            or manifest["input"].get("frames") != EXPECTED[sequence] or manifest["output"].get("mix_uint16_files") != EXPECTED[sequence]
            or manifest.get("strict_loading") != {"gmflow": True, "nvds": True, "dpt": True}):
        raise RuntimeError(f"invalid completed sequence: {path}")
    return manifest


def _aggregate(manifests: dict[str, dict[str, object]]) -> dict[str, object]:
    kinds = ("initial", "forward", "backward", "mix")
    summary = {}
    for kind in kinds:
        rows = [manifest["diagnostic"]["opw_raw_forward_backward_mix"][kind] for manifest in manifests.values()]
        frames = sum(row["frames"] for row in rows)
        total = sum(row["total"] for row in rows)
        summary[kind] = {"total": total, "micro_mean": total / frames,
                         "macro_mean": sum(row["mean"] for row in rows) / len(rows), "frames": frames}
    return {"status": "COMPLETE", "publication": "TEST_ONLY", "method": "nvds_dpt_bidirectional_offline",
            "semantic": "NVDS (not NVDS+), monocular relative inverse-depth, noncausal; never H4/disparity evaluated",
            "sequences": {sequence: {"frames": manifest["input"]["frames"], "opw": manifest["diagnostic"]["opw_raw_forward_backward_mix"],
                                       "manifest_sha256": hashlib.sha256((RESULT / sequence / "run_manifest.json").read_bytes()).hexdigest(),
                                       "evidence_sha256": manifest["evidence"]["sha256"]} for sequence, manifest in manifests.items()},
            "total_frames": sum(manifest["input"]["frames"] for manifest in manifests.values()), "opw": summary,
            "nondeterminism_disclosure": "The pinned upstream enables cuDNN benchmark; direct reruns can differ slightly in floating-point OPW and normalized uint16 hashes. Results are diagnostics, not cross-run bitwise attestations."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=PYTHON)
    args = parser.parse_args()
    if os.environ.get("ARGOS_EXTERNAL_CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("set ARGOS_EXTERNAL_CUDA_VISIBLE_DEVICES=1; GPU 0 is forbidden")
    if (RESULT / "aggregate_diagnostic.json").exists():
        raise FileExistsError(f"refusing to overwrite completed D2 aggregate: {RESULT}")
    RESULT.mkdir(parents=True, exist_ok=True)
    state: dict[str, object] = {"status": "RUNNING", "physical_gpu": 1, "logical_device": "cuda:0", "sequences": {}}
    for sequence, frames in EXPECTED.items():
        destination = RESULT / sequence
        if destination.exists():
            manifest = _manifest(destination, sequence)
            state["sequences"][sequence] = {"status": "RESUMED", "frames": frames, "manifest_sha256": hashlib.sha256((destination / "run_manifest.json").read_bytes()).hexdigest()}
        else:
            command = [str(args.python), str(RUNNER), "--full", "--sequence", sequence, "--output", str(destination), "--python", str(args.python)]
            subprocess.run(command, check=True, cwd=ROOT, env=os.environ | {"ARGOS_EXTERNAL_CUDA_VISIBLE_DEVICES": "1", "PYTHONDONTWRITEBYTECODE": "1"})
            _manifest(destination, sequence)
            state["sequences"][sequence] = {"status": "COMPLETE", "frames": frames, "manifest_sha256": hashlib.sha256((destination / "run_manifest.json").read_bytes()).hexdigest()}
        _atomic_json(RESULT / "state.json", state)
    manifests = {sequence: _manifest(RESULT / sequence, sequence) for sequence in EXPECTED}
    aggregate = _aggregate(manifests)
    if aggregate["total_frames"] != sum(EXPECTED.values()):
        raise RuntimeError("D2 full frame total mismatch")
    _atomic_json(RESULT / "aggregate_diagnostic.json", aggregate)
    _atomic_json(RESULT / "state.json", state | {"status": "COMPLETE"})
    print(RESULT)


if __name__ == "__main__":
    main()
