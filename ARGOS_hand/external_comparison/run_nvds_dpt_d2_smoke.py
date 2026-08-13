"""Run the pinned official NVDS DPT bidirectional smoke without a disparity bridge."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "upstreams" / "nvds"
SMOKE_FINAL = ROOT / "results" / "nvds_dpt_bidirectional_offline" / "d2_kf2_smoke64"
ARGOS_PYTHON = Path("/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python")
SEQUENCES = ("dataset_2_keyframe_2", "dataset_2_keyframe_3", "dataset_2_keyframe_4")
SMOKE_FRAMES = 64


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _records_sha256(records: list[dict[str, object]]) -> str:
    return hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _tree_sha256(root: Path) -> str:
    records = [{"path": str(path.relative_to(root)), "sha256": sha256(path)} for path in sorted(root.rglob("*")) if path.is_file()]
    return _records_sha256(records)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _verify_upstream(repo: Path = UPSTREAM) -> dict[str, object]:
    locked = json.loads((ROOT / "upstreams.lock.json").read_text())["upstreams"]["nvds"]
    actual = {"origin": _git(repo, "remote", "get-url", "origin"), "commit": _git(repo, "rev-parse", "HEAD"),
              "status": _git(repo, "status", "--porcelain"), "license_sha256": sha256(repo / "LICENSE")}
    for key in ("origin", "commit", "license_sha256"):
        if actual[key] != locked[key]:
            raise RuntimeError(f"NVDS upstream {key} mismatch: {actual[key]!r} != {locked[key]!r}")
    if actual["status"]:
        raise RuntimeError("NVDS upstream is dirty; refusing execution")
    return locked | actual


def _locked_checkpoints() -> list[dict[str, object]]:
    lock = json.loads((ROOT / "checkpoints.lock.json").read_text())
    artifacts = lock["checkpoints"]["nvds_official_dpt_bidirectional"]["artifacts"]
    for artifact in artifacts:
        path = ROOT / artifact["path"]
        if not path.is_file() or path.stat().st_size != artifact["bytes"] or sha256(path) != artifact["sha256"]:
            raise RuntimeError(f"locked checkpoint verification failed: {path}")
        # Deliberately prove safe tensor-only deserialization before CUDA is selected.
        state = torch.load(path, map_location="cpu", weights_only=True)
        tensors = state.get("model", state) if isinstance(state, dict) else state
        if not isinstance(tensors, dict) or not tensors or not all(isinstance(v, torch.Tensor) for v in tensors.values()):
            raise RuntimeError(f"weights_only tensor contract failed: {path}")
    return artifacts


def _stage_frames(left: Path, sequence: str, limit: int | None) -> list[dict[str, object]]:
    sys.path[:0] = [str(ROOT.parent / "original_h4" / "scripts"), str(ROOT.parent / "original_h4")]
    from argos_v2.scared_c_data import load_sequence_info  # noqa: PLC0415

    info = load_sequence_info(sequence)
    ids = info.frame_ids if limit is None else info.frame_ids[:limit]
    if not ids or ids != sorted(ids) or (limit is not None and len(ids) != limit):
        raise RuntimeError(f"expected chronological frames for {sequence}")
    left.mkdir(parents=True)
    provenance = []
    for index, frame_id in enumerate(ids):
        source = info.seq_dir / "left" / f"{frame_id}.png"
        bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(source)
        target = left / f"frame_{index:06d}.png"
        if not cv2.imwrite(str(target), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0]):
            raise RuntimeError(f"cannot write {target}")
        provenance.append({"index": index, "frame_id": frame_id, "source": str(source), "source_sha256": sha256(source), "staged_sha256": sha256(target)})
    return provenance


def _mount(runtime: Path, artifacts: list[dict[str, object]]) -> None:
    shutil.copytree(UPSTREAM, runtime, symlinks=True, ignore=shutil.ignore_patterns("__pycache__"))
    paths = {Path(item["path"]).name: ROOT / item["path"] for item in artifacts}
    for destination, source in ((runtime / "NVDS_checkpoints" / "NVDS_Stabilizer.pth", paths["NVDS_Stabilizer.pth"]),
                                (runtime / "dpt" / "checkpoints" / "dpt_large-midas-2f21e586.pt", paths["dpt_large-midas-2f21e586.pt"]),
                                (runtime / "gmflow" / "checkpoints" / "gmflow_sintel-0c07dcb3.pth", paths["gmflow_sintel-0c07dcb3.pth"])):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.unlink(missing_ok=True)
        destination.symlink_to(source)


def _peak_gpu_mb(stop: bool = False) -> int | None:
    try:
        value = subprocess.check_output(["nvidia-smi", "--id=1", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip()
        return int(value)
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def _opw(report: Path) -> dict[str, dict[str, float | int]]:
    text = report.read_text()
    out: dict[str, dict[str, float | int]] = {}
    for label, total, mean, frames in re.findall(r"\*+([^*\n]+)\*+\s*\nall:\s*([^,\s]+)[,\s]+mean:\s*([^,\s]+)[,\s]+frames:\s*(\d+)", text):
        key = label.strip().lower()
        if key in {"initial", "forward", "backward", "mixing"}:
            out["mix" if key == "mixing" else key] = {"total": float(total), "mean": float(mean), "frames": int(frames)}
    if set(out) != {"initial", "forward", "backward", "mix"}:
        raise RuntimeError(f"incomplete official OPW report: {out}")
    return out


def _tail(path: Path, size: int = 4000) -> str:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        handle.seek(max(0, handle.tell() - size))
        return handle.read().decode(errors="replace")


def _validate_published(path: Path) -> None:
    manifest = json.loads((path / "run_manifest.json").read_text())
    evidence = path / manifest["evidence"]["path"]
    if sha256(evidence) != manifest["evidence"]["sha256"]:
        raise RuntimeError("published evidence hash mismatch")
    if manifest["output"]["mix_uint16_tree_sha256"] != _tree_sha256(path / "mix_uint16"):
        raise RuntimeError("published uint16 output tree hash mismatch")
    if manifest["output"]["official_stdout_sha256"] != sha256(path / "official_stdout.txt"):
        raise RuntimeError("published stdout hash mismatch")
    details = json.loads(evidence.read_text())
    official_opw = _opw(path / "official_stdout.txt")
    if details.get("opw") != official_opw or manifest.get("diagnostic", {}).get("opw_raw_forward_backward_mix") != official_opw:
        raise RuntimeError("published OPW does not match official stdout")
    if details["source_input_records_sha256"] != manifest["input"]["source_input_records_sha256"]:
        raise RuntimeError("evidence/source input binding mismatch")
    if details["output"]["mix_uint16_tree_sha256"] != manifest["output"]["mix_uint16_tree_sha256"]:
        raise RuntimeError("evidence/output binding mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=ARGOS_PYTHON)
    parser.add_argument("--previous", type=Path, help="recoverable previous smoke result for comparison")
    parser.add_argument("--sequence", choices=SEQUENCES, default="dataset_2_keyframe_2")
    parser.add_argument("--full", action="store_true", help="use every chronological frame of --sequence")
    parser.add_argument("--output", type=Path, help="atomic destination; defaults to the smoke result")
    args = parser.parse_args()
    final = args.output or SMOKE_FINAL
    frame_limit = None if args.full else SMOKE_FRAMES
    if final.exists():
        raise FileExistsError(f"refusing to overwrite: {final}")
    if args.full and args.previous:
        raise ValueError("--previous is smoke-only")
    if os.environ.get("ARGOS_EXTERNAL_CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("set ARGOS_EXTERNAL_CUDA_VISIBLE_DEVICES=1; GPU 0 is forbidden")
    upstream = _verify_upstream()
    artifacts = _locked_checkpoints()
    final.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(dir=final.parent, prefix=".nvds_dpt.") as temporary:
        temporary_path = Path(temporary)
        runtime, stage = temporary_path / "nvds", temporary_path / "result"
        _mount(runtime, artifacts)
        frames = _stage_frames(runtime / "demo_videos" / args.sequence / "left", args.sequence, frame_limit)
        frame_count = len(frames)
        # The copied upstream still calls torch.load without weights_only.  This local compatibility shim
        # enforces tensor-only loading without changing the pinned upstream checkout.
        shim = temporary_path / "sitecustomize.py"
        shim_source = "import torch\n_load=torch.load\ndef load(*a, **k):\n k.setdefault('weights_only', True); return _load(*a, **k)\ntorch.load=load\n"
        shim.write_text(shim_source)
        command = [str(args.python), "infer_NVDS_dpt_bi.py", "--base_dir", str(stage), "--vnum", args.sequence,
                   "--infer_w", "480", "--infer_h", "384", "--timesall", "2", "--clip_step", "1", "--strict_resume"]
        env = os.environ | {"CUDA_VISIBLE_DEVICES": "1", "PYTHONPATH": f"{temporary_path}{os.pathsep}{os.environ.get('PYTHONPATH', '')}", "PYTHONDONTWRITEBYTECODE": "1"}
        live_log = temporary_path / "official_stdout.txt"
        with live_log.open("w") as log_handle:
            process = subprocess.Popen(command, cwd=runtime, env=env, text=True, stdout=log_handle, stderr=subprocess.STDOUT)
            peak = 0
            while process.poll() is None:
                memory = _peak_gpu_mb()
                if memory is not None:
                    peak = max(peak, memory)
                time.sleep(0.2)
        if process.returncode:
            raise RuntimeError(f"official NVDS failed ({process.returncode}):\n{_tail(live_log)}")
        mix = sorted((stage / "mix" / "gray").glob("frame_*.png"))
        if len(mix) != frame_count:
            raise RuntimeError(f"expected {frame_count} official uint16 mix PNGs, got {len(mix)}")
        for path in mix:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None or image.dtype.name != "uint16" or image.shape != (384, 480):
                raise RuntimeError(f"invalid mix output: {path}")
        final_stage = temporary_path / "final"
        shutil.copytree(stage / "mix" / "gray", final_stage / "mix_uint16")
        shutil.copy2(live_log, final_stage / "official_stdout.txt")
        source_input_records_sha256 = _records_sha256(frames)
        output = {"mix_uint16_files": frame_count, "mix_uint16_tree_sha256": _tree_sha256(final_stage / "mix_uint16"),
                  "official_stdout_sha256": sha256(final_stage / "official_stdout.txt")}
        previous = None
        if args.previous:
            old = json.loads((args.previous / "run_manifest.json").read_text())
            if not (args.previous / "mix_uint16").is_dir() or old.get("method") != "nvds_dpt_bidirectional_offline":
                raise RuntimeError("--previous is not an NVDS DPT smoke result")
            old_output_tree = old.get("output", {}).get("mix_uint16_tree_sha256", _tree_sha256(args.previous / "mix_uint16"))
            previous = {"path": str(args.previous), "old_manifest_format": "current" if "output" in old else "pre_review", "old_output_tree_sha256": old_output_tree,
                        "new_output_tree_sha256": output["mix_uint16_tree_sha256"],
                        "output_tree_equal": old_output_tree == output["mix_uint16_tree_sha256"],
                        "old_opw": old["diagnostic"]["opw_raw_forward_backward_mix"], "new_opw": _opw(stage / "result.txt")}
        evidence = {"method": "nvds_dpt_bidirectional_offline", "semantic": {"model": "NVDS, not NVDS+", "temporal_access": "noncausal", "output": "monocular relative inverse-depth, per-frame uint16 normalized"},
                    "source_input_records_sha256": source_input_records_sha256, "source_frames": frames,
                    "upstream": {"commit": upstream["commit"], "origin": upstream["origin"], "license_sha256": upstream["license_sha256"], "script": "infer_NVDS_dpt_bi.py", "script_sha256": sha256(UPSTREAM / "infer_NVDS_dpt_bi.py")},
                    "command": command, "compatibility_shim": {"sha256": hashlib.sha256(shim_source.encode()).hexdigest(), "effect": "only supplies torch.load(weights_only=True); CPU gate proved all three official files are tensor-only", "numerical_path": "unaltered after deserialization"},
                    "checkpoints": [{"path": item["path"], "source_sha256": item["sha256"], "runtime_weights_only": True} for item in artifacts],
                    "output": output, "opw": _opw(stage / "result.txt"), "runtime": {"physical_gpu": 1, "logical_device": "cuda:0", "cuda_visible_devices": "1", "python": str(args.python), "torch": torch.__version__, "cuda": torch.version.cuda, "seconds": time.monotonic() - started, "peak_memory_mb": peak}, "previous_comparison": previous}
        evidence_path = final_stage / "official_execution_evidence.json"
        evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        metadata = {"status": "COMPLETE", "publication": "TEST_ONLY", "method": "nvds_dpt_bidirectional_offline",
                    "semantic": "monocular relative inverse-depth", "temporal_access": "noncausal", "h4_or_disparity_evaluation": "NOT_APPLICABLE",
                    "input": {"dataset": "SCARED-C", "sequence": args.sequence, "frames": frame_count, "left_rgb_only": True,
                              "source_frames": frames, "source_input_records_sha256": source_input_records_sha256, "inference_width": 480, "inference_height": 384, "resize": "upstream cv2.INTER_CUBIC"},
                    "upstream": {"path": str(UPSTREAM), "commit": upstream["commit"], "origin": upstream["origin"], "license_sha256": upstream["license_sha256"], "script": "infer_NVDS_dpt_bi.py", "script_sha256": sha256(UPSTREAM / "infer_NVDS_dpt_bi.py"),
                                 "command": command, "execution": "direct official script in isolated temporary copy; checkpoint mounts + weights_only shim are external compatibility only",
                                 "compatibility_shim": {"sha256": hashlib.sha256(shim_source.encode()).hexdigest(), "effect": "only supplies torch.load(weights_only=True); CPU gate proved all three official files are tensor-only", "numerical_path": "unaltered after deserialization"}},
                    "checkpoints": [{"path": item["path"], "source_sha256": item["sha256"], "original_sha256": item["sha256"], "derived_sha256": None, "runtime_weights_only": True, "strict": True} for item in artifacts],
                    "strict_loading": {"gmflow": True, "nvds": True, "dpt": True},
                    "runtime": {"physical_gpu": 1, "logical_device": "cuda:0", "cuda_visible_devices": "1", "python": str(args.python),
                                "torch": torch.__version__, "cuda": torch.version.cuda, "seconds": time.monotonic() - started,
                                "peak_memory": {"method": "nvidia-smi physical GPU 1 polling, 200ms", "mb": peak}},
                    "output": output, "evidence": {"path": evidence_path.name, "sha256": sha256(evidence_path)}, "previous_comparison": previous,
                    "official_equivalence": {"status": "DIRECT_OFFICIAL_EXECUTION", "comparison": "not a reimplementation; published uint16 mix files are emitted by the pinned official script"},
                    "diagnostic": {"opw_raw_forward_backward_mix": _opw(stage / "result.txt"), "spatial": "NOT_RUN; unitless relative inverse-depth is never merged with H4"}}
        (final_stage / "run_manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        os.replace(final_stage, final)
    _validate_published(final)
    print(final)


if __name__ == "__main__":
    main()
