"""Run an external method in its isolated interpreter, with a strict NPZ boundary."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from bridge import read_input

ROOT = Path(__file__).resolve().parent
METHODS = {
    "nvds_plus_forward_clip4": ("nvds_plus.py", "nvds_plus_forward_clip4.json"),
    "nvds_plus_bidirectional_offline": ("nvds_plus.py", "nvds_plus_bidirectional_offline.json"),
    "bidastabilizer_bidirectional_offline": ("bidastabilizer.py", "bidastabilizer_bidirectional_offline.json"),
}


def preflight(method: str, input_path: Path) -> dict[str, object]:
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    values, meta = read_input(input_path)
    protocol = json.loads((ROOT / "protocols" / METHODS[method][1]).read_text())
    if protocol["method"] != method or protocol["online_or_h4"]:
        raise ValueError("invalid method protocol")
    return {"status": "PASS", "method": method, "frames": int(values["rgb_left"].shape[0]), "input_sha256": meta["input_sha256"], "causality": protocol["causality"], "future_frames_required": protocol["future_frames_required"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--input", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, help="BiDA only: RAFT-robust bridge input paired with the refined output")
    parser.add_argument("--purpose", choices=("SMOKE_DIAGNOSTIC", "D2_FULL_DIAGNOSTIC", "DRENDS_FULL_DIAGNOSTIC"), default="SMOKE_DIAGNOSTIC")
    parser.add_argument("--python", type=Path, default=Path(sys.executable), help="isolated upstream environment interpreter")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    result = preflight(args.method, args.input)
    if args.preflight:
        print(json.dumps(result, sort_keys=True)); return
    worker, protocol = METHODS[args.method]
    gpu = os.environ.get("ARGOS_EXTERNAL_CUDA_VISIBLE_DEVICES", "")
    if "0" in {item.strip() for item in gpu.split(",") if item.strip()}:
        raise ValueError("physical GPU 0 is reserved; choose a nonzero CUDA device")
    env = os.environ | {"CUDA_VISIBLE_DEVICES": gpu, "PYTHONDONTWRITEBYTECODE": "1"}  # GPU selection is explicit; default never selects GPU0.
    if args.method == "bidastabilizer_bidirectional_offline" and args.raw_output is None:
        raise ValueError("BiDA requires --raw-output so the raw and refined predictions share the same official RAFT result")
    command = [str(args.python), str(ROOT / "workers" / worker), "--input", str(args.input), "--output", str(args.output), "--protocol", str(ROOT / "protocols" / protocol), "--checkpoints", str(ROOT / "checkpoints.lock.json")]
    if args.raw_output is not None:
        command.extend(["--raw-output", str(args.raw_output)])
    if args.method == "bidastabilizer_bidirectional_offline":
        command.extend(["--purpose", args.purpose])
    subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
