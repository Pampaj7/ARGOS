"""Fail-closed NVDS+ process boundary; never deserializes unallowlisted weights."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge import read_input  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True); parser.add_argument("--checkpoints", type=Path, required=True)
    args = parser.parse_args()
    read_input(args.input)
    checkpoint = json.loads(args.checkpoints.read_text())["checkpoints"]["nvds_plus"]
    if checkpoint["status"] != "READY" or not checkpoint.get("sha256"):
        raise RuntimeError(f"NVDS+ is blocked: {checkpoint['reason']}")
    raise RuntimeError("NVDS+ runner is intentionally disabled until its documented stereo/raw-disparity input semantics are independently verified")


if __name__ == "__main__":
    main()
