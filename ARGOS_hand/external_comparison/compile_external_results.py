"""The sole publication gate for allowlisted external comparison source runs."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from package_source_run import _execution_manifest

ROOT = Path(__file__).resolve().parent
FROZEN_COMPILER = ROOT.parent / "original_h4/model_design/comparison/run_definitive_evaluation.py"


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def validate(source_root: Path, datasets: list[str]) -> None:
    for dataset in datasets:
        runs = source_root / dataset
        if not runs.is_dir():
            raise ValueError(f"source dataset missing: {dataset}")
        methods = [path for path in runs.iterdir() if path.is_dir()]
        if not methods:
            raise ValueError(f"source dataset has no methods: {dataset}")
        for run in methods:
            manifest, external = _json(run / "run_manifest.json"), _json(run / "external_method.json")
            for item in (manifest, external):
                execution_id, execution_hash = item.get("execution_manifest_id"), item.get("execution_manifest_sha256")
                if item.get("publication") != "PUBLISHABLE" or not isinstance(execution_id, str) or not execution_id or not isinstance(execution_hash, str) or len(execution_hash) != 64:
                    raise ValueError(f"not publishable: {run}")
            if (manifest["publication"], manifest["execution_manifest_id"], manifest["execution_manifest_sha256"]) != (external["publication"], external["execution_manifest_id"], external["execution_manifest_sha256"]):
                raise ValueError(f"manifest attestation mismatch: {run}")
            input_sha256, prediction_sha256 = external.get("input_sha256"), external.get("prediction_sha256")
            if external.get("method") != run.name or manifest.get("external_method") != external or not all(isinstance(value, str) and len(value) == 64 for value in (input_sha256, prediction_sha256)):
                raise ValueError(f"source execution binding mismatch: {run}")
            if (external.get("source_input_sha256") != input_sha256
                    or not isinstance(external.get("rgb_input_sha256"), str) or len(external["rgb_input_sha256"]) != 64
                    or external.get("source_rgb_input_sha256") != external["rgb_input_sha256"]
                    or not isinstance(external.get("frame_ids"), list) or not external["frame_ids"]):
                raise ValueError(f"source bridge provenance mismatch: {run}")
            execution = _execution_manifest(external["execution_manifest_id"], run.name, input_sha256, prediction_sha256)
            if execution["sha256"] != external["execution_manifest_sha256"]:
                raise ValueError(f"execution manifest attestation mismatch: {run}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=("scared-d2", "scared-d7"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validate(args.source_root, args.datasets)
    subprocess.run([sys.executable, str(FROZEN_COMPILER), "--compile-from", str(args.source_root), "--datasets", *args.datasets, "--output", str(args.output)], check=True)


if __name__ == "__main__":
    main()
