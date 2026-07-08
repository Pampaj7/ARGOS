#!/usr/bin/env python3
"""Evaluation entry point for EGBM-v3-CARE-S.

The training script already writes the full window/streaming evaluation. This wrapper
keeps the requested path stable and reports where to read the artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_ROOT = Path("results/03_temporal_refinement/training/egbm_v3_care_streaming")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    args = p.parse_args()
    summary = args.output_root / "aggregate_summary.json"
    if not summary.exists():
        raise FileNotFoundError(f"{summary} not found; run train_egbm_v3_care_streaming.py first")
    data = json.loads(summary.read_text())
    print(json.dumps({
        "model": data.get("model"),
        "selected_streaming": data.get("selected_streaming", {}).get("all"),
        "full_gt_streaming_test": data.get("full_gt_streaming", {}).get("test"),
        "summary": str(summary),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
