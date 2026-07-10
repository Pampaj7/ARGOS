#!/usr/bin/env python3
"""Build a fixed, deterministic native-resolution validation subset: 3 evenly-spaced frames
per accepted sequence (51 frames total across 17 sequences), independent of any model output.
Saved once and reused identically across all 5 backbones for native-resolution sanity metrics.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from argos_v2.paths import RESULTS_DIR
from argos_v2.scared_c_data import load_sequence_info
from argos_v2.sequences import accepted_sequences

FRAMES_PER_SEQUENCE = 3
OUT_PATH = RESULTS_DIR.parent / "cache_scaredc_backbones/reports_full/native_validation_subset.json"


def main() -> int:
    subset = {}
    for seq in accepted_sequences():
        info = load_sequence_info(seq)
        idx = np.linspace(0, len(info.frame_ids) - 1, FRAMES_PER_SEQUENCE).astype(int)
        idx = sorted(set(idx.tolist()))
        subset[seq] = [info.frame_ids[i] for i in idx]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    total = sum(len(v) for v in subset.values())
    payload = {
        "description": "Deterministic native-resolution validation subset, fixed before inspecting "
                        "any comparative backbone results. Identical frame list reused for all 5 backbones.",
        "frames_per_sequence": FRAMES_PER_SEQUENCE,
        "total_frames": total,
        "sequences": subset,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT_PATH}: {len(subset)} sequences, {total} frames total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
