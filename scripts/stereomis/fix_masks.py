#!/usr/bin/env python3
"""Re-remap curated masks in place, fixing the resolution-mismatch bug in convert.py:
some raw masks ship at half the video frame resolution (e.g. 512x640 vs 1280x1024), which
made remap use rectify maps built for the full-res frame on a half-res source, producing a
mostly-black canvas with a small square artifact instead of a properly aligned mask.

Left/right images are unaffected by this bug (masks only) — no video re-decode needed.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.argos_paths import DATASET_DIR
from scripts.stereomis.convert import RAW, OUT, parse_calib, build_rectify_maps

import cv2

MANIFEST = DATASET_DIR / "StereoMIS/curated/manifests/manifest.csv"


def fix_sequence(seq: str, frame_ids: list[str]) -> int:
    seq_dir = RAW / seq
    K1, D1, K2, D2, R, T, size = parse_calib(seq_dir / "StereoCalibration.ini")
    map1x, map1y, _map2x, _map2y, _P1, _P2 = build_rectify_maps(K1, D1, K2, D2, R, T, size)
    w, h = size
    out_dir = OUT / seq / "mask"
    fixed = 0
    for stem in frame_ids:
        mask_src = seq_dir / "masks" / f"{stem}l.png"
        if not mask_src.exists():
            continue
        mask = cv2.imread(str(mask_src), cv2.IMREAD_GRAYSCALE)
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_r = cv2.remap(mask, map1x, map1y, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out_dir / f"{stem}.png"), mask_r)
        fixed += 1
    return fixed


def main() -> int:
    rows = list(csv.DictReader(MANIFEST.open()))
    by_seq: dict[str, list[str]] = {}
    for r in rows:
        if r["has_mask"] == "True":
            by_seq.setdefault(r["sequence"], []).append(r["frame_id"])

    for seq, frame_ids in by_seq.items():
        n = fix_sequence(seq, frame_ids)
        print(f"{seq}: fixed {n} masks", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
