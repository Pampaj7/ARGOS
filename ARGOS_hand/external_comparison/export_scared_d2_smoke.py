"""Prepare/freeze bounded or complete SCARED D2 BiDA bridge boundaries."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
H4 = ROOT.parent / "original_h4"
sys.path[:0] = [str(ROOT), str(H4 / "scripts"), str(H4)]
from bridge import read_input, write_input  # noqa: E402
from argos_v2.scared_c_data import load_frame_gt, load_frame_lr, load_sequence_info  # noqa: E402
from model_design.comparison.run_comparison import official_scared_protocol_mask  # noqa: E402

D2_SEQUENCES = ("dataset_2_keyframe_2", "dataset_2_keyframe_3", "dataset_2_keyframe_4")
SMOKE_FRAMES = 64
HEIGHT, WIDTH = 144, 180


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **values)
        os.replace(temporary.with_suffix(temporary.suffix + ".npz"), path)
    finally:
        temporary.unlink(missing_ok=True); temporary.with_suffix(temporary.suffix + ".npz").unlink(missing_ok=True)


def _frames(sequence: str, full: bool) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if sequence not in D2_SEQUENCES:
        raise ValueError(f"unsupported D2 sequence: {sequence}")
    info = load_sequence_info(sequence)
    ids = info.frame_ids if full else info.frame_ids[:SMOKE_FRAMES]
    if len(ids) < 3 or ids != sorted(ids):
        raise RuntimeError(f"expected chronological frames for {sequence}")
    left, right, gt, valid = [], [], [], []
    for frame_id in ids:
        image_left, image_right = load_frame_lr(info, frame_id)
        disparity, mask = load_frame_gt(info, frame_id)
        left.append(cv2.resize(image_left, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA))
        right.append(cv2.resize(image_right, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA))
        coverage = cv2.resize(mask.astype(np.float32), (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        numerator = cv2.resize(disparity * mask.astype(np.float32), (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        gt.append(numerator / np.maximum(coverage, 1e-6) * (WIDTH / disparity.shape[1]))
        valid.append(coverage > .5)
    rgb_left = np.ascontiguousarray(np.asarray(left, np.float32).transpose(0, 3, 1, 2))
    rgb_right = np.ascontiguousarray(np.asarray(right, np.float32).transpose(0, 3, 1, 2))
    return {"rgb_left": rgb_left, "rgb_right": rgb_right, "gt_disparity": np.asarray(gt, np.float32)[:, None],
            "gt_valid": np.asarray(valid, bool)[:, None], "frame_ids": np.asarray(ids)}, {
                "dataset": "SCARED-C", "split": "d2", "backbone": "RAFTStereo robust", "sequence_id": sequence,
                "protocol": "d2_full_diagnostic_raw_valid" if full else "smoke_diagnostic_raw_valid",
                "purpose": "D2_FULL_DIAGNOSTIC" if full else "SMOKE_DIAGNOSTIC", "publication": "TEST_ONLY",
                "source_manifest_sha256": sha256(info.seq_dir / "manifest.csv"),
                "calibration": {"fx_px": float(info.fx) * WIDTH / 1280, "baseline_mm": float(info.baseline_mm), "width_scale": WIDTH / 1280},
            }


def seed(path: Path, sequence: str, full: bool) -> None:
    values, metadata = _frames(sequence, full)
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    dummy = np.ones((len(values["frame_ids"]), 1, HEIGHT, WIDTH), np.float32)
    write_input(path, {"rgb_left": values["rgb_left"], "rgb_right": values["rgb_right"], "raw_disparity": dummy,
                       "raw_valid": np.ones_like(dummy, bool), "frame_ids": values["frame_ids"]},
                metadata | {"purpose": f"{metadata['purpose']}_SEED", "publication": "TEST_ONLY"})


def freeze(bridge: Path, evaluation: Path, sequence: str, full: bool) -> None:
    if evaluation.exists() or evaluation.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to overwrite: {evaluation}")
    values, input_meta = read_input(bridge)
    source, metadata = _frames(sequence, full)
    if (values["frame_ids"].tolist() != source["frame_ids"].tolist()
            or not np.array_equal(values["rgb_left"], source["rgb_left"])
            or not np.array_equal(values["rgb_right"], source["rgb_right"])):
        raise ValueError("bridge frame IDs/grid do not match frozen D2 smoke source")
    raw_valid = values["raw_valid"]
    # The frozen evaluator owns this operation.  The diagnostic artifact deliberately
    # has no flow-derived paper-support claim; all temporal supports are true.
    protocol_mask = official_scared_protocol_mask(raw_valid, np.ones_like(raw_valid), (np.ones_like(raw_valid), np.ones_like(raw_valid)))
    arrays = {"gt_disparity": source["gt_disparity"], "gt_valid": source["gt_valid"],
              "protocol_mask": np.asarray(protocol_mask, bool), "keyframe_mask": np.asarray([True] + [False] * (len(values["frame_ids"]) - 1)),
              "adapter_support": raw_valid.astype(bool)}
    _atomic_npz(evaluation, **arrays)
    sidecar = metadata | {"frame_ids": values["frame_ids"].tolist(), "input_sha256": input_meta["input_sha256"],
                          "rgb_input_sha256": input_meta["rgb_input_sha256"], "evaluation_npz_sha256": sha256(evaluation),
                          "bridge_artifact_id": bridge.name, "bridge_npz_sha256": sha256(bridge),
                          "bridge_json_sha256": sha256(bridge.with_suffix(".json"))}
    evaluation.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seed", type=Path); group.add_argument("--bridge", type=Path)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--sequence", choices=D2_SEQUENCES, default=D2_SEQUENCES[0])
    parser.add_argument("--full", action="store_true", help="use all chronological frames of the selected D2 sequence")
    args = parser.parse_args()
    if args.seed:
        if args.evaluation:
            parser.error("--seed writes only the pre-inference bridge")
        seed(args.seed, args.sequence, args.full)
    else:
        if not args.evaluation:
            parser.error("--bridge requires --evaluation")
        freeze(args.bridge, args.evaluation, args.sequence, args.full)


if __name__ == "__main__":
    main()
