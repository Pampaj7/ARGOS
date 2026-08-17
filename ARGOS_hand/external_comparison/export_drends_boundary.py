#!/usr/bin/env python3
"""Export DRENDS recordings as external-comparison boundaries for BiDAStabilizer.

The published head-to-head ran on SCARED-C D2, which is the split our checkpoint was
selected on while BiDAStabilizer had never seen the domain. That comparison is reported
in the paper as an upper bound on our advantage rather than an estimate of it. This
builds the neutral arena: on DRENDS neither method has seen the domain.

Exactly one thing must be true for the comparison to mean anything, and it is the thing
the D2 comparison was originally written to fix: both methods must consume the SAME
frozen disparity. The D2 boundary carried RAFT-Stereo *robust* because that is what the
published BiDA reproduction ran; our DRENDS cache carries RAFT-Stereo *middlebury*.
Rather than re-run either side with the other's weights, this exports the middlebury
predictions we already evaluate TETHER on and hands BiDA that identical array, so the
stored boundary -- not a coincidence of configuration -- is what both read.

No ground truth crosses the boundary: BiDA sees RGB and raw disparity only, and scoring
happens afterwards against a support neither method influences.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
ARGOS = ROOT.parents[1]
H4 = ARGOS / "ARGOS_hand/original_h4"
OUT = ROOT / "results" / "drends_boundary"
RECORDINGS = ("Vid10_Liver_Med", "Vid11_Liver_High", "Vid12_Pancreas_Ext",
              "Vid13_Pancreas_Med", "Vid14_Pancreas_High")

sys.path.insert(0, str(ROOT))
from bridge import write_input  # noqa: E402


def export(recording: str, device: str, max_frames: int | None) -> dict:
    """One recording, on the same canonical grid the evaluation uses."""
    sys.path[:0] = [str(H4), str(H4 / "scripts"), str(ARGOS / "ARGOS_FREEZED/src")]
    import torch
    from model_design.comparison import drends_backbone_transfer as transfer

    base = transfer.base
    _checkpoint, predict = transfer._load_backbone("RAFT-Stereo", torch.device(device))
    records, _info = base.load_drends_records(recording, max_frames)
    frames, (height, width) = base._canonical_frames(records, predict, device=device)

    # [T,3,H,W] float32 RGB and [T,1,H,W] disparity, exactly the grid the frozen module
    # and the metrics already agree on. Anything resampled here would silently make the
    # two sides incomparable again.
    left = np.stack([frame["rgb"][0].cpu().numpy().astype(np.float32) for frame in frames])
    right = np.stack([frame["right_rgb"][0].cpu().numpy().astype(np.float32) for frame in frames])
    disparity = np.stack([frame["raw"][0].cpu().numpy().astype(np.float32) for frame in frames])
    # The evaluator's own validity, not a re-derived one: a mask computed here could
    # disagree with the mask TETHER was scored under and quietly change the support.
    valid = np.stack([frame["raw_valid"][0].cpu().numpy() for frame in frames]).astype(bool)
    # The bridge refuses non-positive disparity on valid pixels; the evaluator's mask
    # already encodes exactly that, so a mismatch here means the contract changed.
    if np.any(disparity[valid] <= 0):
        raise RuntimeError("evaluator marked non-positive disparity valid; contract mismatch")
    ids = np.array([f"{recording}_{index:06d}" for index in range(len(frames))], dtype="<U64")

    destination = OUT / recording / "input.npz"
    info = write_input(destination, {"rgb_left": left, "rgb_right": right,
                                     "raw_disparity": disparity, "raw_valid": valid,
                                     "frame_ids": ids},
                       {"dataset": "drends", "recording": recording,
                        "backbone": "RAFT-Stereo", "backbone_checkpoint": _checkpoint,
                        "grid": [int(height), int(width)],
                        "note": "middlebury weights, matching the cache TETHER is evaluated on; "
                                "BiDA must consume this array and not its own prediction"})
    return {"recording": recording, "frames": int(len(frames)), "grid": [int(height), int(width)],
            "path": str(destination), "input_sha256": info["input_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--recordings", nargs="+", default=list(RECORDINGS))
    parser.add_argument("--max-frames", type=int, help="smoke-test cap; a capped export is not a result")
    args = parser.parse_args()

    for recording in args.recordings:
        record = export(recording, args.device, args.max_frames)
        print(f"{record['recording']:<24} {record['frames']:>5} frames  "
              f"{record['grid']}  {record['input_sha256'][:16]}")
    if args.max_frames is not None:
        print("\nCAPPED EXPORT: smoke test only, not a boundary a comparison may be built on.")


if __name__ == "__main__":
    main()
