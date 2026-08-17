#!/usr/bin/env python3
"""Export DRENDS recordings as external-comparison boundaries for BiDAStabilizer.

The published head-to-head ran on SCARED-C D2, which is the split our checkpoint was
selected on while BiDAStabilizer had never seen the domain. That comparison is reported
in the paper as an upper bound on our advantage rather than an estimate of it. This
builds the neutral arena: on DRENDS neither method has seen the domain.

This is a SEED boundary and carries RGB only. The disparity field is a documented dummy,
for the same reason the D2 seed export writes one: `workers/bidastabilizer.py` reads only
`rgb_left` and `rgb_right`, then replaces `raw_disparity` with its own RAFT-Stereo
*robust* prediction (32 iterations) and writes that back out via `--raw-output`. Anything
we put in this field would be silently discarded, and putting our cached *middlebury*
disparity here would merely look as though the two sides shared an input when they never
did.

The shared input therefore comes from BiDA, not from us, and the order matters: BiDA runs
first and produces the RAFT-robust raw, and TETHER is then run on that stored array. That
is exactly what the D2 comparison does, and it is why its numbers are not comparable with
the middlebury figures in the cross-backbone table.

No ground truth crosses the boundary: BiDA sees RGB only, and scoring happens afterwards
against a support neither method influences.
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
    """One recording's RGB, on the same canonical grid the evaluation uses."""
    sys.path[:0] = [str(H4), str(H4 / "scripts"), str(ARGOS / "ARGOS_FREEZED/src")]
    import torch
    from model_design.comparison import drends_backbone_transfer as transfer

    base = transfer.base
    # The backbone still runs, because _canonical_frames is the only path that produces
    # frames on the grid the evaluation agrees on, and its disparity is what makes the
    # RGB reproducible from a named checkpoint. The disparity itself is not exported.
    _checkpoint, predict = transfer._load_backbone("RAFT-Stereo", torch.device(device))
    records, _info = base.load_drends_records(recording, max_frames)
    # The second return value is the NATIVE shape (720, 1280) that _canonical_frames
    # validates its input against, not the grid the frames come out on -- they are
    # resized to CANONICAL_SIZE, 144x180. Using the returned shape here built a dummy
    # disparity at native size against 144x180 RGB, which the bridge rejected. Take the
    # grid from the array that will actually be written.
    frames, _native = base._canonical_frames(records, predict, device=device)

    # [T,3,H,W] float32 RGB on exactly the grid the frozen module and the metrics already
    # agree on. Anything resampled here would silently make the two sides incomparable.
    left = np.stack([frame["rgb"][0].cpu().numpy().astype(np.float32) for frame in frames])
    right = np.stack([frame["right_rgb"][0].cpu().numpy().astype(np.float32) for frame in frames])
    height, width = left.shape[2], left.shape[3]
    # Dummy disparity, as in the D2 seed export. The stabiliser overwrites this field with
    # its own RAFT-robust prediction, so a real array here would be discarded while
    # implying a shared input that does not exist until BiDA has run.
    dummy = np.ones((len(frames), 1, int(height), int(width)), np.float32)
    ids = np.array([f"{recording}_{index:06d}" for index in range(len(frames))], dtype="<U64")

    destination = OUT / recording / "seed.npz"
    info = write_input(destination, {"rgb_left": left, "rgb_right": right,
                                     "raw_disparity": dummy,
                                     "raw_valid": np.ones_like(dummy, bool),
                                     "frame_ids": ids},
                       {"dataset": "drends", "recording": recording,
                        "publication": "TEST_ONLY", "purpose": "DRENDS_SEED",
                        "rgb_source_backbone_checkpoint": _checkpoint,
                        "grid": [int(height), int(width)],
                        "note": "RGB-only seed; raw_disparity is a dummy the stabilizer replaces "
                                "with its own RAFT-robust prediction, which then becomes the "
                                "shared input TETHER is run on"})
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
