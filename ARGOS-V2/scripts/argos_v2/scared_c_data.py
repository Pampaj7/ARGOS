"""Read SCARED-C corrected_temporal_gt sequences: frame order, images, calibration, GT."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from argos_v2.paths import SCARED_C_SEQ_DIR


@dataclass
class SequenceInfo:
    sequence_id: str
    seq_dir: Path
    frame_ids: list[str]
    fx: float
    baseline_mm: float


def load_sequence_info(sequence_id: str) -> SequenceInfo:
    seq_dir = SCARED_C_SEQ_DIR / sequence_id
    rows = list(csv.DictReader((seq_dir / "manifest.csv").open()))
    if not rows:
        raise RuntimeError(f"empty manifest for {sequence_id}")
    frame_ids = [r["frame_id"] for r in rows]
    return SequenceInfo(
        sequence_id=sequence_id,
        seq_dir=seq_dir,
        frame_ids=frame_ids,
        fx=float(rows[0]["fx"]),
        baseline_mm=float(rows[0]["baseline_mm"]),
    )


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_frame_lr(info: SequenceInfo, frame_id: str) -> tuple[np.ndarray, np.ndarray]:
    left = read_rgb(info.seq_dir / "left" / f"{frame_id}.png")
    right = read_rgb(info.seq_dir / "right" / f"{frame_id}.png")
    return left, right


def load_frame_gt(info: SequenceInfo, frame_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (gt_disp [H,W] float32, gt_valid [H,W] bool) at native SCARED-C resolution."""
    gt_dir = info.seq_dir / "gt"
    disp = np.load(gt_dir / f"{frame_id}_disp.npy").astype(np.float32)
    valid = cv2.imread(str(gt_dir / f"{frame_id}_valid.png"), cv2.IMREAD_GRAYSCALE) > 0
    return disp, valid
