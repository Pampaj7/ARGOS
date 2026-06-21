from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from scripts.argos_paths import RESULTS_DIR
from .types import TemporalBatch


@dataclass(frozen=True)
class RealSequenceSpec:
    sequence_id: str
    start_index: int
    length: int
    crop_height: int
    crop_width: int


class ScaredProgressiveSequenceLoader:
    """Loader for ARGOS indexed SCARED temporal-refinement cache.

    The cache stores per-frame RGB and predictions in original image disparity
    coordinates. StereoAnyVideo is loaded as a teacher/pseudo-target, never GT.
    """

    def __init__(
        self,
        cache_root: Path | None = None,
        index_file: Path | None = None,
    ):
        self.cache_root = cache_root or (
            RESULTS_DIR / "03_temporal_refinement/cache/temporal_refinement_cache/large_v3_s2m2s512_fast"
        )
        self.index_file = index_file or (self.cache_root / "index.csv")
        with self.index_file.open() as f:
            rows = list(csv.DictReader(f))
        self.by_sequence: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            self.by_sequence.setdefault(row["sequence_id"], []).append(row)
        for seq_id, seq_rows in self.by_sequence.items():
            self.by_sequence[seq_id] = sorted(seq_rows, key=lambda r: int(r["center_frame_id"]))

    def sequence_ids(self) -> list[str]:
        return sorted(self.by_sequence)

    def make_spec(
        self,
        sequence_id: str | None = None,
        length: int = 5,
        crop_height: int = 256,
        crop_width: int = 384,
        start_index: int = 0,
    ) -> RealSequenceSpec:
        seq_id = sequence_id or self.sequence_ids()[0]
        if seq_id not in self.by_sequence:
            raise KeyError(f"Unknown sequence {seq_id!r}")
        if len(self.by_sequence[seq_id]) < length:
            raise ValueError(f"Sequence {seq_id} has only {len(self.by_sequence[seq_id])} indexed centers")
        return RealSequenceSpec(seq_id, start_index, length, crop_height, crop_width)

    def load(self, spec: RealSequenceSpec, device: torch.device) -> TemporalBatch:
        rows = self.by_sequence[spec.sequence_id][spec.start_index : spec.start_index + spec.length]
        if len(rows) != spec.length:
            raise ValueError(f"Requested {spec.length} frames, got {len(rows)}")
        h0, w0 = int(rows[0]["height"]), int(rows[0]["width"])
        ch, cw = min(spec.crop_height, h0), min(spec.crop_width, w0)
        y, x = (h0 - ch) // 2, (w0 - cw) // 2
        rgbs, s2m2_s, s2m2_l, sav = [], [], [], []
        valid_mask = []
        gt_disp = []
        for row in rows:
            rgb = self._read_rgb(self.cache_root / row["rgb_center_path"], y, x, ch, cw)
            rgbs.append(torch.from_numpy(rgb).permute(2, 0, 1))
            s2m2_s.append(torch.from_numpy(self._read_disp(row["s2m2_s512_t_path"], y, x, ch, cw)).unsqueeze(0))
            s2m2_l.append(torch.from_numpy(self._read_disp(row["s2m2_l736_t_path"], y, x, ch, cw)).unsqueeze(0))
            sav.append(torch.from_numpy(self._read_disp(row["sav_t_path"], y, x, ch, cw)).unsqueeze(0))
            # The current large_v3 cache has has_gt=False, but keep the shape
            # plumbing ready for future SCARED warped/keyframe GT caches.
            valid_mask.append(torch.zeros(1, ch, cw))
            gt_disp.append(torch.zeros(1, ch, cw))
        batch = TemporalBatch(
            rgb=torch.stack(rgbs).unsqueeze(0).float().to(device),
            s2m2_s_disp=torch.stack(s2m2_s).unsqueeze(0).float().to(device),
            s2m2_l_disp=torch.stack(s2m2_l).unsqueeze(0).float().to(device),
            sav_disp=torch.stack(sav).unsqueeze(0).float().to(device),
            gt_disp=None,
            gt_depth_mm=None,
            valid_mask=None,
            sequence_ids=[spec.sequence_id],
        )
        return batch

    def load_clip(
        self,
        sequence_id: str,
        start_index: int,
        length: int,
        crop_height: int,
        crop_width: int,
        device: torch.device,
    ) -> TemporalBatch:
        return self.load(
            RealSequenceSpec(
                sequence_id=sequence_id,
                start_index=start_index,
                length=length,
                crop_height=crop_height,
                crop_width=crop_width,
            ),
            device=device,
        )

    def _read_rgb(self, path: Path, y: int, x: int, h: int, w: int) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Could not read RGB: {path}")
        img = cv2.cvtColor(img[y : y + h, x : x + w], cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.0

    def _read_disp(self, rel_path: str, y: int, x: int, h: int, w: int) -> np.ndarray:
        arr = np.load(self.cache_root / rel_path, mmap_mode="r")
        return np.asarray(arr[y : y + h, x : x + w], dtype=np.float32)
