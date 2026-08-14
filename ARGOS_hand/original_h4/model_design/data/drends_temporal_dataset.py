"""Training-only DRENDS RAFT cache and causal four-pair clips.

The cache is intentionally local to ``original_h4``: it must not mutate the
immutable ARGOS-V2 cache.  Vid14 is refused here so it cannot be opened by a
training process by accident.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from model_design.comparison.drends_evaluation import (
    CANONICAL_SIZE, HAND, RAFT_CHECKPOINT, _canonical_disparity, _depth,
    _load_raft, _rgb, load_drends_records,
)

HEIGHT, WIDTH = 144, 180
BACKBONE = "RAFT-Stereo"
HELDOUT = "Vid14_Pancreas_High"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_path(root: Path, recording: str) -> Path:
    return root / BACKBONE / recording


def _complete_cache(target: Path, expected: list[str], info: dict) -> bool:
    """Validate a published cache before reusing it."""
    metadata = target / "metadata.json"
    if not ((target / ".complete").is_file() and metadata.is_file()):
        return False
    try:
        old = json.loads(metadata.read_text())
        disparity = np.load(target / "disparity.npy", mmap_mode="r", allow_pickle=False)
        valid = np.load(target / "valid_mask.npy", mmap_mode="r", allow_pickle=False)
        frame_ids = np.load(target / "frame_ids.npy", mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        old.get("frame_ids") == expected
        and old.get("manifest_sha256") == info["manifest_sha256"]
        and old.get("quality_sha256") == info["quality_sha256"]
        and old.get("raft_checkpoint_sha256") == _sha256(RAFT_CHECKPOINT)
        and old.get("shape") == [len(expected), HEIGHT, WIDTH]
        and disparity.shape == (len(expected), HEIGHT, WIDTH)
        and disparity.dtype == np.float16
        and valid.shape == disparity.shape
        and valid.dtype == np.uint8
        and frame_ids.shape == (len(expected),)
        and [str(value) for value in frame_ids.tolist()] == expected
    )


def build_raft_cache(root: Path, recordings: Sequence[str], device: str) -> dict:
    """Atomically create/reuse canonical RAFT raw disparities for train/val only."""
    requested = tuple(recordings)
    if HELDOUT in requested or len(requested) != len(set(requested)):
        raise ValueError("DRENDS training cache must exclude Vid14 and duplicates")
    root.mkdir(parents=True, exist_ok=True)
    report = {}
    checkpoint = predict = None
    for recording in requested:
        print(json.dumps({"event": "drends_cache_start", "recording": recording}), flush=True)
        records, info = load_drends_records(recording)
        target = cache_path(root, recording)
        expected = [item["frame_id"] for item in records]
        if (target / ".complete").is_file():
            if _complete_cache(target, expected, info):
                report[recording] = {"status": "reused", "frames": len(expected), "cache": str(target)}
                print(json.dumps({"event": "drends_cache_reused", "recording": recording, "frames": len(expected)}), flush=True)
                continue
            raise RuntimeError(f"refusing corrupt or mismatched complete cache: {target}")
        if checkpoint is None:
            checkpoint, predict = _load_raft(torch.device(device))
        values, valid = [], []
        for item in records:
            left, right = _rgb(item["_rect_left"], None), _rgb(item["_rect_right"], None)
            native = predict(left, right)[0]
            if left.shape != (720, 1280, 3) or right.shape != left.shape or native.shape != (720, 1280) or not np.isfinite(native).all():
                raise RuntimeError(f"invalid RAFT DRENDS prediction for {recording}/{item['frame_id']}")
            raw = _canonical_disparity(native, WIDTH / 1280.0)
            values.append(raw.astype(np.float16)); valid.append(((raw > 0) & np.isfinite(raw)).astype(np.uint8))
        with tempfile.TemporaryDirectory(dir=root, prefix=f".{recording}.") as temporary:
            stage = Path(temporary)
            np.save(stage / "disparity.npy", np.asarray(values, dtype=np.float16))
            np.save(stage / "valid_mask.npy", np.asarray(valid, dtype=np.uint8))
            np.save(stage / "frame_ids.npy", np.asarray(expected))
            payload = {"recording": recording, "backbone": BACKBONE, "frame_ids": expected,
                       "frame_count": len(expected), "shape": [len(expected), HEIGHT, WIDTH],
                       "disparity_convention": "positive_left; resize 1280->180 then scale 180/1280",
                       "manifest_sha256": info["manifest_sha256"], "quality_sha256": info["quality_sha256"],
                       "raft_checkpoint": str(RAFT_CHECKPOINT), "raft_checkpoint_sha256": _sha256(RAFT_CHECKPOINT),
                       "evaluator_checkpoint": checkpoint, "timing_limit_ms": 100.0,
                       "excluded_timing_frame_ids": info["excluded_timing_frame_ids"]}
            (stage / "metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                # An incomplete cache is solely this experiment's resumable output.
                import shutil; shutil.rmtree(target)
            os.replace(stage, target)
            (target / ".complete").write_text("complete\n")
        report[recording] = {"status": "built", "frames": len(expected), "cache": str(target)}
        print(json.dumps({"event": "drends_cache_built", "recording": recording, "frames": len(expected)}), flush=True)
    return report


@dataclass(frozen=True)
class Clip:
    recording: str
    pair_indices: tuple[int, ...]


class DrendsTemporalClipDataset(Dataset):
    """DRENDS item contract mirrors ``TemporalClipDataset`` exactly."""
    def __init__(self, recordings: Sequence[str], cache_root: Path, *, clip_length: int = 4, coverage_threshold: float = .5) -> None:
        if HELDOUT in recordings:
            raise ValueError("Vid14 is held out and cannot enter this dataset")
        self.recordings, self.cache_root = tuple(recordings), Path(cache_root)
        self.clip_length, self.coverage_threshold = clip_length, coverage_threshold
        self.info, self.records, self._cache, self._frames = {}, {}, {}, {}
        self.clips = []
        for recording in self.recordings:
            rows, info = load_drends_records(recording)
            target = cache_path(self.cache_root, recording)
            if not (target / ".complete").is_file(): raise FileNotFoundError(f"missing complete DRENDS cache: {target}")
            meta = json.loads((target / "metadata.json").read_text())
            ids = [item["frame_id"] for item in rows]
            if meta.get("frame_ids") != ids or meta.get("manifest_sha256") != info["manifest_sha256"] or meta.get("quality_sha256") != info["quality_sha256"]:
                raise RuntimeError(f"DRENDS cache contract mismatch: {recording}")
            self.info[recording], self.records[recording] = info, rows
            self._cache[recording] = (np.load(target / "disparity.npy", mmap_mode="r"), np.load(target / "valid_mask.npy", mmap_mode="r"))
            for start in range(0, len(rows) - clip_length, clip_length): self.clips.append(Clip(recording, tuple(range(start + 1, start + 1 + clip_length))))
        if not self.clips: raise RuntimeError("DRENDS needs at least one complete causal clip")

    def __len__(self): return len(self.clips)

    def preload_frame_data(self, workers: int) -> dict:
        if workers < 1: return {"enabled": False, "frames": 0, "bytes": 0}
        tasks = [(recording, index, row) for recording in self.recordings for index, row in enumerate(self.records[recording])]
        loaded = {recording: [None] * len(rows) for recording, rows in self.records.items()}
        def read(task):
            recording, index, row = task; left, right = _rgb(row["_rect_left"]), _rgb(row["_rect_right"])
            depth, coverage, _ = _depth(row["_depth_left"], row["_mask_left"], WIDTH / 1280.0)
            product = self.info[recording]["focal_baseline_native_px_m"] * 1000.0 * WIDTH / 1280.0
            gt = product / np.maximum(depth, 1e-6)
            return recording, index, (np.ascontiguousarray(left).transpose(2,0,1), np.ascontiguousarray(right).transpose(2,0,1), gt.astype(np.float32), coverage, depth.astype(np.float32), product)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for recording, index, value in pool.map(read, tasks): loaded[recording][index] = value
        self._frames = loaded
        return {"enabled": True, "frames": len(tasks), "bytes": int(sum(v.nbytes for rows in loaded.values() for f in rows for v in f[:5] if hasattr(v, "nbytes"))), "workers": workers}

    def _pair(self, recording: str, current: int) -> dict:
        if not self._frames: raise RuntimeError("preload_frame_data is required for DRENDS training")
        past = current - 1; raw, valid = self._cache[recording]; cur, prev = self._frames[recording][current], self._frames[recording][past]
        row, old = self.records[recording][current], self.records[recording][past]
        return {"raw": torch.from_numpy(np.asarray(raw[current], dtype=np.float32).copy())[None], "past": torch.from_numpy(np.asarray(raw[past], dtype=np.float32).copy())[None],
                "raw_valid": torch.from_numpy((np.asarray(valid[current]) > 0).copy())[None], "past_valid": torch.from_numpy((np.asarray(valid[past]) > 0).copy())[None],
                "current_rgb": torch.from_numpy(cur[0]).float(), "current_right_rgb": torch.from_numpy(cur[1]).float(), "past_rgb": torch.from_numpy(prev[0]).float(),
                "gt": torch.from_numpy(cur[2])[None], "gt_coverage": torch.from_numpy(cur[3].copy())[None], "past_gt": torch.from_numpy(prev[2])[None], "past_gt_coverage": torch.from_numpy(prev[3].copy())[None],
                "gt_valid": torch.from_numpy((cur[3] > self.coverage_threshold).copy())[None], "gt_depth_mm": torch.from_numpy(cur[4])[None], "focal_baseline_mm": torch.tensor(cur[5], dtype=torch.float32),
                "backbone": BACKBONE, "sequence": recording, "past_frame_id": old["frame_id"], "current_frame_id": row["frame_id"], "past_index": past, "current_index": current, "domain": "drends"}

    def __getitem__(self, index):
        clip = self.clips[index]; frames = [self._pair(clip.recording, i) for i in clip.pair_indices]; out = {}
        for key in frames[0]: out[key] = torch.stack([x[key] for x in frames]) if torch.is_tensor(frames[0][key]) else [x[key] for x in frames]
        out["backbone"], out["sequence"], out["domain"] = BACKBONE, clip.recording, "drends"
        return out
