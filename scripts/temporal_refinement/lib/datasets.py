from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TemporalRefinementCacheDataset(Dataset):
    def __init__(
        self,
        cache_root: str | Path,
        sample_ids: list[int] | None = None,
        crop_size: tuple[int, int] = (256, 512),
        random_crop: bool = True,
    ):
        self.cache_root = Path(cache_root)
        self.crop_size = crop_size
        self.random_crop = random_crop
        with (self.cache_root / "index.csv").open() as f:
            rows = list(csv.DictReader(f))
        if sample_ids is not None:
            wanted = {int(x) for x in sample_ids}
            rows = [r for r in rows if int(r["sample_id"]) in wanted]
        self.rows = rows
        if not self.rows:
            raise RuntimeError(f"No samples found under {cache_root}")

    def __len__(self) -> int:
        return len(self.rows)

    def _crop_origin(self, h: int, w: int) -> tuple[int, int]:
        ch, cw = self.crop_size
        if ch > h or cw > w:
            raise RuntimeError(f"Crop {self.crop_size} larger than image {(h, w)}")
        if self.random_crop:
            return random.randint(0, h - ch), random.randint(0, w - cw)
        return (h - ch) // 2, (w - cw) // 2

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int | bool]:
        row = self.rows[idx]
        sample = np.load(self.cache_root / row["sample_path"])
        rgb = sample["center_rgb"]
        disp_window = sample["s2m2_l736_disp_window"].astype(np.float32)
        teacher = sample["stereoanyvideo_disp_center"].astype(np.float32)
        h, w = rgb.shape[:2]
        y, x = self._crop_origin(h, w)
        ch, cw = self.crop_size
        rgb_crop = rgb[y : y + ch, x : x + cw].astype(np.float32) / 255.0
        disp_crop = disp_window[:, y : y + ch, x : x + cw]
        teacher_crop = teacher[y : y + ch, x : x + cw]

        # Normalize disparity channels by a stable surgical-scale constant.
        disp_scale = 128.0
        rgb_t = torch.from_numpy(rgb_crop).permute(2, 0, 1)
        disp_t = torch.from_numpy(disp_crop / disp_scale)
        teacher_t = torch.from_numpy(teacher_crop).unsqueeze(0)
        s2m2_center = torch.from_numpy(disp_crop[disp_crop.shape[0] // 2]).unsqueeze(0)
        x_t = torch.cat([rgb_t, disp_t], dim=0).float()

        out: dict[str, torch.Tensor | str | int | bool] = {
            "input": x_t,
            "s2m2_center": s2m2_center.float(),
            "teacher": teacher_t.float(),
            "sample_id": int(row["sample_id"]),
            "source_sequence": row["source_sequence"],
            "center_frame_id": row["center_frame_id"],
            "has_gt": row["has_gt"] == "True",
        }
        if "gt_disparity" in sample.files:
            valid = sample["valid_mask"][y : y + ch, x : x + cw].astype(bool)
            gt_disp = sample["gt_disparity"][y : y + ch, x : x + cw].astype(np.float32)
            gt_depth = sample["gt_depth"][y : y + ch, x : x + cw].astype(np.float32)
            out["gt_disp"] = torch.from_numpy(gt_disp).unsqueeze(0).float()
            out["gt_depth"] = torch.from_numpy(gt_depth).unsqueeze(0).float()
            out["valid_mask"] = torch.from_numpy(valid).unsqueeze(0)
            out["fx"] = torch.tensor(float(sample["fx"]))
            out["baseline_mm"] = torch.tensor(float(sample["baseline_mm"]))
        return out


class PairedTemporalRefinementCacheDataset(Dataset):
    def __init__(
        self,
        cache_root: str | Path,
        sample_ids: list[int] | None = None,
        crop_size: tuple[int, int] = (256, 512),
        random_crop: bool = True,
        source_sequence: str | None = None,
    ):
        self.base = TemporalRefinementCacheDataset(cache_root, None, crop_size, random_crop=False)
        self.cache_root = Path(cache_root)
        self.crop_size = crop_size
        self.random_crop = random_crop
        wanted = set(sample_ids) if sample_ids is not None else None
        rows = self.base.rows
        if source_sequence is not None:
            rows = [r for r in rows if r["source_sequence"] == source_sequence]
        by_sequence: dict[str, dict[int, dict]] = {}
        for row in rows:
            by_sequence.setdefault(row["source_sequence"], {})[int(row["sample_id"])] = row
        pairs = []
        for sequence, by_id in by_sequence.items():
            for sid in sorted(by_id):
                if sid - 1 not in by_id:
                    continue
                if wanted is not None and (sid not in wanted or sid - 1 not in wanted):
                    continue
                pairs.append((sid - 1, sid, sequence))
        self.pairs = pairs
        if not self.pairs:
            raise RuntimeError(f"No consecutive pairs found under {cache_root}")

    def __len__(self) -> int:
        return len(self.pairs)

    def _row_for(self, sample_id: int, sequence: str):
        for row in self.base.rows:
            if int(row["sample_id"]) == sample_id and row["source_sequence"] == sequence:
                return row
        raise RuntimeError(f"Missing sample {sample_id} in sequence {sequence}")

    def _load_at_crop(self, sample_id: int, sequence: str, y: int, x: int, ch: int, cw: int):
        row = self._row_for(sample_id, sequence)
        sample = np.load(self.cache_root / row["sample_path"])
        rgb = sample["center_rgb"]
        disp_window = sample["s2m2_l736_disp_window"].astype(np.float32)
        teacher = sample["stereoanyvideo_disp_center"].astype(np.float32)
        rgb_crop = rgb[y : y + ch, x : x + cw].astype(np.float32) / 255.0
        disp_crop = disp_window[:, y : y + ch, x : x + cw]
        teacher_crop = teacher[y : y + ch, x : x + cw]
        rgb_t = torch.from_numpy(rgb_crop).permute(2, 0, 1)
        disp_t = torch.from_numpy(disp_crop / 128.0)
        return {
            "input": torch.cat([rgb_t, disp_t], dim=0).float(),
            "s2m2_center": torch.from_numpy(disp_crop[disp_crop.shape[0] // 2]).unsqueeze(0).float(),
            "teacher": torch.from_numpy(teacher_crop).unsqueeze(0).float(),
            "sample_id": sample_id,
            "source_sequence": sequence,
            "center_frame_id": row["center_frame_id"],
        }

    def __getitem__(self, idx: int):
        prev_id, cur_id, sequence = self.pairs[idx]
        row = self._row_for(cur_id, sequence)
        sample = np.load(self.cache_root / row["sample_path"])
        h, w = sample["center_rgb"].shape[:2]
        ch, cw = self.crop_size
        if self.random_crop:
            y, x = random.randint(0, h - ch), random.randint(0, w - cw)
        else:
            y, x = (h - ch) // 2, (w - cw) // 2
        return {
            "prev": self._load_at_crop(prev_id, sequence, y, x, ch, cw),
            "cur": self._load_at_crop(cur_id, sequence, y, x, ch, cw),
        }


class LegacyPairedTemporalRefinementCacheDataset(Dataset):
    def __init__(
        self,
        cache_root: str | Path,
        sample_ids: list[int] | None = None,
        crop_size: tuple[int, int] = (256, 512),
        random_crop: bool = True,
        source_sequence: str = "consecutive32",
    ):
        self.base = TemporalRefinementCacheDataset(cache_root, None, crop_size, random_crop=False)
        self.cache_root = Path(cache_root)
        self.crop_size = crop_size
        self.random_crop = random_crop
        wanted = set(sample_ids) if sample_ids is not None else None
        rows = [r for r in self.base.rows if r["source_sequence"] == source_sequence]
        by_id = {int(r["sample_id"]): r for r in rows}
        pairs = []
        for sid in sorted(by_id):
            if sid - 1 not in by_id:
                continue
            if wanted is not None and (sid not in wanted or sid - 1 not in wanted):
                continue
            pairs.append((sid - 1, sid))
        self.pairs = pairs
        if not self.pairs:
            raise RuntimeError(f"No consecutive pairs found under {cache_root}")

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_at_crop(self, sample_id: int, y: int, x: int, ch: int, cw: int):
        row = next(r for r in self.base.rows if int(r["sample_id"]) == sample_id)
        sample = np.load(self.cache_root / row["sample_path"])
        rgb = sample["center_rgb"]
        disp_window = sample["s2m2_l736_disp_window"].astype(np.float32)
        teacher = sample["stereoanyvideo_disp_center"].astype(np.float32)
        rgb_crop = rgb[y : y + ch, x : x + cw].astype(np.float32) / 255.0
        disp_crop = disp_window[:, y : y + ch, x : x + cw]
        teacher_crop = teacher[y : y + ch, x : x + cw]
        rgb_t = torch.from_numpy(rgb_crop).permute(2, 0, 1)
        disp_t = torch.from_numpy(disp_crop / 128.0)
        return {
            "input": torch.cat([rgb_t, disp_t], dim=0).float(),
            "s2m2_center": torch.from_numpy(disp_crop[disp_crop.shape[0] // 2]).unsqueeze(0).float(),
            "teacher": torch.from_numpy(teacher_crop).unsqueeze(0).float(),
            "sample_id": sample_id,
            "center_frame_id": row["center_frame_id"],
        }

    def __getitem__(self, idx: int):
        prev_id, cur_id = self.pairs[idx]
        row = next(r for r in self.base.rows if int(r["sample_id"]) == cur_id)
        sample = np.load(self.cache_root / row["sample_path"])
        h, w = sample["center_rgb"].shape[:2]
        ch, cw = self.crop_size
        if self.random_crop:
            y, x = random.randint(0, h - ch), random.randint(0, w - cw)
        else:
            y, x = (h - ch) // 2, (w - cw) // 2
        return {
            "prev": self._load_at_crop(prev_id, y, x, ch, cw),
            "cur": self._load_at_crop(cur_id, y, x, ch, cw),
        }


def split_sample_ids(cache_root: str | Path, val_count: int = 5) -> tuple[list[int], list[int]]:
    with (Path(cache_root) / "index.csv").open() as f:
        rows = list(csv.DictReader(f))
    ids = [int(r["sample_id"]) for r in rows]
    gt_ids = [int(r["sample_id"]) for r in rows if r["has_gt"] == "True"]
    # Keep the only GT sample in validation for metrics.
    val = []
    for gid in gt_ids:
        if gid in ids and gid not in val:
            val.append(gid)
    for sid in ids[-val_count:]:
        if sid not in val:
            val.append(sid)
        if len(val) >= val_count:
            break
    train = [sid for sid in ids if sid not in set(val)]
    return train, val
