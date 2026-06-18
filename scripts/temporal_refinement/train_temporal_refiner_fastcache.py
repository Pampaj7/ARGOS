#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from scripts.temporal_refinement.lib.losses import edge_aware_smoothness
from scripts.temporal_refinement.lib.training import (
    colorize,
    eval_pairs,
    eval_single,
    forward_refiner,
    mean_dict,
    save_qualitative,
    set_seed,
    train_one_epoch,
)
from scripts.temporal_refinement.lib.models import ConvGRURefiner, TinyUNetRefiner


TIME_SLOTS = ["tminus2", "tminus1", "t", "tplus1", "tplus2"]


def disp_dir_from_prefix(prefix: str) -> str:
    if prefix == "sav":
        return "sav_disp"
    return f"{prefix}_disp"


def frame_column_for_slot(slot: str) -> str:
    return f"frame_{slot}"


class IndexedTemporalRefinerDataset(Dataset):
    """
    Generic indexed fast-cache dataset.

    It supports different frozen disparity backbones by selecting columns through:
      --backbone-prefix s2m2_s512 / s2m2_l736 / ...

    It keeps output keys compatible with the existing training/eval functions:
      input, s_center, l_teacher, sav_teacher

    Meaning:
      s_center    = center disparity from the selected backbone
      l_teacher   = spatial teacher or backbone anchor, depending on spatial_target
      sav_teacher = temporal teacher, typically StereoAnyVideo
    """

    def __init__(
        self,
        cache_root: Path,
        index_file: str,
        sample_ids: list[int] | None,
        crop_size: tuple[int, int],
        random_crop: bool,
        backbone_prefix: str,
        spatial_teacher_prefix: str,
        temporal_teacher_prefix: str,
        spatial_target: str,
        disp_norm: float,
    ):
        self.cache_root = Path(cache_root)
        self.index_file = index_file
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.backbone_prefix = backbone_prefix
        self.spatial_teacher_prefix = spatial_teacher_prefix
        self.temporal_teacher_prefix = temporal_teacher_prefix
        self.spatial_target = spatial_target
        self.disp_norm = float(disp_norm)

        with (self.cache_root / self.index_file).open() as f:
            rows = list(csv.DictReader(f))

        if sample_ids is not None:
            wanted = {int(x) for x in sample_ids}
            rows = [r for r in rows if int(r["sample_id"]) in wanted]

        self.rows = rows
        if not self.rows:
            raise RuntimeError(f"No rows in {self.cache_root / self.index_file}")

    def __len__(self):
        return len(self.rows)

    def _crop_origin(self, h: int, w: int):
        ch, cw = self.crop_size
        if h < ch or w < cw:
            raise RuntimeError(f"Crop size {self.crop_size} is larger than image size {(h, w)}")
        if self.random_crop:
            return random.randint(0, h - ch), random.randint(0, w - cw)
        return (h - ch) // 2, (w - cw) // 2

    def _path_from_prefix(self, row: dict, prefix: str, slot: str) -> str:
        """
        Prefer an explicit index column, e.g. s2m2_l736_tminus2_path.
        If absent, reconstruct from sequence_id and frame id.
        """
        key = f"{prefix}_{slot}_path"
        if key in row and row[key]:
            return row[key]

        frame_col = frame_column_for_slot(slot)
        if frame_col not in row:
            raise KeyError(f"Missing {key} and fallback frame column {frame_col}")

        seq = row["sequence_id"]
        frame_id = row[frame_col]
        disp_dir = disp_dir_from_prefix(prefix)
        return f"{seq}/{disp_dir}/{frame_id}.npy"

    def _load_disp(self, rel_path: str, y: int, x: int, ch: int, cw: int):
        p = self.cache_root / rel_path
        if not p.exists():
            raise FileNotFoundError(str(p))
        arr = np.load(p, mmap_mode="r")
        return np.asarray(arr[y : y + ch, x : x + cw], dtype=np.float32)

    def load_row_at(self, row: dict, y: int, x: int, ch: int, cw: int):
        rgb_path = self.cache_root / row["rgb_center_path"]
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise RuntimeError(f"Could not read RGB image: {rgb_path}")

        rgb = cv2.cvtColor(rgb[y : y + ch, x : x + cw], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        backbone_window = np.stack(
            [
                self._load_disp(self._path_from_prefix(row, self.backbone_prefix, "tminus2"), y, x, ch, cw),
                self._load_disp(self._path_from_prefix(row, self.backbone_prefix, "tminus1"), y, x, ch, cw),
                self._load_disp(self._path_from_prefix(row, self.backbone_prefix, "t"), y, x, ch, cw),
                self._load_disp(self._path_from_prefix(row, self.backbone_prefix, "tplus1"), y, x, ch, cw),
                self._load_disp(self._path_from_prefix(row, self.backbone_prefix, "tplus2"), y, x, ch, cw),
            ]
        )

        backbone_center = backbone_window[2]

        temporal_teacher = self._load_disp(
            self._path_from_prefix(row, self.temporal_teacher_prefix, "t"),
            y,
            x,
            ch,
            cw,
        )

        if self.spatial_target == "teacher":
            spatial_teacher = self._load_disp(
                self._path_from_prefix(row, self.spatial_teacher_prefix, "t"),
                y,
                x,
                ch,
                cw,
            )
        elif self.spatial_target == "backbone":
            spatial_teacher = backbone_center.copy()
        elif self.spatial_target == "none":
            spatial_teacher = backbone_center.copy()
        else:
            raise ValueError(f"Unknown spatial_target: {self.spatial_target}")

        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1)
        disp_t = torch.from_numpy(backbone_window / self.disp_norm)

        return {
            "input": torch.cat([rgb_t, disp_t], dim=0).float(),
            "s_center": torch.from_numpy(backbone_center).unsqueeze(0).float(),
            "l_teacher": torch.from_numpy(spatial_teacher).unsqueeze(0).float(),
            "sav_teacher": torch.from_numpy(temporal_teacher).unsqueeze(0).float(),
            "sample_id": int(row["sample_id"]),
            "source_sequence": row["sequence_id"],
            "center_frame_id": row["center_frame_id"],
            "has_gt": row.get("has_gt", "False") == "True",
        }

    def load_causal_frame_at(self, row: dict, y: int, x: int, ch: int, cw: int):
        """Load one online timestep: RGB plus only the current backbone disparity."""
        rgb_path = self.cache_root / row["rgb_center_path"]
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise RuntimeError(f"Could not read RGB image: {rgb_path}")

        rgb = cv2.cvtColor(rgb[y : y + ch, x : x + cw], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        backbone_center = self._load_disp(self._path_from_prefix(row, self.backbone_prefix, "t"), y, x, ch, cw)
        temporal_teacher = self._load_disp(self._path_from_prefix(row, self.temporal_teacher_prefix, "t"), y, x, ch, cw)

        if self.spatial_target == "teacher":
            spatial_teacher = self._load_disp(
                self._path_from_prefix(row, self.spatial_teacher_prefix, "t"),
                y,
                x,
                ch,
                cw,
            )
        elif self.spatial_target in {"backbone", "none"}:
            spatial_teacher = backbone_center.copy()
        else:
            raise ValueError(f"Unknown spatial_target: {self.spatial_target}")

        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1)
        disp_t = torch.from_numpy(backbone_center / self.disp_norm).unsqueeze(0)

        return {
            "input": torch.cat([rgb_t, disp_t], dim=0).float(),
            "s_center": torch.from_numpy(backbone_center).unsqueeze(0).float(),
            "l_teacher": torch.from_numpy(spatial_teacher).unsqueeze(0).float(),
            "sav_teacher": torch.from_numpy(temporal_teacher).unsqueeze(0).float(),
            "sample_id": int(row["sample_id"]),
            "source_sequence": row["sequence_id"],
            "center_frame_id": row["center_frame_id"],
            "has_gt": row.get("has_gt", "False") == "True",
        }

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        h, w = int(row["height"]), int(row["width"])
        ch, cw = self.crop_size
        y, x = self._crop_origin(h, w)
        return self.load_row_at(row, y, x, ch, cw)


class PairedIndexedTemporalRefinerDataset(Dataset):
    def __init__(
        self,
        cache_root: Path,
        index_file: str,
        sample_ids: list[int] | None,
        crop_size: tuple[int, int],
        random_crop: bool,
        backbone_prefix: str,
        spatial_teacher_prefix: str,
        temporal_teacher_prefix: str,
        spatial_target: str,
        disp_norm: float,
    ):
        self.single = IndexedTemporalRefinerDataset(
            cache_root=cache_root,
            index_file=index_file,
            sample_ids=None,
            crop_size=crop_size,
            random_crop=False,
            backbone_prefix=backbone_prefix,
            spatial_teacher_prefix=spatial_teacher_prefix,
            temporal_teacher_prefix=temporal_teacher_prefix,
            spatial_target=spatial_target,
            disp_norm=disp_norm,
        )
        self.crop_size = crop_size
        self.random_crop = random_crop

        wanted = {int(x) for x in sample_ids} if sample_ids is not None else None

        by_seq: dict[str, dict[int, dict]] = {}
        for row in self.single.rows:
            by_seq.setdefault(row["sequence_id"], {})[int(row["sample_id"])] = row

        self.pairs = []
        for _seq, by_id in by_seq.items():
            for sid in sorted(by_id):
                if sid - 1 not in by_id:
                    continue
                if wanted is not None and (sid not in wanted or sid - 1 not in wanted):
                    continue
                self.pairs.append((by_id[sid - 1], by_id[sid]))

        if not self.pairs:
            raise RuntimeError(f"No consecutive pairs in {cache_root / index_file}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        prev_row, cur_row = self.pairs[idx]
        h, w = int(cur_row["height"]), int(cur_row["width"])
        ch, cw = self.crop_size

        if h < ch or w < cw:
            raise RuntimeError(f"Crop size {self.crop_size} is larger than image size {(h, w)}")

        if self.random_crop:
            y, x = random.randint(0, h - ch), random.randint(0, w - cw)
        else:
            y, x = (h - ch) // 2, (w - cw) // 2

        return {
            "prev": self.single.load_row_at(prev_row, y, x, ch, cw),
            "cur": self.single.load_row_at(cur_row, y, x, ch, cw),
        }


class ClipIndexedTemporalRefinerDataset(Dataset):
    """Causal clip dataset for recurrent temporal refinement."""

    def __init__(
        self,
        cache_root: Path,
        index_file: str,
        sample_ids: list[int] | None,
        crop_size: tuple[int, int],
        random_crop: bool,
        backbone_prefix: str,
        spatial_teacher_prefix: str,
        temporal_teacher_prefix: str,
        spatial_target: str,
        disp_norm: float,
        sequence_length: int,
    ):
        self.single = IndexedTemporalRefinerDataset(
            cache_root=cache_root,
            index_file=index_file,
            sample_ids=None,
            crop_size=crop_size,
            random_crop=False,
            backbone_prefix=backbone_prefix,
            spatial_teacher_prefix=spatial_teacher_prefix,
            temporal_teacher_prefix=temporal_teacher_prefix,
            spatial_target=spatial_target,
            disp_norm=disp_norm,
        )
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.sequence_length = sequence_length
        wanted = {int(x) for x in sample_ids} if sample_ids is not None else None
        by_seq: dict[str, list[dict]] = {}
        for row in self.single.rows:
            sid = int(row["sample_id"])
            if wanted is not None and sid not in wanted:
                continue
            by_seq.setdefault(row["sequence_id"], []).append(row)
        self.clips: list[list[dict]] = []
        for _seq, rows in by_seq.items():
            rows = sorted(rows, key=lambda r: int(r["center_frame_id"]))
            if sequence_length <= 0:
                self.clips.append(rows)
                continue
            for start in range(0, len(rows) - sequence_length + 1):
                self.clips.append(rows[start : start + sequence_length])
        if not self.clips:
            raise RuntimeError(f"No clips length {sequence_length} in {cache_root / index_file}")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx: int):
        rows = self.clips[idx]
        h, w = int(rows[0]["height"]), int(rows[0]["width"])
        ch, cw = self.crop_size
        if h < ch or w < cw:
            raise RuntimeError(f"Crop size {self.crop_size} is larger than image size {(h, w)}")
        if self.random_crop:
            y, x = random.randint(0, h - ch), random.randint(0, w - cw)
        else:
            y, x = (h - ch) // 2, (w - cw) // 2
        frames = [self.single.load_causal_frame_at(row, y, x, ch, cw) for row in rows]
        out: dict[str, torch.Tensor | list] = {}
        tensor_keys = ["input", "s_center", "l_teacher", "sav_teacher"]
        for key in tensor_keys:
            out[key] = torch.stack([frame[key] for frame in frames], dim=0)
        out["sample_id"] = [frame["sample_id"] for frame in frames]
        out["source_sequence"] = frames[0]["source_sequence"]
        out["center_frame_id"] = [frame["center_frame_id"] for frame in frames]
        out["has_gt"] = [frame["has_gt"] for frame in frames]
        return out


def split_fast_by_sequence(cache_root: Path, index_file: str, val_sequences: int = 1):
    with (cache_root / index_file).open() as f:
        rows = list(csv.DictReader(f))

    sequences = sorted({r["sequence_id"] for r in rows})
    val_seq = set(sequences[-val_sequences:])

    train_ids = [int(r["sample_id"]) for r in rows if r["sequence_id"] not in val_seq]
    val_ids = [int(r["sample_id"]) for r in rows if r["sequence_id"] in val_seq]

    return train_ids, val_ids, sorted(val_seq)


def json_safe_args(args) -> dict:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


LOSS_WEIGHT_KEYS = ("spatial", "abs_sav", "delta_sav", "res", "edge")


def loss_weights_for_epoch(args, epoch: int, initial_weights: dict[str, float] | None = None) -> dict[str, float]:
    """Return active loss weights for 1-indexed epoch.

    Static behavior is preserved unless --loss-schedule is enabled. Scheduled
    behavior keeps the initial CLI weights for schedule_warmup_epochs, then
    linearly interpolates to final weights over schedule_transition_epochs.
    """
    initial = initial_weights or {
        "spatial": float(args.spatial_weight),
        "abs_sav": float(args.abs_sav_weight),
        "delta_sav": float(args.delta_sav_weight),
        "res": float(args.res_weight),
        "edge": float(args.edge_weight),
    }
    if not getattr(args, "loss_schedule", False):
        return initial

    final = {
        "spatial": float(args.final_spatial_weight),
        "abs_sav": float(args.final_abs_sav_weight),
        "delta_sav": float(args.final_delta_sav_weight),
        "res": float(args.final_res_weight),
        "edge": float(args.final_edge_weight),
    }

    warmup_epochs = max(0, int(args.schedule_warmup_epochs))
    transition_epochs = max(0, int(args.schedule_transition_epochs))

    if epoch <= warmup_epochs or transition_epochs == 0:
        alpha = 0.0 if epoch <= warmup_epochs else 1.0
    else:
        transition_epoch = epoch - warmup_epochs
        alpha = min(1.0, transition_epoch / transition_epochs)

    return {key: initial[key] + alpha * (final[key] - initial[key]) for key in LOSS_WEIGHT_KEYS}


def apply_loss_weights(args, weights: dict[str, float]) -> None:
    args.spatial_weight = weights["spatial"]
    args.abs_sav_weight = weights["abs_sav"]
    args.delta_sav_weight = weights["delta_sav"]
    args.res_weight = weights["res"]
    args.edge_weight = weights["edge"]


def clip_to_device(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def make_grad_scaler(device: torch.device, amp_enabled: bool):
    enabled = device.type == "cuda" and amp_enabled
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def ddp_info() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return world_size > 1, rank, local_rank, world_size


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def ddp_mean_dict(values: dict[str, float], device: torch.device) -> dict[str, float]:
    if not dist.is_available() or not dist.is_initialized():
        return values
    keys = sorted(values)
    if not keys:
        return values
    tensor = torch.tensor([float(values[key]) for key in keys], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return {key: float(value) for key, value in zip(keys, tensor.cpu().tolist())}


def convgru_forward_clip(model: ConvGRURefiner, batch, device):
    batch = clip_to_device(batch, device)
    inputs = batch["input"]
    backbone = batch["s_center"]
    hidden = None
    deltas = []
    refined = []
    for t in range(inputs.shape[1]):
        x_t = torch.cat([inputs[:, t, :3], inputs[:, t, 3:4]], dim=1)
        delta_t, hidden = model(x_t, hidden)
        refined_t = torch.clamp(backbone[:, t] + delta_t, min=0.0)
        deltas.append(delta_t)
        refined.append(refined_t)
    return batch, torch.stack(deltas, dim=1), torch.stack(refined, dim=1)


def convgru_clip_loss(model, batch, device, args):
    b, delta, refined = convgru_forward_clip(model, batch, device)
    spatial = F.smooth_l1_loss(refined, b["l_teacher"])
    abs_sav = F.smooth_l1_loss(refined, b["sav_teacher"])
    if refined.shape[1] > 1:
        delta_sav = F.smooth_l1_loss(refined[:, 1:] - refined[:, :-1], b["sav_teacher"][:, 1:] - b["sav_teacher"][:, :-1])
    else:
        delta_sav = refined.new_tensor(0.0)
    residual = torch.mean(torch.abs(delta))
    edge_terms = [edge_aware_smoothness(delta[:, t], b["input"][:, t, :3]) for t in range(delta.shape[1])]
    edge = torch.stack(edge_terms).mean()
    loss = (
        args.spatial_weight * spatial
        + args.abs_sav_weight * abs_sav
        + args.delta_sav_weight * delta_sav
        + args.res_weight * residual
        + args.edge_weight * edge
    )
    return loss, {
        "loss_spatial": float(spatial.detach().cpu()),
        "loss_abs_sav": float(abs_sav.detach().cpu()),
        "loss_delta_sav": float(delta_sav.detach().cpu()),
        "loss_res": float(residual.detach().cpu()),
        "loss_edge": float(edge.detach().cpu()),
    }


def train_convgru_one_epoch(model, loader, optimizer, scaler, device, args):
    model.train()
    rows = []
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda" and args.amp):
            loss, parts = convgru_clip_loss(model, batch, device, args)
        scaler.scale(loss).backward()
        if args.grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        parts["loss"] = float(loss.detach().cpu())
        rows.append(parts)
    return mean_dict(rows)


@torch.no_grad()
def eval_convgru_clips(model, dataset, device):
    model.eval()
    rows = []
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for batch in loader:
        b, delta, refined = convgru_forward_clip(model, batch, device)
        backbone = b["s_center"]
        spatial = b["l_teacher"]
        temporal = b["sav_teacher"]
        row = {
            "mae_s_to_l": float(torch.mean(torch.abs(backbone - spatial)).cpu()),
            "mae_refined_to_l": float(torch.mean(torch.abs(refined - spatial)).cpu()),
            "mae_s_to_sav": float(torch.mean(torch.abs(backbone - temporal)).cpu()),
            "mae_refined_to_sav": float(torch.mean(torch.abs(refined - temporal)).cpu()),
            "residual_abs_mean": float(torch.mean(torch.abs(delta)).cpu()),
            "residual_mean": float(delta.mean().cpu()),
            "residual_std": float(delta.std().cpu()),
            "residual_min": float(delta.min().cpu()),
            "residual_max": float(delta.max().cpu()),
        }
        if refined.shape[1] > 1:
            row.update(
                {
                    "backbone_temporal_diff": float(torch.mean(torch.abs(backbone[:, 1:] - backbone[:, :-1])).cpu()),
                    "refined_temporal_diff": float(torch.mean(torch.abs(refined[:, 1:] - refined[:, :-1])).cpu()),
                    "sav_temporal_diff": float(torch.mean(torch.abs(temporal[:, 1:] - temporal[:, :-1])).cpu()),
                    "teacher_delta_mae": float(torch.mean(torch.abs((refined[:, 1:] - refined[:, :-1]) - (temporal[:, 1:] - temporal[:, :-1]))).cpu()),
                }
            )
        rows.append(row)
    return mean_dict(rows) | {"clip_count": float(len(rows))}


@torch.no_grad()
def save_convgru_qualitative(model, dataset, out_dir: Path, device, max_items=8):
    qdir = out_dir / "qualitative"
    qdir.mkdir(parents=True, exist_ok=True)
    chosen = np.linspace(0, len(dataset) - 1, min(max_items, len(dataset)), dtype=int)
    for idx in chosen:
        batch = dataset[int(idx)]
        collated = {key: value.unsqueeze(0) if torch.is_tensor(value) else [value] for key, value in batch.items()}
        b, delta, refined = convgru_forward_clip(model, collated, device)
        t = refined.shape[1] // 2
        rgb = (b["input"][0, t, :3].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        s = b["s_center"][0, t, 0].cpu().numpy()
        l = b["l_teacher"][0, t, 0].cpu().numpy()
        sav = b["sav_teacher"][0, t, 0].cpu().numpy()
        r = refined[0, t, 0].cpu().numpy()
        d = delta[0, t, 0].cpu().numpy()
        vmax = float(np.nanpercentile(np.concatenate([s.ravel(), l.ravel(), sav.ravel(), r.ravel()]), 99))
        tiles = [
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            colorize(s, vmax),
            colorize(r, vmax),
            colorize(l, vmax),
            colorize(sav, vmax),
            colorize(np.abs(r - l), 8.0, cv2.COLORMAP_MAGMA),
            colorize(np.abs(r - sav), 8.0, cv2.COLORMAP_MAGMA),
            colorize(d, 2.0, cv2.COLORMAP_VIRIDIS),
        ]
        labels = ["RGB", "backbone", "refined", "spatial T", "temporal T", "|R-S|", "|R-T|", "delta"]
        small = []
        for tile, label in zip(tiles, labels):
            tile = cv2.resize(tile, (150, 105), interpolation=cv2.INTER_AREA)
            cv2.putText(tile, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)
            small.append(tile)
        cv2.imwrite(str(qdir / f"clip_{int(batch['sample_id'][t]):06d}.png"), np.concatenate(small, axis=1))


def parse_args(argv=None):
    p = argparse.ArgumentParser()

    p.add_argument("--cache-root", type=Path, default=Path("results/temporal_refinement_cache/large_v3_s2m2s512_fast"))
    p.add_argument("--index-file", type=str, default="index.csv")
    p.add_argument("--out-dir", type=Path, default=Path("results/temporal_refinement_train_fastcache_generic"))

    p.add_argument("--model", choices=["tiny_unet", "convgru"], default="tiny_unet")
    p.add_argument("--backbone-prefix", type=str, default="s2m2_s512")
    p.add_argument("--spatial-teacher-prefix", type=str, default="s2m2_l736")
    p.add_argument("--temporal-teacher-prefix", type=str, default="sav")
    p.add_argument("--spatial-target", choices=["teacher", "backbone", "none"], default="teacher")

    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--crop-height", "--crop-h", type=int, default=384)
    p.add_argument("--crop-width", "--crop-w", type=int, default=640)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--persistent-workers", action="store_true", default=False)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=0)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--spatial-weight", type=float, default=0.40)
    p.add_argument("--abs-sav-weight", type=float, default=0.25)
    p.add_argument("--delta-sav-weight", type=float, default=0.25)
    p.add_argument("--res-weight", type=float, default=0.10)
    p.add_argument("--edge-weight", type=float, default=0.05)

    p.add_argument("--warmup-epochs", type=int, default=0)
    p.add_argument("--warmup-spatial-weight", type=float, default=None)
    p.add_argument("--warmup-abs-sav-weight", type=float, default=None)
    p.add_argument("--warmup-delta-sav-weight", type=float, default=None)
    p.add_argument("--warmup-res-weight", type=float, default=None)
    p.add_argument("--warmup-edge-weight", type=float, default=None)

    p.add_argument("--loss-schedule", action="store_true", default=False)
    p.add_argument("--schedule-warmup-epochs", type=int, default=0)
    p.add_argument("--schedule-transition-epochs", type=int, default=0)
    p.add_argument("--final-spatial-weight", type=float, default=0.40)
    p.add_argument("--final-abs-sav-weight", type=float, default=0.25)
    p.add_argument("--final-delta-sav-weight", type=float, default=0.25)
    p.add_argument("--final-res-weight", type=float, default=0.10)
    p.add_argument("--final-edge-weight", type=float, default=0.05)

    p.add_argument("--disp-norm", type=float, default=128.0)
    p.add_argument("--base-channels", type=int, default=16)
    p.add_argument("--hidden-channels", type=int, default=64)
    p.add_argument("--sequence-length", type=int, default=5)
    p.add_argument("--eval-full-sequences", action="store_true", default=False)
    p.add_argument("--residual-clamp-px", type=float, default=4.0)
    p.add_argument("--grad-clip-norm", type=float, default=0.0)
    p.add_argument("--score-spatial-weight", type=float, default=0.5)

    p.add_argument("--val-sequences", type=int, default=1)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--seed", type=int, default=23)

    return p.parse_args(argv)


def run_training(args):
    ddp, rank, local_rank, world_size = ddp_info()
    if ddp:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    set_seed(args.seed + rank)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "checkpoints").mkdir(exist_ok=True)

    device = torch.device(f"cuda:{local_rank}" if ddp else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    train_ids, val_ids, val_sequences = split_fast_by_sequence(
        args.cache_root,
        args.index_file,
        val_sequences=args.val_sequences,
    )
    if args.max_train_samples and args.max_train_samples > 0:
        train_ids = train_ids[: args.max_train_samples]
    if args.max_val_samples and args.max_val_samples > 0:
        val_ids = val_ids[: args.max_val_samples]

    crop = (args.crop_height, args.crop_width)

    train_pairs = PairedIndexedTemporalRefinerDataset(
        cache_root=args.cache_root,
        index_file=args.index_file,
        sample_ids=train_ids,
        crop_size=crop,
        random_crop=True,
        backbone_prefix=args.backbone_prefix,
        spatial_teacher_prefix=args.spatial_teacher_prefix,
        temporal_teacher_prefix=args.temporal_teacher_prefix,
        spatial_target=args.spatial_target,
        disp_norm=args.disp_norm,
    )

    val_single = IndexedTemporalRefinerDataset(
        cache_root=args.cache_root,
        index_file=args.index_file,
        sample_ids=val_ids,
        crop_size=crop,
        random_crop=False,
        backbone_prefix=args.backbone_prefix,
        spatial_teacher_prefix=args.spatial_teacher_prefix,
        temporal_teacher_prefix=args.temporal_teacher_prefix,
        spatial_target=args.spatial_target,
        disp_norm=args.disp_norm,
    )

    val_pairs = PairedIndexedTemporalRefinerDataset(
        cache_root=args.cache_root,
        index_file=args.index_file,
        sample_ids=val_ids,
        crop_size=crop,
        random_crop=False,
        backbone_prefix=args.backbone_prefix,
        spatial_teacher_prefix=args.spatial_teacher_prefix,
        temporal_teacher_prefix=args.temporal_teacher_prefix,
        spatial_target=args.spatial_target,
        disp_norm=args.disp_norm,
    )
    train_clips = None
    val_clips = None
    if args.model == "convgru":
        train_clips = ClipIndexedTemporalRefinerDataset(
            cache_root=args.cache_root,
            index_file=args.index_file,
            sample_ids=train_ids,
            crop_size=crop,
            random_crop=True,
            backbone_prefix=args.backbone_prefix,
            spatial_teacher_prefix=args.spatial_teacher_prefix,
            temporal_teacher_prefix=args.temporal_teacher_prefix,
            spatial_target=args.spatial_target,
            disp_norm=args.disp_norm,
            sequence_length=args.sequence_length,
        )
        val_clips = ClipIndexedTemporalRefinerDataset(
            cache_root=args.cache_root,
            index_file=args.index_file,
            sample_ids=val_ids,
            crop_size=crop,
            random_crop=False,
            backbone_prefix=args.backbone_prefix,
            spatial_teacher_prefix=args.spatial_teacher_prefix,
            temporal_teacher_prefix=args.temporal_teacher_prefix,
            spatial_target=args.spatial_target,
            disp_norm=args.disp_norm,
            sequence_length=0 if args.eval_full_sequences else args.sequence_length,
        )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }

    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = args.persistent_workers
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_dataset = train_clips if args.model == "convgru" else train_pairs
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None
    if train_sampler is not None:
        loader_kwargs["sampler"] = train_sampler
        loader_kwargs["shuffle"] = False

    loader = torch.utils.data.DataLoader(train_dataset, **loader_kwargs)

    if args.model == "convgru":
        model = ConvGRURefiner(
            in_channels=4,
            base_channels=args.base_channels,
            hidden_channels=args.hidden_channels,
            residual_clamp_px=args.residual_clamp_px,
        ).to(device)
    else:
        model = TinyUNetRefiner(
            in_channels=8,
            base_channels=args.base_channels,
            residual_clamp_px=args.residual_clamp_px,
        ).to(device)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = make_grad_scaler(device, args.amp)
    start_epoch = 1
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1

    if is_main_process():
        (args.out_dir / "config.yaml").write_text(
            f"""cache_root: {args.cache_root}
index_file: {args.index_file}
out_dir: {args.out_dir}
cache_format: indexed_per_frame_float16_npy
ddp:
  enabled: {ddp}
  world_size: {world_size}
model_name: {args.model}
backbone_prefix: {args.backbone_prefix}
spatial_teacher_prefix: {args.spatial_teacher_prefix}
temporal_teacher_prefix: {args.temporal_teacher_prefix}
spatial_target: {args.spatial_target}
epochs: {args.epochs}
batch_size: {args.batch_size}
crop_size: [{args.crop_height}, {args.crop_width}]
num_workers: {args.num_workers}
persistent_workers: {args.persistent_workers}
prefetch_factor: {args.prefetch_factor}
eval_every: {args.eval_every}
save_every: {args.save_every}
max_train_samples: {args.max_train_samples}
max_val_samples: {args.max_val_samples}
lr: {args.lr}
model:
  type: {"ConvGRURefiner" if args.model == "convgru" else "TinyUNetRefiner"}
  base_channels: {args.base_channels}
  hidden_channels: {args.hidden_channels}
  sequence_length: {args.sequence_length}
  eval_full_sequences: {args.eval_full_sequences}
  residual_clamp_px: {args.residual_clamp_px}
  disp_norm: {args.disp_norm}
  grad_clip_norm: {args.grad_clip_norm}
score:
  teacher_delta_mae_plus_spatial_weighted_by: {args.score_spatial_weight}
loss_weights:
  spatial: {args.spatial_weight}
  abs_sav: {args.abs_sav_weight}
  delta_sav: {args.delta_sav_weight}
  residual: {args.res_weight}
  edge: {args.edge_weight}
loss_schedule:
  enabled: {args.loss_schedule}
  warmup_epochs: {args.schedule_warmup_epochs}
  transition_epochs: {args.schedule_transition_epochs}
  final_spatial: {args.final_spatial_weight}
  final_abs_sav: {args.final_abs_sav_weight}
  final_delta_sav: {args.final_delta_sav_weight}
  final_residual: {args.final_res_weight}
  final_edge: {args.final_edge_weight}
warmup:
  epochs: {args.warmup_epochs}
  spatial: {args.warmup_spatial_weight}
  abs_sav: {args.warmup_abs_sav_weight}
  delta_sav: {args.warmup_delta_sav_weight}
  residual: {args.warmup_res_weight}
  edge: {args.warmup_edge_weight}
val_sequences: {val_sequences}
resume: {args.resume}
"""
        )

    log_path = args.out_dir / "train_log.csv"
    fieldnames = [
        "epoch",
        "seconds",
        "train_loss",
        "train_loss_spatial",
        "train_loss_abs_sav",
        "train_loss_delta_sav",
        "train_loss_res",
        "train_loss_edge",
        "active_spatial_weight",
        "active_abs_sav_weight",
        "active_delta_sav_weight",
        "active_res_weight",
        "active_edge_weight",
        "val_mae_refined_to_l",
        "val_mae_refined_to_sav",
        "backbone_temporal_diff",
        "refined_temporal_diff",
        "sav_temporal_diff",
        "teacher_delta_mae",
        "residual_abs_mean",
        "residual_std",
        "runtime_ms_per_frame",
    ]

    if is_main_process():
        with log_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    start = time.time()
    epoch_seconds = []

    best = float("inf")
    best_epoch = -1
    last_val = {}
    last_temporal = {}

    base_weights = {
        "spatial_weight": args.spatial_weight,
        "abs_sav_weight": args.abs_sav_weight,
        "delta_sav_weight": args.delta_sav_weight,
        "res_weight": args.res_weight,
        "edge_weight": args.edge_weight,
    }
    schedule_initial_weights = {
        "spatial": args.spatial_weight,
        "abs_sav": args.abs_sav_weight,
        "delta_sav": args.delta_sav_weight,
        "res": args.res_weight,
        "edge": args.edge_weight,
    }

    warmup_weights = {
        "spatial_weight": args.warmup_spatial_weight,
        "abs_sav_weight": args.warmup_abs_sav_weight,
        "delta_sav_weight": args.warmup_delta_sav_weight,
        "res_weight": args.warmup_res_weight,
        "edge_weight": args.warmup_edge_weight,
    }

    if is_main_process():
        print("Starting generic temporal-refiner training")
        print(f"device={device}")
        print(f"ddp={ddp} world_size={world_size}")
        print(f"model={args.model}")
        print(f"cache_root={args.cache_root}")
        print(f"index_file={args.index_file}")
        print(f"backbone_prefix={args.backbone_prefix}")
        print(f"spatial_teacher_prefix={args.spatial_teacher_prefix}")
        print(f"temporal_teacher_prefix={args.temporal_teacher_prefix}")
        print(f"spatial_target={args.spatial_target}")
        if args.model == "convgru":
            print(
                f"train_clips={len(train_clips)} val_clips={len(val_clips)} "
                f"sequence_length={args.sequence_length} eval_full_sequences={args.eval_full_sequences}",
                flush=True,
            )
        else:
            print(f"train_pairs={len(train_pairs)} val_single={len(val_single)} val_pairs={len(val_pairs)}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if args.loss_schedule:
            active_weights = loss_weights_for_epoch(args, epoch, schedule_initial_weights)
            apply_loss_weights(args, active_weights)
        elif args.warmup_epochs and epoch <= args.warmup_epochs:
            for key, value in warmup_weights.items():
                if value is not None:
                    setattr(args, key, value)
            active_weights = {
                "spatial": args.spatial_weight,
                "abs_sav": args.abs_sav_weight,
                "delta_sav": args.delta_sav_weight,
                "res": args.res_weight,
                "edge": args.edge_weight,
            }
        else:
            for key, value in base_weights.items():
                setattr(args, key, value)
            active_weights = {
                "spatial": args.spatial_weight,
                "abs_sav": args.abs_sav_weight,
                "delta_sav": args.delta_sav_weight,
                "res": args.res_weight,
                "edge": args.edge_weight,
            }

        if is_main_process():
            print(
                "active_loss_weights "
                f"epoch={epoch} "
                f"spatial={active_weights['spatial']:.6f} "
                f"abs_sav={active_weights['abs_sav']:.6f} "
                f"delta_sav={active_weights['delta_sav']:.6f} "
                f"res={active_weights['res']:.6f} "
                f"edge={active_weights['edge']:.6f}",
                flush=True,
            )

        t0 = time.time()
        train = train_convgru_one_epoch(model, loader, opt, scaler, device, args) if args.model == "convgru" else train_one_epoch(model, loader, opt, scaler, device, args)
        train = ddp_mean_dict(train, device)
        seconds = time.time() - t0
        epoch_seconds.append(seconds)

        if is_main_process() and (epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs):
            if args.model == "convgru":
                val = eval_convgru_clips(unwrap_model(model), val_clips, device)
                temporal = val
            else:
                val = eval_single(unwrap_model(model), val_single, device)
                temporal = eval_pairs(unwrap_model(model), val_pairs, device)
            last_val = val
            last_temporal = temporal
        elif is_main_process():
            val = last_val
            temporal = last_temporal
        if ddp:
            dist.barrier()
        if not is_main_process():
            if ddp:
                dist.barrier()
            continue

        if args.model == "convgru":
            score = temporal["teacher_delta_mae"] + args.score_spatial_weight * val["mae_refined_to_l"]
        else:
            score = val["mae_refined_to_l"] + 0.5 * temporal["teacher_delta_mae"]
        checkpoint = {
            "model_state_dict": unwrap_model(model).state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "epoch": epoch,
            "score": score,
            "active_loss_weights": active_weights,
            "args": json_safe_args(args),
        }
        torch.save(checkpoint, args.out_dir / "checkpoints" / "latest.pt")

        if score < best:
            best = score
            best_epoch = epoch
            torch.save(checkpoint, args.out_dir / "checkpoints" / "best.pt")

        if args.save_every and epoch % args.save_every == 0:
            torch.save(checkpoint, args.out_dir / "checkpoints" / f"epoch_{epoch:04d}.pt")

        with log_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(
                {
                    "epoch": epoch,
                    "seconds": seconds,
                    **{f"train_{k}": v for k, v in train.items()},
                    "active_spatial_weight": active_weights["spatial"],
                    "active_abs_sav_weight": active_weights["abs_sav"],
                    "active_delta_sav_weight": active_weights["delta_sav"],
                    "active_res_weight": active_weights["res"],
                    "active_edge_weight": active_weights["edge"],
                    "val_mae_refined_to_l": val["mae_refined_to_l"],
                    "val_mae_refined_to_sav": val["mae_refined_to_sav"],
                    "backbone_temporal_diff": temporal.get("backbone_temporal_diff", temporal.get("s2m2s_temporal_diff", "")),
                    "refined_temporal_diff": temporal["refined_temporal_diff"],
                    "sav_temporal_diff": temporal.get("sav_temporal_diff", ""),
                    "teacher_delta_mae": temporal["teacher_delta_mae"],
                    "residual_abs_mean": val.get("residual_abs_mean", ""),
                    "residual_std": val.get("residual_std", ""),
                    "runtime_ms_per_frame": (seconds * 1000.0 / (len(loader.dataset) * args.sequence_length)) if args.model == "convgru" else (seconds * 1000.0 / (len(loader.dataset) * 2)),
                }
            )

        print(
            f"epoch {epoch:03d} "
            f"seconds={seconds:.1f} "
            f"val_R_L={val['mae_refined_to_l']:.4f} "
            f"val_R_SAV={val['mae_refined_to_sav']:.4f} "
            f"ref_td={temporal['refined_temporal_diff']:.4f} "
            f"teacher_delta={temporal['teacher_delta_mae']:.4f}",
            flush=True,
        )
        if ddp:
            dist.barrier()

    if not is_main_process():
        if ddp:
            dist.destroy_process_group()
        return {}

    if args.model == "convgru":
        final_val = eval_convgru_clips(unwrap_model(model), val_clips, device)
        final_temporal = final_val
    else:
        final_val = eval_single(unwrap_model(model), val_single, device)
        final_temporal = eval_pairs(unwrap_model(model), val_pairs, device)

    metrics = {
        "best_epoch": best_epoch,
        "epoch_seconds": epoch_seconds,
        "seconds_per_epoch_mean": float(np.mean(epoch_seconds)),
        "seconds_per_epoch_median": float(np.median(epoch_seconds)),
        "val": final_val,
        "temporal": final_temporal,
        "runtime_seconds": time.time() - start,
        "peak_vram_mb": float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == "cuda" else 0.0,
        "loss_schedule": {
            "enabled": args.loss_schedule,
            "warmup_epochs": args.schedule_warmup_epochs,
            "transition_epochs": args.schedule_transition_epochs,
            "final_spatial": args.final_spatial_weight,
            "final_abs_sav": args.final_abs_sav_weight,
            "final_delta_sav": args.final_delta_sav_weight,
            "final_res": args.final_res_weight,
            "final_edge": args.final_edge_weight,
        },
        "args": json_safe_args(args),
    }

    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if args.model == "convgru":
        save_convgru_qualitative(unwrap_model(model), val_clips, args.out_dir, device)
    else:
        save_qualitative(unwrap_model(model), val_single, args.out_dir, device)

    if ddp:
        dist.destroy_process_group()
    return metrics


def main():
    args = parse_args()
    metrics = run_training(args)
    if is_main_process():
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
