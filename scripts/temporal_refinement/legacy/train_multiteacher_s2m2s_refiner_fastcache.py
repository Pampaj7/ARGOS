#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from scripts.temporal_refinement.legacy.train_multiteacher_s2m2s_refiner import (
    PairedMultiTeacherCacheDataset,
    MultiTeacherCacheDataset,
    eval_pairs,
    eval_single,
    save_qualitative,
    set_seed,
    split_by_sequence,
    train_one_epoch,
    write_report,
)
from scripts.temporal_refinement.lib.models import TinyUNetRefiner


class IndexedTemporalMultiTeacherDataset(Dataset):
    def __init__(self, cache_root: Path, sample_ids: list[int] | None, crop_size: tuple[int, int], random_crop: bool):
        self.cache_root = Path(cache_root)
        self.crop_size = crop_size
        self.random_crop = random_crop
        with (self.cache_root / "index.csv").open() as f:
            rows = list(csv.DictReader(f))
        if sample_ids is not None:
            wanted = {int(x) for x in sample_ids}
            rows = [r for r in rows if int(r["sample_id"]) in wanted]
        self.rows = rows
        if not rows:
            raise RuntimeError(f"No rows in {cache_root}")

    def __len__(self):
        return len(self.rows)

    def _crop_origin(self, h: int, w: int):
        ch, cw = self.crop_size
        if self.random_crop:
            return random.randint(0, h - ch), random.randint(0, w - cw)
        return (h - ch) // 2, (w - cw) // 2

    def _load_disp(self, rel_path: str, y: int, x: int, ch: int, cw: int):
        arr = np.load(self.cache_root / rel_path, mmap_mode="r")
        return np.asarray(arr[y : y + ch, x : x + cw], dtype=np.float32)

    def load_row_at(self, row: dict, y: int, x: int, ch: int, cw: int):
        rgb = cv2.imread(str(self.cache_root / row["rgb_center_path"]), cv2.IMREAD_COLOR)
        if rgb is None:
            raise RuntimeError(row["rgb_center_path"])
        rgb = cv2.cvtColor(rgb[y : y + ch, x : x + cw], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        s_window = np.stack(
            [
                self._load_disp(row["s2m2_s512_tminus2_path"], y, x, ch, cw),
                self._load_disp(row["s2m2_s512_tminus1_path"], y, x, ch, cw),
                self._load_disp(row["s2m2_s512_t_path"], y, x, ch, cw),
                self._load_disp(row["s2m2_s512_tplus1_path"], y, x, ch, cw),
                self._load_disp(row["s2m2_s512_tplus2_path"], y, x, ch, cw),
            ]
        )
        s_center = s_window[2]
        l_teacher = self._load_disp(row["s2m2_l736_t_path"], y, x, ch, cw)
        sav_teacher = self._load_disp(row["sav_t_path"], y, x, ch, cw)
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1)
        disp_t = torch.from_numpy(s_window / 128.0)
        return {
            "input": torch.cat([rgb_t, disp_t], dim=0).float(),
            "s_center": torch.from_numpy(s_center).unsqueeze(0).float(),
            "l_teacher": torch.from_numpy(l_teacher).unsqueeze(0).float(),
            "sav_teacher": torch.from_numpy(sav_teacher).unsqueeze(0).float(),
            "sample_id": int(row["sample_id"]),
            "source_sequence": row["sequence_id"],
            "center_frame_id": row["center_frame_id"],
            "has_gt": row["has_gt"] == "True",
        }

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        h, w = int(row["height"]), int(row["width"])
        ch, cw = self.crop_size
        y, x = self._crop_origin(h, w)
        return self.load_row_at(row, y, x, ch, cw)


class PairedIndexedTemporalMultiTeacherDataset(Dataset):
    def __init__(self, cache_root: Path, sample_ids: list[int] | None, crop_size: tuple[int, int], random_crop: bool):
        self.single = IndexedTemporalMultiTeacherDataset(cache_root, None, crop_size, random_crop=False)
        self.cache_root = Path(cache_root)
        self.crop_size = crop_size
        self.random_crop = random_crop
        wanted = set(sample_ids) if sample_ids is not None else None
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
            raise RuntimeError(f"No consecutive pairs in {cache_root}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        prev_row, cur_row = self.pairs[idx]
        h, w = int(cur_row["height"]), int(cur_row["width"])
        ch, cw = self.crop_size
        if self.random_crop:
            y, x = random.randint(0, h - ch), random.randint(0, w - cw)
        else:
            y, x = (h - ch) // 2, (w - cw) // 2
        return {
            "prev": self.single.load_row_at(prev_row, y, x, ch, cw),
            "cur": self.single.load_row_at(cur_row, y, x, ch, cw),
        }


def split_fast_by_sequence(cache_root: Path, val_sequences: int = 1):
    with (cache_root / "index.csv").open() as f:
        rows = list(csv.DictReader(f))
    sequences = sorted({r["sequence_id"] for r in rows})
    val_seq = set(sequences[-val_sequences:])
    train_ids = [int(r["sample_id"]) for r in rows if r["sequence_id"] not in val_seq]
    val_ids = [int(r["sample_id"]) for r in rows if r["sequence_id"] in val_seq]
    return train_ids, val_ids, sorted(val_seq)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-root", type=Path, default=Path("results/03_temporal_refinement/cache/large_v3_s2m2s512_fast"))
    p.add_argument("--out-dir", type=Path, default=Path("results/temporal_refinement_train_unet_s2m2s512_fastcache_benchmark"))
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
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=23)
    return p.parse_args()


def run_training(args):
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "checkpoints").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    train_ids, val_ids, val_sequences = split_fast_by_sequence(args.cache_root)
    crop = (args.crop_height, args.crop_width)
    train_pairs = PairedIndexedTemporalMultiTeacherDataset(args.cache_root, train_ids, crop, random_crop=True)
    val_single = IndexedTemporalMultiTeacherDataset(args.cache_root, val_ids, crop, random_crop=False)
    val_pairs = PairedIndexedTemporalMultiTeacherDataset(args.cache_root, val_ids, crop, random_crop=False)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = args.persistent_workers
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = torch.utils.data.DataLoader(train_pairs, **loader_kwargs)
    model = TinyUNetRefiner(in_channels=8, base_channels=16, residual_clamp_px=4.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp)
    (args.out_dir / "config.yaml").write_text(
        f"""cache_root: {args.cache_root}
out_dir: {args.out_dir}
cache_format: indexed_per_frame_float16_npy
epochs: {args.epochs}
batch_size: {args.batch_size}
crop_size: [{args.crop_height}, {args.crop_width}]
num_workers: {args.num_workers}
eval_every: {args.eval_every}
save_every: {args.save_every}
loss_weights:
  spatial: {args.spatial_weight}
  abs_sav: {args.abs_sav_weight}
  delta_sav: {args.delta_sav_weight}
  residual: {args.res_weight}
  edge: {args.edge_weight}
warmup:
  epochs: {args.warmup_epochs}
  spatial: {args.warmup_spatial_weight}
  abs_sav: {args.warmup_abs_sav_weight}
  delta_sav: {args.warmup_delta_sav_weight}
  residual: {args.warmup_res_weight}
  edge: {args.warmup_edge_weight}
val_sequences: {val_sequences}
"""
    )
    log_path = args.out_dir / "train_log.csv"
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "seconds", "train_loss", "train_loss_spatial", "train_loss_abs_sav", "train_loss_delta_sav", "train_loss_res", "train_loss_edge", "val_mae_refined_to_l", "val_mae_refined_to_sav", "refined_temporal_diff", "teacher_delta_mae"])
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
    warmup_weights = {
        "spatial_weight": args.warmup_spatial_weight,
        "abs_sav_weight": args.warmup_abs_sav_weight,
        "delta_sav_weight": args.warmup_delta_sav_weight,
        "res_weight": args.warmup_res_weight,
        "edge_weight": args.warmup_edge_weight,
    }
    for epoch in range(1, args.epochs + 1):
        if args.warmup_epochs and epoch <= args.warmup_epochs:
            for key, value in warmup_weights.items():
                if value is not None:
                    setattr(args, key, value)
        else:
            for key, value in base_weights.items():
                setattr(args, key, value)
        t0 = time.time()
        train = train_one_epoch(model, loader, opt, scaler, device, args)
        seconds = time.time() - t0
        epoch_seconds.append(seconds)
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            val = eval_single(model, val_single, device)
            temporal = eval_pairs(model, val_pairs, device)
            last_val = val
            last_temporal = temporal
        else:
            val = last_val
            temporal = last_temporal
        score = val["mae_refined_to_l"] + 0.5 * temporal["teacher_delta_mae"]
        if score < best:
            best = score
            best_epoch = epoch
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(), "epoch": epoch, "score": score}, args.out_dir / "checkpoints" / "best.pt")
        if args.save_every and epoch % args.save_every == 0:
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(), "epoch": epoch, "score": score}, args.out_dir / "checkpoints" / f"epoch_{epoch:04d}.pt")
        with log_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "seconds", "train_loss", "train_loss_spatial", "train_loss_abs_sav", "train_loss_delta_sav", "train_loss_res", "train_loss_edge", "val_mae_refined_to_l", "val_mae_refined_to_sav", "refined_temporal_diff", "teacher_delta_mae"])
            writer.writerow({
                "epoch": epoch,
                "seconds": seconds,
                **{f"train_{k}": v for k, v in train.items()},
                "val_mae_refined_to_l": val["mae_refined_to_l"],
                "val_mae_refined_to_sav": val["mae_refined_to_sav"],
                "refined_temporal_diff": temporal["refined_temporal_diff"],
                "teacher_delta_mae": temporal["teacher_delta_mae"],
            })
        print(f"epoch {epoch:03d} seconds={seconds:.1f} val_R_L={val['mae_refined_to_l']:.4f} val_R_SAV={val['mae_refined_to_sav']:.4f} ref_td={temporal['refined_temporal_diff']:.4f}", flush=True)
    metrics = {
        "best_epoch": best_epoch,
        "epoch_seconds": epoch_seconds,
        "seconds_per_epoch_mean": float(np.mean(epoch_seconds)),
        "seconds_per_epoch_median": float(np.median(epoch_seconds)),
        "val": eval_single(model, val_single, device),
        "temporal": eval_pairs(model, val_pairs, device),
        "runtime_seconds": time.time() - start,
        "peak_vram_mb": float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == "cuda" else 0.0,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    save_qualitative(model, val_single, args.out_dir, device)
    return metrics


def main():
    args = parse_args()
    metrics = run_training(args)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
