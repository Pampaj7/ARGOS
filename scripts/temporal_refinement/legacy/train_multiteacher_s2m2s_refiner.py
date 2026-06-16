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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from scripts.temporal_refinement.lib.losses import edge_aware_smoothness
from scripts.temporal_refinement.lib.models import TinyUNetRefiner
from scripts.temporal_refinement.legacy.train_debug_unet_refiner import colorize, mean_dict


class MultiTeacherCacheDataset(Dataset):
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
        self.by_id = {int(r["sample_id"]): r for r in rows}
        if not rows:
            raise RuntimeError(f"No rows in {cache_root}")

    def __len__(self):
        return len(self.rows)

    def crop_origin(self, h: int, w: int):
        ch, cw = self.crop_size
        if self.random_crop:
            return random.randint(0, h - ch), random.randint(0, w - cw)
        return (h - ch) // 2, (w - cw) // 2

    def load_row(self, row: dict, y: int, x: int, ch: int, cw: int):
        sample = np.load(self.cache_root / row["sample_path"])
        rgb = sample["center_rgb"][y : y + ch, x : x + cw].astype(np.float32) / 255.0
        s_window = sample["s2m2_s512_disp_window"][:, y : y + ch, x : x + cw].astype(np.float32)
        s_center = sample["s2m2_s512_disp_center"][y : y + ch, x : x + cw].astype(np.float32)
        l_teacher = sample["s2m2_l736_disp_center"][y : y + ch, x : x + cw].astype(np.float32)
        sav_teacher = sample["stereoanyvideo_disp_center"][y : y + ch, x : x + cw].astype(np.float32)
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1)
        disp_t = torch.from_numpy(s_window / 128.0)
        return {
            "input": torch.cat([rgb_t, disp_t], dim=0).float(),
            "s_center": torch.from_numpy(s_center).unsqueeze(0).float(),
            "l_teacher": torch.from_numpy(l_teacher).unsqueeze(0).float(),
            "sav_teacher": torch.from_numpy(sav_teacher).unsqueeze(0).float(),
            "sample_id": int(row["sample_id"]),
            "source_sequence": row["source_sequence"],
            "center_frame_id": row["center_frame_id"],
            "has_gt": row["has_gt"] == "True",
        }

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        sample = np.load(self.cache_root / row["sample_path"])
        h, w = sample["center_rgb"].shape[:2]
        ch, cw = self.crop_size
        y, x = self.crop_origin(h, w)
        return self.load_row(row, y, x, ch, cw)


class PairedMultiTeacherCacheDataset(Dataset):
    def __init__(self, cache_root: Path, sample_ids: list[int] | None, crop_size: tuple[int, int], random_crop: bool):
        self.single = MultiTeacherCacheDataset(cache_root, None, crop_size, random_crop=False)
        self.cache_root = Path(cache_root)
        self.crop_size = crop_size
        self.random_crop = random_crop
        wanted = set(sample_ids) if sample_ids is not None else None
        by_seq: dict[str, dict[int, dict]] = {}
        for row in self.single.rows:
            by_seq.setdefault(row["source_sequence"], {})[int(row["sample_id"])] = row
        self.pairs = []
        for seq, by_id in by_seq.items():
            for sid in sorted(by_id):
                if sid - 1 not in by_id:
                    continue
                if wanted is not None and (sid not in wanted or sid - 1 not in wanted):
                    continue
                self.pairs.append((by_id[sid - 1], by_id[sid]))
        if not self.pairs:
            raise RuntimeError(f"No consecutive pairs under {cache_root}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        prev_row, cur_row = self.pairs[idx]
        sample = np.load(self.cache_root / cur_row["sample_path"])
        h, w = sample["center_rgb"].shape[:2]
        ch, cw = self.crop_size
        if self.random_crop:
            y, x = random.randint(0, h - ch), random.randint(0, w - cw)
        else:
            y, x = (h - ch) // 2, (w - cw) // 2
        return {
            "prev": self.single.load_row(prev_row, y, x, ch, cw),
            "cur": self.single.load_row(cur_row, y, x, ch, cw),
        }


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_by_sequence(cache_root: Path, val_sequences: int = 1):
    with (cache_root / "index.csv").open() as f:
        rows = list(csv.DictReader(f))
    sequences = sorted({r["source_sequence"] for r in rows})
    val_seq = set(sequences[-val_sequences:])
    train_ids = [int(r["sample_id"]) for r in rows if r["source_sequence"] not in val_seq]
    val_ids = [int(r["sample_id"]) for r in rows if r["source_sequence"] in val_seq]
    return train_ids, val_ids, sorted(val_seq)


def to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def forward(model, batch, device):
    batch = to_device(batch, device)
    delta = model(batch["input"])
    refined = torch.clamp(batch["s_center"] + delta, min=0.0)
    return batch, delta, refined


def pair_loss(model, batch, device, args):
    prev, d_prev, r_prev = forward(model, batch["prev"], device)
    cur, d_cur, r_cur = forward(model, batch["cur"], device)
    spatial = F.smooth_l1_loss(r_cur, cur["l_teacher"]) + F.smooth_l1_loss(r_prev, prev["l_teacher"])
    abs_sav = F.smooth_l1_loss(r_cur, cur["sav_teacher"]) + F.smooth_l1_loss(r_prev, prev["sav_teacher"])
    delta_sav = F.smooth_l1_loss(r_cur - r_prev, cur["sav_teacher"] - prev["sav_teacher"])
    residual = torch.mean(torch.abs(d_cur)) + torch.mean(torch.abs(d_prev))
    edge = edge_aware_smoothness(d_cur, cur["input"][:, :3]) + edge_aware_smoothness(d_prev, prev["input"][:, :3])
    loss = args.spatial_weight * spatial + args.abs_sav_weight * abs_sav + args.delta_sav_weight * delta_sav + args.res_weight * residual + args.edge_weight * edge
    return loss, {
        "loss_spatial": float(spatial.detach().cpu()),
        "loss_abs_sav": float(abs_sav.detach().cpu()),
        "loss_delta_sav": float(delta_sav.detach().cpu()),
        "loss_res": float(residual.detach().cpu()),
        "loss_edge": float(edge.detach().cpu()),
    }


def train_one_epoch(model, loader, optimizer, scaler, device, args):
    model.train()
    rows = []
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda" and args.amp):
            loss, parts = pair_loss(model, batch, device, args)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        parts["loss"] = float(loss.detach().cpu())
        rows.append(parts)
    return mean_dict(rows)


@torch.no_grad()
def eval_single(model, dataset, device):
    model.eval()
    rows = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for batch in loader:
        b, delta, refined = forward(model, batch, device)
        rows.append(
            {
                "mae_s_to_l": float(torch.mean(torch.abs(b["s_center"] - b["l_teacher"])).cpu()),
                "mae_refined_to_l": float(torch.mean(torch.abs(refined - b["l_teacher"])).cpu()),
                "mae_s_to_sav": float(torch.mean(torch.abs(b["s_center"] - b["sav_teacher"])).cpu()),
                "mae_refined_to_sav": float(torch.mean(torch.abs(refined - b["sav_teacher"])).cpu()),
                "residual_mean": float(delta.mean().cpu()),
                "residual_std": float(delta.std().cpu()),
                "residual_min": float(delta.min().cpu()),
                "residual_max": float(delta.max().cpu()),
            }
        )
    return mean_dict(rows)


@torch.no_grad()
def eval_pairs(model, dataset, device):
    model.eval()
    rows = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for batch in loader:
        prev, _d_prev, r_prev = forward(model, batch["prev"], device)
        cur, _d_cur, r_cur = forward(model, batch["cur"], device)
        rows.append(
            {
                "s2m2s_temporal_diff": float(torch.mean(torch.abs(cur["s_center"] - prev["s_center"])).cpu()),
                "s2m2l_temporal_diff": float(torch.mean(torch.abs(cur["l_teacher"] - prev["l_teacher"])).cpu()),
                "sav_temporal_diff": float(torch.mean(torch.abs(cur["sav_teacher"] - prev["sav_teacher"])).cpu()),
                "refined_temporal_diff": float(torch.mean(torch.abs(r_cur - r_prev)).cpu()),
                "teacher_delta_mae": float(torch.mean(torch.abs((r_cur - r_prev) - (cur["sav_teacher"] - prev["sav_teacher"]))).cpu()),
            }
        )
    return mean_dict(rows) | {"pair_count": float(len(rows))}


@torch.no_grad()
def save_qualitative(model, dataset, out_dir: Path, device, max_items=8):
    qdir = out_dir / "qualitative"
    qdir.mkdir(parents=True, exist_ok=True)
    chosen = np.linspace(0, len(dataset) - 1, min(max_items, len(dataset)), dtype=int)
    for idx in chosen:
        batch = dataset[int(idx)]
        collated = {k: v.unsqueeze(0) if torch.is_tensor(v) else [v] for k, v in batch.items()}
        b, delta, refined = forward(model, collated, device)
        rgb = (b["input"][0, :3].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        s = b["s_center"][0, 0].cpu().numpy()
        l = b["l_teacher"][0, 0].cpu().numpy()
        sav = b["sav_teacher"][0, 0].cpu().numpy()
        r = refined[0, 0].cpu().numpy()
        d = delta[0, 0].cpu().numpy()
        vmax = float(np.nanpercentile(np.concatenate([s.ravel(), l.ravel(), sav.ravel(), r.ravel()]), 99))
        tiles = [
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            colorize(s, vmax),
            colorize(r, vmax),
            colorize(l, vmax),
            colorize(sav, vmax),
            colorize(np.abs(s - l), 8.0, cv2.COLORMAP_MAGMA),
            colorize(np.abs(r - l), 8.0, cv2.COLORMAP_MAGMA),
            colorize(np.abs(r - sav), 8.0, cv2.COLORMAP_MAGMA),
            colorize(d, 4.0, cv2.COLORMAP_VIRIDIS),
        ]
        labels = ["RGB", "S raw", "Refined", "L teacher", "SAV", "|S-L|", "|R-L|", "|R-SAV|", "delta"]
        small = []
        for tile, label in zip(tiles, labels):
            tile = cv2.resize(tile, (150, 105), interpolation=cv2.INTER_AREA)
            cv2.putText(tile, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)
            small.append(tile)
        cv2.imwrite(str(qdir / f"sample_{int(batch['sample_id']):06d}.png"), np.concatenate(small, axis=1))


def write_report(args, metrics):
    lines = [
        "# S2M2-S@512 Multi-Teacher Temporal Refiner V1",
        "",
        "Tiny U-Net residual refiner trained on frozen S2M2-S@512 predictions.",
        "",
        f"Best epoch: `{metrics['best_epoch']}`",
        f"Peak VRAM: `{metrics['peak_vram_mb']:.1f} MB`",
        f"Runtime: `{metrics['runtime_seconds']:.1f} s`",
        "",
        "## Validation",
        "",
        f"- raw S2M2-S -> S2M2-L MAE: `{metrics['val']['mae_s_to_l']:.4f}` px",
        f"- refined -> S2M2-L MAE: `{metrics['val']['mae_refined_to_l']:.4f}` px",
        f"- raw S2M2-S -> StereoAnyVideo MAE: `{metrics['val']['mae_s_to_sav']:.4f}` px",
        f"- refined -> StereoAnyVideo MAE: `{metrics['val']['mae_refined_to_sav']:.4f}` px",
        f"- raw S2M2-S temporal diff: `{metrics['temporal']['s2m2s_temporal_diff']:.4f}` px",
        f"- refined temporal diff: `{metrics['temporal']['refined_temporal_diff']:.4f}` px",
        f"- S2M2-L temporal diff: `{metrics['temporal']['s2m2l_temporal_diff']:.4f}` px",
        f"- StereoAnyVideo temporal diff: `{metrics['temporal']['sav_temporal_diff']:.4f}` px",
        f"- teacher-delta MAE: `{metrics['temporal']['teacher_delta_mae']:.4f}` px",
        "",
        "## Interpretation",
        "",
        "This first run tests whether a small residual head can move the fast S2M2-S@512 backbone toward S2M2-L spatial structure while distilling StereoAnyVideo temporal dynamics.",
    ]
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-root", type=Path, default=Path("results/temporal_refinement_cache/large_v2_s2m2s512"))
    p.add_argument("--out-dir", type=Path, default=Path("results/temporal_refinement_train_unet_s2m2s512_v1"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--crop-height", "--crop-h", type=int, default=384)
    p.add_argument("--crop-width", "--crop-w", type=int, default=640)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--spatial-weight", type=float, default=0.40)
    p.add_argument("--abs-sav-weight", type=float, default=0.25)
    p.add_argument("--delta-sav-weight", type=float, default=0.25)
    p.add_argument("--res-weight", type=float, default=0.10)
    p.add_argument("--edge-weight", type=float, default=0.05)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=13)
    return p.parse_args()


def main():
    args = parse_args()
    start = time.time()
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "checkpoints").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    train_ids, val_ids, val_sequences = split_by_sequence(args.cache_root, val_sequences=1)
    crop = (args.crop_height, args.crop_width)
    train_pairs = PairedMultiTeacherCacheDataset(args.cache_root, train_ids, crop, random_crop=True)
    val_single = MultiTeacherCacheDataset(args.cache_root, val_ids, crop, random_crop=False)
    val_pairs = PairedMultiTeacherCacheDataset(args.cache_root, val_ids, crop, random_crop=False)
    train_loader = DataLoader(train_pairs, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model = TinyUNetRefiner(in_channels=8, base_channels=16, residual_clamp_px=4.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp)
    (args.out_dir / "config.yaml").write_text(
        f"""cache_root: {args.cache_root}
out_dir: {args.out_dir}
model: TinyUNetRefiner
backbone: S2M2-S@512
spatial_teacher: S2M2-L@736
temporal_teacher: StereoAnyVideo@384x640
crop_size: [{args.crop_height}, {args.crop_width}]
batch_size: {args.batch_size}
epochs: {args.epochs}
lr: {args.lr}
loss_weights:
  spatial_s2m2l: {args.spatial_weight}
  abs_stereoanyvideo: {args.abs_sav_weight}
  delta_stereoanyvideo: {args.delta_sav_weight}
  residual_l1: {args.res_weight}
  edge_aware: {args.edge_weight}
val_sequences: {val_sequences}
"""
    )
    log_path = args.out_dir / "train_log.csv"
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_loss_spatial", "train_loss_abs_sav", "train_loss_delta_sav", "train_loss_res", "train_loss_edge", "val_mae_refined_to_l", "val_mae_refined_to_sav", "refined_temporal_diff", "teacher_delta_mae"])
        writer.writeheader()
    best_score = float("inf")
    best_epoch = -1
    last_val = {}
    last_temporal = {}
    for epoch in range(1, args.epochs + 1):
        train = train_one_epoch(model, train_loader, opt, scaler, device, args)
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            val = eval_single(model, val_single, device)
            temporal = eval_pairs(model, val_pairs, device)
            last_val, last_temporal = val, temporal
        else:
            val, temporal = last_val, last_temporal
        score = val.get("mae_refined_to_l", float("inf")) + 0.5 * temporal.get("teacher_delta_mae", float("inf"))
        if score < best_score:
            best_score = score
            best_epoch = epoch
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(), "epoch": epoch, "score": score}, args.out_dir / "checkpoints" / "best.pt")
        if args.save_every and epoch % args.save_every == 0:
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(), "epoch": epoch, "score": score}, args.out_dir / "checkpoints" / f"epoch_{epoch:04d}.pt")
        with log_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_loss_spatial", "train_loss_abs_sav", "train_loss_delta_sav", "train_loss_res", "train_loss_edge", "val_mae_refined_to_l", "val_mae_refined_to_sav", "refined_temporal_diff", "teacher_delta_mae"])
            writer.writerow({
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train.items()},
                "val_mae_refined_to_l": val.get("mae_refined_to_l", ""),
                "val_mae_refined_to_sav": val.get("mae_refined_to_sav", ""),
                "refined_temporal_diff": temporal.get("refined_temporal_diff", ""),
                "teacher_delta_mae": temporal.get("teacher_delta_mae", ""),
            })
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:03d} val_R_L={val.get('mae_refined_to_l', float('nan')):.4f} val_R_SAV={val.get('mae_refined_to_sav', float('nan')):.4f} ref_td={temporal.get('refined_temporal_diff', float('nan')):.4f}", flush=True)
    ckpt = torch.load(args.out_dir / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    metrics = {
        "best_epoch": best_epoch,
        "val_sequences": val_sequences,
        "train_sample_count": len(train_ids),
        "val_sample_count": len(val_ids),
        "val": eval_single(model, val_single, device),
        "temporal": eval_pairs(model, val_pairs, device),
        "runtime_seconds": time.time() - start,
        "peak_vram_mb": float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == "cuda" else 0.0,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    save_qualitative(model, val_single, args.out_dir, device)
    write_report(args, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
