from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from scripts.temporal_refinement.lib.losses import edge_aware_smoothness


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def colorize(x: np.ndarray, vmax=None, cmap=cv2.COLORMAP_TURBO) -> np.ndarray:
    x = x.astype(np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros((*x.shape, 3), dtype=np.uint8)
    if vmax is None:
        lo = float(np.nanpercentile(x[finite], 1))
        hi = float(np.nanpercentile(x[finite], 99))
    else:
        lo, hi = 0.0, float(vmax)
    if hi <= lo:
        return np.zeros((*x.shape, 3), dtype=np.uint8)
    y = (np.clip((x - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(y, cmap)


def mean_dict(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({k for row in rows for k in row})
    out = {}
    for key in keys:
        vals = [row[key] for row in rows if key in row and np.isfinite(row[key])]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    return out


def to_device(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def forward_refiner(model, batch, device):
    batch = to_device(batch, device)
    delta = model(batch["input"])
    refined = torch.clamp(batch["s_center"] + delta, min=0.0)
    return batch, delta, refined


def pair_loss(model, batch, device, args):
    prev, d_prev, r_prev = forward_refiner(model, batch["prev"], device)
    cur, d_cur, r_cur = forward_refiner(model, batch["cur"], device)
    spatial = F.smooth_l1_loss(r_cur, cur["l_teacher"]) + F.smooth_l1_loss(r_prev, prev["l_teacher"])
    abs_sav = F.smooth_l1_loss(r_cur, cur["sav_teacher"]) + F.smooth_l1_loss(r_prev, prev["sav_teacher"])
    delta_sav = F.smooth_l1_loss(r_cur - r_prev, cur["sav_teacher"] - prev["sav_teacher"])
    residual = torch.mean(torch.abs(d_cur)) + torch.mean(torch.abs(d_prev))
    edge = edge_aware_smoothness(d_cur, cur["input"][:, :3]) + edge_aware_smoothness(d_prev, prev["input"][:, :3])
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
        b, delta, refined = forward_refiner(model, batch, device)
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
        prev, _d_prev, r_prev = forward_refiner(model, batch["prev"], device)
        cur, _d_cur, r_cur = forward_refiner(model, batch["cur"], device)
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
        collated = {key: value.unsqueeze(0) if torch.is_tensor(value) else [value] for key, value in batch.items()}
        b, delta, refined = forward_refiner(model, collated, device)
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
        labels = ["RGB", "backbone", "refined", "spatial T", "temporal T", "|B-S|", "|R-S|", "|R-T|", "delta"]
        small = []
        for tile, label in zip(tiles, labels):
            tile = cv2.resize(tile, (150, 105), interpolation=cv2.INTER_AREA)
            cv2.putText(tile, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)
            small.append(tile)
        cv2.imwrite(str(qdir / f"sample_{int(batch['sample_id']):06d}.png"), np.concatenate(small, axis=1))
