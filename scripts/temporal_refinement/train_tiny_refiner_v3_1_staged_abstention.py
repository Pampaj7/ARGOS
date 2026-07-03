#!/usr/bin/env python3
"""Train staged abstaining refiner: detector first, residual second."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_tiny_refiner_v1_full_gt import (
    DEFAULT_TARGETS_ROOT,
    DISP_SCALE,
    Sample,
    charbonnier,
    colorize,
    finite_mean,
    load_shards,
    masked_mean,
    parse_bool,
    read_rows,
    split_sequences,
    unwrap,
    write_csv,
)


DEFAULT_OUTPUT_ROOT = Path("results/03_temporal_refinement/training/tiny_refiner_v3_1_staged_abstention")
DEFAULT_BALANCED_SPLIT = Path("results/03_temporal_refinement/training/refiner_failure_analysis/proposed_balanced_split.json")
THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01)


def load_samples_with_split(targets_root: Path, split_json: Path, max_frames: int) -> tuple[dict[str, list[str]], dict[str, list[Sample]]]:
    rows = read_rows(targets_root / "frame_targets_index.csv")
    sequence_ids = sorted({r["sequence_id"] for r in rows})
    if split_json.exists():
        splits = json.loads(split_json.read_text())
        splits = {k: [s for s in v if s in sequence_ids] for k, v in splits.items()}
    else:
        splits = split_sequences(sequence_ids)
    by_split: dict[str, list[Sample]] = {k: [] for k in ("train", "val", "test")}
    split_of = {seq: split for split, seqs in splits.items() for seq in seqs}
    for row in rows:
        split = split_of.get(row["sequence_id"])
        if split not in by_split:
            continue
        if max_frames > 0 and len(by_split[split]) >= max_frames:
            continue
        by_split[split].append(
            Sample(
                sequence_id=row["sequence_id"],
                frame_id=row["frame_id"],
                frame_index=int(row["frame_index"]),
                offset=int(row["frame_offset"]),
                target_path=Path(row["target_path"]),
            )
        )
    return splits, by_split


def make_features_from_raws(raws: np.ndarray, valids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = raws[0]
    median = np.median(raws, axis=0).astype(np.float32)
    mean = np.mean(raws, axis=0).astype(np.float32)
    var = np.var(raws, axis=0).astype(np.float32)
    gx = np.zeros_like(raw, dtype=np.float32)
    gy = np.zeros_like(raw, dtype=np.float32)
    gx[:, 1:] = raw[:, 1:] - raw[:, :-1]
    gy[1:, :] = raw[1:, :] - raw[:-1, :]
    edge = np.sqrt(gx * gx + gy * gy)
    dt1 = np.abs(raws[0] - raws[1]).astype(np.float32) if raws.shape[0] > 1 else np.zeros_like(raw)
    features = [
        *(raws / DISP_SCALE),
        *valids,
        dt1 / DISP_SCALE,
        mean / DISP_SCALE,
        median / DISP_SCALE,
        var / (DISP_SCALE * DISP_SCALE),
        np.abs(raw - median) / DISP_SCALE,
        gx / DISP_SCALE,
        gy / DISP_SCALE,
        edge / DISP_SCALE,
    ]
    return np.stack(features, axis=0).astype(np.float32), edge, var


def crop_arrays(shard: dict[str, np.ndarray], offset: int, y: int, x: int, size: int, context_frames: int) -> dict[str, np.ndarray]:
    ys = slice(y, y + size)
    xs = slice(x, x + size)
    indices = [max(0, offset - i) for i in range(context_frames)]
    raws = np.stack([shard["raw_disp"][i, ys, xs].astype(np.float32) for i in indices], axis=0)
    valids = np.stack([shard["valid_mask"][i, ys, xs].astype(np.float32) for i in indices], axis=0)
    xfeat, edge, var = make_features_from_raws(raws, valids)
    raw = raws[0]
    gt = shard["gt_disp"][offset, ys, xs].astype(np.float32)
    valid = valids[0]
    delta = shard["delta_disp_gt_minus_raw"][offset, ys, xs].astype(np.float32)
    return {"x": xfeat, "raw": raw, "gt": gt, "delta": delta, "valid": valid, "edge": edge, "temporal_var": var}


class BalancedCropDataset(Dataset):
    categories = ("hard", "good", "boundary", "temporal", "random")

    def __init__(self, samples: list[Sample], shards: dict[Path, dict[str, np.ndarray]], args: argparse.Namespace):
        self.samples = samples
        self.shards = shards
        self.args = args
        self.rng = random.Random(1234)
        self.counts = {k: 0 for k in self.categories}

    def __len__(self) -> int:
        return self.args.crops_per_epoch

    def _score_crop(self, shard: dict[str, np.ndarray], offset: int, y: int, x: int, category: str) -> float:
        s = self.args.crop_size
        raw = shard["raw_disp"][offset, y : y + s, x : x + s].astype(np.float32)
        gt = shard["gt_disp"][offset, y : y + s, x : x + s].astype(np.float32)
        valid = shard["valid_mask"][offset, y : y + s, x : x + s].astype(np.float32)
        if valid.mean() < 0.05:
            return -1.0
        err = np.abs(raw - gt)
        if category == "hard":
            return float((((err >= self.args.bad_threshold_px) | (err >= 10.0)) * valid).mean())
        if category == "good":
            return float(((err < self.args.good_threshold_px) * valid).mean())
        if category == "boundary":
            gx = np.zeros_like(raw)
            gy = np.zeros_like(raw)
            gx[:, 1:] = raw[:, 1:] - raw[:, :-1]
            gy[1:, :] = raw[1:, :] - raw[:-1, :]
            return float((np.sqrt(gx * gx + gy * gy) * valid).mean())
        if category == "temporal":
            idx = [max(0, offset - i) for i in range(self.args.context_frames)]
            raws = np.stack([shard["raw_disp"][i, y : y + s, x : x + s].astype(np.float32) for i in idx], axis=0)
            return float((np.var(raws, axis=0) * valid).mean())
        return self.rng.random()

    def _pick_crop(self, shard: dict[str, np.ndarray], offset: int, category: str) -> tuple[int, int]:
        h, w = shard["raw_disp"].shape[1:]
        s = min(self.args.crop_size, h, w)
        best = (0.0, 0, 0)
        tries = 1 if category == "random" else self.args.crop_candidate_tries
        for _ in range(tries):
            y = self.rng.randint(0, max(0, h - s))
            x = self.rng.randint(0, max(0, w - s))
            score = self._score_crop(shard, offset, y, x, category)
            if score >= best[0]:
                best = (score, y, x)
        return best[1], best[2]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        category = self.categories[idx % len(self.categories)]
        sample = self.samples[self.rng.randrange(len(self.samples))]
        shard = self.shards[sample.target_path]
        y, x = self._pick_crop(shard, sample.offset, category)
        item = crop_arrays(shard, sample.offset, y, x, min(self.args.crop_size, shard["raw_disp"].shape[1], shard["raw_disp"].shape[2]), self.args.context_frames)
        raw_err = np.abs(item["raw"] - item["gt"])
        self.counts[category] += 1
        return {
            "x": torch.from_numpy(item["x"]),
            "raw": torch.from_numpy(item["raw"][None]),
            "gt": torch.from_numpy(item["gt"][None]),
            "delta": torch.from_numpy(item["delta"][None]),
            "valid": torch.from_numpy(item["valid"][None]),
            "raw_bad": torch.from_numpy((raw_err >= self.args.bad_threshold_px).astype(np.float32)[None]),
            "raw_good": torch.from_numpy((raw_err < self.args.good_threshold_px).astype(np.float32)[None]),
            "category": category,
        }


class FullFrameDataset(Dataset):
    def __init__(self, samples: list[Sample], shards: dict[Path, dict[str, np.ndarray]], context_frames: int):
        self.samples = samples
        self.shards = shards
        self.context_frames = context_frames

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        shard = self.shards[sample.target_path]
        h, w = shard["raw_disp"].shape[1:]
        item = crop_arrays(shard, sample.offset, 0, 0, min(h, w) if h == w else h, self.context_frames) if h == w else None
        if item is None:
            indices = [max(0, sample.offset - i) for i in range(self.context_frames)]
            raws = np.stack([shard["raw_disp"][i].astype(np.float32) for i in indices], axis=0)
            valids = np.stack([shard["valid_mask"][i].astype(np.float32) for i in indices], axis=0)
            xfeat, _edge, var = make_features_from_raws(raws, valids)
            raw = raws[0]
            gt = shard["gt_disp"][sample.offset].astype(np.float32)
            valid = valids[0]
            delta = shard["delta_disp_gt_minus_raw"][sample.offset].astype(np.float32)
        else:
            xfeat, raw, gt, valid, delta, var = item["x"], item["raw"], item["gt"], item["valid"], item["delta"], item["temporal_var"]
        return {
            "x": torch.from_numpy(xfeat),
            "raw": torch.from_numpy(raw[None]),
            "gt": torch.from_numpy(gt[None]),
            "delta": torch.from_numpy(delta[None]),
            "valid": torch.from_numpy(valid[None]),
            "temporal_var": torch.from_numpy(var[None]),
            "sequence_id": sample.sequence_id,
            "frame_id": sample.frame_id,
        }


class AbstentionCropRefiner(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 64):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_channels, hidden, 3, padding=1), nn.GroupNorm(8, hidden), nn.SiLU(inplace=True)]
        for _ in range(5):
            layers += [nn.Conv2d(hidden, hidden, 3, padding=1), nn.GroupNorm(8, hidden), nn.SiLU(inplace=True)]
        self.trunk = nn.Sequential(*layers)
        self.bad_head = nn.Conv2d(hidden, 1, 1)
        self.residual_head = nn.Conv2d(hidden, 1, 1)
        nn.init.zeros_(self.bad_head.weight)
        nn.init.zeros_(self.bad_head.bias)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, x: torch.Tensor, residual_scale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.trunk(x)
        bad_logit = self.bad_head(feat)
        residual = residual_scale * torch.tanh(self.residual_head(feat))
        p_bad = torch.sigmoid(bad_logit)
        return bad_logit, p_bad, residual


@dataclass
class MetricAcc:
    valid: float = 0.0
    raw_abs: float = 0.0
    soft_abs: float = 0.0
    hard_abs: float = 0.0
    raw_bad1: float = 0.0
    raw_bad3: float = 0.0
    soft_bad1: float = 0.0
    soft_bad3: float = 0.0
    hard_bad1: float = 0.0
    hard_bad3: float = 0.0
    raw_good: float = 0.0
    new_bad3: float = 0.0
    fixed_bad3: float = 0.0
    degraded_good_abs: float = 0.0
    modified: float = 0.0
    p_good: float = 0.0
    p_mid: float = 0.0
    p_bad: float = 0.0
    n_good: float = 0.0
    n_mid: float = 0.0
    n_bad: float = 0.0
    corr_abs: float = 0.0
    det_tp: float = 0.0
    det_fp: float = 0.0
    det_fn: float = 0.0

    _FIELDS = (
        "valid", "raw_abs", "soft_abs", "hard_abs", "raw_bad1", "raw_bad3", "soft_bad1",
        "soft_bad3", "hard_bad1", "hard_bad3", "raw_good", "new_bad3", "fixed_bad3",
        "degraded_good_abs", "modified", "corr_abs", "p_good", "p_mid", "p_bad",
        "n_good", "n_mid", "n_bad", "det_tp", "det_fp", "det_fn",
    )

    def add(self, raw, gt, valid, p_bad, residual, threshold: float) -> None:
        hard_mask = (p_bad >= threshold).float()
        soft = raw + p_bad * residual
        hard = raw + hard_mask * residual
        raw_err = torch.abs(raw - gt)
        soft_err = torch.abs(soft - gt)
        hard_err = torch.abs(hard - gt)
        v = valid
        good = v * (raw_err < 1.0).float()
        bad = v * (raw_err >= 3.0).float()
        mid = v * ((raw_err >= 1.0) & (raw_err < 3.0)).float()
        raw_bad3 = (raw_err > 3.0).float()
        hard_bad3 = (hard_err > 3.0).float()
        sums = torch.stack([
            v.sum(),
            (raw_err * v).sum(),
            (soft_err * v).sum(),
            (hard_err * v).sum(),
            ((raw_err > 1.0).float() * v).sum(),
            (raw_bad3 * v).sum(),
            ((soft_err > 1.0).float() * v).sum(),
            ((soft_err > 3.0).float() * v).sum(),
            ((hard_err > 1.0).float() * v).sum(),
            (hard_bad3 * v).sum(),
            good.sum(),
            ((raw_err < 1.0).float() * hard_bad3 * v).sum(),
            (raw_bad3 * (hard_err <= 3.0).float() * v).sum(),
            (torch.clamp(hard_err - raw_err, min=0.0) * good).sum(),
            (hard_mask * v).sum(),
            (torch.abs(hard_mask * residual) * v).sum(),
            (p_bad * good).sum(),
            (p_bad * mid).sum(),
            (p_bad * bad).sum(),
            good.sum(),
            mid.sum(),
            bad.sum(),
            (hard_mask * bad).sum(),
            (hard_mask * (1.0 - (bad > 0).float()) * v).sum(),
            ((1.0 - hard_mask) * bad).sum(),
        ]).detach().cpu().tolist()
        if sums[0] <= 0:
            return
        for name, s in zip(self._FIELDS, sums):
            setattr(self, name, getattr(self, name) + float(s))

    def row(self, threshold: float) -> dict[str, float]:
        v = max(self.valid, 1.0)
        return {
            "threshold": threshold,
            "raw_mae": self.raw_abs / v,
            "refined_soft_mae": self.soft_abs / v,
            "refined_hard_mae": self.hard_abs / v,
            "raw_bad1": 100.0 * self.raw_bad1 / v,
            "raw_bad3": 100.0 * self.raw_bad3 / v,
            "refined_soft_bad1": 100.0 * self.soft_bad1 / v,
            "refined_soft_bad3": 100.0 * self.soft_bad3 / v,
            "refined_hard_bad1": 100.0 * self.hard_bad1 / v,
            "refined_hard_bad3": 100.0 * self.hard_bad3 / v,
            "new_bad3_from_raw_good_pct": 100.0 * self.new_bad3 / max(self.raw_good, 1.0),
            "fixed_bad3_pct": 100.0 * self.fixed_bad3 / max(self.raw_bad3, 1.0),
            "degraded_good_pixel_mean": self.degraded_good_abs / max(self.raw_good, 1.0),
            "fraction_modified_pct": 100.0 * self.modified / v,
            "p_bad_good_mean": self.p_good / max(self.n_good, 1.0),
            "p_bad_mid_mean": self.p_mid / max(self.n_mid, 1.0),
            "p_bad_bad_mean": self.p_bad / max(self.n_bad, 1.0),
            "correction_magnitude_mean": self.corr_abs / v,
            "precision_at_threshold": self.det_tp / max(self.det_tp + self.det_fp, 1.0),
            "recall_at_threshold": self.det_tp / max(self.det_tp + self.det_fn, 1.0),
            "mae_improvement_pct": 100.0 * ((self.raw_abs / v) - (self.hard_abs / v)) / max(self.raw_abs / v, 1e-8),
            "bad3_improvement_pct": 100.0 * ((self.raw_bad3 / v) - (self.hard_bad3 / v)) / max(self.raw_bad3 / v, 1e-8),
        }


def focal_bce(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, gamma: float) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = torch.sigmoid(logits)
    pt = torch.where(target > 0.5, prob, 1.0 - prob)
    return masked_mean(bce * torch.pow(1.0 - pt, gamma), valid)


def detector_loss_batch(model: nn.Module, batch: dict[str, torch.Tensor], args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    x = batch["x"].to(device, non_blocking=True)
    valid = batch["valid"].to(device, non_blocking=True)
    raw_bad = batch["raw_bad"].to(device, non_blocking=True) * valid
    bad_logit, p_bad, _residual = model(x, args.residual_scale)
    loss = focal_bce(bad_logit, raw_bad, valid, args.focal_gamma)
    return loss, {
        "loss": float(loss.detach().cpu()),
        "bad_detection_loss": float(loss.detach().cpu()),
        "p_bad_mean": float(masked_mean(p_bad, valid).detach().cpu()),
        "raw_bad_frac": float(masked_mean(raw_bad, valid).detach().cpu()),
    }


def residual_loss_batch(model: nn.Module, batch: dict[str, torch.Tensor], args: argparse.Namespace, device: torch.device, detector_weight: float) -> tuple[torch.Tensor, dict[str, float]]:
    x = batch["x"].to(device, non_blocking=True)
    raw = batch["raw"].to(device, non_blocking=True)
    gt = batch["gt"].to(device, non_blocking=True)
    delta = batch["delta"].to(device, non_blocking=True)
    valid = batch["valid"].to(device, non_blocking=True)
    raw_bad = batch["raw_bad"].to(device, non_blocking=True) * valid
    raw_good = batch["raw_good"].to(device, non_blocking=True) * valid
    bad_logit, p_bad, residual = model(x, args.residual_scale)
    p_for_refine = p_bad.detach() if detector_weight <= 0.0 else p_bad
    refined = raw + p_for_refine * residual
    raw_err = torch.abs(raw - gt)
    clipped_refined = torch.clamp(charbonnier(refined - gt), max=args.robust_loss_clip_px)
    full_loss = masked_mean(clipped_refined, valid)
    det_loss = focal_bce(bad_logit, raw_bad, valid, args.focal_gamma)
    res_loss = masked_mean(charbonnier(residual - delta), raw_bad) if float(raw_bad.sum()) > 0 else full_loss.new_tensor(0.0)
    preserve = masked_mean(torch.abs(p_for_refine * residual), raw_good) if float(raw_good.sum()) > 0 else full_loss.new_tensor(0.0)
    sparse = masked_mean(p_bad, valid) + masked_mean(p_bad, raw_good) if float(raw_good.sum()) > 0 else masked_mean(p_bad, valid)
    loss = (
        args.full_weight * full_loss
        + detector_weight * det_loss
        + args.residual_weight * res_loss
        + args.preserve_weight * preserve
        + args.sparsity_weight * sparse
    )
    return loss, {
        "loss": float(loss.detach().cpu()),
        "full_loss": float(full_loss.detach().cpu()),
        "bad_detection_loss": float(det_loss.detach().cpu()),
        "residual_hard_loss": float(res_loss.detach().cpu()),
        "preserve_loss": float(preserve.detach().cpu()),
        "sparsity_loss": float(sparse.detach().cpu()),
        "p_bad_mean": float(masked_mean(p_bad, valid).detach().cpu()),
        "raw_error_mean": float(masked_mean(raw_err, valid).detach().cpu()),
    }


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, args: argparse.Namespace, device: torch.device, stage: str, detector_weight: float = 0.0) -> dict[str, float]:
    model.train()
    rows: list[dict[str, float]] = []
    for batch in loader:
        if stage == "detector":
            loss, metrics = detector_loss_batch(model, batch, args, device)
        else:
            loss, metrics = residual_loss_batch(model, batch, args, device, detector_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        rows.append(metrics)
    return {k: finite_mean([r[k] for r in rows]) for k in rows[0]}


def auc_ap(labels: list[np.ndarray], scores: list[np.ndarray]) -> tuple[float, float]:
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        y = np.concatenate(labels)
        p = np.concatenate(scores)
        if len(np.unique(y)) < 2:
            return float("nan"), float("nan")
        return float(roc_auc_score(y, p)), float(average_precision_score(y, p))
    except Exception:
        return float("nan"), float("nan")


def write_csv_union(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, args: argparse.Namespace, device: torch.device, split: str, per_sequence: bool = True) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model.eval()
    acc = {thr: MetricAcc() for thr in THRESHOLDS}
    seq_acc: dict[tuple[str, float], MetricAcc] = {}
    auc_labels: list[np.ndarray] = []
    auc_scores: list[np.ndarray] = []
    auc_left = args.max_auc_pixels
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        raw = batch["raw"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        _logit, p_bad, residual = model(x, args.residual_scale)
        for thr, meter in acc.items():
            meter.add(raw, gt, valid, p_bad, residual, thr)
        if per_sequence:
            for i, seq in enumerate(batch["sequence_id"]):
                for thr in THRESHOLDS:
                    seq_acc.setdefault((str(seq), thr), MetricAcc()).add(raw[i : i + 1], gt[i : i + 1], valid[i : i + 1], p_bad[i : i + 1], residual[i : i + 1], thr)
        if auc_left > 0:
            y = ((torch.abs(raw - gt) >= args.bad_threshold_px) & (valid > 0)).detach().cpu().numpy().reshape(-1).astype(np.uint8)
            m = (valid > 0).detach().cpu().numpy().reshape(-1)
            p = p_bad.detach().cpu().numpy().reshape(-1)
            idx = np.flatnonzero(m)
            if idx.size:
                take = min(auc_left, idx.size, 20_000)
                pick = np.linspace(0, idx.size - 1, take, dtype=np.int64)
                auc_labels.append(y[idx[pick]])
                auc_scores.append(p[idx[pick]])
                auc_left -= take
    auc, ap = auc_ap(auc_labels, auc_scores) if auc_labels else (float("nan"), float("nan"))
    rows = []
    for thr in THRESHOLDS:
        row = {"split": split, **acc[thr].row(thr), "bad_pixel_auc": auc, "bad_pixel_ap": ap}
        rows.append(row)
    seq_rows = []
    for (seq, thr), meter in sorted(seq_acc.items()):
        seq_rows.append({"split": split, "sequence_id": seq, **meter.row(thr)})
    return rows, seq_rows


def choose_threshold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    feasible = [r for r in rows if r["refined_hard_mae"] <= r["raw_mae"] and r["refined_hard_bad3"] <= r["raw_bad3"]]
    pool = feasible or rows
    return min(pool, key=lambda r: (r["refined_hard_mae"], r["refined_hard_bad3"]))


@torch.no_grad()
def write_diagnostics(model: nn.Module, dataset: FullFrameDataset, out_dir: Path, args: argparse.Namespace, device: torch.device, split: str, threshold: float, count: int = 4) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for idx in np.linspace(0, max(0, len(dataset) - 1), min(count, len(dataset)), dtype=int):
        item = dataset[int(idx)]
        _logit, p_bad, residual = model(item["x"][None].to(device), args.residual_scale)
        raw = item["raw"][0].numpy()
        gt = item["gt"][0].numpy()
        valid = item["valid"][0].numpy()
        prob = p_bad[0, 0].cpu().numpy()
        res = residual[0, 0].cpu().numpy()
        hard_mask = (prob >= threshold).astype(np.float32)
        refined = raw + hard_mask * res
        raw_err = np.abs(raw - gt)
        refined_err = np.abs(refined - gt)
        vmax = float(np.percentile(gt[valid > 0], 98)) if np.any(valid > 0) else 64.0
        tiles = [
            colorize(raw, vmax),
            colorize(gt, vmax),
            colorize(raw_err, 16.0, cv2.COLORMAP_MAGMA),
            colorize(prob, 1.0, cv2.COLORMAP_VIRIDIS),
            colorize(hard_mask, 1.0, cv2.COLORMAP_VIRIDIS),
            colorize(np.abs(res), args.residual_scale, cv2.COLORMAP_MAGMA),
            colorize(refined, vmax),
            colorize(refined_err, 16.0, cv2.COLORMAP_MAGMA),
            colorize(np.clip(raw_err - refined_err, -8.0, 8.0), 8.0, cv2.COLORMAP_TURBO),
            colorize(item["temporal_var"][0].numpy(), 32.0, cv2.COLORMAP_MAGMA),
        ]
        cv2.imwrite(str(out_dir / f"{split}_{item['sequence_id']}_{item['frame_id']}.png"), np.concatenate(tiles, axis=1))


def make_loader(ds: Dataset, batch_size: int, num_workers: int, shuffle: bool, prefetch_factor: int) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets-root", type=Path, default=DEFAULT_TARGETS_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--balanced-split-json", type=Path, default=DEFAULT_BALANCED_SPLIT)
    p.add_argument("--context-frames", type=int, default=4)
    p.add_argument("--crop-size", type=int, default=96)
    p.add_argument("--crop-candidate-tries", type=int, default=12)
    p.add_argument("--crops-per-epoch", type=int, default=60000)
    p.add_argument("--detector-epochs", type=int, default=8)
    p.add_argument("--residual-epochs", type=int, default=22)
    p.add_argument("--residual-freeze-detector-epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--detector-lr", type=float, default=5e-4)
    p.add_argument("--residual-lr", type=float, default=5e-4)
    p.add_argument("--unfrozen-detector-lr", type=float, default=5e-5)
    p.add_argument("--num-workers", type=int, default=24)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--residual-scale", type=float, default=3.0)
    p.add_argument("--bad-threshold-px", type=float, default=3.0)
    p.add_argument("--good-threshold-px", type=float, default=1.0)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--full-weight", type=float, default=0.5)
    p.add_argument("--bad-loss-weight", type=float, default=1.0)
    p.add_argument("--residual-weight", type=float, default=0.5)
    p.add_argument("--preserve-weight", type=float, default=1.0)
    p.add_argument("--sparsity-weight", type=float, default=0.10)
    p.add_argument("--robust-loss-clip-px", type=float, default=10.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--early-stop-patience", type=int, default=6)
    p.add_argument("--max-auc-pixels", type=int, default=200000)
    p.add_argument("--overwrite", nargs="?", const=True, default=False, type=parse_bool)
    return p.parse_args()


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for p in module.parameters():
        p.requires_grad = enabled


def core_model(model: nn.Module) -> AbstentionCropRefiner:
    return unwrap(model)  # type: ignore[return-value]


def make_detector_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    m = core_model(model)
    set_requires_grad(m.trunk, True)
    set_requires_grad(m.bad_head, True)
    set_requires_grad(m.residual_head, False)
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.detector_lr)


def make_residual_optimizer(model: nn.Module, args: argparse.Namespace, freeze_detector: bool) -> torch.optim.Optimizer:
    m = core_model(model)
    set_requires_grad(m.trunk, not freeze_detector)
    set_requires_grad(m.bad_head, not freeze_detector)
    set_requires_grad(m.residual_head, True)
    if freeze_detector:
        return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.residual_lr)
    return torch.optim.AdamW(
        [
            {"params": list(m.residual_head.parameters()), "lr": args.residual_lr},
            {"params": list(m.trunk.parameters()) + list(m.bad_head.parameters()), "lr": args.unfrozen_detector_lr},
        ]
    )


def checkpoint_payload(model: nn.Module, args: argparse.Namespace, splits: dict[str, list[str]], input_channels: int, params: int, epoch: int, threshold: float, val_metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_state_dict": unwrap(model).state_dict(),
        "args": vars(args),
        "splits": splits,
        "input_channels": input_channels,
        "parameter_count": params,
        "epoch": epoch,
        "threshold": threshold,
        "val_metrics": val_metrics,
    }


def main() -> int:
    args = parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(args.output_root)
        shutil.rmtree(args.output_root)
    (args.output_root / "checkpoints").mkdir(parents=True)
    (args.output_root / "diagnostics").mkdir()
    sys.stdout = (args.output_root / "stdout.log").open("w", buffering=1)
    sys.stderr = (args.output_root / "stderr.log").open("w", buffering=1)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    splits, by_split = load_samples_with_split(args.targets_root, args.balanced_split_json, args.max_frames)
    all_samples = by_split["train"] + by_split["val"] + by_split["test"]
    shards = load_shards(all_samples)
    train_ds = BalancedCropDataset(by_split["train"], shards, args)
    full_ds = {split: FullFrameDataset(samples, shards, args.context_frames) for split, samples in by_split.items()}
    train_loader = make_loader(train_ds, args.batch_size, args.num_workers, True, args.prefetch_factor)
    eval_loaders = {split: make_loader(ds, args.eval_batch_size, max(0, args.num_workers // 2), False, args.prefetch_factor) for split, ds in full_ds.items()}

    input_channels = args.context_frames * 2 + 8
    model = AbstentionCropRefiner(input_channels).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    params = sum(p.numel() for p in model.parameters())

    run_lines = [
        f"device={device}",
        f"cuda_device_count={torch.cuda.device_count() if device.type == 'cuda' else 0}",
        f"params={params}",
        f"splits={json.dumps(splits)}",
        f"frames={{'train': {len(full_ds['train'])}, 'val': {len(full_ds['val'])}, 'test': {len(full_ds['test'])}}}",
        f"shards_loaded={len(shards)}",
        f"input_channels={input_channels} context_frames={args.context_frames}",
        f"crop_size={args.crop_size} crops_per_epoch={args.crops_per_epoch} batch_size={args.batch_size}",
        f"detector_epochs={args.detector_epochs} residual_epochs={args.residual_epochs} residual_freeze_detector_epochs={args.residual_freeze_detector_epochs}",
        f"thresholds={THRESHOLDS}",
    ]
    (args.output_root / "run.log").write_text("\n".join(run_lines) + "\n")
    start = time.perf_counter()
    detector_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []

    detector_optimizer = make_detector_optimizer(model, args)
    for epoch in range(1, args.detector_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, detector_optimizer, args, device, "detector")
        val_thr_rows, _ = evaluate(model, eval_loaders["val"], args, device, "val")
        chosen = choose_threshold(val_thr_rows)
        detector_rows.append({"stage": "detector", "epoch": epoch, **train_metrics, **{f"val_{k}": v for k, v in chosen.items() if isinstance(v, (int, float, str))}})
        threshold_rows.extend({"stage": "detector", "epoch": epoch, **r} for r in val_thr_rows)
        run_lines.append(
            f"stage=detector epoch={epoch} loss={train_metrics['loss']:.6f} p_bad={train_metrics['p_bad_mean']:.4f} "
            f"val_auc={chosen['bad_pixel_auc']:.4f} val_ap={chosen['bad_pixel_ap']:.4f} "
            f"val_prec={chosen['precision_at_threshold']:.4f} val_rec={chosen['recall_at_threshold']:.4f} thr={chosen['threshold']}"
        )
        (args.output_root / "run.log").write_text("\n".join(run_lines) + "\n")

    val_thr_rows, _ = evaluate(model, eval_loaders["val"], args, device, "val")
    detector_chosen = choose_threshold(val_thr_rows)
    torch.save(
        checkpoint_payload(model, args, splits, input_channels, params, args.detector_epochs, float(detector_chosen["threshold"]), detector_chosen),
        args.output_root / "checkpoints" / "detector.pt",
    )

    best_score = float("inf")
    best_epoch = 0
    current_optimizer: torch.optim.Optimizer | None = None
    for epoch in range(1, args.residual_epochs + 1):
        freeze_detector = epoch <= args.residual_freeze_detector_epochs
        if epoch == 1 or epoch == args.residual_freeze_detector_epochs + 1:
            current_optimizer = make_residual_optimizer(model, args, freeze_detector)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            current_optimizer,
            args,
            device,
            "residual",
            0.0 if freeze_detector else args.bad_loss_weight,
        )
        val_thr_rows, _ = evaluate(model, eval_loaders["val"], args, device, "val")
        chosen = choose_threshold(val_thr_rows)
        score = chosen["refined_hard_mae"] + max(0.0, chosen["refined_hard_bad3"] - chosen["raw_bad3"]) * 0.02
        residual_rows.append({"stage": "residual", "epoch": epoch, "detector_frozen": freeze_detector, **train_metrics})
        val_rows.append({"stage": "residual", "epoch": epoch, "detector_frozen": freeze_detector, **chosen})
        threshold_rows.extend({"stage": "residual", "epoch": epoch, **r} for r in val_thr_rows)
        if score < best_score:
            best_score = score
            best_epoch = epoch
            torch.save(
                checkpoint_payload(model, args, splits, input_channels, params, epoch, float(chosen["threshold"]), chosen),
                args.output_root / "checkpoints" / "best.pt",
            )
        run_lines.append(
            f"stage=residual epoch={epoch} frozen_detector={freeze_detector} loss={train_metrics['loss']:.6f} p_bad={train_metrics['p_bad_mean']:.4f} "
            f"val_raw={chosen['raw_mae']:.6f} val_hard={chosen['refined_hard_mae']:.6f} "
            f"val_bad3={chosen['raw_bad3']:.3f}->{chosen['refined_hard_bad3']:.3f} "
            f"thr={chosen['threshold']} modified={chosen['fraction_modified_pct']:.2f}% auc={chosen['bad_pixel_auc']:.4f}"
        )
        (args.output_root / "run.log").write_text("\n".join(run_lines) + "\n")
        if epoch - best_epoch >= args.early_stop_patience:
            run_lines.append(f"early_stop epoch={epoch} best_epoch={best_epoch} patience={args.early_stop_patience}")
            (args.output_root / "run.log").write_text("\n".join(run_lines) + "\n")
            break

    ckpt = torch.load(args.output_root / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    unwrap(model).load_state_dict(ckpt["model_state_dict"])
    best_threshold = float(ckpt["threshold"])
    all_thr_rows: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    selected_rows: dict[str, dict[str, Any]] = {}
    for split, loader in eval_loaders.items():
        rows, seq_rows = evaluate(model, loader, args, device, split)
        all_thr_rows.extend({"best_residual_epoch": ckpt["epoch"], **r} for r in rows)
        sequence_rows.extend(r for r in seq_rows if abs(float(r["threshold"]) - best_threshold) < 1e-6)
        selected_rows[split] = min(rows, key=lambda r: abs(float(r["threshold"]) - best_threshold))

    write_csv_union(args.output_root / "train_detector_log.csv", detector_rows)
    write_csv(args.output_root / "train_residual_log.csv", residual_rows)
    write_csv(args.output_root / "val_metrics.csv", val_rows)
    write_csv(args.output_root / "test_metrics.csv", [selected_rows["test"]])
    write_csv_union(args.output_root / "threshold_sweep.csv", threshold_rows + all_thr_rows)
    write_csv(args.output_root / "sequence_metrics.csv", sequence_rows)
    shutil.copyfile(args.output_root / "checkpoints" / "best.pt", args.output_root / "checkpoints" / "final.pt")
    planned = args.crops_per_epoch * (args.detector_epochs + int(ckpt["epoch"]))
    write_csv(
        args.output_root / "crop_sampling_summary.csv",
        [{"category": k, "planned_samples": planned // len(train_ds.categories)} for k in train_ds.categories],
    )
    write_diagnostics(model, full_ds["val"], args.output_root / "diagnostics", args, device, "val", best_threshold)
    write_diagnostics(model, full_ds["test"], args.output_root / "diagnostics", args, device, "test", best_threshold)

    summary = {
        "targets_root": str(args.targets_root),
        "output_root": str(args.output_root),
        "params": params,
        "best_residual_epoch": ckpt["epoch"],
        "best_threshold": best_threshold,
        "elapsed_seconds": time.perf_counter() - start,
        "frames": {k: len(v) for k, v in full_ds.items()},
        "splits": splits,
        "stage_a_detector_epochs": args.detector_epochs,
        "stage_b_residual_epochs_requested": args.residual_epochs,
        "train_selected_threshold": selected_rows["train"],
        "val_selected_threshold": selected_rows["val"],
        "test_selected_threshold": selected_rows["test"],
        "no_teacher_inference": True,
    }
    (args.output_root / "aggregate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_root / "README.md").write_text(
        f"""# Tiny Refiner v3.1 Staged Abstention

Detector-first then residual correction for full-frame identity fallback.
Training used only existing low-resolution S2M2->GT target shards from `{args.targets_root}`.
No S2M2, SAV, RAFT, DINO, oracle, or teacher inference was run.

- Parameters: `{params}`
- Detector epochs: `{args.detector_epochs}`
- Best residual epoch: `{ckpt['epoch']}`
- Selected hard threshold: `{best_threshold}`
- Frames: `{summary['frames']}`
- Crop size: `{args.crop_size}`
- Crops per epoch: `{args.crops_per_epoch}`
- Batch size: `{args.batch_size}`
- Context frames: `{args.context_frames}`
- Test raw MAE: `{selected_rows['test']['raw_mae']:.6f}`
- Test hard refined MAE: `{selected_rows['test']['refined_hard_mae']:.6f}`
- Test raw/refined bad-3px: `{selected_rows['test']['raw_bad3']:.6f}` -> `{selected_rows['test']['refined_hard_bad3']:.6f}`
- Test modified pixels: `{selected_rows['test']['fraction_modified_pct']:.6f}%`

Stage A trains the bad-pixel detector only. Stage B trains residual correction with the detector frozen first, then optionally unfrozen at a smaller LR.
Full-frame evaluation reports hard abstention thresholds in `threshold_sweep.csv`.
The deployed-safe path is hard abstention: pixels below the selected threshold stay raw S2M2.
"""
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
