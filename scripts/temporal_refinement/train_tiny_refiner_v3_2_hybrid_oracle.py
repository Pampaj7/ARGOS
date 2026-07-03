#!/usr/bin/env python3
"""Fine-tune v3.1 staged abstention refiner with hybrid full-GT + selected oracle targets.

Stage A loads the v3.1 checkpoint. Stage B/C fine-tunes mostly the residual head
(trunk/detector at very low LR) on mixed batches: full-GT safety batches plus
selected-clip oracle_all_available residual batches restricted to
oracle-beneficial hard pixels. No S2M2/SAV/RAFT/DINO inference is run.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "eval_scripts"))

from train_tiny_refiner_v1_full_gt import (  # noqa: E402
    charbonnier,
    finite_mean,
    masked_mean,
    parse_bool,
    write_csv,
)
from train_tiny_refiner_v3_1_staged_abstention import (  # noqa: E402
    DEFAULT_BALANCED_SPLIT,
    AbstentionCropRefiner,
    BalancedCropDataset,
    FullFrameDataset,
    focal_bce,
    evaluate,
    load_samples_with_split,
    make_features_from_raws,
    set_requires_grad,
    unwrap,
    write_csv_union,
    write_diagnostics,
)
from train_tiny_refiner_v1_full_gt import DEFAULT_TARGETS_ROOT, load_shards  # noqa: E402
from evaluate_v3_1_on_selected_oracle_clips import (  # noqa: E402
    auc_ap,
    diagnostic,
    metric,
    predict_clip,
    read_csv,
    summarize,
)


DEFAULT_ORACLE_TARGETS = Path("results/03_temporal_refinement/evaluation/distillation_targets_selected_clips")
DEFAULT_BASE_CHECKPOINT = Path("results/03_temporal_refinement/training/tiny_refiner_v3_1_staged_abstention/checkpoints/best.pt")
DEFAULT_OUTPUT_ROOT = Path("results/03_temporal_refinement/training/tiny_refiner_v3_2_hybrid_oracle")
SWEEP_THRESHOLDS = (0.3, 0.5, 0.7, 0.9, 0.95, 1.01)
IDENTITY_THRESHOLD = 1.01


@dataclass
class Clip:
    clip_id: str
    sequence_id: str
    failure_mode: str
    frame_ids: list[str]
    raws: np.ndarray
    gts: np.ndarray
    valids: np.ndarray
    oracle: np.ndarray
    sav: np.ndarray | None
    sup_mask: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.sup_mask = np.zeros_like(self.valids, dtype=bool)


def load_clips(root: Path, args: argparse.Namespace) -> list[Clip]:
    clip_index = {r["clip_id"]: r for r in read_csv(root / "clip_targets_index.csv")}
    clips: list[Clip] = []
    for clip_dir in sorted((root / "clips").iterdir()):
        if not clip_dir.is_dir():
            continue
        rows = read_csv(clip_dir / "frame_target_index.csv")
        meta = json.loads((clip_dir / "clip_metadata.json").read_text())
        frames = [np.load(r["target_path"]) for r in rows]
        clip = Clip(
            clip_id=clip_dir.name,
            sequence_id=meta["sequence_id"],
            failure_mode=clip_index.get(clip_dir.name, {}).get("dominant_failure_mode", ""),
            frame_ids=[r["frame_id"] for r in rows],
            raws=np.stack([f["raw_disp"].astype(np.float32) for f in frames]),
            gts=np.stack([f["gt_disp"].astype(np.float32) for f in frames]),
            valids=np.stack([f["valid_mask"].astype(np.float32) for f in frames]),
            oracle=np.stack([f["oracle_all_available_disp"].astype(np.float32) for f in frames]),
            sav=np.stack([f["sav_disp"].astype(np.float32) for f in frames]) if "sav_disp" in frames[0].files else None,
        )
        raw_err = np.abs(clip.raws - clip.gts)
        oracle_err = np.abs(clip.oracle - clip.gts)
        improvement = raw_err - oracle_err
        sup = (improvement > args.oracle_min_improvement_px) & (clip.valids > 0)
        if args.oracle_hard_only:
            sup &= raw_err >= args.bad_threshold_px
        clip.sup_mask = sup
        clips.append(clip)
    return clips


class OracleCropDataset(Dataset):
    """Random crops from selected oracle clips, biased toward oracle-beneficial pixels."""

    def __init__(self, clips: list[Clip], args: argparse.Namespace, crops_per_epoch: int):
        self.clips = clips
        self.args = args
        self.crops_per_epoch = crops_per_epoch
        self.frame_pool = [(ci, fi) for ci, c in enumerate(clips) for fi in range(len(c.frame_ids))]
        self.rng = random.Random(4321)

    def __len__(self) -> int:
        return self.crops_per_epoch

    def __getitem__(self, idx: int) -> dict[str, Any]:
        a = self.args
        ci, fi = self.frame_pool[self.rng.randrange(len(self.frame_pool))]
        clip = self.clips[ci]
        h, w = clip.raws.shape[1:]
        s = min(a.crop_size, h, w)
        best = (-1.0, 0, 0)
        for _ in range(a.crop_candidate_tries):
            y = self.rng.randint(0, max(0, h - s))
            x = self.rng.randint(0, max(0, w - s))
            score = float(clip.sup_mask[fi, y : y + s, x : x + s].mean())
            if score >= best[0]:
                best = (score, y, x)
        _, y, x = best
        ys, xs = slice(y, y + s), slice(x, x + s)
        ids = [max(0, fi - i) for i in range(a.context_frames)]
        raws = clip.raws[ids, ys, xs]
        valids = clip.valids[ids, ys, xs]
        xfeat, _edge, _var = make_features_from_raws(raws, valids)
        raw = raws[0]
        gt = clip.gts[fi, ys, xs]
        valid = valids[0]
        oracle_delta = clip.oracle[fi, ys, xs] - raw
        raw_err = np.abs(raw - gt)
        return {
            "x": torch.from_numpy(xfeat),
            "raw": torch.from_numpy(raw[None]),
            "gt": torch.from_numpy(gt[None]),
            "delta": torch.from_numpy(oracle_delta[None].astype(np.float32)),
            "valid": torch.from_numpy(valid[None]),
            "raw_bad": torch.from_numpy((raw_err >= a.bad_threshold_px).astype(np.float32)[None]),
            "raw_good": torch.from_numpy((raw_err < a.good_threshold_px).astype(np.float32)[None]),
            "sup": torch.from_numpy(clip.sup_mask[fi, ys, xs].astype(np.float32)[None]),
        }


def hybrid_loss_batch(model: nn.Module, batch: dict[str, torch.Tensor], args: argparse.Namespace, device: torch.device, source: str) -> tuple[torch.Tensor, dict[str, float]]:
    x = batch["x"].to(device, non_blocking=True)
    raw = batch["raw"].to(device, non_blocking=True)
    gt = batch["gt"].to(device, non_blocking=True)
    delta = batch["delta"].to(device, non_blocking=True)
    valid = batch["valid"].to(device, non_blocking=True)
    raw_bad = batch["raw_bad"].to(device, non_blocking=True) * valid
    raw_good = batch["raw_good"].to(device, non_blocking=True) * valid
    sup = batch["sup"].to(device, non_blocking=True) * valid if "sup" in batch else raw_bad
    bad_logit, p_bad, residual = model(x, args.residual_scale)
    refined = raw + p_bad * residual
    refined_err = torch.abs(refined - gt)
    raw_err = torch.abs(raw - gt)
    full_loss = masked_mean(torch.clamp(charbonnier(refined - gt), max=args.robust_loss_clip_px), valid)
    det_loss = focal_bce(bad_logit, raw_bad, valid, args.focal_gamma)
    # ponytail: target delta clamped to tanh-representable range; larger corrections need a bigger residual_scale, not this loss
    target = delta.clamp(-args.residual_scale, args.residual_scale)
    res_loss = masked_mean(charbonnier(residual - target), sup) if float(sup.sum()) > 0 else full_loss.new_tensor(0.0)
    preserve = masked_mean(torch.abs(p_bad * residual), raw_good) if float(raw_good.sum()) > 0 else full_loss.new_tensor(0.0)
    below3 = valid * (raw_err < args.bad_threshold_px).float()
    new_bad3 = masked_mean(torch.relu(refined_err - args.bad_threshold_px), below3) if float(below3.sum()) > 0 else full_loss.new_tensor(0.0)
    sparse = masked_mean(p_bad, valid) + (masked_mean(p_bad, raw_good) if float(raw_good.sum()) > 0 else full_loss.new_tensor(0.0))
    loss = (
        args.full_weight * full_loss
        + args.detector_weight * det_loss
        + args.residual_weight * res_loss
        + args.preserve_weight * preserve
        + args.new_bad3_weight * new_bad3
        + args.sparsity_weight * sparse
    )
    return loss, {
        f"{source}_loss": float(loss.detach().cpu()),
        f"{source}_full_loss": float(full_loss.detach().cpu()),
        f"{source}_det_loss": float(det_loss.detach().cpu()),
        f"{source}_residual_loss": float(res_loss.detach().cpu()),
        f"{source}_preserve_loss": float(preserve.detach().cpu()),
        f"{source}_new_bad3_loss": float(new_bad3.detach().cpu()),
        f"{source}_sparsity_loss": float(sparse.detach().cpu()),
        f"{source}_p_bad_mean": float(masked_mean(p_bad, valid).detach().cpu()),
        f"{source}_sup_frac": float(masked_mean(sup, valid).detach().cpu()),
    }


def train_one_epoch(model: nn.Module, gt_loader: DataLoader, oracle_loader: DataLoader, optimizer: torch.optim.Optimizer, args: argparse.Namespace, device: torch.device, rng: random.Random) -> dict[str, float]:
    model.train()
    order = ["gt"] * len(gt_loader) + ["oracle"] * len(oracle_loader)
    rng.shuffle(order)
    iters = {"gt": iter(gt_loader), "oracle": iter(oracle_loader)}
    rows: list[dict[str, float]] = []
    for source in order:
        try:
            batch = next(iters[source])
        except StopIteration:
            continue
        loss, metrics = hybrid_loss_batch(model, batch, args, device, source)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        rows.append(metrics)
    out: dict[str, float] = {}
    keys = {k for r in rows for k in r}
    for k in sorted(keys):
        out[k] = finite_mean([r[k] for r in rows if k in r])
    return out


@torch.no_grad()
def evaluate_selected(model: nn.Module, clips: list[Clip], args: argparse.Namespace, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Sweep thresholds over selected clips; returns aggregate rows and per-clip (p_bad, residual)."""
    model.eval()
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    agg = {t: {k: 0.0 for k in ("n", "raw_abs", "ref_abs", "oracle_abs", "sav_abs", "sav_n", "raw_bad3", "ref_bad3", "oracle_bad3", "raw_good", "new_bad3", "modified")} for t in SWEEP_THRESHOLDS}
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for clip in clips:
        _, p_bad, residual, _ = predict_clip(model, clip.raws, clip.valids, args, device, args.residual_scale, 0.5)
        predictions[clip.clip_id] = (p_bad, residual)
        valid = clip.valids > 0
        raw_err = np.abs(clip.raws - clip.gts)
        oracle_err = np.abs(clip.oracle - clip.gts)
        flat_valid = valid.reshape(-1)
        pick = np.flatnonzero(flat_valid)
        if pick.size:
            take = min(args.max_auc_pixels, pick.size)
            pick = pick[np.linspace(0, pick.size - 1, take, dtype=np.int64)]
            labels.append(((raw_err >= args.bad_threshold_px).reshape(-1)[pick]).astype(np.uint8))
            scores.append(p_bad.reshape(-1)[pick])
        for t in SWEEP_THRESHOLDS:
            hard = (p_bad >= t).astype(np.float32)
            refined = clip.raws + hard * residual
            ref_err = np.abs(refined - clip.gts)
            a = agg[t]
            a["n"] += float(valid.sum())
            a["raw_abs"] += float(raw_err[valid].sum())
            a["ref_abs"] += float(ref_err[valid].sum())
            a["oracle_abs"] += float(oracle_err[valid].sum())
            a["raw_bad3"] += float((raw_err[valid] > 3.0).sum())
            a["ref_bad3"] += float((ref_err[valid] > 3.0).sum())
            a["oracle_bad3"] += float((oracle_err[valid] > 3.0).sum())
            good = valid & (raw_err < 1.0)
            a["raw_good"] += float(good.sum())
            a["new_bad3"] += float((ref_err[good] >= 3.0).sum())
            a["modified"] += float(hard[valid].sum())
            if clip.sav is not None:
                sav_err = np.abs(clip.sav - clip.gts)
                a["sav_abs"] += float(sav_err[valid].sum())
                a["sav_n"] += float(valid.sum())
    auc, ap = auc_ap(np.concatenate(scores), np.concatenate(labels)) if scores else (float("nan"), float("nan"))
    rows = []
    for t in SWEEP_THRESHOLDS:
        a = agg[t]
        n = max(a["n"], 1.0)
        raw_mae = a["raw_abs"] / n
        ref_mae = a["ref_abs"] / n
        oracle_mae = a["oracle_abs"] / n
        sav_mae = a["sav_abs"] / max(a["sav_n"], 1.0) if a["sav_n"] > 0 else float("nan")
        rows.append({
            "threshold": t,
            "raw_mae": raw_mae,
            "refined_mae": ref_mae,
            "oracle_all_available_mae": oracle_mae,
            "sav_mae": sav_mae,
            "raw_bad3": 100.0 * a["raw_bad3"] / n,
            "refined_bad3": 100.0 * a["ref_bad3"] / n,
            "oracle_bad3": 100.0 * a["oracle_bad3"] / n,
            "new_bad3_from_raw_good_pct": 100.0 * a["new_bad3"] / max(a["raw_good"], 1.0),
            "modified_pixels_pct": 100.0 * a["modified"] / n,
            "oracle_gap_recovered": (raw_mae - ref_mae) / (raw_mae - oracle_mae) if raw_mae > oracle_mae else float("nan"),
            "sav_gap_recovered": (raw_mae - ref_mae) / (raw_mae - sav_mae) if math.isfinite(sav_mae) and raw_mae > sav_mae else float("nan"),
            "detector_auc": auc,
            "detector_ap": ap,
        })
    return rows, predictions


def selected_frame_rows(clips: list[Clip], predictions: dict[str, tuple[np.ndarray, np.ndarray]], threshold: float, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for clip in clips:
        p_bad, residual = predictions[clip.clip_id]
        hard = (p_bad >= threshold).astype(np.float32)
        refined = clip.raws + hard * residual
        for i, frame_id in enumerate(clip.frame_ids):
            valid = clip.valids[i] > 0
            raw_err = np.abs(clip.raws[i] - clip.gts[i])
            ref_err = np.abs(refined[i] - clip.gts[i])
            good = valid & (raw_err < 1.0)
            bad = valid & (raw_err >= 3.0)
            row: dict[str, Any] = {
                "clip_id": clip.clip_id,
                "sequence_id": clip.sequence_id,
                "frame_id": frame_id,
                "dominant_failure_mode": clip.failure_mode,
                "threshold": threshold,
                "modified_pixels_pct": float(hard[i][valid].mean() * 100.0) if valid.any() else float("nan"),
                "new_bad3_from_raw_good_pct": float((good & (ref_err >= 3.0)).sum() / max(good.sum(), 1) * 100.0),
                "fixed_bad3_pct": float((bad & (ref_err < 3.0)).sum() / max(bad.sum(), 1) * 100.0),
                "detector_auc": float("nan"),
                "detector_ap": float("nan"),
            }
            maps = {"raw": clip.raws[i], "refined": refined[i], "oracle_all_available": clip.oracle[i]}
            if clip.sav is not None:
                maps["sav"] = clip.sav[i]
            for method, pred in maps.items():
                m = metric(pred, clip.gts[i], clip.valids[i])
                row[f"{method}_mae"] = m["mae"]
                row[f"{method}_bad1"] = m["bad1"]
                row[f"{method}_bad3"] = m["bad3"]
            raw, ref = row["raw_mae"], row["refined_mae"]
            oracle = row["oracle_all_available_mae"]
            sav = row.get("sav_mae", float("nan"))
            row["oracle_gap_recovered"] = (raw - ref) / (raw - oracle) if math.isfinite(oracle) and raw > oracle else float("nan")
            row["sav_gap_recovered"] = (raw - ref) / (raw - sav) if math.isfinite(sav) and raw > sav else float("nan")
            rows.append(row)
    return rows


def choose_hybrid_threshold(full_rows: list[dict[str, Any]], sel_rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[float, bool]:
    full_by_thr = {round(float(r["threshold"]), 4): r for r in full_rows}
    feasible: list[dict[str, Any]] = []
    strict: list[dict[str, Any]] = []
    for se in sel_rows:
        thr = round(float(se["threshold"]), 4)
        fg = full_by_thr.get(thr)
        if fg is None or thr >= IDENTITY_THRESHOLD:
            continue
        ok = (
            fg["refined_hard_mae"] <= fg["raw_mae"] + args.full_gt_mae_tolerance
            and fg["refined_hard_bad3"] <= fg["raw_bad3"] + args.full_gt_bad3_tolerance
            and se["refined_mae"] < se["raw_mae"]
            and se["refined_bad3"] <= se["raw_bad3"]
            and se["new_bad3_from_raw_good_pct"] <= args.max_new_bad3_pct
        )
        if ok:
            feasible.append(se)
            if se["new_bad3_from_raw_good_pct"] <= args.baseline_selected_new_bad3:
                strict.append(se)
    pool = strict or feasible
    if not pool:
        return IDENTITY_THRESHOLD, False
    best = min(pool, key=lambda r: (r["refined_mae"], r["refined_bad3"]))
    return float(best["threshold"]), True


def make_loader(ds: Dataset, batch_size: int, num_workers: int, shuffle: bool, prefetch_factor: int) -> DataLoader:
    def reseed(worker_id: int) -> None:
        info = torch.utils.data.get_worker_info()
        if info is not None and hasattr(info.dataset, "rng"):
            info.dataset.rng = random.Random(10_000 * (worker_id + 1) + info.seed % 10_000)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        worker_init_fn=reseed if num_workers > 0 else None,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets-root", type=Path, default=DEFAULT_TARGETS_ROOT)
    p.add_argument("--oracle-targets-root", type=Path, default=DEFAULT_ORACLE_TARGETS)
    p.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--balanced-split-json", type=Path, default=DEFAULT_BALANCED_SPLIT)
    p.add_argument("--context-frames", type=int, default=4)
    p.add_argument("--crop-size", type=int, default=96)
    p.add_argument("--crop-candidate-tries", type=int, default=12)
    p.add_argument("--crops-per-epoch", type=int, default=60000)
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--oracle-batch-ratio", type=float, default=0.25)
    p.add_argument("--full-gt-batch-ratio", type=float, default=0.75)
    p.add_argument("--residual-lr", type=float, default=2e-4)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--detector-lr", type=float, default=1e-5)
    p.add_argument("--freeze-detector", nargs="?", const=True, default=False, type=parse_bool)
    p.add_argument("--num-workers", type=int, default=24)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--residual-scale", type=float, default=3.0)
    p.add_argument("--bad-threshold-px", type=float, default=3.0)
    p.add_argument("--good-threshold-px", type=float, default=1.0)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--oracle-min-improvement-px", type=float, default=1.0)
    p.add_argument("--oracle-hard-only", nargs="?", const=True, default=False, type=parse_bool)
    p.add_argument("--full-weight", type=float, default=0.5)
    p.add_argument("--detector-weight", type=float, default=0.2)
    p.add_argument("--residual-weight", type=float, default=0.5)
    p.add_argument("--preserve-weight", type=float, default=1.0)
    p.add_argument("--new-bad3-weight", type=float, default=0.5)
    p.add_argument("--sparsity-weight", type=float, default=0.10)
    p.add_argument("--robust-loss-clip-px", type=float, default=10.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--early-stop-patience", type=int, default=5)
    p.add_argument("--max-auc-pixels", type=int, default=200000)
    p.add_argument("--full-gt-mae-tolerance", type=float, default=0.0)
    p.add_argument("--full-gt-bad3-tolerance", type=float, default=0.05)
    p.add_argument("--max-new-bad3-pct", type=float, default=5.0)
    p.add_argument("--baseline-selected-new-bad3", type=float, default=4.79)
    p.add_argument("--baseline-selected-mae", type=float, default=11.0421)
    p.add_argument("--baseline-oracle-gap-recovered", type=float, default=0.0622)
    p.add_argument("--diagnostics-per-clip", type=int, default=2)
    p.add_argument("--lr-schedule", choices=("none", "cosine", "step"), default="none")
    p.add_argument("--lr-decay-start-epoch", type=int, default=16)
    p.add_argument("--lr-decay-step-size", type=int, default=8)
    p.add_argument("--lr-decay-gamma", type=float, default=0.5)
    p.add_argument("--overwrite", nargs="?", const=True, default=False, type=parse_bool)
    args = p.parse_args()
    total = args.oracle_batch_ratio + args.full_gt_batch_ratio
    args.oracle_batch_ratio /= total
    args.full_gt_batch_ratio /= total
    return args


def make_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    m: AbstentionCropRefiner = unwrap(model)  # type: ignore[assignment]
    set_requires_grad(m.residual_head, True)
    set_requires_grad(m.trunk, not args.freeze_detector)
    set_requires_grad(m.bad_head, not args.freeze_detector)
    groups = [{"params": list(m.residual_head.parameters()), "lr": args.residual_lr}]
    if not args.freeze_detector:
        groups.append({"params": list(m.trunk.parameters()), "lr": args.backbone_lr})
        groups.append({"params": list(m.bad_head.parameters()), "lr": args.detector_lr})
    return torch.optim.AdamW(groups)


def apply_lr_schedule(optimizer: torch.optim.Optimizer, base_lrs: list[float], epoch: int, args: argparse.Namespace) -> float:
    """Scale each group's LR from its base value; returns the scale factor applied."""
    start = args.lr_decay_start_epoch
    if args.lr_schedule == "none" or epoch < start:
        scale = 1.0
    elif args.lr_schedule == "cosine":
        span = max(1, args.epochs - start)
        progress = min(1.0, (epoch - start) / span)
        scale = 0.5 * (1.0 + math.cos(math.pi * progress))
    else:  # step
        steps = (epoch - start) // max(1, args.lr_decay_step_size)
        scale = args.lr_decay_gamma**steps
    for group, base in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base * scale
    return scale


def checkpoint_payload(model: nn.Module, args: argparse.Namespace, splits: dict[str, list[str]], input_channels: int, epoch: int, threshold: float, safe_gain: bool, val_full: dict[str, Any], val_sel: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_state_dict": unwrap(model).state_dict(),
        "args": vars(args),
        "splits": splits,
        "input_channels": input_channels,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "epoch": epoch,
        "threshold": threshold,
        "safe_oracle_gain": safe_gain,
        "val_full_gt_metrics": val_full,
        "val_selected_metrics": val_sel,
    }


def row_at(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    return min(rows, key=lambda r: abs(float(r["threshold"]) - threshold))


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
    start = time.perf_counter()

    ckpt = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    input_channels = int(ckpt.get("input_channels", args.context_frames * 2 + 8))
    base_threshold = float(ckpt.get("threshold", 0.5))
    model = AbstentionCropRefiner(input_channels).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    params = sum(p.numel() for p in model.parameters())

    splits, by_split = load_samples_with_split(args.targets_root, args.balanced_split_json, args.max_frames)
    all_samples = by_split["train"] + by_split["val"] + by_split["test"]
    shards = load_shards(all_samples)
    clips = load_clips(args.oracle_targets_root, args)

    gt_crops = int(round(args.crops_per_epoch * args.full_gt_batch_ratio))
    oracle_crops = args.crops_per_epoch - gt_crops
    gt_args = argparse.Namespace(**{**vars(args), "crops_per_epoch": gt_crops})
    gt_train_ds = BalancedCropDataset(by_split["train"], shards, gt_args)
    oracle_train_ds = OracleCropDataset(clips, args, oracle_crops)
    full_ds = {split: FullFrameDataset(samples, shards, args.context_frames) for split, samples in by_split.items()}
    gt_loader = make_loader(gt_train_ds, args.batch_size, args.num_workers, True, args.prefetch_factor)
    oracle_loader = make_loader(oracle_train_ds, args.batch_size, max(2, args.num_workers // 3), True, args.prefetch_factor)
    eval_loaders = {split: make_loader(ds, args.eval_batch_size, max(0, args.num_workers // 2), False, args.prefetch_factor) for split, ds in full_ds.items()}

    run_lines = [
        f"device={device} gpus={torch.cuda.device_count() if device.type == 'cuda' else 0}",
        f"params={params} input_channels={input_channels}",
        f"base_checkpoint={args.base_checkpoint} base_threshold={base_threshold} base_epoch={ckpt.get('epoch')}",
        f"splits={json.dumps(splits)}",
        f"frames={{'train': {len(full_ds['train'])}, 'val': {len(full_ds['val'])}, 'test': {len(full_ds['test'])}}}",
        f"clips={len(clips)} clip_frames={sum(len(c.frame_ids) for c in clips)}",
        f"sup_pixels_pct={100.0 * float(np.mean([c.sup_mask.mean() for c in clips])):.3f}",
        f"crops_per_epoch={args.crops_per_epoch} gt_crops={gt_crops} oracle_crops={oracle_crops} batch_size={args.batch_size}",
        f"lrs residual={args.residual_lr} backbone={args.backbone_lr} detector={args.detector_lr} freeze_detector={args.freeze_detector}",
        f"sweep_thresholds={SWEEP_THRESHOLDS}",
    ]
    (args.output_root / "run.log").write_text("\n".join(run_lines) + "\n")

    optimizer = make_optimizer(model, args)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    epoch_rng = random.Random(999)
    train_rows: list[dict[str, Any]] = []
    sweep_full_rows: list[dict[str, Any]] = []
    sweep_sel_rows: list[dict[str, Any]] = []
    best_score = float("inf")
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        lr_scale = apply_lr_schedule(optimizer, base_lrs, epoch, args)
        train_metrics = train_one_epoch(model, gt_loader, oracle_loader, optimizer, args, device, epoch_rng)
        train_metrics["lr_scale"] = lr_scale
        full_val_rows, _ = evaluate(model, eval_loaders["val"], args, device, "val", per_sequence=False)
        sel_rows, _preds = evaluate_selected(model, clips, args, device)
        threshold, safe_gain = choose_hybrid_threshold(full_val_rows, sel_rows, args)
        fg = row_at(full_val_rows, threshold)
        se = row_at(sel_rows, threshold)
        score = se["refined_mae"] if safe_gain else se["raw_mae"] + 1.0
        train_rows.append({"epoch": epoch, "threshold": threshold, "safe_oracle_gain": safe_gain, **train_metrics, "val_full_gt_refined_mae": fg["refined_hard_mae"], "val_full_gt_raw_mae": fg["raw_mae"], "sel_refined_mae": se["refined_mae"], "sel_raw_mae": se["raw_mae"], "sel_new_bad3": se["new_bad3_from_raw_good_pct"], "sel_oracle_gap": se["oracle_gap_recovered"]})
        sweep_full_rows.extend({"epoch": epoch, "split": "val", **r} for r in full_val_rows)
        sweep_sel_rows.extend({"epoch": epoch, **r} for r in sel_rows)
        if score < best_score:
            best_score = score
            best_epoch = epoch
            torch.save(checkpoint_payload(model, args, splits, input_channels, epoch, threshold, safe_gain, fg, se), args.output_root / "checkpoints" / "best.pt")
        run_lines.append(
            f"epoch={epoch} thr={threshold} safe_gain={safe_gain} lr_scale={lr_scale:.4f} "
            f"fullgt_val raw={fg['raw_mae']:.4f} refined={fg['refined_hard_mae']:.4f} bad3={fg['raw_bad3']:.3f}->{fg['refined_hard_bad3']:.3f} "
            f"sel raw={se['raw_mae']:.4f} refined={se['refined_mae']:.4f} gap={se['oracle_gap_recovered']:.4f} "
            f"new_bad3={se['new_bad3_from_raw_good_pct']:.3f}% modified={se['modified_pixels_pct']:.2f}%"
        )
        (args.output_root / "run.log").write_text("\n".join(run_lines) + "\n")
        if epoch - best_epoch >= args.early_stop_patience:
            run_lines.append(f"early_stop epoch={epoch} best_epoch={best_epoch}")
            (args.output_root / "run.log").write_text("\n".join(run_lines) + "\n")
            break

    if best_epoch == 0:
        torch.save(checkpoint_payload(model, args, splits, input_channels, 0, IDENTITY_THRESHOLD, False, {}, {}), args.output_root / "checkpoints" / "best.pt")
    best = torch.load(args.output_root / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    unwrap(model).load_state_dict(best["model_state_dict"])

    final_full_rows: dict[str, list[dict[str, Any]]] = {}
    for split in ("val", "test"):
        rows, _ = evaluate(model, eval_loaders[split], args, device, split, per_sequence=False)
        final_full_rows[split] = rows
    final_sel_rows, predictions = evaluate_selected(model, clips, args, device)
    threshold, safe_gain = choose_hybrid_threshold(final_full_rows["val"], final_sel_rows, args)
    frame_rows = selected_frame_rows(clips, predictions, threshold, args)
    for r in frame_rows:
        r["detector_auc"] = final_sel_rows[0]["detector_auc"]
        r["detector_ap"] = final_sel_rows[0]["detector_ap"]
    failure_rows = summarize(frame_rows, "dominant_failure_mode")

    write_csv_union(args.output_root / "train_log.csv", train_rows)
    write_csv(args.output_root / "full_gt_val_metrics.csv", [row_at(final_full_rows["val"], threshold)])
    write_csv(args.output_root / "full_gt_test_metrics.csv", [row_at(final_full_rows["test"], threshold)])
    write_csv_union(args.output_root / "threshold_sweep_full_gt.csv", sweep_full_rows + [{"epoch": "final", "split": s, **r} for s in ("val", "test") for r in final_full_rows[s]])
    write_csv_union(args.output_root / "threshold_sweep_selected_oracle.csv", sweep_sel_rows + [{"epoch": "final", **r} for r in final_sel_rows])
    write_csv_union(args.output_root / "selected_oracle_metrics.csv", frame_rows)
    write_csv_union(args.output_root / "failure_mode_summary.csv", failure_rows)

    write_diagnostics(model, full_ds["val"], args.output_root / "diagnostics", args, device, "val", threshold)
    write_diagnostics(model, full_ds["test"], args.output_root / "diagnostics", args, device, "test", threshold)
    for clip in clips:
        p_bad, residual = predictions[clip.clip_id]
        hard = (p_bad >= threshold).astype(np.float32)
        refined = clip.raws + hard * residual
        for i in np.linspace(0, len(clip.frame_ids) - 1, min(args.diagnostics_per_clip, len(clip.frame_ids)), dtype=int):
            diagnostic(
                args.output_root / "diagnostics" / f"selected_{clip.clip_id}_{clip.frame_ids[i]}.png",
                clip.raws[i], refined[i], clip.gts[i], clip.oracle[i],
                clip.sav[i] if clip.sav is not None else None,
                clip.valids[i], p_bad[i], hard[i],
            )

    fg_val = row_at(final_full_rows["val"], threshold)
    fg_test = row_at(final_full_rows["test"], threshold)
    se = row_at(final_sel_rows, threshold)
    summary = {
        "targets_root": str(args.targets_root),
        "oracle_targets_root": str(args.oracle_targets_root),
        "base_checkpoint": str(args.base_checkpoint),
        "output_root": str(args.output_root),
        "params": params,
        "best_epoch": best["epoch"],
        "selected_threshold": threshold,
        "safe_oracle_gain": safe_gain,
        "elapsed_seconds": time.perf_counter() - start,
        "full_gt_val": fg_val,
        "full_gt_test": fg_test,
        "selected_oracle": se,
        "baseline_v3_1": {
            "selected_mae": args.baseline_selected_mae,
            "selected_new_bad3": args.baseline_selected_new_bad3,
            "oracle_gap_recovered": args.baseline_oracle_gap_recovered,
        },
        "success_vs_baseline": {
            "oracle_gap_beats_baseline": bool(safe_gain and math.isfinite(se["oracle_gap_recovered"]) and se["oracle_gap_recovered"] > args.baseline_oracle_gap_recovered),
            "selected_mae_beats_baseline": bool(se["refined_mae"] < args.baseline_selected_mae),
            "new_bad3_controlled": bool(se["new_bad3_from_raw_good_pct"] <= args.max_new_bad3_pct),
        },
        "no_teacher_inference": True,
    }
    (args.output_root / "aggregate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_root / "README.md").write_text(
        f"""# Tiny Refiner v3.2 Hybrid Oracle Fine-Tune

Fine-tunes the v3.1 staged abstention checkpoint with mixed batches:
`{args.full_gt_batch_ratio:.0%}` full-GT safety batches from `{args.targets_root}` and
`{args.oracle_batch_ratio:.0%}` selected oracle batches from `{args.oracle_targets_root}`.
Oracle supervision uses `oracle_all_available_disp` restricted to pixels where the oracle
beats raw by more than `{args.oracle_min_improvement_px}px`. SAV is only used through the
oracle selection, never as a global target. No S2M2, SAV, RAFT, or DINO inference was run.

- Base checkpoint: `{args.base_checkpoint}`
- Parameters: `{params}`
- Best epoch: `{best['epoch']}`
- Selected threshold: `{threshold}` (safe oracle gain: `{safe_gain}`)
- Full-GT test: raw MAE `{fg_test['raw_mae']:.4f}` -> refined `{fg_test['refined_hard_mae']:.4f}`, Bad-3 `{fg_test['raw_bad3']:.3f}` -> `{fg_test['refined_hard_bad3']:.3f}`, new Bad-3 from raw-good `{fg_test['new_bad3_from_raw_good_pct']:.3f}%`
- Selected clips: raw MAE `{se['raw_mae']:.4f}` -> refined `{se['refined_mae']:.4f}` (oracle-all `{se['oracle_all_available_mae']:.4f}`)
- Selected oracle gap recovered: `{se['oracle_gap_recovered']:.4f}` (v3.1 baseline `{args.baseline_oracle_gap_recovered}`)
- Selected new Bad-3 from raw-good: `{se['new_bad3_from_raw_good_pct']:.3f}%` (v3.1 baseline `{args.baseline_selected_new_bad3}%`)
- Selected modified pixels: `{se['modified_pixels_pct']:.2f}%`

Threshold selection is constrained: full-GT val must not regress beyond tolerance, selected
Bad-3 must not worsen, and new Bad-3 from raw-good must stay <= `{args.max_new_bad3_pct}%`.
If no threshold satisfies the constraints the run falls back to the identity-safe
threshold `{IDENTITY_THRESHOLD}` and reports no safe oracle gain.
"""
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
