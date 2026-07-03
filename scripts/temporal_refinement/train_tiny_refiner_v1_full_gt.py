#!/usr/bin/env python3
"""Train tiny_refiner_v1 on full sharded S2M2->GT low-res targets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


DEFAULT_TARGETS_ROOT = Path("results/03_temporal_refinement/evaluation/s2m2_gt_refiner_targets_full")
DEFAULT_OUTPUT_ROOT = Path("results/03_temporal_refinement/training/tiny_refiner_v1_full_gt")
DISP_SCALE = 64.0


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def finite_mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def colorize(value: np.ndarray, vmax: float | None = None, cmap: int = cv2.COLORMAP_TURBO) -> np.ndarray:
    arr = np.nan_to_num(value.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if vmax is None:
        vals = arr[np.isfinite(arr)]
        vmax = float(np.percentile(vals, 98)) if vals.size else 1.0
    norm = np.clip(arr / max(float(vmax), 1e-6), 0.0, 1.0)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cmap)


class TinyRefinerV1(nn.Module):
    def __init__(self, in_channels: int = 4, hidden: int = 40):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_channels, hidden, 3, padding=1), nn.GroupNorm(8, hidden), nn.SiLU(inplace=True)]
        for _ in range(4):
            layers += [nn.Conv2d(hidden, hidden, 3, padding=1), nn.GroupNorm(8, hidden), nn.SiLU(inplace=True)]
        layers.append(nn.Conv2d(hidden, 1, 1))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) * DISP_SCALE


@dataclass(frozen=True)
class Sample:
    sequence_id: str
    frame_id: str
    frame_index: int
    offset: int
    target_path: Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def split_sequences(sequence_ids: list[str]) -> dict[str, list[str]]:
    splits = {"train": [], "val": [], "test": []}
    for i, seq in enumerate(sorted(sequence_ids)):
        if i % 7 == 5:
            splits["val"].append(seq)
        elif i % 7 == 6:
            splits["test"].append(seq)
        else:
            splits["train"].append(seq)
    return splits


def load_samples(targets_root: Path, max_frames: int) -> tuple[dict[str, list[str]], dict[str, list[Sample]]]:
    rows = read_rows(targets_root / "frame_targets_index.csv")
    splits = split_sequences(sorted({r["sequence_id"] for r in rows}))
    by_split: dict[str, list[Sample]] = {k: [] for k in splits}
    split_of = {seq: split for split, seqs in splits.items() for seq in seqs}
    for row in rows:
        split = split_of[row["sequence_id"]]
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


def load_shards(samples: list[Sample]) -> dict[Path, dict[str, np.ndarray]]:
    shards: dict[Path, dict[str, np.ndarray]] = {}
    for path in sorted({s.target_path for s in samples}):
        z = np.load(path)
        shards[path] = {k: z[k] for k in ["raw_disp", "gt_disp", "valid_mask", "delta_disp_gt_minus_raw"]}
    return shards


class FullGtDataset(Dataset):
    def __init__(self, samples: list[Sample], shards: dict[Path, dict[str, np.ndarray]]):
        self.samples = samples
        self.shards = shards

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        shard = self.shards[sample.target_path]
        raw = shard["raw_disp"][sample.offset].astype(np.float32)
        gt = shard["gt_disp"][sample.offset].astype(np.float32)
        valid = shard["valid_mask"][sample.offset].astype(np.float32)
        delta = shard["delta_disp_gt_minus_raw"][sample.offset].astype(np.float32)
        prev_offset = max(0, sample.offset - 1)
        prev_raw = shard["raw_disp"][prev_offset].astype(np.float32)
        abs_dt = np.abs(raw - prev_raw)
        x = np.stack([raw / DISP_SCALE, prev_raw / DISP_SCALE, abs_dt / DISP_SCALE, valid], axis=0).astype(np.float32)
        return {
            "x": torch.from_numpy(x),
            "raw": torch.from_numpy(raw[None]),
            "gt": torch.from_numpy(gt[None]),
            "delta": torch.from_numpy(delta[None]),
            "valid": torch.from_numpy(valid[None]),
            "sequence_id": sample.sequence_id,
            "frame_id": sample.frame_id,
        }


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def charbonnier(value: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(value * value + eps * eps)


def bad_rate(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor, tau: float) -> float:
    return float(masked_mean((torch.abs(pred - gt) > tau).float(), valid).detach().cpu() * 100.0)


def batch_metrics(raw: torch.Tensor, refined: torch.Tensor, gt: torch.Tensor, delta_pred: torch.Tensor, delta: torch.Tensor, valid: torch.Tensor) -> dict[str, float]:
    raw_mae = float(masked_mean(torch.abs(raw - gt), valid).detach().cpu())
    refined_mae = float(masked_mean(torch.abs(refined - gt), valid).detach().cpu())
    return {
        "raw_disp_mae_vs_gt": raw_mae,
        "refined_disp_mae_vs_gt": refined_mae,
        "raw_bad_1px": bad_rate(raw, gt, valid, 1.0),
        "refined_bad_1px": bad_rate(refined, gt, valid, 1.0),
        "raw_bad_3px": bad_rate(raw, gt, valid, 3.0),
        "refined_bad_3px": bad_rate(refined, gt, valid, 3.0),
        "residual_l1": float(masked_mean(torch.abs(delta_pred - delta), valid).detach().cpu()),
        "relative_mae_improvement": (raw_mae - refined_mae) / raw_mae if raw_mae > 1e-8 else float("nan"),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, split: str) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    seq_rows: dict[str, list[dict[str, float]]] = {}
    for batch in loader:
        x = batch["x"].to(device)
        raw = batch["raw"].to(device)
        gt = batch["gt"].to(device)
        delta = batch["delta"].to(device)
        valid = batch["valid"].to(device)
        pred_delta = model(x)
        refined = raw + pred_delta
        metrics = batch_metrics(raw, refined, gt, pred_delta, delta, valid)
        rows.append(metrics)
        for seq in set(batch["sequence_id"]):
            mask = [s == seq for s in batch["sequence_id"]]
            idx = torch.tensor(mask, device=device, dtype=torch.bool)
            sm = batch_metrics(raw[idx], refined[idx], gt[idx], pred_delta[idx], delta[idx], valid[idx])
            seq_rows.setdefault(str(seq), []).append(sm)
    out = {"split": split, "frames": len(loader.dataset)}
    for key in rows[0]:
        out[key] = finite_mean([r[key] for r in rows])
    seq_out = [{"split": split, "sequence_id": seq, "frames": sum(1 for s in loader.dataset.samples if s.sequence_id == seq), **{k: finite_mean([r[k] for r in vals]) for k in vals[0]}} for seq, vals in sorted(seq_rows.items())]
    return out, seq_out


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, residual_clip: float) -> dict[str, float]:
    model.train()
    losses: list[float] = []
    for batch in loader:
        x = batch["x"].to(device)
        delta = batch["delta"].to(device).clamp(-residual_clip, residual_clip)
        valid = batch["valid"].to(device)
        pred_delta = model(x).clamp(-residual_clip, residual_clip)
        loss = masked_mean(charbonnier(pred_delta - delta), valid)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return {"loss": finite_mean(losses)}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def write_diagnostics(model: nn.Module, dataset: FullGtDataset, out_dir: Path, device: torch.device, prefix: str, count: int = 4) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for idx in np.linspace(0, max(0, len(dataset) - 1), min(count, len(dataset)), dtype=int):
        item = dataset[int(idx)]
        pred_delta = model(item["x"][None].to(device))[0, 0].cpu().numpy()
        raw = item["raw"][0].numpy()
        gt = item["gt"][0].numpy()
        valid = item["valid"][0].numpy()
        refined = raw + pred_delta
        target_delta = item["delta"][0].numpy()
        vmax = float(np.percentile(gt[valid > 0], 98)) if np.any(valid > 0) else 64.0
        tiles = [
            colorize(raw, vmax),
            colorize(refined, vmax),
            colorize(gt, vmax),
            colorize(np.abs(raw - gt), 16.0, cv2.COLORMAP_MAGMA),
            colorize(np.abs(refined - gt), 16.0, cv2.COLORMAP_MAGMA),
            colorize(np.abs(target_delta), 16.0, cv2.COLORMAP_MAGMA),
            colorize(np.abs(pred_delta), 16.0, cv2.COLORMAP_MAGMA),
            colorize(valid, 1.0, cv2.COLORMAP_VIRIDIS),
        ]
        cv2.imwrite(str(out_dir / f"{prefix}_{item['sequence_id']}_{item['frame_id']}.png"), np.concatenate(tiles, axis=1))


def make_loader(ds: FullGtDataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=True)


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets-root", type=Path, default=DEFAULT_TARGETS_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--residual-clip", type=float, default=64.0)
    p.add_argument("--overwrite", nargs="?", const=True, default=False, type=parse_bool)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(args.output_root)
        shutil.rmtree(args.output_root)
    (args.output_root / "checkpoints").mkdir(parents=True)
    (args.output_root / "diagnostics").mkdir()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    splits, by_split = load_samples(args.targets_root, args.max_frames)
    all_samples = by_split["train"] + by_split["val"] + by_split["test"]
    shards = load_shards(all_samples)
    datasets = {split: FullGtDataset(samples, shards) for split, samples in by_split.items()}
    loaders = {
        "train": make_loader(datasets["train"], args, True),
        "val": make_loader(datasets["val"], args, False),
        "test": make_loader(datasets["test"], args, False),
    }
    model = TinyRefinerV1().to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best = float("inf")
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    run_log = [
        f"device={device}",
        f"params={params}",
        f"splits={json.dumps(splits)}",
        f"frames={{'train': {len(datasets['train'])}, 'val': {len(datasets['val'])}, 'test': {len(datasets['test'])}}}",
        f"shards_loaded={len(shards)}",
    ]
    start = time.perf_counter()
    (args.output_root / "run.log").write_text("\n".join(run_log) + "\n")
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, loaders["train"], optimizer, device, args.residual_clip)
        val_metrics, _seq = evaluate(model, loaders["val"], device, "val")
        train_rows.append({"epoch": epoch, **train_metrics})
        val_rows.append({"epoch": epoch, **val_metrics})
        if val_metrics["refined_disp_mae_vs_gt"] < best:
            best = val_metrics["refined_disp_mae_vs_gt"]
            torch.save(
                {
                    "model_state_dict": unwrap(model).state_dict(),
                    "args": vars(args),
                    "splits": splits,
                    "input_channels": 4,
                    "parameter_count": params,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                args.output_root / "checkpoints" / "best.pt",
            )
        run_log.append(
            f"epoch={epoch} train_loss={train_metrics['loss']:.6f} "
            f"val_raw={val_metrics['raw_disp_mae_vs_gt']:.6f} val_refined={val_metrics['refined_disp_mae_vs_gt']:.6f}"
        )
        (args.output_root / "run.log").write_text("\n".join(run_log) + "\n")

    ckpt = torch.load(args.output_root / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    unwrap(model).load_state_dict(ckpt["model_state_dict"])
    val_metrics, val_seq = evaluate(model, loaders["val"], device, "val")
    test_metrics, test_seq = evaluate(model, loaders["test"], device, "test")
    train_metrics_eval, train_seq = evaluate(model, loaders["train"], device, "train")

    write_csv(args.output_root / "train_log.csv", train_rows)
    write_csv(args.output_root / "val_metrics.csv", val_rows)
    write_csv(args.output_root / "test_metrics.csv", [test_metrics])
    write_csv(args.output_root / "sequence_metrics.csv", train_seq + val_seq + test_seq)
    write_diagnostics(model, datasets["val"], args.output_root / "diagnostics", device, "val")
    write_diagnostics(model, datasets["test"], args.output_root / "diagnostics", device, "test")
    summary = {
        "targets_root": str(args.targets_root),
        "output_root": str(args.output_root),
        "params": params,
        "best_epoch": ckpt["epoch"],
        "elapsed_seconds": time.perf_counter() - start,
        "frames": {k: len(v) for k, v in datasets.items()},
        "splits": splits,
        "train": train_metrics_eval,
        "val": val_metrics,
        "test": test_metrics,
        "no_teacher_inference": True,
    }
    (args.output_root / "aggregate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_root / "README.md").write_text(
        f"""# Tiny Refiner v1 Full-GT

Trained on full low-resolution S2M2->GT residual targets from `{args.targets_root}`.
No S2M2, SAV, RAFT, DINO, or oracle/teacher inference was run during training.

- Parameters: `{params}`
- Best epoch: `{ckpt['epoch']}`
- Split by sequence: `{splits}`
- Frames: `{summary['frames']}`
- Test raw MAE: `{test_metrics['raw_disp_mae_vs_gt']:.6f}`
- Test refined MAE: `{test_metrics['refined_disp_mae_vs_gt']:.6f}`
- Test relative MAE improvement: `{test_metrics['relative_mae_improvement']:.6f}`
"""
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
