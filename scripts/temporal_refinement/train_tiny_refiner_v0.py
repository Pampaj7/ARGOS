#!/usr/bin/env python3
"""Smoke-train a tiny refiner from selected-clip oracle targets only."""

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


DEFAULT_TARGETS_ROOT = Path("results/03_temporal_refinement/evaluation/distillation_targets_selected_clips")
DEFAULT_OUTPUT_ROOT = Path("results/03_temporal_refinement/training/tiny_refiner_v0_smoke")
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
    vmax = max(float(vmax), 1e-6)
    norm = np.clip(arr / vmax, 0.0, 1.0)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cmap)


class TinyRefinerV0(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.GroupNorm(4, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(4, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 2, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)
        return out[:, :1] * DISP_SCALE, out[:, 1:2]


@dataclass(frozen=True)
class Sample:
    clip_id: str
    frame_id: str
    target_path: Path
    prev_target_path: Path | None
    raw_disp_mae: float
    oracle_all_available_disp_mae: float


def read_clip_rows(targets_root: Path) -> tuple[list[dict[str, str]], dict[str, list[Sample]]]:
    with (targets_root / "clip_targets_index.csv").open(newline="") as f:
        clip_rows = list(csv.DictReader(f))
    by_clip: dict[str, list[Sample]] = {}
    for clip in clip_rows:
        clip_id = clip["clip_id"]
        rows_path = targets_root / "clips" / clip_id / "frame_target_index.csv"
        rows = list(csv.DictReader(rows_path.open(newline="")))
        samples: list[Sample] = []
        prev: Path | None = None
        for row in rows:
            path = Path(row["target_path"])
            samples.append(
                Sample(
                    clip_id=clip_id,
                    frame_id=row["frame_id"],
                    target_path=path,
                    prev_target_path=prev,
                    raw_disp_mae=float(row.get("raw_disp_mae") or "nan"),
                    oracle_all_available_disp_mae=float(row.get("oracle_all_available_disp_mae") or "nan"),
                )
            )
            prev = path
        by_clip[clip_id] = samples
    return clip_rows, by_clip


def split_clips(clip_rows: list[dict[str, str]]) -> dict[str, list[str]]:
    ids = [row["clip_id"] for row in clip_rows]
    return {"train": ids[:4], "val": ids[4:5], "test": ids[5:6]}


def cap_samples(samples: list[Sample], max_frames: int) -> list[Sample]:
    return samples[:max_frames] if max_frames > 0 else samples


class TargetDataset(Dataset):
    def __init__(self, samples: list[Sample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def load_raw(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        z = np.load(path)
        oracle = z["oracle_all_available_disp" if "oracle_all_available_disp" in z.files else "oracle_disp_all_available"].astype(np.float32)
        delta = z["delta_disp_oracle_all_available_minus_raw"].astype(np.float32)
        valid = z["valid_mask"].astype(np.float32)
        conf = z["raw_confidence_binary"].astype(np.float32)
        raw = z["raw_disp"].astype(np.float32) if "raw_disp" in z.files else oracle - delta
        gt = z["gt_disp"].astype(np.float32) if "gt_disp" in z.files else oracle
        return raw, oracle, delta, valid, conf, gt

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        raw, oracle, delta, valid, conf, gt = self.load_raw(sample.target_path)
        if sample.prev_target_path is None:
            prev_raw = raw
        else:
            prev_raw, _prev_oracle, _prev_delta, _prev_valid, _prev_conf, _prev_gt = self.load_raw(sample.prev_target_path)
        abs_dt = np.abs(raw - prev_raw)
        x = np.stack([raw / DISP_SCALE, prev_raw / DISP_SCALE, abs_dt / DISP_SCALE, valid], axis=0).astype(np.float32)
        return {
            "x": torch.from_numpy(x),
            "raw": torch.from_numpy(raw[None].astype(np.float32)),
            "oracle": torch.from_numpy(oracle[None].astype(np.float32)),
            "gt": torch.from_numpy(gt[None].astype(np.float32)),
            "delta": torch.from_numpy(delta[None].astype(np.float32)),
            "valid": torch.from_numpy(valid[None].astype(np.float32)),
            "conf": torch.from_numpy(conf[None].astype(np.float32)),
            "clip_id": sample.clip_id,
            "frame_id": sample.frame_id,
            "raw_disp_mae": torch.tensor(sample.raw_disp_mae, dtype=torch.float32),
            "oracle_all_available_disp_mae": torch.tensor(sample.oracle_all_available_disp_mae, dtype=torch.float32),
        }


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    den = mask.sum().clamp_min(1.0)
    return (value * mask).sum() / den


def confidence_metrics(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> dict[str, float]:
    bce = masked_mean(nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none"), valid)
    prob = torch.sigmoid(logits)
    pred = (prob >= 0.5).float()
    acc = masked_mean((pred == target).float(), valid)
    out = {"confidence_bce": float(bce.detach().cpu()), "confidence_accuracy": float(acc.detach().cpu())}
    try:
        from sklearn.metrics import roc_auc_score

        m = valid.detach().cpu().numpy().astype(bool).reshape(-1)
        y = target.detach().cpu().numpy().reshape(-1)[m]
        p = prob.detach().cpu().numpy().reshape(-1)[m]
        out["confidence_auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    except Exception:
        out["confidence_auc"] = float("nan")
    return out


def bad_rate(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor, tau: float) -> float:
    return float(masked_mean((torch.abs(pred - gt) > tau).float(), valid).detach().cpu() * 100.0)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    residual_losses: list[float] = []
    raw_to_oracle: list[float] = []
    refined_to_oracle: list[float] = []
    raw_gt: list[float] = []
    oracle_gt: list[float] = []
    refined_gt: list[float] = []
    raw_bad1: list[float] = []
    refined_bad1: list[float] = []
    oracle_bad1: list[float] = []
    raw_bad3: list[float] = []
    refined_bad3: list[float] = []
    oracle_bad3: list[float] = []
    bces: list[float] = []
    accs: list[float] = []
    aucs: list[float] = []
    for batch in loader:
        x = batch["x"].to(device)
        valid = batch["valid"].to(device)
        target_delta = batch["delta"].to(device)
        conf = batch["conf"].to(device)
        raw = batch["raw"].to(device)
        oracle = batch["oracle"].to(device)
        gt = batch["gt"].to(device)
        pred_delta, conf_logits = model(x)
        refined = raw + pred_delta
        residual_losses.append(float(masked_mean(torch.abs(pred_delta - target_delta), valid).cpu()))
        raw_to_oracle.append(float(masked_mean(torch.abs(raw - oracle), valid).cpu()))
        refined_to_oracle.append(float(masked_mean(torch.abs(refined - oracle), valid).cpu()))
        raw_gt.append(float(masked_mean(torch.abs(raw - gt), valid).cpu()))
        refined_gt.append(float(masked_mean(torch.abs(refined - gt), valid).cpu()))
        oracle_gt.append(float(masked_mean(torch.abs(oracle - gt), valid).cpu()))
        raw_bad1.append(bad_rate(raw, gt, valid, 1.0))
        refined_bad1.append(bad_rate(refined, gt, valid, 1.0))
        oracle_bad1.append(bad_rate(oracle, gt, valid, 1.0))
        raw_bad3.append(bad_rate(raw, gt, valid, 3.0))
        refined_bad3.append(bad_rate(refined, gt, valid, 3.0))
        oracle_bad3.append(bad_rate(oracle, gt, valid, 3.0))
        cm = confidence_metrics(conf_logits, conf, valid)
        bces.append(cm["confidence_bce"])
        accs.append(cm["confidence_accuracy"])
        if math.isfinite(cm["confidence_auc"]):
            aucs.append(cm["confidence_auc"])
    raw_oracle_l1 = finite_mean(raw_to_oracle)
    refined_oracle_l1 = finite_mean(refined_to_oracle)
    gap = raw_oracle_l1 - refined_oracle_l1
    raw_mae = finite_mean(raw_gt)
    refined_mae = finite_mean(refined_gt)
    oracle_mae = finite_mean(oracle_gt)
    gt_gap_den = raw_mae - oracle_mae
    return {
        "raw_disp_mae": raw_mae,
        "refined_disp_mae": refined_mae,
        "oracle_all_available_disp_mae": oracle_mae,
        "raw_to_oracle_l1": raw_oracle_l1,
        "refined_to_oracle_l1": refined_oracle_l1,
        "residual_l1": finite_mean(residual_losses),
        "raw_bad_1px": finite_mean(raw_bad1),
        "refined_bad_1px": finite_mean(refined_bad1),
        "oracle_bad_1px": finite_mean(oracle_bad1),
        "raw_bad_3px": finite_mean(raw_bad3),
        "refined_bad_3px": finite_mean(refined_bad3),
        "oracle_bad_3px": finite_mean(oracle_bad3),
        "confidence_bce": finite_mean(bces),
        "confidence_accuracy": finite_mean(accs),
        "confidence_auc": finite_mean(aucs),
        "target_oracle_gap_recovered": gap / raw_oracle_l1 if raw_oracle_l1 > 1e-8 else float("nan"),
        "oracle_gap_recovered": (raw_mae - refined_mae) / gt_gap_den if abs(gt_gap_den) > 1e-8 else float("nan"),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    confidence_weight: float,
) -> dict[str, float]:
    model.train()
    losses: list[float] = []
    residuals: list[float] = []
    confs: list[float] = []
    for batch in loader:
        x = batch["x"].to(device)
        valid = batch["valid"].to(device)
        delta = batch["delta"].to(device)
        conf = batch["conf"].to(device)
        pred_delta, conf_logits = model(x)
        residual_loss = masked_mean(torch.abs(pred_delta - delta), valid)
        conf_loss = masked_mean(nn.functional.binary_cross_entropy_with_logits(conf_logits, conf, reduction="none"), valid)
        loss = residual_loss + confidence_weight * conf_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        residuals.append(float(residual_loss.detach().cpu()))
        confs.append(float(conf_loss.detach().cpu()))
    return {"loss": finite_mean(losses), "residual_l1": finite_mean(residuals), "confidence_bce": finite_mean(confs)}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def write_diagnostics(model: nn.Module, dataset: TargetDataset, out_dir: Path, device: torch.device, count: int = 4) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for idx in range(min(count, len(dataset))):
        item = dataset[idx]
        x = item["x"][None].to(device)
        pred_delta, conf_logits = model(x)
        raw = item["raw"][0].numpy()
        oracle = item["oracle"][0].numpy()
        gt = item["gt"][0].numpy()
        refined = raw + pred_delta[0, 0].cpu().numpy()
        target_delta = item["delta"][0].numpy()
        pred_delta_np = pred_delta[0, 0].cpu().numpy()
        conf_t = item["conf"][0].numpy()
        conf_p = torch.sigmoid(conf_logits)[0, 0].cpu().numpy()
        raw_err = np.abs(raw - gt)
        refined_err = np.abs(refined - gt)
        oracle_err = np.abs(oracle - gt)
        vmax = float(np.nanpercentile(gt[gt > 0], 98)) if np.any(gt > 0) else 64.0
        tiles = [
            colorize(raw, vmax),
            colorize(refined, vmax),
            colorize(gt, vmax),
            colorize(oracle, vmax),
            colorize(raw_err, 16.0, cv2.COLORMAP_MAGMA),
            colorize(refined_err, 16.0, cv2.COLORMAP_MAGMA),
            colorize(oracle_err, 16.0, cv2.COLORMAP_MAGMA),
            colorize(np.abs(target_delta), 16.0, cv2.COLORMAP_MAGMA),
            colorize(np.abs(pred_delta_np), 16.0, cv2.COLORMAP_MAGMA),
            colorize(conf_t, 1.0, cv2.COLORMAP_VIRIDIS),
            colorize(conf_p, 1.0, cv2.COLORMAP_VIRIDIS),
        ]
        sheet = np.concatenate(tiles, axis=1)
        cv2.imwrite(str(out_dir / f"{item['clip_id']}_{item['frame_id']}.png"), sheet)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets-root", type=Path, default=DEFAULT_TARGETS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--confidence-loss-weight", type=float, default=0.1)
    parser.add_argument("--overwrite", nargs="?", const=True, default=False, type=parse_bool)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(args.output_root)
        shutil.rmtree(args.output_root)
    (args.output_root / "checkpoints").mkdir(parents=True)
    (args.output_root / "diagnostics").mkdir()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    clip_rows, by_clip = read_clip_rows(args.targets_root)
    splits = split_clips(clip_rows)
    train_samples = cap_samples([s for cid in splits["train"] for s in by_clip[cid]], args.max_frames)
    val_samples = [s for cid in splits["val"] for s in by_clip[cid]]
    test_samples = [s for cid in splits["test"] for s in by_clip[cid]]
    if not train_samples or not val_samples:
        raise RuntimeError("Need non-empty train and val splits")

    sample_keys = list(np.load(train_samples[0].target_path).files)
    train_ds = TargetDataset(train_samples)
    val_ds = TargetDataset(val_samples)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = TinyRefinerV0(in_channels=4).to(device)
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best = float("inf")
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    with (args.output_root / "run.log").open("w") as log:
        log.write(f"device={device}\\nparams={params}\\nkeys={','.join(sample_keys)}\\n")
        log.write(f"splits={json.dumps(splits)}\\ntrain_frames={len(train_samples)} val_frames={len(val_samples)} test_frames={len(test_samples)}\\n")
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_one_epoch(model, train_loader, optimizer, device, args.confidence_loss_weight)
            val_metrics = evaluate(model, val_loader, device)
            train_rows.append({"epoch": epoch, **train_metrics})
            val_rows.append({"epoch": epoch, **val_metrics})
            score = val_metrics["refined_disp_mae"]
            if score < best:
                best = score
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "args": vars(args),
                        "input_channels": 4,
                        "parameter_count": params,
                        "npz_keys": sample_keys,
                        "splits": splits,
                        "epoch": epoch,
                        "val_metrics": val_metrics,
                    },
                    args.output_root / "checkpoints" / "best.pt",
                )
            log.write(
                f"epoch={epoch} train_loss={train_metrics['loss']:.6f} "
                f"val_refined_disp_mae={val_metrics['refined_disp_mae']:.6f} "
                f"val_refined_to_oracle_l1={val_metrics['refined_to_oracle_l1']:.6f}\\n"
            )
            log.flush()

    write_csv(args.output_root / "train_log.csv", train_rows)
    write_csv(args.output_root / "val_metrics.csv", val_rows)
    final_val = val_rows[-1].copy()
    final_val.update({"split": "val", "frames": len(val_samples), "test_frames_metadata_only": len(test_samples)})
    write_csv(args.output_root / "test_or_val_predictions_summary.csv", [final_val])
    write_diagnostics(model, val_ds, args.output_root / "diagnostics", device)

    readme = f"""# Tiny Refiner v0 Corrected-Target Training

This training run uses only compressed selected-clip `.npz` targets from `{args.targets_root}`.
No S2M2, SAV, RAFT, DINO, or teacher inference was run.

- Train/val/test split by clip: `{splits}`
- Train frames used: `{len(train_samples)}` (`--max-frames {args.max_frames}`)
- Validation frames: `{len(val_samples)}`
- Held-out test metadata frames: `{len(test_samples)}`
- NPZ keys available: `{sample_keys}`
- Model parameters: `{params}`

Validation reports target-space residual metrics and GT-space metrics from saved `raw_disp`, `gt_disp`, `oracle_all_available_disp`, and `valid_mask` in the corrected `.npz` targets. No external GT loading is required.

Recommended next step after this smoke: continue only if `refined_disp_mae` and diagnostics improve over raw on validation.
"""
    (args.output_root / "README.md").write_text(readme)
    summary = {"output_root": str(args.output_root), "elapsed_seconds": time.perf_counter() - start, "params": params, "final_val": val_rows[-1]}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
