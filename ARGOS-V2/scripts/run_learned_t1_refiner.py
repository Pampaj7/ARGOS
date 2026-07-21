#!/usr/bin/env python3
"""Train and evaluate the minimal causal learned t-1 ARGOS v2 refiner."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

V2_ROOT = Path(__file__).resolve().parents[1]
ROOT = V2_ROOT.parent
sys.path.insert(0, str(V2_ROOT))
sys.path.insert(0, str(V2_ROOT / "scripts"))

from argos_v2.scared_c_data import load_frame_gt, load_sequence_info  # noqa: E402
from model_design.data.temporal_pair_dataset import (  # noqa: E402
    DEFAULT_VALIDATION_SEQUENCES,
    PRIMARY_UNSEEN_BACKBONE,
    SEEN_BACKBONES,
    TemporalPairDataset,
    build_split_manifest,
)
from model_design.external_components.bidavideo import (  # noqa: E402
    BiDAFlowInferenceAdapter,
    temporal_disparity_evidence,
)
from model_design.losses.safety_losses import SafetyLossConfig, learned_t1_losses  # noqa: E402
from model_design.models.learned_t1_refiner import LearnedT1Refiner  # noqa: E402

CACHE_SIZE = (144, 180)
BASELINE_METHODS = (
    "raw",
    "memory",
    "blend_0.1",
    "blend_0.25",
    "blend_0.5",
    "heuristic_gate",
    "learned_refiner",
    "oracle",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "train", "evaluate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(f"A{i}" for i in range(2, 8)), default="A7")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--backbones", nargs="+", default=list(SEEN_BACKBONES))
    parser.add_argument("--train-sequences", nargs="+")
    parser.add_argument("--validation-sequences", nargs="+", default=list(DEFAULT_VALIDATION_SEQUENCES))
    parser.add_argument("--coverage-threshold", type=float, default=0.50)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.25, 0.50, 0.90])
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-train-pairs-per-sequence", type=int, default=256)
    parser.add_argument("--max-validation-pairs-per-sequence", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps", type=int, default=0, help="fixed optimizer steps; mainly smoke")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8))
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=("none", "onecycle"), default="none")
    parser.add_argument("--tau-px", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--contact-sheets", type=int, default=4)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.unlink(missing_ok=True)
    append_csv(path, rows)


def save_json(path: Path, value: object) -> None:
    def clean(item):
        if isinstance(item, dict):
            return {key: clean(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(child) for child in item]
        if isinstance(item, (float, np.floating)) and not math.isfinite(float(item)):
            return None
        if isinstance(item, np.generic):
            return item.item()
        return item
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def atomic_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def make_loader(dataset: TemporalPairDataset, args: argparse.Namespace, *, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        drop_last=False,
        generator=generator,
    )


def to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def build_evidence(adapter: BiDAFlowInferenceAdapter, batch: dict) -> tuple[dict[str, torch.Tensor], float]:
    current_rgb = batch["current_rgb"]
    past_rgb = batch["past_rgb"]
    target = torch.cat((current_rgb, past_rgb), dim=0)
    source = torch.cat((past_rgb, current_rgb), dim=0)
    if current_rgb.is_cuda:
        torch.cuda.synchronize(current_rgb.device)
    start = time.perf_counter()
    inferred = adapter.infer(target, source)
    if current_rgb.is_cuda:
        torch.cuda.synchronize(current_rgb.device)
    flow_ms = (time.perf_counter() - start) * 1000 / current_rgb.shape[0]
    # Clone converts inference tensors into ordinary detached tensors that are safe
    # as constant convolution inputs during refiner backpropagation.
    flows = inferred.clone()
    count = current_rgb.shape[0]
    with torch.no_grad():
        evidence_object = temporal_disparity_evidence(
            batch["raw"],
            batch["past"],
            flows[:count],
            flows[count:],
            current_valid=batch["raw_valid"],
            past_valid=batch["past_valid"],
            current_rgb=current_rgb,
            past_rgb=past_rgb,
        )
        evidence = {name: value.detach() for name, value in evidence_object.as_dict().items()}
        evidence["current_valid"] = batch["raw_valid"].detach()
    return evidence, flow_ms


def loss_config(variant: str) -> SafetyLossConfig:
    config = SafetyLossConfig()
    if variant in {"A2", "A3", "A4", "A5"}:
        return replace(config, clean_weight=0.0, ranking_weight=0.0)
    if variant == "A6":
        return replace(config, ranking_weight=0.0)
    return config


def validation_score(
    model: LearnedT1Refiner,
    adapter: BiDAFlowInferenceAdapter,
    loader: DataLoader,
    device: torch.device,
    coverage_threshold: float,
) -> tuple[float, float, float]:
    model.eval()
    raw_sum = refined_sum = oracle_sum = 0.0
    count = 0
    with torch.no_grad():
        for batch_cpu in loader:
            batch = to_device(batch_cpu, device)
            evidence, _ = build_evidence(adapter, batch)
            output = model(batch["raw"], evidence, batch["current_rgb"])
            valid = (
                (batch["gt_coverage"] > coverage_threshold)
                & batch["raw_valid"].bool()
                & evidence["aligned_validity"].bool()
                & evidence["warp_support"].bool()
            )
            raw_error = (batch["raw"] - batch["gt"]).abs()
            refined_error = (output.disparity - batch["gt"]).abs()
            memory_error = (evidence["aligned_past_disparity"] - batch["gt"]).abs()
            n = int(valid.sum())
            raw_sum += float(raw_error[valid].sum())
            refined_sum += float(refined_error[valid].sum())
            oracle_sum += float(torch.minimum(raw_error, memory_error)[valid].sum())
            count += n
    return raw_sum / max(count, 1), refined_sum / max(count, 1), oracle_sum / max(count, 1)


def train(args: argparse.Namespace, *, smoke: bool = False) -> int:
    seed_everything(args.seed)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = build_split_manifest(
        seed=args.seed,
        coverage_threshold=args.coverage_threshold,
        frame_stride=args.frame_stride,
        validation_sequences=args.validation_sequences,
    )
    train_sequences = args.train_sequences or manifest["train_sequences"]
    backbones = ["S2M2-S"] if smoke else args.backbones
    if PRIMARY_UNSEEN_BACKBONE in backbones:
        raise ValueError("Fast-FoundationStereo cannot enter training or checkpoint selection")
    if smoke:
        train_sequences = [train_sequences[0]]
        args.max_train_pairs_per_sequence = min(args.max_train_pairs_per_sequence, 24)
        args.max_validation_pairs_per_sequence = min(args.max_validation_pairs_per_sequence, 8)
    manifest["actual_train_sequences"] = list(train_sequences)
    manifest["actual_training_backbones"] = list(backbones)
    manifest["mode"] = "smoke" if smoke else "train"
    save_json(args.output / "split_manifest.json", manifest)
    save_json(args.output / "config.json", vars(args) | {"output": str(args.output), "checkpoint": str(args.checkpoint) if args.checkpoint else None})

    train_dataset = TemporalPairDataset(
        backbones,
        train_sequences,
        coverage_threshold=args.coverage_threshold,
        frame_stride=args.frame_stride,
        max_pairs_per_sequence=args.max_train_pairs_per_sequence,
        random_clip_start=True,
        seed=args.seed,
    )
    validation_dataset = TemporalPairDataset(
        backbones,
        args.validation_sequences,
        coverage_threshold=args.coverage_threshold,
        frame_stride=args.frame_stride,
        max_pairs_per_sequence=args.max_validation_pairs_per_sequence,
        random_clip_start=False,
        seed=args.seed,
    )
    train_loader = make_loader(train_dataset, args, shuffle=True)
    validation_loader = make_loader(validation_dataset, args, shuffle=False)
    model = LearnedT1Refiner(args.variant, tau_px=args.tau_px).to(device)
    adapter = BiDAFlowInferenceAdapter("sea_raft", device=device)
    assert all(not parameter.requires_grad for parameter in adapter.model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    target_epochs = args.epochs if not smoke else max(args.epochs, 1_000_000)
    scheduler = None
    if args.scheduler == "onecycle":
        if smoke:
            raise ValueError("onecycle is reserved for fixed-length full training")
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.learning_rate,
            epochs=target_epochs,
            steps_per_epoch=len(train_loader),
        )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    config = loss_config(args.variant)
    start_epoch = 0
    best_epe = math.inf
    best_path = args.output / "checkpoints/best_validation.pt"
    final_path = args.output / "checkpoints/final.pt"
    history_path = args.output / "training_history.csv"
    if args.resume and final_path.exists():
        state = torch.load(final_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        best_epe = float(state["best_validation_epe"])
    elif not args.resume:
        history_path.unlink(missing_ok=True)

    log = (args.output / "run.log").open("a", buffering=1)
    print(f"COMMAND {' '.join(sys.argv)}", file=log)
    print(f"DATA train_pairs={len(train_dataset)} validation_pairs={len(validation_dataset)}", file=log)
    initial_gate = None
    global_step = 0
    for epoch in range(start_epoch, target_epochs):
        model.train()
        sums: defaultdict[str, float] = defaultdict(float)
        batches = 0
        for batch_cpu in train_loader:
            batch = to_device(batch_cpu, device)
            evidence, flow_ms = build_evidence(adapter, batch)
            valid = (
                batch["gt_valid"].bool()
                & batch["raw_valid"].bool()
                & evidence["aligned_validity"].bool()
                & evidence["warp_support"].bool()
            )
            safety_valid = valid
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(batch["raw"], evidence, batch["current_rgb"])
                losses = learned_t1_losses(
                    output,
                    raw=batch["raw"],
                    aligned_memory=evidence["aligned_past_disparity"],
                    gt=batch["gt"],
                    valid=valid,
                    safety_valid=safety_valid,
                    config=config,
                )
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            gate_mean = float((output.g_error * output.c_memory).detach().mean())
            if initial_gate is None:
                initial_gate = gate_mean
            for name, value in losses.items():
                sums[name] += float(value.detach())
            sums["gradient_norm"] += gradient_norm
            sums["gate_mean"] += gate_mean
            sums["update_abs_max"] = max(sums["update_abs_max"], float(output.update.detach().abs().max()))
            sums["flow_latency_ms"] += flow_ms
            batches += 1
            global_step += 1
            if args.steps and global_step >= args.steps:
                break

        raw_epe, refined_epe, oracle_epe = validation_score(
            model, adapter, validation_loader, device, args.coverage_threshold
        )
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{name: value / max(batches, 1) for name, value in sums.items() if name != "update_abs_max"},
            "update_abs_max": sums["update_abs_max"],
            "validation_raw_epe": raw_epe,
            "validation_refined_epe": refined_epe,
            "validation_oracle_epe": oracle_epe,
            "validation_oracle_recovery": (raw_epe - refined_epe) / max(raw_epe - oracle_epe, 1e-8),
        }
        append_csv(history_path, [row])
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_validation_epe": min(best_epe, refined_epe),
            "variant": args.variant,
            "tau_px": args.tau_px,
            "split_manifest": manifest,
            "loss_config": asdict(config),
        }
        if refined_epe < best_epe:
            best_epe = refined_epe
            payload["best_validation_epe"] = best_epe
            atomic_checkpoint(best_path, payload)
        atomic_checkpoint(final_path, payload)
        print(json.dumps(row), file=log)
        if args.steps and global_step >= args.steps:
            break
    finite = all(torch.isfinite(parameter).all() for parameter in model.parameters())
    smoke_result = {
        "passed": bool(
            finite
            and initial_gate is not None
            and row["gradient_norm"] > 0
            and row["gate_mean"] != initial_gate
            and row["update_abs_max"] <= args.tau_px + 1e-5
        ),
        "finite": bool(finite),
        "initial_gate_mean": initial_gate,
        "final_gate_mean": row["gate_mean"],
        "final_gradient_norm": row["gradient_norm"],
        "max_update_px": row["update_abs_max"],
        "initial_loss": None,
        "final_loss": row["total"],
    }
    if history_path.exists():
        history = list(csv.DictReader(history_path.open()))
        if history:
            smoke_result["initial_loss"] = float(history[0]["total"])
            smoke_result["loss_reduction_ratio"] = 1 - row["total"] / max(float(history[0]["total"]), 1e-8)
            if smoke:
                smoke_result["passed"] &= smoke_result["loss_reduction_ratio"] > 0.20
    save_json(args.output / "smoke_result.json", smoke_result)
    print(f"COMPLETE best_validation_epe={best_epe:.6f} smoke={smoke_result}", file=log)
    log.close()
    return 0 if (not smoke or smoke_result["passed"]) else 2


def boundary_mask(gt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    dx = cv2.Sobel(gt, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gt, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.hypot(dx, dy) > 1.0
    edge |= cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    return cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool) & valid


def frame_metrics(raw: np.ndarray, pred: np.ndarray, gt: np.ndarray, common: np.ndarray, boundary: np.ndarray) -> dict:
    count = int(common.sum())
    if not count:
        return {name: math.nan for name in ("epe", "bad1", "bad3", "absrel", "boundary_epe", "new_bad3", "false_update_rate", "clean_degradation_ratio", "clean_update_mean", "frame_degradation") } | {"valid_count": 0, "clean_count": 0, "raw_good3_count": 0, "boundary_count": 0}
    raw_error = np.abs(raw - gt)
    error = np.abs(pred - gt)
    update = np.abs(pred - raw)
    clean = common & (raw_error <= 0.50)
    raw_good3 = common & (raw_error <= 3.0)
    boundary_common = common & boundary
    return {
        "valid_count": count,
        "clean_count": int(clean.sum()),
        "raw_good3_count": int(raw_good3.sum()),
        "boundary_count": int(boundary_common.sum()),
        "epe": float(error[common].mean()),
        "bad1": float((error[common] > 1).mean()),
        "bad3": float((error[common] > 3).mean()),
        "absrel": float((error[common] / np.maximum(gt[common], 1e-6)).mean()),
        "boundary_epe": float(error[boundary_common].mean()) if boundary_common.any() else math.nan,
        "new_bad3": float((error[raw_good3] > 3).mean()) if raw_good3.any() else math.nan,
        "false_update_rate": float((update[clean] > 0.05).mean()) if clean.any() else math.nan,
        "clean_degradation_ratio": float((error[clean] > raw_error[clean] + 0.02).mean()) if clean.any() else math.nan,
        "clean_update_mean": float(update[clean].mean()) if clean.any() else math.nan,
        "frame_degradation": float(error[common].mean() - raw_error[common].mean()),
    }


def calibration_summary(scores: list[float], targets: list[float], *, binary: bool) -> dict:
    if not scores:
        return {"count": 0}
    probability = np.asarray(scores, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    bins = []
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        selected = (probability >= lo) & (probability < hi if hi < 1 else probability <= hi)
        if not selected.any():
            continue
        confidence = float(probability[selected].mean())
        accuracy = float(target[selected].mean())
        fraction = float(selected.mean())
        ece += fraction * abs(confidence - accuracy)
        bins.append({"lo": lo, "hi": hi, "count": int(selected.sum()), "confidence": confidence, "target": accuracy})
    result = {
        "count": len(target),
        "brier": float(np.mean((probability - target) ** 2)),
        "ece_10": float(ece),
        "bins": bins,
    }
    if binary and len(np.unique(target)) == 2:
        chosen = probability >= 0.5
        positive = target > 0.5
        tp = int((chosen & positive).sum())
        fp = int((chosen & ~positive).sum())
        fn = int((~chosen & positive).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        result["average_precision"] = float(average_precision_score(target, probability))
        result["auroc"] = float(roc_auc_score(target, probability))
        result["precision_at_0.5"] = precision
        result["recall_at_0.5"] = recall
        result["f1_at_0.5"] = 2 * precision * recall / max(precision + recall, 1e-12)
    return result


def aggregate_frame_rows(rows: list[dict]) -> tuple[list[dict], dict, dict]:
    sequence_groups: defaultdict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        sequence_groups[(row["namespace"], row["coverage_threshold"], row["backbone"], row["sequence"], row["method"])].append(row)
    sequence_rows = []
    for key, group in sequence_groups.items():
        total = sum(row["valid_count"] for row in group)
        clean_total = sum(row["clean_count"] for row in group)
        good3_total = sum(row["raw_good3_count"] for row in group)
        boundary_total = sum(row["boundary_count"] for row in group)
        out = dict(zip(("namespace", "coverage_threshold", "backbone", "sequence", "method"), key))
        for metric in ("epe", "bad1", "bad3", "absrel"):
            out[metric] = sum(row[metric] * row["valid_count"] for row in group if math.isfinite(row[metric])) / max(total, 1)
        out["boundary_epe"] = sum(row["boundary_epe"] * row["boundary_count"] for row in group if math.isfinite(row["boundary_epe"])) / max(boundary_total, 1)
        for metric, denominator in (("false_update_rate", clean_total), ("clean_degradation_ratio", clean_total), ("clean_update_mean", clean_total), ("new_bad3", good3_total)):
            weight_key = "clean_count" if metric != "new_bad3" else "raw_good3_count"
            out[metric] = sum(row[metric] * row[weight_key] for row in group if math.isfinite(row[metric])) / max(denominator, 1)
        degradations = np.asarray([row["frame_degradation"] for row in group if math.isfinite(row["frame_degradation"])])
        out.update({
            "valid_count": total,
            "clean_count": clean_total,
            "raw_good3_count": good3_total,
            "boundary_count": boundary_total,
            "common_valid_ratio": float(np.mean([row["common_valid_ratio"] for row in group])),
            "frames": len(group),
            "frames_worsened_ratio": float((degradations > 0).mean()) if len(degradations) else math.nan,
            "worst_frame_degradation": float(degradations.max()) if len(degradations) else math.nan,
            "p95_frame_degradation": float(np.quantile(degradations, 0.95)) if len(degradations) else math.nan,
        })
        sequence_rows.append(out)

    aggregate_groups: defaultdict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        aggregate_groups[(row["namespace"], row["coverage_threshold"], row["backbone"], row["method"])].append(row)
    aggregate = []
    for key, group in aggregate_groups.items():
        total = sum(row["valid_count"] for row in group)
        clean_total = sum(row["clean_count"] for row in group)
        good3_total = sum(row["raw_good3_count"] for row in group)
        boundary_total = sum(row["boundary_count"] for row in group)
        out = dict(zip(("namespace", "coverage_threshold", "backbone", "method"), key))
        for metric in ("epe", "bad1", "bad3", "absrel"):
            out[metric] = sum(row[metric] * row["valid_count"] for row in group if math.isfinite(row[metric])) / max(total, 1)
        out["boundary_epe"] = sum(row["boundary_epe"] * row["boundary_count"] for row in group if math.isfinite(row["boundary_epe"])) / max(boundary_total, 1)
        for metric, weight_key, denominator in (
            ("false_update_rate", "clean_count", clean_total),
            ("clean_degradation_ratio", "clean_count", clean_total),
            ("clean_update_mean", "clean_count", clean_total),
            ("new_bad3", "raw_good3_count", good3_total),
        ):
            out[metric] = sum(row[metric] * row[weight_key] for row in group if math.isfinite(row[metric])) / max(denominator, 1)
        out["valid_count"] = total
        out["common_valid_ratio"] = float(np.mean([row["common_valid_ratio"] for row in group]))
        degradations = np.asarray([row["frame_degradation"] for row in group if math.isfinite(row["frame_degradation"])])
        out["frames_worsened_ratio"] = float((degradations > 0).mean()) if len(degradations) else math.nan
        out["worst_frame_degradation"] = float(degradations.max()) if len(degradations) else math.nan
        out["p95_frame_degradation"] = float(np.quantile(degradations, 0.95)) if len(degradations) else math.nan
        aggregate.append(out)

    lookup = {(row["namespace"], row["coverage_threshold"], row["backbone"], row["method"]): row for row in aggregate}
    oracle_analysis = {}
    safety = {}
    for row in aggregate:
        if row["method"] != "learned_refiner":
            continue
        prefix = (row["namespace"], row["coverage_threshold"], row["backbone"])
        raw = lookup[prefix + ("raw",)]
        oracle = lookup[prefix + ("oracle",)]
        denominator = max(raw["epe"] - oracle["epe"], 1e-8)
        key = "/".join(map(str, prefix))
        oracle_analysis[key] = {
            "raw_epe": raw["epe"],
            "learned_epe": row["epe"],
            "oracle_epe": oracle["epe"],
            "oracle_gain": raw["epe"] - oracle["epe"],
            "learned_gain": raw["epe"] - row["epe"],
            "oracle_recovery": (raw["epe"] - row["epe"]) / denominator,
        }
        safety[key] = {name: row[name] for name in ("new_bad3", "false_update_rate", "clean_degradation_ratio", "clean_update_mean", "frames_worsened_ratio", "worst_frame_degradation", "p95_frame_degradation")}
    return sequence_rows, {"rows": aggregate, "oracle_analysis": oracle_analysis}, safety


def contact_sheet(path: Path, rgb: np.ndarray, gt: np.ndarray, raw: np.ndarray, refined: np.ndarray, gate: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def color(value: np.ndarray, lo: float, hi: float) -> np.ndarray:
        scaled = np.clip((value - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        return cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    rgb_bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    panels = [rgb_bgr, color(gt, 0, 32), color(raw, 0, 32), color(refined, 0, 32), color(np.abs(refined - raw), 0, 3), color(gate, 0, 1)]
    labels = ["RGB", "GT", "raw", "refined", "|update|", "g_error*c_memory"]
    for panel, label in zip(panels, labels):
        cv2.putText(panel, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), np.concatenate(panels, axis=1))


def evaluate(args: argparse.Namespace) -> int:
    seed_everything(args.seed)
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for evaluate")
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    variant = state.get("variant", args.variant)
    model = LearnedT1Refiner(variant, tau_px=float(state.get("tau_px", args.tau_px))).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    adapter = BiDAFlowInferenceAdapter("sea_raft", device=device)
    dataset = TemporalPairDataset(
        args.backbones,
        args.validation_sequences,
        coverage_threshold=args.coverage_threshold,
        frame_stride=args.frame_stride,
        max_pairs_per_sequence=args.max_validation_pairs_per_sequence,
        random_clip_start=False,
        seed=args.seed,
    )
    loader = make_loader(dataset, args, shuffle=False)
    rows: list[dict] = []
    sampled: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    flow_latencies: list[float] = []
    model_latencies: list[float] = []
    sheets = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for batch_cpu in loader:
            batch = to_device(batch_cpu, device)
            evidence, flow_ms = build_evidence(adapter, batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            output = model(batch["raw"], evidence, batch["current_rgb"])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            model_ms = (time.perf_counter() - start) * 1000 / batch["raw"].shape[0]
            flow_latencies.append(flow_ms)
            model_latencies.append(model_ms)
            heuristic_gate = 0.5 * evidence["forward_backward_confidence"] * (1 - evidence["photometric_residual"]) * evidence["aligned_validity"].float()
            predictions_t = {
                "raw": batch["raw"],
                "memory": evidence["aligned_past_disparity"],
                "blend_0.1": 0.9 * batch["raw"] + 0.1 * evidence["aligned_past_disparity"],
                "blend_0.25": 0.75 * batch["raw"] + 0.25 * evidence["aligned_past_disparity"],
                "blend_0.5": 0.5 * batch["raw"] + 0.5 * evidence["aligned_past_disparity"],
                "heuristic_gate": batch["raw"] + heuristic_gate * (evidence["aligned_past_disparity"] - batch["raw"]),
                "learned_refiner": output.disparity,
            }
            raw_error_t = (batch["raw"] - batch["gt"]).abs()
            memory_error_t = (evidence["aligned_past_disparity"] - batch["gt"]).abs()
            predictions_t["oracle"] = torch.where(memory_error_t < raw_error_t, evidence["aligned_past_disparity"], batch["raw"])
            for index in range(batch["raw"].shape[0]):
                raw = batch["raw"][index, 0].cpu().numpy()
                gt = batch["gt"][index, 0].cpu().numpy()
                aligned_valid = evidence["aligned_validity"][index, 0].cpu().numpy().astype(bool)
                support = evidence["warp_support"][index, 0].cpu().numpy().astype(bool)
                raw_valid = batch["raw_valid"][index, 0].cpu().numpy().astype(bool)
                coverage = batch["gt_coverage"][index, 0].cpu().numpy()
                score = (output.g_error[index, 0] * output.c_memory[index, 0]).cpu().numpy()
                error_score = output.g_error[index, 0].cpu().numpy()
                memory_score = output.c_memory[index, 0].cpu().numpy()
                memory = evidence["aligned_past_disparity"][index, 0].cpu().numpy()
                raw_error = np.abs(raw - gt)
                memory_error = np.abs(memory - gt)
                for threshold in args.thresholds:
                    common = (coverage > threshold) & raw_valid & aligned_valid & support
                    boundary = boundary_mask(gt, coverage > threshold)
                    base = {
                        "namespace": "cache",
                        "coverage_threshold": threshold,
                        "backbone": batch["backbone"][index],
                        "sequence": batch["sequence"][index],
                        "frame_id": batch["current_frame_id"][index],
                        "common_valid_ratio": float(common.mean()),
                        "flow_latency_ms": flow_ms,
                        "model_latency_ms": model_ms,
                    }
                    for method in BASELINE_METHODS:
                        prediction = predictions_t[method][index, 0].cpu().numpy()
                        rows.append(base | {"method": method} | frame_metrics(raw, prediction, gt, common, boundary))
                    if abs(threshold - args.coverage_threshold) < 1e-8 and common.any():
                        positions = np.flatnonzero(common)
                        take = min(256, len(positions))
                        positions = positions[np.linspace(0, len(positions) - 1, take).astype(int)]
                        backbone = batch["backbone"][index]
                        useful = (memory_error.ravel()[positions] + 0.05 < raw_error.ravel()[positions]).astype(float)
                        error_target = np.clip(raw_error.ravel()[positions] / 3.0, 0, 1)
                        sampled[backbone]["selector_score"].extend(score.ravel()[positions].tolist())
                        sampled[backbone]["memory_score"].extend(memory_score.ravel()[positions].tolist())
                        sampled[backbone]["memory_target"].extend(useful.tolist())
                        sampled[backbone]["error_score"].extend(error_score.ravel()[positions].tolist())
                        sampled[backbone]["error_target"].extend(error_target.tolist())
                if sheets < args.contact_sheets:
                    rgb = batch["current_rgb"][index].permute(1, 2, 0).cpu().numpy()
                    contact_sheet(
                        args.output / "contact_sheets" / f"{batch['backbone'][index]}_{batch['sequence'][index]}_{batch['current_frame_id'][index]}.png",
                        rgb,
                        gt,
                        raw,
                        output.disparity[index, 0].cpu().numpy(),
                        score,
                    )
                    sheets += 1

    sequence_rows, aggregate, safety = aggregate_frame_rows(rows)
    calibration = {}
    for backbone, values in sampled.items():
        calibration[backbone] = {
            "selector": calibration_summary(values["selector_score"], values["memory_target"], binary=True),
            "memory_trust": calibration_summary(values["memory_score"], values["memory_target"], binary=True),
            "raw_error_gate": calibration_summary(values["error_score"], values["error_target"], binary=False),
        }
    runtime = {
        "sea_raft_latency_ms_per_pair": float(np.mean(flow_latencies)),
        "model_latency_ms_per_pair": float(np.mean(model_latencies)),
        "total_latency_ms_per_pair": float(np.mean(flow_latencies) + np.mean(model_latencies)),
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    aggregate["runtime"] = runtime
    aggregate["checkpoint"] = str(args.checkpoint)
    aggregate["variant"] = variant
    write_csv(args.output / "frame_metrics.csv", rows)
    write_csv(args.output / "sequence_metrics.csv", sequence_rows)
    save_json(args.output / "aggregate_summary.json", aggregate)
    save_json(args.output / "calibration_metrics.json", calibration)
    save_json(args.output / "safety_summary.json", safety)
    save_json(args.output / "split_manifest.json", state.get("split_manifest", {}))
    save_json(args.output / "config.json", vars(args) | {"output": str(args.output), "checkpoint": str(args.checkpoint)})
    log = args.output / "run.log"
    with log.open("a") as handle:
        print(f"COMMAND {' '.join(sys.argv)}", file=handle)
        print(f"COMPLETE frames={len(rows)} runtime={runtime}", file=handle)
    return 0


def main() -> int:
    args = parse_args()
    if args.mode == "smoke":
        return train(args, smoke=True)
    if args.mode == "train":
        return train(args)
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
