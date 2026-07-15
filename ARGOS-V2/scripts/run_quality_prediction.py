#!/usr/bin/env python3
"""Train and evaluate ARGOS v2 Q0 candidate-error prediction only."""
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
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))
sys.path.insert(0, str(V2_ROOT / "scripts"))

from model_design.data.quality_prediction_dataset import (  # noqa: E402
    CANDIDATE_NAMES,
    FORBIDDEN_Q0_BACKBONES,
    SEEN_BACKBONES,
    QualityPredictionDataset,
    assemble_quality_candidates,
)
from model_design.data.temporal_pair_dataset import (  # noqa: E402
    DEFAULT_VALIDATION_SEQUENCES,
    build_split_manifest,
)
from model_design.external_components.bidavideo import BiDAFlowInferenceAdapter  # noqa: E402
from model_design.losses.quality_losses import (  # noqa: E402
    QualityLossConfig,
    quality_prediction_losses,
)
from model_design.models.learned_ppm_selector import LearnedPPMSelectorRefiner  # noqa: E402
from model_design.models.learned_t1_refiner import LearnedT1Refiner  # noqa: E402
from model_design.models.quality_predictor import ARCHITECTURES, QualityPredictor  # noqa: E402
from run_learned_ppm_selector import build_evidence  # noqa: E402


EVALUATION_THRESHOLDS = (0.05, 0.25, 0.50, 0.90)
T1_CHECKPOINT = V2_ROOT / "results/learned_t1_refiner/ablations/A2/checkpoints/best_validation.pt"
PPM_CHECKPOINT = V2_ROOT / "results/ppmstereo_validation/learned_selector/checkpoints/best_validation.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "train", "evaluate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="q0_5")
    parser.add_argument("--channels", type=int, default=24)
    parser.add_argument(
        "--trainable-uncertainty", action=argparse.BooleanOptionalAction, default=None,
        help="Override whether sigma heads train; default is true only for q0_5.",
    )
    parser.add_argument(
        "--target-mode",
        choices=("absolute", "log", "advantage", "joint", "uncertainty"),
        default="uncertainty",
    )
    parser.add_argument("--patch-size", type=int, choices=(1, 8, 16), default=1)
    parser.add_argument("--ranking-weight", type=float, default=0.25)
    parser.add_argument("--advantage-weight", type=float, default=0.0)
    parser.add_argument("--uncertainty-weight", type=float, default=1.0)
    parser.add_argument("--indifference-margin", type=float, default=0.10)
    parser.add_argument("--hard-negative-weight", type=float, default=0.0)
    parser.add_argument("--coverage-threshold", type=float, default=0.50)
    parser.add_argument("--backbones", nargs="+", default=list(SEEN_BACKBONES))
    parser.add_argument("--train-sequences", nargs="+")
    parser.add_argument(
        "--validation-sequences", nargs="+", default=list(DEFAULT_VALIDATION_SEQUENCES)
    )
    parser.add_argument("--max-train-samples-per-sequence", type=int, default=256)
    parser.add_argument("--max-validation-samples-per-sequence", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8))
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evaluation-sample-pixels", type=int, default=256)
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_json(value):
    if isinstance(value, dict):
        return {str(key): clean_json(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0])
    for row in rows[1:]:
        keys.extend(key for key in row if key not in keys)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def atomic_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def ensure_seen_only(backbones: list[str] | tuple[str, ...]) -> None:
    forbidden = set(backbones) & set(FORBIDDEN_Q0_BACKBONES)
    if forbidden:
        raise ValueError(f"Q0 forbids unseen backbones: {sorted(forbidden)}")
    unknown = set(backbones) - set(SEEN_BACKBONES)
    if unknown:
        raise ValueError(f"Q0 accepts only the predeclared seen backbones: {sorted(unknown)}")


def make_loader(dataset, args: argparse.Namespace, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        generator=torch.Generator().manual_seed(args.seed),
        drop_last=False,
    )


def to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def loss_config(args: argparse.Namespace) -> QualityLossConfig:
    return QualityLossConfig(
        target_mode=args.target_mode,
        patch_size=args.patch_size,
        indifference_margin_px=args.indifference_margin,
        ranking_weight=args.ranking_weight,
        uncertainty_weight=args.uncertainty_weight,
        advantage_weight=args.advantage_weight,
        hard_negative_weight=args.hard_negative_weight,
    )


def prediction_for_objective(output, target_mode: str) -> torch.Tensor:
    if target_mode == "advantage":
        # Raw-relative quality is defined up to a constant; lower is better.
        value = -output.advantage
        return value - value[:, :1]
    return output.mu


def safe_corr(left: np.ndarray, right: np.ndarray, *, rank: bool = False) -> float:
    if left.size < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return float("nan")
    if rank:
        from scipy.stats import spearmanr

        return float(spearmanr(left, right).statistic)
    return float(np.corrcoef(left, right)[0, 1])


def sampled_flat(value: torch.Tensor, valid: torch.Tensor, maximum: int = 32768) -> tuple[np.ndarray, np.ndarray]:
    truth_index = valid.flatten().nonzero(as_tuple=False).flatten()
    if truth_index.numel() > maximum:
        pick = torch.linspace(0, truth_index.numel() - 1, maximum, device=value.device).long()
        truth_index = truth_index[pick]
    return (
        value.flatten()[truth_index].float().cpu().numpy(),
        truth_index.cpu().numpy(),
    )


@torch.no_grad()
def validate(model, adapter, data_loader, device, args) -> dict[str, float]:
    model.eval()
    sums = defaultdict(float)
    predicted_sample: list[np.ndarray] = []
    target_sample: list[np.ndarray] = []
    sigma_sample: list[np.ndarray] = []
    batches = 0
    for cpu in data_loader:
        batch = to_device(cpu, device)
        evidence, flow_ms = build_evidence(adapter, batch)
        candidates = assemble_quality_candidates(
            batch, evidence,
            coverage_threshold=args.coverage_threshold,
            margin=args.indifference_margin,
        )
        output = model(candidates)
        quality = prediction_for_objective(output, args.target_mode)
        valid = candidates.target_valid[:, :, 0]
        target = candidates.target_error[:, :, 0]
        count = int(valid.sum())
        difference = quality - target
        sums["absolute"] += float((difference.abs() * valid).sum())
        sums["square"] += float((difference.square() * valid).sum())
        sums["bias"] += float((difference * valid).sum())
        sums["count"] += count
        sums["flow_ms"] += flow_ms
        target_inf = target.masked_fill(~valid, torch.inf)
        predicted_inf = quality.masked_fill(~valid, torch.inf)
        best = target_inf.argmin(dim=1)
        selected = predicted_inf.argmin(dim=1)
        pixel_valid = valid.any(dim=1)
        selected_error = target.gather(1, selected[:, None]).squeeze(1)
        best_error = target_inf.min(dim=1).values
        sums["top1"] += float((selected[pixel_valid] == best[pixel_valid]).sum())
        sums["pixel_count"] += int(pixel_valid.sum())
        sums["regret"] += float((selected_error[pixel_valid] - best_error[pixel_valid]).sum())
        correct_pairs = 0.0
        pair_count = 0
        for left in range(len(CANDIDATE_NAMES)):
            for right in range(left + 1, len(CANDIDATE_NAMES)):
                true_delta = target[:, right] - target[:, left]
                pred_delta = quality[:, right] - quality[:, left]
                pair_valid = valid[:, left] & valid[:, right] & (
                    true_delta.abs() > args.indifference_margin
                )
                correct_pairs += float(((true_delta.sign() == pred_delta.sign()) & pair_valid).sum())
                pair_count += int(pair_valid.sum())
        sums["correct_pairs"] += correct_pairs
        sums["pair_count"] += pair_count
        sampled, indices = sampled_flat(quality, valid)
        predicted_sample.append(sampled)
        target_sample.append(target.flatten()[torch.as_tensor(indices, device=device)].float().cpu().numpy())
        sigma_sample.append(output.sigma.flatten()[torch.as_tensor(indices, device=device)].float().cpu().numpy())
        batches += 1
    prediction = np.concatenate(predicted_sample) if predicted_sample else np.empty(0)
    target = np.concatenate(target_sample) if target_sample else np.empty(0)
    sigma = np.concatenate(sigma_sample) if sigma_sample else np.empty(0)
    residual = np.abs(prediction - target)
    count = max(sums["count"], 1)
    pixel_count = max(sums["pixel_count"], 1)
    return {
        "mae": sums["absolute"] / count,
        "rmse": math.sqrt(sums["square"] / count),
        "bias": sums["bias"] / count,
        "pearson": safe_corr(prediction, target),
        "spearman": safe_corr(prediction, target, rank=True),
        "uncertainty_error_correlation": safe_corr(sigma, residual, rank=True),
        "top1_accuracy": sums["top1"] / pixel_count,
        "selected_candidate_regret": sums["regret"] / pixel_count,
        "pairwise_ranking_accuracy": sums["correct_pairs"] / max(sums["pair_count"], 1),
        "flow_latency_ms_per_sample": sums["flow_ms"] / max(batches, 1),
        "valid_count": int(sums["count"]),
    }


def model_payload(model, optimizer, epoch, best_score, args, manifest, means) -> dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_score": best_score,
        "model_config": {
            "architecture": args.architecture,
            "channels": args.channels,
            "candidates": len(CANDIDATE_NAMES),
            "predict_uncertainty": args.trainable_uncertainty,
        },
        "loss_config": asdict(loss_config(args)),
        "constant_candidate_means": means,
        "split_manifest": manifest,
    }


def train(args: argparse.Namespace, *, smoke: bool) -> int:
    seed_all(args.seed)
    ensure_seen_only(args.backbones)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = build_split_manifest(
        seed=args.seed,
        coverage_threshold=args.coverage_threshold,
        validation_sequences=args.validation_sequences,
    )
    train_sequences = list(args.train_sequences or manifest["train_sequences"])
    backbones = ["S2M2-S"] if smoke else list(args.backbones)
    if smoke:
        train_sequences = [train_sequences[0]]
        args.max_train_samples_per_sequence = min(args.max_train_samples_per_sequence, 24)
        args.max_validation_samples_per_sequence = min(args.max_validation_samples_per_sequence, 8)
        args.batch_size = min(args.batch_size, 24)
        if args.steps <= 0:
            args.steps = 40
    manifest.update(
        {
            "actual_train_sequences": train_sequences,
            "actual_training_backbones": backbones,
            "candidate_order": list(CANDIDATE_NAMES),
            "candidate_ages": [0, 1, 2, 4, 8],
            "forbidden_q0_backbones": list(FORBIDDEN_Q0_BACKBONES),
            "mode": "smoke" if smoke else "train",
        }
    )
    save_json(args.output / "config.json", vars(args))
    save_json(args.output / "split_manifest.json", manifest)
    train_data = QualityPredictionDataset(
        backbones, train_sequences,
        max_samples_per_sequence=args.max_train_samples_per_sequence,
        random_clip_start=True,
        seed=args.seed,
    )
    validation_data = QualityPredictionDataset(
        backbones, args.validation_sequences,
        max_samples_per_sequence=args.max_validation_samples_per_sequence,
        seed=args.seed,
    )
    train_loader = make_loader(train_data, args, shuffle=True)
    validation_loader = make_loader(validation_data, args, shuffle=False)
    model = QualityPredictor(
        args.architecture, channels=args.channels,
        predict_uncertainty=args.trainable_uncertainty,
    ).to(device)
    adapter = BiDAFlowInferenceAdapter("sea_raft", device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    config = loss_config(args)
    final_path = args.output / "checkpoints/final.pt"
    best_path = args.output / "checkpoints/best_validation.pt"
    start_epoch = 0
    best_score = math.inf
    candidate_sums = torch.zeros(len(CANDIDATE_NAMES), dtype=torch.float64)
    candidate_counts = torch.zeros_like(candidate_sums)
    history_path = args.output / "training_history.csv"
    if not args.resume:
        history_path.unlink(missing_ok=True)
    if args.resume and final_path.exists():
        state = torch.load(final_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1
        best_score = float(state["best_score"])
        means = state.get("constant_candidate_means")
        if means:
            candidate_sums = torch.tensor(means, dtype=torch.float64)
            candidate_counts.fill_(1)
    log = (args.output / "run.log").open("a", buffering=1)
    print(f"COMMAND {' '.join(sys.argv)}", file=log)
    print(f"DATA train={len(train_data)} validation={len(validation_data)}", file=log)
    first_loss = None
    final_loss = None
    total_steps = 0
    train_start = time.perf_counter()
    epoch_limit = args.epochs if not smoke else max(args.epochs, args.steps)
    for epoch in range(start_epoch, epoch_limit):
        model.train()
        totals = defaultdict(float)
        batches = 0
        stop = False
        for cpu in train_loader:
            batch = to_device(cpu, device)
            evidence, flow_ms = build_evidence(adapter, batch)
            candidates = assemble_quality_candidates(
                batch, evidence,
                coverage_threshold=args.coverage_threshold,
                margin=args.indifference_margin,
            )
            if epoch == 0 and start_epoch == 0:
                valid = candidates.target_valid[:, :, 0]
                target = candidates.target_error[:, :, 0]
                candidate_sums += (target * valid).sum(dim=(0, 2, 3)).double().cpu()
                candidate_counts += valid.sum(dim=(0, 2, 3)).double().cpu()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model(candidates)
                losses = quality_prediction_losses(output, candidates, config)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
            scaler.step(optimizer)
            scaler.update()
            batch_loss = float(losses["total"].detach())
            if first_loss is None:
                first_loss = batch_loss
            final_loss = batch_loss
            for key, value in losses.items():
                totals[key] += float(value.detach())
            totals["gradient_norm"] += grad_norm
            totals["flow_ms"] += flow_ms
            batches += 1
            total_steps += 1
            if args.steps > 0 and total_steps >= args.steps:
                stop = True
                break
        # The smoke repeats one tiny clip. Avoid spending almost all smoke time
        # re-running SEA-RAFT on validation after every single update.
        validation = (
            validate(model, adapter, validation_loader, device, args)
            if (not smoke or stop)
            else {
                "mae": math.nan, "rmse": math.nan, "bias": math.nan,
                "pearson": math.nan, "spearman": math.nan,
                "uncertainty_error_correlation": math.nan,
                "top1_accuracy": math.nan,
                "selected_candidate_regret": math.nan,
                "pairwise_ranking_accuracy": math.nan,
                "flow_latency_ms_per_sample": math.nan,
                "valid_count": 0,
            }
        )
        means = (candidate_sums / candidate_counts.clamp_min(1)).tolist()
        selection_score = (
            validation["selected_candidate_regret"]
            if args.target_mode == "advantage"
            else validation["mae"]
        )
        row = {
            "epoch": epoch,
            "steps": total_steps,
            **{f"train_{key}": value / max(batches, 1) for key, value in totals.items()},
            **{f"validation_{key}": value for key, value in validation.items()},
        }
        append_csv(history_path, row)
        payload = model_payload(model, optimizer, epoch, best_score, args, manifest, means)
        atomic_checkpoint(final_path, payload)
        if math.isfinite(selection_score) and selection_score < best_score:
            best_score = selection_score
            payload["best_score"] = best_score
            atomic_checkpoint(best_path, payload)
        print(json.dumps(clean_json(row)), file=log)
        print(json.dumps(clean_json(row)))
        if stop:
            break
    elapsed = time.perf_counter() - train_start
    if smoke:
        train_metrics = validate(model, adapter, train_loader, device, args)
        smoke_result = {
            "first_loss": first_loss,
            "final_loss": final_loss,
            "loss_reduction_fraction": (
                (first_loss - final_loss) / max(abs(first_loss), 1e-8)
                if first_loss is not None and final_loss is not None else None
            ),
            "nonzero_gradient": bool(row.get("train_gradient_norm", 0) > 0),
            "finite": bool(final_loss is not None and math.isfinite(final_loss)),
            "train_metrics": train_metrics,
            "steps": total_steps,
            "elapsed_seconds": elapsed,
        }
        smoke_result["passed"] = bool(
            smoke_result["finite"]
            and smoke_result["nonzero_gradient"]
            and smoke_result["loss_reduction_fraction"] > 0.20
            and train_metrics["pairwise_ranking_accuracy"] > 0.50
        )
        save_json(args.output / "smoke_result.json", smoke_result)
        print(json.dumps(clean_json(smoke_result), indent=2))
        if not smoke_result["passed"]:
            raise RuntimeError("Q0 overfit smoke did not satisfy its predeclared checks")
    return 0


class EvaluationAccumulator:
    """Exact first moments plus deterministic pixel samples for richer metrics."""

    def __init__(self, sample_pixels: int) -> None:
        self.sample_pixels = sample_pixels
        self.groups: dict[tuple, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.samples: dict[str, list[np.ndarray]] = defaultdict(list)

    def add_error_group(self, key, predicted, target, valid) -> None:
        difference = predicted - target
        absolute = difference.abs()
        huber = torch.where(absolute < 0.25, 0.5 * difference.square() / 0.25, absolute - 0.125)
        count = int(valid.sum())
        group = self.groups[key]
        group["count"] += count
        group["absolute"] += float((difference.abs() * valid).sum())
        group["square"] += float((difference.square() * valid).sum())
        group["bias"] += float((difference * valid).sum())
        group["huber"] += float((huber * valid).sum())
        group["target_sum"] += float((target * valid).sum())
        group["target_square"] += float((target.square() * valid).sum())

    @staticmethod
    def finalize_group(key, value) -> dict:
        count = max(value["count"], 1)
        variance = max(value["target_square"] / count - (value["target_sum"] / count) ** 2, 1e-12)
        mse = value["square"] / count
        return {
            "coverage_threshold": key[0],
            "backbone": key[1],
            "sequence": key[2],
            "candidate": key[3],
            "valid_count": int(value["count"]),
            "mae": value["absolute"] / count,
            "rmse": math.sqrt(mse),
            "mean_bias": value["bias"] / count,
            "huber_loss_beta_0.25": value["huber"] / count,
            "explained_variance_approx": 1.0 - mse / variance,
        }


def deterministic_pixel_indices(valid: torch.Tensor, maximum: int) -> torch.Tensor:
    index = valid.flatten().nonzero(as_tuple=False).flatten()
    if index.numel() > maximum:
        pick = torch.linspace(0, index.numel() - 1, maximum, device=index.device).long()
        index = index[pick]
    return index


def load_reference_models(device: torch.device):
    t1 = LearnedT1Refiner("A2").to(device).eval()
    t1_state = torch.load(T1_CHECKPOINT, map_location=device, weights_only=False)
    t1.load_state_dict(t1_state["model"])
    ppm = LearnedPPMSelectorRefiner().to(device).eval()
    ppm_state = torch.load(PPM_CHECKPOINT, map_location=device, weights_only=False)
    ppm.load_state_dict(ppm_state["model"])
    for model in (t1, ppm):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return t1, ppm


def metric_rows_from_samples(method: str, prediction: np.ndarray, target: np.ndarray, sigma=None) -> dict:
    difference = prediction - target
    row = {
        "method": method,
        "sample_count": int(target.size),
        "mae": float(np.mean(np.abs(difference))) if target.size else float("nan"),
        "rmse": float(np.sqrt(np.mean(difference ** 2))) if target.size else float("nan"),
        "pearson": safe_corr(prediction, target),
        "spearman": safe_corr(prediction, target, rank=True),
        "mean_bias": float(np.mean(difference)) if target.size else float("nan"),
    }
    if sigma is not None and target.size:
        safe_sigma = np.clip(sigma, 1e-3, 10)
        row.update(
            {
                "laplace_nll": float(np.mean(np.abs(difference) / safe_sigma + np.log(2 * safe_sigma))),
                "uncertainty_error_spearman": safe_corr(sigma, np.abs(difference), rank=True),
                "sharpness_mean_sigma": float(np.mean(sigma)),
                "interval_1sigma_coverage": float(np.mean(np.abs(difference) <= sigma)),
                "interval_2sigma_coverage": float(np.mean(np.abs(difference) <= 2 * sigma)),
            }
        )
    return row


def ranking_diagnostics(
    method: str,
    predicted: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    margins: tuple[float, ...],
) -> tuple[list[dict], list[dict]]:
    prediction = np.where(valid, predicted, np.inf)
    truth = np.where(valid, target, np.inf)
    selected = prediction.argmin(1)
    best = truth.argmin(1)
    selected_error = target[np.arange(len(target)), selected]
    best_error = truth.min(1)
    rows = []
    for margin in margins:
        memory_best = truth[:, 1:].min(1)
        memory_useful = memory_best + margin < truth[:, 0]
        predicted_memory_advantage = prediction[:, 0] - prediction[:, 1:].min(1)
        predicted_memory_advantage = np.nan_to_num(
            predicted_memory_advantage, nan=-1e6, neginf=-1e6, posinf=1e6
        )
        action = selected != 0
        true_positive = np.sum(action & memory_useful)
        precision = true_positive / max(np.sum(action), 1)
        recall = true_positive / max(np.sum(memory_useful), 1)
        from sklearn.metrics import average_precision_score, roc_auc_score

        auroc = (
            float(roc_auc_score(memory_useful, predicted_memory_advantage))
            if np.unique(memory_useful).size == 2 else float("nan")
        )
        average_precision = (
            float(average_precision_score(memory_useful, predicted_memory_advantage))
            if memory_useful.any() else float("nan")
        )
        rows.append(
            {
                "method": method,
                "indifference_margin_px": margin,
                "sample_count": len(target),
                "top1_accuracy": float(np.mean(selected == best)),
                "top2_recall": float(np.mean([best[i] in np.argsort(prediction[i])[:2] for i in range(len(best))])),
                "pairwise_ranking_accuracy": pairwise_accuracy(predicted, target, valid, margin),
                "selected_candidate_regret": float(np.mean(selected_error - best_error)),
                "normalized_candidate_regret": float(np.mean(selected_error - best_error) / max(np.mean(truth[:, 0] - best_error), 1e-8)),
                "raw_null_accuracy": float(np.mean((selected == 0) == (~memory_useful))),
                "memory_action_precision": float(precision),
                "memory_action_recall": float(recall),
                "raw_vs_best_memory_auroc": auroc,
                "raw_vs_best_memory_average_precision": average_precision,
            }
        )
    confusion = []
    matrix = np.zeros((len(CANDIDATE_NAMES), len(CANDIDATE_NAMES)), dtype=np.int64)
    for true, pred in zip(best, selected):
        matrix[true, pred] += 1
    for true in range(len(CANDIDATE_NAMES)):
        for pred in range(len(CANDIDATE_NAMES)):
            confusion.append(
                {
                    "method": method,
                    "true_candidate": CANDIDATE_NAMES[true],
                    "predicted_candidate": CANDIDATE_NAMES[pred],
                    "count": int(matrix[true, pred]),
                }
            )
    return rows, confusion


def pairwise_accuracy(predicted, target, valid, margin) -> float:
    correct = total = 0
    for left in range(len(CANDIDATE_NAMES)):
        for right in range(left + 1, len(CANDIDATE_NAMES)):
            delta = target[:, right] - target[:, left]
            pair_valid = valid[:, left] & valid[:, right] & (np.abs(delta) > margin)
            correct += np.sum(np.sign(delta[pair_valid]) == np.sign((predicted[:, right] - predicted[:, left])[pair_valid]))
            total += np.sum(pair_valid)
    return float(correct / max(total, 1))


def evaluate(args: argparse.Namespace) -> int:
    seed_all(args.seed)
    ensure_seen_only(args.backbones)
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for evaluate")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config = state["model_config"]
    # The evaluation manifest must describe the loaded frozen checkpoint, not
    # parser defaults that are irrelevant to inference.
    args.architecture = model_config["architecture"]
    args.channels = model_config["channels"]
    args.trainable_uncertainty = model_config.get("predict_uncertainty")
    frozen_loss_config = state["loss_config"]
    args.patch_size = frozen_loss_config["patch_size"]
    args.ranking_weight = frozen_loss_config["ranking_weight"]
    args.advantage_weight = frozen_loss_config["advantage_weight"]
    args.uncertainty_weight = frozen_loss_config["uncertainty_weight"]
    args.indifference_margin = frozen_loss_config["indifference_margin_px"]
    args.hard_negative_weight = frozen_loss_config.get("hard_negative_weight", 0.0)
    model = QualityPredictor(**model_config).to(device).eval()
    model.load_state_dict(state["model"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    args.target_mode = frozen_loss_config["target_mode"]
    constant_means = np.asarray(state["constant_candidate_means"], dtype=np.float32)
    t1, ppm = load_reference_models(device)
    adapter = BiDAFlowInferenceAdapter("sea_raft", device=device)
    data = QualityPredictionDataset(
        args.backbones, args.validation_sequences,
        max_samples_per_sequence=args.max_validation_samples_per_sequence,
        seed=args.seed,
    )
    data_loader = make_loader(data, args, shuffle=False)
    manifest = build_split_manifest(
        seed=args.seed,
        coverage_threshold=args.coverage_threshold,
        validation_sequences=args.validation_sequences,
    )
    manifest.update(
        {
            "evaluated_backbones": list(args.backbones),
            "candidate_order": list(CANDIDATE_NAMES),
            "unseen_backbones_touched": [],
            "diagnostic_argmin_only": True,
        }
    )
    save_json(args.output / "config.json", vars(args))
    save_json(args.output / "split_manifest.json", manifest)
    log = (args.output / "run.log").open("a", buffering=1)
    print(f"COMMAND {' '.join(sys.argv)}", file=log)
    accumulator = EvaluationAccumulator(args.evaluation_sample_pixels)
    sampled_target: list[np.ndarray] = []
    sampled_valid: list[np.ndarray] = []
    method_predictions: dict[str, list[np.ndarray]] = defaultdict(list)
    q0_sigma: list[np.ndarray] = []
    sampled_failure: dict[str, list[np.ndarray]] = defaultdict(list)
    sampled_backbone: list[np.ndarray] = []
    sampled_sequence: list[np.ndarray] = []
    q0_raw_gate: list[np.ndarray] = []
    t1_memory_gate: list[np.ndarray] = []
    t1_error_gate: list[np.ndarray] = []
    total_flow_ms = total_model_ms = peak_memory = batches = total_samples = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for cpu in data_loader:
            batch = to_device(cpu, device)
            evidence, flow_ms = build_evidence(adapter, batch)
            # Primary-threshold targets feed sampled ranking diagnostics.
            candidates = assemble_quality_candidates(
                batch, evidence,
                coverage_threshold=args.coverage_threshold,
                margin=args.indifference_margin,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            tick = time.perf_counter()
            output = model(candidates)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            total_model_ms += (time.perf_counter() - tick) * 1000
            total_flow_ms += flow_ms * batch["raw"].shape[0]
            batches += 1
            total_samples += batch["raw"].shape[0]
            predicted_quality = prediction_for_objective(output, args.target_mode)
            ppm_output = ppm(batch["raw"], batch["raw_valid"], evidence, batch["ages"][0])
            ppm_quality = torch.cat(
                (
                    torch.zeros_like(ppm_output.candidate_logits[:, 0]),
                    -ppm_output.candidate_logits[:, :, 0],
                ), dim=1,
            )
            t1_evidence = {
                name: (value[:, 0] if value.ndim == 5 else value)
                for name, value in evidence.items()
            }
            t1_evidence["current_valid"] = batch["raw_valid"]
            t1_output = t1(batch["raw"], t1_evidence)
            raw = candidates.disparity[:, 0, 0]
            median = torch.nan_to_num(candidates.consensus_median[:, 0], nan=0.0)
            cmc_quality = torch.cat(
                (
                    (raw - median).abs()[:, None],
                    (candidates.disparity[:, 1:, 0] - median[:, None]).abs(),
                ), dim=1,
            )
            heuristic = torch.cat(
                (
                    torch.zeros_like(raw)[:, None],
                    -(
                        candidates.forward_backward_confidence[:, 1:, 0]
                        * torch.exp(-candidates.photometric_residual[:, 1:, 0])
                    ),
                ), dim=1,
            )
            constant = torch.as_tensor(constant_means, device=device).view(1, -1, 1, 1).expand_as(predicted_quality)
            batch_methods = {
                "constant_mean": constant,
                "heuristic_fb_photo": heuristic,
                "ppm_selector": ppm_quality,
                "cmc_spread": cmc_quality,
                "q0_quality_predictor": predicted_quality,
            }
            # Exact error metrics for Q0 at every required coverage threshold.
            for threshold in EVALUATION_THRESHOLDS:
                threshold_candidates = assemble_quality_candidates(
                    batch, evidence, coverage_threshold=threshold,
                    margin=args.indifference_margin,
                )
                target = threshold_candidates.target_error[:, :, 0]
                valid = threshold_candidates.target_valid[:, :, 0]
                # Reduce whole groups before transferring scalars to the CPU.
                # The previous per-frame reduction was exact but introduced
                # hundreds of thousands of unnecessary CUDA synchronizations.
                groups = {("ALL", "ALL")}
                groups.update((backbone, "ALL") for backbone in set(batch["backbone"]))
                groups.update(("ALL", sequence) for sequence in set(batch["sequence"]))
                groups.update(zip(batch["backbone"], batch["sequence"]))
                for group_backbone, group_sequence in groups:
                    selected_rows = torch.tensor(
                        [
                            (group_backbone == "ALL" or backbone == group_backbone)
                            and (group_sequence == "ALL" or sequence == group_sequence)
                            for backbone, sequence in zip(batch["backbone"], batch["sequence"])
                        ],
                        device=device, dtype=torch.bool,
                    )
                    spatial_rows = selected_rows[:, None, None]
                    for candidate_index, candidate_name in enumerate(CANDIDATE_NAMES):
                        mask = valid[:, candidate_index] & spatial_rows
                        accumulator.add_error_group(
                            (threshold, group_backbone, group_sequence, candidate_name),
                            predicted_quality[:, candidate_index],
                            target[:, candidate_index], mask,
                        )
            # Fixed, deterministic per-frame sample for ranking/correlation.
            target = candidates.target_error[:, :, 0]
            valid = candidates.target_valid[:, :, 0]
            base_valid = valid[:, 0]
            for local_index in range(target.shape[0]):
                index = deterministic_pixel_indices(
                    base_valid[local_index], args.evaluation_sample_pixels
                )
                if not index.numel():
                    continue
                target_flat = target[local_index].reshape(len(CANDIDATE_NAMES), -1)[:, index].T
                valid_flat = valid[local_index].reshape(len(CANDIDATE_NAMES), -1)[:, index].T
                sampled_target.append(target_flat.float().cpu().numpy())
                sampled_valid.append(valid_flat.cpu().numpy())
                sampled_backbone.append(np.repeat(batch["backbone"][local_index], len(index)))
                sampled_sequence.append(np.repeat(batch["sequence"][local_index], len(index)))
                for name, value in batch_methods.items():
                    method_predictions[name].append(
                        value[local_index].reshape(len(CANDIDATE_NAMES), -1)[:, index].T.float().cpu().numpy()
                    )
                q0_sigma.append(
                    output.sigma[local_index].reshape(len(CANDIDATE_NAMES), -1)[:, index].T.float().cpu().numpy()
                )
                q0_raw_gate.append(
                    t1_output.g_error[local_index, 0].flatten()[index].float().cpu().numpy()
                )
                t1_memory_gate.append(
                    t1_output.c_memory[local_index, 0].flatten()[index].float().cpu().numpy()
                )
                t1_error_gate.append(
                    t1_output.g_error[local_index, 0].flatten()[index].float().cpu().numpy()
                )
                for failure_name, failure_map in candidates.failure_masks.items():
                    sampled_failure[failure_name].append(
                        failure_map[local_index, 0].flatten()[index].cpu().numpy()
                    )
    peak_memory = (
        torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
    )
    target = np.concatenate(sampled_target)
    valid = np.concatenate(sampled_valid).astype(bool)
    sigma = np.concatenate(q0_sigma)
    sample_backbone = np.concatenate(sampled_backbone)
    sample_sequence = np.concatenate(sampled_sequence)
    predictions = {name: np.concatenate(parts) for name, parts in method_predictions.items()}
    error_rows = [
        accumulator.finalize_group(key, value)
        for key, value in sorted(accumulator.groups.items(), key=lambda item: str(item[0]))
    ]
    # Add deterministic sampled correlations at every primary-threshold
    # aggregate/backbone/sequence level. Exact MAE/RMSE remain pixel-weighted.
    for row in error_rows:
        if row["coverage_threshold"] != args.coverage_threshold:
            continue
        candidate_index = CANDIDATE_NAMES.index(row["candidate"])
        mask = valid[:, candidate_index].copy()
        if row["backbone"] != "ALL":
            mask &= sample_backbone == row["backbone"]
        if row["sequence"] != "ALL":
            mask &= sample_sequence == row["sequence"]
        prediction = predictions["q0_quality_predictor"][mask, candidate_index]
        truth = target[mask, candidate_index]
        row["pearson"] = safe_corr(prediction, truth)
        row["spearman"] = safe_corr(prediction, truth, rank=True)
    write_csv(args.output / "error_prediction_metrics.csv", error_rows)
    ranking_rows: list[dict] = []
    confusion_rows: list[dict] = []
    margins = (0.05, 0.10, 0.25, 0.50)
    for name, prediction in predictions.items():
        rows, confusion = ranking_diagnostics(name, prediction, target, valid, margins)
        ranking_rows.extend(rows)
        confusion_rows.extend(confusion)
    write_csv(args.output / "ranking_metrics.csv", ranking_rows)
    write_csv(args.output / "candidate_confusion_matrix.csv", confusion_rows)
    flat_valid = valid.flatten()
    uncertainty_rows = []
    reliability_rows = []
    quantile_rows = []
    for candidate_index, candidate_name in enumerate(CANDIDATE_NAMES):
        mask = valid[:, candidate_index]
        candidate_prediction = predictions["q0_quality_predictor"][mask, candidate_index]
        candidate_target = target[mask, candidate_index]
        candidate_sigma = sigma[mask, candidate_index]
        uncertainty_row = {
            "candidate": candidate_name, **metric_rows_from_samples(
                "q0_quality_predictor",
                candidate_prediction, candidate_target, candidate_sigma,
            )}
        uncertainty_row["interval_calibration_error"] = 0.5 * (
            abs(uncertainty_row["interval_1sigma_coverage"] - (1 - math.exp(-1)))
            + abs(uncertainty_row["interval_2sigma_coverage"] - (1 - math.exp(-2)))
        )
        uncertainty_rows.append(uncertainty_row)
        residual = np.abs(candidate_prediction - candidate_target)
        sigma_edges = np.quantile(candidate_sigma, np.linspace(0, 1, 11))
        target_edges = np.quantile(candidate_target, np.linspace(0, 1, 11))
        for bin_index in range(10):
            upper_inclusive = bin_index == 9
            sigma_bin = (candidate_sigma >= sigma_edges[bin_index]) & (
                candidate_sigma <= sigma_edges[bin_index + 1]
                if upper_inclusive else candidate_sigma < sigma_edges[bin_index + 1]
            )
            target_bin = (candidate_target >= target_edges[bin_index]) & (
                candidate_target <= target_edges[bin_index + 1]
                if upper_inclusive else candidate_target < target_edges[bin_index + 1]
            )
            reliability_rows.append({
                "candidate": candidate_name, "sigma_decile": bin_index + 1,
                "count": int(sigma_bin.sum()),
                "mean_sigma": float(candidate_sigma[sigma_bin].mean()) if sigma_bin.any() else None,
                "mean_absolute_residual": float(residual[sigma_bin].mean()) if sigma_bin.any() else None,
            })
            quantile_rows.append({
                "candidate": candidate_name, "true_error_decile": bin_index + 1,
                "count": int(target_bin.sum()),
                "mean_true_error": float(candidate_target[target_bin].mean()) if target_bin.any() else None,
                "prediction_mae": float(residual[target_bin].mean()) if target_bin.any() else None,
                "mean_predicted_error": float(candidate_prediction[target_bin].mean()) if target_bin.any() else None,
            })
    write_csv(args.output / "uncertainty_metrics.csv", uncertainty_rows)
    write_csv(args.output / "uncertainty_reliability.csv", reliability_rows)
    write_csv(args.output / "error_quantile_metrics.csv", quantile_rows)
    # Risk-coverage is diagnostic argmin only, never a deployed selector.
    q0_prediction = np.where(valid, predictions["q0_quality_predictor"], np.inf)
    selected = q0_prediction.argmin(1)
    best = np.where(valid, target, np.inf).argmin(1)
    selected_error = target[np.arange(len(target)), selected]
    best_error = np.where(valid, target, np.inf).min(1)
    selected_sigma = sigma[np.arange(len(sigma)), selected]
    predicted_advantage = q0_prediction[:, 0] - q0_prediction.min(1)
    risk_rows = []
    for ordering, confidence in (
        ("low_uncertainty", -selected_sigma),
        ("large_predicted_advantage", predicted_advantage),
    ):
        order = np.argsort(-confidence)
        for coverage in (0.01, 0.05, 0.10, 0.20, 0.50, 1.00):
            take = order[: max(1, round(len(order) * coverage))]
            action = selected[take] != 0
            useful = selected_error[take] + args.indifference_margin < target[take, 0]
            risk_rows.append(
                {
                    "ordering": ordering,
                    "coverage": coverage,
                    "sample_count": len(take),
                    "selected_candidate_regret": float(np.mean(selected_error[take] - best_error[take])),
                    "candidate_precision": float(np.sum(action & useful) / max(np.sum(action), 1)),
                    "true_gain_available": float(np.mean(target[take, 0] - best_error[take])),
                    "clean_pixel_prevalence": float(np.mean(target[take, 0] <= 0.50)),
                    "top1_accuracy": float(np.mean(selected[take] == best[take])),
                }
            )
    write_csv(args.output / "risk_coverage.csv", risk_rows)
    per_backbone = [
        row for row in error_rows
        if row["coverage_threshold"] == args.coverage_threshold
        and row["sequence"] == "ALL" and row["backbone"] != "ALL"
    ]
    per_sequence = [
        row for row in error_rows
        if row["coverage_threshold"] == args.coverage_threshold
        and row["backbone"] == "ALL" and row["sequence"] != "ALL"
    ]
    write_csv(args.output / "per_backbone.csv", per_backbone)
    write_csv(args.output / "per_sequence.csv", per_sequence)
    # Direct comparison to the existing t-1 heads on their actual Q0 signals.
    raw_target = target[:, 0]
    best_memory = np.where(valid[:, 1:], target[:, 1:], np.inf).min(1)
    memory_label = best_memory + args.indifference_margin < raw_target
    raw_gate = np.concatenate(q0_raw_gate)
    memory_gate = np.concatenate(t1_memory_gate)
    from sklearn.metrics import average_precision_score, roc_auc_score

    selector_comparison = {
        "learned_t1_raw_error_gate": {
            "spearman_vs_raw_error": safe_corr(raw_gate, raw_target, rank=True),
            "pearson_vs_raw_error": safe_corr(raw_gate, raw_target),
        },
        "learned_t1_memory_gate": {
            "auroc_best_memory_beats_raw": float(roc_auc_score(memory_label, memory_gate)),
            "average_precision_best_memory_beats_raw": float(average_precision_score(memory_label, memory_gate)),
        },
    }
    failure_analysis = {}
    ppm_selected_all = np.where(valid, predictions["ppm_selector"], np.inf).argmin(1)
    for failure_name, parts in sampled_failure.items():
        mask = np.concatenate(parts).astype(bool)
        failure_analysis[failure_name] = {
            "sample_count": int(mask.sum()),
            "prevalence": float(mask.mean()),
            "q0_top1_accuracy": float(np.mean(selected[mask] == best[mask])) if mask.any() else None,
            "q0_regret": float(np.mean(selected_error[mask] - best_error[mask])) if mask.any() else None,
            "ppm_regret": (
                float(np.mean(
                    target[mask][np.arange(int(mask.sum())), ppm_selected_all[mask]]
                    - best_error[mask]
                )) if mask.any() else None
            ),
        }
    primary_ranking = {
        row["method"]: row for row in ranking_rows
        if row["indifference_margin_px"] == args.indifference_margin
    }
    aggregate = {
        "stage": "Q0 quality prediction only",
        "cache_grid": True,
        "refinement_or_selection_implemented": False,
        "diagnostic_argmin_only": True,
        "evaluated_backbones": list(args.backbones),
        "unseen_backbones_touched": [],
        "primary_coverage_threshold": args.coverage_threshold,
        "q0_ranking": primary_ranking.get("q0_quality_predictor"),
        "baseline_ranking": {name: row for name, row in primary_ranking.items() if name != "q0_quality_predictor"},
        "selector_comparison": selector_comparison,
        "failure_analysis": failure_analysis,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    save_json(args.output / "aggregate_summary.json", aggregate)
    runtime = {
        "samples": total_samples,
        "sea_raft_latency_ms_per_sample": total_flow_ms / max(total_samples, 1),
        "quality_model_latency_ms_per_batch": total_model_ms / max(batches, 1),
        "quality_model_latency_ms_per_sample": total_model_ms / max(total_samples, 1),
        "peak_gpu_memory_gib_total_pipeline": peak_memory,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    save_json(args.output / "runtime_summary.json", runtime)
    print(json.dumps(clean_json(aggregate), indent=2))
    return 0


def main() -> int:
    args = parse_args()
    if args.mode == "smoke":
        return train(args, smoke=True)
    if args.mode == "train":
        return train(args, smoke=False)
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
