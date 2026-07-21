#!/usr/bin/env python3
"""Fit, freeze and evaluate the minimal ARGOS v2 feature-support guard."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
ARGOS_ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ARGOS_ROOT)]

from model_design.models.support_guard import (  # noqa: E402
    METHODS, SupportGuard, SupportProvenance, fit_support_reference,
    guarded_output, quantile_threshold, save_reference,
)
from run_calibration_shift_audit import (  # noqa: E402
    HELDOUT, SEEN_BACKBONES, as_tensor, inspect_step, iter_d4d, iter_scared,
    iter_serv, iter_stereomis,
)
from run_ood_generalization import (  # noqa: E402
    FrozenARGOS, H, W, geometry_metrics, read_rgb, resize_disparity,
    resize_gt, rgb_tensor, s2m2_model, verify_frozen,
)

OUT_DEFAULT = ROOT / "results/support_guard_validation"
SPLIT_PATH = ROOT / "results/raw_error_abstention/full/split_manifest.json"
QUANTILES = (0.90, 0.95, 0.975, 0.99, 0.995)
FEATURE_NAMES = tuple(f"penultimate_{index:02d}" for index in range(24))


def save_json(path: Path, value: Any) -> None:
    def clean(item: Any) -> Any:
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(v) for v in item]
        return item
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dataframe_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def parse_frame(record: dict[str, Any], dataset: str) -> tuple[str, str, str]:
    parts = str(record["frame"]).split(":")
    if dataset in {"SCARED-C-fit", "SCARED-C-calibration", "SCARED-C-test", "Fast-FoundationStereo", "CREStereo"}:
        return parts[0], parts[1], parts[2]
    if dataset == "SERV-CT":
        return "S2M2-S", parts[1], parts[-1]
    if dataset == "D4D":
        return "S2M2-S", parts[1], parts[-1]
    if dataset == "StereoMIS":
        return "S2M2-S", parts[1], parts[-1]
    return "S2M2-S", parts[1] if len(parts) > 1 else dataset, parts[-1]


@dataclass
class ForwardFrame:
    dataset: str
    backbone: str
    sequence: str
    frame_id: str
    raw: np.ndarray
    gt: np.ndarray | None
    gt_valid: np.ndarray | None
    pipeline_support: np.ndarray
    base_update: np.ndarray
    base_authorization: np.ndarray
    scores: dict[str, np.ndarray]
    score_latency_ms: dict[str, float]
    aligned_raw: np.ndarray
    current_rgb: np.ndarray


def forward_record(
    dataset: str,
    record: dict[str, Any],
    pipe: FrozenARGOS,
    guard: SupportGuard | None,
    capture: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[ForwardFrame, torch.Tensor]:
    raw = as_tensor(record["raw"], device)
    past = as_tensor(record["past"], device)
    raw_valid = as_tensor(record["raw_valid"], device, True)
    past_valid = as_tensor(record["past_valid"], device, True)
    current_rgb = rgb_tensor(record["current_rgb"], device)
    past_rgb = rgb_tensor(record["past_rgb"], device)
    output = inspect_step(pipe, raw, raw_valid, current_rgb, past, past_valid, past_rgb, capture)
    feature = output["penultimate"]
    scores: dict[str, np.ndarray] = {}
    latency: dict[str, float] = {}
    if guard is not None:
        for method in METHODS:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            score = guard.score(feature, method)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latency[method] = (time.perf_counter() - start) * 1000
            scores[method] = score[0, 0].float().cpu().numpy()
    backbone, sequence, frame_id = parse_frame(record, dataset)
    n = lambda tensor: tensor[0, 0].float().cpu().numpy()
    frame = ForwardFrame(
        dataset=dataset, backbone=backbone, sequence=sequence, frame_id=frame_id,
        raw=np.asarray(record["raw"], dtype=np.float32),
        gt=None if record["gt"] is None else np.asarray(record["gt"], dtype=np.float32),
        gt_valid=None if record["gt_valid"] is None else np.asarray(record["gt_valid"], dtype=bool),
        pipeline_support=(n(output["warp_support"]) > .5) & (n(output["aligned_valid"]) > .5) & np.asarray(record["raw_valid"], dtype=bool),
        base_update=n(output["update_signed"]), base_authorization=n(output["authorization"]) > .5,
        scores=scores, score_latency_ms=latency, aligned_raw=n(output["aligned_disparity"]),
        current_rgb=np.asarray(record["current_rgb"]),
    )
    return frame, feature.detach()


def sample_indices(mask: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    indices = np.flatnonzero(mask.ravel())
    if len(indices) > count:
        indices = rng.choice(indices, size=count, replace=False)
    return indices


def fit_reference_support(
    pipe: FrozenARGOS,
    capture: dict[str, torch.Tensor],
    split: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[SupportGuard, dict[str, Any]]:
    rng = np.random.default_rng(args.seed)
    grouped: dict[tuple[str, str, str], list[np.ndarray]] = defaultdict(list)
    records = iter_scared(SEEN_BACKBONES, split["train_sequences"], args.max_train_pairs)
    for record in records:
        frame, feature = forward_record("SCARED-C-fit", record, pipe, None, capture, args.device_obj)
        valid = frame.pipeline_support & np.asarray(record["gt_valid"], dtype=bool)
        error = np.abs(frame.raw - np.asarray(record["gt"], dtype=np.float32))
        flat_feature = feature[0].permute(1, 2, 0).float().cpu().numpy().reshape(-1, feature.shape[1])
        for status, mask in (("clean", valid & (error <= .5)), ("error", valid & (error > .5))):
            index = sample_indices(mask, args.fit_pixels_per_status_frame, rng)
            if len(index):
                grouped[(frame.backbone, frame.sequence, status)].append(flat_feature[index])
    expected = {(backbone, sequence, status) for backbone in SEEN_BACKBONES
                for sequence in split["train_sequences"] for status in ("clean", "error")}
    missing = sorted(expected.difference(grouped))
    if missing or not grouped:
        raise RuntimeError(f"missing balanced SCARED-C support groups: {missing}")
    merged = {key: np.concatenate(parts) for key, parts in grouped.items()}
    take = min(args.fit_pixels_per_group, min(len(value) for value in merged.values()))
    balanced, counts = [], {}
    for key in sorted(merged):
        value = merged[key]
        index = rng.choice(len(value), size=take, replace=False)
        balanced.append(value[index])
        counts["|".join(key)] = int(take)
    features = np.concatenate(balanced).astype(np.float32)
    provenance = SupportProvenance(
        dataset="SCARED-C", split="training", backbones=tuple(SEEN_BACKBONES),
        sequences=tuple(split["train_sequences"]), seed=args.seed,
    )
    reference = fit_support_reference(
        features, feature_names=FEATURE_NAMES, provenance=provenance,
        bank_size=args.bank_size, knn_k=args.knn_k,
    )
    save_reference(args.output / "support_reference.npz", reference)
    manifest = {
        "dataset": "SCARED-C", "split": "training", "backbones": SEEN_BACKBONES,
        "sequences": split["train_sequences"], "balanced_group_definition": "backbone|sequence|clean_or_error",
        "samples_per_group": take, "group_counts": counts, "total_fit_vectors": len(features),
        "feature_names": list(FEATURE_NAMES), "feature_dim": len(FEATURE_NAMES),
        "bank_size": len(reference.reference_bank), "knn_k": reference.knn_k,
        "shrinkage": reference.shrinkage, "memory_bytes": reference.memory_bytes,
        "reference_sha256": sha256(args.output / "support_reference.npz"),
        "forbidden_sources": ["Fast-FoundationStereo", "CREStereo", "SERV-CT", "D4D", "StereoMIS"],
    }
    save_json(args.output / "support_reference_manifest.json", manifest)
    return SupportGuard(reference).to(args.device_obj).eval(), manifest


def collect_calibration(
    pipe: FrozenARGOS, guard: SupportGuard, capture: dict[str, torch.Tensor],
    split: dict[str, Any], args: argparse.Namespace,
) -> tuple[list[ForwardFrame], dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(args.seed + 1)
    frames, score_samples = [], defaultdict(list)
    feature_samples = []
    for record in iter_scared(SEEN_BACKBONES, split["calibration_sequences"], args.max_calibration_pairs):
        frame, feature = forward_record("SCARED-C-calibration", record, pipe, guard, capture, args.device_obj)
        frames.append(frame)
        valid = frame.pipeline_support & np.asarray(record["gt_valid"], dtype=bool)
        index = sample_indices(valid, args.calibration_score_pixels_frame, rng)
        for method in METHODS:
            score_samples[method].append(frame.scores[method].ravel()[index])
        feature_index = index[:args.calibration_feature_pixels_frame]
        flat = feature[0].permute(1, 2, 0).float().cpu().numpy().reshape(-1, feature.shape[1])
        feature_samples.append(flat[feature_index])
    return frames, {key: np.concatenate(value) for key, value in score_samples.items()}, np.concatenate(feature_samples)


def method_frame_metrics(frame: ForwardFrame, method: str, accepted: np.ndarray) -> dict[str, Any]:
    if method == "raw":
        update = np.zeros_like(frame.raw)
    elif method == "balanced_no_guard":
        update = frame.base_update
    else:
        update = np.where(accepted, frame.base_update, 0.0)
    refined = frame.raw + update
    common_valid = frame.pipeline_support if frame.gt_valid is None else frame.gt_valid
    geometry = geometry_metrics(frame.raw, refined, update, frame.pipeline_support,
                                common_valid, frame.gt) if frame.gt is not None else {}
    eligible = frame.pipeline_support
    base_count = int((frame.base_authorization & eligible).sum())
    kept_count = int((frame.base_authorization & accepted & eligible).sum()) if method not in {"raw", "balanced_no_guard"} else (0 if method == "raw" else base_count)
    raw_temporal = float(np.abs(frame.raw - frame.aligned_raw)[eligible].mean()) if eligible.any() else math.nan
    refined_temporal = float(np.abs(refined - frame.aligned_raw)[eligible].mean()) if eligible.any() else math.nan
    return {
        "dataset": frame.dataset, "backbone": frame.backbone, "sequence": frame.sequence,
        "frame_id": frame.frame_id, "method": method,
        "support_accepted_count": int((accepted & eligible).sum()) if method not in {"raw", "balanced_no_guard"} else int(eligible.sum()),
        "support_eligible_count": int(eligible.sum()), "base_authorized_count": base_count,
        "guard_authorized_count": kept_count,
        "support_acceptance": float(accepted[eligible].mean()) if eligible.any() and method not in {"raw", "balanced_no_guard"} else 1.0,
        "authorization_retention": kept_count / max(1, base_count),
        "raw_mc_temporal_error": raw_temporal, "refined_mc_temporal_error": refined_temporal,
        "temporal_delta": refined_temporal - raw_temporal,
        **geometry,
    }


def operational_support(frame: ForwardFrame, method: str, threshold: float, granularity: str) -> np.ndarray:
    pixel = np.isfinite(frame.scores[method]) & (frame.scores[method] <= threshold)
    if granularity == "pixel":
        return pixel
    if granularity != "frame":
        raise ValueError("granularity must be pixel or frame")
    value = frame.scores[method][frame.pipeline_support]
    frame_accept = bool(value.size and np.isfinite(value).any() and np.nanmedian(value) <= threshold)
    return np.full_like(frame.pipeline_support, frame_accept, dtype=bool)


def aggregate_rows(rows: list[dict[str, Any]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    output = []
    frame = pd.DataFrame(rows)
    for identity, group in frame.groupby(list(group_keys), dropna=False):
        if not isinstance(identity, tuple):
            identity = (identity,)
        result = dict(zip(group_keys, identity))
        result["frames"] = int(len(group))
        result["support_eligible_count"] = int(group.support_eligible_count.sum())
        result["support_accepted_count"] = int(group.support_accepted_count.sum())
        result["support_acceptance"] = result["support_accepted_count"] / max(1, result["support_eligible_count"])
        result["base_authorized_count"] = int(group.base_authorized_count.sum())
        result["guard_authorized_count"] = int(group.guard_authorized_count.sum())
        result["authorization_retention"] = result["guard_authorized_count"] / max(1, result["base_authorized_count"])
        for key in ("raw_epe", "refined_epe", "raw_bad1", "refined_bad1", "raw_bad3", "refined_bad3", "new_bad3", "raw_boundary_epe", "refined_boundary_epe"):
            if key in group and group[key].notna().any():
                weight = group.get("valid_count", pd.Series(np.ones(len(group)), index=group.index)).fillna(0).to_numpy()
                valid = group[key].notna().to_numpy() & (weight > 0)
                result[key] = float(np.average(group.loc[valid, key], weights=weight[valid])) if valid.any() else math.nan
        for key, weight_key in (("false_update_rate", "clean_count"), ("clean_degradation", "clean_count"),
                                ("mean_clean_update", "clean_count"), ("intervention_precision", "modified_count"),
                                ("harmful_modified_rate", "modified_count")):
            if key in group and group[key].notna().any():
                weight = group.get(weight_key, pd.Series(np.ones(len(group)), index=group.index)).fillna(0).to_numpy()
                valid = group[key].notna().to_numpy() & (weight > 0)
                result[key] = float(np.average(group.loc[valid, key], weights=weight[valid])) if valid.any() else math.nan
        for key in ("raw_mc_temporal_error", "refined_mc_temporal_error", "temporal_delta"):
            result[key] = float(group[key].mean())
        if "raw_epe" in group and group.raw_epe.notna().any():
            delta = group.refined_epe - group.raw_epe
            result["refined_minus_raw_epe"] = float(result["refined_epe"] - result["raw_epe"])
            result["frames_worsened"] = float((delta > 0).mean())
            result["worst_frame_degradation"] = float(delta.max())
            result["p95_frame_degradation"] = float(delta.quantile(.95))
        output.append(result)
    return output


def shifted_feature_scores(guard: SupportGuard, features: np.ndarray, method: str, seed: int) -> tuple[float, dict[str, float]]:
    rng = np.random.default_rng(seed)
    mean = guard.mean[0, :, 0, 0].cpu().numpy()
    std = guard.std[0, :, 0, 0].cpu().numpy()
    shifts = {
        "radial_scale_1.75": mean + (features - mean) * 1.75,
        "alternating_2sd": features + std * np.where(np.arange(features.shape[1]) % 2, -2.0, 2.0),
        "gaussian_1.5sd": features + rng.normal(size=features.shape).astype(np.float32) * std * 1.5,
    }
    def score(value: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(value.T[:, :, None]).permute(2, 0, 1).unsqueeze(-1).to(guard.mean.device)
        return guard.score(tensor, method).flatten().float().cpu().numpy()
    real = score(features.astype(np.float32))
    synthetic = {name: score(value.astype(np.float32)) for name, value in shifts.items()}
    labels = np.concatenate((np.zeros(len(real)), np.ones(sum(len(x) for x in synthetic.values()))))
    values = np.concatenate((real, *(synthetic.values())))
    return float(roc_auc_score(labels, values)), {name: float(value.mean()) for name, value in synthetic.items()}


def benchmark_knn_bank_sizes(guard: SupportGuard, features: np.ndarray) -> list[dict[str, Any]]:
    """Small fixed feature benchmark; it does not affect method selection."""
    sample = features[:min(2048, len(features))].astype(np.float32)
    tensor = torch.from_numpy(sample.T[:, :, None]).permute(2, 0, 1).unsqueeze(-1).to(guard.mean.device)
    z, _ = guard.standardized(tensor)
    flat = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])
    rows = []
    for size in sorted({min(size, len(guard.reference_bank)) for size in (256, 1024, 4096)}):
        bank = guard.reference_bank[:size].to(dtype=flat.dtype)
        if flat.device.type == "cuda":
            torch.cuda.synchronize(flat.device)
        start = time.perf_counter()
        distance = torch.cdist(flat, bank)
        _ = distance.topk(min(guard.knn_k, size), largest=False, dim=1).values.mean(dim=1)
        if flat.device.type == "cuda":
            torch.cuda.synchronize(flat.device)
        rows.append({"bank_size": int(size), "feature_vectors": len(flat),
                     "latency_ms": (time.perf_counter() - start) * 1000,
                     "bank_memory_bytes": int(bank.numel() * bank.element_size())})
    return rows


def select_thresholds(
    frames: list[ForwardFrame], score_samples: dict[str, np.ndarray],
    feature_samples: np.ndarray, guard: SupportGuard, split: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    provenance = SupportProvenance("SCARED-C", "calibration", tuple(SEEN_BACKBONES),
                                   tuple(split["calibration_sequences"]), args.seed)
    shift_auc, shift_means = {}, {}
    for method in METHODS:
        shift_auc[method], shift_means[method] = shifted_feature_scores(guard, feature_samples, method, args.seed)
    base_rows = [method_frame_metrics(frame, "balanced_no_guard", np.ones_like(frame.pipeline_support)) for frame in frames]
    raw_rows = [method_frame_metrics(frame, "raw", np.zeros_like(frame.pipeline_support)) for frame in frames]
    base = aggregate_rows(base_rows, ("method",))[0]
    raw = aggregate_rows(raw_rows, ("method",))[0]
    base_gain = raw["raw_epe"] - base["refined_epe"]
    table = []
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for method in METHODS:
        for quantile in QUANTILES:
            threshold = quantile_threshold(score_samples[method], quantile, provenance=provenance)
            for granularity in ("pixel", "frame"):
                variant = f"{method}_{granularity}"
                rows = [method_frame_metrics(frame, variant, operational_support(frame, method, threshold, granularity)) for frame in frames]
                aggregate = aggregate_rows(rows, ("method",))[0]
                guard_gain = raw["raw_epe"] - aggregate["refined_epe"]
                retained = guard_gain / base_gain if base_gain > 1e-8 else 1.0
                row = {
                    "variant": variant, "method": method, "granularity": granularity,
                    "quantile": quantile, "threshold": threshold,
                    "synthetic_shift_auroc": shift_auc[method], "retained_a2_gain": retained,
                    "raw_epe": raw["raw_epe"], "balanced_epe": base["refined_epe"],
                    "guarded_epe": aggregate["refined_epe"], "support_acceptance": aggregate["support_acceptance"],
                    "authorization_retention": aggregate["authorization_retention"],
                    "false_update_rate": aggregate.get("false_update_rate", math.nan),
                    "clean_degradation": aggregate.get("clean_degradation", math.nan),
                    "intervention_precision": aggregate.get("intervention_precision", math.nan),
                }
                table.append(row)
                if retained >= .70 and row["false_update_rate"] < .05 and row["clean_degradation"] < .03 and row["authorization_retention"] > 0:
                    candidates[variant].append(row)
    chosen_per_variant = {}
    for variant in sorted({row["variant"] for row in table}):
        pool = candidates[variant]
        if not pool:
            pool = [row for row in table if row["variant"] == variant]
        chosen_per_variant[variant] = sorted(pool, key=lambda row: (-row["synthetic_shift_auroc"], row["quantile"], -row["retained_a2_gain"]))[0]
    chosen_per_method = {}
    for method in METHODS:
        pool = [chosen_per_variant[f"{method}_{granularity}"] for granularity in ("pixel", "frame")]
        chosen_per_method[method] = sorted(pool, key=lambda row: (-row["synthetic_shift_auroc"], row["quantile"], -row["retained_a2_gain"]))[0]
    shrinkage = score_samples["shrinkage"]
    knn = score_samples["knn"]
    correlation = float(np.corrcoef(shrinkage, knn)[0, 1])
    run_g4 = correlation < .65
    # G4 is deliberately skipped unless score complementarity is substantial.
    selected = sorted(chosen_per_variant.values(), key=lambda row: (-row["synthetic_shift_auroc"], row["quantile"], -row["retained_a2_gain"]))[0]
    policy = {
        "selected_method": selected["method"], "selected_quantile": selected["quantile"],
        "selected_granularity": selected["granularity"], "selected_variant": selected["variant"],
        "selected_threshold": selected["threshold"], "per_method": chosen_per_method,
        "per_variant": chosen_per_variant,
        "g2_g3_score_correlation": correlation, "g4_run": run_g4,
        "g4_reason": "run only when G2/G3 correlation <0.65" if run_g4 else "skipped by YAGNI: G2/G3 were not materially complementary",
        "selection_dataset": "SCARED-C", "selection_split": "calibration",
        "selection_sequences": split["calibration_sequences"], "candidate_quantiles": QUANTILES,
        "synthetic_shift_auc": shift_auc, "synthetic_shift_score_means": shift_means,
        "base_gain": base_gain,
    }
    return policy, table


def freeze_support_policy(output: Path, policy: dict[str, Any], reference_manifest: dict[str, Any], frozen: dict[str, Any]) -> dict[str, Any]:
    manifest = {
        "status": "frozen_before_any_unseen_or_ood_loader", "policy": policy,
        "support_reference_sha256": reference_manifest["reference_sha256"],
        "original_frozen_artifacts": frozen,
        "forbidden_during_fit_selection": ["Fast-FoundationStereo", "CREStereo", "SERV-CT", "D4D", "SCARED structured-light", "StereoMIS"],
    }
    save_json(output / "frozen_manifest.json", manifest)
    manifest["manifest_sha256"] = sha256(output / "frozen_manifest.json")
    return manifest


def iter_scared_structured_light(device: torch.device, limit: int = 0) -> Iterator[dict[str, Any]]:
    model, infer = s2m2_model(device)
    root = ARGOS_ROOT / "dataset/SCARED/curated/geometric_gt/strong_keyframes_rectified"
    directories = sorted(path for path in root.glob("dataset_*/keyframe_*") if path.is_dir())
    if limit:
        directories = directories[:limit]
    for directory in directories:
        left, right = read_rgb(directory / "left_rectified.png"), read_rgb(directory / "right_rectified.png")
        raw_native, _, _ = infer(model, left, right, device, 512)
        raw = resize_disparity(raw_native)
        gt_native = np.load(directory / "gt_disparity.npy").astype(np.float32)
        valid_native = cv2.imread(str(directory / "valid_mask.png"), cv2.IMREAD_GRAYSCALE) > 0
        gt, gt_valid, _ = resize_gt(gt_native, valid_native)
        valid = np.isfinite(raw) & (raw > 0)
        yield {"frame": f"SCARED-SL:{directory.parent.name}:{directory.name}", "raw": raw,
               "raw_valid": valid, "past": raw, "past_valid": valid,
               "current_rgb": left, "past_rgb": left, "gt": gt, "gt_valid": gt_valid}


def save_contact_sheet(path: Path, frame: ForwardFrame, accepted: np.ndarray) -> None:
    rgb = cv2.resize(frame.current_rgb, (W, H), interpolation=cv2.INTER_AREA)
    guarded = frame.raw + np.where(accepted, frame.base_update, 0.0)
    scale = max(float(np.quantile(frame.raw, .99)), 1e-3)
    def colour(value: np.ndarray, vmax: float) -> np.ndarray:
        u8 = np.clip(value / vmax * 255, 0, 255).astype(np.uint8)
        return cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    support = np.repeat((accepted[..., None] * 255).astype(np.uint8), 3, axis=2)
    sheet = np.concatenate((rgb, colour(frame.raw, scale), colour(guarded, scale), support), axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def evaluate_dataset(
    dataset: str, records: Iterable[dict[str, Any]], pipe: FrozenARGOS,
    guard: SupportGuard, capture: dict[str, torch.Tensor], policy: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[float]]:
    rows, score_rows, acceptance_rows, latencies = [], [], [], []
    selected_method = policy["selected_method"]
    selected_granularity = policy["selected_granularity"]
    selected_threshold = float(policy["selected_threshold"])
    diagnostics = 0
    for record in records:
        frame, _ = forward_record(dataset, record, pipe, guard, capture, args.device_obj)
        for method in METHODS:
            value = frame.scores[method][frame.pipeline_support]
            if value.size:
                score_rows.append({"dataset": dataset, "backbone": frame.backbone, "sequence": frame.sequence,
                                   "method": method, "count": int(value.size), "mean": float(value.mean()),
                                   "std": float(value.std()), "p50": float(np.quantile(value, .5)),
                                   "p90": float(np.quantile(value, .9)), "p95": float(np.quantile(value, .95)),
                                   "p99": float(np.quantile(value, .99))})
            latencies.append(frame.score_latency_ms[method])
        methods = {"raw": np.zeros_like(frame.pipeline_support),
                   "balanced_no_guard": np.ones_like(frame.pipeline_support)}
        for method in METHODS:
            for granularity in ("pixel", "frame"):
                variant = f"{method}_{granularity}"
                threshold = float(policy["per_variant"][variant]["threshold"])
                methods[variant] = operational_support(frame, method, threshold, granularity)
        methods["selected_guard"] = operational_support(frame, selected_method, selected_threshold, selected_granularity)
        if frame.gt is not None:
            base_refined = frame.raw + frame.base_update
            methods["oracle_support"] = np.abs(base_refined - frame.gt) < np.abs(frame.raw - frame.gt)
        for method, accepted in methods.items():
            rows.append(method_frame_metrics(frame, method, accepted))
            acceptance_rows.append({"dataset": dataset, "backbone": frame.backbone, "sequence": frame.sequence,
                                    "frame_id": frame.frame_id, "method": method,
                                    "support_acceptance": rows[-1]["support_acceptance"],
                                    "authorization_retention": rows[-1]["authorization_retention"]})
        if dataset == "StereoMIS" and diagnostics < 3:
            save_contact_sheet(args.output / "diagnostics" / f"StereoMIS_{frame.sequence}_{frame.frame_id}.png",
                               frame, methods["selected_guard"])
            diagnostics += 1
    return rows, score_rows, acceptance_rows, latencies


def run_frozen_evaluations(
    pipe: FrozenARGOS, guard: SupportGuard, capture: dict[str, torch.Tensor],
    split: dict[str, Any], policy: dict[str, Any], args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[float]]:
    # This function is called only after freeze_support_policy has serialized the
    # selected method and thresholds.  No result below can change that policy.
    specs: list[tuple[str, Iterable[dict[str, Any]]]] = [
        ("SCARED-C-test", iter_scared(SEEN_BACKBONES, split["seen_test_sequences"], args.max_test_pairs)),
        ("Fast-FoundationStereo", iter_scared(["Fast-FoundationStereo"], split["seen_test_sequences"], args.max_test_pairs)),
        ("CREStereo", iter_scared(["CREStereo"], split["seen_test_sequences"], args.max_test_pairs)),
        ("SERV-CT", iter_serv()),
        ("SCARED-structured-light", iter_scared_structured_light(args.device_obj, args.static_limit)),
        ("D4D", iter_d4d(args.d4d_windows)),
        ("StereoMIS", iter_stereomis(args.stereomis_samples_per_sequence, args.device_obj)),
    ]
    all_rows, all_scores, all_acceptance, all_latencies = [], [], [], []
    for dataset, records in specs:
        print(f"[D1] frozen evaluation: {dataset}", flush=True)
        rows, scores, acceptance, latencies = evaluate_dataset(dataset, records, pipe, guard, capture, policy, args)
        all_rows.extend(rows); all_scores.extend(scores); all_acceptance.extend(acceptance); all_latencies.extend(latencies)
    return all_rows, all_scores, all_acceptance, all_latencies


def verdicts(per_dataset: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(per_dataset)
    selected = frame[frame.method == "selected_guard"].set_index("dataset")
    base = frame[frame.method == "balanced_no_guard"].set_index("dataset")
    raw = frame[frame.method == "raw"].set_index("dataset")
    result = {}
    for dataset in selected.index:
        row = selected.loc[dataset]
        verdict, reason = "DIAGNOSTIC", "no dense GT"
        if dataset in {"SCARED-C-test", "Fast-FoundationStereo", "CREStereo"}:
            clean = row.get("clean_degradation", math.nan)
            false = row.get("false_update_rate", math.nan)
            base_gain = raw.loc[dataset].raw_epe - base.loc[dataset].refined_epe
            guard_gain = raw.loc[dataset].raw_epe - row.refined_epe
            retained = guard_gain / base_gain if base_gain > 1e-8 else 1.0
            required_retention = .70 if dataset == "SCARED-C-test" else .60
            safe = clean < .03 and false < .05 and row.authorization_retention > 0 and retained >= required_retention
            verdict, reason = ("GO", "in-support gain/safety preserved") if safe else ("NO-GO", "in-support transfer or safety lost")
        elif dataset == "SERV-CT":
            safe = row.get("false_update_rate", 1) < .15 and row.get("clean_degradation", 1) < .10 and row.get("refined_minus_raw_epe", 1) <= .01
            verdict, reason = ("GO", "cross-domain failure suppressed") if safe else ("NO-GO", "SERV-CT remains unsafe")
        elif dataset == "D4D":
            safe = row.get("false_update_rate", 1) < .15 and row.get("clean_degradation", 1) < .10 and row.get("refined_bad3", 1) <= row.get("raw_bad3", 0) + 1e-4
            verdict, reason = ("GO", "anchor safety restored") if safe else ("NO-GO", "D4D anchor safety remains insufficient")
        elif dataset == "SCARED-structured-light":
            safe = abs(row.get("refined_minus_raw_epe", 1)) <= .01 and row.get("clean_degradation", 1) < .03
            verdict, reason = ("GO", "approximately identity preserving") if safe else ("NO-GO", "static geometry degraded")
        result[dataset] = {"verdict": verdict, "reason": reason,
                           "support_acceptance": row.support_acceptance,
                           "authorization_retention": row.authorization_retention,
                           "gain_retained": retained if dataset in {"SCARED-C-test", "Fast-FoundationStereo", "CREStereo"} else None}
    required = [result.get(key, {}).get("verdict") == "GO" for key in
                ("SCARED-C-test", "Fast-FoundationStereo", "CREStereo", "SERV-CT", "D4D", "SCARED-structured-light")]
    result["overall"] = {"verdict": "STRONG GO" if all(required) else "NO-GO", "all_required_pass": all(required)}
    return result


def write_readme(output: Path, policy: dict[str, Any], verdict: dict[str, Any], runtime: dict[str, Any]) -> None:
    lines = [
        "# ARGOS v2 Support Guard Validation", "",
        "Frozen D1 validation of `a_final = a_error AND a_support`. No neural module was trained and no OOD dataset participated in fitting or threshold selection.", "",
        "## Frozen policy", "",
        f"- representation: 24-channel Raw Error Detector penultimate feature;",
        f"- method: `{policy['selected_method']}`;",
        f"- granularity: `{policy['selected_granularity']}`;",
        f"- calibration acceptance quantile: `{policy['selected_quantile']}`;",
        f"- threshold: `{policy['selected_threshold']}`;",
        f"- G2/G3 correlation: `{policy['g2_g3_score_correlation']}`; G4: `{policy['g4_reason']}`.", "",
        "## Verdict", "", f"**{verdict['overall']['verdict']}**", "",
        "See `per_dataset.csv`, `safety_summary.json`, `verdicts.json` and `aggregate_summary.json` for exact results.", "",
        "## Runtime", "", f"Median isolated support-score latency: {runtime['support_score_latency_ms_median']:.4f} ms; compact reference memory: {runtime['reference_memory_bytes']} bytes.", "",
        "All rejected pixels use `torch.where` and are bit-exact raw. StereoMIS results are no-reference diagnostics only.", "",
    ]
    (output / "README.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--max-train-pairs", type=int, default=32)
    parser.add_argument("--max-calibration-pairs", type=int, default=160)
    parser.add_argument("--max-test-pairs", type=int, default=160)
    parser.add_argument("--fit-pixels-per-status-frame", type=int, default=16)
    parser.add_argument("--fit-pixels-per-group", type=int, default=512)
    parser.add_argument("--calibration-score-pixels-frame", type=int, default=64)
    parser.add_argument("--calibration-feature-pixels-frame", type=int, default=16)
    parser.add_argument("--bank-size", type=int, default=4096)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--d4d-windows", type=int, default=156)
    parser.add_argument("--stereomis-samples-per-sequence", type=int, default=128)
    parser.add_argument("--static-limit", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.max_train_pairs = args.max_calibration_pairs = args.max_test_pairs = 1
        args.fit_pixels_per_status_frame = 4
        args.fit_pixels_per_group = 4
        args.calibration_score_pixels_frame = 8
        args.calibration_feature_pixels_frame = 4
        args.bank_size = 32
        args.knn_k = 3
        args.d4d_windows = 1
        args.stereomis_samples_per_sequence = 1
        args.static_limit = 1
    args.device_obj = torch.device(args.device)
    return args


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    (args.output / "run.log").write_text(" ".join(sys.argv) + "\n")
    split = json.loads(SPLIT_PATH.read_text())
    save_json(args.output / "config.json", {key: value for key, value in vars(args).items() if key != "device_obj"})
    save_json(args.output / "split_manifest.json", split)
    frozen = verify_frozen()
    pipe = FrozenARGOS(args.device_obj)
    capture: dict[str, torch.Tensor] = {}
    hook = pipe.detector.encoder.register_forward_hook(lambda _module, _inputs, output: capture.update(penultimate=output.detach()))
    print("[D1] fitting SCARED-C training support", flush=True)
    guard, reference_manifest = fit_reference_support(pipe, capture, split, args)
    print("[D1] selecting thresholds on SCARED-C calibration only", flush=True)
    calibration_frames, score_samples, feature_samples = collect_calibration(pipe, guard, capture, split, args)
    bank_benchmark = benchmark_knn_bank_sizes(guard, feature_samples)
    policy, threshold_rows = select_thresholds(calibration_frames, score_samples, feature_samples, guard, split, args)
    dataframe_csv(args.output / "threshold_selection.csv", threshold_rows)
    frozen_support = freeze_support_policy(args.output, policy, reference_manifest, frozen)
    # Calibration rows become part of the final method comparison, but never OOD selection.
    calibration_eval_rows = []
    for frame in calibration_frames:
        for method in ("raw", "balanced_no_guard"):
            calibration_eval_rows.append(method_frame_metrics(frame, method, np.ones_like(frame.pipeline_support)))
        for method in METHODS:
            for granularity in ("pixel", "frame"):
                variant = f"{method}_{granularity}"
                threshold = policy["per_variant"][variant]["threshold"]
                calibration_eval_rows.append(method_frame_metrics(frame, variant, operational_support(frame, method, threshold, granularity)))
        calibration_eval_rows.append(method_frame_metrics(frame, "selected_guard", operational_support(
            frame, policy["selected_method"], policy["selected_threshold"], policy["selected_granularity"])))
    rows, score_rows, acceptance_rows, latency = run_frozen_evaluations(pipe, guard, capture, split, policy, args)
    hook.remove()
    all_rows = calibration_eval_rows + rows
    dataframe_csv(args.output / "frame_metrics.csv", all_rows)
    sequence = aggregate_rows(all_rows, ("dataset", "backbone", "sequence", "method"))
    backbone = aggregate_rows(all_rows, ("dataset", "backbone", "method"))
    dataset = aggregate_rows(all_rows, ("dataset", "method"))
    dataframe_csv(args.output / "sequence_metrics.csv", sequence)
    dataframe_csv(args.output / "per_backbone.csv", backbone)
    dataframe_csv(args.output / "per_dataset.csv", dataset)
    dataframe_csv(args.output / "method_comparison.csv", dataset)
    dataframe_csv(args.output / "support_score_summary.csv", score_rows)
    dataframe_csv(args.output / "support_acceptance.csv", acceptance_rows)
    safety = [row for row in dataset if row["method"] in {"balanced_no_guard", "selected_guard"}]
    save_json(args.output / "safety_summary.json", safety)
    verdict = verdicts(dataset)
    save_json(args.output / "verdicts.json", verdict)
    runtime = {
        "support_score_latency_ms_mean": float(np.mean(latency)),
        "support_score_latency_ms_median": float(np.median(latency)),
        "support_score_latency_ms_p95": float(np.quantile(latency, .95)),
        "reference_memory_bytes": reference_manifest["memory_bytes"],
        "bank_size": reference_manifest["bank_size"], "knn_k": reference_manifest["knn_k"],
        "knn_bank_size_benchmark": bank_benchmark,
        "frozen_baseline_runtime": json.loads((ROOT / "results/ood_generalization/runtime_summary.json").read_text()),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(args.device_obj)) if args.device_obj.type == "cuda" else 0,
    }
    save_json(args.output / "runtime_summary.json", runtime)
    aggregate = {"policy": policy, "reference": reference_manifest, "frozen": frozen_support,
                 "verdicts": verdict, "runtime": runtime, "datasets": dataset}
    save_json(args.output / "aggregate_summary.json", aggregate)
    (args.output / "SUPPORT_GUARD_AUDIT.md").write_text((ROOT / "model_design/SUPPORT_GUARD_AUDIT.md").read_text())
    write_readme(args.output, policy, verdict, runtime)
    print(json.dumps({"output": str(args.output), "selected": policy["selected_method"],
                      "granularity": policy["selected_granularity"], "quantile": policy["selected_quantile"],
                      "verdict": verdict["overall"]["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
