#!/usr/bin/env python3
"""Stage-gated frozen DINOv3 representation validation for ARGOS v2."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))
sys.path.insert(0, str(V2_ROOT / "scripts"))

from argos_v2.scared_c_data import load_sequence_info, read_rgb  # noqa: E402
from model_design.external_components.bidavideo import (  # noqa: E402
    BiDAFlowInferenceAdapter,
    photometric_consistency,
)
from model_design.external_components.dinov3 import FrozenDINOv3, warp_dino_feature  # noqa: E402
from model_design.data.temporal_memory_dataset import TemporalMemoryDataset  # noqa: E402
from model_design.data.temporal_pair_dataset import (  # noqa: E402
    DEFAULT_VALIDATION_SEQUENCES,
    SEEN_BACKBONES,
    build_split_manifest,
)
from model_design.external_components.bidavideo import temporal_disparity_evidence  # noqa: E402
from model_design.models.dinov3_memory_selector import (  # noqa: E402
    DINORepresentationSelector,
    VARIANTS,
    selector_targets,
)
from model_design.models.learned_ppm_selector import LearnedPPMSelectorRefiner  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "probe", "ranking"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolutions", nargs="+", default=["256x320", "320x400"])
    parser.add_argument("--layers", nargs="+", type=int, default=[5, 11, 17, 23])
    parser.add_argument("--backbones", nargs="+", default=list(SEEN_BACKBONES))
    parser.add_argument("--sequences", nargs="+", default=list(DEFAULT_VALIDATION_SEQUENCES))
    parser.add_argument("--samples-per-sequence", type=int, default=24)
    parser.add_argument("--coverage-threshold", type=float, default=0.50)
    parser.add_argument("--useful-margin", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--train-samples-per-sequence", type=int, default=16)
    parser.add_argument("--validation-samples-per-sequence", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    return parser.parse_args()


def parse_size(value: str) -> tuple[int, int]:
    height, width = value.lower().split("x")
    return int(height), int(width)


def json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def clean(item):
        if isinstance(item, dict):
            return {str(key): clean(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(child) for child in item]
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item

    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def rgb_tensor(image: np.ndarray, size: tuple[int, int] | None = None) -> torch.Tensor:
    if size is not None:
        image = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None]


def quantiles(values: torch.Tensor) -> dict[str, float | None]:
    values = values.detach().float().flatten()
    values = values[torch.isfinite(values)]
    if not values.numel():
        return {"count": 0, "mean": None, "p05": None, "p50": None, "p95": None}
    q = torch.quantile(values, torch.tensor([0.05, 0.5, 0.95], device=values.device))
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "p05": float(q[0]),
        "p50": float(q[1]),
        "p95": float(q[2]),
    }


def selected_pairs() -> list[tuple[str, int]]:
    # Fixed held-out seen sequences: tissue/specularity/boundaries and a broad
    # motion range. This is a sanity sample, never an architecture conclusion.
    return [
        ("dataset_7_keyframe_1", 32),
        ("dataset_7_keyframe_1", 160),
        ("dataset_7_keyframe_2", 80),
        ("dataset_7_keyframe_3", 120),
        ("dataset_7_keyframe_4", 40),
        ("dataset_7_keyframe_4", 200),
    ]


def smoke(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    dino = FrozenDINOv3(device=device)
    flow_model = BiDAFlowInferenceAdapter("sea_raft", device=device)
    pairs = []
    for sequence, current_index in selected_pairs():
        info = load_sequence_info(sequence)
        current_index = min(current_index, len(info.frame_ids) - 1)
        if current_index < 1:
            continue
        current_id, past_id = info.frame_ids[current_index], info.frame_ids[current_index - 1]
        current = read_rgb(info.seq_dir / "left" / f"{current_id}.png")
        past = read_rgb(info.seq_dir / "left" / f"{past_id}.png")
        pairs.append((sequence, current_id, past_id, current, past))

    cache_current = torch.cat([rgb_tensor(item[3], (144, 180)) for item in pairs]).to(device)
    cache_past = torch.cat([rgb_tensor(item[4], (144, 180)) for item in pairs]).to(device)
    flow_current_past = flow_model.infer(cache_current, cache_past)
    photo = photometric_consistency(cache_current, cache_past, flow_current_past)
    records = []
    runtime = {}
    for resolution_text in args.resolutions:
        resolution = parse_size(resolution_text)
        # Time on one real frame; extract the controlled batch separately.
        _, timing = dino.measure(
            rgb_tensor(pairs[0][3]), layers=args.layers, input_size=resolution, warmup=2, repeats=5
        )
        runtime[resolution_text] = {
            "latency_ms_per_frame": timing.latency_ms_per_frame,
            "peak_gpu_memory_bytes": timing.peak_gpu_memory_bytes,
            "patch_grid": [resolution[0] // 16, resolution[1] // 16],
        }
        current_native = torch.cat([rgb_tensor(item[3]) for item in pairs])
        past_native = torch.cat([rgb_tensor(item[4]) for item in pairs])
        current_features = dino.extract(current_native, layers=args.layers, input_size=resolution)
        past_features = dino.extract(past_native, layers=args.layers, input_size=resolution)
        for layer, current_map, past_map in zip(
            args.layers, current_features.feature_maps, past_features.feature_maps, strict=True
        ):
            aligned = warp_dino_feature(past_map, flow_current_past)
            cosine = F.cosine_similarity(current_map.float(), aligned.warped.float(), dim=1)
            nearby_negative = F.cosine_similarity(
                current_map.float(), torch.roll(aligned.warped.float(), shifts=2, dims=-1), dim=1
            )
            photo_grid = F.interpolate(
                photo.robust_normalized_residual, current_map.shape[-2:], mode="bilinear", align_corners=True
            )[:, 0]
            valid = aligned.valid[:, 0]
            valid_photo = photo_grid[valid]
            high_photo_threshold = torch.quantile(valid_photo, 0.75) if valid_photo.numel() else torch.tensor(1.0, device=device)
            high_photo = valid & (photo_grid >= high_photo_threshold)
            occluded = ~aligned.support[:, 0]
            unaligned_cosine = F.cosine_similarity(current_map.float(), past_map.float(), dim=1)
            records.append(
                {
                    "resolution": resolution_text,
                    "layer": layer,
                    "shape": list(current_map.shape),
                    "feature": quantiles(current_map),
                    "corresponding_cosine": quantiles(cosine[valid]),
                    "nearby_negative_cosine": quantiles(nearby_negative[valid]),
                    "strong_photometric_change_cosine": quantiles(cosine[high_photo]),
                    "occluded_unaligned_cosine": quantiles(unaligned_cosine[occluded]),
                    "support_ratio": float(valid.float().mean()),
                }
            )
    result = {
        "purpose": "tiny real-frame sanity only; not architecture evidence",
        "pairs": [
            {"sequence": item[0], "current_frame_id": item[1], "past_frame_id": item[2]}
            for item in pairs
        ],
        "layers": args.layers,
        "runtime": runtime,
        "records": records,
    }
    json_dump(output / "feature_smoke.json", result)
    (output / "run.log").write_text("DINO smoke completed successfully\n")
    print(json.dumps(result, indent=2))
    return 0


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _single_batch(item: dict, device: torch.device) -> dict:
    return {
        key: value[None].to(device) if torch.is_tensor(value) else value
        for key, value in item.items()
    }


def _native_temporal_rgb(item: dict) -> tuple[torch.Tensor, torch.Tensor]:
    info = load_sequence_info(item["sequence"])
    current = read_rgb(info.seq_dir / "left" / f"{item['current_frame_id']}.png")
    current_tensor = rgb_tensor(current)
    memories = []
    for frame_id in item["past_frame_ids"]:
        memories.append(rgb_tensor(read_rgb(info.seq_dir / "left" / f"{frame_id}.png")))
    return current_tensor, torch.cat(memories)


def _build_four_age_evidence(
    flow_model: BiDAFlowInferenceAdapter,
    batch: dict,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    m = batch["past"].shape[1]
    current_rgb = batch["current_rgb"].expand(m, -1, -1, -1)
    past_rgb = batch["past_rgb"][0]
    flow_cp = flow_model.infer(current_rgb, past_rgb)
    flow_pc = flow_model.infer(past_rgb, current_rgb)
    raw = batch["raw"].expand(m, -1, -1, -1)
    raw_valid = batch["raw_valid"].expand(m, -1, -1, -1)
    value = temporal_disparity_evidence(
        raw,
        batch["past"][0],
        flow_cp,
        flow_pc,
        current_valid=raw_valid,
        past_valid=batch["past_valid"][0],
        current_rgb=current_rgb,
        past_rgb=past_rgb,
    )
    return value.as_dict(), flow_cp


def _classification_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    if labels.min() == labels.max():
        return {"memory_better_auroc": float("nan"), "memory_better_ap": float("nan")}
    return {
        "memory_better_auroc": float(roc_auc_score(labels, scores)),
        "memory_better_ap": float(average_precision_score(labels, scores)),
    }


def probe(args: argparse.Namespace) -> int:
    """GT-controlled non-parametric layer/resolution gate on seen backbones."""
    if any(name == "Fast-FoundationStereo" for name in args.backbones):
        raise ValueError("Fast-FoundationStereo is forbidden during DINO tuning")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = TemporalMemoryDataset(
        args.backbones,
        args.sequences,
        ages=(1, 2, 4, 8),
        max_samples_per_sequence=args.samples_per_sequence,
        random_clip_start=True,
        seed=args.seed,
    )
    dino = FrozenDINOv3(device=device)
    flow_model = BiDAFlowInferenceAdapter("sea_raft", device=device)
    keys = [(parse_size(text), text, layer) for text in args.resolutions for layer in args.layers]
    accum = {
        (text, layer): {
            "scores": [], "labels": [], "top_correct": 0, "top_count": 0,
            "pair_correct": 0, "pair_count": 0,
        }
        for _size, text, layer in keys
    }
    for index in range(len(dataset)):
        item = dataset[index]
        batch = _single_batch(item, device)
        evidence, flow_cp = _build_four_age_evidence(flow_model, batch)
        raw_error = (batch["raw"][0] - batch["gt"][0]).abs()
        memory_error = (evidence["aligned_past_disparity"] - batch["gt"][0]).abs()
        candidate_valid = evidence["aligned_validity"] & evidence["warp_support"]
        base_valid = (batch["gt_coverage"][0] > args.coverage_threshold) & batch["raw_valid"][0]
        current_native, past_native = _native_temporal_rgb(item)
        all_native = torch.cat((current_native, past_native))
        for resolution_text in args.resolutions:
            resolution = parse_size(resolution_text)
            features = dino.extract(all_native, layers=args.layers, input_size=resolution)
            for layer, feature in zip(args.layers, features.feature_maps, strict=True):
                current_map = feature[:1].expand(4, -1, -1, -1)
                aligned = warp_dino_feature(feature[1:], flow_cp)
                score = F.cosine_similarity(current_map.float(), aligned.warped.float(), dim=1, eps=1e-6)[:, None]
                score = F.interpolate(score, raw_error.shape[-2:], mode="bilinear", align_corners=True)
                valid = candidate_valid & base_valid
                advantage = raw_error - memory_error
                labels = advantage > args.useful_margin
                state = accum[(resolution_text, layer)]
                # Deterministic spatial thinning controls memory without changing examples.
                sampled = valid[..., ::4, ::4]
                state["scores"].append(score[..., ::4, ::4][sampled].detach().cpu().numpy())
                state["labels"].append(labels[..., ::4, ::4][sampled].detach().cpu().numpy())
                score_valid = score.masked_fill(~valid, -torch.inf)
                err_valid = memory_error.masked_fill(~valid, torch.inf)
                any_valid = valid.any(dim=0)
                predicted = score_valid.argmax(dim=0)
                best = err_valid.argmin(dim=0)
                state["top_correct"] += int(((predicted == best) & any_valid).sum())
                state["top_count"] += int(any_valid.sum())
                for first in range(4):
                    for second in range(first + 1, 4):
                        pair_valid = valid[first] & valid[second]
                        meaningful = pair_valid & ((memory_error[first] - memory_error[second]).abs() > args.useful_margin)
                        predicted_first = score[first] > score[second]
                        actual_first = memory_error[first] < memory_error[second]
                        state["pair_correct"] += int(((predicted_first == actual_first) & meaningful).sum())
                        state["pair_count"] += int(meaningful.sum())

    rows = []
    for (resolution, layer), state in accum.items():
        scores = np.concatenate(state["scores"])
        labels = np.concatenate(state["labels"]).astype(np.uint8)
        row = {
            "resolution": resolution,
            "layer": layer,
            **_classification_metrics(scores, labels),
            "best_memory_top1_accuracy": state["top_correct"] / max(state["top_count"], 1),
            "pairwise_ranking_accuracy": state["pair_correct"] / max(state["pair_count"], 1),
            "sampled_candidate_pixels": len(scores),
            "memory_better_prevalence": float(labels.mean()),
        }
        rows.append(row)
    rows.sort(key=lambda row: (row["resolution"], row["layer"]))
    write_csv(args.output / "feature_layer_probe.csv", rows)
    resolution_rows = []
    for resolution in args.resolutions:
        candidates = [row for row in rows if row["resolution"] == resolution]
        best = max(candidates, key=lambda row: (row["memory_better_ap"], row["pairwise_ranking_accuracy"]))
        resolution_rows.append({"resolution": resolution, **{f"best_{key}": value for key, value in best.items() if key != "resolution"}})
    write_csv(args.output / "feature_resolution_probe.csv", resolution_rows)
    manifest = build_split_manifest(validation_sequences=args.sequences)
    manifest.update({
        "stage": "DINO layer/resolution probe",
        "actual_backbones": args.backbones,
        "actual_sequences": args.sequences,
        "samples_per_sequence_per_backbone": args.samples_per_sequence,
        "Fast-FoundationStereo_accessed": False,
    })
    json_dump(args.output / "split_manifest.json", manifest)
    json_dump(args.output / "config.json", vars(args))
    (args.output / "run.log").write_text(
        "DINO layer/resolution probe completed; Fast-FoundationStereo not accessed.\n"
    )
    print(json.dumps(rows, indent=2))
    return 0


def _pool_mean(value: torch.Tensor, valid: torch.Tensor, size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    numerator = F.adaptive_avg_pool2d(value * valid.float(), size)
    coverage = F.adaptive_avg_pool2d(valid.float(), size)
    return numerator / coverage.clamp_min(1e-6), coverage


class _ListDataset(Dataset):
    def __init__(self, examples: list[dict]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        return self.examples[index]


def _frozen_projection(device: torch.device) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260713)
    value = torch.randn(16, 1024, generator=generator)
    return F.normalize(value, dim=1).to(device)


@torch.no_grad()
def build_representation_examples(
    dataset: TemporalMemoryDataset,
    dino: FrozenDINOv3,
    flow_model: BiDAFlowInferenceAdapter,
    *,
    resolution: tuple[int, int],
    layers: list[int],
    coverage_threshold: float,
    device: torch.device,
) -> list[dict]:
    """Build compact in-memory regional descriptors; nothing is persisted."""
    grid = (resolution[0] // 16, resolution[1] // 16)
    projection = _frozen_projection(device)
    feature_cache: dict[tuple[str, str], torch.Tensor] = {}
    flow_cache: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
    dino_pair_cache: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
    examples = []

    def projected_frames(item: dict) -> tuple[torch.Tensor, torch.Tensor]:
        ids = [item["current_frame_id"], *item["past_frame_ids"]]
        missing = [frame_id for frame_id in ids if (item["sequence"], frame_id) not in feature_cache]
        if missing:
            info = load_sequence_info(item["sequence"])
            images = torch.cat([
                rgb_tensor(read_rgb(info.seq_dir / "left" / f"{frame_id}.png")) for frame_id in missing
            ])
            extracted = dino.extract(images, layers=layers, input_size=resolution)
            per_frame = torch.stack([
                torch.einsum("oc,bchw->bohw", projection, feature.float()).half().cpu()
                for feature in extracted.feature_maps
            ], dim=1)  # [frames,L,16,h,w]
            for frame_id, value in zip(missing, per_frame, strict=True):
                feature_cache[(item["sequence"], frame_id)] = value
        current = feature_cache[(item["sequence"], ids[0])]
        memories = torch.stack([feature_cache[(item["sequence"], frame_id)] for frame_id in ids[1:]])
        return current, memories

    for index in range(len(dataset)):
        item = dataset[index]
        batch = _single_batch(item, device)
        temporal_key = (item["sequence"], item["current_index"])
        if temporal_key not in flow_cache:
            m = batch["past"].shape[1]
            current_rgb = batch["current_rgb"].expand(m, -1, -1, -1)
            past_rgb = batch["past_rgb"][0]
            flow_cache[temporal_key] = (
                flow_model.infer(current_rgb, past_rgb).half().cpu(),
                flow_model.infer(past_rgb, current_rgb).half().cpu(),
            )
        flow_cp_cpu, flow_pc_cpu = flow_cache[temporal_key]
        flow_cp, flow_pc = flow_cp_cpu.to(device).float(), flow_pc_cpu.to(device).float()
        m = batch["past"].shape[1]
        current_rgb = batch["current_rgb"].expand(m, -1, -1, -1)
        past_rgb = batch["past_rgb"][0]
        value = temporal_disparity_evidence(
            batch["raw"].expand(m, -1, -1, -1), batch["past"][0], flow_cp, flow_pc,
            current_valid=batch["raw_valid"].expand(m, -1, -1, -1),
            past_valid=batch["past_valid"][0], current_rgb=current_rgb, past_rgb=past_rgb,
        )
        evidence = {name: tensor[None] for name, tensor in value.as_dict().items()}
        geom = LearnedPPMSelectorRefiner.normalized_candidate_inputs(
            batch["raw"], batch["raw_valid"], evidence, batch["ages"][0]
        )[0]
        geom = F.interpolate(geom.reshape(m * 12, 1, 144, 180), grid, mode="bilinear", align_corners=True)
        geom = geom.reshape(m, 12, *grid).half().cpu()

        if temporal_key not in dino_pair_cache:
            current_feature, memory_features = projected_frames(item)
            layer_descriptors = []
            for layer_index in range(len(layers)):
                current_map = current_feature[layer_index][None].expand(m, -1, -1, -1).to(device).float()
                memory_map = memory_features[:, layer_index].to(device).float()
                aligned = warp_dino_feature(memory_map, flow_cp)
                cosine = F.cosine_similarity(current_map, aligned.warped, dim=1)[:, None]
                descriptor = torch.cat((
                    current_map,
                    aligned.warped,
                    (current_map - aligned.warped).abs(),
                    (current_map - aligned.warped).square()[:, :15],
                    cosine,
                ), dim=1)
                layer_descriptors.append(descriptor.half().cpu())
            dino_pair_cache[temporal_key] = (
                torch.stack(layer_descriptors),
                aligned.support.half().cpu(),
            )
        dino_descriptor, _dino_support = dino_pair_cache[temporal_key]

        rgb = torch.cat((batch["current_rgb"], batch["past_rgb"][0]), dim=0).float() / 255.0
        rgb = F.interpolate(rgb, grid, mode="bilinear", align_corners=True).half().cpu()
        gt, raw = batch["gt"][0], batch["raw"][0]
        base_valid = (batch["gt_coverage"][0] > coverage_threshold) & batch["raw_valid"][0]
        raw_error, raw_coverage = _pool_mean((raw - gt).abs()[None], base_valid[None], grid)
        memory_error = (value.aligned_past_disparity - gt).abs()
        candidate_pixel_valid = value.aligned_validity & value.warp_support & base_valid
        memory_region_error, memory_coverage = _pool_mean(memory_error, candidate_pixel_valid, grid)
        candidate_valid = memory_coverage > 0.50
        region_valid = (raw_coverage > 0.50)[0]
        errors = torch.cat((raw_error, memory_region_error), dim=0).half().cpu()
        errors[1:] = errors[1:].masked_fill(~candidate_valid.cpu(), 1000.0)
        examples.append({
            "geom": geom,
            "rgb": rgb,
            "dino": dino_descriptor,
            "candidate_valid": candidate_valid.cpu(),
            "errors": errors,
            "region_valid": region_valid.cpu(),
            "backbone_index": torch.tensor(list(SEEN_BACKBONES).index(item["backbone"])),
            "age": item["ages"],
        })
    return examples


def _ranking_loss(output, errors, candidate_valid, region_valid, margin: float) -> dict[str, torch.Tensor]:
    target, useful = selector_targets(errors, candidate_valid, margin=margin)
    valid = region_valid[:, 0].bool()
    ce = F.cross_entropy(output.logits, target, reduction="none")
    listwise = ce[valid].mean()
    candidate_mask = candidate_valid[:, :, 0].bool() & valid[:, None]
    binary = F.binary_cross_entropy_with_logits(
        output.candidate_logits, useful[:, :, 0], reduction="none"
    )
    memory_better = binary[candidate_mask].mean()
    pairwise_terms = []
    for first in range(4):
        for second in range(first + 1, 4):
            difference = errors[:, first + 1, 0] - errors[:, second + 1, 0]
            pair_valid = candidate_mask[:, first] & candidate_mask[:, second] & (difference.abs() > margin)
            sign = -torch.sign(difference)  # +1 means first is better and should score higher
            term = F.softplus(-sign * (output.candidate_logits[:, first] - output.candidate_logits[:, second]))
            if pair_valid.any():
                pairwise_terms.append(term[pair_valid].mean())
    pairwise = torch.stack(pairwise_terms).mean() if pairwise_terms else listwise.new_zeros(())
    total = listwise + 0.5 * memory_better + 0.25 * pairwise
    return {"total": total, "listwise": listwise, "memory_better": memory_better, "pairwise": pairwise}


@torch.no_grad()
def evaluate_ranker(model, loader, device, margin: float) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    model.eval()
    scores, labels, score_groups = [], [], []
    selected, targets, region_groups, confidences, corrects = [], [], [], [], []
    regrets, entropies, pair_correct, pair_count = [], [], 0, 0
    top3_correct, top3_count = 0, 0
    selected_age = np.zeros(5, dtype=np.int64)
    for cpu in loader:
        batch = {key: value.to(device) for key, value in cpu.items() if torch.is_tensor(value)}
        output = model(batch["geom"].float(), batch["rgb"].float(), batch["dino"].float(), batch["candidate_valid"])
        target, useful = selector_targets(batch["errors"].float(), batch["candidate_valid"], margin=margin)
        valid = batch["region_valid"][:, 0].bool()
        candidate_mask = batch["candidate_valid"][:, :, 0].bool() & valid[:, None]
        scores.append(torch.sigmoid(output.candidate_logits)[candidate_mask].cpu().numpy())
        labels.append(useful[:, :, 0][candidate_mask].cpu().numpy())
        score_group_map = batch["backbone_index"][:, None, None, None].expand_as(candidate_mask)
        score_groups.append(score_group_map[candidate_mask].cpu().numpy())
        prediction = output.probabilities.argmax(dim=1)
        top3 = output.probabilities.topk(k=3, dim=1).indices
        top3_correct += int(((top3 == target[:, None]).any(dim=1) & valid).sum())
        top3_count += int(valid.sum())
        selected.append(prediction[valid].cpu().numpy())
        targets.append(target[valid].cpu().numpy())
        region_group_map = batch["backbone_index"][:, None, None].expand_as(valid)
        region_groups.append(region_group_map[valid].cpu().numpy())
        confidence = output.probabilities.max(dim=1).values
        confidences.append(confidence[valid].cpu().numpy())
        corrects.append((prediction == target)[valid].cpu().numpy())
        probability = output.probabilities.clamp_min(1e-12)
        entropies.append((-(probability * probability.log()).sum(dim=1))[valid].cpu().numpy())
        error_map = batch["errors"][:, :, 0].float()
        chosen = error_map.gather(1, prediction[:, None]).squeeze(1)
        best = error_map.min(dim=1).values
        regrets.append((chosen - best)[valid].cpu().numpy())
        for age_index in range(5):
            selected_age[age_index] += int(((prediction == age_index) & valid).sum())
        for first in range(4):
            for second in range(first + 1, 4):
                difference = error_map[:, first + 1] - error_map[:, second + 1]
                pair_valid = candidate_mask[:, first] & candidate_mask[:, second] & (difference.abs() > margin)
                predicted_first = output.candidate_logits[:, first] > output.candidate_logits[:, second]
                actual_first = difference < 0
                pair_correct += int(((predicted_first == actual_first) & pair_valid).sum())
                pair_count += int(pair_valid.sum())
    score = np.concatenate(scores); label = np.concatenate(labels).astype(np.uint8)
    score_group = np.concatenate(score_groups)
    selected_np = np.concatenate(selected); target_np = np.concatenate(targets)
    region_group = np.concatenate(region_groups)
    confidence_np = np.concatenate(confidences); correct_np = np.concatenate(corrects)
    regret_np = np.concatenate(regrets); entropy_np = np.concatenate(entropies)
    predicted_useful = score >= 0.5
    tp = int((predicted_useful & (label == 1)).sum()); fp = int((predicted_useful & (label == 0)).sum())
    fn = int((~predicted_useful & (label == 1)).sum())
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    ece = 0.0
    for low in np.linspace(0, 0.9, 10):
        mask = (confidence_np >= low) & (confidence_np < low + 0.1)
        if mask.any():
            ece += mask.mean() * abs(confidence_np[mask].mean() - correct_np[mask].mean())
    total_selected = selected_age.sum()
    metrics = {
        "memory_better_auroc": float(roc_auc_score(label, score)),
        "memory_better_ap": float(average_precision_score(label, score)),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "top1_best_candidate_accuracy": float((selected_np == target_np).mean()),
        "top3_candidate_recall": top3_correct / max(top3_count, 1),
        "pairwise_ranking_accuracy": pair_correct / max(pair_count, 1),
        "selected_memory_regret": float(regret_np.mean()),
        "calibration_ece": float(ece),
        "brier_memory_better": float(np.mean((score - label) ** 2)),
        "raw_null_abstention_accuracy": float(((selected_np == 0) == (target_np == 0)).mean()),
        "raw_null_selection_rate": float((selected_np == 0).mean()),
        "memory_weight_entropy": float(entropy_np.mean()),
        **{f"selection_rate_class_{index}": int(count) / max(int(total_selected), 1) for index, count in enumerate(selected_age)},
    }
    for group_index, backbone in enumerate(SEEN_BACKBONES):
        candidate_group = score_group == group_index
        region_group_mask = region_group == group_index
        prefix = backbone.replace("-", "_")
        metrics[f"{prefix}_memory_better_auroc"] = float(roc_auc_score(label[candidate_group], score[candidate_group]))
        metrics[f"{prefix}_memory_better_ap"] = float(average_precision_score(label[candidate_group], score[candidate_group]))
        metrics[f"{prefix}_top1_accuracy"] = float((selected_np[region_group_mask] == target_np[region_group_mask]).mean())
        metrics[f"{prefix}_raw_null_accuracy"] = float(
            ((selected_np[region_group_mask] == 0) == (target_np[region_group_mask] == 0)).mean()
        )
        metrics[f"{prefix}_selected_regret"] = float(regret_np[region_group_mask].mean())
    return metrics


def ranking(args: argparse.Namespace) -> int:
    if any(name == "Fast-FoundationStereo" for name in args.backbones):
        raise ValueError("Fast-FoundationStereo is forbidden during representation selection")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = build_split_manifest()
    train_data = TemporalMemoryDataset(
        args.backbones, manifest["train_sequences"], ages=(1, 2, 4, 8),
        max_samples_per_sequence=args.train_samples_per_sequence, random_clip_start=True, seed=args.seed,
    )
    val_data = TemporalMemoryDataset(
        args.backbones, manifest["validation_sequences"], ages=(1, 2, 4, 8),
        max_samples_per_sequence=args.validation_samples_per_sequence, random_clip_start=False, seed=args.seed,
    )
    dino = FrozenDINOv3(device=device)
    flow_model = BiDAFlowInferenceAdapter("sea_raft", device=device)
    resolution = (256, 320)
    layers = [5, 11, 17, 23]
    train_examples = build_representation_examples(
        train_data, dino, flow_model, resolution=resolution, layers=layers,
        coverage_threshold=args.coverage_threshold, device=device,
    )
    val_examples = build_representation_examples(
        val_data, dino, flow_model, resolution=resolution, layers=layers,
        coverage_threshold=args.coverage_threshold, device=device,
    )
    train_loader = DataLoader(_ListDataset(train_examples), batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(_ListDataset(val_examples), batch_size=args.batch_size, shuffle=False)
    rows, states = [], {}
    for variant in VARIANTS:
        torch.manual_seed(args.seed)
        model = DINORepresentationSelector(variant).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        first_loss = None
        for epoch in range(args.epochs):
            model.train()
            for cpu in train_loader:
                batch = {key: value.to(device) for key, value in cpu.items() if torch.is_tensor(value)}
                optimizer.zero_grad(set_to_none=True)
                output = model(batch["geom"].float(), batch["rgb"].float(), batch["dino"].float(), batch["candidate_valid"])
                losses = _ranking_loss(output, batch["errors"].float(), batch["candidate_valid"],
                                       batch["region_valid"], args.useful_margin)
                if first_loss is None:
                    first_loss = float(losses["total"])
                losses["total"].backward(); optimizer.step()
        metrics = evaluate_ranker(model, val_loader, device, args.useful_margin)
        row = {
            "variant": variant,
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "first_training_loss": first_loss,
            **metrics,
        }
        rows.append(row)
        states[variant] = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    write_csv(args.output / "ranking_metrics.csv", rows)
    write_csv(args.output / "selector_metrics.csv", rows)
    # AP is the primary Stage-A bottleneck; regret breaks ties.
    best = max(rows, key=lambda row: (row["memory_better_ap"], -row["selected_memory_regret"]))
    checkpoint = {
        "variant": best["variant"], "model": states[best["variant"]], "layers": layers,
        "resolution": resolution, "seed": args.seed, "selection_rule": "max validation memory-better AP",
    }
    checkpoint_dir = args.output / "checkpoints"; checkpoint_dir.mkdir(exist_ok=True)
    torch.save(checkpoint, checkpoint_dir / "best_representation_selector.pt")
    torch.save(checkpoint, checkpoint_dir / "final_representation_selector.pt")
    manifest.update({
        "stage": "P0-P6 frozen representation ranking", "actual_training_backbones": args.backbones,
        "train_samples": len(train_examples), "validation_samples": len(val_examples),
        "DINO_resolution": list(resolution), "DINO_layers": layers,
        "Fast-FoundationStereo_accessed": False,
    })
    json_dump(args.output / "split_manifest.json", manifest)
    json_dump(args.output / "config.json", vars(args))
    json_dump(args.output / "aggregate_summary.json", {"stage_a_best": best, "variants": rows})
    with (args.output / "run.log").open("a") as handle:
        handle.write(f"P0-P6 ranking completed; selected {best['variant']}; Fast-FoundationStereo not accessed.\n")
    print(json.dumps(rows, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    if args.mode == "smoke":
        return smoke(args)
    if args.mode == "probe":
        return probe(args)
    return ranking(args)


if __name__ == "__main__":
    raise SystemExit(main())
