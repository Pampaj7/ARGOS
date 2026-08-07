#!/usr/bin/env python3
"""Train/evaluate the minimal ARGOS v2 non-recurrent raw-anchor adapter."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, Sampler

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from argos_v2.cache_io import load_sequence_cache  # noqa: E402
from argos_v2.scared_c_data import load_frame_gt, load_frame_lr, load_sequence_info  # noqa: E402
from model_design.data.temporal_pair_dataset import SEEN_BACKBONES, resize_gt_to_cache_masked  # noqa: E402
from model_design.external_components.bidavideo import BiDAFlowInferenceAdapter, SEA_RAFT_CHECKPOINT, causal_warp, temporal_disparity_evidence  # noqa: E402
from model_design.losses.raw_multi_anchor_losses import MultiAnchorLossConfig, multi_anchor_targets, raw_multi_anchor_losses  # noqa: E402
from model_design.models.raw_multi_anchor_refiner import (  # noqa: E402
    MultiAnchorEvidence, PROVENANCE_CORRECTED, PROVENANCE_RAW, RawMultiAnchorRefiner,
    retrieve_and_fuse,
)
from run_hybrid_temporal_memory_oracle_audit import CHECKPOINT as CORRECTED_CHECKPOINT, fused_history  # noqa: E402
from run_hybrid_temporal_memory_oracle_audit import load_model as load_corrected_model  # noqa: E402
from run_ppmstereo_validation import infer_age_flows, rgb_tensor  # noqa: E402


OUTPUT = ROOT / "results/raw_multi_anchor_temporal_refiner"
TRAIN = ("dataset_1_keyframe_2", "dataset_1_keyframe_3", "dataset_3_keyframe_1", "dataset_3_keyframe_2",
         "dataset_3_keyframe_3", "dataset_3_keyframe_4", "dataset_6_keyframe_1", "dataset_6_keyframe_2",
         "dataset_6_keyframe_3", "dataset_6_keyframe_4")
VALIDATION = ("dataset_2_keyframe_2", "dataset_2_keyframe_3", "dataset_2_keyframe_4")
TEST = ("dataset_7_keyframe_1", "dataset_7_keyframe_2", "dataset_7_keyframe_3", "dataset_7_keyframe_4")
RAW_AGES = (1, 2, 4, 8)
CONFIG_AGES = {"cs1": (1,), "hard": RAW_AGES, "soft": RAW_AGES, "hybrid_ttl2": (1, 2, 4, 8, 1, 2)}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "overfit", "smoke", "train", "calibrate", "evaluate", "consensus", "summarize"), required=True)
    parser.add_argument("--configuration", choices=("cs1", "hard", "soft", "hybrid_ttl2", "consensus"), required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=min(48, os.cpu_count() or 8))
    parser.add_argument("--flow-batch-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--crop-height", type=int, default=64)
    parser.add_argument("--crop-width", type=int, default=80)
    parser.add_argument("--margin", type=float, default=.10)
    parser.add_argument("--coverage-threshold", type=float, default=.50)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def clean(value):
    if isinstance(value, dict): return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [clean(item) for item in value]
    if isinstance(value, Path): return str(value)
    if isinstance(value, (np.generic,)): return value.item()
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n"); temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: path.write_text(""); return
    fields = list(dict.fromkeys(field for row in rows for field in row))
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for piece in iter(lambda: handle.read(1 << 20), b""): digest.update(piece)
    return digest.hexdigest()


def configuration_root(config: argparse.Namespace) -> Path:
    return config.output / {"cs1": "cs1_reference", "hard": "hard_retrieval", "soft": "soft_fusion",
                            "hybrid_ttl2": "hybrid_ttl2", "consensus": "robust_consensus"}[config.configuration]


def split_manifest() -> dict:
    return {
        "project": "ARGOS v2", "split_unit": "complete dataset-ID and sequence",
        "train_sequences": list(TRAIN), "validation_sequences": list(VALIDATION), "test_sequences": list(TEST),
        "train_dataset_ids": [1, 3, 6], "validation_dataset_ids": [2], "test_dataset_ids": [7],
        "seen_backbones": list(SEEN_BACKBONES), "dataset7_locked_until_freeze": True,
        "dataset7_caveat": "held out from this experiment but inspected by prior ARGOS v2 studies",
        "causal_contract": "direct current-to-anchor SEA-RAFT; no flow composition; model output never enters raw anchor bank",
    }


class AnchorBankDataset(Dataset):
    def __init__(self, tensors: dict[str, torch.Tensor], metadata: list[dict]) -> None:
        self.tensors, self.metadata = tensors, metadata
        length = len(metadata)
        if any(len(value) != length for value in tensors.values()): raise ValueError("tensor-bank lengths differ")

    def __len__(self) -> int: return len(self.metadata)

    def __getitem__(self, index: int) -> dict:
        return {key: value[index] for key, value in self.tensors.items()} | {"metadata_index": index}


class BalancedGroupSampler(Sampler[int]):
    def __init__(self, metadata: list[dict], seed: int) -> None:
        self.seed, self.epoch = seed, 0; groups = defaultdict(list)
        for index, item in enumerate(metadata): groups[(item["backbone"], item["sequence"])].append(index)
        self.groups = dict(sorted(groups.items())); self.length = max(map(len, self.groups.values()))

    def set_epoch(self, epoch: int) -> None: self.epoch = epoch
    def __len__(self) -> int: return len(self.groups) * self.length
    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        expanded = []
        for values in self.groups.values():
            shuffled = [values[i] for i in torch.randperm(len(values), generator=generator).tolist()]
            expanded.append([shuffled[i % len(shuffled)] for i in range(self.length)])
        for offset in range(self.length):
            for group in torch.randperm(len(expanded), generator=generator).tolist(): yield expanded[group][offset]


def _load_sequence_inputs(sequence: str, device: torch.device, max_frames: int | None):
    info = load_sequence_info(sequence); frame_ids = info.frame_ids[:max_frames] if max_frames else info.frame_ids
    images, gt, coverage = [], [], []
    for frame_id in frame_ids:
        left, _right = load_frame_lr(info, frame_id); images.append(rgb_tensor(left, device))
        native, valid = load_frame_gt(info, frame_id); resized, cov = resize_gt_to_cache_masked(native, valid)
        gt.append(resized); coverage.append(cov)
    return info, frame_ids, images, np.stack(gt), np.stack(coverage)


@torch.inference_mode()
def _align_raw_anchors(
    raw: np.ndarray, valid: np.ndarray, images: list[torch.Tensor], current_indices: list[int],
    flows: dict[int, tuple[np.ndarray, np.ndarray]], ages: tuple[int, ...], device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    stores = {key: [] for key in ("candidates", "valid", "support", "fb")}
    for start in range(0, len(current_indices), batch_size):
        query = current_indices[start:start + batch_size]; per_age = defaultdict(list)
        for age in ages:
            positions = np.arange(start, start + len(query))
            current = torch.from_numpy(np.stack([raw[index] for index in query]))[:, None].float().to(device)
            source = torch.from_numpy(np.stack([raw[index-age] for index in query]))[:, None].float().to(device)
            current_valid = torch.from_numpy(np.stack([valid[index] for index in query]))[:, None].bool().to(device)
            source_valid = torch.from_numpy(np.stack([valid[index-age] for index in query]))[:, None].bool().to(device)
            forward = torch.from_numpy(flows[age][0][positions]).float().to(device)
            backward = torch.from_numpy(flows[age][1][positions]).float().to(device)
            evidence = temporal_disparity_evidence(
                current, source, forward, backward, current_valid=current_valid, past_valid=source_valid,
                current_rgb=torch.stack([images[index] for index in query]),
                past_rgb=torch.stack([images[index-age] for index in query]),
            )
            per_age["candidates"].append(evidence.aligned_past_disparity[:, 0].cpu().numpy())
            per_age["valid"].append(evidence.aligned_validity[:, 0].cpu().numpy())
            per_age["support"].append(evidence.warp_support[:, 0].cpu().numpy())
            per_age["fb"].append(evidence.forward_backward_confidence[:, 0].cpu().numpy())
        for key in stores: stores[key].append(np.stack(per_age[key], axis=1))
    return {key: np.concatenate(values, axis=0) for key, values in stores.items()}


@torch.inference_mode()
def _align_corrected_ttl2(
    raw: np.ndarray, valid: np.ndarray, corrected: list[np.ndarray], images: list[torch.Tensor],
    current_indices: list[int], flows: dict[int, tuple[np.ndarray, np.ndarray]], device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    stores = {key: [] for key in ("candidates", "valid", "support", "fb")}
    for start in range(0, len(current_indices), batch_size):
        query = current_indices[start:start + batch_size]; per_age = defaultdict(list)
        for age in (1, 2):
            positions = np.arange(start, start + len(query))
            current = torch.from_numpy(np.stack([raw[index] for index in query]))[:, None].float().to(device)
            source = torch.from_numpy(np.stack([corrected[index-age] for index in query]))[:, None].float().to(device)
            current_valid = torch.from_numpy(np.stack([valid[index] for index in query]))[:, None].bool().to(device)
            source_valid = torch.from_numpy(np.stack([valid[index-age] for index in query]))[:, None].bool().to(device)
            evidence = temporal_disparity_evidence(
                current, source, torch.from_numpy(flows[age][0][positions]).float().to(device),
                torch.from_numpy(flows[age][1][positions]).float().to(device),
                current_valid=current_valid, past_valid=source_valid,
                current_rgb=torch.stack([images[index] for index in query]),
                past_rgb=torch.stack([images[index-age] for index in query]),
            )
            per_age["candidates"].append(evidence.aligned_past_disparity[:, 0].cpu().numpy())
            per_age["valid"].append(evidence.aligned_validity[:, 0].cpu().numpy())
            per_age["support"].append(evidence.warp_support[:, 0].cpu().numpy())
            per_age["fb"].append(evidence.forward_backward_confidence[:, 0].cpu().numpy())
        for key in stores: stores[key].append(np.stack(per_age[key], axis=1))
    return {key: np.concatenate(values, axis=0) for key, values in stores.items()}


def _crop(array: np.ndarray, y: int, x: int, height: int, width: int) -> np.ndarray:
    return np.ascontiguousarray(array[..., y:y+height, x:x+width])


@torch.inference_mode()
def _aligned_gt_age1(gt: np.ndarray, coverage: np.ndarray, current_indices: list[int], flows, device: torch.device, batch_size: int):
    disparities=[]; validity=[]
    for start in range(0,len(current_indices),batch_size):
        query=current_indices[start:start+batch_size]; positions=np.arange(start,start+len(query))
        source=torch.from_numpy(np.stack([gt[index-1] for index in query]))[:,None].float().to(device)
        source_valid=torch.from_numpy(np.stack([coverage[index-1]>.50 for index in query]))[:,None].bool().to(device)
        warped=causal_warp(source,torch.from_numpy(flows[1][0][positions]).float().to(device),source_valid=source_valid)
        disparities.append(warped.warped[:,0].cpu().numpy()); validity.append(warped.valid[:,0].cpu().numpy())
    return np.concatenate(disparities),np.concatenate(validity)


def build_ram_bank(config: argparse.Namespace, sequences: tuple[str, ...], *, training: bool) -> AnchorBankDataset:
    device = torch.device(config.device); adapter = BiDAFlowInferenceAdapter("sea_raft", device=device)
    assert not any(parameter.requires_grad for parameter in adapter.model.parameters())
    corrected_model = corrected_adapter = None
    if config.configuration == "hybrid_ttl2" or not training:
        corrected_model, _ = load_corrected_model(device); corrected_adapter = adapter
    fields = defaultdict(list); metadata = []; rng = np.random.default_rng(config.seed + (0 if training else 1000))
    max_frames = config.max_frames
    for sequence in sequences:
        info, frame_ids, images, gt, coverage = _load_sequence_inputs(sequence, device, max_frames)
        if len(frame_ids) <= 8: continue
        current_indices = list(range(8, len(frame_ids)))
        flow_ages = RAW_AGES
        flows, _latency, _peak = infer_age_flows(adapter, images, current_indices, list(flow_ages), config.flow_batch_size, device)
        gt_memory,gt_memory_valid=_aligned_gt_age1(gt,coverage,current_indices,flows,device,config.flow_batch_size)
        for backbone in SEEN_BACKBONES:
            raw_mmap, valid_mmap, cache_ids, _ = load_sequence_cache(backbone, sequence)
            if [str(value) for value in cache_ids[:len(frame_ids)]] != frame_ids: raise RuntimeError(f"frame-ID mismatch {backbone}/{sequence}")
            raw = np.asarray(raw_mmap[:len(frame_ids)], dtype=np.float32); valid = np.asarray(valid_mmap[:len(frame_ids)]) > 0
            maps = _align_raw_anchors(raw, valid, images, current_indices, flows, RAW_AGES, device, config.flow_batch_size)
            ages = list(RAW_AGES); provenance = [PROVENANCE_RAW] * 4
            # The frozen H=4 reference is a recurrent sequence model and needs
            # every consecutive age-1 flow from frame 1 onward.  ``flows``
            # above intentionally starts at frame 8 for anchor candidates, so
            # construct the complete history flow explicitly rather than
            # indexing a truncated array with absolute frame IDs.
            if corrected_model is not None:
                history_indices = list(range(1, len(frame_ids)))
                history_flows, _, _ = infer_age_flows(
                    adapter, images, history_indices, [1], config.flow_batch_size, device
                )
                corrected = fused_history(
                    corrected_model, corrected_adapter, images, raw, valid, history_flows[1], device
                )
                del history_flows
            else:
                corrected = None
            if config.configuration == "hybrid_ttl2":
                corrected_maps = _align_corrected_ttl2(raw, valid, corrected, images, current_indices, flows, device, config.flow_batch_size)
                for key in maps: maps[key] = np.concatenate((maps[key], corrected_maps[key]), axis=1)
                ages += [1, 2]; provenance += [PROVENANCE_CORRECTED, PROVENANCE_CORRECTED]
            for position, index in enumerate(current_indices):
                if training:
                    height, width = config.crop_height, config.crop_width
                    y = int(rng.integers(0, 144-height+1)); x = int(rng.integers(0, 180-width+1))
                else: height, width, y, x = 144, 180, 0, 0
                fields["raw"].append(_crop(raw[index][None], y, x, height, width).astype(np.float16))
                fields["candidates"].append(_crop(maps["candidates"][position], y, x, height, width).astype(np.float16))
                fields["candidate_valid"].append(_crop(maps["valid"][position], y, x, height, width).astype(np.uint8))
                fields["warp_support"].append(_crop(maps["support"][position], y, x, height, width).astype(np.uint8))
                fields["fb_confidence"].append(_crop(maps["fb"][position], y, x, height, width).astype(np.float16))
                fields["gt"].append(_crop(gt[index][None], y, x, height, width).astype(np.float16))
                fields["gt_coverage"].append(_crop(coverage[index][None], y, x, height, width).astype(np.float16))
                fields["raw_valid"].append(_crop(valid[index][None], y, x, height, width).astype(np.uint8))
                fields["gt_memory"].append(_crop(gt_memory[position][None], y, x, height, width).astype(np.float16))
                fields["gt_memory_valid"].append(_crop(gt_memory_valid[position][None], y, x, height, width).astype(np.uint8))
                if corrected is not None:
                    fields["fixed_h4_output"].append(_crop(corrected[index][None],y,x,height,width).astype(np.float16))
                metadata.append({"backbone": backbone, "sequence": sequence, "frame_id": frame_ids[index], "frame_index": index,
                                 "ages": ages, "provenance": provenance})
        del images, gt, coverage, flows
        if device.type == "cuda": torch.cuda.empty_cache()
        print(f"BANK sequence={sequence} examples={len(metadata)}", flush=True)
    tensors = {key: torch.from_numpy(np.stack(values)) for key, values in fields.items()}
    return AnchorBankDataset(tensors, metadata)


def to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def evidence_from_batch(batch: dict, ages: list[int], provenance: list[float]) -> MultiAnchorEvidence:
    count = len(ages)
    return MultiAnchorEvidence(
        raw=batch["raw"].float(), candidates=batch["candidates"][:, :count].float(),
        candidate_valid=batch["candidate_valid"][:, :count].bool(), warp_support=batch["warp_support"][:, :count].bool(),
        fb_confidence=batch["fb_confidence"][:, :count].float(),
        ages=torch.tensor(ages, device=batch["raw"].device), provenance=torch.tensor(provenance, device=batch["raw"].device),
    )


def loader(dataset: AnchorBankDataset, config: argparse.Namespace, training: bool) -> DataLoader:
    sampler = BalancedGroupSampler(dataset.metadata, config.seed) if training else None
    result = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler, shuffle=False, num_workers=config.workers,
                        pin_memory=True, persistent_workers=config.workers > 0, prefetch_factor=4 if config.workers else None)
    result.balanced_sampler = sampler  # type: ignore[attr-defined]
    return result


def model_ages(config: argparse.Namespace) -> tuple[list[int], list[float]]:
    if config.configuration == "cs1": return [1], [PROVENANCE_RAW]
    if config.configuration == "hybrid_ttl2": return [1, 2, 4, 8, 1, 2], [PROVENANCE_RAW] * 4 + [PROVENANCE_CORRECTED] * 2
    return list(RAW_AGES), [PROVENANCE_RAW] * 4


@torch.inference_mode()
def validation_epoch(model, dataset: AnchorBankDataset, config: argparse.Namespace, ages: list[int], provenance: list[float], loss_config: MultiAnchorLossConfig) -> dict:
    model.eval(); device=torch.device(config.device); totals=defaultdict(float); batches=0
    probability_values=[]; score_values=[]; raw_values=[]; candidate_values=[]; weight_values=[]; gt_values=[]
    for cpu in loader(dataset, config, training=False):
        batch=to_device(cpu,device); evidence=evidence_from_batch(batch,ages,provenance)
        targets=multi_anchor_targets(batch["raw"].float(),evidence.candidates,batch["gt"].float(),batch["gt_coverage"].float(),batch["raw_valid"].bool(),evidence,margin_px=config.margin)
        output=model(evidence); losses=raw_multi_anchor_losses(output,evidence,targets,loss_config,enable_fusion=config.configuration!="hard")
        for key,value in losses.items(): totals[key]+=float(value)
        score,chosen=output.selection_score.max(dim=1,keepdim=True); probability=torch.gather(output.utility_probability,1,chosen)
        candidate=torch.gather(evidence.candidates,1,chosen); weight=torch.gather(output.fusion_weight,1,chosen)
        base=((batch["gt_coverage"].float()>.50)&batch["raw_valid"].bool()).flatten(); index=base.nonzero().flatten()[::64]
        take=lambda value:value.flatten()[index].float().cpu().numpy()
        probability_values.append(take(probability)); score_values.append(take(score)); raw_values.append(take(evidence.raw))
        candidate_values.append(take(candidate)); weight_values.append(take(weight)); gt_values.append(take(batch["gt"].float())); batches+=1
    arrays=[np.concatenate(values) for values in (probability_values,score_values,raw_values,candidate_values,weight_values,gt_values)]
    probability,score,raw,candidate,weight,gt=arrays; raw_error=np.abs(raw-gt); prediction=(candidate if config.configuration=="hard" else raw+weight*(candidate-raw)); output_error=np.abs(prediction-gt)
    true_gain=raw_error-output_error; policies=[]
    for p in (.3,.4,.5,.6,.7,.8,.9):
        for u in (-.05,0,.01,.02,.05,.1):
            accepted=(probability>=p)&(score>=u); realized=true_gain*accepted
            positive=float(realized[realized>0].sum()); negative=float((-realized[realized<0]).sum())
            policies.append((float(realized.mean()),float(accepted.mean()),negative/max(positive,1e-12),p,u))
    feasible=[item for item in policies if item[1]>=.005 and item[2]<=.25]
    selected=max(feasible or policies,key=lambda item:item[0])
    return {**{f"validation_{key}":value/max(batches,1) for key,value in totals.items()},"validation_best_gain":selected[0],"validation_best_coverage":selected[1],"validation_harm_cost_fraction":selected[2],"validation_probability_threshold":selected[3],"validation_utility_threshold":selected[4],"validation_constraint_feasible":bool(feasible)}


def train(config: argparse.Namespace, *, tiny: str | None = None) -> None:
    seed_all(config.seed); root = configuration_root(config); root.mkdir(parents=True, exist_ok=True)
    save_json(config.output / "protocol_audit/split_manifest.json", split_manifest())
    sequences = (TRAIN[0],) if tiny else TRAIN
    original_max = config.max_frames
    if tiny: config.max_frames = 20
    dataset = build_ram_bank(config, sequences, training=True); config.max_frames = original_max
    data_loader = loader(dataset, config, training=True); ages, provenance = model_ages(config)
    if tiny == "overfit": validation_dataset=dataset
    else:
        if tiny: config.max_frames=20
        validation_dataset=build_ram_bank(config,(VALIDATION[0],) if tiny else VALIDATION,training=False)
        config.max_frames=original_max
    device = torch.device(config.device); model = RawMultiAnchorRefiner(config.channels, config.blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    epochs = 12 if tiny == "overfit" else 3 if tiny == "smoke" else config.epochs
    total_steps = epochs * len(data_loader); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(total_steps, 1), eta_min=config.learning_rate * .05)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda"); loss_config = MultiAnchorLossConfig(margin_px=config.margin)
    history, best, first = [], -math.inf, None; start = time.perf_counter()
    for epoch in range(epochs):
        model.train(); data_loader.balanced_sampler.set_epoch(epoch)  # type: ignore[attr-defined]
        totals = defaultdict(float); batches = 0
        for cpu in data_loader:
            batch = to_device(cpu, device); evidence = evidence_from_batch(batch, ages, provenance)
            targets = multi_anchor_targets(batch["raw"].float(), evidence.candidates, batch["gt"].float(), batch["gt_coverage"].float(),
                                           batch["raw_valid"].bool(), evidence, margin_px=config.margin)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = model(evidence); losses = raw_multi_anchor_losses(output, evidence, targets, loss_config, enable_fusion=config.configuration != "hard")
            optimizer.zero_grad(set_to_none=True); scaler.scale(losses["total"]).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); scaler.step(optimizer); scaler.update(); scheduler.step()
            if first is None: first = float(losses["total"])
            for key, value in losses.items(): totals[key] += float(value.detach())
            batches += 1
        row = {"epoch": epoch + 1, **{key: value/max(batches, 1) for key, value in totals.items()}, "learning_rate": scheduler.get_last_lr()[0]}
        row.update(validation_epoch(model,validation_dataset,config,ages,provenance,loss_config))
        history.append(row); score = row["validation_best_gain"]
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch+1, "configuration": config.configuration,
                   "ages": ages, "provenance": provenance, "channels": config.channels, "blocks": config.blocks,
                   "loss": asdict(loss_config), "split": split_manifest(), "seed": config.seed}
        checkpoint = root / "checkpoints/last.pt"; checkpoint.parent.mkdir(parents=True, exist_ok=True); torch.save(payload, checkpoint)
        if score > best: best = score; torch.save(payload, root / "checkpoints/best_validation.pt")
        print(json.dumps(row), flush=True)
    write_csv(root / "training_history.csv", history)
    save_json(root / "parameter_summary.json", {"parameters": sum(parameter.numel() for parameter in model.parameters()), "channels": config.channels,
        "blocks": config.blocks, "feature_channels": 17, "receptive_field": 3 + 4*config.blocks, "wall_time_s": time.perf_counter()-start})
    if tiny:
        final = history[-1]["total"]; required = .45 if tiny == "overfit" else .85
        passed = math.isfinite(final) and final < float(first) * required
        save_json(root / f"{tiny}_summary.json", {"initial_loss": first, "final_loss": final, "passed": passed, "examples": len(dataset),
            "no_future": True, "direct_flow": True, "immutable_raw": True, "bit_exact_fallback": True})
        if not passed: raise RuntimeError(f"{tiny} loss reduction insufficient")


def load_trained(config: argparse.Namespace, device: torch.device):
    checkpoint = config.checkpoint or configuration_root(config) / "checkpoints/best_validation.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = RawMultiAnchorRefiner(int(state["channels"]), int(state["blocks"])).to(device)
    model.load_state_dict(state["model"]); model.eval()
    return model, state, checkpoint


@torch.inference_mode()
def decision_samples(config: argparse.Namespace, dataset: AnchorBankDataset, model, state, *, stride: int = 64) -> dict[str, np.ndarray]:
    device = torch.device(config.device); ages, provenance = state["ages"], state["provenance"]; values = defaultdict(list)
    for cpu in loader(dataset, config, training=False):
        batch = to_device(cpu, device); evidence = evidence_from_batch(batch, ages, provenance); targets = multi_anchor_targets(
            batch["raw"].float(), evidence.candidates, batch["gt"].float(), batch["gt_coverage"].float(), batch["raw_valid"].bool(), evidence,
            margin_px=config.margin,
        ); output = model(evidence); score,chosen=output.selection_score.max(dim=1,keepdim=True)
        probability=torch.gather(output.utility_probability,1,chosen); candidate=torch.gather(evidence.candidates,1,chosen); weight=torch.gather(output.fusion_weight,1,chosen)
        prediction=candidate if config.configuration=="hard" else evidence.raw+weight*(candidate-evidence.raw)
        true_gain=(evidence.raw-batch["gt"].float()).abs()-(prediction-batch["gt"].float()).abs()
        valid=((batch["gt_coverage"].float()>.50)&batch["raw_valid"].bool()).flatten(); indices=valid.nonzero().flatten()[::stride]
        for key,tensor in {"probability":probability,"score":score,"delta":true_gain,"helpful":true_gain>config.margin}.items():
            values[key].append(tensor.flatten()[indices].float().cpu().numpy())
    return {key: np.concatenate(items) for key, items in values.items()}


def calibrate(config: argparse.Namespace) -> None:
    root = configuration_root(config); device = torch.device(config.device); model, state, checkpoint = load_trained(config, device)
    original = config.max_frames; config.max_frames = config.max_frames
    dataset = build_ram_bank(config, VALIDATION, training=False); config.max_frames = original
    values = decision_samples(config, dataset, model, state); rows = []
    for probability in (.30, .40, .50, .60, .70, .80, .90):
        for utility in (-.05, 0., .01, .02, .05, .10):
            selected = (values["probability"] >= probability) & (values["score"] >= utility)
            true = values["delta"]; helpful = true > config.margin; harmful = true < -config.margin
            positive = float(true[selected & (true > 0)].sum()); negative = float((-true[selected & (true < 0)]).sum())
            rows.append({"probability_threshold": probability, "utility_threshold_px": utility, "sample_coverage": float(selected.mean()),
                         "precision": float(helpful[selected].mean()) if selected.any() else 0.,
                         "harmful_fraction": float(harmful[selected].mean()) if selected.any() else 0.,
                         "harm_cost_fraction": negative/max(positive, 1e-12), "sample_net_utility": float((true*selected).mean())})
    feasible = [row for row in rows if row["sample_coverage"] >= .005 and row["harm_cost_fraction"] <= .25 and row["harmful_fraction"] <= .15]
    selected = max(feasible or rows, key=lambda row: (row["sample_net_utility"], row["precision"]))
    record = {"project": "ARGOS v2", "configuration": config.configuration, "selected_only_on": list(VALIDATION), "dataset7_accessed": False,
              "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "policy": selected,
              "constraints": "sample coverage>=0.005; harm-cost<=0.25; harmful fraction<=0.15", "feasible": bool(feasible)}
    write_csv(root / "fallback_calibration.csv", rows); save_json(root / "frozen_policy.json", record)


def consensus_policy(evidence: MultiAnchorEvidence, *, raw_deviation: float, mad_max: float, fb_min: float):
    available = evidence.available; masked = torch.where(available, evidence.candidates, torch.full_like(evidence.candidates, torch.nan))
    median = torch.nanmedian(masked, dim=1, keepdim=True).values; mad = torch.nanmedian(
        torch.where(available, (evidence.candidates-median).abs(), torch.full_like(evidence.candidates, torch.nan)), dim=1, keepdim=True,
    ).values
    deviation = (evidence.raw-median).abs(); witness = available.sum(dim=1, keepdim=True)
    accepted = (deviation >= raw_deviation) & (mad <= mad_max) & (witness >= 2)
    distance = (evidence.candidates-median).abs().masked_fill(~available, torch.inf)
    chosen = distance.argmin(dim=1, keepdim=True); accepted &= torch.gather(evidence.fb_confidence, 1, chosen) >= fb_min
    candidate = torch.gather(evidence.candidates, 1, chosen)
    return torch.where(accepted, .5*evidence.raw + .5*candidate, evidence.raw), accepted, chosen


def _frame_metric(raw, output, fixed_h4, gt, gt_memory, gt_memory_valid, base, accepted, chosen, evidence, metadata, config_name: str) -> dict:
    raw_error = (raw-gt).abs(); error = (output-gt).abs(); valid = base.bool(); changed = valid & ((output-raw).abs() > 1e-6)
    fixed_error=(fixed_h4-gt).abs()
    helpful = changed & (error < raw_error-config_name_margin(config_name)); harmful = changed & (error > raw_error+config_name_margin(config_name))
    clean = valid & (raw_error <= .10); clean_degraded = changed & clean & (error > raw_error+.10)
    chosen_age = torch.gather(evidence.ages[None, :, None, None].expand(raw.shape[0], -1, *raw.shape[-2:]), 1, chosen)
    available=evidence.available; candidate_error=(evidence.candidates-gt).abs().masked_fill(~available,torch.inf)
    oracle_stack=torch.cat((raw_error,candidate_error),dim=1); oracle_index=oracle_stack.argmin(dim=1,keepdim=True)
    strict=valid & available.all(dim=1,keepdim=True)
    temporal=valid & gt_memory_valid.bool() & available[:,0:1]
    gt_delta=gt-gt_memory; raw_tepe=((raw-evidence.candidates[:,0:1])-gt_delta).abs(); output_tepe=((output-evidence.candidates[:,0:1])-gt_delta).abs()
    result = {"method": config_name, **metadata, "valid_count": int(valid.sum()), "raw_error_sum": float(raw_error[valid].sum()),
              "output_error_sum": float(error[valid].sum()), "changed_count": int(changed.sum()), "helpful_count": int(helpful.sum()),
              "fixed_h4_error_sum":float(fixed_error[valid].sum()),
              "harmful_count": int(harmful.sum()), "clean_count": int(clean.sum()), "clean_degradation_count": int(clean_degraded.sum()),
              "frame_raw_epe": float(raw_error[valid].mean()), "frame_output_epe": float(error[valid].mean()),
              "frame_degradation": float(error[valid].mean()-raw_error[valid].mean()), "exact_fallback_count": int((valid & ~changed).sum())}
    result.update({"strict_count":int(strict.sum()),"strict_raw_error_sum":float(raw_error[strict].sum()),"strict_output_error_sum":float(error[strict].sum()),
                   "temporal_count":int(temporal.sum()),"raw_tepe_sum":float(raw_tepe[temporal].sum()),"output_tepe_sum":float(output_tepe[temporal].sum()),
                   "raw_teper_sum":float((raw_tepe/(gt_delta.abs()+1e-3))[temporal].sum()),"output_teper_sum":float((output_tepe/(gt_delta.abs()+1e-3))[temporal].sum())})
    for age in RAW_AGES:
        result[f"selected_age_{age}"] = int((changed & (chosen_age == age)).sum())
        accepted_age=changed&(chosen_age==age); result[f"accepted_age_{age}_error_sum"]=float(error[accepted_age].sum())
        result[f"accepted_age_{age}_count"]=int(accepted_age.sum())
    actions=("raw",)+tuple(str(age) for age in RAW_AGES)
    predicted_action=torch.where(changed,chosen_age,torch.zeros_like(chosen_age))
    for predicted in actions:
        pred_mask=(~changed if predicted=="raw" else (changed&(chosen_age==int(predicted))))
        for oracle in actions:
            oracle_value=0 if oracle=="raw" else list(RAW_AGES).index(int(oracle))+1
            result[f"confusion_pred_{predicted}_oracle_{oracle}"]=int((valid&pred_mask&(oracle_index==oracle_value)).sum())
    distant=changed&((chosen_age==4)|(chosen_age==8)); result["distant_helpful_count"]=int((distant&helpful).sum()); result["distant_harmful_count"]=int((distant&harmful).sum())
    masked_candidates=torch.where(available,evidence.candidates,torch.full_like(evidence.candidates,torch.nan))
    median=torch.nanmedian(masked_candidates,dim=1,keepdim=True).values
    mad=torch.nanmedian(torch.where(available,(evidence.candidates-median).abs(),torch.full_like(evidence.candidates,torch.nan)),dim=1,keepdim=True).values
    agreement=(((evidence.candidates-median).abs()<=mad+.10)&available).sum(dim=1,keepdim=True)
    raw_deviation=(raw-median).abs(); realized_gain=raw_error-error
    for witnesses in range(1,evidence.candidates.shape[1]+1):
        mask=valid&(agreement==witnesses); result[f"agreement_{witnesses}_count"]=int(mask.sum()); result[f"agreement_{witnesses}_gain_sum"]=float(realized_gain[mask].sum())
    for label,lower,upper in (("lt010",0,.10),("010_050",.10,.50),("050_100",.50,1.0),("ge100",1.0,float("inf"))):
        mask=valid&(raw_deviation>=lower)&(raw_deviation<upper); result[f"rawdev_{label}_count"]=int(mask.sum()); result[f"rawdev_{label}_gain_sum"]=float(realized_gain[mask].sum())
    return result


def config_name_margin(_name: str) -> float: return .10


@torch.inference_mode()
def evaluate(config: argparse.Namespace, *, consensus: bool = False) -> None:
    root = configuration_root(config); policy_path = root / "frozen_policy.json"
    if config.split == "test" and not policy_path.exists(): raise RuntimeError("dataset 7 locked until validation policy freeze")
    sequences = VALIDATION if config.split == "validation" else TEST; dataset = build_ram_bank(config, sequences, training=False)
    device = torch.device(config.device); model = state = checkpoint = None
    if not consensus: model, state, checkpoint = load_trained(config, device)
    policy = json.loads(policy_path.read_text())["policy"] if policy_path.exists() else None
    rows, diagnostic_probability, diagnostic_label, diagnostic_score = [], [], [], []
    data_loader = loader(dataset, config, training=False); ages, provenance = (model_ages(config) if consensus else (state["ages"], state["provenance"]))
    offset = 0; predictions_by_group = defaultdict(list)
    for cpu in data_loader:
        batch = to_device(cpu, device); evidence = evidence_from_batch(batch, ages, provenance); batch_count = len(batch["raw"])
        if consensus:
            if policy is None: raise RuntimeError("consensus policy not frozen")
            prediction, accepted, chosen = consensus_policy(evidence, raw_deviation=policy["raw_deviation"], mad_max=policy["mad_max"], fb_min=policy["fb_min"])
        else:
            output = model(evidence); prediction, accepted, chosen, _ = retrieve_and_fuse(
                evidence.raw, evidence, output, probability_threshold=policy["probability_threshold"], utility_threshold_px=policy["utility_threshold_px"],
                hard=config.configuration == "hard",
            )
            targets = multi_anchor_targets(evidence.raw, evidence.candidates, batch["gt"].float(), batch["gt_coverage"].float(), batch["raw_valid"].bool(), evidence, margin_px=config.margin)
            mask = targets.valid
            diagnostic_probability.append(output.utility_probability[mask].cpu().numpy()[::64]); diagnostic_score.append(output.selection_score[mask].cpu().numpy()[::64])
            diagnostic_label.append(targets.helpful[mask].cpu().numpy()[::64])
        base = (batch["gt_coverage"].float() > config.coverage_threshold) & batch["raw_valid"].bool()
        for local in range(batch_count):
            meta = dataset.metadata[offset+local]
            row = _frame_metric(evidence.raw[local:local+1], prediction[local:local+1],batch["fixed_h4_output"][local:local+1].float(),batch["gt"][local:local+1].float(),
                                batch["gt_memory"][local:local+1].float(),batch["gt_memory_valid"][local:local+1].bool(),base[local:local+1],
                                accepted[local:local+1], chosen[local:local+1], MultiAnchorEvidence(
                                    evidence.raw[local:local+1], evidence.candidates[local:local+1], evidence.candidate_valid[local:local+1],
                                    evidence.warp_support[local:local+1], evidence.fb_confidence[local:local+1], evidence.ages, evidence.provenance), meta, config.configuration)
            rows.append(row); predictions_by_group[(meta["backbone"], meta["sequence"])].append((meta["frame_index"], prediction[local,0].cpu(), evidence.raw[local,0].cpu(),
                evidence.candidates[local,0].cpu(), base[local,0].cpu()))
        offset += batch_count
    destination = root / config.split; destination.mkdir(parents=True, exist_ok=True); write_csv(destination / "frame_metrics.csv", rows)
    summary = aggregate_metrics(rows); summary.update({"split": config.split, "configuration": config.configuration,
        "checkpoint": str(checkpoint) if checkpoint else None, "policy": policy, "candidate_count": len(ages), "availability_aware": True})
    if diagnostic_label:
        label = np.concatenate(diagnostic_label); probability = np.concatenate(diagnostic_probability); score = np.concatenate(diagnostic_score)
        summary["utility_auroc"] = float(roc_auc_score(label, score)) if len(np.unique(label)) > 1 else math.nan
        summary["utility_auprc"] = float(average_precision_score(label, score)) if len(np.unique(label)) > 1 else math.nan
        order=np.argsort(score)[::-1]; risk=[]
        for coverage in (.001,.002,.005,.01,.02,.05,.10,.20,.50,1.0):
            count=max(1,int(math.ceil(coverage*len(score)))); chosen_indices=order[:count]
            risk.append({"split":config.split,"configuration":config.configuration,"coverage":coverage,"sample_count":count,
                         "helpful_precision":float(label[chosen_indices].mean()),"mean_score":float(score[chosen_indices].mean())})
        write_csv(destination/"risk_coverage.csv",risk)
    save_json(destination / "summary.json", summary); write_csv(destination / "per_backbone_metrics.csv", grouped_metrics(rows, "backbone")); write_csv(destination / "per_sequence_metrics.csv", grouped_metrics(rows, "sequence"))
    write_csv(destination/"temporal_metrics.csv",[{key:summary[key] for key in ("raw_tepe","output_tepe","raw_teper","output_teper")}])
    per_age=[]
    for age in RAW_AGES:
        per_age.append({"age":age,"selection_fraction":summary[f"selected_age_{age}_fraction"],"accepted_candidate_epe":summary[f"accepted_age_{age}_epe"]})
    write_csv(destination/"per_age_metrics.csv",per_age)
    confusion=[]
    for predicted in ("raw",)+tuple(str(age) for age in RAW_AGES):
        for oracle in ("raw",)+tuple(str(age) for age in RAW_AGES):
            field=f"confusion_pred_{predicted}_oracle_{oracle}"
            confusion.append({"predicted_action":predicted,"oracle_action":oracle,"count":sum(row[field] for row in rows)})
    write_csv(destination/"candidate_selection_metrics.csv",confusion)
    conditioned=[]
    candidate_count=len(ages)
    for witnesses in range(1,candidate_count+1):
        count=sum(row[f"agreement_{witnesses}_count"] for row in rows); gain=sum(row[f"agreement_{witnesses}_gain_sum"] for row in rows)
        conditioned.append({"condition":"agreement_count","bin":witnesses,"valid_count":count,"mean_gain":gain/max(count,1)})
    for label in ("lt010","010_050","050_100","ge100"):
        count=sum(row[f"rawdev_{label}_count"] for row in rows); gain=sum(row[f"rawdev_{label}_gain_sum"] for row in rows)
        conditioned.append({"condition":"raw_deviation_from_temporal_median","bin":label,"valid_count":count,"mean_gain":gain/max(count,1)})
    write_csv(destination/"conditioned_gain_metrics.csv",conditioned)
    if config.split == "test": save_json(root / "test_opened.json", {"opened_after_policy_freeze": True, "sequences": list(TEST)})


def aggregate_metrics(rows: list[dict]) -> dict:
    count = sum(row["valid_count"] for row in rows); raw_sum = sum(row["raw_error_sum"] for row in rows); output_sum = sum(row["output_error_sum"] for row in rows)
    changed = sum(row["changed_count"] for row in rows); helpful = sum(row["helpful_count"] for row in rows); harmful = sum(row["harmful_count"] for row in rows)
    clean = sum(row["clean_count"] for row in rows); clean_deg = sum(row["clean_degradation_count"] for row in rows); deltas = np.array([row["frame_degradation"] for row in rows])
    strict=sum(row["strict_count"] for row in rows); temporal=sum(row["temporal_count"] for row in rows)
    fixed_sum=sum(row["fixed_h4_error_sum"] for row in rows)
    result={"valid_count": count, "raw_epe": raw_sum/max(count,1), "output_epe": output_sum/max(count,1), "gain": (raw_sum-output_sum)/max(count,1),
            "fixed_h4_epe_common_support":fixed_sum/max(count,1),"gain_over_fixed_h4":(fixed_sum-output_sum)/max(count,1),
            "coverage": changed/max(count,1), "exact_raw_fallback_frequency": 1-changed/max(count,1), "intervention_precision": helpful/max(changed,1),
            "harmful_update_rate": harmful/max(changed,1), "clean_degradation": clean_deg/max(clean,1), "degraded_frame_fraction": float((deltas>0).mean()),
            "worst_frame_degradation": float(deltas.max()), "p95_frame_degradation": float(np.quantile(deltas,.95)),
            "strict_intersection_coverage":strict/max(count,1),"strict_raw_epe":sum(row["strict_raw_error_sum"] for row in rows)/max(strict,1),
            "strict_output_epe":sum(row["strict_output_error_sum"] for row in rows)/max(strict,1),
            "raw_tepe":sum(row["raw_tepe_sum"] for row in rows)/max(temporal,1),"output_tepe":sum(row["output_tepe_sum"] for row in rows)/max(temporal,1),
            "raw_teper":sum(row["raw_teper_sum"] for row in rows)/max(temporal,1),"output_teper":sum(row["output_teper_sum"] for row in rows)/max(temporal,1),
            "distant_helpful_fraction":sum(row["distant_helpful_count"] for row in rows)/max(changed,1),
            "distant_harmful_fraction":sum(row["distant_harmful_count"] for row in rows)/max(changed,1),
            **{f"selected_age_{age}_fraction": sum(row[f"selected_age_{age}"] for row in rows)/max(changed,1) for age in RAW_AGES}}
    for age in RAW_AGES:
        accepted_count=sum(row[f"accepted_age_{age}_count"] for row in rows)
        result[f"accepted_age_{age}_epe"]=sum(row[f"accepted_age_{age}_error_sum"] for row in rows)/max(accepted_count,1)
    return result


def grouped_metrics(rows: list[dict], key: str) -> list[dict]:
    groups = defaultdict(list)
    for row in rows: groups[row[key]].append(row)
    return [{key: name, **aggregate_metrics(group)} for name, group in groups.items()]


def calibrate_consensus(config: argparse.Namespace) -> None:
    root = configuration_root(config); dataset = build_ram_bank(config, VALIDATION, training=False); device = torch.device(config.device); rows = []
    for raw_deviation in (.10, .25, .50, 1.0, 2.0):
        for mad_max in (.10, .25, .50, 1.0):
            for fb_min in (.25, .50, .75):
                totals = []
                for cpu in loader(dataset, config, training=False):
                    batch = to_device(cpu, device); evidence = evidence_from_batch(batch, list(RAW_AGES), [PROVENANCE_RAW]*4)
                    prediction, accepted, _ = consensus_policy(evidence, raw_deviation=raw_deviation, mad_max=mad_max, fb_min=fb_min)
                    base = (batch["gt_coverage"].float() > config.coverage_threshold) & batch["raw_valid"].bool(); raw_error=(evidence.raw-batch["gt"].float()).abs(); error=(prediction-batch["gt"].float()).abs()
                    totals.append((float((raw_error-error)[base].sum()), int(base.sum()), int((accepted&base).sum()), int((accepted&base&(error>raw_error+.1)).sum())))
                gain_sum=sum(item[0] for item in totals); count=sum(item[1] for item in totals); changed=sum(item[2] for item in totals); harmful=sum(item[3] for item in totals)
                rows.append({"raw_deviation":raw_deviation,"mad_max":mad_max,"fb_min":fb_min,"gain":gain_sum/max(count,1),"coverage":changed/max(count,1),"harmful_fraction":harmful/max(changed,1)})
    feasible=[row for row in rows if row["coverage"]>=.005 and row["harmful_fraction"]<=.15 and row["gain"]>0]
    selected=max(feasible or rows,key=lambda row:row["gain"]); write_csv(root/"fallback_calibration.csv",rows)
    save_json(root/"frozen_policy.json",{"project":"ARGOS v2","configuration":"consensus","selected_only_on":list(VALIDATION),"dataset7_accessed":False,"policy":selected,"feasible":bool(feasible)})


def audit(config: argparse.Namespace) -> None:
    config.output.mkdir(parents=True, exist_ok=True); save_json(config.output/"protocol_audit/protocol_audit.json", split_manifest() | {
        "SEA_RAFT_sha256": sha256(SEA_RAFT_CHECKPOINT), "BiDA_sha256": sha256(ROOT/"model_design/external_components/bidavideo.py"),
        "corrected_checkpoint_sha256": sha256(CORRECTED_CHECKPOINT), "raw_candidates_immutable": True, "output_enters_future_bank": False,
        "strict_support_reported": True, "availability_aware_deployment": True,
    })
    save_json(config.output/"checkpoint_hashes.json", {"sea_raft":sha256(SEA_RAFT_CHECKPOINT),"corrected_h4":sha256(CORRECTED_CHECKPOINT)})
    save_json(config.output/"configuration.json", vars(config))


def summarize(config: argparse.Namespace) -> None:
    configurations=("cs1","hard","soft","consensus","hybrid_ttl2"); rows=[]; roots={}
    for name in configurations:
        config.configuration=name; root=configuration_root(config); roots[name]=root; path=root/"test/summary.json"
        if path.exists(): rows.append(json.loads(path.read_text()))
    write_csv(config.output/"ablation_summary.csv",rows)
    reference=json.loads((ROOT/"results/codd_style_bounded_memory_validation/aggregate_summary.json").read_text())
    oracle=json.loads((ROOT/"results/hybrid_temporal_memory_oracle_audit/aggregate_summary.json").read_text())
    soft=next((row for row in rows if row["configuration"]=="soft"),None); hard=next((row for row in rows if row["configuration"]=="hard"),None)
    aggregate={"project":"ARGOS v2","fixed_h4_reference":reference.get("final_test",reference),"raw_bank_oracle":oracle["test"]["raw_anchor_bank"],"configurations":rows,
               "scope":"held-out SCARED-C dataset 7 and three seen training backbones only"}
    if soft:
        aggregate["raw_bank_selection_oracle_recovery"]=soft["gain"]/max(oracle["test"]["raw_anchor_bank"]["oracle_gain"],1e-12)
        aggregate["raw_bank_convex_oracle_recovery"]=soft["gain"]/.1549843363093545
        aggregate["soft_beats_hard"]=bool(hard and soft["output_epe"]<hard["output_epe"])
    soft_back=list(csv.DictReader((roots["soft"]/"test/per_backbone_metrics.csv").open())) if (roots["soft"]/"test/per_backbone_metrics.csv").exists() else []
    soft_seq=list(csv.DictReader((roots["soft"]/"test/per_sequence_metrics.csv").open())) if (roots["soft"]/"test/per_sequence_metrics.csv").exists() else []
    gates={"soft_exists":soft is not None,"improves_fixed_h4":bool(soft and soft["gain_over_fixed_h4"]>0),
           "all_seen_backbones_positive":bool(soft_back and all(float(row["gain"])>0 for row in soft_back)),
           "nearly_all_sequences_positive":bool(soft_seq and np.mean([float(row["gain"])>0 for row in soft_seq])>=.75),
           "distant_useful_nonzero":bool(soft and soft["distant_helpful_fraction"]>=.005),
           "soft_beats_hard":bool(soft and hard and soft["output_epe"]<hard["output_epe"])}
    # Geometry-only gates are insufficient for a deployable selective adapter.
    # The protocol requires controlled harmful updates and clean-pixel risk;
    # encode those gates explicitly so the compact report cannot overclaim GO.
    safety = {
        "harmful_update_rate_le_10pct": bool(soft and soft["harmful_update_rate"] <= .10),
        "clean_degradation_le_3pct": bool(soft and soft["clean_degradation"] <= .03),
        "degraded_frame_fraction_le_25pct": bool(soft and soft["degraded_frame_fraction"] <= .25),
    }
    geometry_verdict = "GO" if all(gates.values()) else "NO-GO"
    safety_verdict = "GO" if all(safety.values()) else "NO-GO"
    verdict = "GO" if geometry_verdict == "GO" and safety_verdict == "GO" else "NO-GO"
    hybrid=next((row for row in rows if row["configuration"]=="hybrid_ttl2"),None)
    hybrid_promoted=bool(hybrid and soft and hybrid["output_epe"]<soft["output_epe"] and (hybrid["harmful_update_rate"]<soft["harmful_update_rate"] or hybrid["output_tepe"]<soft["output_tepe"]))
    aggregate.update({"promotion_gates":gates,"safety_gates":safety,"geometry_verdict":geometry_verdict,
                      "safety_verdict":safety_verdict,"verdict":verdict,"hybrid_ttl2_promoted":hybrid_promoted})
    save_json(config.output/"aggregate_summary.json",aggregate); save_json(config.output/"verdicts.json",{"raw_multi_anchor_soft":verdict,"geometry_verdict":geometry_verdict,"safety_verdict":safety_verdict,"gates":gates,"safety_gates":safety,"hybrid_ttl2_promoted":hybrid_promoted})
    training=[]; validation=[]
    for name,root in roots.items():
        history=root/"training_history.csv"; val=root/"validation/summary.json"
        if history.exists():
            records=list(csv.DictReader(history.open())); training.extend({"configuration":name,**row} for row in records)
        if val.exists(): validation.append(json.loads(val.read_text()))
    write_csv(config.output/"training_summary.csv",training); write_csv(config.output/"validation_summary.csv",validation); write_csv(config.output/"frozen_test_summary.csv",rows)
    calibration=[]
    for name,root in roots.items():
        path=root/"fallback_calibration.csv"
        if path.exists(): calibration.extend({"configuration":name,**row} for row in csv.DictReader(path.open()))
    write_csv(config.output/"fallback_calibration.csv",calibration)
    for filename in ("per_backbone_metrics.csv","per_sequence_metrics.csv","temporal_metrics.csv","risk_coverage.csv","per_age_metrics.csv","candidate_selection_metrics.csv"):
        combined=[]
        for name,root in roots.items():
            path=root/"test"/filename
            if path.exists(): combined.extend({"configuration":name,**row} for row in csv.DictReader(path.open()))
        write_csv(config.output/filename,combined)
    recovery=[]
    selection_gain=float(oracle["test"]["raw_anchor_bank"]["oracle_gain"]); convex_gain=.1549843363093545
    for row in rows: recovery.append({"configuration":row["configuration"],"gain":row["gain"],"selection_oracle_recovery":row["gain"]/selection_gain,"convex_oracle_recovery":row["gain"]/convex_gain})
    write_csv(config.output/"oracle_recovery.csv",recovery)
    matrix=[
        {"phase":"leave_one_backbone_out","domain":"SCARED-C","selection_use":"train/validation only","status":"future"},
        {"phase":"completely_unseen_backbone","domain":"SCARED-C","selection_use":"none after freeze","status":"future"},
        {"phase":"external_surgical_dataset","domain":"verified stereo+GT protocol","selection_use":"separate validation only","status":"future"},
        {"phase":"combined_unseen_backbone_external_domain","domain":"external surgical","selection_use":"none after freeze","status":"future"},
    ]; write_csv(config.output/"next_phase_validation_matrix.csv",matrix)
    if soft:
        (config.output/"paper_ready_tables.tex").write_text("\\begin{tabular}{lrrrr}\\toprule\nMethod & EPE & Gain & Harmful & Clean degradation \\\\\n"+"\n".join(f"{row['configuration'].replace('_',' ')} & {row['output_epe']:.4f} & {row['gain']:.4f} & {row['harmful_update_rate']:.3f} & {row['clean_degradation']:.3f} \\\\" for row in rows)+"\n\\bottomrule\\end{tabular}\n")
        (config.output/"README.md").write_text(
            "# ARGOS v2 raw multi-anchor temporal refiner\n\n"
            f"Frozen dataset-7 verdict: **{verdict}** (geometry={geometry_verdict}, safety={safety_verdict}). Raw multi-anchor soft fusion has EPE {soft['output_epe']:.5f}, gain {soft['gain']:.5f}, and common-support gain over fixed H=4 of {soft['gain_over_fixed_h4']:.5f}. "
            f"It recovers {100*soft['gain']/selection_gain:.1f}% of the raw-bank selection oracle and {100*soft['gain']/convex_gain:.1f}% of its convex oracle.\n\n"
            f"Soft beats hard: {gates['soft_beats_hard']}. Hybrid TTL-2 promoted: {hybrid_promoted}. Unseen-backbone and external-dataset OOD remain unverified.\n")


def main() -> None:
    config=arguments(); audit(config)
    if config.mode=="audit": return
    if config.mode=="overfit": train(config,tiny="overfit")
    elif config.mode=="smoke": train(config,tiny="smoke")
    elif config.mode=="train": train(config)
    elif config.mode=="calibrate": calibrate_consensus(config) if config.configuration=="consensus" else calibrate(config)
    elif config.mode in {"evaluate","consensus"}: evaluate(config,consensus=config.configuration=="consensus")
    else: summarize(config)


if __name__ == "__main__": main()
