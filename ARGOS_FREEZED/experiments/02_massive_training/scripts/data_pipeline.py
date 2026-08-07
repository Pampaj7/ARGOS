"""Canonical ARGOS v2 train/validation evidence construction without ARGOS-V2 imports."""
from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from argos_freezed.alignment.bida_pull_warp import causal_warp, temporal_disparity_evidence
from argos_freezed.alignment.sea_raft_adapter import SEARAFTFlowAdapter
from argos_freezed.models.raw_multi_anchor_refiner import MultiAnchorEvidence
from campaign_common import ANCHOR_AGES, SEEN_BACKBONES, guard_no_d7

ARGOS_ROOT = Path("/dtu/p1/leopam/ARGOS")
CACHE_DIR = ARGOS_ROOT / "ARGOS-V2/cache_scaredc_backbones"
SEQUENCE_DIR = ARGOS_ROOT / "dataset/SCARED-C/curated/geometric_gt/corrected_temporal_gt"
CACHE_HEIGHT, CACHE_WIDTH = 144, 180


@dataclass(frozen=True)
class SequenceInfo:
    sequence_id: str
    seq_dir: Path
    frame_ids: list[str]
    fx: float
    baseline_mm: float


def load_sequence_info(sequence_id: str) -> SequenceInfo:
    guard_no_d7(sequence_id)
    seq_dir = SEQUENCE_DIR / sequence_id
    rows = list(csv.DictReader((seq_dir / "manifest.csv").open()))
    if not rows: raise RuntimeError(f"empty manifest for {sequence_id}")
    return SequenceInfo(sequence_id, seq_dir, [row["frame_id"] for row in rows], float(rows[0]["fx"]), float(rows[0]["baseline_mm"]))


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None: raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_frame_left(info: SequenceInfo, frame_id: str) -> np.ndarray:
    return read_rgb(info.seq_dir / "left" / f"{frame_id}.png")


def load_frame_gt(info: SequenceInfo, frame_id: str) -> tuple[np.ndarray, np.ndarray]:
    directory = info.seq_dir / "gt"
    disparity = np.load(directory / f"{frame_id}_disp.npy").astype(np.float32)
    valid = cv2.imread(str(directory / f"{frame_id}_valid.png"), cv2.IMREAD_GRAYSCALE) > 0
    return disparity, valid


def load_sequence_cache(backbone: str, sequence_id: str):
    guard_no_d7(sequence_id)
    directory = CACHE_DIR / backbone / sequence_id
    metadata = json.loads((directory / "metadata.json").read_text())
    return (np.load(directory / "disparity.npy", mmap_mode="r"), np.load(directory / "valid_mask.npy", mmap_mode="r"),
            np.load(directory / "frame_ids.npy"), metadata)


def resize_gt_to_cache_masked(disparity: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coverage = cv2.resize(valid.astype(np.float32), (CACHE_WIDTH, CACHE_HEIGHT), interpolation=cv2.INTER_AREA)
    numerator = cv2.resize(disparity.astype(np.float32) * valid.astype(np.float32), (CACHE_WIDTH, CACHE_HEIGHT), interpolation=cv2.INTER_AREA)
    cache = numerator / np.maximum(coverage, 1e-6); cache *= CACHE_WIDTH / disparity.shape[1]
    return cache.astype(np.float32), coverage


def rgb_tensor(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    resized = cv2.resize(rgb, (CACHE_WIDTH, CACHE_HEIGHT), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(np.ascontiguousarray(resized)).permute(2, 0, 1).float().to(device)


def infer_age_flows(adapter, images, current_indices, ages, batch_size, device):
    flows, latency = {}, {}
    for age in ages:
        forward, backward, elapsed = [], [], 0.0
        for start in range(0, len(current_indices), batch_size):
            indices = current_indices[start:start + batch_size]
            current = torch.stack([images[index] for index in indices]); past = torch.stack([images[index-age] for index in indices])
            if device.type == "cuda": torch.cuda.synchronize(device)
            tick = time.perf_counter(); inferred = adapter.infer(torch.cat((current, past)), torch.cat((past, current)))
            if device.type == "cuda": torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - tick; count = len(indices)
            forward.extend(inferred[:count].cpu().numpy().astype(np.float32)); backward.extend(inferred[count:].cpu().numpy().astype(np.float32))
        flows[age] = (np.stack(forward), np.stack(backward)); latency[age] = elapsed * 1000 / max(len(current_indices), 1)
    return flows, latency


class AnchorBankDataset(Dataset):
    def __init__(self, tensors: dict[str, torch.Tensor], metadata: list[dict]) -> None:
        self.tensors, self.metadata = tensors, metadata
        if any(len(value) != len(metadata) for value in tensors.values()): raise ValueError("tensor-bank lengths differ")
    def __len__(self): return len(self.metadata)
    def __getitem__(self, index): return {key: value[index] for key, value in self.tensors.items()} | {"metadata_index": index}


class BalancedGroupSampler(Sampler[int]):
    def __init__(self, metadata, seed):
        self.seed, self.epoch = seed, 0; groups = defaultdict(list)
        for index, item in enumerate(metadata): groups[(item["backbone"], item["sequence"])].append(index)
        self.groups = dict(sorted(groups.items())); self.length = max(map(len, self.groups.values()))
    def set_epoch(self, epoch): self.epoch = epoch
    def __len__(self): return len(self.groups) * self.length
    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch); expanded = []
        for values in self.groups.values():
            shuffled = [values[i] for i in torch.randperm(len(values), generator=generator).tolist()]
            expanded.append([shuffled[i % len(shuffled)] for i in range(self.length)])
        for offset in range(self.length):
            for group in torch.randperm(len(expanded), generator=generator).tolist(): yield expanded[group][offset]


def _load_inputs(sequence: str, device: torch.device, max_frames: int | None):
    info = load_sequence_info(sequence); frame_ids = info.frame_ids[:max_frames] if max_frames else info.frame_ids
    images, gt, coverage = [], [], []
    for frame_id in frame_ids:
        images.append(rgb_tensor(load_frame_left(info, frame_id), device))
        native, valid = load_frame_gt(info, frame_id); resized, cov = resize_gt_to_cache_masked(native, valid)
        gt.append(resized); coverage.append(cov)
    return info, frame_ids, images, np.stack(gt), np.stack(coverage)


@torch.inference_mode()
def _align(raw, valid, images, current_indices, flows, device, batch_size):
    stores = {key: [] for key in ("candidates", "candidate_valid", "warp_support", "fb_confidence")}
    for start in range(0, len(current_indices), batch_size):
        query = current_indices[start:start + batch_size]; per_age = defaultdict(list)
        for age in ANCHOR_AGES:
            positions = np.arange(start, start + len(query))
            current = torch.from_numpy(np.stack([raw[index] for index in query]))[:, None].float().to(device)
            source = torch.from_numpy(np.stack([raw[index-age] for index in query]))[:, None].float().to(device)
            current_valid = torch.from_numpy(np.stack([valid[index] for index in query]))[:, None].bool().to(device)
            source_valid = torch.from_numpy(np.stack([valid[index-age] for index in query]))[:, None].bool().to(device)
            evidence = temporal_disparity_evidence(current, source,
                torch.from_numpy(flows[age][0][positions]).float().to(device), torch.from_numpy(flows[age][1][positions]).float().to(device),
                current_valid=current_valid, past_valid=source_valid, current_rgb=torch.stack([images[index] for index in query]),
                past_rgb=torch.stack([images[index-age] for index in query]))
            per_age["candidates"].append(evidence.aligned_past_disparity[:, 0].cpu().numpy())
            per_age["candidate_valid"].append(evidence.aligned_validity[:, 0].cpu().numpy())
            per_age["warp_support"].append(evidence.warp_support[:, 0].cpu().numpy())
            per_age["fb_confidence"].append(evidence.forward_backward_confidence[:, 0].cpu().numpy())
        for key in stores: stores[key].append(np.stack(per_age[key], axis=1))
    return {key: np.concatenate(value) for key, value in stores.items()}


def build_ram_bank(sequences, *, training, seed, device, flow_batch_size=32, max_frames=None, progress=print):
    guard_no_d7(sequences); adapter = SEARAFTFlowAdapter(device=device); fields = defaultdict(list); metadata = []
    rng = np.random.default_rng(seed + (0 if training else 1000)); flow_runtime = 0.0
    for sequence in sequences:
        _, frame_ids, images, gt, coverage = _load_inputs(sequence, device, max_frames)
        if len(frame_ids) <= 8: continue
        indices = list(range(8, len(frame_ids))); tick = time.perf_counter()
        flows, _ = infer_age_flows(adapter, images, indices, ANCHOR_AGES, flow_batch_size, device); flow_runtime += time.perf_counter() - tick
        for backbone in SEEN_BACKBONES:
            raw_mmap, valid_mmap, cache_ids, _ = load_sequence_cache(backbone, sequence)
            if [str(value) for value in cache_ids[:len(frame_ids)]] != frame_ids: raise RuntimeError(f"frame-ID mismatch {backbone}/{sequence}")
            raw = np.asarray(raw_mmap[:len(frame_ids)], dtype=np.float32); valid = np.asarray(valid_mmap[:len(frame_ids)]) > 0
            maps = _align(raw, valid, images, indices, flows, device, flow_batch_size)
            for position, index in enumerate(indices):
                if training:
                    height, width = 64, 80; y = int(rng.integers(0, 81)); x = int(rng.integers(0, 101))
                else: height, width, y, x = 144, 180, 0, 0
                crop = lambda array: np.ascontiguousarray(array[..., y:y+height, x:x+width])
                fields["raw"].append(crop(raw[index][None]).astype(np.float16))
                for key in ("candidates", "fb_confidence"): fields[key].append(crop(maps[key][position]).astype(np.float16))
                for key in ("candidate_valid", "warp_support"): fields[key].append(crop(maps[key][position]).astype(np.uint8))
                fields["gt"].append(crop(gt[index][None]).astype(np.float16)); fields["gt_coverage"].append(crop(coverage[index][None]).astype(np.float16))
                fields["raw_valid"].append(crop(valid[index][None]).astype(np.uint8))
                metadata.append({"backbone": backbone, "sequence": sequence, "frame_id": frame_ids[index], "frame_index": index, "ages": list(ANCHOR_AGES)})
        del images, gt, coverage, flows
        if device.type == "cuda": torch.cuda.empty_cache()
        progress(f"BANK sequence={sequence} examples={len(metadata)}")
    return AnchorBankDataset({key: torch.from_numpy(np.stack(value)) for key, value in fields.items()}, metadata), flow_runtime


def make_loader(dataset, *, seed, batch_size, workers, training):
    sampler = BalancedGroupSampler(dataset.metadata, seed) if training else None
    result = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=workers, pin_memory=True,
                        persistent_workers=workers > 0, prefetch_factor=4 if workers else None)
    result.balanced_sampler = sampler
    return result


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def evidence_from_batch(batch):
    return MultiAnchorEvidence(batch["raw"].float(), batch["candidates"].float(), batch["candidate_valid"].bool(),
        batch["warp_support"].bool(), batch["fb_confidence"].float(), torch.tensor(ANCHOR_AGES, device=batch["raw"].device),
        torch.zeros(4, device=batch["raw"].device))
