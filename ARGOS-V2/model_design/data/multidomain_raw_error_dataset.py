"""Minimal multi-domain supervision for the frozen ARGOS v2 proposal pipeline.

Only genuine SCARED-C processed GT, D4D Zivid anchor GT and (for the explicit
M2 fold) SERV-CT CT-derived GT are exposed.  D4D context frames provide causal
evidence only: supervision is present exclusively at the current anchor.
"""
from __future__ import annotations

import csv
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from model_design.data.raw_error_dataset import RawErrorTargets
from model_design.data.temporal_pair_dataset import resize_gt_to_cache_masked


V2_ROOT = Path(__file__).resolve().parents[2]
ARGOS_ROOT = V2_ROOT.parent
SCRIPTS = V2_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

D4D_ROOT = ARGOS_ROOT / "results/03_temporal_refinement/ood/d4d_s2m2_zero_shot"
D4D_GT_ROOT = ARGOS_ROOT / "dataset/D4D/processed/keyframe_stereo_gt_curated"
SERV_ROOT = ARGOS_ROOT / "results/03_temporal_refinement/ood/prepared/servct"
MULTIDOMAIN_CACHE_ROOT = V2_ROOT / "cache_multidomain_backbones"
CACHE_HW = (144, 180)
FORBIDDEN_GT_TOKENS = ("stereo_depth", "igev", "igev++")
OOD_TRAINING_BACKBONES = ("RAFT-Stereo", "StereoAnywhere")


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _rgb_cache(image: np.ndarray) -> torch.Tensor:
    h, w = CACHE_HW
    value = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(np.ascontiguousarray(value)).permute(2, 0, 1).float()


def _resize_disparity(value: np.ndarray) -> np.ndarray:
    h, w = CACHE_HW
    value = np.asarray(value, np.float32)
    return cv2.resize(value, (w, h), interpolation=cv2.INTER_LINEAR) * (w / value.shape[1])


def verify_geometric_gt_path(path: str | Path, domain: str) -> Path:
    """Reject prediction-derived labels and verify the canonical GT provenance."""
    value = Path(path)
    if not value.is_absolute():
        value = ARGOS_ROOT / value
    lowered = str(value).lower()
    if any(token in lowered for token in FORBIDDEN_GT_TOKENS):
        raise ValueError(f"prediction-derived source is forbidden as GT: {value}")
    if domain == "D4D" and "keyframe_stereo_gt_curated" not in lowered:
        raise ValueError(f"D4D GT must be curated Zivid anchor geometry: {value}")
    if domain == "SERV-CT" and "servct_argos" not in lowered:
        raise ValueError(f"SERV-CT GT must be CT-derived ARGOS geometry: {value}")
    if not value.exists():
        raise FileNotFoundError(value)
    return value


@dataclass(frozen=True)
class MultiDomainRecord:
    domain: str
    backbone: str
    sequence: str
    specimen: str
    current_frame_id: str
    past_frame_id: str
    source_index: int
    supervision_source: str


class FrozenMultiDomainPredictions:
    """Read-only [T,144,180] predictions from the separately validated OOD cache.

    This class intentionally has no GT fields and accepts only the two *seen*
    training backbones added to D4D/SERV.  The S2M2-S legacy shards remain the
    canonical source for existing experiments.
    """

    def __init__(self, backbone: str, domain: str, *, cache_root: Path = MULTIDOMAIN_CACHE_ROOT) -> None:
        if backbone not in OOD_TRAINING_BACKBONES:
            raise ValueError(f"OOD cache is restricted to training backbones: {backbone}")
        self.path = cache_root / backbone / domain
        if not (self.path / ".complete").exists():
            raise FileNotFoundError(f"validated OOD cache missing: {self.path}")
        self.disparity = np.load(self.path / "disparity.npy", mmap_mode="r")
        self.valid = np.load(self.path / "valid_mask.npy", mmap_mode="r")
        ids = np.load(self.path / "frame_ids.npy").tolist()
        if self.disparity.shape != self.valid.shape or self.disparity.shape[1:] != CACHE_HW:
            raise RuntimeError(f"invalid OOD cache tensor contract: {self.path}")
        self._index = {str(frame_id): index for index, frame_id in enumerate(ids)}
        if len(self._index) != len(ids):
            raise RuntimeError(f"duplicate frame IDs in OOD cache: {self.path}")

    def get(self, frame_id: str) -> tuple[np.ndarray, np.ndarray]:
        try:
            index = self._index[frame_id]
        except KeyError as exc:
            raise KeyError(f"frame {frame_id} absent from OOD cache {self.path}") from exc
        disparity = np.asarray(self.disparity[index], dtype=np.float32)
        valid = np.asarray(self.valid[index], dtype=bool) & np.isfinite(disparity) & (disparity > 0)
        return disparity, valid


def _normalize_backbones(backbone: str | Sequence[str]) -> tuple[str, ...]:
    values = (backbone,) if isinstance(backbone, str) else tuple(backbone)
    if not values or len(set(values)) != len(values):
        raise ValueError("backbones must be a non-empty unique sequence")
    if any(value != "S2M2-S" and value not in OOD_TRAINING_BACKBONES for value in values):
        raise ValueError(f"unsupported OOD training backbone(s): {values}")
    return values


class D4DAnchorDataset(Dataset):
    """One genuinely supervised causal t-1 pair per validated Zivid anchor."""

    def __init__(self, specimens: Sequence[str], *, backbone: str | Sequence[str] = "S2M2-S",
                 cache_root: Path = MULTIDOMAIN_CACHE_ROOT, max_records: int | None = None) -> None:
        self.specimens = tuple(specimens)
        self.backbones = _normalize_backbones(backbone)
        self._predictions = {name: (None if name == "S2M2-S"
                                    else FrozenMultiDomainPredictions(name, "D4D", cache_root=cache_root))
                             for name in self.backbones}
        gt_rows = {
            row["anchor_id"]: row
            for row in csv.DictReader((D4D_GT_ROOT / "manifests/valid_and_warning_manifest.csv").open())
        }
        contexts = {
            row["anchor_id"]: row
            for row in csv.DictReader((D4D_ROOT / "context_manifest.csv").open())
        }
        index_rows = list(csv.DictReader((D4D_ROOT / "d4d_index.csv").open()))
        self._rows: list[dict] = []
        self.records: list[MultiDomainRecord] = []
        for row in index_rows:
            if row["specimen"] not in self.specimens:
                continue
            anchor_id = row["sequence_id"]
            gt = gt_rows.get(anchor_id)
            context = contexts.get(anchor_id)
            if gt is None or context is None:
                raise RuntimeError(f"missing canonical D4D metadata for {anchor_id}")
            gt_path = verify_geometric_gt_path(gt["gt_disparity_path"], "D4D")
            valid_path = verify_geometric_gt_path(gt["valid_mask_path"], "D4D")
            stems = context["context_stems"].split(";")[::-1]
            if len(stems) != 4:
                raise RuntimeError(f"D4D context must have four causal frames: {anchor_id}")
            enriched = dict(row, gt_path=str(gt_path), valid_path=str(valid_path), stems=stems)
            for name in self.backbones:
                self._rows.append(enriched)
                self.records.append(MultiDomainRecord(
                    domain="D4D", backbone=name,
                    sequence=f'{row["specimen"]}__{row["session"]}', specimen=row["specimen"],
                    current_frame_id=stems[3], past_frame_id=stems[2], source_index=len(self._rows) - 1,
                    supervision_source="Zivid structured-light curated anchor",
                ))
        if not self.records:
            raise RuntimeError(f"no validated D4D anchors for specimens {self.specimens}")
        if max_records is not None:
            if max_records < 1:
                raise ValueError("max_records must be positive")
            self._rows = self._rows[:max_records]
            self.records = self.records[:max_records]
        self._rgb_maps: dict = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        from run_ood_generalization import d4d_rgb

        record, row = self.records[index], self._rows[index]
        shard = np.load(row["target_path"])
        predictions = self._predictions[record.backbone]
        if predictions is None:
            raws = shard["raw_disp"].astype(np.float32)
            raw, past = _resize_disparity(raws[3]), _resize_disparity(raws[2])
        else:
            cache_prefix = f"{row['specimen']}__{row['session']}__"
            raw, raw_valid = predictions.get(cache_prefix + row["stems"][3])
            past, past_valid = predictions.get(cache_prefix + row["stems"][2])
        # Reuse the validated D4D benchmark shard: this is the canonical Zivid
        # anchor after pose/rectification filtering and uses the same 178x224
        # disparity contract as the stored raw prediction. The original paths
        # were verified at construction time solely to enforce provenance.
        gt_anchor = shard["gt_disp"][3].astype(np.float32)
        gt_valid_anchor = shard["valid_mask"][3].astype(bool) & np.isfinite(gt_anchor) & (gt_anchor > 0)
        gt_anchor = np.where(gt_valid_anchor, gt_anchor, 0.0)
        gt, coverage = resize_gt_to_cache_masked(gt_anchor, gt_valid_anchor)
        current_rgb = d4d_rgb(row, row["stems"][3], self._rgb_maps)
        past_rgb = d4d_rgb(row, row["stems"][2], self._rgb_maps)
        if predictions is None:
            raw_valid = np.isfinite(raw) & (raw > 0)
            past_valid = np.isfinite(past) & (past > 0)
        # Preserve invalidity in explicit masks while keeping all tensors finite;
        # masked arithmetic still propagates NaN through losses (NaN * 0).
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        past = np.nan_to_num(past, nan=0.0, posinf=0.0, neginf=0.0)
        gt = np.nan_to_num(gt, nan=0.0, posinf=0.0, neginf=0.0)
        return {
            "raw": torch.from_numpy(raw.copy())[None],
            "past": torch.from_numpy(past.copy())[None],
            "raw_valid": torch.from_numpy(raw_valid.copy())[None],
            "past_valid": torch.from_numpy(past_valid.copy())[None],
            "current_rgb": _rgb_cache(current_rgb), "past_rgb": _rgb_cache(past_rgb),
            "gt": torch.from_numpy(np.ascontiguousarray(gt))[None],
            "gt_coverage": torch.from_numpy(np.ascontiguousarray(coverage))[None],
            "gt_valid": torch.from_numpy(np.ascontiguousarray(coverage > .5))[None],
            "domain": record.domain, "backbone": record.backbone,
            "sequence": record.sequence, "specimen": record.specimen,
            "past_frame_id": record.past_frame_id, "current_frame_id": record.current_frame_id,
            "supervision_source": record.supervision_source,
        }


class SERVCTPairDataset(Dataset):
    """Weak-replay causal pairs with CT-derived geometry; used only by M2."""

    def __init__(self, sequences: Sequence[str], *, backbone: str | Sequence[str] = "S2M2-S",
                 cache_root: Path = MULTIDOMAIN_CACHE_ROOT) -> None:
        self.sequences = tuple(sequences)
        self.backbones = _normalize_backbones(backbone)
        self._predictions = {name: (None if name == "S2M2-S"
                                    else FrozenMultiDomainPredictions(name, "SERV-CT", cache_root=cache_root))
                             for name in self.backbones}
        manifest = list(csv.DictReader((SERV_ROOT / "sequence_manifest.csv").open()))
        self._sequence_rows: dict[str, list[dict]] = {}
        self.records: list[MultiDomainRecord] = []
        for sequence in self.sequences:
            rows = sorted((r for r in manifest if r["sequence_id"] == sequence), key=lambda r: int(r["order_index"]))
            if len(rows) < 2:
                raise RuntimeError(f"SERV-CT sequence has no causal pairs: {sequence}")
            for row in rows:
                verify_geometric_gt_path(row["gt_disp_path"], "SERV-CT")
            self._sequence_rows[sequence] = rows
            for current in range(1, len(rows)):
                for name in self.backbones:
                    self.records.append(MultiDomainRecord(
                        domain="SERV-CT", backbone=name, sequence=sequence, specimen=sequence,
                        current_frame_id=rows[current]["frame_id"], past_frame_id=rows[current - 1]["frame_id"],
                        source_index=current, supervision_source="CT-derived disparity",
                    ))
        self._shards: dict[str, object] = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        rows = self._sequence_rows[record.sequence]
        current = record.source_index
        if record.sequence not in self._shards:
            self._shards[record.sequence] = np.load(SERV_ROOT / "shards" / f"{record.sequence}.npz")
        shard = self._shards[record.sequence]
        predictions = self._predictions[record.backbone]
        if predictions is None:
            raw = shard["raw_disp"][current].astype(np.float32)
            past = shard["raw_disp"][current - 1].astype(np.float32)
        else:
            raw, raw_valid = predictions.get(f"{record.sequence}__{record.current_frame_id}")
            past, past_valid = predictions.get(f"{record.sequence}__{record.past_frame_id}")
        gt = shard["gt_disp"][current].astype(np.float32)
        valid = shard["valid_mask"][current].astype(bool)
        if predictions is None:
            raw_valid = np.isfinite(raw) & (raw > 0)
            past_valid = np.isfinite(past) & (past > 0)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        past = np.nan_to_num(past, nan=0.0, posinf=0.0, neginf=0.0)
        gt = np.nan_to_num(gt, nan=0.0, posinf=0.0, neginf=0.0)
        return {
            "raw": torch.from_numpy(raw.copy())[None], "past": torch.from_numpy(past.copy())[None],
            "raw_valid": torch.from_numpy(raw_valid.copy())[None],
            "past_valid": torch.from_numpy(past_valid.copy())[None],
            "current_rgb": _rgb_cache(_read_rgb(Path(rows[current]["left_path"]))),
            "past_rgb": _rgb_cache(_read_rgb(Path(rows[current - 1]["left_path"]))),
            "gt": torch.from_numpy(gt.copy())[None],
            "gt_coverage": torch.from_numpy(valid.astype(np.float32))[None],
            "gt_valid": torch.from_numpy(valid.copy())[None],
            "domain": record.domain, "backbone": record.backbone,
            "sequence": record.sequence, "specimen": record.specimen,
            "past_frame_id": record.past_frame_id, "current_frame_id": record.current_frame_id,
            "supervision_source": record.supervision_source,
        }


class MultiDomainRawErrorDataset(Dataset):
    """Concatenate named sources while preserving explicit sample provenance."""

    def __init__(self, datasets: Mapping[str, Dataset]) -> None:
        if not datasets:
            raise ValueError("at least one domain dataset is required")
        self.datasets = dict(datasets)
        self._lookup: list[tuple[str, int]] = []
        self.records: list[MultiDomainRecord] = []
        for domain, dataset in self.datasets.items():
            source_records = getattr(dataset, "records", None)
            if source_records is None or len(source_records) != len(dataset):
                raise ValueError(f"dataset {domain} must expose one record per sample")
            for local_index, source in enumerate(source_records):
                if isinstance(source, MultiDomainRecord):
                    record = source
                else:
                    record = MultiDomainRecord(
                        domain=domain, backbone=source.backbone, sequence=source.sequence,
                        specimen=source.sequence, current_frame_id=source.current_frame_id,
                        past_frame_id=source.past_frame_id, source_index=local_index,
                        supervision_source="SCARED-C corrected temporal pseudo-GT",
                    )
                if record.domain != domain:
                    raise ValueError(f"record/domain mismatch: {record.domain} != {domain}")
                self._lookup.append((domain, local_index)); self.records.append(record)

    def __len__(self) -> int:
        return len(self._lookup)

    def __getitem__(self, index: int) -> dict:
        domain, local = self._lookup[index]
        sample = dict(self.datasets[domain][local])
        sample.setdefault("domain", domain)
        sample.setdefault("specimen", sample["sequence"])
        sample.setdefault("supervision_source", "SCARED-C corrected temporal pseudo-GT")
        # Default collation requires identical keys across heterogeneous samples.
        # Native cache indices do not exist for anchor-local D4D/SERV supervision;
        # frame IDs remain the authoritative mapping for every domain.
        sample.setdefault("past_index", -1)
        sample.setdefault("current_index", -1)
        return sample


class DomainBalancedSampler(Sampler[int]):
    """Deterministic ratio sampler, balanced over backbone and sequence/session."""

    def __init__(self, dataset: MultiDomainRawErrorDataset, ratios: Mapping[str, float],
                 *, samples_per_epoch: int, seed: int) -> None:
        if samples_per_epoch < 1:
            raise ValueError("samples_per_epoch must be positive")
        self.dataset, self.samples_per_epoch, self.seed = dataset, int(samples_per_epoch), int(seed)
        if set(ratios) != set(dataset.datasets):
            raise ValueError("ratio domains must exactly match dataset domains")
        total = float(sum(ratios.values()))
        if total <= 0 or any(value < 0 for value in ratios.values()):
            raise ValueError("ratios must be non-negative with positive sum")
        self.ratios = {key: float(value) / total for key, value in ratios.items()}
        self.epoch = 0
        self._groups: dict[str, dict[tuple[str, str], list[int]]] = {}
        for index, record in enumerate(dataset.records):
            self._groups.setdefault(record.domain, {}).setdefault((record.backbone, record.sequence), []).append(index)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def domain_counts(self) -> dict[str, int]:
        exact = {d: self.samples_per_epoch * r for d, r in self.ratios.items()}
        counts = {d: int(np.floor(v)) for d, v in exact.items()}
        remainder = self.samples_per_epoch - sum(counts.values())
        order = sorted(exact, key=lambda d: (-(exact[d] - counts[d]), d))
        for domain in order[:remainder]:
            counts[domain] += 1
        return counts

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        selected: list[int] = []
        for domain, count in self.domain_counts().items():
            groups = self._groups[domain]
            keys = sorted(groups)
            shuffled = {key: list(rng.permutation(groups[key])) for key in keys}
            cursors = {key: 0 for key in keys}
            for step in range(count):
                key = keys[step % len(keys)]
                values = shuffled[key]
                cursor = cursors[key]
                if cursor and cursor % len(values) == 0:
                    values[:] = rng.permutation(values)
                selected.append(values[cursor % len(values)])
                cursors[key] += 1
        return iter(np.asarray(selected)[rng.permutation(len(selected))].tolist())

    def __len__(self) -> int:
        return self.samples_per_epoch


def stratified_raw_error_targets(targets: RawErrorTargets, *, pixels_per_bin: int,
                                 seed: int) -> RawErrorTargets:
    """Balance clean, moderate and large-error pixels without changing labels."""
    if pixels_per_bin <= 0:
        return targets
    rng = np.random.default_rng(seed)
    selected = torch.zeros_like(targets.regression_valid, dtype=torch.bool)
    error = targets.error.detach().cpu().numpy()
    valid = targets.regression_valid.detach().cpu().numpy().astype(bool)
    for batch in range(error.shape[0]):
        strata = (
            valid[batch] & (error[batch] <= .5),
            valid[batch] & (error[batch] > .5) & (error[batch] <= 3.0),
            valid[batch] & (error[batch] > 3.0),
        )
        for mask in strata:
            index = np.flatnonzero(mask.ravel())
            if len(index) > pixels_per_bin:
                index = rng.choice(index, size=pixels_per_bin, replace=False)
            flat = selected[batch].view(-1)
            if len(index):
                flat[torch.from_numpy(index).to(flat.device)] = True
    regression = targets.regression_valid & selected
    return RawErrorTargets(
        error=targets.error, label=targets.label,
        regression_valid=regression,
        classification_valid=targets.classification_valid & selected,
        clean=targets.clean & selected,
    )


def manifest_digest(records: Sequence[MultiDomainRecord]) -> str:
    payload = "\n".join(
        "|".join((
            str(getattr(r, "domain", "SCARED-C")),
            str(r.backbone), str(r.sequence), str(getattr(r, "specimen", r.sequence)),
            str(r.past_frame_id), str(r.current_frame_id),
            str(getattr(r, "supervision_source", "SCARED-C corrected temporal pseudo-GT")),
        ))
        for r in records
    )
    return hashlib.sha256(payload.encode()).hexdigest()


__all__ = [
    "D4DAnchorDataset", "DomainBalancedSampler", "MultiDomainRawErrorDataset",
    "MultiDomainRecord", "SERVCTPairDataset", "manifest_digest",
    "stratified_raw_error_targets", "verify_geometric_gt_path",
]
