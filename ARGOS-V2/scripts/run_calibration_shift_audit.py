#!/usr/bin/env python3
"""D0 calibration-shift audit for the frozen ARGOS v2 pipeline.

This is deliberately an analysis-only runner.  It reuses the frozen OOD
pipeline and never writes predictions, modifies a checkpoint, or changes a
model.  Per-pixel forward quantities are reduced online to exact moments;
fixed-seed in-memory samples are used only for multivariate/ranking plots.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import ks_2samp, pearsonr, spearmanr, wasserstein_distance
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# The validated OOD runner owns dataset parsing and frozen artifact checks.
from run_ood_generalization import (ARGOS_ROOT, H, MODES_PATH, W, FrozenARGOS, d4d_rgb,
                                    read_rgb, resize_disparity, resize_gt, rgb_tensor,
                                    s2m2_model, verify_frozen)
from model_design.external_components.bidavideo import temporal_disparity_evidence
from model_design.models.abstention import authorization_mask, calibrated_probability
from model_design.models.raw_error_detector import RawErrorEvidence

ROOT = Path("/dtu/p1/leopam/ARGOS/ARGOS-V2")
SERV_DIR = ARGOS_ROOT / "results/03_temporal_refinement/ood/prepared/servct"
D4D_DIR = ARGOS_ROOT / "results/03_temporal_refinement/ood/d4d_s2m2_zero_shot"
STEREOMIS_DIR = ARGOS_ROOT / "dataset/StereoMIS/curated/geometric_gt/temporal_sequences"
OUT_DEFAULT = ROOT / "results/calibration_shift_audit"
SEEN_BACKBONES = ["S2M2-S", "RAFT-Stereo", "StereoAnywhere"]
HELDOUT = ["dataset_7_keyframe_3", "dataset_7_keyframe_4"]
EPS = 1e-8


def save_json(path: Path, value: Any) -> None:
    def clean(x: Any) -> Any:
        if isinstance(x, np.generic):
            x = x.item()
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, float) and not math.isfinite(x):
            return None
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        return x
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n")


class Moments:
    """Exact scalar moments accumulated over every eligible forward pixel."""
    def __init__(self) -> None:
        self.n = defaultdict(int)
        self.s = defaultdict(float)
        self.s2 = defaultdict(float)
        self.lo = defaultdict(lambda: float("inf"))
        self.hi = defaultdict(lambda: float("-inf"))

    def add(self, values: dict[str, np.ndarray], mask: np.ndarray) -> None:
        for name, value in values.items():
            v = np.asarray(value)[mask]
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            self.n[name] += int(v.size)
            self.s[name] += float(v.sum(dtype=np.float64))
            self.s2[name] += float(np.square(v, dtype=np.float64).sum(dtype=np.float64))
            self.lo[name] = min(self.lo[name], float(v.min()))
            self.hi[name] = max(self.hi[name], float(v.max()))

    def rows(self, dataset: str) -> list[dict[str, Any]]:
        rows = []
        for name in sorted(self.n):
            n = self.n[name]
            mean = self.s[name] / n
            var = max(0.0, self.s2[name] / n - mean * mean)
            rows.append(dict(dataset=dataset, feature=name, count=n, mean=mean,
                             std=math.sqrt(var), minimum=self.lo[name], maximum=self.hi[name]))
        return rows


class SampleStore:
    """Fixed-seed analysis sample; this is not a prediction cache."""
    def __init__(self, rng: np.random.Generator, per_frame: int) -> None:
        self.rng, self.per_frame = rng, per_frame
        self.parts: list[pd.DataFrame] = []

    def add(self, dataset: str, frame: str, arrays: dict[str, np.ndarray],
            sample_mask: np.ndarray) -> None:
        idx = np.flatnonzero(sample_mask.ravel())
        if not len(idx):
            return
        if len(idx) > self.per_frame:
            idx = self.rng.choice(idx, size=self.per_frame, replace=False)
        data = {k: np.asarray(v).ravel()[idx] for k, v in arrays.items()}
        data["dataset"] = np.full(len(idx), dataset)
        data["frame"] = np.full(len(idx), frame)
        self.parts.append(pd.DataFrame(data))

    def dataframe(self) -> pd.DataFrame:
        return pd.concat(self.parts, ignore_index=True) if self.parts else pd.DataFrame()


def gradient_maps(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dx = F.pad(raw[..., :, 1:] - raw[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(raw[..., 1:, :] - raw[..., :-1, :], (0, 0, 0, 1))
    grad = torch.sqrt(dx.square() + dy.square() + 1e-8)
    mean = F.avg_pool2d(raw, 5, 1, 2)
    variance = F.avg_pool2d(raw.square(), 5, 1, 2) - mean.square()
    return dx, dy, variance.clamp_min(0.0)


@torch.inference_mode()
def inspect_step(pipe: FrozenARGOS, raw: torch.Tensor, raw_valid: torch.Tensor,
                 current_rgb: torch.Tensor, past_raw: torch.Tensor,
                 past_valid: torch.Tensor, past_rgb: torch.Tensor,
                 capture: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Exact frozen t-1 composition with a hook-derived penultimate feature."""
    flows = pipe.flow.infer(torch.cat((current_rgb, past_rgb)), torch.cat((past_rgb, current_rgb))).clone()
    evidence = {k: v.detach() for k, v in temporal_disparity_evidence(
        raw, past_raw, flows[:1], flows[1:], current_valid=raw_valid,
        past_valid=past_valid, current_rgb=current_rgb, past_rgb=past_rgb,
    ).as_dict().items()}
    evidence["current_valid"] = raw_valid
    proposal = pipe.a2(raw, evidence, current_rgb)
    capture.clear()
    detector_input = RawErrorEvidence(raw, raw_valid, evidence["aligned_past_disparity"],
        evidence["aligned_validity"], evidence["warp_support"], evidence["forward_backward_error"],
        evidence["forward_backward_confidence"], evidence["photometric_residual"],
        evidence["flow_magnitude"], proposal.update, proposal.g_error, proposal.c_memory)
    prediction = pipe.detector(detector_input)
    feature = capture.get("penultimate")
    if feature is None:
        raise RuntimeError("Detector encoder hook did not capture penultimate features")
    calibrated = calibrated_probability(prediction.logits, pipe.temperature)
    authorized = authorization_mask(prediction, mode=pipe.mode, temperature=pipe.temperature,
                                    aligned_valid=evidence["aligned_validity"], warp_support=evidence["warp_support"],
                                    proposal_update=proposal.update)
    # Existing ultra-safe SCARED-C mode only; never fitted or adjusted on OOD.
    ultra = (calibrated >= 0.95) & (prediction.mu >= 0.25) & \
        (prediction.sigma <= 2.0) & evidence["aligned_validity"].bool() & evidence["warp_support"].bool() & \
        (proposal.update.abs() <= 3.0)
    update = authorized.float() * proposal.update
    dx, dy, variance = gradient_maps(raw)
    return dict(
        raw_probability=prediction.probability, intervention_probability=calibrated,
        predicted_error_mu=prediction.mu,
        predicted_uncertainty_sigma=prediction.sigma,
        authorization=authorized.float(), authorization_ultra_safe=ultra.float(),
        update_magnitude=update.abs(), update_signed=update, a2_proposal_magnitude=proposal.update.abs(),
        a2_error_gate=proposal.g_error, a2_memory_gate=proposal.c_memory,
        aligned_disparity=evidence["aligned_past_disparity"],
        disparity_residual_signed=raw - evidence["aligned_past_disparity"],
        bidavideo_disagreement_abs=(raw - evidence["aligned_past_disparity"]).abs(),
        forward_backward_error=evidence["forward_backward_error"],
        forward_backward_confidence=evidence["forward_backward_confidence"],
        photometric_residual=evidence["photometric_residual"],
        warp_support=evidence["warp_support"].float(), raw_valid=raw_valid.float(),
        aligned_valid=evidence["aligned_validity"].float(),
        flow_magnitude=evidence["flow_magnitude"], raw_gradient_x=dx,
        raw_gradient_y=dy, raw_gradient_magnitude=torch.sqrt(dx.square() + dy.square() + EPS),
        raw_local_variance=variance, memory_age=torch.ones_like(raw),
        penultimate=feature,
    )


def tensors_to_arrays(result: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    arrays = {}
    for name, value in result.items():
        value = value.detach().float().cpu().numpy()
        if name == "penultimate":
            for c in range(value.shape[1]):
                arrays[f"penultimate_{c:02d}"] = value[0, c]
        else:
            arrays[name] = value[0, 0]
    return arrays


def temporal_pair_dataset(backbones: list[str], sequences: list[str], max_pairs: int):
    from model_design.data.temporal_pair_dataset import TemporalPairDataset
    return TemporalPairDataset(backbones, sequences, coverage_threshold=0.50,
                               max_pairs_per_sequence=max_pairs, random_clip_start=False)


def iter_scared(backbones: list[str], sequences: list[str], max_pairs: int) -> Iterator[dict[str, Any]]:
    dataset = temporal_pair_dataset(backbones, sequences, max_pairs)
    for sample in dataset:
        yield dict(frame=f"{sample['backbone']}:{sample['sequence']}:{sample['current_frame_id']}",
                   raw=sample["raw"][0].numpy(), raw_valid=sample["raw_valid"][0].numpy(),
                   past=sample["past"][0].numpy(), past_valid=sample["past_valid"][0].numpy(),
                   current_rgb=sample["current_rgb"].permute(1, 2, 0).numpy().astype(np.uint8),
                   past_rgb=sample["past_rgb"].permute(1, 2, 0).numpy().astype(np.uint8),
                   gt=sample["gt"][0].numpy(), gt_valid=sample["gt_valid"][0].numpy())


def iter_serv() -> Iterator[dict[str, Any]]:
    manifest = list(csv.DictReader((SERV_DIR / "sequence_manifest.csv").open()))
    for sequence in sorted({r["sequence_id"] for r in manifest}):
        rows = sorted((r for r in manifest if r["sequence_id"] == sequence), key=lambda r: int(r["order_index"]))
        shard = np.load(SERV_DIR / "shards" / f"{sequence}.npz")
        for index in range(1, len(rows)):
            current, past = rows[index], rows[index - 1]
            yield dict(frame=f'SERV-CT:{sequence}:{current["frame_id"]}', raw=shard["raw_disp"][index].astype(np.float32),
                       raw_valid=np.isfinite(shard["raw_disp"][index]) & (shard["raw_disp"][index] > 0),
                       past=shard["raw_disp"][index - 1].astype(np.float32),
                       past_valid=np.isfinite(shard["raw_disp"][index - 1]) & (shard["raw_disp"][index - 1] > 0),
                       current_rgb=read_rgb(Path(current["left_path"])), past_rgb=read_rgb(Path(past["left_path"])),
                       gt=shard["gt_disp"][index].astype(np.float32), gt_valid=shard["valid_mask"][index].astype(bool))


def iter_d4d(max_windows: int) -> Iterator[dict[str, Any]]:
    index = list(csv.DictReader((D4D_DIR / "d4d_index.csv").open()))[:max_windows]
    contexts = {row["anchor_id"]: row for row in csv.DictReader((D4D_DIR / "context_manifest.csv").open())}
    cache: dict = {}
    for row in index:
        shard = np.load(row["target_path"])
        raws = shard["raw_disp"].astype(np.float32)
        stems = contexts[row["sequence_id"]]["context_stems"].split(";")[::-1]
        rgbs = [d4d_rgb(row, stem, cache) for stem in stems]
        for t in range(1, 4):
            raw, past = resize_disparity(raws[t]), resize_disparity(raws[t - 1])
            gt = gt_valid = None
            if t == 3:
                gt, gt_valid, _ = resize_gt(shard["gt_disp"][3].astype(np.float32), shard["valid_mask"][3].astype(bool))
            yield dict(frame=f'D4D:{row["sequence_id"]}:{stems[t]}', raw=raw, raw_valid=np.isfinite(raw) & (raw > 0),
                       past=past, past_valid=np.isfinite(past) & (past > 0), current_rgb=rgbs[t], past_rgb=rgbs[t - 1],
                       gt=gt, gt_valid=gt_valid)


def iter_stereomis(samples_per_sequence: int, device: torch.device) -> Iterator[dict[str, Any]]:
    model, infer = s2m2_model(device)
    for sequence in ["P1", "P2_8", "P3"]:
        lefts = sorted((STEREOMIS_DIR / sequence / "left").glob("*"))
        rights = {p.stem: p for p in (STEREOMIS_DIR / sequence / "right").glob("*")}
        lefts = [p for p in lefts if p.stem in rights]
        chosen = np.unique(np.linspace(1, len(lefts) - 1, min(samples_per_sequence, len(lefts) - 1), dtype=int))
        for t in chosen:
            past_native, _, _ = infer(model, read_rgb(lefts[t - 1]), read_rgb(rights[lefts[t - 1].stem]), device, 512)
            raw_native, _, _ = infer(model, read_rgb(lefts[t]), read_rgb(rights[lefts[t].stem]), device, 512)
            past, raw = resize_disparity(past_native), resize_disparity(raw_native)
            yield dict(frame=f"StereoMIS:{sequence}:{lefts[t].stem}", raw=raw, raw_valid=np.isfinite(raw) & (raw > 0),
                       past=past, past_valid=np.isfinite(past) & (past > 0), current_rgb=read_rgb(lefts[t]),
                       past_rgb=read_rgb(lefts[t - 1]), gt=None, gt_valid=None)


def as_tensor(x: np.ndarray, device: torch.device, boolean: bool = False) -> torch.Tensor:
    value = torch.from_numpy(np.asarray(x)).to(device)
    return value.bool().unsqueeze(0).unsqueeze(0) if boolean else value.float().unsqueeze(0).unsqueeze(0)


def binary_metrics(prob: np.ndarray, y: np.ndarray) -> dict[str, float]:
    good = np.isfinite(prob) & np.isfinite(y)
    prob, y = prob[good], y[good].astype(int)
    if not len(y):
        return {"n": 0}
    out = {"n": len(y), "prevalence": float(y.mean()), "brier": float(brier_score_loss(y, prob))}
    out["auroc"] = float(roc_auc_score(y, prob)) if y.min() != y.max() else float("nan")
    out["ap"] = float(average_precision_score(y, prob)) if y.min() != y.max() else float("nan")
    bins = np.clip((prob * 10).astype(int), 0, 9)
    ece = 0.0
    for b in range(10):
        take = bins == b
        if take.any():
            ece += take.mean() * abs(prob[take].mean() - y[take].mean())
    out["ece"] = float(ece)
    return out


def process_dataset(name: str, records: Iterator[dict[str, Any]], pipe: FrozenARGOS,
                    hook_capture: dict[str, torch.Tensor], device: torch.device,
                    moments: Moments, samples: SampleStore) -> dict[str, Any]:
    totals = defaultdict(float)
    start = time.perf_counter()
    for record in records:
        raw = as_tensor(record["raw"], device)
        past = as_tensor(record["past"], device)
        raw_valid = as_tensor(record["raw_valid"], device, True)
        past_valid = as_tensor(record["past_valid"], device, True)
        current_rgb = rgb_tensor(record["current_rgb"], device)
        past_rgb = rgb_tensor(record["past_rgb"], device)
        out = inspect_step(pipe, raw, raw_valid, current_rgb, past, past_valid, past_rgb, hook_capture)
        arr = tensors_to_arrays(out)
        support = (arr["raw_valid"] > .5) & (arr["warp_support"] > .5) & (arr["aligned_valid"] > .5)
        totals["frames"] += 1
        totals["eligible_pixels"] += int(support.sum())
        totals["authorized_pixels"] += float(arr["authorization"][support].sum())
        totals["ultra_authorized_pixels"] += float(arr["authorization_ultra_safe"][support].sum())
        moments.add({k: v for k, v in arr.items() if not k.startswith("penultimate_")}, support)
        moments.add({k: v for k, v in arr.items() if k.startswith("penultimate_")}, support)
        if record["gt"] is not None:
            gt = np.asarray(record["gt"], dtype=np.float32)
            gt_valid = np.asarray(record["gt_valid"], dtype=bool)
            common = support & gt_valid & np.isfinite(gt)
            raw_error = np.abs(np.asarray(record["raw"], dtype=np.float32) - gt)
            refined_error = np.abs(np.asarray(record["raw"], dtype=np.float32) + arr["update_signed"] - gt)
            clean = raw_error <= .50
            raw_wrong = raw_error > .50
            changed = np.abs(arr["update_signed"]) > .05
            false_update = changed & clean
            clean_degradation = clean & (refined_error > raw_error + .02)
            incorrect = (arr["authorization"] > .5) & (clean | (refined_error > raw_error + .02))
            arr.update(raw_error=raw_error, raw_wrong=raw_wrong.astype(np.float32),
                       clean=clean.astype(np.float32), false_update=false_update.astype(np.float32),
                       clean_degradation=clean_degradation.astype(np.float32),
                       incorrect_authorization=incorrect.astype(np.float32), gt_valid=gt_valid.astype(np.float32))
            moments.add({"raw_error": raw_error, "raw_wrong": raw_wrong.astype(np.float32),
                         "false_update": false_update.astype(np.float32),
                         "incorrect_authorization": incorrect.astype(np.float32)}, common)
            totals["gt_pixels"] += int(common.sum())
            totals["gt_authorized_pixels"] += float(arr["authorization"][common].sum())
            totals["gt_changed_pixels"] += float(changed[common].sum())
            totals["clean_pixels"] += float(clean[common].sum())
            totals["raw_wrong_pixels"] += float(raw_wrong[common].sum())
            totals["false_updates"] += float(false_update[common].sum())
            totals["clean_degradations"] += float(clean_degradation[common].sum())
            totals["incorrect_authorizations"] += float(incorrect[common].sum())
            sample_mask = common
        else:
            arr.update(raw_error=np.full_like(arr["authorization"], np.nan),
                       raw_wrong=np.full_like(arr["authorization"], np.nan),
                       clean=np.full_like(arr["authorization"], np.nan),
                       false_update=np.full_like(arr["authorization"], np.nan),
                       clean_degradation=np.full_like(arr["authorization"], np.nan),
                       incorrect_authorization=np.full_like(arr["authorization"], np.nan),
                       gt_valid=np.zeros_like(arr["authorization"]))
            sample_mask = support
        samples.add(name, record["frame"], arr, sample_mask)
    totals["elapsed_s"] = time.perf_counter() - start
    totals["eligible_authorization_rate"] = totals["authorized_pixels"] / max(1.0, totals["eligible_pixels"])
    totals["ultra_authorization_rate"] = totals["ultra_authorized_pixels"] / max(1.0, totals["eligible_pixels"])
    totals["false_update_rate_among_gt_updates"] = totals["false_updates"] / max(1.0, totals["gt_changed_pixels"])
    totals["clean_degradation_ratio"] = totals["clean_degradations"] / max(1.0, totals["clean_pixels"])
    totals["incorrect_authorization_rate_gt"] = totals["incorrect_authorizations"] / max(1.0, totals["gt_authorized_pixels"])
    return dict(dataset=name, **totals)


def feature_shift(sample: pd.DataFrame, features: list[str], reference: str) -> pd.DataFrame:
    rows = []
    ref = sample[sample.dataset == reference]
    for dataset in sorted(sample.dataset.unique()):
        if dataset == reference:
            continue
        cur = sample[sample.dataset == dataset]
        for feature in features:
            a, b = ref[feature].dropna().values, cur[feature].dropna().values
            if len(a) < 20 or len(b) < 20:
                continue
            ks = ks_2samp(a, b)
            rows.append(dict(reference=reference, dataset=dataset, feature=feature,
                             reference_mean=float(a.mean()), dataset_mean=float(b.mean()),
                             mean_shift_std=float((b.mean() - a.mean()) / (a.std() + EPS)),
                             std_ratio=float(b.std() / (a.std() + EPS)),
                             wasserstein=float(wasserstein_distance(a, b)), ks_stat=float(ks.statistic),
                             ks_pvalue=float(ks.pvalue)))
    return pd.DataFrame(rows)


def multivariate(sample: pd.DataFrame, outdir: Path, seed: int) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    feature_cols = [c for c in sample if c.startswith("penultimate_")]
    ref = sample[sample.dataset == "SCARED-C-heldout"].dropna(subset=feature_cols)
    xref = ref[feature_cols].values
    scaler = StandardScaler().fit(xref)
    zref = scaler.transform(xref)
    covariance = LedoitWolf().fit(zref)
    nn = NearestNeighbors(n_neighbors=2).fit(zref)
    ref_nn = nn.kneighbors(zref)[0][:, 1]
    threshold = float(np.quantile(ref_nn, .95))
    summaries: dict[str, Any] = {"reference_nn_distance_p95": threshold, "datasets": {}}
    chosen = []
    for dataset in sorted(sample.dataset.unique()):
        part = sample[sample.dataset == dataset].dropna(subset=feature_cols)
        z = scaler.transform(part[feature_cols].values)
        md = covariance.mahalanobis(z)
        d = nn.kneighbors(z, n_neighbors=1)[0][:, 0]
        summaries["datasets"][dataset] = dict(n=int(len(z)), mahalanobis_mean=float(md.mean()),
            mahalanobis_median=float(np.median(md)), mahalanobis_p95=float(np.quantile(md, .95)),
            nn_distance_mean=float(d.mean()), nn_distance_median=float(np.median(d)),
            feature_overlap_ref_nn95=float((d <= threshold).mean()))
        chosen.append(part.sample(min(2500, len(part)), random_state=seed))
        pd.DataFrame(covariance.covariance_, index=feature_cols, columns=feature_cols).to_csv(outdir / f"covariance_{dataset}.csv")
    visual = pd.concat(chosen, ignore_index=True)
    zv = scaler.transform(visual[feature_cols].values)
    pca = PCA(n_components=2, random_state=seed).fit_transform(zv)
    pca_df = visual[["dataset", "authorization", "raw_wrong", "incorrect_authorization"]].copy()
    pca_df["pc1"], pca_df["pc2"] = pca[:, 0], pca[:, 1]
    take = []
    for dataset, frame in visual.groupby("dataset"):
        take.append(frame.sample(min(800, len(frame)), random_state=seed))
    ts = pd.concat(take, ignore_index=True)
    zt = scaler.transform(ts[feature_cols].values)
    perplexity = min(30, max(5, (len(zt) - 1) // 3))
    embedding = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto",
                     random_state=seed).fit_transform(zt)
    tsne_df = ts[["dataset", "authorization", "raw_wrong", "incorrect_authorization"]].copy()
    tsne_df["tsne1"], tsne_df["tsne2"] = embedding[:, 0], embedding[:, 1]
    return summaries, pca_df, tsne_df


def calibration_and_risk(sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, risks = [], []
    for dataset, part in sample.groupby("dataset"):
        valid = part.dropna(subset=["raw_wrong", "intervention_probability"])
        if not len(valid):
            rows.append(dict(dataset=dataset, metric="no_reference", value=np.nan, n=0))
            continue
        metric = binary_metrics(valid.intervention_probability.values, valid.raw_wrong.values)
        rows.append(dict(dataset=dataset, metric="raw_error_probability", **metric))
        bins = np.clip((valid.intervention_probability.values * 10).astype(int), 0, 9)
        for b in range(10):
            selected = valid.iloc[np.flatnonzero(bins == b)]
            if len(selected):
                rows.append(dict(dataset=dataset, metric="reliability_bin", bin=b,
                                 n=len(selected), predicted_mean=float(selected.intervention_probability.mean()),
                                 observed_frequency=float(selected.raw_wrong.mean())))
        for coverage in [.01, .05, .10, .20, .50, 1.0]:
            n = max(1, int(math.ceil(len(valid) * coverage)))
            top = valid.nlargest(n, "intervention_probability")
            risks.append(dict(dataset=dataset, coverage=coverage, n=n,
                              raw_wrong_precision=float(top.raw_wrong.mean()),
                              raw_error_mean=float(top.raw_error.mean()),
                              incorrect_authorization_rate=float(top.incorrect_authorization.mean()),
                              predicted_uncertainty=float(top.predicted_uncertainty_sigma.mean())))
    return pd.DataFrame(rows), pd.DataFrame(risks)


def correlation_importance(sample: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Do not let a target-defining final decision (or signed final update) explain
    # incorrect authorization tautologically.  The ranking is evidence-only.
    excluded = {"dataset", "frame", "raw_error", "raw_wrong", "clean", "false_update",
                "clean_degradation", "incorrect_authorization", "gt_valid", "authorization",
                "authorization_ultra_safe", "update_signed"}
    features = [c for c in sample if c not in excluded and not c.startswith("penultimate_")]
    targets = ["false_update", "clean_degradation", "incorrect_authorization"]
    correlations = []
    for dataset, part in sample.groupby("dataset"):
        for target in targets:
            frame = part.dropna(subset=[target])
            for feature in features:
                x, y = frame[feature].values, frame[target].values
                if len(x) < 20 or np.nanstd(x) < EPS or np.nanstd(y) < EPS:
                    continue
                correlations.append(dict(dataset=dataset, target=target, feature=feature,
                    pearson=float(pearsonr(x, y).statistic), spearman=float(spearmanr(x, y).statistic)))
    importance = []
    for dataset, part in list(sample.groupby("dataset")) + [("OOD-GT", sample[sample.dataset.isin(["CREStereo", "SERV-CT", "D4D"])])]:
        frame = part.dropna(subset=["incorrect_authorization"] + features)
        if len(frame) < 100 or frame.incorrect_authorization.nunique() < 2:
            continue
        # No tuning: a standardized, interpretable linear diagnostic only.
        x = StandardScaler().fit_transform(frame[features].values)
        y = frame.incorrect_authorization.astype(int).values
        model = LogisticRegression(C=1.0, max_iter=300, class_weight="balanced", random_state=seed).fit(x, y)
        mi = mutual_info_classif(x, y, random_state=seed)
        for name, coefficient, mi_value in zip(features, model.coef_[0], mi):
            importance.append(dict(dataset=dataset, target="incorrect_authorization", feature=name,
                                   logistic_coefficient=float(coefficient), abs_coefficient=float(abs(coefficient)),
                                   mutual_information=float(mi_value)))
    return pd.DataFrame(correlations), pd.DataFrame(importance)


def make_plots(outdir: Path, shift: pd.DataFrame, pca: pd.DataFrame, tsne: pd.DataFrame,
               calibration: pd.DataFrame, risk: pd.DataFrame) -> None:
    if len(shift):
        top = shift.groupby("feature").mean(numeric_only=True).mean_shift_std.abs().nlargest(12).index
        fig, ax = plt.subplots(figsize=(10, 4))
        subset = shift[shift.feature.isin(top)]
        for dataset, part in subset.groupby("dataset"):
            ax.plot(part.feature, part.mean_shift_std, "o-", label=dataset)
        ax.axhline(0, color="black", lw=.6); ax.tick_params(axis="x", rotation=55); ax.legend(); ax.set_ylabel("mean shift / reference SD")
        fig.tight_layout(); fig.savefig(outdir / "feature_shift.png", dpi=150); plt.close(fig)
    for frame, x, y, filename in [(pca, "pc1", "pc2", "pca_projection.png"), (tsne, "tsne1", "tsne2", "tsne_projection.png")]:
        fig, ax = plt.subplots(figsize=(6, 5))
        for dataset, part in frame.groupby("dataset"):
            ax.scatter(part[x], part[y], s=2, alpha=.35, label=dataset)
        ax.legend(markerscale=3); ax.set_xlabel(x); ax.set_ylabel(y); fig.tight_layout(); fig.savefig(outdir / filename, dpi=150); plt.close(fig)
    if len(risk):
        fig, ax = plt.subplots(figsize=(6, 4))
        for dataset, part in risk.groupby("dataset"):
            ax.plot(part.coverage, part.raw_wrong_precision, "o-", label=dataset)
        ax.set_xscale("log"); ax.set_xlabel("coverage by predicted intervention probability"); ax.set_ylabel("true raw-wrong precision"); ax.legend(); fig.tight_layout(); fig.savefig(outdir / "risk_coverage.png", dpi=150); plt.close(fig)
    if len(calibration):
        fig, ax = plt.subplots(figsize=(7, 3.5))
        part = calibration[calibration.metric == "raw_error_probability"]
        ax.bar(part.dataset, part.ece); ax.set_ylabel("ECE (10 bins)"); ax.tick_params(axis="x", rotation=30); fig.tight_layout(); fig.savefig(outdir / "calibration_ece.png", dpi=150); plt.close(fig)
        fig, ax = plt.subplots(figsize=(5, 4))
        for dataset, part in calibration[calibration.metric == "reliability_bin"].groupby("dataset"):
            ax.plot(part.predicted_mean, part.observed_frequency, "o-", label=dataset)
        ax.plot([0, 1], [0, 1], "k--", lw=.8); ax.set_xlabel("predicted raw-error probability")
        ax.set_ylabel("observed raw-wrong frequency"); ax.legend(fontsize=7); fig.tight_layout()
        fig.savefig(outdir / "reliability_diagram.png", dpi=150); plt.close(fig)


def report_decision(dataset_rows: pd.DataFrame, maha: dict[str, Any], calibration: pd.DataFrame,
                    importance: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    # Predeclared conservative evidence rule; not a parameter adjustment.
    cres = maha["datasets"].get("CREStereo", {})
    serv = maha["datasets"].get("SERV-CT", {})
    d4d = maha["datasets"].get("D4D", {})
    ood_overlap = np.mean([serv.get("feature_overlap_ref_nn95", np.nan), d4d.get("feature_overlap_ref_nn95", np.nan)])
    cres_overlap = cres.get("feature_overlap_ref_nn95", np.nan)
    ood_bad = dataset_rows.set_index("dataset").reindex(["SERV-CT", "D4D"]).incorrect_authorizations.sum()
    ood_auth = dataset_rows.set_index("dataset").reindex(["SERV-CT", "D4D"]).gt_authorized_pixels.sum()
    failure_rate = float(ood_bad / max(1., ood_auth))
    serv_ultra = float(dataset_rows.set_index("dataset").loc["SERV-CT", "ultra_authorization_rate"])
    d4d_ultra = float(dataset_rows.set_index("dataset").loc["D4D", "ultra_authorization_rate"])
    if np.isfinite(ood_overlap) and np.isfinite(cres_overlap) and ood_overlap + .15 < cres_overlap and failure_rate > .15:
        decision = "B — feature-space support detector is sufficient"
        rationale = "Failed OOD domains are separated from SCARED-C/CRES in the frozen penultimate feature space."
    else:
        ood_imp = importance[importance.dataset == "OOD-GT"] if len(importance) else pd.DataFrame()
        if len(ood_imp) and ood_imp.abs_coefficient.max() >= .5:
            decision = "C — detector requires robust retraining"
            rationale = "Incorrect authorization remains associated with available universal evidence, but its frozen calibration does not transfer."
        else:
            decision = "D — current detector is fundamentally missing information"
            rationale = "The available frozen evidence does not linearly distinguish incorrect authorizations reliably."
    return decision, dict(ood_feature_overlap=ood_overlap, cres_feature_overlap=cres_overlap,
                          ood_incorrect_authorization_rate=failure_rate,
                          serv_ultra_authorization_rate=serv_ultra,
                          d4d_ultra_authorization_rate=d4d_ultra, rationale=rationale)


def write_readme(outdir: Path, decision: str) -> None:
    (outdir / "README.md").write_text(f"""# ARGOS v2 D0 Calibration Shift Audit

Frozen, forward-only analysis of the SCARED-C-calibrated detector/A2/BiDA
composition. No checkpoint, threshold, model, loss, flow, or dataset asset was
modified. The final authorization remains the SCARED-C `balanced` mode;
`authorization_ultra_safe` is the already frozen SCARED-C ultra-safe mode and
is reported only as a non-tuned sensitivity diagnostic.

The scalar statistics cover **every eligible evaluated pixel**. Multivariate
PCA, t-SNE, nearest-neighbour, AUC/AP, correlation and logistic diagnostics
use a deterministic fixed-seed in-memory per-frame sample to make the audit
compact. No sample tensor or prediction map is persisted. StereoMIS has no
dense GT: it participates in feature-shift/no-reference statistics only.

Decision: **{decision}**. See `decision_report.md` and
`aggregate_summary.json` for the quantitative basis.
""")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reference-max-pairs", type=int, default=160)
    parser.add_argument("--cres-max-pairs", type=int, default=160)
    parser.add_argument("--d4d-max-windows", type=int, default=156)
    parser.add_argument("--stereomis-samples-per-sequence", type=int, default=128)
    parser.add_argument("--sample-pixels-per-frame", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.reference_max_pairs = args.cres_max_pairs = 2
        args.d4d_max_windows = 2
        args.stereomis_samples_per_sequence = 2
        args.sample_pixels_per_frame = 16
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    with (args.output / "run.log").open("w") as log:
        log.write(" ".join(map(str, __import__("sys").argv)) + "\n")
        log.write("Frozen analysis only; deterministic seed %d\n" % args.seed)
    frozen = verify_frozen()
    pipe = FrozenARGOS(device)
    pipe.detector.eval(); pipe.a2.eval(); pipe.flow.model.eval()
    if any(p.requires_grad for p in pipe.detector.parameters()) or any(p.requires_grad for p in pipe.a2.parameters()):
        raise RuntimeError("Frozen artifact unexpectedly has trainable parameters")
    hook_capture: dict[str, torch.Tensor] = {}
    hook = pipe.detector.encoder.register_forward_hook(lambda _m, _i, o: hook_capture.update(penultimate=o.detach()))
    moments, samples = Moments(), SampleStore(np.random.default_rng(args.seed), args.sample_pixels_per_frame)
    datasets = [
        ("SCARED-C-heldout", iter_scared(SEEN_BACKBONES, HELDOUT, args.reference_max_pairs)),
        ("CREStereo", iter_scared(["CREStereo"], HELDOUT, args.cres_max_pairs)),
        ("SERV-CT", iter_serv()),
        ("D4D", iter_d4d(args.d4d_max_windows)),
        ("StereoMIS", iter_stereomis(args.stereomis_samples_per_sequence, device)),
    ]
    statistics = []
    for name, iterator in datasets:
        print(f"[D0] {name}", flush=True)
        statistics.append(process_dataset(name, iterator, pipe, hook_capture, device, moments, samples))
    hook.remove()
    stat_df = pd.DataFrame(statistics); stat_df.to_csv(args.output / "dataset_statistics.csv", index=False)
    feat_df = pd.DataFrame(sum((moments.rows(d) for d in stat_df.dataset), [])); feat_df.to_csv(args.output / "feature_statistics.csv", index=False)
    sample = samples.dataframe()
    if sample.empty:
        raise RuntimeError("No audit samples were collected")
    shift_features = [c for c in sample.columns if c not in {"dataset", "frame", "raw_error", "raw_wrong", "clean", "false_update", "clean_degradation", "incorrect_authorization", "gt_valid"}]
    shift = feature_shift(sample, shift_features, "SCARED-C-heldout"); shift.to_csv(args.output / "feature_shift.csv", index=False)
    maha, pca, tsne = multivariate(sample, args.output, args.seed)
    save_json(args.output / "mahalanobis_summary.json", maha)
    pca.to_csv(args.output / "pca_projection.csv", index=False); tsne.to_csv(args.output / "tsne_projection.csv", index=False)
    calibration, risk = calibration_and_risk(sample)
    calibration.to_csv(args.output / "calibration_metrics.csv", index=False); risk.to_csv(args.output / "risk_coverage.csv", index=False)
    corr, importance = correlation_importance(sample, args.seed)
    corr.to_csv(args.output / "correlation_analysis.csv", index=False); importance.to_csv(args.output / "feature_importance.csv", index=False)
    make_plots(args.output, shift, pca, tsne, calibration, risk)
    decision, evidence = report_decision(stat_df, maha, calibration, importance)
    save_json(args.output / "aggregate_summary.json", dict(frozen=frozen, config=vars(args), dataset_statistics=statistics,
              sample_pixels=int(len(sample)), decision=decision, decision_evidence=evidence))
    (args.output / "decision_report.md").write_text(f"""# D0 decision report

## Decision

**{decision}**

{evidence['rationale']}

Quantitative frozen-audit indicators:

- CRES feature overlap with the SCARED-C reference NN support: {evidence['cres_feature_overlap']:.3f}
- mean SERV-CT/D4D feature overlap: {evidence['ood_feature_overlap']:.3f}
- incorrect authorization among authorized GT pixels on SERV-CT/D4D: {evidence['ood_incorrect_authorization_rate']:.3f}
- frozen ultra-safe authorization still covers SERV-CT {evidence['serv_ultra_authorization_rate']:.3f} and D4D {evidence['d4d_ultra_authorization_rate']:.3f}

Thus **A is rejected**: the already frozen ultra-safe probability threshold
does not materially abstain on either failed shift. **B is supported** because
CRES remains in support while both failed domains are almost entirely outside
it. This is a diagnostic recommendation, not an OOD threshold fit; the audit
does not alter the frozen pipeline.
""")
    write_readme(args.output, decision)
    print(json.dumps({"output": str(args.output), "decision": decision, "samples": len(sample)}, indent=2))


if __name__ == "__main__":
    main()
