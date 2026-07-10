#!/usr/bin/env python3
"""Run small ARGOS v2 external-component probes on validated SCARED-C cache entries.

This is a diagnostic probe, not a final refiner. It uses official RAFT/SEA-RAFT paths
when `--with-flow` is set and records blocked components instead of substituting fake flow.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "ARGOS-V2"
OUT = V2 / "component_probes"
sys.path.insert(0, str(V2 / "scripts"))
sys.path.insert(0, str(V2))
sys.path.insert(0, str(OUT))

from argos_v2.cache_io import is_complete, load_sequence_cache  # noqa: E402
from argos_v2.metrics import resize_gt_to_cache_corrected  # noqa: E402
from argos_v2.paths import CACHE_HEIGHT, CACHE_WIDTH  # noqa: E402
from argos_v2.scared_c_data import load_frame_gt, load_frame_lr, load_sequence_info  # noqa: E402
from model_design.external_components import bidavideo as bida  # noqa: E402
from model_design.external_components import endostreamdepth as endo  # noqa: E402
from model_design.external_components import ppmstereo as ppm  # noqa: E402

TRAIN_BACKBONES = ["S2M2-S", "RAFT-Stereo", "StereoAnywhere"]
SEQUENCES = ["dataset_3_keyframe_1", "dataset_7_keyframe_4"]
CASES = [("dataset_3_keyframe_1", 32), ("dataset_3_keyframe_1", 160), ("dataset_7_keyframe_4", 1099)]
AGES = [1, 2, 4, 8, 16]
VISUAL_GT_COVERAGE_THRESHOLD = 0.05
COVERAGE_THRESHOLDS = [0.05, 0.25, 0.50, 0.90]
FLOW_CACHE_DIR = OUT / "flow_cache"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def fmean(x: np.ndarray) -> float | None:
    return float(x.mean()) if x.size else None


def corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def resized_left_rgb(sequence_id: str, frame_id: str) -> np.ndarray:
    info = load_sequence_info(sequence_id)
    left, _right = load_frame_lr(info, frame_id)
    return cv2.resize(left, (CACHE_WIDTH, CACHE_HEIGHT), interpolation=cv2.INTER_AREA)


def gt_cache(sequence_id: str, frame_id: str, coverage_threshold: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    info = load_sequence_info(sequence_id)
    disp, valid = load_frame_gt(info, frame_id)
    return resize_gt_to_cache_corrected(disp, valid, disp.shape[1], coverage_threshold=coverage_threshold)


def upsample_disp_to_native(disp_cache: np.ndarray, native_shape: tuple[int, int]) -> np.ndarray:
    native_h, native_w = native_shape
    disp_native = cv2.resize(disp_cache.astype(np.float32), (native_w, native_h), interpolation=cv2.INTER_LINEAR)
    return disp_native * (native_w / float(CACHE_WIDTH))


def upsample_mask_to_native(mask_cache: np.ndarray, native_shape: tuple[int, int]) -> np.ndarray:
    native_h, native_w = native_shape
    return cv2.resize(mask_cache.astype(np.uint8), (native_w, native_h), interpolation=cv2.INTER_NEAREST) > 0


def epe_count_ratio(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> tuple[float | None, int, float]:
    m = mask & np.isfinite(pred) & np.isfinite(gt) & (gt > 0)
    return epe(pred, gt, m), int(m.sum()), float(m.mean())


def cache_case(backbone: str, sequence_id: str) -> dict:
    disp, valid, frame_ids, metadata = load_sequence_cache(backbone, sequence_id, mmap=True)
    return {"disp": disp, "valid": valid, "frame_ids": [str(x) for x in frame_ids], "metadata": metadata}


def epe(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float | None:
    m = mask & np.isfinite(pred) & np.isfinite(gt) & (gt > 0)
    if not m.any():
        return None
    return float(np.abs(pred[m].astype(np.float32) - gt[m].astype(np.float32)).mean())


def boundary_mask(gt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(gt, dtype=np.float32)
    gy = np.zeros_like(gt, dtype=np.float32)
    gx[:, 1:] = np.abs(gt[:, 1:] - gt[:, :-1])
    gy[1:, :] = np.abs(gt[1:, :] - gt[:-1, :])
    grad = gx + gy
    vals = grad[valid & np.isfinite(grad)]
    if vals.size == 0:
        return np.zeros_like(valid)
    return valid & (grad >= np.quantile(vals, 0.90))


def torch_warp(source: np.ndarray, flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        src = torch.from_numpy(source.astype(np.float32))[None, None]
        flw = torch.from_numpy(flow.astype(np.float32))[None]
        warped = bida.warp_disparity_original(src, flw)[0, 0].cpu().numpy()
        support = bida.support_mask(src, flw)[0, 0].cpu().numpy() > 0.5
    return warped, support


def fb_consistency(flow_ab: np.ndarray, flow_ba: np.ndarray) -> tuple[float, float]:
    wx, support_x = torch_warp(flow_ba[0], flow_ab)
    wy, support_y = torch_warp(flow_ba[1], flow_ab)
    support = support_x & support_y
    err = np.sqrt((flow_ab[0] + wx) ** 2 + (flow_ab[1] + wy) ** 2)
    if not support.any():
        return math.nan, 0.0
    return float(err[support].mean()), float((support & (err < 1.0)).mean())


def load_flow_models(names: list[str]) -> tuple[dict, list[dict]]:
    models: dict = {}
    rows: list[dict] = []
    for name in names:
        t0 = time.perf_counter()
        try:
            if name == "RAFT":
                from model_design.external_components.raft import OfficialRAFT
                models[name] = OfficialRAFT()
            elif name == "SEA-RAFT":
                from model_design.external_components.sea_raft import OfficialSEARAFT
                models[name] = OfficialSEARAFT()
            else:
                raise ValueError(f"unknown flow model {name}")
            rows.append({
                "component": name,
                "status": "pass",
                "runtime_ms": (time.perf_counter() - t0) * 1000.0,
                "peak_memory_mb": 0.0,
                "notes": "official model loaded",
            })
        except Exception as exc:
            rows.append({
                "component": name,
                "status": "blocked",
                "runtime_ms": (time.perf_counter() - t0) * 1000.0,
                "peak_memory_mb": 0.0,
                "notes": f"{type(exc).__name__}: {exc}",
            })
    return models, rows


def flow_cache_path(flow_name: str, sequence_id: str, cur_idx: int, age: int, direction: str) -> Path:
    return FLOW_CACHE_DIR / flow_name / f"{sequence_id}_idx{cur_idx:06d}_age{age}_{direction}.npy"


def compute_flows(models: dict, cases: list[tuple[str, int]], generate_missing: bool = True) -> tuple[dict, list[dict]]:
    flows: dict = {}
    rows: list[dict] = []
    for flow_name, model in models.items():
        for sequence_id, cur_idx in cases:
            info = load_sequence_info(sequence_id)
            for age in AGES:
                past_idx = cur_idx - age
                if past_idx < 0:
                    continue
                cur_id = info.frame_ids[cur_idx]
                past_id = info.frame_ids[past_idx]
                cur_img = resized_left_rgb(sequence_id, cur_id)
                past_img = resized_left_rgb(sequence_id, past_id)
                for direction, img_a, img_b in [
                    ("current_to_past", cur_img, past_img),
                    ("past_to_current", past_img, cur_img),
                ]:
                    key = (flow_name, sequence_id, cur_idx, age, direction)
                    cache_path = flow_cache_path(flow_name, sequence_id, cur_idx, age, direction)
                    try:
                        if cache_path.exists():
                            flow = np.load(cache_path).astype(np.float32)
                            runtime_ms = 0.0
                            peak_memory_mb = 0.0
                            status = "pass_cached"
                        elif generate_missing:
                            out = model.infer(img_a, img_b)
                            flow = out["flow"]
                            cache_path.parent.mkdir(parents=True, exist_ok=True)
                            np.save(cache_path, flow.astype(np.float32))
                            runtime_ms = out["runtime_ms"]
                            peak_memory_mb = out["peak_memory_mb"]
                            status = "pass"
                        else:
                            raise FileNotFoundError(cache_path)
                        flows[key] = flow
                        rows.append({
                            "flow_model": flow_name,
                            "sequence_id": sequence_id,
                            "current_frame": cur_id,
                            "age": age,
                            "direction": direction,
                            "runtime_ms": runtime_ms,
                            "peak_memory_mb": peak_memory_mb,
                            "status": status,
                            "notes": "",
                        })
                    except Exception as exc:
                        rows.append({
                            "flow_model": flow_name,
                            "sequence_id": sequence_id,
                            "current_frame": cur_id,
                            "age": age,
                            "direction": direction,
                            "runtime_ms": None,
                            "peak_memory_mb": None,
                            "status": "fail",
                            "notes": f"{type(exc).__name__}: {exc}",
                        })
    return flows, rows


def run_bida_probe(flows: dict, flow_rows: list[dict], cases: list[tuple[str, int]]) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    flow_summary: list[dict] = []
    for flow_name in sorted({r["flow_model"] for r in flow_rows if str(r["status"]).startswith("pass")}):
        for sequence_id, cur_idx in cases:
            flow_cp = flows.get((flow_name, sequence_id, cur_idx, 1, "current_to_past"))
            flow_pc = flows.get((flow_name, sequence_id, cur_idx, 1, "past_to_current"))
            if flow_cp is None or flow_pc is None:
                continue
            fb_mean, fb_valid = fb_consistency(flow_cp, flow_pc)
            info = load_sequence_info(sequence_id)
            cur_id = info.frame_ids[cur_idx]
            past_id = info.frame_ids[cur_idx - 1]
            runtimes = [
                r for r in flow_rows
                if r["flow_model"] == flow_name and r["sequence_id"] == sequence_id
                and r["current_frame"] == cur_id and r["age"] == 1 and str(r["status"]).startswith("pass")
            ]
            gt_native, gt_native_valid = load_frame_gt(info, cur_id)
            for backbone in TRAIN_BACKBONES:
                if not is_complete(backbone, sequence_id):
                    continue
                cache = cache_case(backbone, sequence_id)
                raw = np.asarray(cache["disp"][cur_idx], dtype=np.float32)
                pred_valid = np.asarray(cache["valid"][cur_idx]) > 0
                prev = np.asarray(cache["disp"][cur_idx - 1], dtype=np.float32)
                prev_valid = np.asarray(cache["valid"][cur_idx - 1], dtype=np.float32)
                aligned, support = torch_warp(prev, flow_cp)
                aligned_valid_f, support_valid = torch_warp(prev_valid, flow_cp)
                aligned_valid = support & support_valid & (aligned_valid_f > 0.5)
                raw_native = upsample_disp_to_native(raw, gt_native.shape)
                aligned_native = upsample_disp_to_native(aligned, gt_native.shape)
                raw_valid_native = upsample_mask_to_native(pred_valid, gt_native.shape)
                aligned_valid_native = upsample_mask_to_native(aligned_valid, gt_native.shape)
                native_common = gt_native_valid & raw_valid_native & aligned_valid_native
                raw_native_epe, native_count, native_ratio = epe_count_ratio(raw_native, gt_native, native_common)
                aligned_native_epe, _, _ = epe_count_ratio(aligned_native, gt_native, native_common)
                for coverage_threshold in COVERAGE_THRESHOLDS:
                    gt, gt_valid = gt_cache(sequence_id, cur_id, coverage_threshold)
                    edge = boundary_mask(gt, gt_valid)
                    common = gt_valid & pred_valid & aligned_valid
                    raw_epe, common_count, common_ratio = epe_count_ratio(raw, gt, common)
                    aligned_epe, _, _ = epe_count_ratio(aligned, gt, common)
                    raw_err = np.abs(raw - gt)
                    aligned_err = np.abs(aligned - gt)
                    row = {
                        "flow_model": flow_name,
                        "backbone": backbone,
                        "sequence_id": sequence_id,
                        "current_frame": cur_id,
                        "previous_frame": past_id,
                        "coverage_threshold": coverage_threshold,
                        "epe_raw_common_cache_px": raw_epe,
                        "epe_candidate_common_cache_px": aligned_epe,
                        "common_valid_count": common_count,
                        "common_valid_ratio": common_ratio,
                        "epe_raw_native_px": raw_native_epe,
                        "epe_aligned_native_px": aligned_native_epe,
                        "native_valid_count": native_count,
                        "native_valid_ratio": native_ratio,
                        "boundary_epe_raw_common_cache_px": epe(raw, gt, common & edge),
                        "boundary_epe_candidate_common_cache_px": epe(aligned, gt, common & edge),
                        "fb_consistency_mean_px": fb_mean,
                        "fb_valid_ratio_lt1px": fb_valid,
                        "aligned_better_frame": bool(aligned_epe is not None and raw_epe is not None and aligned_epe < raw_epe),
                        "aligned_better_pixel_ratio": fmean((aligned_err[common] < raw_err[common]).astype(np.float32)) if common.any() else None,
                        "clean_pixel_degradation_ratio": fmean(((raw_err[common] < 1.0) & (aligned_err[common] > raw_err[common] + 1.0)).astype(np.float32)) if common.any() else None,
                        "temporal_disagreement_px": fmean(np.abs(raw[common] - aligned[common])) if common.any() else None,
                        "notes": "BiDAVideo original flow_warp; common-mask evaluation",
                    }
                    rows.append(row)
            flow_summary.append({
                "flow_model": flow_name,
                "sequence_id": sequence_id,
                "current_frame": cur_id,
                "age": 1,
                "runtime_ms_pair_mean": float(np.mean([float(r["runtime_ms"]) for r in runtimes])) if runtimes else None,
                "peak_memory_mb_pair_max": float(np.max([float(r["peak_memory_mb"]) for r in runtimes])) if runtimes else None,
                "valid_warp_ratio": float(bida.support_mask(torch.zeros(1, 1, CACHE_HEIGHT, CACHE_WIDTH), torch.from_numpy(flow_cp)[None]).mean().item()),
                "forward_backward_consistency_mean_px": fb_mean,
                "forward_backward_valid_ratio_lt1px": fb_valid,
            })
    return rows, flow_summary


def feature_stack(current: np.ndarray, candidates: list[np.ndarray], supports: list[np.ndarray], ages: list[int]) -> torch.Tensor:
    all_disp = [current] + candidates
    med = np.median(current[np.isfinite(current) & (current > 0)])
    med = float(med) if np.isfinite(med) and med > 0 else 1.0
    feats = []
    for i, disp in enumerate(all_disp):
        support = np.ones_like(current, dtype=np.float32) if i == 0 else supports[i - 1].astype(np.float32)
        age = 0 if i == 0 else ages[i - 1]
        grad = np.zeros_like(current, dtype=np.float32)
        grad[:, 1:] += np.abs(disp[:, 1:] - disp[:, :-1])
        grad[1:, :] += np.abs(disp[1:, :] - disp[:-1, :])
        diff = np.abs(disp - current) / med
        age_plane = np.full_like(current, age / max(ages), dtype=np.float32)
        feats.append(np.stack([
            disp / med,
            np.clip(grad / med, 0, 5),
            support,
            np.clip(diff, 0, 5),
            np.exp(-np.clip(diff, 0, 5)),
            age_plane,
        ], axis=0))
    return torch.from_numpy(np.stack(feats, axis=1)[None].astype(np.float32))


def aggregate(candidates: list[np.ndarray], supports: list[np.ndarray], indices: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    num = np.zeros_like(candidates[0], dtype=np.float32)
    den = np.zeros_like(candidates[0], dtype=np.float32)
    for idx, weight in zip(indices, weights):
        sup = supports[int(idx)].astype(np.float32)
        num += candidates[int(idx)] * sup * float(weight)
        den += sup * float(weight)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 1e-6), den > 1e-6


def true_oracle_metrics(raw: np.ndarray, raw_valid: np.ndarray, candidates: list[np.ndarray], supports: list[np.ndarray],
                        gt: np.ndarray, gt_valid: np.ndarray) -> dict:
    support_any = np.logical_or.reduce(supports) if supports else np.zeros_like(gt_valid)
    mask = gt_valid & raw_valid & support_any
    if not mask.any():
        return {
            "oracle_best_memory_per_pixel_epe": None,
            "oracle_raw_or_memory_per_pixel_epe": None,
            "oracle_best_memory_per_region_epe": None,
            "oracle_gain_over_raw": None,
        }
    raw_err = np.abs(raw - gt)
    err_stack = np.stack([np.where(s, np.abs(c - gt), np.inf) for c, s in zip(candidates, supports)], axis=0)
    best_mem = err_stack.min(axis=0)
    raw_e = float(raw_err[mask].mean())
    best_mem_e = float(best_mem[mask].mean())
    raw_or_mem_e = float(np.minimum(raw_err, best_mem)[mask].mean())
    region_e = region_oracle_epe(err_stack, mask)
    return {
        "oracle_best_memory_per_pixel_epe": best_mem_e,
        "oracle_raw_or_memory_per_pixel_epe": raw_or_mem_e,
        "oracle_best_memory_per_region_epe": region_e,
        "oracle_gain_over_raw": raw_e - raw_or_mem_e,
    }


def region_oracle_epe(err_stack: np.ndarray, mask: np.ndarray, grid: int = 4) -> float | None:
    vals = []
    h, w = mask.shape
    for y0 in np.linspace(0, h, grid + 1, dtype=int)[:-1]:
        y1 = min(h, y0 + math.ceil(h / grid))
        for x0 in np.linspace(0, w, grid + 1, dtype=int)[:-1]:
            x1 = min(w, x0 + math.ceil(w / grid))
            m = mask[y0:y1, x0:x1]
            if not m.any():
                continue
            region_errs = err_stack[:, y0:y1, x0:x1]
            means = [float(e[m & np.isfinite(e)].mean()) if np.isfinite(e[m]).any() else np.inf for e in region_errs]
            best = int(np.argmin(means))
            selected = region_errs[best][m & np.isfinite(region_errs[best])]
            if selected.size:
                vals.append(selected)
    if not vals:
        return None
    return float(np.concatenate(vals).mean())


def run_ppm_probe(flows: dict, cases: list[tuple[str, int]]) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    candidate_signal_rows: list[dict] = []
    flow_name = "SEA-RAFT" if any(k[0] == "SEA-RAFT" for k in flows) else ("RAFT" if any(k[0] == "RAFT" for k in flows) else None)
    if flow_name is None:
        return rows, candidate_signal_rows
    for sequence_id, cur_idx in cases:
        info = load_sequence_info(sequence_id)
        cur_id = info.frame_ids[cur_idx]
        gt_native, gt_native_valid = load_frame_gt(info, cur_id)
        for backbone in TRAIN_BACKBONES:
            if not is_complete(backbone, sequence_id):
                continue
            cache = cache_case(backbone, sequence_id)
            current = np.asarray(cache["disp"][cur_idx], dtype=np.float32)
            current_valid = np.asarray(cache["valid"][cur_idx]) > 0
            candidates: list[np.ndarray] = []
            supports: list[np.ndarray] = []
            used_ages: list[int] = []
            for age in AGES:
                flow = flows.get((flow_name, sequence_id, cur_idx, age, "current_to_past"))
                if flow is None:
                    continue
                past = np.asarray(cache["disp"][cur_idx - age], dtype=np.float32)
                past_valid = np.asarray(cache["valid"][cur_idx - age], dtype=np.float32)
                warped, support = torch_warp(past, flow)
                warped_valid, support_valid = torch_warp(past_valid, flow)
                candidates.append(warped)
                supports.append(support & support_valid & (warped_valid > 0.5))
                used_ages.append(age)
            if not candidates:
                continue
            feats = feature_stack(current, candidates, supports, used_ages)
            sim_matrix = ppm.compute_qk_similarity_exact(feats, feats, t=len(used_ages) + 1)
            similarity = sim_matrix[0, 0, 0, 1:].float()
            med = np.median(current[current_valid & np.isfinite(current)])
            med = float(med) if np.isfinite(med) and med > 0 else 1.0
            confidence = torch.tensor([
                float(s.mean()) * math.exp(-float(np.abs(c - current)[s].mean() / med)) if s.any() else 0.0
                for c, s in zip(candidates, supports)
            ], dtype=torch.float32)
            score, penalty = ppm.quality_aware_scores(similarity[None], confidence[None])
            clean_quality = confidence[None]
            redundancy_vals = []
            for i, (candidate_i, support_i) in enumerate(zip(candidates, supports)):
                vals = []
                for j, (candidate_j, support_j) in enumerate(zip(candidates, supports)):
                    common_support = support_i & support_j
                    if i == j or common_support.sum() < 10:
                        continue
                    a = candidate_i[common_support].astype(np.float32)
                    b = candidate_j[common_support].astype(np.float32)
                    n = min(a.size, b.size, 2000)
                    vals.append(float(np.corrcoef(a[:n], b[:n])[0, 1]) if np.std(a[:n]) > 0 and np.std(b[:n]) > 0 else 0.0)
                redundancy_vals.append(max(vals) if vals else 0.0)
            redundancy = torch.tensor([redundancy_vals], dtype=torch.float32)
            ages_t = torch.tensor([used_ages], dtype=torch.float32)
            sys.path.insert(0, str(V2 / "external_code_backbone_needed"))
            try:
                from ppmstereo.memory_selection.pick_and_play import score_memory, select_topk_and_weights  # type: ignore
                clean_scores = score_memory(clean_quality, similarity[None], redundancy, torch.ones_like(score), ages_t)
                clean_indices, clean_weights = select_topk_and_weights(clean_scores, k=min(5, len(used_ages)))
            finally:
                sys.path.remove(str(V2 / "external_code_backbone_needed"))

            for coverage_threshold in COVERAGE_THRESHOLDS:
                gt, gt_valid = gt_cache(sequence_id, cur_id, coverage_threshold)
                for age, candidate, support, sim, conf, sc in zip(used_ages, candidates, supports, similarity, confidence, score[0]):
                    common = gt_valid & current_valid & support
                    candidate_epe, common_count, common_ratio = epe_count_ratio(candidate, gt, common)
                    raw_candidate_epe, _, _ = epe_count_ratio(current, gt, common)
                    candidate_signal_rows.append({
                        "sequence_id": sequence_id,
                        "backbone": backbone,
                        "current_frame": cur_id,
                        "coverage_threshold": coverage_threshold,
                        "age": age,
                        "ppm_similarity": float(sim),
                        "ppm_confidence_adapter": float(conf),
                        "ppm_score": float(sc),
                        "support_ratio": float(support.mean()),
                        "epe_raw_common_cache_px": raw_candidate_epe,
                        "candidate_epe_cache_px": candidate_epe,
                        "common_valid_count": common_count,
                        "common_valid_ratio": common_ratio,
                    })

            for k in [1, 3, 5]:
                k_eff = min(k, len(used_ages))
                baselines = []
                baselines.append(("t_minus_1_only", np.array([0]), np.array([1.0], dtype=np.float32)))
                baselines.append(("uniform_all", np.arange(len(used_ages)), np.ones(len(used_ages), dtype=np.float32) / len(used_ages)))
                idx, values, weights = ppm.topk_with_modulation(score, k_eff)
                baselines.append(("ppmstereo_original_score_topk", idx[0].numpy(), np.ones(k_eff, dtype=np.float32) / k_eff))
                baselines.append(("ppmstereo_original_modulated_diag", idx[0].numpy(), weights[0].numpy()))
                clean_k = min(k_eff, clean_indices.shape[1])
                baselines.append(("cleanroom_exported_topk", clean_indices[0, :clean_k].numpy(), clean_weights[0, :clean_k].numpy()))
                for method, selected, w in baselines:
                    agg, agg_valid = aggregate(candidates, supports, selected, w)
                    memory_native = upsample_disp_to_native(agg, gt_native.shape)
                    current_native = upsample_disp_to_native(current, gt_native.shape)
                    agg_valid_native = upsample_mask_to_native(agg_valid, gt_native.shape)
                    current_valid_native = upsample_mask_to_native(current_valid, gt_native.shape)
                    native_common = gt_native_valid & current_valid_native & agg_valid_native
                    raw_native_epe, native_count, native_ratio = epe_count_ratio(current_native, gt_native, native_common)
                    memory_native_epe, _, _ = epe_count_ratio(memory_native, gt_native, native_common)
                    for coverage_threshold in COVERAGE_THRESHOLDS:
                        gt, gt_valid = gt_cache(sequence_id, cur_id, coverage_threshold)
                        common = gt_valid & agg_valid & current_valid
                        raw_epe, common_count, common_ratio = epe_count_ratio(current, gt, common)
                        method_epe, _, _ = epe_count_ratio(agg, gt, common)
                        oracle = true_oracle_metrics(current, current_valid, candidates, supports, gt, gt_valid)
                        rows.append({
                            "flow_model_for_warp": flow_name,
                            "backbone": backbone,
                            "sequence_id": sequence_id,
                            "current_frame": cur_id,
                            "coverage_threshold": coverage_threshold,
                            "k": k,
                            "method": method,
                            "selected_ages": ";".join(str(used_ages[int(i)]) for i in selected),
                            "weights": ";".join(f"{float(x):.4f}" for x in w),
                            "epe_raw_common_cache_px": raw_epe,
                            "epe_candidate_common_cache_px": method_epe,
                            "memory_epe_cache_px": method_epe,
                            "common_valid_count": common_count,
                            "common_valid_ratio": common_ratio,
                            "epe_raw_native_px": raw_native_epe,
                            "epe_memory_native_px": memory_native_epe,
                            "native_valid_count": native_count,
                            "native_valid_ratio": native_ratio,
                            "improves_over_raw": bool(method_epe is not None and raw_epe is not None and method_epe < raw_epe),
                            "clean_pixel_degradation_ratio": None if not common.any() else fmean(((np.abs(current - gt)[common] < 1.0) & (np.abs(agg - gt)[common] > np.abs(current - gt)[common] + 1.0)).astype(np.float32)),
                            **oracle,
                            "notes": "PPMStereo scoring/top-k math evaluated with a deterministic untrained ARGOS feature adapter; full flash_attn readout not isolated",
                        })
    return rows, candidate_signal_rows


def run_endo_probe() -> list[dict]:
    rows = endo.import_status()
    sequence_id, cur_idx = CASES[0]
    cache = cache_case("S2M2-S", sequence_id)
    start = max(0, cur_idx - 8)
    pred = torch.from_numpy(np.asarray(cache["disp"][start:cur_idx], dtype=np.float32))[None]
    valid = torch.from_numpy((np.asarray(cache["valid"][start:cur_idx]) > 0).astype(np.float32))[None]
    frozen = pred[:, :1].repeat(1, pred.shape[1], 1, 1)
    try:
        raw_loss = float(endo.temporal_consistency_loss_actual(pred, valid))
        frozen_loss = float(endo.temporal_consistency_loss_actual(frozen, valid))
        rows.append({
            "component": "temporal_consistency_loss real S2M2 sequence",
            "status": "pass",
            "reason": f"raw_loss={raw_loss:.6f}; frozen_first_frame_loss={frozen_loss:.6f}; lower frozen means loss can reward freezing",
        })
    except Exception as exc:
        rows.append({"component": "temporal_consistency_loss real S2M2 sequence", "status": "fail", "reason": f"{type(exc).__name__}: {exc}"})
    return rows


def save_contact_sheets(bida_rows: list[dict], flows: dict) -> None:
    out_dir = OUT / "diagnostic_contact_sheets"
    out_dir.mkdir(exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()
    for row in bida_rows:
        sequence_id = row["sequence_id"]
        cur_id = row["current_frame"]
        cur_idx = load_sequence_info(sequence_id).frame_ids.index(cur_id)
        flow = flows.get((row["flow_model"], sequence_id, cur_idx, 1, "current_to_past"))
        if flow is None:
            continue
        cache = cache_case(row["backbone"], sequence_id)
        info = load_sequence_info(sequence_id)
        gt_native, valid_native = load_frame_gt(info, cur_id)
        gt, gt_valid = resize_gt_to_cache_corrected(
            gt_native, valid_native, gt_native.shape[1], coverage_threshold=VISUAL_GT_COVERAGE_THRESHOLD
        )
        raw = np.asarray(cache["disp"][cur_idx], dtype=np.float32)
        prev = np.asarray(cache["disp"][cur_idx - 1], dtype=np.float32)
        aligned, support = torch_warp(prev, flow)
        left = resized_left_rgb(sequence_id, cur_id)
        raw_err = np.where(gt_valid, np.abs(raw - gt), 0)
        aligned_err = np.where(gt_valid & support, np.abs(aligned - gt), 0)
        gt_panel = cv2.cvtColor((gt_valid.astype(np.uint8) * 255), cv2.COLOR_GRAY2RGB)
        support_panel = cv2.cvtColor((support.astype(np.uint8) * 255), cv2.COLOR_GRAY2RGB)
        panels = [left, gt_panel, heat(raw_err), heat(aligned_err), support_panel]
        sheet = np.concatenate(panels, axis=1)
        cv2.imwrite(str(out_dir / f"{row['flow_model']}_{row['backbone']}_{sequence_id}_{cur_id}.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def heat(arr: np.ndarray) -> np.ndarray:
    vals = arr[np.isfinite(arr)]
    vmax = np.quantile(vals, 0.98) if vals.size else 1.0
    img = np.clip(arr / max(float(vmax), 1e-6), 0, 1)
    return cv2.cvtColor(cv2.applyColorMap((img * 255).astype(np.uint8), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def write_docs(flow_models: list[str], flow_load_rows: list[dict], bida_rows: list[dict], ppm_rows: list[dict], endo_rows: list[dict],
               write_architecture: bool = True) -> None:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cache_sequences": SEQUENCES,
        "cases": [{"sequence_id": s, "frame_index": i} for s, i in CASES],
        "training_backbones_used_for_decisions": TRAIN_BACKBONES,
        "flow_models_requested": flow_models,
        "flow_model_load_status": flow_load_rows,
        "bida_rows": len(bida_rows),
        "ppm_rows": len(ppm_rows),
        "endostream_rows": len(endo_rows),
        "bida_original_losses_sha256": bida.source_sha256(),
    }
    (OUT / "probe_manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT / "README.md").write_text(
        "# ARGOS v2 Component Probes\n\n"
        "Small deterministic probes over validated SCARED-C cache entries. The scripts use cloned external "
        "implementations where executable, and mark dependency-coupled pieces as reference-only instead of "
        "substituting unrelated components.\n\n"
        "Run:\n\n```bash\n.miniconda/envs/argos/bin/python3 ARGOS-V2/component_probes/run_component_probes.py --with-flow --flow-models SEA-RAFT RAFT\n```\n"
    )
    (OUT / "repository_component_map.md").write_text(component_map_md())
    if write_architecture:
        (OUT / "architecture_decisions.md").write_text(decisions_md(flow_load_rows, bida_rows, ppm_rows, endo_rows))


def component_map_md() -> str:
    return """# Repository Component Map

| Mechanism | Repository | Source file | Original shape | Wrapper shape | Direct original code used | Causal | Future frames | Probe result |
|---|---|---|---|---|---|---|---|---|
| BiDA flow warp | `external/bidavideo` | `train_utils/losses.py::flow_warp` | tensor `[B,C,H,W]`, flow `[B,2,H,W]` | same | yes | yes if past-only | no | executed |
| BiDA propagation/fusion | `external/bidavideo` | `models/core/bidastabilizer.py::forward` | sequence `[B,T,C,H,W]` | none | inspected | no | yes | reference-only, backward/future pass |
| RAFT flow | `external/RAFT` | `demo.py`, `core/raft.py` | RGB pair `[B,3,H,W]` | same at 144x180 | yes | pairwise causal | no | see `raft_vs_searaft.csv` |
| SEA-RAFT flow | `external/SEA-RAFT` | `custom.py`, `core/raft.py` | RGB pair `[B,3,H,W]` | same at 144x180 | yes | pairwise causal | no | see `raft_vs_searaft.csv` |
| PPMStereo QK similarity | `external/PPMStereo` | `ppmstereo.py::compute_qk_similarity` | `[B,C,T,H,W]` | adapter features `[B,6,T,H,W]` | compared against original method | past-only in probe | no | executed |
| PPMStereo top-k/modulation | `external/PPMStereo` | `ppmstereo.py` lines 504-541 | score `[B,1,T,T]` | candidate score `[B,M]` | math preserved | past-only in probe | no | executed, readout diagnostic only |
| PPMStereo flash-attn readout | `external/PPMStereo` | `ppmstereo.py::forward_update_block` | cost/update features | none | no | sequence-level | possible | reference-only, cost-volume coupled |
| EndoStreamDepth Mamba/xLSTM | `external/EndoStreamDepth` | `mamba.py`, `xlstm_block.py` | DPT tokens `[B,L,C]` | none | import/instantiate tested | yes | no | dependency-coupled |
| Endo temporal loss | `external/EndoStreamDepth` | `util/loss.py::temporal_consistency_loss` | `[B,T,H,W]` | cache disparity `[B,T,144,180]` | yes | training loss | no | executed |
"""


def decisions_md(flow_load_rows: list[dict], bida_rows: list[dict], ppm_rows: list[dict], endo_rows: list[dict]) -> str:
    def avg(rows: list[dict], key: str) -> float | None:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    flow_status = ", ".join(f"{r['component']}={r['status']}" for r in flow_load_rows)
    raw = avg(bida_rows, "epe_raw_common_cache_px")
    aligned = avg(bida_rows, "epe_candidate_common_cache_px")
    aligned_better = sum(1 for r in bida_rows if r.get("aligned_better_frame") is True)
    endo_summary = "; ".join(f"{r['component']}:{r['status']}" for r in endo_rows)
    thresholds = sorted({r.get("coverage_threshold") for r in bida_rows + ppm_rows if r.get("coverage_threshold") is not None})
    return f"""# Preliminary Architecture Notes From Component Probes

All conclusions here are preliminary. Use the corrected CSVs for decisions: common-mask cache metrics,
native-grid metrics, coverage sensitivity, and true memory oracle fields are now explicit.

Flow load status: {flow_status or 'not run'}.

Mean raw current EPE: {raw}. Mean aligned previous EPE: {aligned}. Aligned-better frames: {aligned_better}/{len(bida_rows)}.
Coverage thresholds evaluated: {thresholds}.
PPMStereo ranking is intentionally not summarized here; tiny-mask minima are not architecture evidence.
Endo status: {endo_summary}.

1. Which BiDAStabilizer tricks are genuinely useful?
   Preliminary reusable trick: exact `grid + flow`, `align_corners=True` alignment convention plus support and forward-backward consistency signals.
2. Does explicit flow-based alignment improve usable temporal evidence?
   Preliminary: useful evidence, but not a direct replacement for raw disparity in this probe.
3. RAFT or SEA-RAFT?
   Preliminary: compare `raft_vs_searaft.csv`; keep both until the corrected metrics are reviewed.
4. Should flow run online or be precomputed?
   Preliminary: cached low-resolution flow is enough for this probe layer.
5. Does PPMStereo scoring/top-k math evaluated with a deterministic untrained ARGOS feature adapter beat simple recent-frame selection?
   Preliminary only; do not conclude until common-mask, native-grid, coverage-sensitivity, and true-oracle CSVs have been reviewed.
6. Does redundancy-aware selection matter?
   Preliminary only; the invalid global oracle has been removed.
7. Does dynamic memory modulation matter?
   Preliminary only; full flash-attn readout remains reference-only.
8. What K is supported by evidence?
   Preliminary only; no K conclusion until the corrected CSVs are inspected.
9. Is the actual EndoStreamDepth Mamba state useful on ARGOS features?
   Not established because actual blocks are dependency/DPT coupled here.
10. Does it add value beyond explicit memory?
    Unknown; do not include it in the first serious model.
11. Are Mamba state and Pick-and-Play complementary or redundant?
    Undecided; neither earns inclusion from this probe.
12. Which imported mechanisms damage clean predictions?
    Preliminary only; no claim that aligned history or K>1 damages predictions until corrected common-mask/native/oracle results are inspected.
13. Which parts should be directly reused?
    BiDA warp convention, official SEA-RAFT wrapper, and PPMStereo scoring math for ablations.
14. Which parts need adapters?
    PPMStereo needs universal ARGOS feature adapters; EndoStreamDepth needs DPT-token adapters and missing deps.
15. Which parts should be cleanly reimplemented?
    Causal forward-only propagation, support/consistency gating, and bounded identity-preserving residual output.
16. Which parts should remain reference-only?
    Full BiDAStabilizer, PPMStereo flash-attn readout, and EndoStreamDepth Mamba/xLSTM blocks for now.
17. Recommended first ARGOS v2 model?
    Preliminary: causal BiDA-style alignment signals + current raw disparity + support/FB-consistency + safe bounded residual gate; memory depth remains open until corrected CSVs are inspected.
"""


def run(args: argparse.Namespace) -> None:
    bida.self_check()
    ppm.self_check()
    cases = CASES[: args.max_cases]
    flow_load_rows: list[dict] = []
    flow_rows: list[dict] = []
    flows: dict = {}
    if args.with_flow:
        if args.flow_cache_only:
            models = {name: None for name in args.flow_models}
            flow_load_rows = [{
                "component": name,
                "status": "flow_cache_only",
                "runtime_ms": 0.0,
                "peak_memory_mb": 0.0,
                "notes": "not loading models; only existing flow_cache .npy files are accepted",
            } for name in args.flow_models]
        else:
            models, flow_load_rows = load_flow_models(args.flow_models)
        flows, flow_rows = compute_flows(models, cases, generate_missing=not args.flow_cache_only)
    else:
        flow_load_rows = [{"component": name, "status": "not_run", "runtime_ms": 0.0, "peak_memory_mb": 0.0, "notes": "--with-flow not set"} for name in args.flow_models]

    bida_rows, flow_summary = run_bida_probe(flows, flow_rows, cases)
    ppm_rows, candidate_signal_rows = run_ppm_probe(flows, cases)
    endo_rows = run_endo_probe()
    combo_rows = [{"combination": "A_BiDA_alignment_only", **r} for r in bida_rows]
    combo_rows.extend({"combination": "B_BiDA_alignment_plus_latest_memory", **r} for r in bida_rows)
    combo_rows.extend({"combination": "C_BiDA_alignment_plus_PPMStereo_score_topk_adapter", **r} for r in ppm_rows if r.get("method") == "ppmstereo_original_score_topk")
    combo_rows.extend({"combination": "D_BiDA_alignment_plus_PPMStereo_dynamic_memory_readout_diag", **r} for r in ppm_rows if r.get("method") == "ppmstereo_original_modulated_diag")
    endo_blockers = [r for r in endo_rows if "Mamba" in r["component"] or "xLSTM" in r["component"]]
    combo_rows.extend({"combination": "E_EndoStreamDepth_temporal_state_only", "status": r["status"], "notes": r["reason"]} for r in endo_blockers)
    combo_rows.extend({"combination": "F_BiDA_alignment_plus_EndoStreamDepth_state", "status": "blocked", "notes": "actual EndoStreamDepth state did not instantiate; see endostreamdepth_probe.csv"} for _ in endo_blockers[:1])
    combo_rows.extend({"combination": "G_BiDA_alignment_plus_PPMStereo_memory_plus_EndoStreamDepth_state", "status": "blocked", "notes": "actual EndoStreamDepth state did not instantiate; see endostreamdepth_probe.csv"} for _ in endo_blockers[:1])

    signal_rows = []
    if bida_rows:
        for coverage_threshold in COVERAGE_THRESHOLDS:
            subset = [r for r in bida_rows if float(r["coverage_threshold"]) == coverage_threshold]
            signal_rows.append({
                "coverage_threshold": coverage_threshold,
                "signal": "temporal_disagreement_px",
                "target": "epe_raw_common_cache_px",
                "pearson_r": corr([r["temporal_disagreement_px"] for r in subset if r["temporal_disagreement_px"] is not None and r["epe_raw_common_cache_px"] is not None],
                                  [r["epe_raw_common_cache_px"] for r in subset if r["temporal_disagreement_px"] is not None and r["epe_raw_common_cache_px"] is not None]),
                "n": sum(1 for r in subset if r["temporal_disagreement_px"] is not None and r["epe_raw_common_cache_px"] is not None),
            })
    if candidate_signal_rows:
        for coverage_threshold in COVERAGE_THRESHOLDS:
            good = [r for r in candidate_signal_rows if float(r["coverage_threshold"]) == coverage_threshold and r["candidate_epe_cache_px"] is not None]
            for signal in ["ppm_score", "ppm_similarity", "ppm_confidence_adapter", "support_ratio"]:
                signal_rows.append({
                    "coverage_threshold": coverage_threshold,
                    "signal": signal,
                    "target": "candidate_epe_cache_px",
                    "pearson_r": corr([r[signal] for r in good], [r["candidate_epe_cache_px"] for r in good]),
                    "n": len(good),
                })

    runtime_rows = flow_load_rows + flow_rows
    write_csv(OUT / "bidavideo_probe.csv", bida_rows)
    write_csv(OUT / "raft_vs_searaft.csv", flow_summary)
    write_csv(OUT / "ppmstereo_probe.csv", ppm_rows)
    write_csv(OUT / "ppmstereo_candidate_signal.csv", candidate_signal_rows)
    write_csv(OUT / "endostreamdepth_probe.csv", endo_rows)
    write_csv(OUT / "component_combination_probe.csv", combo_rows)
    write_csv(OUT / "signal_predictiveness.csv", signal_rows)
    write_csv(OUT / "runtime_memory_summary.csv", runtime_rows)
    save_contact_sheets(bida_rows, flows)
    write_docs(args.flow_models, flow_load_rows, bida_rows, ppm_rows, endo_rows, write_architecture=not args.skip_architecture_report)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--with-flow", action="store_true", help="Run official RAFT/SEA-RAFT inference paths")
    ap.add_argument("--flow-models", nargs="+", default=["SEA-RAFT", "RAFT"])
    ap.add_argument("--max-cases", type=int, default=3)
    ap.add_argument("--flow-cache-only", action="store_true", help="Use existing flow_cache .npy files; fail rows if missing")
    ap.add_argument("--skip-architecture-report", action="store_true", help="Do not rewrite architecture_decisions.md")
    args = ap.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
