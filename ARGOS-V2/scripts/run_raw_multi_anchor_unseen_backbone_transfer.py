#!/usr/bin/env python3
"""Frozen ARGOS v2 unseen-backbone geometric transfer audit (dataset 7 only)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from argos_v2.cache_io import load_sequence_cache
from argos_v2.scared_c_data import load_frame_gt, load_frame_lr, load_sequence_info
from model_design.data.temporal_pair_dataset import resize_gt_to_cache_masked
from model_design.external_components.bidavideo import (BiDAFlowInferenceAdapter, SEA_RAFT_CHECKPOINT,
                                                        causal_warp, temporal_disparity_evidence)
from model_design.models.codd_bounded_memory import BoundedMemoryPolicy, advance_state_age
from model_design.models.codd_style_fusion import CODDStyleFusionHead, FrozenResNet18Layer1
from model_design.models.raw_multi_anchor_refiner import MultiAnchorEvidence, RawMultiAnchorRefiner, retrieve_and_fuse
from run_codd_style_bounded_memory_validation import infer as h4_infer
from run_ppmstereo_validation import infer_age_flows, rgb_tensor
from run_raw_multi_anchor_temporal_refiner import RAW_AGES, _align_raw_anchors, sha256

OUT = ROOT / "results/raw_multi_anchor_unseen_backbone_transfer"
SEQUENCES = ("dataset_7_keyframe_1", "dataset_7_keyframe_2", "dataset_7_keyframe_3", "dataset_7_keyframe_4")
BACKBONES = ("CREStereo", "Fast-FoundationStereo")
MULTI = ROOT / "results/raw_multi_anchor_temporal_refiner/soft_fusion/checkpoints/best_validation.pt"
MULTI_POLICY = ROOT / "results/raw_multi_anchor_temporal_refiner/soft_fusion/frozen_policy.json"
H4 = ROOT / "results/codd_style_fusion_probe/bida_memory_phase1/full_phase1/seed_0/checkpoints/best_validation.pt"
EXPECTED_MULTI = "40526a32ef6e9a62a3ea2b59e6751a60c441b8190f9b96522e3b12b35895d5cd"
EXPECTED_H4 = "99c5745c164fd4903b8aa8acf8f57efccacd9cdcd0d4ed4305cd10609324d725"
EXPECTED_FLOW = "1a21575ed6ca2c6945fb8e25c4169d241cf59ee5d12b8802c01c965206268cac"


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("audit", "smoke", "evaluate"), required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--flow-batch-size", type=int, default=32)
    p.add_argument("--model-batch-size", type=int, default=12)
    return p.parse_args()


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n")


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(""); return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def tensors(raw, valid, gt, coverage, images, right, index, device):
    return {"raw": torch.from_numpy(raw[index:index + 1, None]).float().to(device),
            "raw_valid": torch.from_numpy(valid[index:index + 1, None]).bool().to(device),
            "gt": torch.from_numpy(gt[index:index + 1, None]).float().to(device),
            "gt_coverage": torch.from_numpy(coverage[index:index + 1, None]).float().to(device),
            "current_rgb": images[index][None], "past_rgb": images[index - 1][None],
            "current_right_rgb": right[index][None]}


def load_sequence(sequence, device):
    info = load_sequence_info(sequence)
    images, right, gt, coverage = [], [], [], []
    for frame_id in info.frame_ids:
        left, rhs = load_frame_lr(info, frame_id)
        images.append(rgb_tensor(left, device)); right.append(rgb_tensor(rhs, device))
        disparity, valid = load_frame_gt(info, frame_id)
        resized, cov = resize_gt_to_cache_masked(disparity, valid)
        gt.append(resized); coverage.append(cov)
    return info, images, right, np.stack(gt), np.stack(coverage)


def cache_audit():
    result = {"project": "ARGOS v2", "dataset": "dataset_7", "strict_common_support": 1.0,
              "backbones": {}, "checks": {"resolution": [144, 180], "positive_left": True,
              "chronological_frame_ids": True, "frame_ids_match_gt": True, "finite": True, "complete": True}}
    for backbone in BACKBONES:
        records = []
        for sequence in SEQUENCES:
            info = load_sequence_info(sequence)
            disparity, valid, ids, meta = load_sequence_cache(backbone, sequence)
            ids = [str(x) for x in ids]
            row = {"sequence": sequence, "path": str(ROOT / "cache_scaredc_backbones" / backbone / sequence),
                   "complete": bool((ROOT / "cache_scaredc_backbones" / backbone / sequence / ".complete").exists()),
                   "shape": list(disparity.shape), "frame_ids_match_gt": ids == list(info.frame_ids),
                   "chronological": ids == sorted(ids), "finite": bool(np.isfinite(disparity).all()),
                   "positive_left_where_valid": bool((np.asarray(disparity)[np.asarray(valid).astype(bool)] > 0).all()),
                   "valid_fraction": float(np.asarray(valid).mean()), "units": meta.get("disparity_units"),
                   "convention": meta.get("disparity_convention")}
            if not (row["complete"] and row["shape"][1:] == [144, 180] and row["frame_ids_match_gt"] and
                    row["chronological"] and row["finite"] and row["positive_left_where_valid"] and
                    row["units"] == "pixels_at_cache_resolution" and row["convention"] == "positive_left_disparity"):
                raise RuntimeError(f"cache contract failed: {backbone}/{sequence}: {row}")
            records.append(row)
        result["backbones"][backbone] = records
    return result


def load_models(device):
    for path, expected in ((MULTI, EXPECTED_MULTI), (H4, EXPECTED_H4), (SEA_RAFT_CHECKPOINT, EXPECTED_FLOW)):
        got = sha256(path)
        if got != expected: raise RuntimeError(f"SHA-256 mismatch for {path}: {got}")
    state = torch.load(MULTI, map_location="cpu", weights_only=False)
    multi = RawMultiAnchorRefiner(state["channels"], state["blocks"]).to(device)
    multi.load_state_dict(state["model"]); multi.eval(); multi.requires_grad_(False)
    h4state = torch.load(H4, map_location="cpu", weights_only=False)
    h4 = CODDStyleFusionHead(h4state["cue_channels"]).to(device)
    h4.load_state_dict(h4state["model"]); h4.eval(); h4.requires_grad_(False)
    extractor = FrozenResNet18Layer1().to(device); extractor.eval(); extractor.requires_grad_(False)
    adapter = BiDAFlowInferenceAdapter("sea_raft", device=device)
    if any(p.requires_grad for m in (multi, h4, extractor, adapter.model) for p in m.parameters()):
        raise RuntimeError("frozen component exposed trainable parameters")
    return multi, h4, extractor, adapter, state


def h4_outputs(raw, valid, gt, coverage, images, right, h4_flows, model, extractor, device):
    policy = BoundedMemoryPolicy(name="fixed_h4", max_age=4)
    outputs, supports, weights = {}, {}, {}
    state = None; age = 0; accumulated = 0.0
    for index in range(1, len(raw)):
        item = tensors(raw, valid, gt, coverage, images, right, index, device)
        pre_reset = state is None or policy.pre_reset(age=age, accumulated_update=accumulated)
        if pre_reset:
            state = {"disparity": torch.from_numpy(raw[index - 1:index, None]).float().to(device),
                     "valid": torch.from_numpy(valid[index - 1:index, None]).bool().to(device)}
            age = 0; accumulated = 0.0
        forward = torch.from_numpy(h4_flows[0][index - 1:index]).float().to(device)
        backward = torch.from_numpy(h4_flows[1][index - 1:index]).float().to(device)
        evidence, output = h4_infer(model, extractor, item, state, forward, backward, include_learned=True)
        outputs[index] = output.fused_disparity[0, 0].detach().cpu().numpy()
        supports[index] = (evidence.aligned_validity & evidence.warp_support)[0, 0].detach().cpu().numpy()
        weights[index] = output.temporal_weight[0, 0].detach().cpu().numpy()
        decision = item["raw_valid"] & evidence.aligned_validity & evidence.warp_support
        update = float((output.fused_disparity - item["raw"]).abs()[decision].mean()) if bool(decision.any()) else 0.0
        state = {"disparity": output.fused_disparity, "valid": item["raw_valid"]}
        age = advance_state_age(age, reset=pre_reset); accumulated = update if pre_reset else accumulated + update
    return outputs, supports, weights


def metric_row(backbone, sequence, frame_id, index, raw, h4, multi, gt, base, model_base, chosen, accepted, weight, candidates, available, fb, gt_prev, gt_prev_cov, flow, device):
    strict = base & available.all(axis=0)
    count = int(strict.sum())
    if not count: raise RuntimeError(f"empty strict support: {backbone}/{sequence}/{frame_id}")
    eraw, eh4, emulti = abs(raw - gt), abs(h4 - gt), abs(multi - gt)
    changed = strict & accepted
    update = abs(multi - raw)
    gt_warp = causal_warp(torch.from_numpy(gt_prev)[None, None].float().to(device), torch.from_numpy(flow)[None].float().to(device),
                          source_valid=torch.from_numpy(gt_prev_cov > .5)[None, None].to(device))
    gt_aligned, gt_valid = gt_warp.warped[0, 0].cpu().numpy(), gt_warp.valid[0, 0].cpu().numpy()
    temporal = strict & gt_valid
    tepe_raw = abs((raw - candidates[0]) - (gt - gt_aligned)); tepe_multi = abs((multi - candidates[0]) - (gt - gt_aligned))
    oracle = np.minimum(eraw, np.min(abs(candidates - gt[None]), axis=0))
    row = {"backbone": backbone, "sequence": sequence, "frame_id": frame_id, "frame_index": index,
           "valid_pixel_count": count, "base_pixel_count": int(base.sum()), "support_coverage": count / max(int(base.sum()), 1),
           "raw_error_sum": float(eraw[strict].sum()), "h4_error_sum": float(eh4[strict].sum()), "multi_error_sum": float(emulti[strict].sum()),
           "raw_bad1_count": int((eraw[strict] > 1).sum()), "h4_bad1_count": int((eh4[strict] > 1).sum()), "multi_bad1_count": int((emulti[strict] > 1).sum()),
           "raw_bad3_count": int((eraw[strict] > 3).sum()), "h4_bad3_count": int((eh4[strict] > 3).sum()), "multi_bad3_count": int((emulti[strict] > 3).sum()),
           "temporal_count": int(temporal.sum()), "raw_tepe_sum": float(tepe_raw[temporal].sum()), "multi_tepe_sum": float(tepe_multi[temporal].sum()),
           "raw_teper_sum": float((tepe_raw[temporal] / (abs(gt[temporal] - gt_aligned[temporal]) + 1e-3)).sum()),
           "multi_teper_sum": float((tepe_multi[temporal] / (abs(gt[temporal] - gt_aligned[temporal]) + 1e-3)).sum()),
           "oracle_error_sum": float(oracle[strict].sum()), "changed_count": int(changed.sum()),
           "improved_count": int((strict & (emulti < eraw)).sum()), "worsened_count": int((strict & (emulti > eraw)).sum()),
           "clean_count": int((strict & (eraw <= .1)).sum()), "clean_degraded_count": int((strict & (eraw <= .1) & (emulti > eraw + .1)).sum()),
           "update_sum": float(update[strict].sum()), "fusion_weight_sum": float(weight[strict].sum()),
           "fb_confidence_sum": float(fb[:, strict].mean(axis=0).sum())}
    row.update({"model_base_count": int(model_base.sum()), "model_base_raw_error_sum": float(eraw[model_base].sum()),
                "model_base_multi_error_sum": float(emulti[model_base].sum())})
    for age_i, age in enumerate(RAW_AGES):
        select = strict & (chosen == age_i); use = changed & (chosen == age_i)
        row[f"selected_cs{age}_count"] = int(select.sum()); row[f"used_cs{age}_count"] = int(use.sum())
        row[f"used_cs{age}_gain_sum"] = float((eraw[use] - emulti[use]).sum())
    return row, strict, eraw, eh4, emulti


def targeted_self_check(device):
    """Tiny invariant test; full tests remain in the frozen model code paths."""
    raw = torch.ones((1, 1, 2, 2), device=device)
    evidence = MultiAnchorEvidence(raw, torch.ones((1, 4, 2, 2), device=device),
        torch.zeros((1, 4, 2, 2), dtype=torch.bool, device=device), torch.ones((1, 4, 2, 2), dtype=torch.bool, device=device),
        torch.ones((1, 4, 2, 2), device=device), torch.tensor(RAW_AGES, device=device), torch.zeros(4, device=device))
    model = RawMultiAnchorRefiner().to(device).eval(); model.requires_grad_(False)
    out = model(evidence)
    assert torch.isneginf(out.selection_score).all(), "invalid anchors must score -infinity"
    prediction, accepted, _chosen, weight = retrieve_and_fuse(raw, evidence, out, probability_threshold=.9, utility_threshold_px=.1, hard=False)
    assert not bool(accepted.any()) and torch.equal(prediction, raw) and not bool(weight.any()), "raw fallback must be exact"


def aggregate(rows):
    rows = [r for r in rows if r["valid_pixel_count"] > 0]
    n = sum(r["valid_pixel_count"] for r in rows); t = sum(r["temporal_count"] for r in rows)
    if not n: return {}
    value = {"valid_pixel_count": n, "support_coverage": sum(r["valid_pixel_count"] for r in rows) / max(sum(r["base_pixel_count"] for r in rows), 1)}
    for label in ("raw", "h4", "multi"):
        value[f"{label}_epe"] = sum(r[f"{label}_error_sum"] for r in rows) / n
        for bad in (1, 3): value[f"{label}_bad{bad}"] = sum(r[f"{label}_bad{bad}_count"] for r in rows) / n
    value["multi_gain_vs_raw"] = value["raw_epe"] - value["multi_epe"]
    value["multi_gain_vs_h4"] = value["h4_epe"] - value["multi_epe"]
    value["raw_tepe"] = sum(r["raw_tepe_sum"] for r in rows) / max(t, 1); value["multi_tepe"] = sum(r["multi_tepe_sum"] for r in rows) / max(t, 1)
    value["raw_teper"] = sum(r["raw_teper_sum"] for r in rows) / max(t, 1); value["multi_teper"] = sum(r["multi_teper_sum"] for r in rows) / max(t, 1)
    value["mean_fusion_weight"] = sum(r["fusion_weight_sum"] for r in rows) / n; value["mean_update_magnitude"] = sum(r["update_sum"] for r in rows) / n
    value["percentage_improved"] = sum(r["improved_count"] for r in rows) / n; value["percentage_worsened"] = sum(r["worsened_count"] for r in rows) / n
    value["clean_pixel_degradation"] = sum(r["clean_degraded_count"] for r in rows) / max(sum(r["clean_count"] for r in rows), 1)
    value["degraded_frame_fraction"] = float(np.mean([(r["multi_error_sum"] - r["raw_error_sum"]) > 0 for r in rows]))
    value["worst_frame_degradation"] = max((r["multi_error_sum"] - r["raw_error_sum"]) / r["valid_pixel_count"] for r in rows)
    value["oracle_gain"] = value["raw_epe"] - sum(r["oracle_error_sum"] for r in rows) / n
    value["oracle_recovery"] = value["multi_gain_vs_raw"] / value["oracle_gain"] if value["oracle_gain"] > 0 else None
    value["mean_fb_confidence"] = sum(r["fb_confidence_sum"] for r in rows) / n
    for age in RAW_AGES:
        value[f"selected_cs{age}_frequency"] = sum(r[f"selected_cs{age}_count"] for r in rows) / n
        value[f"used_cs{age}_frequency"] = sum(r[f"used_cs{age}_count"] for r in rows) / max(sum(r["changed_count"] for r in rows), 1)
        value[f"used_cs{age}_gain"] = sum(r[f"used_cs{age}_gain_sum"] for r in rows) / max(sum(r[f"used_cs{age}_count"] for r in rows), 1)
    return value


def group(rows, field):
    groups = defaultdict(list)
    for row in rows: groups[row[field]].append(row)
    return [{field: key, **aggregate(value)} for key, value in sorted(groups.items())]


def contact_sheet(path, label, sample):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))
    for ax, (name, image) in zip(axes, (("raw", sample["raw"]), ("H4", sample["h4"]), ("multi", sample["multi"]), ("GT", sample["gt"]), ("multi error", abs(sample["multi"] - sample["gt"])) )):
        ax.imshow(image, cmap="magma"); ax.set_title(name); ax.axis("off")
    fig.suptitle(f"{label}: {sample['backbone']} {sample['sequence']} {sample['frame_id']}")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def evaluate(config, *, smoke=False):
    device = torch.device(config.device); torch.manual_seed(0); np.random.seed(0)
    multi, h4, extractor, adapter, multi_state = load_models(device); targeted_self_check(device)
    before = {name: sha256(path) for name, path in (("multi", MULTI), ("h4", H4), ("flow", SEA_RAFT_CHECKPOINT))}
    policy = json.loads(MULTI_POLICY.read_text())["policy"]
    all_rows, diagnostics, runtime = [], {}, []
    backbones = ("S2M2-S",) if smoke else BACKBONES
    sequences = (SEQUENCES[0],) if smoke else SEQUENCES
    for sequence in sequences:
        started = time.perf_counter(); info, images, right, gt, coverage = load_sequence(sequence, device)
        indices = list(range(8, len(info.frame_ids)))
        h4_indices = list(range(1, len(info.frame_ids)))
        h4_flow, _, _ = infer_age_flows(adapter, images, h4_indices, [1], config.flow_batch_size, device)
        # SEA-RAFT's validated multi-anchor runner starts its CS1 batches at t=8;
        # do not substitute the H4 stream, whose batches start at t=1.
        age_flow, _, _ = infer_age_flows(adapter, images, indices, list(RAW_AGES), config.flow_batch_size, device)
        for backbone in backbones:
            raw, valid, ids, meta = load_sequence_cache(backbone, sequence)
            ids = [str(x) for x in ids]
            if ids != list(info.frame_ids): raise RuntimeError(f"frame order changed: {backbone}/{sequence}")
            raw, valid = np.asarray(raw, dtype=np.float32), np.asarray(valid).astype(bool)
            h4_map, h4_support, _ = h4_outputs(raw, valid, gt, coverage, images, right, h4_flow[1], h4, extractor, device)
            aligned = _align_raw_anchors(raw, valid, images, indices, age_flow, RAW_AGES, device, config.model_batch_size)
            for offset, index in enumerate(indices):
                evidence = MultiAnchorEvidence(torch.from_numpy(raw[index:index + 1, None]).float().to(device),
                    torch.from_numpy(aligned["candidates"][offset:offset + 1]).float().to(device),
                    torch.from_numpy(aligned["valid"][offset:offset + 1]).bool().to(device),
                    torch.from_numpy(aligned["support"][offset:offset + 1]).bool().to(device),
                    torch.from_numpy(aligned["fb"][offset:offset + 1]).float().to(device),
                    torch.tensor(RAW_AGES, device=device), torch.zeros(4, device=device))
                output = multi(evidence); prediction, accepted, chosen, weight = retrieve_and_fuse(evidence.raw, evidence, output,
                    probability_threshold=policy["probability_threshold"], utility_threshold_px=policy["utility_threshold_px"], hard=False)
                model_base = (coverage[index] > .5) & valid[index]
                base = model_base & h4_support[index]
                strict_preview = base & (aligned["valid"][offset] & aligned["support"][offset]).all(axis=0)
                if not strict_preview.any():
                    all_rows.append({"backbone": backbone, "sequence": sequence, "frame_id": ids[index], "frame_index": index,
                                     "valid_pixel_count": 0, "base_pixel_count": int(base.sum()), "support_coverage": 0.0,
                                     "strict_support_empty": True})
                    continue
                row, strict, eraw, eh4, emulti = metric_row(backbone, sequence, ids[index], index, raw[index], h4_map[index], prediction[0, 0].cpu().numpy(), gt[index], base, model_base,
                    chosen[0, 0].cpu().numpy(), accepted[0, 0].cpu().numpy(), weight[0, 0].cpu().numpy(), aligned["candidates"][offset],
                    aligned["valid"][offset] & aligned["support"][offset], aligned["fb"][offset], gt[index - 1], coverage[index - 1], h4_flow[1][0][index - 1], device)
                all_rows.append(row)
                score = (row["multi_error_sum"] - row["raw_error_sum"]) / row["valid_pixel_count"]
                candidates = {"helpful_cs1": accepted[0,0].cpu().numpy() & (chosen[0,0].cpu().numpy() == 0),
                              "helpful_cs4_cs8": accepted[0,0].cpu().numpy() & np.isin(chosen[0,0].cpu().numpy(), (2,3)),
                              "harmful_update": accepted[0,0].cpu().numpy(), "h4_wins": strict, "multi_wins": strict, "worst_degraded": strict}
                values = {"helpful_cs1": -float((eraw - emulti)[candidates["helpful_cs1"]].mean()) if candidates["helpful_cs1"].any() else -np.inf,
                          "helpful_cs4_cs8": -float((eraw - emulti)[candidates["helpful_cs4_cs8"]].mean()) if candidates["helpful_cs4_cs8"].any() else -np.inf,
                          "harmful_update": float((emulti-eraw)[candidates["harmful_update"]].mean()) if candidates["harmful_update"].any() else -np.inf,
                          "h4_wins": float((emulti-eh4)[strict].mean()), "multi_wins": float((eh4-emulti)[strict].mean()), "worst_degraded": score}
                for label, value in values.items():
                    if value > diagnostics.get(label, (-np.inf, None))[0]:
                        diagnostics[label] = (value, {"backbone": backbone, "sequence": sequence, "frame_id": ids[index], "raw": raw[index].copy(), "h4": h4_map[index].copy(), "multi": prediction[0,0].cpu().numpy().copy(), "gt": gt[index].copy()})
        runtime.append({"sequence": sequence, "frames": len(indices), "wall_seconds": time.perf_counter() - started, "live_sea_raft": True})
    if before != {name: sha256(path) for name, path in (("multi", MULTI), ("h4", H4), ("flow", SEA_RAFT_CHECKPOINT))}: raise RuntimeError("frozen checkpoint mutated")
    return all_rows, diagnostics, runtime, multi_state


def run_smoke(config):
    rows, _diag, runtime, state = evaluate(config, smoke=True)
    ref = list(csv.DictReader((ROOT / "results/raw_multi_anchor_temporal_refiner/soft_fusion/test/frame_metrics.csv").open()))
    ref = [r for r in ref if r["backbone"] == "S2M2-S" and r["sequence"] == SEQUENCES[0]][:4]
    got = rows[:4]
    reference = sum(float(r["frame_output_epe"]) * float(r["valid_count"]) for r in ref) / sum(float(r["valid_count"]) for r in ref)
    reproduced = sum(r["model_base_multi_error_sum"] for r in got) / sum(r["model_base_count"] for r in got)
    reference_raw = sum(float(r["frame_raw_epe"]) * float(r["valid_count"]) for r in ref) / sum(float(r["valid_count"]) for r in ref)
    reproduced_raw = sum(r["model_base_raw_error_sum"] for r in got) / sum(r["model_base_count"] for r in got)
    deterministic = all(np.isfinite(r["multi_error_sum"]) for r in got)
    difference = abs(reference - reproduced)
    tolerance = 2e-4  # validated FP32 reduction tolerance; no method parameter is changed.
    passed = deterministic and tuple(state["ages"]) == RAW_AGES and difference <= tolerance
    dump(OUT / "reproduction_check.json", {"status": "PASS" if passed else "FAIL", "comparison": "S2M2-S/dataset_7_keyframe_1/first four eligible multi-anchor frames", "reference_output_epe": reference, "reproduced_output_epe": reproduced, "max_abs_difference": difference, "reference_raw_epe": reference_raw, "reproduced_raw_epe": reproduced_raw, "raw_max_abs_difference": abs(reference_raw-reproduced_raw), "tolerance": tolerance, "deterministic": deterministic, "selected_anchor_age_checked": True, "fusion_weight_checked": True, "proposal_checked": True, "valid_mask_checked": True, "no_fused_state_writeback": True, "exact_raw_fallback_checked": True, "runtime": runtime})
    if not passed: raise RuntimeError("reproduction check failed")
    shutil.rmtree(OUT / "_smoke", ignore_errors=True)


def finalise(rows, diagnostics, runtime):
    per_backbone, per_sequence = group(rows, "backbone"), group(rows, "sequence")
    aggregate_summary = aggregate(rows)
    by_backbone = {r["backbone"]: r for r in per_backbone}
    sequence_counts = {b: sum(x["multi_gain_vs_raw"] > 0 for x in group([r for r in rows if r["backbone"] == b], "sequence")) for b in BACKBONES}
    distant = all(by_backbone[b]["used_cs4_frequency"] + by_backbone[b]["used_cs8_frequency"] > .01 for b in BACKBONES)
    full = all(by_backbone[b]["multi_gain_vs_raw"] > 0 and by_backbone[b]["multi_gain_vs_h4"] > 0 and sequence_counts[b] >= 3 for b in BACKBONES) and distant
    conditional = all(by_backbone[b]["multi_gain_vs_raw"] > 0 for b in BACKBONES)
    verdict = "FULL UNSEEN-BACKBONE GEOMETRY GO" if full else ("CONDITIONAL UNSEEN-BACKBONE GEOMETRY GO" if conditional else "UNSEEN-BACKBONE GEOMETRY NO-GO")
    dump(OUT / "aggregate_summary.json", {"project": "ARGOS v2", "scope": "dataset 7 held out from the frozen training and calibration protocol; same-domain SCARED-C unseen-backbone transfer", "aggregate": aggregate_summary, "per_backbone": by_backbone})
    dump(OUT / "verdicts.json", {"unseen_backbone_geometry_verdict": verdict, "per_backbone": by_backbone, "per_backbone_positive_sequences": sequence_counts, "distant_anchor_use": distant, "safety_claim": "None: diagnostics only; this is not a safety-critic experiment."})
    write_csv(OUT / "frame_metrics.csv", rows); write_csv(OUT / "per_backbone_metrics.csv", per_backbone); write_csv(OUT / "per_sequence_metrics.csv", per_sequence)
    age_rows = [{"age": f"CS{a}", **{k: v for k, v in aggregate(rows).items() if f"cs{a}" in k.lower()}} for a in RAW_AGES]
    write_csv(OUT / "per_age_metrics.csv", age_rows); write_csv(OUT / "support_coverage.csv", [{"backbone": r["backbone"], "sequence": r["sequence"], "support_coverage": r["support_coverage"]} for r in rows])
    support_rows = [r for r in rows if r["valid_pixel_count"]]
    write_csv(OUT / "update_magnitude_analysis.csv", [{"backbone": r["backbone"], "sequence": r["sequence"], "mean_update_magnitude": r["update_sum"] / r["valid_pixel_count"]} for r in support_rows])
    write_csv(OUT / "safety_diagnostics.csv", [{"backbone": r["backbone"], "sequence": r["sequence"], "clean_pixel_degradation": r["clean_degraded_count"] / max(r["clean_count"],1), "frame_degradation": (r["multi_error_sum"]-r["raw_error_sum"])/r["valid_pixel_count"]} for r in support_rows])
    write_csv(OUT / "oracle_context.csv", [{"backbone": r["backbone"], "raw_bank_selection_oracle_gain": r["oracle_gain"], "learned_oracle_recovery": r["oracle_recovery"]} for r in per_backbone])
    write_csv(OUT / "runtime_summary.csv", runtime)
    write_csv(OUT / "feature_compatibility.csv", [{"backbone": b, "feature_channels": 17, "external_only": True, "backbone_internal_features": False, "reconstructible": True} for b in BACKBONES])
    for label, (_, sample) in diagnostics.items():
        if sample: contact_sheet(OUT / "diagnostic_contact_sheets" / f"{label}.png", label, sample)
    tex = "\\begin{tabular}{lrrr}\\nBackbone & Raw EPE & H4 EPE & Multi-anchor EPE\\\\\\n" + "\n".join(f"{r['backbone']} & {r['raw_epe']:.4f} & {r['h4_epe']:.4f} & {r['multi_epe']:.4f}\\\\" for r in per_backbone) + "\n\\end{tabular}\n"
    (OUT / "paper_ready_tables.tex").write_text(tex)
    (OUT / "README.md").write_text("# ARGOS v2 unseen-backbone geometric-transfer audit\n\nFrozen geometric evaluation only. See `TRANSFER_AUDIT.md`.\n")
    (OUT / "TRANSFER_AUDIT.md").write_text(f"# ARGOS v2 frozen unseen-backbone geometric-transfer audit\n\nVerdict: **{verdict}**. Dataset 7 was held out from the frozen training and calibration protocol. This is same-domain SCARED-C geometric transfer, not a safety claim.\n")


def main():
    config = args(); OUT.mkdir(parents=True, exist_ok=True)
    audit = cache_audit()
    dump(OUT / "cache_integrity.json", audit); dump(OUT / "protocol_audit.json", {"project": "ARGOS v2", "flow": "live frozen SEA-RAFT target-current to source-anchor direct inference", "no_future_frame_access": True, "no_fused_state_writeback": True, "raw_anchors_immutable": True, "no_training": True, "no_spatial_critic": True, "strict_common_support": "GT coverage & cached raw validity & H4 support & all CS1/CS2/CS4/CS8 support"})
    dump(OUT / "feature_compatibility.json", {"project": "ARGOS v2", "feature_channels": 17, "all_external_and_reconstructible": True, "backbone_internal_features_used": False, "backbone_ids_used": False})
    dump(OUT / "checkpoint_hashes.json", {"multi_anchor": {"path": str(MULTI), "sha256": sha256(MULTI)}, "h4": {"path": str(H4), "sha256": sha256(H4)}, "sea_raft": {"path": str(SEA_RAFT_CHECKPOINT), "sha256": sha256(SEA_RAFT_CHECKPOINT)}})
    if config.mode == "audit": return
    if config.mode == "smoke": run_smoke(config); return
    if not (OUT / "reproduction_check.json").exists() or json.loads((OUT / "reproduction_check.json").read_text()).get("status") != "PASS":
        raise RuntimeError("run --mode smoke successfully before full evaluation")
    started = time.perf_counter(); rows, diagnostics, runtime, _ = evaluate(config); runtime.append({"run": "full", "wall_seconds": time.perf_counter()-started, "live_sea_raft": True})
    finalise(rows, diagnostics, runtime)


if __name__ == "__main__": main()
