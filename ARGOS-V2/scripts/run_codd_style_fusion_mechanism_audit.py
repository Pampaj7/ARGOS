#!/usr/bin/env python3
"""ARGOS v2 post-hoc mechanism audit for frozen CODD-style Phase-1 checkpoints.

This script deliberately distinguishes historical raw-t-1 selection, actual
recurrent-state selection, and the GT-only convex-fusion ceiling.  It creates
no disparity/flow caches and never opens a dataset outside the strict SCARED-C
split supplied on the command line.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_codd_style_fusion_probe import (  # noqa: E402
    CACHE_WIDTH, NATIVE_WIDTH, TEST, VALIDATION, codd_config, frame, make_dataset,
    manifest, run_clip, seed_all, to_device, valid_mask,
)
from model_design.data.temporal_pair_dataset import SEEN_BACKBONES, TemporalPairDataset  # noqa: E402
from model_design.external_components.bidavideo import (  # noqa: E402
    BiDAFlowInferenceAdapter, causal_warp, temporal_disparity_evidence,
)
from model_design.models.codd_style_fusion import (  # noqa: E402
    CODDStyleFusionHead, FrozenResNet18Layer1, build_codd_cues,
    convex_fusion_oracle, hard_endpoint_fusion,
)


PHASE1_ROOT = ROOT / "results/codd_style_fusion_probe/bida_memory_phase1/full_phase1"
OUT_ROOT = ROOT / "results/codd_style_fusion_mechanism_audit"
MAGNITUDE_THRESHOLDS = (0.01, 0.05, 0.10, 0.50)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "combine", "hard_validation", "hard_test"), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--preload-workers", type=int, default=20)
    parser.add_argument("--coverage-threshold", type=float, default=.50)
    parser.add_argument("--clip-length", type=int, default=4)
    parser.add_argument("--source-root", type=Path, default=PHASE1_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--hard-threshold", type=float)
    parser.add_argument("--evaluation-mode", choices=("both", "clip_reset", "reset_every4_all_pairs", "continuous_streaming"), default="both")
    return parser.parse_args()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(value: torch.Tensor, mask: torch.Tensor) -> float:
    return float(value[mask].mean()) if bool(mask.any()) else float("nan")


def metadata(item: dict) -> tuple[str, str, str]:
    def text(key: str) -> str:
        value = item[key]
        return str(value[0] if isinstance(value, list) else value)
    return text("sequence"), text("backbone"), text("current_frame_id")


def common_support(item: dict, historical, recurrent) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    historical_mask = valid_mask(item, historical)
    recurrent_mask = valid_mask(item, recurrent)
    return historical_mask & recurrent_mask, historical_mask, recurrent_mask


def frame_metrics(
    *, item: dict, historical_memory: torch.Tensor, recurrent_memory: torch.Tensor,
    output, common: torch.Tensor, historical_mask: torch.Tensor, recurrent_mask: torch.Tensor,
    mode: str, step_since_reset: int, gt_aligned: torch.Tensor, gt_aligned_valid: torch.Tensor,
) -> tuple[dict, list[dict]]:
    """One common-mask candidate/oracle decomposition and magnitude sweep."""
    sequence, backbone, frame_id = metadata(item)
    raw, gt = item["raw"], item["gt"]
    fused = output.fused_disparity
    raw_error = (raw - gt).abs()
    historical_error = (historical_memory - gt).abs()
    recurrent_error = (recurrent_memory - gt).abs()
    fused_error = (fused - gt).abs()
    historical_oracle = torch.minimum(raw_error, historical_error)
    recurrent_endpoint = torch.minimum(raw_error, recurrent_error)
    convex, w_star = convex_fusion_oracle(raw, recurrent_memory, gt)
    convex_error = (convex - gt).abs()
    temporal = output.temporal_weight
    delta = (fused - raw).abs()
    endpoint_gain = raw_error - recurrent_endpoint
    beyond_endpoint = recurrent_endpoint - fused_error
    interpolation_advantage = beyond_endpoint.clamp_min(0)
    clean = common & (raw_error <= .10)
    harmed = common & (fused_error > raw_error + .10)
    between = common & (gt >= torch.minimum(raw, recurrent_memory)) & (gt <= torch.maximum(raw, recurrent_memory))
    better_both = common & (fused_error < recurrent_endpoint)
    worse_both = common & (fused_error > torch.maximum(raw_error, recurrent_error))
    temporal_mask = common & gt_aligned_valid
    gt_delta = gt - gt_aligned
    raw_tepe = ((raw - historical_memory) - gt_delta).abs()
    fused_tepe = ((fused - recurrent_memory) - gt_delta).abs()

    row = {
        "mode": mode, "sequence": sequence, "backbone": backbone, "frame_id": frame_id,
        "step_since_reset": step_since_reset, "valid_count": int(common.sum()),
        "historical_only_count": int((historical_mask & ~recurrent_mask).sum()),
        "recurrent_only_count": int((recurrent_mask & ~historical_mask).sum()),
        "raw_epe": scalar(raw_error, common),
        "historical_memory_epe": scalar(historical_error, common),
        "recurrent_memory_epe": scalar(recurrent_error, common),
        "fused_epe": scalar(fused_error, common),
        "historical_selection_oracle_epe": scalar(historical_oracle, common),
        "recurrent_selection_oracle_epe": scalar(recurrent_endpoint, common),
        "convex_fusion_oracle_epe": scalar(convex_error, common),
        "historical_selection_gain": scalar(raw_error - historical_oracle, common),
        "recurrent_selection_gain": scalar(raw_error - recurrent_endpoint, common),
        "convex_fusion_gain": scalar(raw_error - convex_error, common),
        "fused_gain": scalar(raw_error - fused_error, common),
        "endpoint_selection_gain": scalar(endpoint_gain, common),
        "beyond_endpoint_gain_signed": scalar(beyond_endpoint, common),
        "interpolation_advantage": scalar(interpolation_advantage, common),
        "recurrent_candidate_improvement": scalar(historical_error - recurrent_error, common),
        "gt_between_endpoint_fraction": float(between.sum() / common.sum()),
        "fused_better_than_both_fraction": float(better_both.sum() / common.sum()),
        "fused_worse_than_both_fraction": float(worse_both.sum() / common.sum()),
        "harmful_update_rate": float(harmed.sum() / common.sum()),
        "clean_pixel_degradation": float((harmed & clean).sum() / clean.sum().clamp_min(1)),
        "reset_mean": scalar(output.reset_weight, common),
        "fusion_mean": scalar(output.fusion_weight, common),
        "temporal_weight_mean": scalar(temporal, common),
        "temporal_weight_p95": float(torch.quantile(temporal[common], .95)),
        "recurrent_raw_difference_mean": scalar((recurrent_memory - raw).abs(), common),
        "delta_fusion_mean": scalar(delta, common),
        "delta_fusion_p95": float(torch.quantile(delta[common], .95)),
        "w_star_mean": scalar(w_star, common),
        "temporal_valid_count": int(temporal_mask.sum()),
        "raw_tepe": scalar(raw_tepe, temporal_mask),
        "fused_tepe": scalar(fused_tepe, temporal_mask),
        "raw_teper": scalar(raw_tepe / (gt_delta.abs() + 1e-3), temporal_mask),
        "fused_teper": scalar(fused_tepe / (gt_delta.abs() + 1e-3), temporal_mask),
    }
    # Diagnostic thresholded policies have exact raw fallback outside their
    # accepted mask; the underlying soft fused output remains unchanged.
    magnitude_rows=[]
    for threshold in MAGNITUDE_THRESHOLDS:
        selected = common & (delta > threshold)
        thresholded = torch.where(selected, fused, raw)
        thresholded_error = (thresholded - gt).abs()
        harmed_selected = selected & (thresholded_error > raw_error + .10)
        clean_harm = harmed_selected & clean
        magnitude_rows.append({
            "mode": mode, "sequence": sequence, "backbone": backbone, "frame_id": frame_id,
            "threshold_cache_px": threshold, "threshold_native_px": threshold * NATIVE_WIDTH / CACHE_WIDTH,
            "valid_count": int(common.sum()), "coverage": float(selected.sum() / common.sum()),
            "epe": scalar(thresholded_error, common), "gain": scalar(raw_error - thresholded_error, common),
            "harmful_update_rate": float(harmed_selected.sum() / common.sum()),
            "harmful_selected_fraction": float(harmed_selected.sum() / selected.sum().clamp_min(1)),
            "clean_pixel_degradation": float(clean_harm.sum() / clean.sum().clamp_min(1)),
        })
    return row, magnitude_rows


@torch.no_grad()
def one_step(model, extractor, adapter, item: dict, state: dict | None, *, time_since_reset: int, mode: str):
    """One causal pair, retaining both raw historical and recurrent candidates."""
    if state is None:
        state = {"disparity": item["past"], "valid": item["past_valid"].bool(),
                 "gt": item["past_gt"], "gt_coverage": item["past_gt_coverage"]}
    forward = adapter.current_to_past(item["current_rgb"], item["past_rgb"])
    backward = adapter.past_to_current(item["past_rgb"], item["current_rgb"])
    recurrent = temporal_disparity_evidence(item["raw"], state["disparity"], forward, backward,
        current_valid=item["raw_valid"], past_valid=state["valid"], current_rgb=item["current_rgb"], past_rgb=item["past_rgb"])
    historical = temporal_disparity_evidence(item["raw"], item["past"], forward, backward,
        current_valid=item["raw_valid"], past_valid=item["past_valid"], current_rgb=item["current_rgb"], past_rgb=item["past_rgb"])
    cues = build_codd_cues(extractor, raw=item["raw"], aligned_memory=recurrent.aligned_past_disparity,
        current_rgb=item["current_rgb"], current_right_rgb=item["current_right_rgb"], past_rgb=item["past_rgb"],
        flow_current_to_past=forward, flow_magnitude=recurrent.flow_magnitude,
        forward_backward_confidence=recurrent.forward_backward_confidence, warp_support=recurrent.warp_support,
        aligned_valid=recurrent.aligned_validity)
    output = model(cues, item["raw"], recurrent.aligned_past_disparity)
    common, hmask, rmask = common_support(item, historical, recurrent)
    # State must advance even when a frame has no paired GT/warp support, but
    # that frame is not an evaluable common-support observation.
    next_state = {"disparity": output.fused_disparity, "valid": item["raw_valid"].bool(),
                  "gt": item["gt"], "gt_coverage": item["gt_coverage"]}
    if not bool(common.any()):
        return None, next_state
    gt_warp = causal_warp(state["gt"], forward, source_valid=state["gt_coverage"] > .50)
    return frame_metrics(item=item, historical_memory=historical.aligned_past_disparity,
        recurrent_memory=recurrent.aligned_past_disparity, output=output, common=common,
        historical_mask=hmask, recurrent_mask=rmask, mode=mode,
        step_since_reset=time_since_reset, gt_aligned=gt_warp.warped, gt_aligned_valid=gt_warp.valid), next_state


@torch.no_grad()
def evaluate_clip_reset(model, extractor, adapter, dataset, config) -> tuple[list[dict], list[dict]]:
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.workers,
        persistent_workers=config.workers > 0, pin_memory=True, prefetch_factor=4 if config.workers else None)
    rows=[]; magnitude=[]
    for cpu in loader:
        clip=to_device(cpu, next(model.parameters()).device)
        _, states=run_clip(model, extractor, adapter, clip, config, training=False)
        for index, state in enumerate(states, start=1):
            item, output=state["item"], state["output"]
            # The original Phase-1 raw-t-1 candidate and actual recurrent state
            # share a common candidate definition here.
            historical = type("Evidence", (), {"aligned_past_disparity": state["raw_memory"]})()
            recurrent = type("Evidence", (), {"aligned_past_disparity": state["state_memory"]})()
            common=state["raw_memory_valid"] & state["state_mask"]
            hmask, rmask=state["raw_memory_valid"], state["state_mask"]
            if not bool(common.any()):
                continue
            row, sweep=frame_metrics(item=item, historical_memory=historical.aligned_past_disparity,
                recurrent_memory=recurrent.aligned_past_disparity, output=output, common=common,
                historical_mask=hmask, recurrent_mask=rmask, mode="clip_reset", step_since_reset=index,
                gt_aligned=state["gt_aligned"], gt_aligned_valid=state["gt_aligned_valid"])
            rows.append(row); magnitude.extend(sweep)
    return rows, magnitude


@torch.no_grad()
def evaluate_continuous(model, extractor, adapter, dataset, config) -> tuple[list[dict], list[dict]]:
    loader=DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.workers,
        persistent_workers=config.workers > 0, pin_memory=True, prefetch_factor=4 if config.workers else None)
    rows=[]; magnitude=[]; state=None; prior=None; step=0
    for cpu in loader:
        item=to_device(cpu, next(model.parameters()).device)
        sequence, backbone, _=metadata(item)
        key=(sequence, backbone)
        if key != prior:
            state=None; step=0; prior=key
        step += 1
        observation, state=one_step(model, extractor, adapter, item, state, time_since_reset=step, mode="continuous_streaming")
        if observation is not None:
            row, sweep=observation
            rows.append(row); magnitude.extend(sweep)
    return rows, magnitude


@torch.no_grad()
def evaluate_reset_every_four(model, extractor, adapter, dataset, config) -> tuple[list[dict], list[dict]]:
    """Phase-1 reset policy evaluated on every pair, matching streaming support."""
    loader=DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.workers,
        persistent_workers=config.workers > 0, pin_memory=True, prefetch_factor=4 if config.workers else None)
    rows=[]; magnitude=[]; state=None; prior=None; step=0
    for cpu in loader:
        item=to_device(cpu, next(model.parameters()).device)
        sequence, backbone, _=metadata(item)
        key=(sequence, backbone)
        if key != prior:
            state=None; step=0; prior=key
        step += 1
        if (step - 1) % config.clip_length == 0:
            state=None
        observation, state=one_step(model, extractor, adapter, item, state, time_since_reset=((step - 1) % config.clip_length) + 1, mode="reset_every4_all_pairs")
        if observation is not None:
            row, sweep=observation
            rows.append(row); magnitude.extend(sweep)
    return rows, magnitude


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {}
    weights=np.asarray([row["valid_count"] for row in rows], dtype=float)
    weights/=weights.sum()
    result={}
    for key in rows[0]:
        if key in {"mode", "sequence", "backbone", "frame_id", "step_since_reset", "valid_count"}:
            continue
        values=np.asarray([row[key] for row in rows], dtype=float)
        if np.isfinite(values).all():
            result[key]=float(np.sum(weights*values))
    for oracle in ("historical_selection", "recurrent_selection", "convex_fusion"):
        gain=result.get(f"{oracle}_gain", float("nan"))
        result[f"{oracle}_normalized_gain"]=float(result["fused_gain"] / gain) if gain > 1e-12 else float("nan")
    result["frames_worsened_fraction"]=float(np.mean([row["fused_gain"] < 0 for row in rows]))
    result["worst_frame_degradation"]=float(max(-row["fused_gain"] for row in rows))
    return result


def grouped(rows: list[dict], field: str) -> list[dict]:
    groups=defaultdict(list)
    for row in rows: groups[row[field]].append(row)
    return [{field:value, **aggregate(group)} for value,group in sorted(groups.items())]


def aggregate_magnitude(rows: list[dict]) -> list[dict]:
    grouped_rows=defaultdict(list)
    for row in rows: grouped_rows[(row["mode"], row["threshold_cache_px"])].append(row)
    result=[]
    for (mode, threshold), group in sorted(grouped_rows.items()):
        weights=np.asarray([row["valid_count"] for row in group],dtype=float); weights/=weights.sum()
        output={"mode":mode,"threshold_cache_px":threshold,"threshold_native_px":group[0]["threshold_native_px"]}
        for key in ("coverage","epe","gain","harmful_update_rate","harmful_selected_fraction","clean_pixel_degradation"):
            output[key]=float(sum(weight*row[key] for weight,row in zip(weights,group)))
        output["degraded_frame_fraction"]=float(np.mean([row["gain"]<0 for row in group]))
        output["worst_frame_degradation"]=float(max(-row["gain"] for row in group))
        result.append(output)
    return result


def checkpoint(config, seed: int):
    path=config.source_root/f"seed_{seed}"/"checkpoints/best_validation.pt"
    state=torch.load(path,map_location="cpu",weights_only=False)
    model=CODDStyleFusionHead(state["cue_channels"]).to(config.device)
    model.load_state_dict(state["model"]); model.eval()
    return model, {"seed":seed,"path":str(path),"sha256":sha256(path),"cue_channels":state["cue_channels"],"epoch":state["epoch"]}


def config_for_runner(config: argparse.Namespace) -> argparse.Namespace:
    # run_clip expects these Phase-1 fixed configuration attributes.
    return argparse.Namespace(coverage_threshold=config.coverage_threshold, clip_length=config.clip_length,
        tau_reset_native_px=5.0, tau_fusion_native_px=1.0, alpha_reg=.2,
        memory_state="recurrent", disable_learned_stereo_evidence=False,
        max_clips_per_sequence=None, seed=config.seed, workers=config.workers)


def run_audit(config: argparse.Namespace) -> None:
    if config.seed is None: raise ValueError("--seed is required for audit")
    seed_all(config.seed); device=torch.device(config.device); runner_config=config_for_runner(config)
    model, checkpoint_info=checkpoint(config, config.seed)
    extractor=FrozenResNet18Layer1().to(device); adapter=BiDAFlowInferenceAdapter("sea_raft",device=device)
    assert not any(p.requires_grad for p in extractor.parameters())
    assert not any(p.requires_grad for p in adapter.model.parameters())
    output=config.output_root
    save_json(output/"checkpoint_hashes.json", checkpoint_info)
    save_json(output/"protocol_audit.json", {**manifest(runner_config), "stage":"post-hoc frozen-checkpoint mechanism audit",
        "continuous_streaming":"state resets only at (sequence, backbone) boundaries", "no_future_access":True})
    if config.evaluation_mode in ("both", "clip_reset"):
        clips=make_dataset(TEST, runner_config); clips.pairs.preload_frame_data(config.preload_workers)
        clip_rows, clip_magnitude=evaluate_clip_reset(model, extractor, adapter, clips, runner_config)
        write_csv(output/"clip_reset_rows.csv",clip_rows); write_csv(output/"clip_reset_magnitude.csv",clip_magnitude)
        save_json(output/"clip_reset_summary.json", {"seed":config.seed,"summary":aggregate(clip_rows),"checkpoint":checkpoint_info})
    if config.evaluation_mode in ("both", "continuous_streaming"):
        pairs=TemporalPairDataset(SEEN_BACKBONES, TEST, coverage_threshold=config.coverage_threshold, include_right_rgb=True)
        pairs.preload_frame_data(config.preload_workers)
        continuous_rows, continuous_magnitude=evaluate_continuous(model, extractor, adapter, pairs, runner_config)
        write_csv(output/"continuous_streaming_rows.csv",continuous_rows); write_csv(output/"continuous_streaming_magnitude.csv",continuous_magnitude)
        save_json(output/"continuous_streaming_summary.json", {"seed":config.seed,"summary":aggregate(continuous_rows),"checkpoint":checkpoint_info})
    if config.evaluation_mode == "reset_every4_all_pairs":
        pairs=TemporalPairDataset(SEEN_BACKBONES, TEST, coverage_threshold=config.coverage_threshold, include_right_rgb=True)
        pairs.preload_frame_data(config.preload_workers)
        reset_rows, reset_magnitude=evaluate_reset_every_four(model, extractor, adapter, pairs, runner_config)
        write_csv(output/"reset_every4_all_pairs_rows.csv",reset_rows); write_csv(output/"reset_every4_all_pairs_magnitude.csv",reset_magnitude)
        save_json(output/"reset_every4_all_pairs_summary.json", {"seed":config.seed,"summary":aggregate(reset_rows),"checkpoint":checkpoint_info})


def combine(config: argparse.Namespace) -> None:
    output=config.output_root/"posthoc_oracles"/"canonical"
    clip=json.loads((output/"clip_reset_summary.json").read_text())["summary"]
    reset=json.loads((output/"reset_every4_all_pairs_summary.json").read_text())["summary"]
    stream=json.loads((output/"continuous_streaming_summary.json").read_text())["summary"]
    clip_rows=list(csv.DictReader((output/"clip_reset_rows.csv").open()))
    stream_rows=list(csv.DictReader((output/"continuous_streaming_rows.csv").open()))
    reset_rows=list(csv.DictReader((output/"reset_every4_all_pairs_rows.csv").open()))
    clip_magnitude=list(csv.DictReader((output/"clip_reset_magnitude.csv").open()))
    stream_magnitude=list(csv.DictReader((output/"continuous_streaming_magnitude.csv").open()))
    reset_magnitude=list(csv.DictReader((output/"reset_every4_all_pairs_magnitude.csv").open()))
    # CSV parsing restores numeric cells as strings; re-use the original JSON
    # summaries for headline values and keep frame-level CSV compact.
    rows=clip_rows+reset_rows+stream_rows
    write_csv(output/"oracle_comparison.csv",rows); write_csv(output/"gain_decomposition.csv",rows)
    write_csv(output/"intervention_magnitude.csv",clip_magnitude+reset_magnitude+stream_magnitude)
    write_csv(output/"streaming_comparison.csv",[{"mode":"clip_reset_phase1_subsampled",**clip},{"mode":"reset_every4_all_pairs",**reset},{"mode":"continuous_streaming",**stream}])
    # Grouping below needs numeric values, so reload through a small converter.
    numeric=[]
    for row in rows:
        converted={key:(float(value) if key not in {"mode","sequence","backbone","frame_id"} else value) for key,value in row.items()}
        converted["valid_count"]=int(float(row["valid_count"])); converted["step_since_reset"]=int(float(row["step_since_reset"]))
        numeric.append(converted)
    write_csv(output/"per_backbone_metrics.csv",grouped(numeric,"backbone"))
    write_csv(output/"per_sequence_metrics.csv",grouped(numeric,"sequence"))
    write_csv(output/"temporal_metrics.csv",rows)
    common={"checkpoint_policy":"single canonical Phase-1 seed 0: lowest ID-2 validation fused EPE among existing frozen checkpoints",
            "clip_rows":len(clip_rows),"reset_every4_rows":len(reset_rows),"continuous_rows":len(stream_rows),"all_common_counts_positive":all(row["valid_count"]>0 for row in numeric),
            "historical_recurrent_mask_difference_pixels":int(sum(row["historical_only_count"]+row["recurrent_only_count"] for row in numeric)),
            "same_common_support_contract":"historical mask intersect recurrent mask for every reported comparison"}
    save_json(output/"common_support_audit.json",common)
    save_json(output/"aggregate_summary.json",{"seed_count":1,"checkpoint_policy":common["checkpoint_policy"],"clip_reset_phase1_subsampled":clip,"reset_every4_all_pairs":reset,
        "continuous_streaming":stream,"units":"cache-grid disparity pixels at width 180",
        "interpretation":"historical, recurrent-selection, and convex-fusion normalised gains are distinct quantities"})


def main() -> None:
    config=arguments()
    if config.mode == "combine": combine(config); return
    if config.mode == "audit":
        config.output_root=config.output_root/"posthoc_oracles"/"canonical"
        run_audit(config); return
    raise NotImplementedError("hard endpoint evaluation is enabled after the frozen oracle audit is reviewed")


if __name__ == "__main__":
    main()
