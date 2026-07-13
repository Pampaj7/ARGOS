#!/usr/bin/env python3
"""Train/evaluate the minimal ARGOS v2 learned long-memory selector."""
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

from model_design.data.temporal_memory_dataset import TemporalMemoryDataset  # noqa: E402
from model_design.data.temporal_pair_dataset import (  # noqa: E402
    DEFAULT_VALIDATION_SEQUENCES,
    PRIMARY_UNSEEN_BACKBONE,
    SEEN_BACKBONES,
    build_split_manifest,
)
from model_design.external_components.bidavideo import (  # noqa: E402
    BiDAFlowInferenceAdapter,
    temporal_disparity_evidence,
)
from model_design.losses.ppm_losses import PPMLossConfig, learned_ppm_losses  # noqa: E402
from model_design.models.learned_ppm_selector import LearnedPPMSelectorRefiner  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "train", "evaluate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--backbones", nargs="+", default=list(SEEN_BACKBONES))
    parser.add_argument("--train-sequences", nargs="+")
    parser.add_argument("--validation-sequences", nargs="+", default=list(DEFAULT_VALIDATION_SEQUENCES))
    parser.add_argument("--ages", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--coverage-threshold", type=float, default=0.50)
    parser.add_argument("--max-train-samples-per-sequence", type=int, default=256)
    parser.add_argument("--max-validation-samples-per-sequence", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8))
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def save_json(path: Path, value: object) -> None:
    def clean(item):
        if isinstance(item, dict):
            return {key: clean(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(child) for child in item]
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def append_csv(path: Path, row: dict) -> None:
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loader(dataset: TemporalMemoryDataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        generator=torch.Generator().manual_seed(args.seed),
    )


def to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def build_evidence(adapter: BiDAFlowInferenceAdapter, batch: dict) -> tuple[dict[str, torch.Tensor], float]:
    b, m = batch["past"].shape[:2]
    current_rgb = batch["current_rgb"][:, None].expand(-1, m, -1, -1, -1)
    current_flat = current_rgb.reshape(b * m, *current_rgb.shape[2:])
    past_rgb = batch["past_rgb"].reshape(b * m, *batch["past_rgb"].shape[2:])
    target = torch.cat((current_flat, past_rgb), dim=0)
    source = torch.cat((past_rgb, current_flat), dim=0)
    if target.is_cuda:
        torch.cuda.synchronize(target.device)
    tick = time.perf_counter()
    inferred = adapter.infer(target, source).detach()
    if target.is_cuda:
        torch.cuda.synchronize(target.device)
    latency = (time.perf_counter() - tick) * 1000 / b
    raw = batch["raw"][:, None].expand(-1, m, -1, -1, -1).reshape(b * m, 1, *batch["raw"].shape[-2:])
    raw_valid = batch["raw_valid"][:, None].expand(-1, m, -1, -1, -1).reshape_as(raw)
    past = batch["past"].reshape(b * m, 1, *batch["past"].shape[-2:])
    past_valid = batch["past_valid"].reshape_as(past)
    with torch.no_grad():
        value = temporal_disparity_evidence(
            raw,
            past,
            inferred[: b * m],
            inferred[b * m :],
            current_valid=raw_valid,
            past_valid=past_valid,
            current_rgb=current_flat,
            past_rgb=past_rgb,
        )
    return {
        name: tensor.detach().reshape(b, m, *tensor.shape[1:])
        for name, tensor in value.as_dict().items()
    }, latency


@torch.no_grad()
def validate(model, adapter, data_loader, device, coverage_threshold) -> tuple[float, float, float, float]:
    model.eval()
    sums = defaultdict(float)
    count = 0
    for cpu in data_loader:
        batch = to_device(cpu, device)
        evidence, _ = build_evidence(adapter, batch)
        output = model(batch["raw"], batch["raw_valid"], evidence, batch["ages"][0])
        valid = (
            (batch["gt_coverage"] > coverage_threshold)
            & batch["raw_valid"].bool()
            & evidence["aligned_validity"][:, 0]
            & evidence["warp_support"][:, 0]
        )
        raw_error = (batch["raw"] - batch["gt"]).abs()
        refined_error = (output.disparity - batch["gt"]).abs()
        memory_error = (evidence["aligned_past_disparity"] - batch["gt"][:, None]).abs()
        memory_error = memory_error.masked_fill(~(evidence["aligned_validity"] & evidence["warp_support"]), torch.inf)
        oracle_t1 = torch.minimum(raw_error, memory_error[:, 0])
        oracle_multi = torch.minimum(raw_error, memory_error.min(dim=1).values)
        n = int(valid.sum())
        sums["raw"] += float(raw_error[valid].sum())
        sums["refined"] += float(refined_error[valid].sum())
        sums["t1"] += float(oracle_t1[valid].sum())
        sums["multi"] += float(oracle_multi[valid].sum())
        count += n
    return tuple(sums[key] / max(count, 1) for key in ("raw", "refined", "t1", "multi"))


def train(args: argparse.Namespace, smoke: bool) -> int:
    seed_all(args.seed)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = build_split_manifest(validation_sequences=args.validation_sequences)
    train_sequences = args.train_sequences or manifest["train_sequences"]
    backbones = ["S2M2-S"] if smoke else args.backbones
    if PRIMARY_UNSEEN_BACKBONE in backbones:
        raise ValueError("unseen backbone cannot enter training")
    if smoke:
        train_sequences = [train_sequences[0]]
        args.max_train_samples_per_sequence = 24
        args.max_validation_samples_per_sequence = 8
    manifest.update({"actual_train_sequences": train_sequences, "actual_training_backbones": backbones, "ages": args.ages})
    save_json(args.output / "split_manifest.json", manifest)
    save_json(args.output / "config.json", vars(args))
    train_data = TemporalMemoryDataset(backbones, train_sequences, ages=args.ages, max_samples_per_sequence=args.max_train_samples_per_sequence, random_clip_start=True, seed=args.seed)
    val_data = TemporalMemoryDataset(backbones, args.validation_sequences, ages=args.ages, max_samples_per_sequence=args.max_validation_samples_per_sequence, seed=args.seed)
    train_loader = loader(train_data, args, True)
    val_loader = loader(val_data, args, False)
    model = LearnedPPMSelectorRefiner().to(device)
    adapter = BiDAFlowInferenceAdapter("sea_raft", device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_config = PPMLossConfig()
    best = math.inf
    start_epoch = 0
    final_path = args.output / "checkpoints/final.pt"
    best_path = args.output / "checkpoints/best_validation.pt"
    if args.resume and final_path.exists():
        state = torch.load(final_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1; best = state["best_validation_epe"]
    history = args.output / "training_history.csv"
    if not args.resume:
        history.unlink(missing_ok=True)
    log = (args.output / "run.log").open("a", buffering=1)
    print(f"COMMAND {' '.join(sys.argv)}", file=log)
    print(f"DATA train={len(train_data)} validation={len(val_data)}", file=log)
    step = 0
    first_loss = None
    for epoch in range(start_epoch, args.epochs if not smoke else 1000000):
        model.train(); sums = defaultdict(float); batches = 0
        for cpu in train_loader:
            batch = to_device(cpu, device)
            evidence, flow_ms = build_evidence(adapter, batch)
            valid = (
                (batch["gt_coverage"] > args.coverage_threshold)
                & batch["raw_valid"].bool()
                & evidence["aligned_validity"][:, 0]
                & evidence["warp_support"][:, 0]
            )
            candidate_valid = evidence["aligned_validity"] & evidence["warp_support"]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(batch["raw"], batch["raw_valid"], evidence, batch["ages"][0])
                losses = learned_ppm_losses(
                    output,
                    raw=batch["raw"], candidates=evidence["aligned_past_disparity"],
                    candidate_valid=candidate_valid, gt=batch["gt"], valid=valid,
                    config=loss_config,
                )
            scaler.scale(losses["total"]).backward(); scaler.unscale_(optimizer)
            grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
            scaler.step(optimizer); scaler.update()
            for name, value in losses.items(): sums[name] += float(value.detach())
            sums["gradient_norm"] += grad; sums["flow_latency_ms"] += flow_ms
            sums["memory_mass"] += float(output.play_weights.sum(dim=1).mean().detach())
            batches += 1; step += 1
            if first_loss is None: first_loss = float(losses["total"].detach())
            if args.steps and step >= args.steps: break
        raw, refined, t1, multi = validate(model, adapter, val_loader, device, args.coverage_threshold)
        row = {"epoch": epoch, "step": step, **{key: value/max(batches,1) for key,value in sums.items()}, "validation_raw_epe": raw, "validation_refined_epe": refined, "validation_t1_oracle_epe": t1, "validation_multi_oracle_epe": multi, "additional_oracle_recovery": (t1-refined)/max(t1-multi,1e-8)}
        append_csv(history, row)
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best_validation_epe": min(best, refined), "ages": args.ages, "manifest": manifest, "loss_config": asdict(loss_config)}
        final_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, final_path)
        if refined < best: best = refined; torch.save(payload, best_path)
        print(json.dumps(row), file=log)
        if args.steps and step >= args.steps: break
    smoke_result = {"passed": bool(row["gradient_norm"] > 0 and row["total"] < first_loss * 0.85 and math.isfinite(row["validation_refined_epe"])), "initial_loss": first_loss, "final_loss": row["total"], "gradient_norm": row["gradient_norm"], "memory_mass": row["memory_mass"], "required_loss_reduction": 0.15}
    save_json(args.output / "smoke_result.json", smoke_result)
    print(f"COMPLETE best={best} smoke={smoke_result}", file=log); log.close()
    return 0 if not smoke or smoke_result["passed"] else 2


def evaluate(args: argparse.Namespace) -> int:
    if args.checkpoint is None: raise ValueError("--checkpoint required")
    seed_all(args.seed); device = torch.device(args.device); args.output.mkdir(parents=True, exist_ok=True)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = LearnedPPMSelectorRefiner().to(device); model.load_state_dict(state["model"]); model.eval()
    adapter = BiDAFlowInferenceAdapter("sea_raft", device=device)
    data = TemporalMemoryDataset(args.backbones, args.validation_sequences, ages=state["ages"], max_samples_per_sequence=args.max_validation_samples_per_sequence, seed=args.seed)
    data_loader = loader(data, args, False)
    rows=[]; flow_times=[]; model_times=[]
    with torch.no_grad():
        for cpu in data_loader:
            batch=to_device(cpu,device); evidence,flow_ms=build_evidence(adapter,batch); flow_times.append(flow_ms)
            tick=time.perf_counter(); output=model(batch["raw"],batch["raw_valid"],evidence,batch["ages"][0]); torch.cuda.synchronize(device); model_times.append((time.perf_counter()-tick)*1000/batch["raw"].shape[0])
            candidate_error=(evidence["aligned_past_disparity"]-batch["gt"][:,None]).abs().masked_fill(~(evidence["aligned_validity"]&evidence["warp_support"]),torch.inf)
            raw_error=(batch["raw"]-batch["gt"]).abs(); learned_error=(output.disparity-batch["gt"]).abs(); oracle=torch.minimum(raw_error,candidate_error.min(1).values)
            valid=(batch["gt_coverage"]>args.coverage_threshold)&batch["raw_valid"].bool()&evidence["aligned_validity"][:,0]&evidence["warp_support"][:,0]
            for index in range(batch["raw"].shape[0]):
                mask=valid[index]; n=int(mask.sum()); clean=mask&(raw_error[index]<=.5); deg=float(learned_error[index][mask].mean()-raw_error[index][mask].mean()) if n else math.nan
                all_choice = torch.cat(
                    (output.raw_abstain_weight[index][None], output.play_weights[index]), dim=0
                ).argmax(dim=0)
                row={"backbone":batch["backbone"][index],"sequence":batch["sequence"][index],"frame_id":batch["current_frame_id"][index],"valid_count":n,"raw_error_sum":float(raw_error[index][mask].sum()),"learned_error_sum":float(learned_error[index][mask].sum()),"oracle_error_sum":float(oracle[index][mask].sum()),"clean_count":int(clean.sum()),"clean_degraded_count":int((clean&(learned_error[index]>raw_error[index]+.02)).sum()),"false_update_count":int((clean&(output.update[index].abs()>.05)).sum()),"frame_degradation":deg,"memory_mass":float(output.play_weights[index].sum(0)[mask].mean()) if n else math.nan,"selected_raw_count":int(((all_choice==0)&mask).sum())}
                for age_index, age in enumerate(state["ages"]):
                    row[f"age_{age}_weight_sum"] = float(output.play_weights[index, age_index][mask].sum())
                    row[f"selected_age_{age}_count"] = int(((all_choice == age_index + 1) & mask).sum())
                rows.append(row)
    write_path=args.output/"frame_metrics.csv"; write_path.unlink(missing_ok=True)
    for row in rows: append_csv(write_path,row)
    groups=defaultdict(list)
    for row in rows: groups[(row["backbone"],row["sequence"])].append(row)
    sequence=[]
    for (backbone,seq),values in groups.items():
        n=sum(v["valid_count"] for v in values); clean=sum(v["clean_count"] for v in values); deg=np.asarray([v["frame_degradation"] for v in values])
        sequence.append({"backbone":backbone,"sequence":seq,"frames":len(values),"valid_count":n,"raw_epe":sum(v["raw_error_sum"] for v in values)/max(n,1),"learned_epe":sum(v["learned_error_sum"] for v in values)/max(n,1),"oracle_epe":sum(v["oracle_error_sum"] for v in values)/max(n,1),"clean_degradation_ratio":sum(v["clean_degraded_count"] for v in values)/max(clean,1),"false_update_rate":sum(v["false_update_count"] for v in values)/max(clean,1),"frames_worsened_ratio":float((deg>0).mean()),"worst_frame_degradation":float(deg.max()),"p95_frame_degradation":float(np.percentile(deg,95))})
    seq_path=args.output/"sequence_metrics.csv"; seq_path.unlink(missing_ok=True)
    for row in sequence: append_csv(seq_path,row)
    n=sum(v["valid_count"] for v in rows); raw=sum(v["raw_error_sum"] for v in rows)/max(n,1); learned=sum(v["learned_error_sum"] for v in rows)/max(n,1); oracle=sum(v["oracle_error_sum"] for v in rows)/max(n,1)
    summary={"backbones":args.backbones,"raw_epe":raw,"learned_epe":learned,"multi_oracle_epe":oracle,"oracle_recovery":(raw-learned)/max(raw-oracle,1e-8),"additional_gain_beyond_existing_t1_model":"compare against baseline_summary.json","flow_latency_ms":float(np.mean(flow_times)),"model_latency_ms":float(np.mean(model_times)),"parameter_count":sum(p.numel() for p in model.parameters())}
    selected_total = n
    age_statistics = {
        str(age): {
            "mean_play_weight": sum(r[f"age_{age}_weight_sum"] for r in rows) / max(n, 1),
            "argmax_selection_ratio": sum(r[f"selected_age_{age}_count"] for r in rows) / max(selected_total, 1),
        }
        for age in state["ages"]
    }
    calibration={"mean_memory_mass":sum(value["mean_play_weight"] for value in age_statistics.values()),"raw_abstain_argmax_ratio":sum(r["selected_raw_count"] for r in rows)/max(selected_total,1),"age_statistics":age_statistics,"effective_memory_horizon":sum(float(age)*value["mean_play_weight"] for age,value in age_statistics.items())/max(sum(value["mean_play_weight"] for value in age_statistics.values()),1e-8)}
    save_json(args.output/"aggregate_summary.json",summary); save_json(args.output/"safety_summary.json",sequence); save_json(args.output/"selector_calibration.json",calibration)
    return 0


def main() -> int:
    args=parse_args()
    return evaluate(args) if args.mode=="evaluate" else train(args,args.mode=="smoke")


if __name__ == "__main__": raise SystemExit(main())
