#!/usr/bin/env python3
"""Train/evaluate the controlled ARGOS v2 E2-E5 latent-state ladder."""
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
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))
sys.path.insert(0, str(V2_ROOT / "scripts"))

from model_design.data.temporal_clip_dataset import TemporalClipDataset  # noqa: E402
from model_design.data.temporal_pair_dataset import (  # noqa: E402
    DEFAULT_VALIDATION_SEQUENCES,
    PRIMARY_UNSEEN_BACKBONE,
    SEEN_BACKBONES,
    build_split_manifest,
)
from model_design.external_components.bidavideo import BiDAFlowInferenceAdapter  # noqa: E402
from model_design.external_components.endostreamdepth import CausalState  # noqa: E402
from model_design.losses.safety_losses import SafetyLossConfig, learned_t1_losses  # noqa: E402
from model_design.models.latent_t1_refiner import LatentT1Refiner  # noqa: E402
from model_design.models.learned_t1_refiner import LearnedT1Refiner  # noqa: E402
from run_learned_t1_refiner import (  # noqa: E402
    aggregate_frame_rows,
    atomic_checkpoint,
    boundary_mask,
    build_evidence,
    frame_metrics,
    save_json,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "train", "evaluate", "diagnose"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("E2", "E3", "E4", "E5"), default="E3")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--a2-checkpoint", type=Path, default=V2_ROOT / "results/learned_t1_refiner/ablations/A2/checkpoints/best_validation.pt")
    parser.add_argument("--backbones", nargs="+", default=list(SEEN_BACKBONES))
    parser.add_argument("--train-sequences", nargs="+")
    parser.add_argument("--validation-sequences", nargs="+", default=list(DEFAULT_VALIDATION_SEQUENCES))
    parser.add_argument("--coverage-threshold", type=float, default=0.50)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.25, 0.50, 0.90])
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--burn-in", type=int, default=1)
    parser.add_argument("--detach-interval", type=int, default=0, help="0 means once at clip boundary")
    parser.add_argument("--max-train-clips-per-sequence", type=int, default=16)
    parser.add_argument("--max-validation-clips-per-sequence", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8))
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--feature-channels", type=int, default=32)
    parser.add_argument("--state-channels", type=int, default=16)
    parser.add_argument("--tau-px", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-diagnostic-clips", type=int, default=8)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def serializable_args(args: argparse.Namespace) -> dict:
    """Convert every CLI path, including baseline paths, before JSON logging."""
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def validate_tuning_backbones(backbones: list[str]) -> None:
    forbidden = {PRIMARY_UNSEEN_BACKBONE, "CREStereo"} & set(backbones)
    if forbidden:
        raise ValueError(f"unseen backbones cannot enter training/tuning: {sorted(forbidden)}")
    unknown = set(backbones) - set(SEEN_BACKBONES)
    if unknown:
        raise ValueError(f"training supports only the seen pool: {sorted(unknown)}")


def to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def loader(dataset: TemporalClipDataset, args: argparse.Namespace, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0, drop_last=False,
        generator=torch.Generator().manual_seed(args.seed),
    )


def time_slice(batch: dict, step: int) -> dict:
    result = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim >= 2 and value.shape[1] == batch["raw"].shape[1]:
            result[key] = value[:, step]
        elif key in {"past_frame_id", "current_frame_id", "past_index", "current_index"}:
            result[key] = value[step]
        else:
            result[key] = value
    return result


def current_ids(batch: dict, step: int) -> list[str]:
    # String frame fields collate as T lists, each containing B strings.
    values = batch["current_frame_id"]
    return list(values[step]) if values and isinstance(values[0], (list, tuple)) else list(values)


def placeholder_evidence(frame: dict) -> dict[str, torch.Tensor]:
    raw = frame["raw"]
    zero = torch.zeros_like(raw)
    return {"current_valid": frame["raw_valid"], "aligned_past_disparity": raw,
            "aligned_validity": frame["raw_valid"], "warp_support": frame["raw_valid"],
            "forward_backward_error": zero, "forward_backward_confidence": zero,
            "photometric_residual": zero, "flow_magnitude": zero,
            "absolute_disparity_disagreement": zero}


def evidence_for_frame(adapter: BiDAFlowInferenceAdapter | None, frame: dict) -> tuple[dict[str, torch.Tensor], float]:
    if adapter is None:
        return placeholder_evidence(frame), 0.0
    evidence, latency = build_evidence(adapter, frame)
    evidence["absolute_disparity_disagreement"] = (
        frame["raw"] - evidence["aligned_past_disparity"]
    ).abs()
    return evidence, latency


def state_loss_config(variant: str) -> SafetyLossConfig:
    base = SafetyLossConfig()
    return replace(base, memory_gate_weight=0.0) if variant == "E2" else base


def make_datasets(args: argparse.Namespace, *, smoke: bool = False) -> tuple[TemporalClipDataset, TemporalClipDataset, dict]:
    manifest = build_split_manifest(seed=args.seed, coverage_threshold=args.coverage_threshold)
    train_sequences = list(args.train_sequences or manifest["train_sequences"])
    backbones = ["S2M2-S"] if smoke else list(args.backbones)
    validate_tuning_backbones(backbones)
    if smoke:
        train_sequences = [train_sequences[0]]
    train = TemporalClipDataset(
        backbones, train_sequences, clip_length=args.clip_length,
        max_clips_per_sequence=min(2, args.max_train_clips_per_sequence) if smoke else args.max_train_clips_per_sequence,
        random_clip_selection=True, seed=args.seed, coverage_threshold=args.coverage_threshold,
    )
    validation = TemporalClipDataset(
        backbones, args.validation_sequences[:1] if smoke else args.validation_sequences,
        clip_length=args.clip_length,
        max_clips_per_sequence=1 if smoke else args.max_validation_clips_per_sequence,
        random_clip_selection=False, seed=args.seed, coverage_threshold=args.coverage_threshold,
    )
    manifest.update({
        "version": 2, "actual_train_sequences": train_sequences,
        "actual_training_backbones": backbones, "clip_length": args.clip_length,
        "burn_in": args.burn_in, "detach_interval": args.detach_interval,
        "loss_application": "all frames after burn-in", "state_reset": "every clip/sequence boundary",
        "bptt": "full clip unless detach_interval > 0", "gradient_clipping": 5.0,
    })
    return train, validation, manifest


@torch.no_grad()
def validation_epe(model: LatentT1Refiner, adapter: BiDAFlowInferenceAdapter | None,
                   data: DataLoader, device: torch.device, threshold: float) -> tuple[float, float]:
    model.eval(); raw_sum = refined_sum = 0.0; count = 0
    for cpu in data:
        batch = to_device(cpu, device); state = None
        for step in range(batch["raw"].shape[1]):
            frame = time_slice(batch, step); evidence, _ = evidence_for_frame(adapter, frame)
            output = model(frame["raw"], evidence, state, sequence_ids=list(batch["sequence"]),
                           frame_indices=frame["current_index"])
            state = output.state
            valid = (frame["gt_coverage"] > threshold) & frame["raw_valid"].bool()
            if model.uses_bida:
                valid &= evidence["aligned_validity"].bool() & evidence["warp_support"].bool()
            raw_error = (frame["raw"] - frame["gt"]).abs(); error = (output.disparity - frame["gt"]).abs()
            raw_sum += float(raw_error[valid].sum()); refined_sum += float(error[valid].sum()); count += int(valid.sum())
    return raw_sum / max(count, 1), refined_sum / max(count, 1)


def train(args: argparse.Namespace, *, smoke: bool) -> int:
    seed_everything(args.seed); device = torch.device(args.device); args.output.mkdir(parents=True, exist_ok=True)
    train_set, validation_set, manifest = make_datasets(args, smoke=smoke)
    save_json(args.output / "config.json", serializable_args(args))
    save_json(args.output / "split_manifest.json", manifest)
    train_loader = loader(train_set, args, shuffle=True); validation_loader = loader(validation_set, args, shuffle=False)
    model = LatentT1Refiner(args.variant, feature_channels=args.feature_channels,
                            state_channels=args.state_channels, tau_px=args.tau_px).to(device)
    adapter = BiDAFlowInferenceAdapter("sea_raft", device=device) if model.uses_bida else None
    if adapter is not None:
        assert all(not parameter.requires_grad for parameter in adapter.model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    config = state_loss_config(args.variant); history_path = args.output / "training_history.csv"
    final_path = args.output / "checkpoints/final.pt"; best_path = args.output / "checkpoints/best_validation.pt"
    start_epoch = 0; global_step = 0; best = math.inf
    if args.resume and final_path.exists():
        saved = torch.load(final_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        start_epoch = saved["epoch"] + 1; global_step = saved["global_step"]; best = saved["best_validation_epe"]
    elif not args.resume:
        history_path.unlink(missing_ok=True)
    log = (args.output / "run.log").open("a", buffering=1)
    print("COMMAND " + " ".join(sys.argv), file=log)
    print(f"DATA train_clips={len(train_set)} val_clips={len(validation_set)}", file=log)
    first_loss = None; first_geometry = None; initial_state_injection = float(model.state_injection.detach())
    target_epochs = max(args.epochs, 100000) if smoke and args.steps else args.epochs
    stop = False
    for epoch in range(start_epoch, target_epochs):
        model.train(); sums: defaultdict[str, float] = defaultdict(float); batches = 0
        for cpu in train_loader:
            batch = to_device(cpu, device); state = None; per_step = []; optimizer.zero_grad(set_to_none=True)
            flow_ms = 0.0
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                for step in range(batch["raw"].shape[1]):
                    frame = time_slice(batch, step); evidence, latency = evidence_for_frame(adapter, frame); flow_ms += latency
                    output = model(frame["raw"], evidence, state, sequence_ids=list(batch["sequence"]),
                                   frame_indices=frame["current_index"])
                    state = output.state
                    valid = frame["gt_valid"].bool() & frame["raw_valid"].bool()
                    if model.uses_bida:
                        valid &= evidence["aligned_validity"].bool() & evidence["warp_support"].bool()
                    if step >= args.burn_in:
                        per_step.append(learned_t1_losses(
                            output, raw=frame["raw"], aligned_memory=evidence["aligned_past_disparity"],
                            gt=frame["gt"], valid=valid, safety_valid=valid, config=config,
                        ))
                    if args.detach_interval and (step + 1) % args.detach_interval == 0:
                        state = state.detach()
                if not per_step:
                    raise RuntimeError("burn-in leaves no supervised frame")
                losses = {key: torch.stack([item[key] for item in per_step]).mean() for key in per_step[0]}
            scaler.scale(losses["total"]).backward(); scaler.unscale_(optimizer)
            grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)); scaler.step(optimizer); scaler.update()
            if first_loss is None:
                first_loss = float(losses["total"].detach())
                first_geometry = float(losses["geometry"].detach())
            for key, value in losses.items(): sums[key] += float(value.detach())
            sums["gradient_norm"] += grad; sums["flow_latency_ms"] += flow_ms / batch["raw"].shape[1]
            sums["state_injection"] += float(torch.tanh(model.state_injection.detach()))
            sums["state_norm"] += float(torch.stack([value.float().square().mean().sqrt() for value in state.tensors]).mean())
            sums["update_abs_max"] = max(sums["update_abs_max"], float(output.update.detach().abs().max()))
            batches += 1; global_step += 1
            if args.steps and global_step >= args.steps: stop = True; break
        raw_epe, refined_epe = validation_epe(model, adapter, validation_loader, device, args.coverage_threshold)
        row = {"epoch": epoch, "global_step": global_step,
               **{key: value / max(batches, 1) for key, value in sums.items() if key != "update_abs_max"},
               "update_abs_max": sums["update_abs_max"], "validation_raw_epe": raw_epe,
               "validation_refined_epe": refined_epe, "validation_gain": raw_epe - refined_epe}
        append = not history_path.exists() or history_path.stat().st_size == 0
        with history_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row));
            if append: writer.writeheader()
            writer.writerow(row)
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                   "global_step": global_step, "best_validation_epe": min(best, refined_epe),
                   "variant": args.variant, "feature_channels": args.feature_channels,
                   "state_channels": args.state_channels, "tau_px": args.tau_px,
                   "split_manifest": manifest, "loss_config": asdict(config)}
        if refined_epe < best: best = refined_epe; atomic_checkpoint(best_path, payload)
        atomic_checkpoint(final_path, payload); print(json.dumps(row), file=log)
        if stop: break

    # Demonstrate learned history dependence on a deterministic held-out clip.
    model.eval(); diagnostic = next(iter(loader(validation_set, args, shuffle=False))); diagnostic = to_device(diagnostic, device)
    true_outputs = []; zero_outputs = []; state = None
    with torch.no_grad():
        for step in range(diagnostic["raw"].shape[1]):
            frame = time_slice(diagnostic, step); evidence, _ = evidence_for_frame(adapter, frame)
            true = model(frame["raw"], evidence, state, sequence_ids=list(diagnostic["sequence"]), frame_indices=frame["current_index"])
            state = true.state; zero = model(frame["raw"], evidence, None, sequence_ids=list(diagnostic["sequence"]), frame_indices=frame["current_index"])
            true_outputs.append(true.disparity); zero_outputs.append(zero.disparity)
    history_difference = float((torch.stack(true_outputs) - torch.stack(zero_outputs)).abs().mean())
    finite = all(torch.isfinite(parameter).all() for parameter in model.parameters())
    geometry_reduction = 1 - row["geometry"] / max(first_geometry or 1, 1e-8)
    total_reduction = 1 - row["total"] / max(first_loss or 1, 1e-8)
    result = {"passed": bool(finite and first_loss is not None and total_reduction > 0.10
                              and geometry_reduction > 0.20
                              and row["gradient_norm"] > 0 and row["state_norm"] > 0
                              and history_difference > 1e-6 and row["update_abs_max"] <= args.tau_px + 1e-5),
              "finite": finite, "initial_loss": first_loss, "final_loss": row["total"],
              "loss_reduction_ratio": total_reduction,
              "initial_geometry_loss": first_geometry, "final_geometry_loss": row["geometry"],
              "geometry_loss_reduction_ratio": geometry_reduction,
              "history_output_difference_px": history_difference,
              "initial_state_injection": initial_state_injection,
              "final_state_injection": float(torch.tanh(model.state_injection.detach())),
              "state_norm": row["state_norm"], "gradient_norm": row["gradient_norm"],
              "max_update_px": row["update_abs_max"]}
    save_json(args.output / "smoke_result.json", result); print("COMPLETE " + json.dumps(result), file=log); log.close()
    return 0 if (not smoke or result["passed"]) else 2


def load_state_model(args: argparse.Namespace, device: torch.device) -> tuple[LatentT1Refiner, dict]:
    if args.checkpoint is None: raise ValueError("--checkpoint is required")
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = LatentT1Refiner(saved["variant"], feature_channels=saved["feature_channels"],
                            state_channels=saved["state_channels"], tau_px=saved["tau_px"]).to(device)
    model.load_state_dict(saved["model"]); model.eval(); return model, saved


def relabel_state(state: CausalState, ids: list[str], indices: torch.Tensor) -> CausalState:
    return CausalState(state.scales, state.tensors, tuple(ids), indices.to(state.frame_indices), state.update_counts)


@torch.no_grad()
def diagnose(args: argparse.Namespace) -> int:
    seed_everything(args.seed); device = torch.device(args.device); model, saved = load_state_model(args, device)
    adapter = BiDAFlowInferenceAdapter("sea_raft", device=device) if model.uses_bida else None
    dataset = TemporalClipDataset(args.backbones, args.validation_sequences, clip_length=args.clip_length,
                                  max_clips_per_sequence=args.max_diagnostic_clips,
                                  coverage_threshold=args.coverage_threshold)
    args.batch_size = 1; data = loader(dataset, args, shuffle=False)
    modes = ["true", "zero", "stale", "shuffled", "wrong_sequence", "reset_at_max_change"]
    horizons = [1, 2, 4, 8, 16]
    # error sum, valid pixels, output difference, state RMS sum, frame count
    sums = {mode: [0.0, 0, 0.0, 0.0, 0] for mode in modes + [f"horizon_{h}" for h in horizons]}
    donor_state: CausalState | None = None
    reset_rows = []
    for clip_no, cpu in enumerate(data):
        if clip_no >= args.max_diagnostic_clips: break
        batch = to_device(cpu, device); steps = batch["raw"].shape[1]
        frames = [time_slice(batch, step) for step in range(steps)]
        evidences = [evidence_for_frame(adapter, frame)[0] for frame in frames]
        severity = [float((evidence["photometric_residual"] + evidence["flow_magnitude"].clamp(0, 32) / 32
                          + 1.0 - evidence["warp_support"].float()).mean()) for evidence in evidences]
        reset_step = int(np.argmax(severity))
        order_by_mode = {mode: (list(reversed(range(steps))) if mode == "shuffled" else list(range(steps))) for mode in modes}
        reference: dict[int, torch.Tensor] = {}
        true_final_state = None
        for mode, order in order_by_mode.items():
            state = None
            if mode == "wrong_sequence" and donor_state is not None:
                first_index = frames[0]["current_index"] - 1
                state = relabel_state(donor_state, list(batch["sequence"]), first_index)
            for logical, step in enumerate(order):
                frame = frames[step]; evidence = evidences[step]
                if mode == "reset_at_max_change" and step == reset_step:
                    state = None
                input_state = None if mode == "zero" else state
                index = torch.tensor([logical], device=device) if mode == "shuffled" else frame["current_index"]
                output = model(frame["raw"], evidence, input_state, sequence_ids=list(batch["sequence"]), frame_indices=index)
                if mode != "stale" or state is None: state = output.state
                valid = (frame["gt_coverage"] > args.coverage_threshold) & frame["raw_valid"].bool()
                if model.uses_bida: valid &= evidence["aligned_validity"].bool() & evidence["warp_support"].bool()
                error = (output.disparity - frame["gt"]).abs(); sums[mode][0] += float(error[valid].sum()); sums[mode][1] += int(valid.sum())
                state_rms = torch.stack([value.float().square().mean().sqrt() for value in output.state.tensors]).mean()
                sums[mode][3] += float(state_rms); sums[mode][4] += 1
                if mode == "true": reference[step] = output.disparity; true_final_state = output.state
                else: sums[mode][2] += float((output.disparity - reference.get(step, output.disparity)).abs().mean())
            if mode == "reset_at_max_change":
                reset_rows.append({"clip": clip_no, "sequence": batch["sequence"][0], "reset_step": reset_step,
                                   "change_severity": severity[reset_step]})
        if true_final_state is not None:
            donor_state = true_final_state.detach().clone()
        for horizon in horizons:
            state = None
            for step in range(steps):
                if step % horizon == 0: state = None
                frame = frames[step]; evidence = evidences[step]
                output = model(frame["raw"], evidence, state, sequence_ids=list(batch["sequence"]), frame_indices=frame["current_index"]); state = output.state
                valid = (frame["gt_coverage"] > args.coverage_threshold) & frame["raw_valid"].bool()
                if model.uses_bida: valid &= evidence["aligned_validity"].bool() & evidence["warp_support"].bool()
                key = f"horizon_{horizon}"; sums[key][0] += float((output.disparity-frame["gt"]).abs()[valid].sum()); sums[key][1] += int(valid.sum())
                sums[key][2] += float((output.disparity-reference[step]).abs().mean())
                sums[key][3] += float(torch.stack([value.float().square().mean().sqrt() for value in state.tensors]).mean()); sums[key][4] += 1
    rows = [{"mode": key, "epe": values[0]/max(values[1],1), "valid_count": values[1],
             "mean_output_difference_from_true_px": values[2]/max(values[4],1),
             "mean_state_rms": values[3]/max(values[4],1)}
            for key, values in sums.items()]
    save_json(args.output / "state_usage_summary.json", {"rows": rows, "checkpoint": str(args.checkpoint),
              "interpretation_rule": "promotion requires true history to differ from and outperform zero/shuffled history"})
    write_csv(args.output / "state_horizon_metrics.csv", [row for row in rows if row["mode"].startswith("horizon_")])
    write_csv(args.output / "reset_stress_metrics.csv", reset_rows)
    return 0


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> int:
    seed_everything(args.seed); device = torch.device(args.device); model, saved = load_state_model(args, device)
    if PRIMARY_UNSEEN_BACKBONE in args.backbones and set(args.backbones) != {PRIMARY_UNSEEN_BACKBONE}:
        raise ValueError("primary unseen evaluation must be isolated")
    adapter = BiDAFlowInferenceAdapter("sea_raft", device=device) if model.uses_bida else None
    compare_adapter = adapter if adapter is not None else BiDAFlowInferenceAdapter("sea_raft", device=device)
    a2_saved = torch.load(args.a2_checkpoint, map_location=device, weights_only=False)
    a2 = LearnedT1Refiner(a2_saved["variant"], tau_px=a2_saved["tau_px"]).to(device); a2.load_state_dict(a2_saved["model"]); a2.eval()
    dataset = TemporalClipDataset(args.backbones, args.validation_sequences, clip_length=args.clip_length,
                                  max_clips_per_sequence=args.max_validation_clips_per_sequence,
                                  coverage_threshold=args.coverage_threshold)
    data = loader(dataset, args, shuffle=False); rows = []; latencies = []; state_norms = []; peak_state_bytes = 0
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    for cpu in data:
        batch = to_device(cpu, device); state = None; previous_state_pred = None
        for step in range(batch["raw"].shape[1]):
            frame = time_slice(batch, step); evidence, flow_ms = evidence_for_frame(adapter, frame)
            if not model.uses_bida:
                # Paired comparisons still use the canonical BiDA common mask/A2 evidence.
                compare_evidence, flow_ms = evidence_for_frame(compare_adapter, frame)
            else: compare_evidence = evidence
            if device.type == "cuda": torch.cuda.synchronize(device)
            start = time.perf_counter(); output = model(frame["raw"], evidence, state,
                sequence_ids=list(batch["sequence"]), frame_indices=frame["current_index"])
            if device.type == "cuda": torch.cuda.synchronize(device)
            latencies.append((time.perf_counter()-start)*1000/frame["raw"].shape[0]); state = output.state
            state_norms.append(float(torch.stack([value.float().square().mean().sqrt() for value in state.tensors]).mean()))
            peak_state_bytes = max(peak_state_bytes, sum(value.numel()*value.element_size() for value in state.tensors))
            a2_output = a2(frame["raw"], compare_evidence, frame["current_rgb"])
            memory = compare_evidence["aligned_past_disparity"]; raw_error_t=(frame["raw"]-frame["gt"]).abs(); mem_error_t=(memory-frame["gt"]).abs()
            predictions = {"raw": frame["raw"], "flow_warp_blend_0.25": .75*frame["raw"]+.25*memory,
                           "learned_bida_t1_A2": a2_output.disparity, "latent_state": output.disparity,
                           "t1_oracle": torch.where(mem_error_t < raw_error_t, memory, frame["raw"])}
            for index in range(frame["raw"].shape[0]):
                raw=frame["raw"][index,0].cpu().numpy(); gt=frame["gt"][index,0].cpu().numpy(); coverage=frame["gt_coverage"][index,0].cpu().numpy()
                common_base=frame["raw_valid"][index,0].cpu().numpy().astype(bool) & compare_evidence["aligned_validity"][index,0].cpu().numpy().astype(bool) & compare_evidence["warp_support"][index,0].cpu().numpy().astype(bool)
                for threshold in args.thresholds:
                    common=(coverage>threshold)&common_base; boundary=boundary_mask(gt,coverage>threshold)
                    base={"namespace":"cache-grid-from-cached-predictions","coverage_threshold":threshold,
                          "backbone":batch["backbone"][index],"sequence":batch["sequence"][index],
                          "frame_id":current_ids(batch,step)[index],"common_valid_ratio":float(common.mean())}
                    for name,prediction in predictions.items():
                        rows.append(base|{"method":name}|frame_metrics(raw,prediction[index,0].cpu().numpy(),gt,common,boundary))
    sequence_rows, aggregate, safety = aggregate_frame_rows(rows)
    runtime={"state_model_latency_ms_per_frame":float(np.mean(latencies)),"state_memory_bytes":peak_state_bytes,
             "state_memory_mb":peak_state_bytes/2**20,"parameter_count":sum(p.numel() for p in model.parameters()),
             "peak_gpu_memory_mb":torch.cuda.max_memory_allocated(device)/2**20 if device.type=="cuda" else 0,
             "mean_state_rms":float(np.mean(state_norms))}
    aggregate.update({"runtime":runtime,"variant":model.variant,"checkpoint":str(args.checkpoint),
                      "validated_ppmstereo_reference":"results/ppmstereo_validation/aggregate_summary.json"})
    args.output.mkdir(parents=True,exist_ok=True); write_csv(args.output/"frame_metrics.csv",rows); write_csv(args.output/"sequence_metrics.csv",sequence_rows)
    save_json(args.output/"aggregate_summary.json",aggregate); save_json(args.output/"safety_summary.json",safety); save_json(args.output/"runtime_summary.json",runtime)
    save_json(args.output/"split_manifest.json",saved["split_manifest"]); save_json(args.output/"config.json",serializable_args(args))
    (args.output/"run.log").write_text("COMMAND "+" ".join(sys.argv)+f"\nCOMPLETE frames={len(rows)}\n")
    return 0


def main() -> int:
    args=parse_args()
    if args.burn_in >= args.clip_length: raise ValueError("burn-in must be shorter than clip")
    if args.mode=="smoke": return train(args,smoke=True)
    if args.mode=="train": return train(args,smoke=False)
    if args.mode=="diagnose": return diagnose(args)
    return evaluate(args)


if __name__ == "__main__": raise SystemExit(main())
