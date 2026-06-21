from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from scripts.argos_paths import DATASET_DIR, RESULTS_DIR
from .config import ExperimentConfig, ModuleConfig, load_experiment_config
from .losses import compute_playground_loss, compute_playground_metrics
from .model import PlaygroundModel
from .modules import MOTION_ESTIMATORS, WARPERS
from .real_data import ScaredProgressiveSequenceLoader
from .types import FusionOutput, TemporalBatch


REAL_SMOKE_CONFIGS = {
    "fixed_ema": "fixed_ema.yaml",
    "flow_warped": "flow_warped.yaml",
    "flow_warped_convgru": "flow_warped_convgru.yaml",
    "dual_memory": "dual_memory.yaml",
    "dual_memory_teacher_distillation": "dual_memory_teacher_distillation.yaml",
    "uncertainty_guided_dual_memory": "uncertainty_guided_dual_memory.yaml",
}

SHORT_RACE_CONFIGS = {
    "dual_memory": "dual_memory.yaml",
    "dual_memory_teacher_distillation": "dual_memory_teacher_distillation.yaml",
    "uncertainty_guided_dual_memory": "uncertainty_guided_dual_memory.yaml",
}


def make_synthetic_batch(
    batch_size: int = 1,
    sequence_length: int = 5,
    height: int = 64,
    width: int = 96,
    device: torch.device | None = None,
) -> TemporalBatch:
    device = torch.device("cpu") if device is None else device
    rgb = torch.rand(batch_size, sequence_length, 3, height, width, device=device)
    yy = torch.linspace(0.0, 1.0, height, device=device).view(1, 1, 1, height, 1)
    xx = torch.linspace(0.0, 1.0, width, device=device).view(1, 1, 1, 1, width)
    tt = torch.linspace(0.0, 1.0, sequence_length, device=device).view(1, sequence_length, 1, 1, 1)
    base = 20.0 + 32.0 * xx + 8.0 * yy + 2.0 * torch.sin(tt * 6.28318)
    noise = torch.randn(batch_size, sequence_length, 1, height, width, device=device)
    s2m2_s = base + 0.8 * noise
    s2m2_l = base + 0.4 * noise + 0.2
    sav = base + 0.25 * torch.roll(noise, shifts=1, dims=-1)
    gt = base
    valid = torch.ones_like(gt)
    return TemporalBatch(
        rgb=rgb,
        s2m2_s_disp=s2m2_s.float(),
        s2m2_l_disp=s2m2_l.float(),
        sav_disp=sav.float(),
        gt_disp=gt.float(),
        gt_depth_mm=100.0 / torch.clamp(gt.float(), min=1.0),
        valid_mask=valid,
        sequence_ids=[f"synthetic_{i}" for i in range(batch_size)],
    )


def run_one_batch(
    config: ExperimentConfig,
    device: torch.device,
    backward: bool = True,
) -> dict[str, float | str | bool]:
    workflow = config.workflow
    batch = make_synthetic_batch(
        batch_size=workflow.batch_size,
        sequence_length=workflow.sequence_length,
        height=workflow.crop_height,
        width=workflow.crop_width,
        device=device,
    )
    model = PlaygroundModel(config).to(device)
    has_trainable_params = any(p.requires_grad for p in model.parameters())
    if backward and not has_trainable_params:
        batch.s2m2_s_disp.requires_grad_(True)
        batch.s2m2_l_disp.requires_grad_(True)
        batch.sav_disp.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) if has_trainable_params else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    output = model(batch)
    loss, loss_metrics = compute_playground_loss(config, output, batch)
    did_backward = False
    if backward and optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        did_backward = True
    elif backward and loss.requires_grad:
        loss.backward()
        did_backward = True
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    metrics = compute_playground_metrics(output, batch)
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return {
        "experiment_name": config.experiment_name,
        "fusion": config.fusion.name,
        "motion_estimator": config.motion_estimator.name,
        "warper": config.warper.name,
        "uncertainty": config.uncertainty.name,
        "semantic_prior": config.semantic_prior.name,
        "did_backward": did_backward,
        "trainable_params": float(trainable_params),
        "total_params": float(total_params),
        "runtime_ms": elapsed_ms,
        "peak_vram_mb": peak_vram_mb,
        **loss_metrics,
        **metrics,
    }


def run_config_smoke(config_path: Path, output_dir: Path, device: torch.device) -> dict[str, float | str | bool]:
    config = load_experiment_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = run_one_batch(config, device=device, backward=True)
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
    (output_dir / "metrics.csv").write_text(_single_row_csv(metrics))
    (output_dir / "report.md").write_text(_report_text(config, metrics))
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    ref_dir = output_dir / "reference_images"
    ref_dir.mkdir(exist_ok=True)
    _write_reference_image(config, ref_dir / "synthetic_reference.png", device)
    return metrics


def run_all_smokes(config_dir: Path, output_dir: Path, device: torch.device) -> list[dict[str, float | str | bool]]:
    rows = []
    for path in sorted(config_dir.glob("*.yaml")):
        rows.append(run_config_smoke(path, output_dir / path.stem, device))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.csv").write_text(_rows_csv(rows))
    (output_dir / "memory_runtime_report.csv").write_text(_rows_csv(rows))
    (output_dir / "report.md").write_text(_summary_report(rows))
    return rows


def run_real_gpu_smokes(
    config_dir: Path,
    output_dir: Path,
    sequence_id: str = "test_dataset_9_keyframe_1",
    sequence_length: int = 5,
    crop_height: int = 256,
    crop_width: int = 384,
    raft_iters: int = 6,
) -> list[dict[str, object]]:
    """Run one-batch real-data smoke tests on a CUDA device.

    This is intentionally short: one progressive SCARED sequence, one
    forward/backward per config, and no checkpoint writes.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("Real playground smoke is GPU-only. CUDA is not available.")
    device = torch.device("cuda")
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = ScaredProgressiveSequenceLoader()
    spec = loader.make_spec(
        sequence_id=sequence_id,
        length=sequence_length,
        crop_height=crop_height,
        crop_width=crop_width,
    )
    batch = loader.load(spec, device=device)
    rows: list[dict[str, object]] = []
    log_lines: list[str] = []
    for filename in REAL_SMOKE_CONFIGS.values():
        config = _force_real_modules(load_experiment_config(config_dir / filename), raft_iters)
        log_lines.append(f"Running {config.experiment_name}")
        row = _run_real_one(config, batch, output_dir)
        rows.append(row)
        log_lines.append(f"  runtime_ms={row['runtime_ms']:.1f} peak_vram_mb={row['peak_vram_mb']:.1f}")
    _write_rows_csv(output_dir / "real_gpu_smoke_report.csv", rows)
    _write_real_report(output_dir / "report.md", rows, sequence_id)
    (output_dir / "run.log").write_text("\n".join(log_lines) + "\n")
    return rows


def run_short_race(
    config_dir: Path,
    output_dir: Path,
    epochs: int = 5,
    updates_per_epoch: int = 6,
    sequence_length: int = 5,
    crop_height: int = 256,
    crop_width: int = 384,
    val_sequence_id: str = "test_dataset_9_keyframe_1",
    raft_iters: int = 6,
    seed: int = 7,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("Short race is GPU-only. CUDA is not available.")
    device = torch.device("cuda")
    output_dir.mkdir(parents=True, exist_ok=True)
    ref_dir = output_dir / "reference_images"
    ref_dir.mkdir(exist_ok=True)
    log_lines: list[str] = []
    rng = random.Random(seed)

    loader = ScaredProgressiveSequenceLoader()
    sequence_ids = loader.sequence_ids()
    if val_sequence_id not in sequence_ids:
        val_sequence_id = sequence_ids[-1]
    train_sequences = [seq for seq in sequence_ids if seq != val_sequence_id]
    log_lines.append(f"train_sequences={train_sequences}")
    log_lines.append(f"val_sequence={val_sequence_id}")

    shared_motion = MOTION_ESTIMATORS.build("raft_fb_consistency", iters=raft_iters, model_size="small").to(device)
    precomputed_warper = WARPERS.build("flow").to(device)
    val_length = len(loader.by_sequence[val_sequence_id])
    val_batch = loader.load_clip(val_sequence_id, 0, val_length, crop_height, crop_width, device)
    with torch.no_grad():
        val_batch.motion = shared_motion(val_batch)

    confidence_rows = _confidence_rows(val_batch, val_batch.motion)
    _write_rows_csv(output_dir / "confidence_validation.csv", confidence_rows)
    _write_confidence_images(val_batch, val_batch.motion, ref_dir)

    configs = {
        name: _force_short_race_modules(load_experiment_config(config_dir / filename))
        for name, filename in SHORT_RACE_CONFIGS.items()
    }
    fixed_ema_config = _force_short_race_modules(load_experiment_config(config_dir / "fixed_ema.yaml"))
    runtime_rows = _runtime_breakdown(
        configs={**{"fixed_ema": fixed_ema_config}, **configs},
        batch=loader.load_clip(val_sequence_id, 0, sequence_length, crop_height, crop_width, device),
        shared_motion=shared_motion,
        output_dir=output_dir,
    )
    _write_rows_csv(output_dir / "runtime_breakdown.csv", runtime_rows)
    runtime_by_model = {str(row["experiment_name"]): row for row in runtime_rows}

    models = {name: PlaygroundModel(cfg).to(device) for name, cfg in configs.items()}
    optimizers = {name: torch.optim.AdamW(models[name].parameters(), lr=1e-4) for name in models}
    bad_counts = {name: 0 for name in models}
    aborted: set[str] = set()
    metric_rows: list[dict[str, object]] = []

    for epoch in range(1, epochs + 1):
        plan = _make_training_plan(loader, train_sequences, updates_per_epoch, sequence_length, rng)
        for name, model in models.items():
            if name in aborted:
                continue
            model.train()
            train_summaries = []
            for seq_id, start in plan:
                batch = loader.load_clip(seq_id, start, sequence_length, crop_height, crop_width, device)
                with torch.no_grad():
                    batch.motion = shared_motion(batch)
                opt = optimizers[name]
                before = _parameter_vector(model)
                opt.zero_grad(set_to_none=True)
                output = model(batch)
                loss, loss_metrics = compute_playground_loss(configs[name], output, batch)
                if not torch.isfinite(loss):
                    aborted.add(name)
                    log_lines.append(f"ABORT {name}: non-finite loss at epoch {epoch}")
                    break
                loss.backward()
                grad_norm = _grad_norm(model)
                opt.step()
                update_norm = float(torch.linalg.vector_norm(_parameter_vector(model) - before).detach().cpu())
                train_summaries.append({"loss": float(loss.detach().cpu()), "grad_norm": grad_norm, "update_norm": update_norm, **loss_metrics})
                if grad_norm <= 1e-10 or update_norm <= 1e-12:
                    aborted.add(name)
                    log_lines.append(f"ABORT {name}: vanished gradients/updates at epoch {epoch}")
                    break
            if name in aborted:
                continue
            val_row = _validate_model(
                name=name,
                config=configs[name],
                model=model,
                batch=val_batch,
                motion=val_batch.motion,
                runtime_row=runtime_by_model.get(name, {}),
                epoch=epoch,
                train_summary=_mean_dicts(train_summaries),
            )
            metric_rows.append(val_row)
            if (
                float(val_row["fused_to_sav_mae"]) > float(val_row["raw_s2m2_s_to_sav_mae"])
                and float(val_row["fused_to_sav_mae"]) > float(val_row["fixed_ema_reference_to_sav_mae"])
            ):
                bad_counts[name] += 1
            else:
                bad_counts[name] = 0
            if _weights_collapsed(val_row) or bad_counts[name] >= 3:
                aborted.add(name)
                log_lines.append(f"ABORT {name}: collapse or worse-than-baselines at epoch {epoch}")

    final_outputs = {}
    for name, model in models.items():
        if name not in aborted:
            model.eval()
            with torch.no_grad():
                output = model(val_batch)
            final_outputs[name] = output
            _write_final_reference_grid(name, val_batch, output, precomputed_warper, ref_dir)

    _write_rows_csv(output_dir / "short_race_metrics.csv", metric_rows)
    _write_short_race_report(output_dir / "report.md", metric_rows, runtime_rows, confidence_rows, sorted(final_outputs))
    (output_dir / "run.log").write_text("\n".join(log_lines) + "\n")
    return {"metrics": metric_rows, "runtime": runtime_rows, "confidence": confidence_rows, "active_models": sorted(final_outputs)}


def run_gt_short_race(
    config_dir: Path,
    output_dir: Path,
    epochs: int = 5,
    updates_per_epoch: int = 6,
    sequence_length: int = 5,
    crop_height: int = 256,
    crop_width: int = 384,
    raft_iters: int = 6,
    seed: int = 11,
    min_valid_ratio: float = 0.2,
) -> dict[str, object]:
    """Train two playground fusions briefly, then validate on SCARED GT.

    Validation intentionally reuses the existing SCARED temporal-GT benchmark
    artifacts: rectified frames, calibration JSONs, GT disparity/depth/masks,
    cached S2M2-S@512 predictions, and cached StereoAnyVideo predictions. The
    reported motion-compensated temporal metric uses the same Farneback warp
    implementation as the previous fairness/EMA checks.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("GT short race is GPU-only. CUDA is not available.")
    device = torch.device("cuda")
    output_dir.mkdir(parents=True, exist_ok=True)
    ref_dir = output_dir / "reference_images"
    ref_dir.mkdir(exist_ok=True)
    run_log: list[str] = []
    rng = random.Random(seed)

    train_loader = ScaredProgressiveSequenceLoader()
    train_sequences = [seq for seq in train_loader.sequence_ids() if seq != "test_dataset_9_keyframe_3"]
    if not train_sequences:
        train_sequences = train_loader.sequence_ids()
    run_log.append(f"train_sequences={train_sequences}")
    run_log.append("gt_validation_sequence=test_dataset_9_keyframe_3")
    run_log.append("StereoAnyVideo is teacher/pseudo-target only, never GT.")
    run_log.append("Motion-compensated reporting metric: OpenCV Farneback, matching previous SCARED fairness checks.")

    gt_frames = _load_gt_frame_records(min_valid_ratio=min_valid_ratio)
    gt_batch = _load_gt_temporal_batch(gt_frames, device=device)
    raw_seq = gt_batch.s2m2_s_disp.detach().cpu().numpy()[0, :, 0]
    sav_seq = gt_batch.sav_disp.detach().cpu().numpy()[0, :, 0]
    ema_seq = _ema_np_sequence(list(raw_seq), alpha=0.5)
    s2m2_runtime_ms, s2m2_peak = _prediction_metadata("S2M2-S_512")
    sav_runtime_ms, sav_peak = _prediction_metadata("StereoAnyVideo_384x640")

    shared_motion = MOTION_ESTIMATORS.build("raft_fb_consistency", iters=raft_iters, model_size="small").to(device)
    t_motion = time.perf_counter()
    gt_motion = _farneback_motion_for_batch(gt_frames, device)
    gt_raft_runtime_ms = (time.perf_counter() - t_motion) * 1000.0 / max(int(gt_batch.rgb.shape[1]), 1)
    gt_batch.motion = gt_motion
    gt_motion_peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    run_log.append(f"gt_validation_motion=OpenCV Farneback full-res, matching benchmark metric")
    run_log.append(f"gt_motion_runtime_ms_per_frame={gt_raft_runtime_ms:.4f}")
    configs = {
        "dual_memory": _force_short_race_modules(load_experiment_config(config_dir / "dual_memory.yaml")),
        "uncertainty_guided_dual_memory": _force_short_race_modules(
            load_experiment_config(config_dir / "uncertainty_guided_dual_memory.yaml")
        ),
    }
    models = {name: PlaygroundModel(cfg).to(device) for name, cfg in configs.items()}
    optimizers = {name: torch.optim.AdamW(models[name].parameters(), lr=1e-4) for name in models}
    bad_counts = {name: 0 for name in models}
    aborted: set[str] = set()
    metric_rows: list[dict[str, object]] = []
    latest_outputs: dict[str, FusionOutput] = {}
    latest_sequences: dict[str, list[np.ndarray]] = {}
    latest_runtime: dict[str, dict[str, float]] = {}

    for epoch in range(1, epochs + 1):
        plan = _make_training_plan(train_loader, train_sequences, updates_per_epoch, sequence_length, rng)
        for name, model in models.items():
            if name in aborted:
                continue
            model.train()
            train_summaries: list[dict[str, float]] = []
            for seq_id, start in plan:
                batch = train_loader.load_clip(seq_id, start, sequence_length, crop_height, crop_width, device)
                with torch.no_grad():
                    batch.motion = shared_motion(batch)
                opt = optimizers[name]
                before = _parameter_vector(model)
                opt.zero_grad(set_to_none=True)
                output = model(batch)
                loss, loss_metrics = compute_playground_loss(configs[name], output, batch)
                if not torch.isfinite(loss):
                    aborted.add(name)
                    run_log.append(f"ABORT {name}: non-finite loss at epoch {epoch}")
                    break
                loss.backward()
                grad_norm = _grad_norm(model)
                opt.step()
                update_norm = float(torch.linalg.vector_norm(_parameter_vector(model) - before).detach().cpu())
                train_summaries.append({"loss": float(loss.detach().cpu()), "grad_norm": grad_norm, "update_norm": update_norm, **loss_metrics})
                if grad_norm <= 1e-10 or update_norm <= 1e-12:
                    aborted.add(name)
                    run_log.append(f"ABORT {name}: vanished gradients/updates at epoch {epoch}")
                    break
            if name in aborted:
                continue

            output, runtime_info = _infer_playground_full_sequence(
                model,
                gt_batch,
                precomputed_motion=gt_motion,
                raft_runtime_ms=gt_raft_runtime_ms,
                motion_peak_vram_mb=gt_motion_peak_mb,
            )
            pred_seq = [x for x in output.fused_disparity.detach().cpu().numpy()[0, :, 0]]
            row = _summary_for_prediction_sequence(
                method=name,
                sequence_id="test_dataset_9_keyframe_3",
                preds=pred_seq,
                gt_frames=gt_frames,
                runtime_ms=s2m2_runtime_ms + runtime_info["raft_runtime_ms"] + runtime_info["fusion_runtime_ms"],
                peak_vram_mb=max(float(s2m2_peak), runtime_info["peak_vram_mb"]),
                epoch=epoch,
                train_summary=_mean_dicts(train_summaries),
                extra={
                    "raft_runtime_ms": runtime_info["raft_runtime_ms"],
                    "fusion_runtime_ms": runtime_info["fusion_runtime_ms"],
                    "relative_depth_mae_vs_raw_pct": "",
                    "relative_depth_mae_vs_ema_pct": "",
                    "relative_motion_comp_vs_raw_pct": "",
                    "relative_motion_comp_vs_ema_pct": "",
                    "w_raw_mean": float(output.source_weights[:, :, 0:1].mean().detach().cpu()),
                    "w_short_mean": float(output.source_weights[:, :, 1:2].mean().detach().cpu()),
                    "w_long_mean": float(output.source_weights[:, :, 2:3].mean().detach().cpu()),
                    "uncertainty_mean": float(output.uncertainty_map.mean().detach().cpu()),
                    "residual_abs_mean": float(torch.abs(output.residual_map).mean().detach().cpu()),
                    "finite_outputs": bool(torch.isfinite(output.fused_disparity).all().item()),
                },
            )
            metric_rows.append(row)
            latest_outputs[name] = output
            latest_sequences[name] = pred_seq
            latest_runtime[name] = runtime_info
            if _weights_collapsed(row) or not row["finite_outputs"]:
                aborted.add(name)
                run_log.append(f"ABORT {name}: collapse/non-finite outputs at epoch {epoch}")
                continue
            raw_base = _summary_for_prediction_sequence(
                "raw_s2m2_s", "test_dataset_9_keyframe_3", list(raw_seq), gt_frames, s2m2_runtime_ms, s2m2_peak
            )
            ema_base = _summary_for_prediction_sequence(
                "fixed_ema", "test_dataset_9_keyframe_3", ema_seq, gt_frames, s2m2_runtime_ms, s2m2_peak
            )
            worse_geometry = float(row["depth_mae_mm"]) > float(raw_base["depth_mae_mm"]) and float(row["depth_mae_mm"]) > float(ema_base["depth_mae_mm"])
            strong_degradation = float(row["depth_mae_mm"]) > min(float(raw_base["depth_mae_mm"]), float(ema_base["depth_mae_mm"])) + 0.25
            if worse_geometry or strong_degradation:
                bad_counts[name] += 1
            else:
                bad_counts[name] = 0
            if bad_counts[name] >= 3:
                aborted.add(name)
                run_log.append(f"ABORT {name}: worse geometry than raw/EMA for 3 validations at epoch {epoch}")

    baseline_rows = [
        _summary_for_prediction_sequence("raw_s2m2_s", "test_dataset_9_keyframe_3", list(raw_seq), gt_frames, s2m2_runtime_ms, s2m2_peak),
        _summary_for_prediction_sequence("fixed_ema", "test_dataset_9_keyframe_3", ema_seq, gt_frames, s2m2_runtime_ms, s2m2_peak),
        _summary_for_prediction_sequence("stereoanyvideo", "test_dataset_9_keyframe_3", list(sav_seq), gt_frames, sav_runtime_ms, sav_peak),
    ]
    final_model_rows = [_latest_metric_row(metric_rows, name) for name in configs if _latest_metric_row(metric_rows, name)]
    final_rows = baseline_rows + final_model_rows
    _add_relative_changes(final_rows)

    _write_rows_csv(output_dir / "gt_short_race_metrics.csv", metric_rows + baseline_rows)
    _write_rows_csv(output_dir / "per_sequence_metrics.csv", final_rows)
    _write_rows_csv(output_dir / "runtime_summary.csv", _runtime_summary_rows(final_rows, latest_runtime))
    _write_gt_reference_images(gt_frames, gt_batch, list(raw_seq), ema_seq, list(sav_seq), latest_sequences, latest_outputs, ref_dir)
    _write_gt_short_race_report(output_dir / "report.md", final_rows, metric_rows, min_valid_ratio)
    (output_dir / "run.log").write_text("\n".join(run_log) + "\n")
    return {"final_rows": final_rows, "metric_rows": metric_rows, "output_dir": str(output_dir)}


def _force_real_modules(config: ExperimentConfig, raft_iters: int) -> ExperimentConfig:
    motion = ModuleConfig("raft_fb_consistency", {"iters": raft_iters, "model_size": "small"})
    warper = ModuleConfig("identity", {}) if config.fusion.name == "fixed_ema" else ModuleConfig("flow", {})
    return replace(config, motion_estimator=motion, warper=warper)


def _force_short_race_modules(config: ExperimentConfig) -> ExperimentConfig:
    return replace(config, motion_estimator=ModuleConfig("precomputed", {}), warper=ModuleConfig("flow", {}))


def _make_training_plan(
    loader: ScaredProgressiveSequenceLoader,
    train_sequences: list[str],
    updates: int,
    length: int,
    rng: random.Random,
) -> list[tuple[str, int]]:
    plan = []
    for _ in range(updates):
        seq = rng.choice(train_sequences)
        max_start = max(0, len(loader.by_sequence[seq]) - length)
        plan.append((seq, rng.randint(0, max_start)))
    return plan


def _parameter_vector(model: torch.nn.Module) -> torch.Tensor:
    params = [p.detach().flatten().float() for p in model.parameters() if p.requires_grad]
    if not params:
        return torch.empty(0)
    return torch.cat(params)


def _grad_norm(model: torch.nn.Module) -> float:
    grads = [p.grad.detach().flatten().float() for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    return float(torch.linalg.vector_norm(torch.cat(grads)).detach().cpu())


def _mean_dicts(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({k for row in rows for k in row})
    return {f"train_{key}": float(np.mean([row[key] for row in rows if key in row])) for key in keys}


def _runtime_breakdown(
    configs: dict[str, ExperimentConfig],
    batch: TemporalBatch,
    shared_motion: torch.nn.Module,
    output_dir: Path,
    warmup: int = 10,
    measured: int = 30,
) -> list[dict[str, object]]:
    device = batch.rgb.device
    models = {name: PlaygroundModel(cfg).to(device).eval() for name, cfg in configs.items()}
    motion_times = []
    with torch.no_grad():
        for idx in range(warmup + measured):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            motion = shared_motion(batch)
            torch.cuda.synchronize(device)
            if idx >= warmup:
                motion_times.append((time.perf_counter() - start) * 1000.0)
        batch.motion = motion
        order = list(models)
        fusion_times: dict[str, list[float]] = {name: [] for name in models}
        rng = random.Random(123)
        for _ in range(warmup):
            rng.shuffle(order)
            for name in order:
                _ = models[name](batch)
        for _ in range(measured):
            rng.shuffle(order)
            for name in order:
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                _ = models[name](batch)
                torch.cuda.synchronize(device)
                fusion_times[name].append((time.perf_counter() - start) * 1000.0)
    rows = []
    raft_median = _percentile(motion_times, 50)
    raft_p95 = _percentile(motion_times, 95)
    for name, times in fusion_times.items():
        fusion_median = _percentile(times, 50)
        fusion_p95 = _percentile(times, 95)
        rows.append(
            {
                "experiment_name": name,
                "raft_runtime_median_ms": raft_median,
                "raft_runtime_p95_ms": raft_p95,
                "fusion_runtime_median_ms": fusion_median,
                "fusion_runtime_p95_ms": fusion_p95,
                "total_runtime_median_ms": raft_median + fusion_median,
                "total_runtime_p95_ms": raft_p95 + fusion_p95,
                "warmup_iterations": warmup,
                "measured_iterations": measured,
                "execution_order": "randomized",
            }
        )
    (output_dir / "run.log").write_text(
        "Runtime validation loads shared RAFT and all fusion models once; model/CUDA init excluded.\n"
    )
    return rows


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def _validate_model(
    name: str,
    config: ExperimentConfig,
    model: torch.nn.Module,
    batch: TemporalBatch,
    motion: dict[str, torch.Tensor],
    runtime_row: dict[str, object],
    epoch: int,
    train_summary: dict[str, float],
) -> dict[str, object]:
    model.eval()
    with torch.no_grad():
        batch.motion = motion
        output = model(batch)
        loss, loss_metrics = compute_playground_loss(config, output, batch)
        metrics = compute_playground_metrics(output, batch)
        temporal = _motion_temporal_metrics(output.fused_disparity, motion)
    weights = output.source_weights.detach()
    uncertainty = output.uncertainty_map.detach()
    residual = output.residual_map.detach()
    row: dict[str, object] = {
        "epoch": epoch,
        "experiment_name": name,
        "loss": float(loss.detach().cpu()),
        "gt_disp_mae": "",
        "gt_depth_mae_mm": "",
        "bad_2mm_pct": "",
        "fused_raw_temporal_mae": temporal["raw_temporal_mae"],
        "motion_compensated_temporal_mae": temporal["motion_compensated_temporal_mae"],
        "peak_vram_mb": torch.cuda.max_memory_allocated(batch.rgb.device) / (1024**2),
        "w_raw_mean": float(weights[:, :, 0:1].mean().cpu()),
        "w_raw_std": float(weights[:, :, 0:1].std().cpu()),
        "w_short_mean": float(weights[:, :, 1:2].mean().cpu()),
        "w_short_std": float(weights[:, :, 1:2].std().cpu()),
        "w_long_mean": float(weights[:, :, 2:3].mean().cpu()),
        "w_long_std": float(weights[:, :, 2:3].std().cpu()),
        "uncertainty_mean": float(uncertainty.mean().cpu()),
        "uncertainty_std": float(uncertainty.std().cpu()),
        "residual_abs_mean": float(torch.abs(residual).mean().cpu()),
        "finite_outputs": bool(torch.isfinite(output.fused_disparity).all().item()),
        **loss_metrics,
        **metrics,
        **{k: v for k, v in runtime_row.items() if k != "experiment_name"},
        **train_summary,
    }
    return row


def _motion_temporal_metrics(disp: torch.Tensor, motion: dict[str, torch.Tensor]) -> dict[str, float]:
    if disp.shape[1] <= 1:
        return {"raw_temporal_mae": 0.0, "motion_compensated_temporal_mae": 0.0}
    flow = motion["flow"]
    valid = motion.get("valid", torch.ones_like(disp))
    raw = torch.mean(torch.abs(disp[:, 1:] - disp[:, :-1]))
    warped_prev = []
    warper = WARPERS.build("flow").to(disp.device)
    for i in range(1, disp.shape[1]):
        warped_prev.append(warper(disp[:, i - 1], flow[:, i]))
    warped = torch.stack(warped_prev, dim=1)
    conf = valid[:, 1:].clamp(0.0, 1.0)
    denom = conf.sum().clamp(min=1.0)
    motion_comp = (conf * torch.abs(disp[:, 1:] - warped)).sum() / denom
    return {
        "raw_temporal_mae": float(raw.detach().cpu()),
        "motion_compensated_temporal_mae": float(motion_comp.detach().cpu()),
    }


def _weights_collapsed(row: dict[str, object]) -> bool:
    means = [float(row.get("w_raw_mean", 0.0)), float(row.get("w_short_mean", 0.0)), float(row.get("w_long_mean", 0.0))]
    return max(means) > 0.985


def _confidence_rows(batch: TemporalBatch, motion: dict[str, torch.Tensor]) -> list[dict[str, object]]:
    conf = motion["valid"].detach().float().cpu().numpy().reshape(-1)
    fb = motion.get("fb_error", torch.zeros_like(motion["valid"])).detach().float().cpu().numpy().reshape(-1)
    flow = motion["magnitude"].detach().float().cpu().numpy().reshape(-1)
    occ = 1.0 - conf
    return [
        {
            "sequence_id": " ".join(batch.sequence_ids),
            "confidence_mean": float(np.mean(conf)),
            "confidence_std": float(np.std(conf)),
            "confidence_p5": float(np.percentile(conf, 5)),
            "confidence_p50": float(np.percentile(conf, 50)),
            "confidence_p95": float(np.percentile(conf, 95)),
            "confidence_frac_below_0_50": float(np.mean(conf < 0.50)),
            "confidence_frac_below_0_80": float(np.mean(conf < 0.80)),
            "confidence_frac_below_0_95": float(np.mean(conf < 0.95)),
            "occlusion_prevalence": float(np.mean(occ > 0.50)),
            "fb_error_mean_px": float(np.mean(fb)),
            "flow_magnitude_mean": float(np.mean(flow)),
            "flow_magnitude_p95": float(np.percentile(flow, 95)),
        }
    ]


@torch.no_grad()
def _write_confidence_images(batch: TemporalBatch, motion: dict[str, torch.Tensor], out_dir: Path) -> None:
    flow = motion["magnitude"][0, :, 0].detach().cpu().numpy()
    conf = motion["valid"][0, :, 0].detach().cpu().numpy()
    occ = 1.0 - conf
    candidates = {
        "normal_motion": int(np.argsort(np.mean(flow, axis=(1, 2)))[len(flow) // 2]),
        "strongest_motion": int(np.argmax(np.mean(flow, axis=(1, 2)))),
        "strongest_occlusion": int(np.argmax(np.mean(occ, axis=(1, 2)))),
    }
    for name, idx in candidates.items():
        rgb = (batch.rgb[0, idx].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)[..., ::-1]
        tiles = [
            ("RGB", rgb),
            ("flow", _colorize(flow[idx])),
            ("confidence", _colorize(conf[idx], 1.0)),
            ("occlusion", _colorize(occ[idx], 1.0)),
        ]
        rendered = []
        for label, img in tiles:
            tile = cv2.resize(img, (220, 150), interpolation=cv2.INTER_AREA)
            cv2.putText(tile, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            rendered.append(tile)
        cv2.imwrite(str(out_dir / f"confidence_{name}.png"), np.concatenate(rendered, axis=1))


@torch.no_grad()
def _write_final_reference_grid(
    name: str,
    batch: TemporalBatch,
    output: FusionOutput,
    warper: torch.nn.Module,
    out_dir: Path,
) -> None:
    raw = batch.s2m2_s_disp[0, -1, 0].detach().cpu().numpy()
    ema = _fixed_ema_tensor(batch.s2m2_s_disp)[0, -1, 0].detach().cpu().numpy()
    fused = output.fused_disparity[0, -1, 0].detach().cpu().numpy()
    teacher = batch.sav_disp[0, -1, 0].detach().cpu().numpy()
    weights = output.source_weights[0, -1].detach().cpu().numpy()
    confidence = output.diagnostics.get("flow_confidence")
    conf_np = confidence[0, -1, 0].detach().cpu().numpy() if torch.is_tensor(confidence) else np.ones_like(raw)
    residual = output.residual_map[0, -1, 0].detach().cpu().numpy()
    uncertainty = output.uncertainty_map[0, -1, 0].detach().cpu().numpy()
    rgb = (batch.rgb[0, -1].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)[..., ::-1]
    short = output.diagnostics.get("warped_memory")
    short_np = short[0, -1, 0].detach().cpu().numpy() if torch.is_tensor(short) else ema
    long_np = short_np
    motion_error = np.abs(fused - short_np)
    geom_error = np.abs(fused - teacher)
    tiles = [
        ("RGB", rgb),
        ("raw S2M2-S", _colorize(raw)),
        ("fixed EMA", _colorize(ema)),
        ("short memory", _colorize(short_np)),
        ("long memory", _colorize(long_np)),
        ("fused", _colorize(fused)),
        ("SAV teacher", _colorize(teacher)),
        ("w_raw", _colorize(weights[0], 1.0)),
        ("w_short", _colorize(weights[1], 1.0)),
        ("w_long", _colorize(weights[2], 1.0)),
        ("uncertainty", _colorize(uncertainty, 1.0)),
        ("confidence", _colorize(conf_np, 1.0)),
        ("residual", _colorize(np.abs(residual), 2.0)),
        ("geom err", _colorize(geom_error, 12.0)),
        ("motion err", _colorize(motion_error, 12.0)),
    ]
    rows = []
    for row_tiles in [tiles[:5], tiles[5:10], tiles[10:]]:
        rendered = []
        for label, img in row_tiles:
            tile = cv2.resize(img, (180, 120), interpolation=cv2.INTER_AREA)
            cv2.putText(tile, label, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            rendered.append(tile)
        rows.append(np.concatenate(rendered, axis=1))
    cv2.imwrite(str(out_dir / f"{name}_final.png"), np.concatenate(rows, axis=0))


def _write_short_race_report(
    path: Path,
    metrics: list[dict[str, object]],
    runtime: list[dict[str, object]],
    confidence: list[dict[str, object]],
    active_models: list[str],
) -> None:
    final = {}
    for row in metrics:
        final[str(row["experiment_name"])] = row
    ranked = sorted(
        final.values(),
        key=lambda r: (
            float(r.get("gt_depth_mae_mm") or np.inf),
            float(r["motion_compensated_temporal_mae"]),
            float(r["fused_to_sav_mae"]),
        ),
    )
    best_simple = "dual_memory" if "dual_memory" in final else (str(ranked[0]["experiment_name"]) if ranked else "")
    teacher_candidates = [name for name in ["dual_memory_teacher_distillation", "uncertainty_guided_dual_memory"] if name in final]
    best_teacher = min(
        teacher_candidates,
        key=lambda n: (float(final[n]["motion_compensated_temporal_mae"]), float(final[n]["fused_to_sav_mae"])),
        default="",
    )
    overnight = min(active_models, key=lambda n: float(final[n]["motion_compensated_temporal_mae"])) if active_models else ""
    lines = [
        "# Short Race Report",
        "",
        "StereoAnyVideo is used only as teacher/pseudo-target, never as GT.",
        "Current large_v3 cache rows have no GT, so selection falls back to motion-compensated temporal consistency and teacher proximity.",
        "",
        f"- best simple model: `{best_simple}`",
        f"- best teacher-distilled model: `{best_teacher}`",
        f"- overnight candidate: `{overnight}`",
        "",
        "## Final Epoch Metrics",
        "",
        "| Model | Fused->SAV MAE | Motion-comp temporal MAE | W raw | W short | W long | Residual |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranked:
        lines.append(
            f"| {row['experiment_name']} | {float(row['fused_to_sav_mae']):.4f} | "
            f"{float(row['motion_compensated_temporal_mae']):.4f} | {float(row['w_raw_mean']):.4f} | "
            f"{float(row['w_short_mean']):.4f} | {float(row['w_long_mean']):.4f} | {float(row['residual_abs_mean']):.4f} |"
        )
    lines += [
        "",
        "## Runtime Protocol",
        "",
        "Shared RAFT and all fusion models are loaded once. CUDA/model initialization is excluded. Runtime uses 10 warm-up iterations, 30 synchronized measured iterations, and randomized config execution order.",
        "",
        "## Confidence Validation",
        "",
    ]
    if confidence:
        c = confidence[0]
        lines.append(
            f"confidence mean={float(c['confidence_mean']):.4f}, std={float(c['confidence_std']):.4f}, "
            f"P5/P50/P95={float(c['confidence_p5']):.4f}/{float(c['confidence_p50']):.4f}/{float(c['confidence_p95']):.4f}, "
            f"occlusion prevalence={float(c['occlusion_prevalence']):.4f}."
        )
    lines += [
        "",
        "## Fixed EMA Runtime Diagnosis",
        "",
        "Fixed EMA can look slow if RAFT/model initialization is included in per-config timing. The fair breakdown separates shared RAFT runtime from fusion runtime; use `runtime_breakdown.csv` for decisions.",
    ]
    path.write_text("\n".join(lines) + "\n")


def _run_real_one(config: ExperimentConfig, batch: TemporalBatch, output_dir: Path) -> dict[str, object]:
    device = batch.rgb.device
    model = PlaygroundModel(config).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        batch.s2m2_s_disp.requires_grad_(True)
        batch.s2m2_l_disp.requires_grad_(True)
        batch.sav_disp.requires_grad_(True)
    optimizer = torch.optim.AdamW(params, lr=1e-4) if params else None
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    output = model(batch)
    loss, loss_metrics = compute_playground_loss(config, output, batch)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.sqrt(sum(torch.sum(p.grad.detach().float() ** 2) for p in params if p.grad is not None))
        nonzero_grad = bool(float(grad_norm.detach().cpu()) > 0.0)
        optimizer.step()
    else:
        loss.backward()
        grad_norm = batch.s2m2_s_disp.grad.detach().float().norm()
        nonzero_grad = bool(float(grad_norm.detach().cpu()) > 0.0)
    torch.cuda.synchronize(device)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    flow_magnitude = output.diagnostics.get("flow_magnitude")
    flow_confidence = output.diagnostics.get("flow_confidence")
    fb_error = output.diagnostics.get("fb_error")
    flow_magnitude_mean = float(flow_magnitude.mean().detach().cpu()) if torch.is_tensor(flow_magnitude) else 0.0
    flow_confidence_mean = float(flow_confidence.mean().detach().cpu()) if torch.is_tensor(flow_confidence) else 1.0
    fb_error_mean = float(fb_error[:, 1:].mean().detach().cpu()) if torch.is_tensor(fb_error) and fb_error.shape[1] > 1 else 0.0
    _write_real_reference_grid(config.experiment_name, batch, output, output_dir / "reference_images")
    return {
        "experiment_name": config.experiment_name,
        "fusion": config.fusion.name,
        "motion_estimator": config.motion_estimator.name,
        "warper": config.warper.name,
        "teacher_usage": "StereoAnyVideo teacher/pseudo-target, not GT",
        "runtime_ms": runtime_ms,
        "peak_vram_mb": peak_vram_mb,
        "shape_ok": tuple(output.fused_disparity.shape) == tuple(batch.s2m2_s_disp.shape),
        "finite_outputs": bool(torch.isfinite(output.fused_disparity).all().item()),
        "nonzero_gradients": nonzero_grad,
        "grad_norm": float(grad_norm.detach().cpu()),
        "sequence_ids": " ".join(batch.sequence_ids),
        "has_gt": batch.gt_disp is not None,
        "flow_magnitude_mean": flow_magnitude_mean,
        "flow_confidence_mean": flow_confidence_mean,
        "raft_fb_error_mean_px": fb_error_mean,
        **loss_metrics,
        **compute_playground_metrics(output, batch),
    }


@torch.no_grad()
def _write_real_reference_grid(name: str, batch: TemporalBatch, output: FusionOutput, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = batch.s2m2_s_disp[0, -1, 0].detach().cpu().numpy()
    ema = _fixed_ema_tensor(batch.s2m2_s_disp)[0, -1, 0].detach().cpu().numpy()
    fused = output.fused_disparity[0, -1, 0].detach().cpu().numpy()
    teacher = batch.sav_disp[0, -1, 0].detach().cpu().numpy()
    rgb = (batch.rgb[0, -1].detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)[..., ::-1]
    warped = output.diagnostics.get("warped_memory")
    warped_np = warped[0, -1, 0].detach().cpu().numpy() if torch.is_tensor(warped) else ema
    residual = output.residual_map[0, -1, 0].detach().cpu().numpy()
    uncertainty = output.uncertainty_map[0, -1, 0].detach().cpu().numpy()
    reset = output.reset_map[0, -1, 0].detach().cpu().numpy()
    weights = output.source_weights[0, -1].detach().cpu().numpy()
    alpha = output.alpha_map[0, -1, 0].detach().cpu().numpy()
    flow_mag = output.diagnostics.get("flow_magnitude")
    confidence = output.diagnostics.get("flow_confidence")
    flow_np = flow_mag[0, -1, 0].detach().cpu().numpy() if torch.is_tensor(flow_mag) else np.zeros_like(raw)
    conf_np = confidence[0, -1, 0].detach().cpu().numpy() if torch.is_tensor(confidence) else np.ones_like(raw)
    error = np.abs(fused - teacher)
    tiles = [
        ("RGB", rgb),
        ("raw S2M2-S", _colorize(raw)),
        ("fixed EMA", _colorize(ema)),
        ("warped prev", _colorize(warped_np)),
        ("fused", _colorize(fused)),
        ("SAV teacher", _colorize(teacher)),
        ("flow", _colorize(flow_np)),
        ("confidence", _colorize(conf_np, 1.0)),
        ("reset/occ", _colorize(reset, 1.0)),
        ("alpha/w_raw", _colorize(alpha if weights.shape[0] < 3 else weights[0], 1.0)),
        ("residual", _colorize(np.abs(residual), 2.0)),
        ("error", _colorize(error, 12.0)),
    ]
    rendered = []
    for label, img in tiles:
        tile = cv2.resize(img, (180, 120), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, label, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        rendered.append(tile)
    grid = np.concatenate([np.concatenate(rendered[:6], axis=1), np.concatenate(rendered[6:], axis=1)], axis=0)
    cv2.imwrite(str(out_dir / f"{name}.png"), grid)


def _fixed_ema_tensor(raw: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    frames = []
    previous = raw[:, 0]
    for i in range(raw.shape[1]):
        current = raw[:, i] if i == 0 else alpha * raw[:, i] + (1.0 - alpha) * previous
        frames.append(current)
        previous = current
    return torch.stack(frames, dim=1)


def _write_real_report(path: Path, rows: list[dict[str, object]], sequence_id: str) -> None:
    ready = [
        str(row["experiment_name"])
        for row in rows
        if row.get("finite_outputs") and row.get("shape_ok") and row.get("nonzero_gradients")
    ]
    lines = [
        "# Real GPU Smoke Report",
        "",
        f"Sequence: `{sequence_id}`",
        "",
        "StereoAnyVideo is used only as a teacher/pseudo-target and is never labelled as GT.",
        "",
        "## Real vs Placeholder Modules",
        "",
        "- real: cached S2M2-S@512, cached S2M2-L@736, cached StereoAnyVideo teacher, real SCARED progressive RGB sequence loader, flow warping, torchvision RAFT frozen optical flow, forward-backward confidence masks.",
        "- placeholder/remains interface-only: semantic masks unless provided by future instrument/specularity segmenters; GT metrics for this cache because `has_gt=False` in large_v3.",
        "",
        "## RAFT Warping",
        "",
        "RAFT stores backward flow current-to-previous. Warping samples previous disparity into current coordinates with `grid_sample`. Forward-backward confidence is computed from backward + warped forward closure.",
        "",
        "## Ready For Short Training",
        "",
    ]
    lines.extend(f"- `{name}`" for name in ready[:3])
    lines += [
        "",
        "## Metrics",
        "",
        "| Config | Runtime ms | Peak VRAM MB | finite | shape | grad | fused->SAV MAE | alpha mean | reset mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment_name']} | {float(row['runtime_ms']):.1f} | {float(row['peak_vram_mb']):.1f} | "
            f"{row['finite_outputs']} | {row['shape_ok']} | {row['nonzero_gradients']} | "
            f"{float(row['fused_to_sav_mae']):.4f} | {float(row['alpha_mean']):.4f} | {float(row['reset_mean']):.4f} |"
        )
    lines += [
        "",
        "## RAFT FB Consistency",
        "",
        "| Config | Flow mag mean | Confidence mean | FB error mean px |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment_name']} | {float(row['flow_magnitude_mean']):.4f} | "
            f"{float(row['flow_confidence_mean']):.4f} | {float(row['raft_fb_error_mean_px']):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def _load_gt_frame_records(min_valid_ratio: float) -> list[dict[str, object]]:
    sequence_root = DATASET_DIR / "SCARED/curated/temporal_gt/test_dataset_9_keyframe_3"
    metadata_csv = sequence_root / "metadata.csv"
    with metadata_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    records: list[dict[str, object]] = []
    for row in rows:
        if float(row["valid_pixel_ratio"]) < min_valid_ratio:
            continue
        records.append(
            {
                "sequence_id": row["sequence_id"],
                "frame_id": row["frame_id"],
                "left_path": DATASET_DIR.parent / row["left_path"],
                "gt_disp_path": DATASET_DIR.parent / row["disparity_float32_path"],
                "gt_depth_path": DATASET_DIR.parent / row["depth_float32_path"],
                "valid_mask_path": DATASET_DIR.parent / row["valid_mask_path"],
                "calibration_path": DATASET_DIR.parent / row["calibration_path"],
                "valid_pixel_ratio": float(row["valid_pixel_ratio"]),
            }
        )
    return records


def _load_gt_temporal_batch(frames: list[dict[str, object]], device: torch.device) -> TemporalBatch:
    pred_root = RESULTS_DIR / "03_temporal_refinement/evaluation/gt_temporal_test_dataset_9_keyframe_3/predictions"
    rgb_tensors, s_tensors, l_tensors, sav_tensors, gt_tensors, depth_tensors, mask_tensors = [], [], [], [], [], [], []
    for frame in frames:
        fid = str(frame["frame_id"])
        rgb = _read_rgb_np(Path(frame["left_path"]))
        rgb_tensors.append(torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1))
        s_tensors.append(torch.from_numpy(np.load(pred_root / "S2M2-S_512" / f"{fid}.npy").astype(np.float32)).unsqueeze(0))
        l_tensors.append(torch.from_numpy(np.load(pred_root / "S2M2-L_736" / f"{fid}.npy").astype(np.float32)).unsqueeze(0))
        sav_tensors.append(torch.from_numpy(np.load(pred_root / "StereoAnyVideo_384x640" / f"{fid}.npy").astype(np.float32)).unsqueeze(0))
        gt_tensors.append(torch.from_numpy(np.load(Path(frame["gt_disp_path"])).astype(np.float32)).unsqueeze(0))
        depth_tensors.append(torch.from_numpy(np.load(Path(frame["gt_depth_path"])).astype(np.float32)).unsqueeze(0))
        mask_tensors.append(torch.from_numpy(np.load(Path(frame["valid_mask_path"])).astype(np.float32)).unsqueeze(0))
    return TemporalBatch(
        rgb=torch.stack(rgb_tensors).unsqueeze(0).float().to(device),
        s2m2_s_disp=torch.stack(s_tensors).unsqueeze(0).float().to(device),
        s2m2_l_disp=torch.stack(l_tensors).unsqueeze(0).float().to(device),
        sav_disp=torch.stack(sav_tensors).unsqueeze(0).float().to(device),
        gt_disp=torch.stack(gt_tensors).unsqueeze(0).float().to(device),
        gt_depth_mm=torch.stack(depth_tensors).unsqueeze(0).float().to(device),
        valid_mask=torch.stack(mask_tensors).unsqueeze(0).float().to(device),
        sequence_ids=["test_dataset_9_keyframe_3"],
    )


def _read_rgb_np(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _prediction_metadata(prediction_name: str) -> tuple[float, float]:
    path = (
        RESULTS_DIR
        / "03_temporal_refinement/evaluation/gt_temporal_test_dataset_9_keyframe_3/predictions"
        / prediction_name
        / "metadata.json"
    )
    meta = json.loads(path.read_text())
    return float(meta.get("avg_runtime_ms", math.nan)), float(meta.get("peak_vram_mb", math.nan))


@torch.no_grad()
def _infer_playground_full_sequence(
    model: torch.nn.Module,
    batch: TemporalBatch,
    precomputed_motion: dict[str, torch.Tensor],
    raft_runtime_ms: float,
    motion_peak_vram_mb: float,
) -> tuple[FusionOutput, dict[str, float]]:
    device = batch.rgb.device
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    batch.motion = precomputed_motion
    model.eval()
    torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    output = model(batch)
    torch.cuda.synchronize(device)
    fusion_runtime = (time.perf_counter() - t1) * 1000.0 / max(int(batch.rgb.shape[1]), 1)
    peak = max(motion_peak_vram_mb, torch.cuda.max_memory_allocated(device) / (1024**2))
    return output, {"raft_runtime_ms": raft_runtime_ms, "fusion_runtime_ms": fusion_runtime, "peak_vram_mb": peak}


def _ema_np_sequence(seq: list[np.ndarray], alpha: float) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    previous: np.ndarray | None = None
    for current in seq:
        if previous is None:
            filtered = current.astype(np.float32).copy()
        else:
            filtered = alpha * current.astype(np.float32) + (1.0 - alpha) * previous
        out.append(filtered.astype(np.float32))
        previous = filtered
    return out


def _summary_for_prediction_sequence(
    method: str,
    sequence_id: str,
    preds: list[np.ndarray],
    gt_frames: list[dict[str, object]],
    runtime_ms: float,
    peak_vram_mb: float,
    epoch: int | str = "",
    train_summary: dict[str, float] | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    frame_rows = []
    raw_vals, mc_vals = [], []
    rgbs = [_read_rgb_np(Path(frame["left_path"])) for frame in gt_frames]
    flows = [None] + [_farneback_flow(rgbs[i - 1], rgbs[i]) for i in range(1, len(rgbs))]
    masks = []
    for pred, frame in zip(preds, gt_frames):
        gt_disp = np.load(Path(frame["gt_disp_path"])).astype(np.float32)
        gt_depth = np.load(Path(frame["gt_depth_path"])).astype(np.float32)
        valid_mask = np.load(Path(frame["valid_mask_path"])).astype(bool)
        valid = valid_mask & np.isfinite(pred) & np.isfinite(gt_disp) & np.isfinite(gt_depth) & (pred > 0.1) & (gt_disp > 0) & (gt_depth > 0)
        masks.append(valid)
        if valid.any():
            disp_err = np.abs(pred[valid] - gt_disp[valid])
            pred_depth = _depth_from_disp(pred, Path(frame["calibration_path"]))
            depth_err = np.abs(pred_depth[valid] - gt_depth[valid])
            frame_rows.append(
                {
                    "disp_mae": float(np.mean(disp_err)),
                    "depth_mae": float(np.mean(depth_err)),
                    "bad_2mm": float(np.mean(depth_err > 2.0) * 100.0),
                    "gt_valid_pixels": int(valid_mask.sum()),
                    "evaluated_pixels": int(valid.sum()),
                    "coverage_pct": float(valid.sum() / max(int(valid_mask.sum()), 1) * 100.0),
                }
            )
    for i in range(1, len(preds)):
        raw_valid = masks[i - 1] & masks[i]
        if raw_valid.any():
            raw_vals.append(float(np.mean(np.abs(preds[i][raw_valid] - preds[i - 1][raw_valid]))))
        warped_prev = _warp_prev_to_current(preds[i - 1], flows[i])
        mc_valid = masks[i] & np.isfinite(warped_prev) & (warped_prev > 0.1)
        if mc_valid.any():
            mc_vals.append(float(np.mean(np.abs(preds[i][mc_valid] - warped_prev[mc_valid]))))
    row: dict[str, object] = {
        "method": method,
        "sequence_id": sequence_id,
        "epoch": epoch,
        "frames": len(gt_frames),
        "depth_mae_mm": _safe_mean([r["depth_mae"] for r in frame_rows]),
        "disp_mae_px": _safe_mean([r["disp_mae"] for r in frame_rows]),
        "bad_2mm_pct": _safe_mean([r["bad_2mm"] for r in frame_rows]),
        "raw_temporal_mae": _safe_mean(raw_vals),
        "motion_compensated_temporal_mae": _safe_mean(mc_vals),
        "gt_valid_pixels": int(sum(r["gt_valid_pixels"] for r in frame_rows)),
        "evaluated_pixels": int(sum(r["evaluated_pixels"] for r in frame_rows)),
        "valid_coverage_pct": _safe_mean([r["coverage_pct"] for r in frame_rows]),
        "runtime_ms": runtime_ms,
        "peak_vram_mb": peak_vram_mb,
        "motion_metric": "OpenCV Farneback previous-disparity warp, same as previous SCARED fairness checks",
        "positive_disparity_policy": "pred_disp > 0.1 and GT-valid mask",
    }
    if train_summary:
        row.update(train_summary)
    if extra:
        row.update(extra)
    return row


def _safe_mean(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(finite)) if finite else math.nan


def _farneback_flow(prev_rgb: np.ndarray, cur_rgb: np.ndarray) -> np.ndarray:
    prev_gray = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
    cur_gray = cv2.cvtColor(cur_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(
        prev_gray,
        cur_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=25,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    ).astype(np.float32)


def _farneback_motion_for_batch(frames: list[dict[str, object]], device: torch.device) -> dict[str, torch.Tensor]:
    rgbs = [_read_rgb_np(Path(frame["left_path"])) for frame in frames]
    t = len(rgbs)
    h, w = rgbs[0].shape[:2]
    flow = np.zeros((1, t, 2, h, w), dtype=np.float32)
    magnitude = np.zeros((1, t, 1, h, w), dtype=np.float32)
    valid = np.ones((1, t, 1, h, w), dtype=np.float32)
    fb_error = np.zeros((1, t, 1, h, w), dtype=np.float32)
    for i in range(1, t):
        # Store backward flow current -> previous, matching the playground warper convention.
        backward = _farneback_flow(rgbs[i], rgbs[i - 1])
        flow[0, i] = np.moveaxis(backward, -1, 0)
        magnitude[0, i, 0] = np.linalg.norm(backward, axis=-1)
    return {
        "flow": torch.from_numpy(flow).to(device),
        "valid": torch.from_numpy(valid).to(device),
        "magnitude": torch.from_numpy(magnitude).to(device),
        "fb_error": torch.from_numpy(fb_error).to(device),
    }


def _warp_prev_to_current(prev: np.ndarray, flow_prev_to_cur: np.ndarray | None) -> np.ndarray:
    if flow_prev_to_cur is None:
        return prev.astype(np.float32)
    h, w = prev.shape
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = xx - flow_prev_to_cur[..., 0]
    map_y = yy - flow_prev_to_cur[..., 1]
    return cv2.remap(prev.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)


def _depth_from_disp(disp: np.ndarray, calibration_path: Path) -> np.ndarray:
    calib = json.loads(calibration_path.read_text())
    if "fx" in calib and "baseline_mm" in calib:
        fx, baseline = float(calib["fx"]), float(calib["baseline_mm"])
    else:
        p1 = np.array(calib["P1"]["data"], dtype=np.float64).reshape(calib["P1"]["rows"], calib["P1"]["cols"])
        p2 = np.array(calib["P2"]["data"], dtype=np.float64).reshape(calib["P2"]["rows"], calib["P2"]["cols"])
        fx = float(p1[0, 0])
        baseline = float(abs(p2[0, 3] / p2[0, 0]))
    return fx * baseline / np.maximum(disp.astype(np.float32), 1e-6)


def _latest_metric_row(rows: list[dict[str, object]], name: str) -> dict[str, object] | None:
    matches = [row for row in rows if row.get("method") == name]
    if not matches:
        matches = [row for row in rows if row.get("experiment_name") == name]
    if not matches:
        return None
    return sorted(matches, key=lambda row: int(row.get("epoch") or 0))[-1]


def _add_relative_changes(rows: list[dict[str, object]]) -> None:
    by_method = {str(row["method"]): row for row in rows}
    raw = by_method.get("raw_s2m2_s")
    ema = by_method.get("fixed_ema")
    if raw is None or ema is None:
        return
    for row in rows:
        for base_name, base in [("raw", raw), ("ema", ema)]:
            depth_base = float(base["depth_mae_mm"])
            motion_base = float(base["motion_compensated_temporal_mae"])
            row[f"relative_depth_mae_vs_{base_name}_pct"] = 100.0 * (float(row["depth_mae_mm"]) - depth_base) / max(depth_base, 1e-9)
            row[f"relative_motion_comp_vs_{base_name}_pct"] = 100.0 * (
                float(row["motion_compensated_temporal_mae"]) - motion_base
            ) / max(motion_base, 1e-9)


def _runtime_summary_rows(final_rows: list[dict[str, object]], latest_runtime: dict[str, dict[str, float]]) -> list[dict[str, object]]:
    rows = []
    for row in final_rows:
        name = str(row["method"])
        runtime = latest_runtime.get(name, {})
        rows.append(
            {
                "method": name,
                "runtime_ms": row.get("runtime_ms", ""),
                "peak_vram_mb": row.get("peak_vram_mb", ""),
                "validation_motion_runtime_ms": runtime.get("raft_runtime_ms", ""),
                "fusion_runtime_ms": runtime.get("fusion_runtime_ms", ""),
                "notes": "learned fusion runtime is S2M2-S + full-res Farneback validation motion + fusion; raw/SAV use cached benchmark metadata",
            }
        )
    return rows


@torch.no_grad()
def _write_gt_reference_images(
    frames: list[dict[str, object]],
    batch: TemporalBatch,
    raw_seq: list[np.ndarray],
    ema_seq: list[np.ndarray],
    sav_seq: list[np.ndarray],
    learned: dict[str, list[np.ndarray]],
    outputs: dict[str, FusionOutput],
    out_dir: Path,
) -> None:
    if not learned:
        return
    dual = learned.get("dual_memory") or next(iter(learned.values()))
    uncertainty_model = learned.get("uncertainty_guided_dual_memory") or dual
    motion_scores = [0.0]
    for i in range(1, len(raw_seq)):
        motion_scores.append(float(np.mean(np.abs(raw_seq[i] - raw_seq[i - 1]))))
    geom_scores = []
    for pred, frame in zip(dual, frames):
        gt = np.load(Path(frame["gt_disp_path"])).astype(np.float32)
        valid = np.load(Path(frame["valid_mask_path"])).astype(bool) & (gt > 0) & np.isfinite(pred) & (pred > 0.1)
        geom_scores.append(float(np.mean(np.abs(pred[valid] - gt[valid]))) if valid.any() else 0.0)
    picks = {
        "normal_sequence": len(frames) // 2,
        "strongest_motion": int(np.argmax(motion_scores)),
        "worst_geometric_case": int(np.argmax(geom_scores)),
    }
    for label, idx in picks.items():
        idx = max(0, min(idx, len(frames) - 1))
        frame = frames[idx]
        rgb = cv2.cvtColor(_read_rgb_np(Path(frame["left_path"])), cv2.COLOR_RGB2BGR)
        gt = np.load(Path(frame["gt_disp_path"])).astype(np.float32)
        gt_depth = np.load(Path(frame["gt_depth_path"])).astype(np.float32)
        valid = np.load(Path(frame["valid_mask_path"])).astype(bool)
        output = outputs.get("dual_memory") or next(iter(outputs.values()))
        weights = output.source_weights[0, idx].detach().cpu().numpy()
        uncertainty = output.uncertainty_map[0, idx, 0].detach().cpu().numpy()
        short = output.diagnostics.get("warped_short_memory")
        long = output.diagnostics.get("warped_long_memory")
        short_np = short[0, idx, 0].detach().cpu().numpy() if torch.is_tensor(short) else ema_seq[idx]
        long_np = long[0, idx, 0].detach().cpu().numpy() if torch.is_tensor(long) else ema_seq[idx]
        pred_depth = _depth_from_disp(dual[idx], Path(frame["calibration_path"]))
        geom_error = np.where(valid, np.abs(pred_depth - gt_depth), np.nan)
        motion_error = np.zeros_like(gt, dtype=np.float32)
        if idx > 0:
            flow = _farneback_flow(_read_rgb_np(Path(frames[idx - 1]["left_path"])), _read_rgb_np(Path(frame["left_path"])))
            warped = _warp_prev_to_current(dual[idx - 1], flow)
            motion_error = np.abs(dual[idx] - warped)
        tiles = [
            ("RGB", rgb),
            ("GT disp", _colorize(gt)),
            ("raw S2M2-S", _colorize(raw_seq[idx])),
            ("fixed EMA", _colorize(ema_seq[idx])),
            ("dual memory", _colorize(dual[idx])),
            ("uncertainty dual", _colorize(uncertainty_model[idx])),
            ("SAV teacher", _colorize(sav_seq[idx])),
            ("short memory", _colorize(short_np)),
            ("long memory", _colorize(long_np)),
            ("w short", _colorize(weights[1], 1.0)),
            ("w long", _colorize(weights[2], 1.0)),
            ("uncertainty", _colorize(uncertainty, 1.0)),
            ("geom err mm", _colorize(geom_error, 8.0)),
            ("motion err px", _colorize(motion_error, 4.0)),
        ]
        rendered = []
        for tile_label, image in tiles:
            tile = cv2.resize(image, (190, 145), interpolation=cv2.INTER_AREA)
            cv2.rectangle(tile, (0, 0), (tile.shape[1], 24), (0, 0, 0), -1)
            cv2.putText(tile, tile_label, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            rendered.append(tile)
        rows = [np.concatenate(rendered[:7], axis=1), np.concatenate(rendered[7:], axis=1)]
        cv2.imwrite(str(out_dir / f"{label}.png"), np.concatenate(rows, axis=0))


def _write_gt_short_race_report(path: Path, final_rows: list[dict[str, object]], metric_rows: list[dict[str, object]], min_valid_ratio: float) -> None:
    ranked = sorted(final_rows, key=lambda row: (float(row["depth_mae_mm"]), float(row["motion_compensated_temporal_mae"])))
    learned = [row for row in final_rows if row["method"] in {"dual_memory", "uncertainty_guided_dual_memory"}]
    raw_or_ema = [row for row in final_rows if row["method"] in {"raw_s2m2_s", "fixed_ema"}]
    best_baseline_depth = min(float(row["depth_mae_mm"]) for row in raw_or_ema)
    best_baseline_motion = min(float(row["motion_compensated_temporal_mae"]) for row in raw_or_ema)
    worthy = [
        row
        for row in learned
        if float(row["depth_mae_mm"]) <= best_baseline_depth
        and float(row["motion_compensated_temporal_mae"]) < best_baseline_motion
    ]
    lines = [
        "# GT Short Race Report",
        "",
        "- temporal metric directly comparable to previous SCARED benchmark: yes",
        "- same sequence: `test_dataset_9_keyframe_3`",
        f"- same GT frame filter: valid-pixel ratio >= `{min_valid_ratio}`",
        "- same positive-disparity policy: `pred_disp > 0.1` inside GT-valid mask",
        "- same motion-compensation implementation: OpenCV Farneback previous-disparity warp",
        "- learned fusion validation also uses full-resolution Farneback motion so the temporal values are directly comparable to the previous benchmark",
        "- StereoAnyVideo is teacher/upper bound only, never GT",
        "",
        "## Final GT Ranking",
        "",
        "| method | depth MAE mm | disp MAE px | Bad-2mm % | raw temporal px | motion-comp px | coverage % | runtime ms | VRAM MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranked:
        lines.append(
            f"| {row['method']} | {float(row['depth_mae_mm']):.4f} | {float(row['disp_mae_px']):.4f} | "
            f"{float(row['bad_2mm_pct']):.4f} | {float(row['raw_temporal_mae']):.4f} | "
            f"{float(row['motion_compensated_temporal_mae']):.4f} | {float(row['valid_coverage_pct']):.2f} | "
            f"{float(row['runtime_ms']):.2f} | {float(row['peak_vram_mb']):.1f} |"
        )
    best_light = min([row for row in final_rows if row["method"] in {"fixed_ema", "dual_memory", "uncertainty_guided_dual_memory"}], key=lambda row: (float(row["depth_mae_mm"]), float(row["motion_compensated_temporal_mae"])))
    lines += [
        "",
        f"- GT-based best lightweight causal model: `{best_light['method']}`",
        f"- overnight-worthy playground model: `{worthy[0]['method'] if worthy else 'none'}`",
        "",
        "## Training Note",
        "",
        "The two learned fusions were trained for exactly 5 epochs from the real progressive cache. Validation was full-sequence causal on the GT protocol sequence, with state reset at the sequence boundary.",
        f"Validation rows written: `{len(metric_rows)}` learned epoch rows.",
    ]
    path.write_text("\n".join(lines) + "\n")


def _write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _single_row_csv(row: dict[str, object]) -> str:
    return _rows_csv([row])


def _rows_csv(rows: list[dict[str, object]]) -> str:
    if not rows:
        return ""
    keys = sorted({k for row in rows for k in row})
    import io

    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _report_text(config: ExperimentConfig, metrics: dict[str, object]) -> str:
    lines = [
        f"# {config.experiment_name}",
        "",
        "One-batch synthetic smoke test for the modular temporal-refinement playground.",
        "",
        "## Modules",
        "",
        f"- stereo source: `{config.stereo_source.name}`",
        f"- motion estimator: `{config.motion_estimator.name}`",
        f"- warper: `{config.warper.name}`",
        f"- uncertainty: `{config.uncertainty.name}`",
        f"- teacher: `{config.teacher.name}`",
        f"- semantic prior: `{config.semantic_prior.name}`",
        f"- fusion: `{config.fusion.name}`",
        "",
        "## Metrics",
        "",
    ]
    for key in sorted(metrics):
        lines.append(f"- `{key}`: {metrics[key]}")
    return "\n".join(lines) + "\n"


def _summary_report(rows: list[dict[str, object]]) -> str:
    lines = [
        "# ARGOS Temporal Refinement Playground Smoke Report",
        "",
        "All listed configs were assembled from YAML, executed for one forward pass, and given a backward smoke step. Trainable configs also execute an optimizer step.",
        "",
        "| Experiment | Fusion | Backward | Runtime ms | Peak VRAM MB | Params |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment_name']} | {row['fusion']} | {row['did_backward']} | "
            f"{float(row['runtime_ms']):.2f} | {float(row['peak_vram_mb']):.1f} | {int(float(row['trainable_params']))} |"
        )
    lines += [
        "",
        "Baseline comparison is always available in the metrics as raw S2M2-S, fixed EMA 0.50, and StereoAnyVideo teacher distance.",
        "Raw/fixed-EMA configs have no trainable parameters, so backward is checked against input tensors rather than an optimizer step.",
    ]
    return "\n".join(lines) + "\n"


@torch.no_grad()
def _write_reference_image(config: ExperimentConfig, path: Path, device: torch.device) -> None:
    batch = make_synthetic_batch(1, config.workflow.sequence_length, config.workflow.crop_height, config.workflow.crop_width, device)
    model = PlaygroundModel(config).to(device)
    output = model(batch)
    raw = batch.s2m2_s_disp[0, -1, 0].detach().cpu().numpy()
    sav = batch.sav_disp[0, -1, 0].detach().cpu().numpy()
    fused = output.fused_disparity[0, -1, 0].detach().cpu().numpy()
    uncertainty = output.uncertainty_map[0, -1, 0].detach().cpu().numpy()
    reset = output.reset_map[0, -1, 0].detach().cpu().numpy()
    residual = output.residual_map[0, -1, 0].detach().cpu().numpy()
    alpha = output.alpha_map[0, -1, 0].detach().cpu().numpy()
    weights = output.source_weights[0, -1].detach().cpu().numpy()
    rgb = (batch.rgb[0, -1].detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    fixed_ema = _fixed_ema_np(batch.s2m2_s_disp[0, :, 0].detach().cpu().numpy())[-1]
    temporal_error = np.abs(fused - sav)
    warped_memory = output.diagnostics.get("warped_memory")
    if torch.is_tensor(warped_memory):
        warped = warped_memory[0, -1, 0].detach().cpu().numpy()
    else:
        warped = fixed_ema
    tiles = [
        ("RGB", rgb[..., ::-1]),
        ("raw", _colorize(raw)),
        ("fixed EMA", _colorize(fixed_ema)),
        ("fused", _colorize(fused)),
        ("warped memory", _colorize(warped)),
        ("weights raw", _colorize(weights[0], 1.0)),
        ("uncertainty", _colorize(uncertainty, 1.0)),
        ("reset", _colorize(reset, 1.0)),
        ("residual", _colorize(np.abs(residual), 2.0)),
        ("temporal error", _colorize(temporal_error, 4.0)),
    ]
    small = []
    for label, img in tiles:
        tile = cv2.resize(img, (160, 110), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        small.append(tile)
    grid = np.concatenate([np.concatenate(small[:5], axis=1), np.concatenate(small[5:], axis=1)], axis=0)
    cv2.imwrite(str(path), grid)


def _fixed_ema_np(raw: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    frames = []
    previous = raw[0]
    for i in range(raw.shape[0]):
        current = raw[i] if i == 0 else alpha * raw[i] + (1.0 - alpha) * previous
        frames.append(current)
        previous = current
    return np.stack(frames)


def _colorize(x: np.ndarray, vmax: float | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros((*x.shape, 3), dtype=np.uint8)
    lo = 0.0 if vmax is not None else float(np.nanpercentile(x[finite], 1))
    hi = float(vmax) if vmax is not None else float(np.nanpercentile(x[finite], 99))
    if hi <= lo:
        hi = lo + 1.0
    y = (np.clip((x - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(y, cv2.COLORMAP_TURBO)
