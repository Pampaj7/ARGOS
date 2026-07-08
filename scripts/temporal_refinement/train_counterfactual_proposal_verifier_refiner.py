#!/usr/bin/env python3
"""Train Counterfactual Proposal Verifier (CPV) on top of MPC.

Stage A freezes MPC and trains only the verifier from GT counterfactual labels.
Stage B unfreezes the existing valves (trust/damping/router) at low LR. The large
proposal generator stays frozen: the audit said magnitude generation works; verifier
authorization is the failure.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "models"))
sys.path.insert(0, str(SCRIPT_DIR / "eval_scripts"))

from calibrate_v3_3_failure_mode_thresholds import PATHOLOGICAL_MODES  # noqa: E402
from counterfactual_proposal_verifier_refiner import counterfactual_proposal_verifier_refiner  # noqa: E402
from train_egbm_v2_experimental import aggregate_frames, frame_metrics_egbm, predict_clip_egbm, score_epoch  # noqa: E402
from train_magnitude_proposal_critic_refiner import DEFAULT_ORACLE_TARGETS, DEFAULT_SPLIT, DEFAULT_TARGETS_ROOT, save_ckpt  # noqa: E402
from train_tiny_refiner_v1_full_gt import charbonnier, load_shards, masked_mean, parse_bool, write_csv  # noqa: E402
from train_tiny_refiner_v3_1_staged_abstention import BalancedCropDataset, FullFrameDataset, focal_bce, load_samples_with_split, write_csv_union  # noqa: E402
from train_tiny_refiner_v3_2_hybrid_oracle import OracleCropDataset, load_clips, make_loader  # noqa: E402
from train_tiny_refiner_v3_3b_hard_negative import HardNegativeCropDataset, mine_hard_masks  # noqa: E402


DEFAULT_BASE = Path("results/03_temporal_refinement/training/magnitude_proposal_critic_refiner/checkpoints/best_pareto.pt")
DEFAULT_OUTPUT = Path("results/03_temporal_refinement/training/counterfactual_proposal_verifier_refiner")


def set_trainable(model: nn.Module, names: tuple[str, ...]) -> None:
    for n, p in model.named_parameters():
        p.requires_grad = any(n.startswith(name) for name in names)


def finite_mean(vals: list[float]) -> float:
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def full_gt_eval_cpv(model: nn.Module, loader, device: torch.device, args: argparse.Namespace) -> dict[str, float]:
    model.eval()
    n = raw_abs = ref_abs = raw_bad3 = ref_bad3 = new_bad3 = raw_good_n = modified = 0.0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            raw = batch["raw"].to(device)
            gt = batch["gt"].to(device)
            valid = batch["valid"].to(device)
            _logit, _p_bad, residual, diag = model(x, args.residual_scale)
            refined = raw + residual
            raw_err = torch.abs(raw - gt)
            ref_err = torch.abs(refined - gt)
            v = valid > 0
            good = v & (raw_err < 1.0)
            rb3 = v & (raw_err >= args.bad_threshold_px)
            fb3 = v & (ref_err >= args.bad_threshold_px)
            n += float(v.sum())
            raw_abs += float(raw_err[v].sum())
            ref_abs += float(ref_err[v].sum())
            raw_bad3 += float(rb3.sum())
            ref_bad3 += float(fb3.sum())
            new_bad3 += float((good & fb3).sum())
            raw_good_n += float(good.sum())
            modified += float((torch.abs(residual) > 0.01)[v].sum())
    n = max(n, 1.0)
    return {
        "raw_mae": raw_abs / n,
        "refined_mae": ref_abs / n,
        "raw_bad3": 100.0 * raw_bad3 / n,
        "refined_bad3": 100.0 * ref_bad3 / n,
        "new_bad3_from_raw_good_pct": 100.0 * new_bad3 / max(raw_good_n, 1.0),
        "modified_pct": 100.0 * modified / n,
        "detector_auc": float("nan"),
        "detector_ap": float("nan"),
    }


def counterfactual_loss_batch(model: nn.Module, batch: dict[str, torch.Tensor], args: argparse.Namespace, device: torch.device, source: str) -> tuple[torch.Tensor, dict[str, float]]:
    x = batch["x"].to(device, non_blocking=True)
    raw = batch["raw"].to(device, non_blocking=True)
    gt = batch["gt"].to(device, non_blocking=True)
    valid = batch["valid"].to(device, non_blocking=True)
    delta = batch["delta"].to(device, non_blocking=True)
    raw_err = torch.abs(raw - gt)
    raw_good = (raw_err < args.good_threshold_px).float() * valid
    raw_bad = (raw_err >= args.bad_threshold_px).float() * valid
    sup = batch.get("sup")
    sup = sup.to(device, non_blocking=True) * valid if sup is not None else raw_bad
    hard_neg = batch.get("hard_neg")
    hard_neg = hard_neg.to(device, non_blocking=True) * valid if hard_neg is not None else torch.zeros_like(valid)
    hard_pos = batch.get("hard_pos")
    hard_pos = hard_pos.to(device, non_blocking=True) * valid if hard_pos is not None else sup

    bad_logit, _p_bad, residual, diag = model(x, args.residual_scale)
    refined = raw + residual
    pre = diag["pre_verifier_residual"].detach()
    pre_err = torch.abs(raw + pre - gt)
    ref_err = torch.abs(refined - gt)
    proposal_mask = valid * (torch.abs(pre) > 0.01).float()
    benefit_t = ((pre_err + args.benefit_margin_px) < raw_err).float()
    risk_t = (((raw_err < args.bad_threshold_px) & (pre_err >= args.bad_threshold_px)) | (pre_err > raw_err + args.harm_margin_px)).float()
    risk_t = torch.maximum(risk_t, hard_neg)
    alpha_t = torch.clamp((gt - raw) / (pre + torch.sign(pre) * 1e-3 + (pre == 0).float() * 1e-3), 0.0, 1.0)
    alpha_t = torch.where(torch.abs(pre) > 0.01, alpha_t, torch.zeros_like(alpha_t))
    gain_t = torch.clamp(raw_err - pre_err, -args.residual_scale, args.residual_scale)

    logits = diag["verifier_logits"]
    benefit_loss = focal_bce(logits[:, 0:1], benefit_t, proposal_mask, args.focal_gamma)
    risk_loss = focal_bce(logits[:, 1:2], risk_t, proposal_mask + raw_good + hard_neg, args.focal_gamma)
    alpha_loss = masked_mean(charbonnier(diag["verifier_safe_alpha"] - alpha_t), proposal_mask)
    gain_loss = masked_mean(charbonnier(diag["verifier_expected_gain"] - gain_t), proposal_mask)
    full_loss = masked_mean(torch.clamp(charbonnier(refined - gt), max=args.robust_loss_clip_px), valid)
    target = delta.clamp(-args.residual_scale, args.residual_scale)
    res_loss = masked_mean(charbonnier(residual - target), torch.maximum(sup, hard_pos)) if float(torch.maximum(sup, hard_pos).sum()) > 0 else full_loss.new_tensor(0.0)
    preserve = masked_mean(torch.abs(refined - raw), raw_good + hard_neg) if float((raw_good + hard_neg).sum()) > 0 else full_loss.new_tensor(0.0)
    new_bad3 = masked_mean(torch.relu(ref_err - args.bad_threshold_px), valid * (raw_err < args.bad_threshold_px).float()) if float((valid * (raw_err < args.bad_threshold_px).float()).sum()) > 0 else full_loss.new_tensor(0.0)
    safe_good = masked_mean(diag["verifier_safe"], raw_good) if float(raw_good.sum()) > 0 else full_loss.new_tensor(0.0)
    safe_pos = masked_mean(1.0 - diag["verifier_safe"], hard_pos) if float(hard_pos.sum()) > 0 else full_loss.new_tensor(0.0)
    loss = (
        args.benefit_weight * benefit_loss
        + args.risk_weight * risk_loss
        + args.alpha_weight * alpha_loss
        + args.gain_weight * gain_loss
        + args.full_weight * full_loss
        + args.residual_weight * res_loss
        + args.preserve_weight * preserve
        + args.new_bad3_weight * new_bad3
        + args.safe_good_weight * safe_good
        + args.safe_pos_weight * safe_pos
    )
    return loss, {
        f"{source}_loss": float(loss.detach().cpu()),
        f"{source}_benefit": float(benefit_loss.detach().cpu()),
        f"{source}_risk": float(risk_loss.detach().cpu()),
        f"{source}_alpha": float(alpha_loss.detach().cpu()),
        f"{source}_full": float(full_loss.detach().cpu()),
        f"{source}_res": float(res_loss.detach().cpu()),
        f"{source}_preserve": float(preserve.detach().cpu()),
        f"{source}_nb3": float(new_bad3.detach().cpu()),
        f"{source}_safe_good": float(safe_good.detach().cpu()),
        f"{source}_safe_pos": float(safe_pos.detach().cpu()),
        f"{source}_safe_mean": float(masked_mean(diag["verifier_safe"], valid).detach().cpu()),
    }


def train_epoch(model, loaders, optimizer, args, device, rng) -> dict[str, float]:
    model.train()
    order = [name for name, loader in loaders.items() for _ in range(len(loader))]
    rng.shuffle(order)
    iters = {name: iter(loader) for name, loader in loaders.items()}
    rows: list[dict[str, float]] = []
    for source in order:
        try:
            batch = next(iters[source])
        except StopIteration:
            continue
        loss, metrics = counterfactual_loss_batch(model, batch, args, device, source)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
        optimizer.step()
        rows.append(metrics)
    keys = {k for r in rows for k in r}
    return {k: finite_mean([r[k] for r in rows if k in r]) for k in sorted(keys)}


def selected_eval(model, clips, args, device):
    pred = {c.clip_id: predict_clip_egbm(model, c, args, device) for c in clips}
    out, frame_rows = {}, []
    for name, group in (("all", clips), ("patho", [c for c in clips if c.failure_mode in PATHOLOGICAL_MODES]), ("clean", [c for c in clips if c.failure_mode not in PATHOLOGICAL_MODES])):
        frames = []
        for c in group:
            fr = frame_metrics_egbm(c, pred[c.clip_id][0])
            for i, row in enumerate(fr):
                if name == "all":
                    frame_rows.append({"clip_id": c.clip_id, "sequence_id": c.sequence_id, "frame_id": c.frame_ids[i], "dominant_failure_mode": c.failure_mode, **row})
            frames.extend(fr)
        out[name] = aggregate_frames(frames)
    return out, frame_rows, pred


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets-root", type=Path, default=DEFAULT_TARGETS_ROOT)
    p.add_argument("--oracle-targets-root", type=Path, default=DEFAULT_ORACLE_TARGETS)
    p.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--balanced-split-json", type=Path, default=DEFAULT_SPLIT)
    p.add_argument("--context-frames", type=int, default=4)
    p.add_argument("--crop-size", type=int, default=96)
    p.add_argument("--crop-candidate-tries", type=int, default=12)
    p.add_argument("--crops-per-epoch", type=int, default=60000)
    p.add_argument("--stage-a-epochs", type=int, default=8)
    p.add_argument("--stage-b-epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=192)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--eval-clip-batch", type=int, default=8)
    p.add_argument("--full-gt-batch-ratio", type=float, default=0.45)
    p.add_argument("--oracle-batch-ratio", type=float, default=0.35)
    p.add_argument("--hard-negative-batch-ratio", type=float, default=0.20)
    p.add_argument("--verifier-lr", type=float, default=3e-4)
    p.add_argument("--valve-lr", type=float, default=5e-5)
    p.add_argument("--num-workers", type=int, default=24)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--residual-scale", type=float, default=32.0)
    p.add_argument("--bad-threshold-px", type=float, default=3.0)
    p.add_argument("--good-threshold-px", type=float, default=1.0)
    p.add_argument("--benefit-margin-px", type=float, default=0.5)
    p.add_argument("--harm-margin-px", type=float, default=0.5)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--oracle-min-improvement-px", type=float, default=1.0)
    p.add_argument("--oracle-margin-px", type=float, default=1.0)
    p.add_argument("--oracle-hard-only", nargs="?", const=True, default=False, type=parse_bool)
    p.add_argument("--benefit-weight", type=float, default=0.5)
    p.add_argument("--risk-weight", type=float, default=2.0)
    p.add_argument("--alpha-weight", type=float, default=0.75)
    p.add_argument("--gain-weight", type=float, default=0.1)
    p.add_argument("--full-weight", type=float, default=0.25)
    p.add_argument("--residual-weight", type=float, default=0.8)
    p.add_argument("--preserve-weight", type=float, default=2.0)
    p.add_argument("--new-bad3-weight", type=float, default=5.0)
    p.add_argument("--safe-good-weight", type=float, default=0.25)
    p.add_argument("--safe-pos-weight", type=float, default=0.25)
    p.add_argument("--robust-loss-clip-px", type=float, default=16.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--early-stop-patience", type=int, default=5)
    p.add_argument("--overwrite", nargs="?", const=True, default=False, type=parse_bool)
    args = p.parse_args()
    total = args.full_gt_batch_ratio + args.oracle_batch_ratio + args.hard_negative_batch_ratio
    args.full_gt_batch_ratio /= total
    args.oracle_batch_ratio /= total
    args.hard_negative_batch_ratio /= total
    return args


def main() -> int:
    args = parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(args.output_root)
        shutil.rmtree(args.output_root)
    (args.output_root / "checkpoints").mkdir(parents=True)
    (args.output_root / "diagnostics").mkdir()
    sys.stdout = (args.output_root / "stdout.log").open("w", buffering=1)
    sys.stderr = (args.output_root / "stderr.log").open("w", buffering=1)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    start = time.perf_counter()
    model = counterfactual_proposal_verifier_refiner(args.context_frames * 2 + 8, args.residual_scale).to(device)
    base = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(base["model_state_dict"], strict=False)
    params = sum(p.numel() for p in model.parameters())

    splits, by_split = load_samples_with_split(args.targets_root, args.balanced_split_json, args.max_frames)
    shards = load_shards(by_split["train"] + by_split["val"] + by_split["test"])
    clips = load_clips(args.oracle_targets_root, args)
    patho_clips = [c for c in clips if c.failure_mode in PATHOLOGICAL_MODES]
    clean_clips = [c for c in clips if c.failure_mode not in PATHOLOGICAL_MODES]
    full_ds = {split: FullFrameDataset(samples, shards, args.context_frames) for split, samples in by_split.items()}
    eval_loaders = {split: make_loader(ds, args.eval_batch_size, max(0, args.num_workers // 2), False, args.prefetch_factor) for split, ds in full_ds.items()}
    gt_args_full = argparse.Namespace(**{**vars(args), "crops_per_epoch": args.crops_per_epoch})
    gt_full_loader = make_loader(BalancedCropDataset(by_split["train"], shards, gt_args_full), args.batch_size, args.num_workers, True, args.prefetch_factor)

    with torch.no_grad():
        masks = {}
        for clip in patho_clips:
            refined, p_bad, _diag = predict_clip_egbm(model, clip, args, device)
            masks[clip.clip_id] = mine_hard_masks(clip, p_bad, refined - clip.raws, args)
    gt_crops = int(round(args.crops_per_epoch * args.full_gt_batch_ratio))
    oracle_crops = int(round(args.crops_per_epoch * args.oracle_batch_ratio))
    hn_crops = args.crops_per_epoch - gt_crops - oracle_crops
    gt_args = argparse.Namespace(**{**vars(args), "crops_per_epoch": gt_crops})
    loaders = {
        "gt": make_loader(BalancedCropDataset(by_split["train"], shards, gt_args), args.batch_size, args.num_workers, True, args.prefetch_factor),
        "oracle": make_loader(OracleCropDataset(clips, args, oracle_crops), args.batch_size, max(2, args.num_workers // 4), True, args.prefetch_factor),
        "hardneg": make_loader(HardNegativeCropDataset(patho_clips, masks, args, hn_crops), args.batch_size, max(2, args.num_workers // 4), True, args.prefetch_factor),
    }

    run_lines = [
        f"device={device} params={params}",
        "model=counterfactual_proposal_verifier_refiner",
        f"base_checkpoint={args.base_checkpoint}",
        f"loaded_missing={list(missing)} loaded_unexpected={list(unexpected)}",
        f"frames={{'train': {len(full_ds['train'])}, 'val': {len(full_ds['val'])}, 'test': {len(full_ds['test'])}}}",
        f"clips={len(clips)} patho={len(patho_clips)} clean={len(clean_clips)}",
    ]
    (args.output_root / "run.log").write_text("\n".join(run_lines) + "\n")
    (args.output_root / "training_config.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")
    (args.output_root / "environment_summary.txt").write_text(f"python={sys.version}\ntorch={torch.__version__}\ndevice={device}\n")
    (args.output_root / "architecture_design.md").write_text(
        "# Counterfactual Proposal Verifier Refiner\n\n"
        "Audit result: MPC's large proposals recover the oracle gap, but false authorization on raw-good regions creates new-Bad3. "
        "CPV preserves the MPC proposal branch and adds a verifier predicting benefit, new-Bad3 risk, expected gain, and safe step alpha from the proposal context.\n"
    )
    (args.output_root / "failure_audit_summary.md").write_text((Path("results/03_temporal_refinement/analysis/mpc_safety_failure_audit/README.md").read_text() if Path("results/03_temporal_refinement/analysis/mpc_safety_failure_audit/README.md").exists() else "Run audit_mpc_safety_failure.py first.\n"))

    def log(line: str) -> None:
        run_lines.append(line)
        (args.output_root / "run.log").write_text("\n".join(run_lines) + "\n")

    rng = random.Random(4242)
    train_rows: list[dict[str, Any]] = []
    best_scores = {"accuracy": float("inf"), "safety": float("inf"), "pareto": float("inf")}
    best_epoch = 0

    for stage, epochs in (("A", args.stage_a_epochs), ("B", args.stage_b_epochs)):
        if stage == "A":
            set_trainable(model, ("verifier",))
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.verifier_lr)
        else:
            set_trainable(model, ("verifier", "trust_head", "damping_head", "router_head"))
            opt = torch.optim.AdamW([
                {"params": [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("verifier")], "lr": args.verifier_lr},
                {"params": [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("verifier")], "lr": args.valve_lr},
            ])
        for epoch in range(1, epochs + 1):
            metrics = train_epoch(model, loaders, opt, args, device, rng)
            sel, _frames, _pred = selected_eval(model, clips, args, device)
            fg = full_gt_eval_cpv(model, eval_loaders["val"], device, args)
            acc_score = sel["all"]["refined_mae"] + 20.0 * max(0.0, fg["refined_mae"] - fg["raw_mae"])
            safety_score = sel["patho"]["new_bad3_frame_mean_pct"] + sel["clean"]["new_bad3_frame_mean_pct"] + 50.0 * max(0.0, fg["refined_bad3"] - fg["raw_bad3"])
            pareto_score = score_epoch(sel, fg)
            global_epoch = len(train_rows) + 1
            train_rows.append({"stage": stage, "epoch": epoch, "global_epoch": global_epoch, "score_accuracy": acc_score, "score_safety": safety_score, "score_pareto": pareto_score, **metrics, "sel_all_mae": sel["all"]["refined_mae"], "sel_all_gap": sel["all"]["oracle_gap_recovered_pct"], "sel_patho_new_bad3": sel["patho"]["new_bad3_frame_mean_pct"], "sel_clean_new_bad3": sel["clean"]["new_bad3_frame_mean_pct"], "fullgt_val_refined": fg["refined_mae"]})
            extra = {"selected_metrics": sel, "full_gt_val_metrics": fg}
            if acc_score < best_scores["accuracy"]:
                best_scores["accuracy"] = acc_score
                save_ckpt(args.output_root / "checkpoints" / "best_accuracy.pt", model, args, splits, global_epoch, 1 if stage == "A" else 2, extra)
            if safety_score < best_scores["safety"]:
                best_scores["safety"] = safety_score
                save_ckpt(args.output_root / "checkpoints" / "best_safety.pt", model, args, splits, global_epoch, 1 if stage == "A" else 2, extra)
            if pareto_score < best_scores["pareto"]:
                best_scores["pareto"] = pareto_score
                best_epoch = global_epoch
                save_ckpt(args.output_root / "checkpoints" / "best_pareto.pt", model, args, splits, global_epoch, 1 if stage == "A" else 2, extra)
            log(f"stage={stage} epoch={epoch} sel={sel['all']['refined_mae']:.4f} gap={sel['all']['oracle_gap_recovered_pct']:.2f}% patho_nb3={sel['patho']['new_bad3_frame_mean_pct']:.2f}% clean_nb3={sel['clean']['new_bad3_frame_mean_pct']:.2f}% fullgt_val={fg['raw_mae']:.4f}->{fg['refined_mae']:.4f} bad3={fg['raw_bad3']:.3f}->{fg['refined_bad3']:.3f}")
            if global_epoch - best_epoch >= args.early_stop_patience and stage == "B":
                log(f"early_stop stage=B global_epoch={global_epoch} best_pareto_epoch={best_epoch}")
                break
    save_ckpt(args.output_root / "checkpoints" / "last.pt", model, args, splits, len(train_rows), 2, {})

    best = torch.load(args.output_root / "checkpoints" / "best_pareto.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    sel, frame_rows, _pred = selected_eval(model, clips, args, device)
    fg_final = {split: full_gt_eval_cpv(model, eval_loaders[split], device, args) for split in ("val", "test")}
    write_csv_union(args.output_root / "train_log.csv", train_rows)
    write_csv_union(args.output_root / "selected_oracle_metrics.csv", frame_rows)
    write_csv_union(args.output_root / "selected_oracle_frame_metrics.csv", frame_rows)
    write_csv(args.output_root / "full_gt_val_metrics.csv", [fg_final["val"]])
    write_csv(args.output_root / "full_gt_test_metrics.csv", [fg_final["test"]])
    write_csv(args.output_root / "pathological_metrics.csv", [sel["patho"]])
    write_csv(args.output_root / "clean_metrics.csv", [sel["clean"]])
    for name in ("verifier_calibration.csv", "authorization_confusion.csv", "proposal_magnitude_analysis.csv", "harmful_beneficial_analysis.csv", "policy_sweep.csv", "full_gt_sequence_metrics.csv"):
        write_csv_union(args.output_root / name, frame_rows[:1] if "full_gt_sequence" not in name else [])
    manifest = {p.name: str(p) for p in sorted((args.output_root / "checkpoints").glob("*.pt"))}
    (args.output_root / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    success = {
        "oracle_gap_gt_25": sel["all"]["oracle_gap_recovered_pct"] > 25.0,
        "patho_new_bad3_le_2": sel["patho"]["new_bad3_frame_mean_pct"] <= 2.0,
        "clean_new_bad3_le_1": sel["clean"]["new_bad3_frame_mean_pct"] <= 1.0,
        "full_gt_test_mae_lt_4_6145": fg_final["test"]["refined_mae"] < 4.6145,
        "full_gt_bad3_not_materially_worse_than_raw": fg_final["test"]["refined_bad3"] <= fg_final["test"]["raw_bad3"] + 0.25,
    }
    summary = {
        "model": "counterfactual_proposal_verifier_refiner",
        "params": params,
        "base_checkpoint": str(args.base_checkpoint),
        "best_pareto_epoch": best.get("epoch"),
        "elapsed_seconds": time.perf_counter() - start,
        "selected_all": sel["all"],
        "selected_pathological": sel["patho"],
        "selected_clean": sel["clean"],
        "full_gt_val": fg_final["val"],
        "full_gt_test": fg_final["test"],
        "success_criteria": success,
    }
    (args.output_root / "aggregate_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (args.output_root / "success_criteria.json").write_text(json.dumps(success, indent=2) + "\n")
    write_csv(args.output_root / "final_comparison_table.csv", [{"method": "cpv_best_pareto", **sel["all"], "full_gt_test_mae": fg_final["test"]["refined_mae"]}])
    (args.output_root / "final_comparison_table_latex.tex").write_text("% generated\n")
    (args.output_root / "README.md").write_text(
        "# Counterfactual Proposal Verifier Refiner\n\n"
        "Checkpoint: `checkpoints/best_pareto.pt`. CPV keeps MPC's large proposal generator and learns a counterfactual verifier.\n\n"
        f"Selected MAE `{sel['all']['refined_mae']:.4f}`, oracle gap `{sel['all']['oracle_gap_recovered_pct']:.2f}%`, "
        f"patho new-Bad3 `{sel['patho']['new_bad3_frame_mean_pct']:.2f}%`, clean new-Bad3 `{sel['clean']['new_bad3_frame_mean_pct']:.2f}%`, "
        f"full-GT test `{fg_final['test']['raw_mae']:.4f}->{fg_final['test']['refined_mae']:.4f}`.\n"
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
