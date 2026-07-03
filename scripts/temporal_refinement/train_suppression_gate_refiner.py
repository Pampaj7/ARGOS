#!/usr/bin/env python3
"""Train the Suppression-Only Gate (SOG) over a frozen v3.2c refiner.

Single stage: mixed batches (50% full-GT balanced crops, 25% oracle crops from the 4
clean clips, 25% hard-negative crops from the 2 pathological clips), one uniform loss —
all batch types carry raw/gt/valid/x, so the mix only shapes the sampling distribution:

  final = raw + s * hard_correction,  hard_correction = (p_bad >= 0.7) * residual_v32c
  L = w_acc  * clamped charbonnier(final - gt)            on valid
    + w_nb3  * relu(|final - gt| - 3)                     on raw-good(<3px) pixels
    + w_keep * (1 - s)                                    on pixels where v3.2c clearly helps

Only the ~90k gate parameters train; v3.2c stays frozen (its full-GT behavior is the
upper envelope by construction). No S2M2/SAV/RAFT/DINO inference. Single GPU, no
DataParallel, cudnn.benchmark off (EGBM crash lessons).
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
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "models"))
sys.path.insert(0, str(SCRIPT_DIR / "eval_scripts"))

from train_tiny_refiner_v1_full_gt import (  # noqa: E402
    DEFAULT_TARGETS_ROOT,
    charbonnier,
    finite_mean,
    load_shards,
    masked_mean,
    parse_bool,
    write_csv,
)
from train_tiny_refiner_v3_1_staged_abstention import (  # noqa: E402
    DEFAULT_BALANCED_SPLIT,
    AbstentionCropRefiner,
    BalancedCropDataset,
    FullFrameDataset,
    evaluate,
    load_samples_with_split,
    write_csv_union,
)
from train_tiny_refiner_v3_2_hybrid_oracle import (  # noqa: E402
    OracleCropDataset,
    load_clips,
    make_loader,
    row_at,
    selected_frame_rows,
)
from train_tiny_refiner_v3_3b_hard_negative import (  # noqa: E402
    HardNegativeCropDataset,
    eval_selected_groups,
    mine_hard_masks,
    score_epoch,
)
from calibrate_v3_3_failure_mode_thresholds import PATHOLOGICAL_MODES  # noqa: E402
from evaluate_v3_1_on_selected_oracle_clips import predict_clip  # noqa: E402
from suppression_gate import SuppressionGate, V32CWithSuppression  # noqa: E402


DEFAULT_BASE_CHECKPOINT = Path("results/03_temporal_refinement/training/tiny_refiner_v3_2c_hybrid_oracle_freeze_detector_long/checkpoints/best.pt")
DEFAULT_ORACLE_TARGETS = Path("results/03_temporal_refinement/evaluation/distillation_targets_selected_clips")
DEFAULT_OUTPUT_ROOT = Path("results/03_temporal_refinement/training/fable_light_experimental_refiner")
EVAL_THRESHOLD = 0.7
BASELINES = {
    "v3.2c": {"selected_mae": 11.0054, "gap_pct": 7.03, "patho_new_bad3": 15.77, "clean_new_bad3": 0.89, "full_gt_test_mae": 4.6145},
    "v4_tiny": {"selected_mae": 11.0669, "gap_pct": 5.67, "patho_new_bad3": 0.33, "full_gt_test_mae": 4.7763},
    "v3.3_threshold_only": {"selected_mae": 11.1062, "gap_pct": 4.80, "patho_new_bad3": 6.69},
}


def gate_loss_batch(model: V32CWithSuppression, batch: dict[str, torch.Tensor], args: argparse.Namespace, device: torch.device, source: str) -> tuple[torch.Tensor, dict[str, float]]:
    x = batch["x"].to(device, non_blocking=True)
    raw = batch["raw"].to(device, non_blocking=True)
    gt = batch["gt"].to(device, non_blocking=True)
    valid = batch["valid"].to(device, non_blocking=True)
    _logit, p_bad, residual, s = model.forward_with_s(x, args.residual_scale)
    hard = (p_bad >= EVAL_THRESHOLD).float()
    final = raw + hard * s * residual
    refined_v32c = raw + hard * residual  # frozen base's own correction (no grad through it)
    raw_err = torch.abs(raw - gt)
    final_err = torch.abs(final - gt)
    v32c_err = torch.abs(refined_v32c - gt)
    zero = s.sum() * 0.0
    acc = masked_mean(torch.clamp(charbonnier(final - gt), max=args.robust_loss_clip_px), valid)
    below3 = valid * (raw_err < args.bad_threshold_px).float()
    nb3 = masked_mean(torch.relu(final_err - args.bad_threshold_px), below3) if float(below3.sum()) > 0 else zero
    helps = valid * ((v32c_err + args.help_margin_px) < raw_err).float() * hard
    keep = masked_mean(1.0 - s, helps) if float(helps.sum()) > 0 else zero
    loss = args.acc_weight * acc + args.new_bad3_weight * nb3 + args.keep_weight * keep
    return loss, {
        f"{source}_loss": float(loss.detach().cpu()),
        f"{source}_acc": float(acc.detach().cpu()),
        f"{source}_nb3": float(nb3.detach().cpu()),
        f"{source}_keep": float(keep.detach().cpu()),
        f"{source}_s_mean": float(masked_mean(s, valid).detach().cpu()),
        f"{source}_s_on_hard": float(masked_mean(s, valid * hard).detach().cpu()) if float((valid * hard).sum()) > 0 else float("nan"),
    }


def train_one_epoch(model: V32CWithSuppression, loaders: dict[str, DataLoader], optimizer: torch.optim.Optimizer, args: argparse.Namespace, device: torch.device, rng: random.Random) -> dict[str, float]:
    model.train()
    model.base.eval()  # frozen base stays in eval mode (GroupNorm is stateless but be explicit)
    order = [name for name, loader in loaders.items() for _ in range(len(loader))]
    rng.shuffle(order)
    iters = {name: iter(loader) for name, loader in loaders.items()}
    rows: list[dict[str, float]] = []
    for source in order:
        try:
            batch = next(iters[source])
        except StopIteration:
            continue
        loss, metrics = gate_loss_batch(model, batch, args, device, source)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.gate.parameters(), args.grad_clip)
        optimizer.step()
        rows.append(metrics)
    keys = {k for r in rows for k in r}
    return {k: finite_mean([r[k] for r in rows if k in r]) for k in sorted(keys)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets-root", type=Path, default=DEFAULT_TARGETS_ROOT)
    p.add_argument("--oracle-targets-root", type=Path, default=DEFAULT_ORACLE_TARGETS)
    p.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--balanced-split-json", type=Path, default=DEFAULT_BALANCED_SPLIT)
    p.add_argument("--context-frames", type=int, default=4)
    p.add_argument("--crop-size", type=int, default=96)
    p.add_argument("--crop-candidate-tries", type=int, default=12)
    p.add_argument("--crops-per-epoch", type=int, default=60000)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--full-gt-batch-ratio", type=float, default=0.50)
    p.add_argument("--oracle-batch-ratio", type=float, default=0.25)
    p.add_argument("--hard-negative-batch-ratio", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--num-workers", type=int, default=24)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--residual-scale", type=float, default=3.0)
    p.add_argument("--bad-threshold-px", type=float, default=3.0)
    p.add_argument("--good-threshold-px", type=float, default=1.0)
    p.add_argument("--oracle-min-improvement-px", type=float, default=1.0)
    p.add_argument("--oracle-margin-px", type=float, default=1.0)
    p.add_argument("--oracle-hard-only", nargs="?", const=True, default=False, type=parse_bool)
    p.add_argument("--acc-weight", type=float, default=1.0)
    p.add_argument("--new-bad3-weight", type=float, default=4.0)
    p.add_argument("--keep-weight", type=float, default=0.5)
    p.add_argument("--help-margin-px", type=float, default=0.25)
    p.add_argument("--gate-hidden", type=int, default=48)
    p.add_argument("--gate-depth", type=int, default=4)
    p.add_argument("--gate-init-bias", type=float, default=4.0)
    p.add_argument("--robust-loss-clip-px", type=float, default=10.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--early-stop-patience", type=int, default=4)
    p.add_argument("--max-auc-pixels", type=int, default=200000)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--diagnostics-per-clip", type=int, default=2)
    p.add_argument("--fresh", nargs="?", const=True, default=False, type=parse_bool,
                   help="wipe the output root first (default: merge into existing dir, e.g. alongside benchmark artifacts)")
    args = p.parse_args()
    total = args.full_gt_batch_ratio + args.oracle_batch_ratio + args.hard_negative_batch_ratio
    args.full_gt_batch_ratio /= total
    args.oracle_batch_ratio /= total
    args.hard_negative_batch_ratio /= total
    return args


def main() -> int:
    args = parse_args()
    if args.fresh and args.output_root.exists():
        shutil.rmtree(args.output_root)
    (args.output_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (args.output_root / "diagnostics").mkdir(exist_ok=True)
    sys.stdout = (args.output_root / "stdout.log").open("w", buffering=1)
    sys.stderr = (args.output_root / "stderr.log").open("w", buffering=1)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    start = time.perf_counter()

    ckpt = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    input_channels = int(ckpt.get("input_channels", args.context_frames * 2 + 8))
    base = AbstentionCropRefiner(input_channels)
    base.load_state_dict(ckpt["model_state_dict"])
    gate = SuppressionGate(input_channels + 4, hidden=args.gate_hidden, depth=args.gate_depth, init_bias=args.gate_init_bias)
    model = V32CWithSuppression(base, gate, EVAL_THRESHOLD).to(device)
    gate_params = sum(p.numel() for p in gate.parameters())
    base_params = sum(p.numel() for p in base.parameters())

    splits, by_split = load_samples_with_split(args.targets_root, args.balanced_split_json, args.max_frames)
    shards = load_shards(by_split["train"] + by_split["val"] + by_split["test"])
    clips = load_clips(args.oracle_targets_root, args)
    patho_clips = [c for c in clips if c.failure_mode in PATHOLOGICAL_MODES]
    clean_clips = [c for c in clips if c.failure_mode not in PATHOLOGICAL_MODES]
    full_ds = {split: FullFrameDataset(samples, shards, args.context_frames) for split, samples in by_split.items()}
    eval_loaders = {split: make_loader(ds, args.eval_batch_size, max(0, args.num_workers // 2), False, args.prefetch_factor) for split, ds in full_ds.items()}

    # hard negatives mined once from frozen v3.2c (stable all training: base never changes)
    model.eval()
    masks = {}
    eval_args = argparse.Namespace(batch_size=32, context_frames=args.context_frames, residual_scale=args.residual_scale)
    with torch.no_grad():
        for clip in patho_clips:
            _, p_bad, residual, _m = predict_clip(model.base, clip.raws, clip.valids, eval_args, device, args.residual_scale, EVAL_THRESHOLD)
            masks[clip.clip_id] = mine_hard_masks(clip, p_bad, residual, args)
    hn_px = int(sum(m["hard_neg"].sum() for m in masks.values()))
    hp_px = int(sum(m["hard_pos"].sum() for m in masks.values()))

    gt_crops = int(round(args.crops_per_epoch * args.full_gt_batch_ratio))
    oracle_crops = int(round(args.crops_per_epoch * args.oracle_batch_ratio))
    hn_crops = args.crops_per_epoch - gt_crops - oracle_crops
    gt_args = argparse.Namespace(**{**vars(args), "crops_per_epoch": gt_crops})
    loaders = {
        "gt": make_loader(BalancedCropDataset(by_split["train"], shards, gt_args), args.batch_size, args.num_workers, True, args.prefetch_factor),
        "oracle": make_loader(OracleCropDataset(clean_clips, args, oracle_crops), args.batch_size, max(2, args.num_workers // 4), True, args.prefetch_factor),
        "hardneg": make_loader(HardNegativeCropDataset(patho_clips, masks, args, hn_crops), args.batch_size, max(2, args.num_workers // 4), True, args.prefetch_factor),
    }
    optimizer = torch.optim.AdamW(gate.parameters(), lr=args.lr)

    run_lines = [
        f"device={device} gate_params={gate_params} base_params={base_params} (base frozen)",
        f"base_checkpoint={args.base_checkpoint} base_epoch={ckpt.get('epoch')} threshold={EVAL_THRESHOLD}",
        f"frames={{'train': {len(full_ds['train'])}, 'val': {len(full_ds['val'])}, 'test': {len(full_ds['test'])}}}",
        f"clips={len(clips)} patho={len(patho_clips)} clean={len(clean_clips)} hard_neg_px={hn_px} hard_pos_px={hp_px}",
        f"crops gt={gt_crops} oracle={oracle_crops} hardneg={hn_crops} batch={args.batch_size} epochs={args.epochs}",
        f"weights acc={args.acc_weight} nb3={args.new_bad3_weight} keep={args.keep_weight} help_margin={args.help_margin_px}px",
    ]
    (args.output_root / "run.log").write_text("\n".join(run_lines) + "\n")

    def log(line: str) -> None:
        run_lines.append(line)
        (args.output_root / "run.log").write_text("\n".join(run_lines) + "\n")

    rng = random.Random(555)
    train_rows: list[dict[str, Any]] = []
    best_score = float("inf")
    best_epoch = 0
    epoch = 0
    for epoch in range(1, args.epochs + 1):
        metrics = train_one_epoch(model, loaders, optimizer, args, device, rng)
        sel_by_thr, _ = eval_selected_groups(model, clips, eval_args, device)
        sel = sel_by_thr[EVAL_THRESHOLD]
        fg_rows, _ = evaluate(model, eval_loaders["val"], args, device, "val", per_sequence=False)
        fg = row_at(fg_rows, EVAL_THRESHOLD)
        score = score_epoch(sel, fg)
        train_rows.append({
            "epoch": epoch, "score": score, **metrics,
            "sel_all_mae": sel["all"]["refined_mae"], "sel_all_gap": sel["all"]["oracle_gap_recovered_pct"],
            "sel_patho_new_bad3": sel["patho"]["new_bad3_frame_mean_pct"], "sel_clean_new_bad3": sel["clean"]["new_bad3_frame_mean_pct"],
            "sel_modified": sel["all"]["modified_pct"], "fullgt_val_refined": fg["refined_hard_mae"],
        })
        if score < best_score:
            best_score = score
            best_epoch = epoch
            torch.save({
                "gate_state_dict": gate.state_dict(), "args": vars(args), "splits": splits,
                "base_checkpoint": str(args.base_checkpoint), "gate_params": gate_params,
                "epoch": epoch, "threshold": EVAL_THRESHOLD, "selected_metrics": sel, "full_gt_val_metrics": fg,
            }, args.output_root / "checkpoints" / "best.pt")
        log(
            f"epoch={epoch} score={score:.4f} sel_mae={sel['all']['refined_mae']:.4f} gap={sel['all']['oracle_gap_recovered_pct']:.2f}% "
            f"patho_nb3={sel['patho']['new_bad3_frame_mean_pct']:.2f}% clean_nb3={sel['clean']['new_bad3_frame_mean_pct']:.2f}% "
            f"mod={sel['all']['modified_pct']:.2f}% s_hard={metrics.get('gt_s_on_hard', float('nan')):.3f} "
            f"fullgt_val={fg['raw_mae']:.4f}->{fg['refined_hard_mae']:.4f}"
        )
        if epoch - best_epoch >= args.early_stop_patience:
            log(f"early_stop epoch={epoch} best_epoch={best_epoch}")
            break

    best = torch.load(args.output_root / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    gate.load_state_dict(best["gate_state_dict"])

    sel_by_thr, predictions = eval_selected_groups(model, clips, eval_args, device)
    sel = sel_by_thr[EVAL_THRESHOLD]
    frame_rows = selected_frame_rows(clips, predictions, EVAL_THRESHOLD, args)
    fg_final = {}
    for split in ("val", "test"):
        rows, _ = evaluate(model, eval_loaders[split], args, device, split, per_sequence=False)
        fg_final[split] = row_at(rows, EVAL_THRESHOLD)

    write_csv_union(args.output_root / "train_log.csv", train_rows)
    write_csv_union(args.output_root / "selected_oracle_metrics.csv", frame_rows)
    write_csv(args.output_root / "full_gt_val_metrics.csv", [fg_final["val"]])
    write_csv(args.output_root / "full_gt_test_metrics.csv", [fg_final["test"]])

    v32c = BASELINES["v3.2c"]
    success = {
        "selected_mae_at_most_11_03": bool(sel["all"]["refined_mae"] <= 11.03),
        "gap_at_least_6_8pct": bool(sel["all"]["oracle_gap_recovered_pct"] >= 6.8),
        "patho_new_bad3_below_8pct": bool(sel["patho"]["new_bad3_frame_mean_pct"] < 8.0),
        "clean_new_bad3_at_most_1pct": bool(sel["clean"]["new_bad3_frame_mean_pct"] <= 1.0),
        "full_gt_test_beats_raw": bool(fg_final["test"]["refined_hard_mae"] < fg_final["test"]["raw_mae"]),
    }
    summary = {
        "model": "suppression_only_gate_over_frozen_v3.2c",
        "output_root": str(args.output_root),
        "gate_params": gate_params,
        "base_params_frozen": base_params,
        "best_epoch": best["epoch"],
        "epochs_run": epoch,
        "threshold": EVAL_THRESHOLD,
        "elapsed_seconds": time.perf_counter() - start,
        "hard_neg_pixels_mined": hn_px,
        "hard_pos_pixels_mined": hp_px,
        "selected_all": sel["all"],
        "selected_pathological": sel["patho"],
        "selected_clean": sel["clean"],
        "full_gt_val": fg_final["val"],
        "full_gt_test": fg_final["test"],
        "baselines": BASELINES,
        "success_criteria": success,
        "no_teacher_inference": True,
    }
    (args.output_root / "aggregate_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(success, indent=2))
    print(json.dumps({"selected_all": sel["all"], "patho": sel["patho"], "clean": sel["clean"]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
