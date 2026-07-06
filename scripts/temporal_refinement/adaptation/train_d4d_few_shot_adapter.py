#!/usr/bin/env python3
"""D4D few-shot adaptation pilot trainer (canonical path for both models, all modes).

Hypothesis: SCARED-trained refiners keep large-error correction skill on D4D but their
activation policy false-activates on raw-good pixels; light target calibration should fix it.

Modes:
  zero_shot        no training (eval only)
  calibration_only train minimal activation-policy params
  head_only        train task heads, freeze feature trunk
  full             fine-tune all refiner params
  scratch          same arch, random init, full training (sample-efficiency control)

Protocol (frozen eval): train sessions come from the existing few_shot split train.csv
MINUS any session in the frozen session_disjoint validation/test (drops recorded in
audit/split_inventory.csv). ALL runs are selected on the frozen session_disjoint
validation and reported on the frozen session_disjoint test -> comparable across sizes.

Supervision: pixel-level Zivid GT at the anchor frame only (4-frame causal shards from
the zero-shot pipeline). Loss (predefined):
  L = L1(refined, gt | valid)
    + 1.0  * mean(|applied|  | valid & raw_err<1px)   # raw-good preservation
    + 0.05 * mean(|applied|  | valid)                 # identity preference / sparsity
Checkpoint selection (predefined, validation only):
  combined = refined_MAE + 0.02*new_Bad3_pct + 0.5*harmful_rate   (lower = better)
Selectivity (test reporting):
  selectivity = pooled_delta_MAE(raw_err>6px) - max(0, -pooled_delta_MAE(raw_err<1px))
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

import sys
ROOT = Path("/dtu/p1/leopam/ARGOS")
for p in ("scripts/temporal_refinement/ood/eval", "scripts/temporal_refinement",
          "scripts/temporal_refinement/models", "scripts/temporal_refinement/eval_scripts"):
    sys.path.insert(0, str(ROOT / p))
from evaluate_ood_refiners import frame_metrics, load_samples_and_shards, edge_map, EPS  # noqa: E402
from train_tiny_refiner_v3_1_staged_abstention import FullFrameDataset  # noqa: E402
from model_registry import build_registry  # noqa: E402

INDEX = ROOT / "results/03_temporal_refinement/ood/d4d_s2m2_zero_shot/d4d_index.csv"
SPLITS = ROOT / "dataset/D4D/processed/keyframe_stereo_gt/splits"
OUT_ROOT = ROOT / "results/03_temporal_refinement/adaptation/d4d_few_shot_pilot/runs"

W_GOOD, W_SPARSE = 1.0, 0.05
LAM_NB3, LAM_HARM = 0.02, 0.5
GROUPS = {
    "v3.2c": {
        "calibration_only": ["bad_head"],
        "head_only": ["bad_head", "residual_head"],
    },
    "EGBM-v3-CARE-S": {
        "calibration_only": ["bad_head", "damping_head", "threshold_head", "router_head", "boundary_atten"],
        "head_only": ["bad_head", "damping_head", "threshold_head", "router_head", "boundary_atten",
                      "expert_heads", "boundary_head", "care_head"],
    },
}


def split_sessions(csv_path: Path) -> set:
    return {(r["specimen_id"], r["session_id"]) for r in csv.DictReader(csv_path.open())}


def subset(samples, index_meta, sessions):
    keep = []
    for s in samples:
        m = index_meta[s.sequence_id]
        if (m["specimen"], m["session"]) in sessions:
            keep.append(s)
    return keep


def set_trainable(model, mode, model_name):
    for p in model.parameters():
        p.requires_grad = mode in ("full", "scratch")
    if mode in ("calibration_only", "head_only"):
        prefixes = GROUPS[model_name][mode]
        for n, p in model.named_parameters():
            if any(n == pre or n.startswith(pre + ".") for pre in prefixes):
                p.requires_grad = True
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    return train_p, tot


def soft_apply(entry, model, x, raw_t):
    """Differentiable refined map. v3.2c: soft p_bad application (its training convention);
    EGBM: internally gated residual."""
    out = model(x, entry.residual_scale)
    bad_logit, p_bad, residual = out[0], out[1], out[2]
    if entry.mode == "threshold_gate":
        applied = p_bad * residual
    else:
        applied = residual
    return raw_t + applied[:, 0], applied[:, 0], p_bad[:, 0]


@torch.no_grad()
def evaluate(entry, model, samples, shards, device, want_bins=False):
    ds = FullFrameDataset(samples, shards, 4)
    rows = []
    bins_acc = {}  # label -> [raw_sum, ref_sum, n]
    BINS = [0, 1, 3, 6, 12, 1e9]; LABELS = ["<1", "1-3", "3-6", "6-12", ">12"]
    for i in range(len(ds)):
        item = ds[i]
        raw = item["raw"][0].numpy().astype(np.float32)
        gt = item["gt"][0].numpy().astype(np.float32)
        valid = item["valid"][0].numpy() > 0.5
        if valid.sum() == 0:
            continue
        x = item["x"].unsqueeze(0).to(device)
        raw_t = item["raw"][0].to(device)
        # inference-time application = the model's own policy (hard threshold for v3.2c)
        refined_t, _, _, _ = entry.refine(x, raw_t, device)
        refined = refined_t.detach().cpu().numpy().astype(np.float32)
        d = frame_metrics(raw, refined, gt, valid, edge_map(raw))
        d["sequence_id"] = samples[i].sequence_id
        rows.append(d)
        if want_bins:
            er = np.abs(raw - gt)[valid]; ef = np.abs(refined - gt)[valid]
            idx = np.digitize(er, BINS)
            for b in range(1, len(BINS)):
                sel = idx == b
                if sel.any():
                    a = bins_acc.setdefault(LABELS[b - 1], [0.0, 0.0, 0])
                    a[0] += er[sel].sum(); a[1] += ef[sel].sum(); a[2] += int(sel.sum())
    def wmean(key):
        v = [r[key] for r in rows if key in r and r[key] == r[key]]
        return float(np.mean(v)) if v else float("nan")
    agg = {"anchors": len(rows), "refined_mae": wmean("refined_mae"), "raw_mae": wmean("raw_mae"),
           "delta_mae": wmean("delta_mae"), "refined_bad1": wmean("refined_bad1"),
           "refined_bad3": wmean("refined_bad3"), "refined_bad5": wmean("refined_bad5"),
           "new_bad3_pct": wmean("new_bad3_pct_of_rawgood"), "harmful_rate": wmean("harmful_rate"),
           "beneficial_rate": wmean("beneficial_rate"), "modified_pixel_ratio": wmean("modified_pixel_ratio"),
           "boundary_refined_mae": wmean("boundary_refined_mae"),
           "pct_anchors_improved": float(np.mean([r["delta_mae"] > EPS for r in rows]) * 100) if rows else 0.0}
    agg["combined_score"] = agg["refined_mae"] + LAM_NB3 * agg["new_bad3_pct"] + LAM_HARM * agg["harmful_rate"]
    bins = {}
    if want_bins:
        for lab, a in bins_acc.items():
            bins[lab] = {"pixels": a[2], "raw_mae": a[0] / a[2], "refined_mae": a[1] / a[2],
                         "delta_mae": (a[0] - a[1]) / a[2]}
        big = [bins[l] for l in (">12", "6-12") if l in bins]
        good = bins.get("<1")
        pooled_big = (sum(b["raw_mae"] * b["pixels"] for b in big) - sum(b["refined_mae"] * b["pixels"] for b in big)) / max(sum(b["pixels"] for b in big), 1)
        agg["selectivity"] = pooled_big - max(0.0, -(good["delta_mae"] if good else 0.0))
        agg["delta_lt1"] = good["delta_mae"] if good else float("nan")
        agg["delta_gt6_pooled"] = pooled_big
    return agg, rows, bins


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=["v3.2c", "EGBM-v3-CARE-S"])
    ap.add_argument("--adaptation-mode", required=True,
                    choices=["zero_shot", "calibration_only", "head_only", "full", "scratch"])
    ap.add_argument("--split", required=True, help="few_shot split name e.g. 4session_seed1, or 'none' for zero_shot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-updates", type=int, default=0)
    ap.add_argument("--output-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    run_id = f"{args.model.replace('.','_')}__{args.adaptation_mode}__{args.split}__seed{args.seed}"
    out = args.output_root / run_id
    out.mkdir(parents=True, exist_ok=True)

    samples, shards, index_rows = load_samples_and_shards(INDEX)
    meta = {r["sequence_id"]: r for r in index_rows}
    frozen_val = split_sessions(SPLITS / "session_disjoint/validation.csv")
    frozen_test = split_sessions(SPLITS / "session_disjoint/test.csv")
    val_s = subset(samples, meta, frozen_val)
    test_s = subset(samples, meta, frozen_test)
    train_s = []
    if args.adaptation_mode != "zero_shot":
        tr_sessions = split_sessions(SPLITS / "few_shot" / args.split / "train.csv") - frozen_val - frozen_test
        train_s = subset(samples, meta, tr_sessions)
        # drop zero-valid anchors from training
        keep = []
        for s in train_s:
            z = shards[s.target_path]
            if z["valid_mask"][s.offset].sum() > 0:
                keep.append(s)
        train_s = keep
    cfg = {"run_id": run_id, "model": args.model, "mode": args.adaptation_mode, "split": args.split,
           "seed": args.seed, "epochs": args.epochs, "patience": args.patience, "batch": args.batch,
           "train_anchors": len(train_s), "val_anchors": len(val_s), "test_anchors": len(test_s),
           "loss": {"w_good": W_GOOD, "w_sparse": W_SPARSE},
           "selection": {"lam_newbad3": LAM_NB3, "lam_harm": LAM_HARM},
           "protocol": "few_shot train sessions minus frozen; frozen session_disjoint val/test"}
    if args.dry_run:
        print(json.dumps(cfg, indent=2, default=float)); return 0

    reg = {e.name: e for e in build_registry()}
    entry = reg[args.model]
    if args.adaptation_mode == "scratch":
        ck = torch.load(entry.checkpoint, map_location="cpu", weights_only=False)
        model = entry.build(ck.get("input_channels", 16), entry.residual_scale).to(device)
        if entry.mode == "threshold_gate" and entry.threshold is None:
            entry.threshold = float(ck.get("threshold", 0.5))
        entry._model = model
    else:
        model = entry.load(device)
    lr = args.lr or ({"calibration_only": 1e-3, "head_only": 3e-4, "full": 3e-5, "scratch": 3e-4}
                     .get(args.adaptation_mode, 1e-4))
    t0 = time.time()
    history = []
    if args.adaptation_mode == "zero_shot":
        best_state = None
    else:
        train_p, tot_p = set_trainable(model, args.adaptation_mode, args.model)
        cfg.update({"trainable_params": train_p, "total_params": tot_p, "lr": lr,
                    "trainable_pct": round(100 * train_p / tot_p, 3)})
        frozen_snapshot = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
        ds = FullFrameDataset(train_s, shards, 4)
        if len(ds) == 0:
            cfg["status"] = "skipped_no_valid_train_anchors"
            (out / "config.json").write_text(json.dumps(cfg, indent=2, default=float) + "\n")
            print(json.dumps({"run_id": run_id, "status": "skipped_no_valid_train_anchors"}))
            return 0
        loss = torch.zeros(())
        best = (1e9, -1)
        updates = 0
        for epoch in range(args.epochs):
            model.train()
            order = np.random.permutation(len(ds))
            for i0 in range(0, len(ds), args.batch):
                items = [ds[int(j)] for j in order[i0:i0 + args.batch]]
                x = torch.stack([it["x"] for it in items]).to(device)
                raw = torch.stack([it["raw"][0] for it in items]).to(device)
                gt = torch.stack([it["gt"][0] for it in items]).to(device)
                valid = torch.stack([it["valid"][0] for it in items]).to(device) > 0.5
                refined, applied, _p = soft_apply(entry, model, x, raw)
                er = (raw - gt).abs()
                good = valid & (er < 1.0)
                l_fit = (refined - gt).abs()[valid].mean()
                l_good = applied.abs()[good].mean() if good.any() else torch.zeros((), device=device)
                l_sparse = applied.abs()[valid].mean()
                loss = l_fit + W_GOOD * l_good + W_SPARSE * l_sparse
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                updates += 1
                if args.max_updates and updates >= args.max_updates:
                    break
            model.eval()
            vagg, _, _ = evaluate(entry, model, val_s, shards, device)
            history.append({"epoch": epoch, "loss": float(loss), "val_mae": vagg["refined_mae"],
                            "val_newbad3": vagg["new_bad3_pct"], "val_harm": vagg["harmful_rate"],
                            "val_combined": vagg["combined_score"]})
            if vagg["combined_score"] < best[0] - 1e-5:
                best = (vagg["combined_score"], epoch)
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "cfg": cfg,
                            "val": vagg}, out / "best_combined.pt")
            if epoch - best[1] >= args.patience or (args.max_updates and updates >= args.max_updates):
                break
        # frozen-params bitwise check
        drift = [n for n, p in model.named_parameters()
                 if n in frozen_snapshot and not torch.equal(p.detach(), frozen_snapshot[n])]
        cfg["frozen_param_drift"] = drift
        cfg["updates"] = updates
        cfg["best_epoch"] = best[1]
        # reload selected checkpoint
        ckb = torch.load(out / "best_combined.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ckb["model_state_dict"]); model.eval()

    tagg, trows, tbins = evaluate(entry, model, test_s, shards, device, want_bins=True)
    cfg["wall_time_s"] = round(time.time() - t0, 1)
    cfg["test"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in tagg.items()}
    (out / "config.json").write_text(json.dumps(cfg, indent=2, default=float) + "\n")
    with (out / "test_metrics.csv").open("w", newline="") as f:
        if trows:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in trows for k in r})); w.writeheader(); w.writerows(trows)
    (out / "raw_error_bin_metrics.json").write_text(json.dumps(tbins, indent=2, default=float) + "\n")
    if history:
        with (out / "train_log.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0].keys())); w.writeheader(); w.writerows(history)
    print(json.dumps({"run_id": run_id, "train_anchors": len(train_s),
                      "trainable_params": cfg.get("trainable_params", 0),
                      "best_epoch": cfg.get("best_epoch"), "frozen_drift": cfg.get("frozen_param_drift", []),
                      "test_mae": tagg["refined_mae"], "test_delta_mae": tagg["delta_mae"],
                      "test_newbad3": tagg["new_bad3_pct"], "test_harmful": tagg["harmful_rate"],
                      "delta_lt1": tagg.get("delta_lt1"), "delta_gt6": tagg.get("delta_gt6_pooled"),
                      "selectivity": tagg.get("selectivity")}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
