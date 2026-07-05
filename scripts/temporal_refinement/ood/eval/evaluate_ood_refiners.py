#!/usr/bin/env python3
"""Zero-shot OOD benchmark for ARGOS temporal stereo refiners.

Consumes OOD shards (build_ood_shards.py) through the models' own FullFrameDataset,
runs every registered refiner with its selected primary checkpoint (no OOD tuning),
and reports the accuracy + safety metric battery vs GT. `raw` (S2M2-S, no refiner) is
the baseline. Produces per-frame / per-sequence / per-model CSVs, safety + correction
analysis, qualitative panels, and key plots.

Usage:
  python evaluate_ood_refiners.py --dataset servct \
      --index results/03_temporal_refinement/ood/prepared/servct/frame_targets_index.csv \
      --out   results/03_temporal_refinement/ood/zero_shot_benchmark
  add --smoke to run one sequence and skip plots.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path("/dtu/p1/leopam/ARGOS")
TRAIN = ROOT / "scripts/temporal_refinement"
for p in (TRAIN, TRAIN / "models", TRAIN / "eval_scripts", Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from train_tiny_refiner_v3_1_staged_abstention import FullFrameDataset  # noqa: E402
from model_registry import build_registry, ModelEntry  # noqa: E402

EPS = 0.1  # px, "modified pixel" / improvement threshold
BADS = (1.0, 3.0, 5.0)
LARGE = (3.0, 6.0, 12.0, 20.0)


# ----------------------------- data ---------------------------------------
def load_samples_and_shards(index_csv: Path):
    rows = list(csv.DictReader(index_csv.open()))
    shards: dict[Path, dict] = {}
    samples = []
    for r in rows:
        tp = Path(r["target_path"])
        if tp not in shards:
            z = np.load(tp)
            shards[tp] = {k: z[k] for k in ("raw_disp", "gt_disp", "valid_mask", "delta_disp_gt_minus_raw")}
        samples.append(SimpleNamespace(
            sequence_id=r["sequence_id"], frame_id=r["frame_id"],
            frame_index=int(r["frame_index"]), offset=int(r["frame_offset"]),
            target_path=tp, dataset=r.get("dataset", ""), split=r.get("split", ""),
            continuity_flag=r.get("continuity_flag", ""),
        ))
    return samples, shards, rows


def edge_map(raw: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(raw); gy = np.zeros_like(raw)
    gx[:, 1:] = raw[:, 1:] - raw[:, :-1]
    gy[1:, :] = raw[1:, :] - raw[:-1, :]
    return np.sqrt(gx * gx + gy * gy)


# ----------------------------- metrics ------------------------------------
def frame_metrics(raw, refined, gt, valid, edge) -> dict:
    m = valid
    n = int(m.sum())
    raw, refined, gt = raw[m], refined[m], gt[m]
    er, ef = np.abs(raw - gt), np.abs(refined - gt)
    applied = refined - raw
    d = {"valid_px": n}
    d["raw_mae"] = float(er.mean()); d["refined_mae"] = float(ef.mean())
    d["delta_mae"] = d["raw_mae"] - d["refined_mae"]
    d["rel_impr"] = d["delta_mae"] / max(d["raw_mae"], 1e-6)
    d["raw_rmse"] = float(np.sqrt((er ** 2).mean())); d["refined_rmse"] = float(np.sqrt((ef ** 2).mean()))
    for t in BADS:
        d[f"raw_bad{int(t)}"] = float((er >= t).mean() * 100)
        d[f"refined_bad{int(t)}"] = float((ef >= t).mean() * 100)
    # corrections
    modified = np.abs(applied) > EPS
    nmod = int(modified.sum())
    d["modified_pixel_ratio"] = nmod / max(n, 1)
    improved = modified & (ef < er - EPS)
    harmed = modified & (ef > er + EPS)
    neutral = modified & ~improved & ~harmed
    d["beneficial_rate"] = int(improved.sum()) / max(nmod, 1)
    d["harmful_rate"] = int(harmed.sum()) / max(nmod, 1)
    d["neutral_rate"] = int(neutral.sum()) / max(nmod, 1)
    d["net_benefit_ratio"] = int(improved.sum()) / max(int(harmed.sum()), 1)
    d["mean_beneficial_reduction"] = float((er - ef)[improved].mean()) if improved.any() else 0.0
    d["mean_harmful_increase"] = float((ef - er)[harmed].mean()) if harmed.any() else 0.0
    # new-bad / fixed-bad relative to raw-good (Bad3, Bad1)
    for t in (1.0, 3.0):
        raw_good = er < t; raw_bad = er >= t
        new_bad = raw_good & (ef >= t)
        fixed_bad = raw_bad & (ef < t)
        d[f"new_bad{int(t)}_pct_of_rawgood"] = int(new_bad.sum()) / max(int(raw_good.sum()), 1) * 100
        d[f"fixed_bad{int(t)}_pct_of_rawbad"] = int(fixed_bad.sum()) / max(int(raw_bad.sum()), 1) * 100
    # correction sign accuracy + magnitude ratio (where raw was wrong & modified)
    need = modified & (er > EPS)
    if need.any():
        want = np.sign(gt - raw)[need]; got = np.sign(applied)[need]
        d["correction_sign_accuracy"] = float((want == got).mean())
    else:
        d["correction_sign_accuracy"] = float("nan")
    true_corr = np.abs(gt - raw)[modified]
    d["correction_magnitude_ratio"] = float(np.abs(applied)[modified].mean() / max(true_corr.mean(), 1e-6)) if modified.any() else 0.0
    # overshoot/undershoot: among modified where raw wrong, did we cross gt (overshoot) or fall short
    if need.any():
        crossed = np.sign(raw - gt)[need] != np.sign(refined - gt)[need]
        d["overshoot_rate"] = float((crossed & (ef[need] > EPS)).mean())
        d["undershoot_rate"] = float((~crossed).mean())
    else:
        d["overshoot_rate"] = d["undershoot_rate"] = float("nan")
    # correction magnitude distribution
    ac = np.abs(applied)
    d["corr_p50"] = float(np.percentile(ac, 50)); d["corr_p95"] = float(np.percentile(ac, 95))
    d["corr_p99"] = float(np.percentile(ac, 99)); d["corr_max"] = float(ac.max())
    for t in LARGE:
        d[f"frac_corr_gt{int(t)}px"] = float((ac > t).mean())
    # catastrophic harmful: large corrections that hurt
    d["catastrophic_harmful_gt6px"] = int((harmed & (np.abs(applied) > 6)).sum()) / max(n, 1)
    # top-k% damage contribution (harmful error increase concentration)
    inc = np.clip(ef - er, 0, None)
    tot = inc.sum()
    if tot > 0:
        s = np.sort(inc)[::-1]
        for pct in (0.1, 1.0, 5.0, 10.0):
            k = max(1, int(len(s) * pct / 100))
            d[f"top{pct}pct_damage_contrib"] = float(s[:k].sum() / tot)
    else:
        for pct in (0.1, 1.0, 5.0, 10.0):
            d[f"top{pct}pct_damage_contrib"] = 0.0
    # boundary vs interior
    b = edge[m] > 1.0
    if b.any():
        d["boundary_raw_mae"] = float(er[b].mean()); d["boundary_refined_mae"] = float(ef[b].mean())
    if (~b).any():
        d["interior_raw_mae"] = float(er[~b].mean()); d["interior_refined_mae"] = float(ef[~b].mean())
    return d


def aux_safety(aux: dict, valid: np.ndarray) -> dict:
    out = {}
    m = valid
    for key in ("trust", "damping", "gate", "large_magnitude", "verifier_benefit",
                "verifier_new_bad3_risk", "verifier_safe_alpha"):
        if key in aux:
            v = aux[key][0, 0].detach().cpu().numpy()
            out[f"mean_{key}"] = float(v[m].mean()) if m.any() else float(v.mean())
    if "large_proposal" in aux:
        lp = np.abs(aux["large_proposal"][0, 0].detach().cpu().numpy())
        out["large_proposal_util_gt6px"] = float((lp[m] > 6).mean()) if m.any() else 0.0
        out["large_proposal_mean_abs"] = float(lp[m].mean()) if m.any() else 0.0
    return out


# ----------------------------- run ----------------------------------------
def run_model(entry: ModelEntry | None, samples, shards, device, context_frames=4):
    """entry=None -> raw baseline (residual 0). Returns list of per-frame dicts."""
    ds = FullFrameDataset(samples, shards, context_frames)
    if entry is not None:
        entry.load(device)
    per_frame = []
    for i in range(len(ds)):
        item = ds[i]
        raw = item["raw"][0].numpy().astype(np.float32)
        gt = item["gt"][0].numpy().astype(np.float32)
        valid = item["valid"][0].numpy() > 0.5
        edge = edge_map(raw)
        aux = {}
        if entry is None:
            refined = raw.copy(); rt = 0.0
        else:
            x = item["x"].unsqueeze(0).to(device)
            raw_t = item["raw"][0].to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            refined_t, applied_t, p_bad_t, aux = entry.refine(x, raw_t, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            rt = (time.perf_counter() - t0) * 1000
            refined = refined_t.detach().cpu().numpy().astype(np.float32)
        d = frame_metrics(raw, refined, gt, valid, edge)
        d.update({"sequence_id": samples[i].sequence_id, "frame_id": samples[i].frame_id,
                  "split": samples[i].split, "continuity_flag": samples[i].continuity_flag,
                  "runtime_ms": rt})
        if aux:
            d.update(aux_safety(aux, valid))
        per_frame.append((d, raw, refined, gt, valid, edge,
                          (aux if entry is not None else {})))
    return per_frame


def mean_over(frames, key):
    vals = [f[key] for f in frames if key in f and f[key] == f[key]]  # drop nan
    return float(np.mean(vals)) if vals else float("nan")


def aggregate(per_frame_dicts, model, dataset):
    keys = [k for k in per_frame_dicts[0] if isinstance(per_frame_dicts[0][k], (int, float))]
    agg = {"model": model, "dataset": dataset, "frames": len(per_frame_dicts)}
    for k in keys:
        agg[k] = mean_over(per_frame_dicts, k)
    return agg


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    lead = [k for k in ("model", "dataset", "sequence_id", "frame_id") if k in keys]
    keys = lead + [k for k in keys if k not in lead]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def qualitative(out_dir: Path, model: str, rec, max_panels=3):
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    picks = rec[:max_panels]
    for (d, raw, refined, gt, valid, edge, aux) in picks:
        vmax = float(np.percentile(gt[valid], 99)) if valid.any() else 1.0

        def cz(a, mx, cmap=cv2.COLORMAP_TURBO):
            a = np.clip(a, 0, mx) / max(mx, 1e-6)
            return cv2.applyColorMap((a * 255).astype(np.uint8), cmap)
        er, ef = np.abs(raw - gt), np.abs(refined - gt)
        applied = refined - raw
        benef = (np.abs(applied) > EPS) & (ef < er - EPS) & valid
        harm = (np.abs(applied) > EPS) & (ef > er + EPS) & valid
        tiles = [cz(raw, vmax), cz(refined, vmax), cz(gt, vmax),
                 cz(er, 8, cv2.COLORMAP_MAGMA), cz(ef, 8, cv2.COLORMAP_MAGMA),
                 cz(np.abs(applied), 8, cv2.COLORMAP_MAGMA),
                 (benef[..., None] * np.array([0, 200, 0], np.uint8)).astype(np.uint8),
                 (harm[..., None] * np.array([0, 0, 220], np.uint8)).astype(np.uint8)]
        labels = ["raw", "refined", "GT", "raw|err|", "ref|err|", "|applied|", "beneficial", "harmful"]
        h = raw.shape[0]
        strip = []
        for t, lab in zip(tiles, labels):
            t = cv2.resize(t, (int(180 * h / raw.shape[0] * raw.shape[1] / raw.shape[1]), h), interpolation=cv2.INTER_NEAREST) if False else t
            cv2.putText(t, lab, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            strip.append(t)
        panel = np.concatenate(strip, axis=1)
        cv2.imwrite(str(out_dir / f"{model}_{d['sequence_id']}_{d['frame_id']}.png"), panel)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="servct")
    ap.add_argument("--index", type=Path,
                    default=ROOT / "results/03_temporal_refinement/ood/prepared/servct/frame_targets_index.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "results/03_temporal_refinement/ood/zero_shot_benchmark")
    ap.add_argument("--smoke", action="store_true", help="one sequence, no plots")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    samples, shards, rows = load_samples_and_shards(args.index)
    if args.smoke:
        first_seq = samples[0].sequence_id
        samples = [s for s in samples if s.sequence_id == first_seq]
    out = args.out / ("smoke" if args.smoke else args.dataset)
    out.mkdir(parents=True, exist_ok=True)
    (out / "qualitative").mkdir(exist_ok=True)

    registry = build_registry()
    print(f"[{args.dataset}] {len(samples)} frames, {len({s.sequence_id for s in samples})} sequences, "
          f"{len(registry)} models + raw, device={device}")

    all_frames: list[dict] = []
    all_seq: list[dict] = []
    all_model: list[dict] = []
    runs = [("raw", None)] + [(e.name, e) for e in registry]
    for name, entry in runs:
        rec = run_model(entry, samples, shards, device)
        fdicts = [r[0] for r in rec]
        for d in fdicts:
            d2 = {"model": name, "dataset": args.dataset, **d}
            all_frames.append(d2)
        # per-sequence
        seqs = sorted({d["sequence_id"] for d in fdicts})
        for sid in seqs:
            sub = [d for d in fdicts if d["sequence_id"] == sid]
            row = aggregate(sub, name, args.dataset); row["sequence_id"] = sid
            all_seq.append(row)
        # per-model (dataset-level)
        mrow = aggregate(fdicts, name, args.dataset)
        all_model.append(mrow)
        print(f"  {name:16s} raw_mae={mrow['raw_mae']:.3f} refined_mae={mrow['refined_mae']:.3f} "
              f"dMAE={mrow['delta_mae']:+.3f} bad3={mrow['refined_bad3']:.1f} "
              f"newBad3={mrow.get('new_bad3_pct_of_rawgood', float('nan')):.2f} "
              f"harmful={mrow['harmful_rate']:.2f} rt={mrow['runtime_ms']:.1f}ms")
        if entry is not None:
            qualitative(out / "qualitative", name, rec)

    write_csv(out / f"{args.dataset}_frame_metrics.csv", all_frames)
    write_csv(out / f"{args.dataset}_sequence_metrics.csv", all_seq)
    write_csv(out / f"{args.dataset}_model_comparison.csv", all_model)

    # safety subset
    safety_keys = ["model", "dataset", "frames", "refined_mae", "delta_mae",
                   "new_bad3_pct_of_rawgood", "new_bad1_pct_of_rawgood", "harmful_rate",
                   "beneficial_rate", "net_benefit_ratio", "catastrophic_harmful_gt6px",
                   "corr_p99", "corr_max", "frac_corr_gt6px", "frac_corr_gt12px", "frac_corr_gt20px",
                   "top1.0pct_damage_contrib", "mean_trust", "large_proposal_util_gt6px"]
    write_csv(out / "safety_metrics.csv", [{k: m.get(k) for k in safety_keys if k in m} for m in all_model])

    summary = {
        "dataset": args.dataset, "device": str(device),
        "n_frames": len(samples), "n_sequences": len({s.sequence_id for s in samples}),
        "models": [m["model"] for m in all_model],
        "protocol": "zero-shot; selected primary checkpoints; no OOD tuning; lowres 0.25 grid; causal 4-frame window",
        "per_model": {m["model"]: {k: m[k] for k in ("raw_mae", "refined_mae", "delta_mae", "rel_impr",
                      "refined_bad3", "new_bad3_pct_of_rawgood", "harmful_rate", "beneficial_rate",
                      "net_benefit_ratio", "corr_p99", "runtime_ms") if k in m} for m in all_model},
    }
    (out / "aggregate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["per_model"], indent=2))
    if not args.smoke:
        try:
            make_plots(out, all_model, args.dataset)
        except Exception as e:  # plots are non-critical
            print(f"[plots] skipped: {e}")
    return 0


def make_plots(out: Path, all_model, dataset):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pdir = out / "diagnostics"; pdir.mkdir(exist_ok=True)
    names = [m["model"] for m in all_model]

    def bar(key, title, fname, pct=False):
        vals = [m.get(key, float("nan")) for m in all_model]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(names, vals, color=["#888" if n == "raw" else "#3b7" for n in names])
        ax.set_title(title); ax.set_ylabel(key); ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(pdir / fname, dpi=110); plt.close(fig)

    bar("refined_mae", f"OOD {dataset}: refined MAE (px, lower=better)", "ood_mae_comparison.png")
    bar("refined_bad3", f"OOD {dataset}: refined Bad-3 %", "ood_bad3_comparison.png")
    bar("new_bad3_pct_of_rawgood", f"OOD {dataset}: NEW Bad-3 from raw-good (%)", "ood_newbad3_comparison.png")
    bar("net_benefit_ratio", f"OOD {dataset}: net benefit (beneficial/harmful)", "ood_net_benefit.png")
    bar("corr_p99", f"OOD {dataset}: correction magnitude p99 (px)", "correction_magnitude_shift.png")


if __name__ == "__main__":
    raise SystemExit(main())
