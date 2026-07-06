#!/usr/bin/env python3
"""Aggregate the D4D few-shot pilot matrix: mean±std across seeds, tables + plots."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/dtu/p1/leopam/ARGOS")
BASE = ROOT / "results/03_temporal_refinement/adaptation/d4d_few_shot_pilot"
RUNS = BASE / "runs"

KEYS = ["refined_mae", "delta_mae", "refined_bad3", "new_bad3_pct", "harmful_rate",
        "pct_anchors_improved", "modified_pixel_ratio", "delta_lt1", "delta_gt6_pooled",
        "selectivity", "combined_score"]


def load_runs():
    rows = []
    for cfg_p in sorted(RUNS.glob("*/config.json")):
        cfg = json.loads(cfg_p.read_text())
        t = cfg.get("test", {})
        size = cfg["split"].split("_")[0] if cfg["split"] != "none" else "0session"
        rows.append({"run_id": cfg["run_id"], "model": cfg["model"], "mode": cfg["mode"],
                     "size": size, "seed": cfg["seed"], "split": cfg["split"],
                     "train_anchors": cfg.get("train_anchors", 0),
                     "trainable_params": cfg.get("trainable_params", 0),
                     "best_epoch": cfg.get("best_epoch"), "wall_time_s": cfg.get("wall_time_s"),
                     **{k: t.get(k) for k in KEYS}})
    return rows


def main():
    rows = load_runs()
    (BASE / "aggregate").mkdir(exist_ok=True)
    with (BASE / "run_manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # aggregate mean±std by (model, mode, size)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["mode"], r["size"])].append(r)
    agg = []
    for (m, mode, size), rr in sorted(groups.items()):
        row = {"model": m, "mode": mode, "size": size, "n_seeds": len(rr),
               "train_anchors": ";".join(str(x["train_anchors"]) for x in rr),
               "trainable_params": rr[0]["trainable_params"]}
        for k in KEYS:
            v = [x[k] for x in rr if x[k] is not None and x[k] == x[k]]
            row[f"{k}_mean"] = round(float(np.mean(v)), 4) if v else None
            row[f"{k}_std"] = round(float(np.std(v)), 4) if len(v) > 1 else 0.0
        agg.append(row)
    with (BASE / "aggregate" / "aggregate_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0].keys())); w.writeheader(); w.writerows(agg)

    # plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        (BASE / "figures").mkdir(exist_ok=True)
        sizes_order = ["0session", "1session", "2session", "4session", "8session"]
        for metric, fname, title in [
            ("refined_mae", "mae_vs_sessions.png", "Test MAE vs sessions (frozen test)"),
            ("new_bad3_pct", "newbad3_vs_sessions.png", "new-Bad3 % vs sessions"),
            ("harmful_rate", "harmful_vs_sessions.png", "harmful rate vs sessions"),
            ("delta_lt1", "rawgood_delta_vs_sessions.png", "raw<1px delta MAE (0=no damage)"),
            ("delta_gt6_pooled", "largeerr_delta_vs_sessions.png", "raw>6px delta MAE (higher=better)"),
            ("selectivity", "selectivity_vs_sessions.png", "selectivity score"),
        ]:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
            for ax, model in zip(axes, ["v3.2c", "EGBM-v3-CARE-S"]):
                for mode in ["calibration_only", "head_only", "full", "scratch"]:
                    xs, ys, es = [], [], []
                    for i, s in enumerate(sizes_order):
                        g = [a for a in agg if a["model"] == model and a["mode"] == mode and a["size"] == s]
                        if g and g[0][f"{metric}_mean"] is not None:
                            xs.append(i); ys.append(g[0][f"{metric}_mean"]); es.append(g[0][f"{metric}_std"])
                    if xs:
                        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=mode)
                z = [a for a in agg if a["model"] == model and a["mode"] == "zero_shot"]
                if z and z[0][f"{metric}_mean"] is not None:
                    ax.axhline(z[0][f"{metric}_mean"], color="gray", ls="--", lw=1, label="zero-shot")
                ax.set_title(model); ax.set_xticks(range(len(sizes_order)))
                ax.set_xticklabels([s.replace("session", "s") for s in sizes_order])
                ax.grid(alpha=0.3)
            axes[0].legend(fontsize=8); fig.suptitle(title)
            fig.tight_layout(); fig.savefig(BASE / "figures" / fname, dpi=110); plt.close(fig)
        print("plots written")
    except Exception as e:
        print(f"plots skipped: {e}")
    print(json.dumps({"runs": len(rows), "groups": len(agg)}, indent=2))


if __name__ == "__main__":
    main()
