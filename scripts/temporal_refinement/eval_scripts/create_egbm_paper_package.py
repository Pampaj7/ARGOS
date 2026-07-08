#!/usr/bin/env python3
"""Create the ICRA-ready EGBM paper package from existing evaluation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path


SRC = Path("results/03_temporal_refinement/training/egbm_final_evaluation")
OUT = Path("results/03_temporal_refinement/training/egbm_paper_package")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, "")
        return default if value == "" else float(value)
    except (TypeError, ValueError):
        return default


def fmt(x: float, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:.{digits}f}"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n")


def latex_escape(s: str) -> str:
    return (
        str(s)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def latex_table(rows: list[dict[str, str]], cols: list[tuple[str, str]], caption: str, label: str) -> str:
    align = "l" + "r" * (len(cols) - 1)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(title for _key, title in cols) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        vals = []
        for key, _title in cols:
            vals.append(latex_escape(row.get(key, "")))
        lines.append(" & ".join(vals) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def numeric_table_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for r in rows:
        out.append(
            {
                "method": r["method"],
                "selected_mae": fmt(f(r, "selected_mae"), 3),
                "gap": fmt(f(r, "oracle_gap_recovered_pct"), 2),
                "patho_new_bad3": fmt(f(r, "patho_new_bad3_pct"), 2),
                "clean_new_bad3": fmt(f(r, "clean_new_bad3_pct"), 2),
                "full_gt_test_mae": fmt(f(r, "full_gt_test_mae"), 3),
                "runtime": fmt(f(r, "runtime_ms_frame"), 2),
            }
        )
    return out


def plot_bar(fig_dir: Path, name: str, rows: list[dict[str, str]], key: str, ylabel: str, title: str, lower_better: bool = True) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = [(r["method"], f(r, key)) for r in rows if not math.isnan(f(r, key))]
    labels, vals = zip(*data)
    colors = ["#2f6f8f" if "EGBM" not in label else "#c23b22" for label in labels]
    plt.figure(figsize=(9.0, 4.5))
    plt.bar(range(len(vals)), vals, color=colors)
    plt.xticks(range(len(vals)), labels, rotation=35, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    if lower_better:
        plt.gca().invert_yaxis()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_dir / name, dpi=300)
    plt.close()


def plot_runtime(fig_dir: Path, runtime: dict[str, float], rows: list[dict[str, str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["S2M2 raw", "S2M2+EGBM"]
    s2m2 = runtime["s2m2_assumed_ms"]
    vals = [s2m2, runtime["estimated_total_ms"]]
    plt.figure(figsize=(6.2, 4.0))
    plt.bar(labels, vals, color=["#777777", "#c23b22"])
    plt.axhline(runtime["system_budget_ms"], color="black", linestyle="--", linewidth=1.2, label="100 ms budget")
    plt.ylabel("Runtime (ms/frame)")
    plt.title("Online Runtime Budget")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "runtime_budget_comparison.png", dpi=300)
    plt.close()


def plot_scatter(fig_dir: Path, rows: list[dict[str, str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [f(r, "patho_new_bad3_pct") for r in rows if not math.isnan(f(r, "patho_new_bad3_pct")) and not math.isnan(f(r, "oracle_gap_recovered_pct"))]
    ys = [f(r, "oracle_gap_recovered_pct") for r in rows if not math.isnan(f(r, "patho_new_bad3_pct")) and not math.isnan(f(r, "oracle_gap_recovered_pct"))]
    labels = [r["method"] for r in rows if not math.isnan(f(r, "patho_new_bad3_pct")) and not math.isnan(f(r, "oracle_gap_recovered_pct"))]
    plt.figure(figsize=(6.8, 4.8))
    for x, y, label in zip(xs, ys, labels):
        color = "#c23b22" if "EGBM" in label else "#2f6f8f"
        plt.scatter(x, y, s=70, color=color)
        plt.text(x + 0.15, y + 0.15, label, fontsize=8)
    plt.xlabel("Pathological new Bad-3 (%)")
    plt.ylabel("Oracle gap recovered (%)")
    plt.title("Accuracy-Safety Pareto")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_dir / "accuracy_safety_pareto.png", dpi=300)
    plt.close()

    xs = [f(r, "modified_pixels_pct") for r in rows if not math.isnan(f(r, "modified_pixels_pct")) and not math.isnan(f(r, "global_frame_mean_new_bad3_pct"))]
    ys = [f(r, "global_frame_mean_new_bad3_pct") for r in rows if not math.isnan(f(r, "modified_pixels_pct")) and not math.isnan(f(r, "global_frame_mean_new_bad3_pct"))]
    labels = [r["method"] for r in rows if not math.isnan(f(r, "modified_pixels_pct")) and not math.isnan(f(r, "global_frame_mean_new_bad3_pct"))]
    plt.figure(figsize=(6.8, 4.8))
    for x, y, label in zip(xs, ys, labels):
        color = "#c23b22" if "EGBM" in label else "#2f6f8f"
        plt.scatter(x, y, s=70, color=color)
        plt.text(x + 0.6, y + 0.05, label, fontsize=8)
    plt.xlabel("Modified pixels (%)")
    plt.ylabel("Frame-mean new Bad-3 (%)")
    plt.title("Modification Rate vs. Safety")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_dir / "modified_vs_newbad3.png", dpi=300)
    plt.close()


def plot_damping(fig_dir: Path, damping_rows: list[dict[str, str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clips = []
    hard_neg = []
    hard_pos = []
    for clip in sorted({r["clip_id"] for r in damping_rows}):
        hn = next((r for r in damping_rows if r["clip_id"] == clip and r["group"] == "hard_neg"), None)
        hp = next((r for r in damping_rows if r["clip_id"] == clip and r["group"] == "hard_pos"), None)
        if hn and hp:
            clips.append(hn["failure_mode"])
            hard_neg.append(f(hn, "damping_mean"))
            hard_pos.append(f(hp, "damping_mean"))
    x = range(len(clips))
    plt.figure(figsize=(6.4, 4.0))
    plt.bar([i - 0.18 for i in x], hard_neg, width=0.36, label="hard negatives", color="#777777")
    plt.bar([i + 0.18 for i in x], hard_pos, width=0.36, label="hard positives", color="#c23b22")
    plt.xticks(list(x), clips, rotation=20, ha="right")
    plt.ylabel("Mean damping")
    plt.title("EGBM Damping Separates Harmful and Helpful Corrections")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "damping_hardpos_hardneg.png", dpi=300)
    plt.close()


def plot_threshold(fig_dir: Path, rows: list[dict[str, str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_rows = [r for r in rows if r["group"] == "all"]
    th = [f(r, "base_threshold") for r in all_rows]
    mae = [f(r, "refined_mae") for r in all_rows]
    newb = [f(r, "new_bad3_frame_mean_pct") for r in all_rows]
    fig, ax1 = plt.subplots(figsize=(6.4, 4.0))
    ax1.plot(th, mae, marker="o", color="#c23b22", label="MAE")
    ax1.set_xlabel("Damping threshold")
    ax1.set_ylabel("Selected MAE", color="#c23b22")
    ax2 = ax1.twinx()
    ax2.plot(th, newb, marker="s", color="#2f6f8f", label="new Bad-3")
    ax2.set_ylabel("Frame-mean new Bad-3 (%)", color="#2f6f8f")
    ax1.grid(alpha=0.25)
    plt.title("EGBM Threshold Sweep")
    fig.tight_layout()
    plt.savefig(fig_dir / "threshold_sweep_egbm.png", dpi=300)
    plt.close()


def copy_diagnostic_threshold(src: Path, fig_dir: Path) -> None:
    # ponytail: keep the source diagnostic too; regenerated figure above is the paper one.
    src_plot = src / "diagnostics" / "threshold_sweep_mae.png"
    if src_plot.exists():
        shutil.copy2(src_plot, fig_dir / "source_threshold_sweep_mae.png")


def make_figures(fig_dir: Path, src: Path, comparison: list[dict[str, str]], damping: list[dict[str, str]], threshold: list[dict[str, str]], runtime: dict[str, float]) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_bar(fig_dir, "final_comparison_selected_mae.png", comparison, "selected_mae", "Selected clips MAE (px)", "Selected-Clip Accuracy", True)
    plot_bar(fig_dir, "oracle_gap_recovered.png", comparison, "oracle_gap_recovered_pct", "Oracle gap recovered (%)", "Oracle/SAV Headroom Recovered", False)
    plot_bar(fig_dir, "patho_new_bad3_comparison.png", comparison, "patho_new_bad3_pct", "Pathological new Bad-3 (%)", "Safety on Pathological Clips", True)
    plot_bar(fig_dir, "full_gt_test_mae_comparison.png", comparison, "full_gt_test_mae", "Full-GT test MAE (px)", "Full-Dataset Generalization", True)
    plot_runtime(fig_dir, runtime, comparison)
    plot_damping(fig_dir, damping)
    plot_scatter(fig_dir, comparison)
    plot_threshold(fig_dir, threshold)
    copy_diagnostic_threshold(src, fig_dir)


def make_markdown(summary: dict, comparison: list[dict[str, str]]) -> dict[str, str]:
    e = summary["selected"]["all"]
    p = summary["selected"]["pathological"]
    c = summary["selected"]["clean"]
    t = summary["full_gt_test"]
    r = summary["runtime"]
    readme = f"""
# EGBM Paper Package

## Executive Summary

EGBM is the new main ARGOS refiner branch. It is the first model that simultaneously improves selected oracle clips, suppresses pathological overcorrection, beats raw S2M2 on the full held-out GT test split, and remains inside the online robotic perception budget.

Final verdict: **new main branch / strong breakthrough**.

## ICRA Framing

This package frames EGBM as **failure-aware online stereo refinement for surgical robotic perception**. The robotics contribution is not a new offline stereo foundation model; it is an online-compatible safety layer for frozen S2M2 predictions that detects when temporal/boundary corrections are helpful and damps them when they would create new geometric failures.

## Key Numbers

- Selected clips MAE: raw `{e['raw_mae']:.4f}` -> EGBM `{e['refined_mae']:.4f}`
- Oracle gap recovered: `{e['oracle_gap_recovered_pct']:.2f}%`
- Pathological new Bad-3: `{p['new_bad3_frame_mean_pct']:.2f}%`
- Clean new Bad-3: `{c['new_bad3_frame_mean_pct']:.2f}%`
- Global frame-mean new Bad-3: `{e['new_bad3_frame_mean_pct']:.2f}%`
- Pixel-weighted new Bad-3: `{e['new_bad3_pixel_weighted_pct']:.2f}%`
- Full-GT test MAE: raw `{t['raw_mae']:.4f}` -> EGBM `{t['refined_mae']:.4f}`
- Full-GT test Bad-3: `{t['raw_bad3']:.3f}%` -> `{t['refined_bad3']:.3f}%`
- Refiner runtime: `{r['fp32_ms_per_frame_batched']:.2f}` ms/frame
- Estimated S2M2+EGBM runtime: `{r['estimated_total_ms']:.2f}` ms/frame under the `{r['system_budget_ms']:.0f}` ms online budget
- Parameters: `{r['params']:,}`

## Contributions

1. A failure-aware online disparity refiner that improves frozen S2M2 without running heavy teachers online.
2. An event-gated boundary/memory formulation that separates helpful hard positives from harmful hard negatives.
3. A safety analysis centered on new Bad-3, modified-pixel behavior, and pathological clip transitions.
4. A runtime result showing accuracy-oriented refinement is feasible within a surgical robotic perception budget.

## Recommended Paper Figures/Tables

- Figure 1: ARGOS pipeline with frozen S2M2 plus EGBM safety refiner.
- Figure 2: Accuracy-safety Pareto plot (`figures/accuracy_safety_pareto.png`).
- Figure 3: Damping hard positives vs hard negatives (`figures/damping_hardpos_hardneg.png`).
- Figure 4: Qualitative pathological before/after panels from `qualitative_plan.md`.
- Table 1: Final comparison table (`latex/final_comparison_table.tex`).
- Table 2: Damping analysis (`latex/damping_analysis_table.tex`).
"""
    outline = """
# Paper Outline

## Title Options

1. Failure-Aware Online Stereo Refinement for Surgical Robotic Perception
2. Event-Gated Boundary-Memory Refinement for Safe Surgical Stereo
3. Online Safety-Gated Refinement of Frozen Stereo Matchers in Robotic Surgery

## Abstract Draft

Accurate depth perception is a core requirement for surgical robotic autonomy, yet high-quality temporal stereo methods are often too expensive for online deployment. We study online refinement of frozen S2M2 disparity predictions using rectified temporal ground truth from SCARED. Prior dense and gated residual refiners improve selected cases but either overcorrect raw-good pixels or fail to generalize. We propose EGBM, an event-gated boundary-memory refiner that learns when to apply temporal/boundary corrections and when to damp them. EGBM recovers 20.37% of the oracle/SAV headroom on selected failure clips, reduces full-GT held-out MAE from 4.6690 to 4.5226 px, and keeps pathological new Bad-3 to 1.30%, while adding 6.25 ms/frame to a 62 ms S2M2 pipeline. These results support failure-aware refinement as a practical path toward safer online stereo perception in robotic surgery.

## Contributions

- A compact online refiner for frozen S2M2 surgical stereo predictions.
- A failure-aware damping mechanism that suppresses harmful corrections on raw-good pixels.
- A selected-clip oracle evaluation measuring how much SAV/oracle headroom is recovered.
- A safety-first evaluation protocol using new Bad-3 and pathological failure modes.

## Section Outline

1. Introduction: robotic stereo needs online accuracy and safety, not only offline best depth.
2. Related Work: surgical stereo, temporal stereo, SAV/TCSM/StereoDiffusion, edge-aware refinement.
3. Dataset and Evaluation: rectified SCARED temporal GT, selected oracle clips, full-GT split.
4. Method: frozen S2M2, EGBM branches, event gate, damping, online runtime.
5. Experiments: baselines, selected oracle headroom, full-GT generalization, runtime.
6. Safety Analysis: new Bad-3, modified pixels, hard positives/negatives, threshold sweep.
7. Limitations: selected clips are small, no live robot deployment yet, no direct SAV online.
8. Conclusion: EGBM is the new main branch for online-compatible surgical stereo refinement.

## Robotics Positioning

The value is deployment behavior: EGBM keeps the S2M2+refiner pipeline below 100 ms/frame and optimizes against new geometric failures, which matters for closed-loop perception more than offline photorealistic depth alone.

## Reviewer Risks and Rebuttal Notes

- Overfitting concern: full-GT held-out test improves raw S2M2, so selected-clip gains are not isolated.
- High modified-pixel rate: new Bad-3 stays below 1% globally and 1.30% on pathological clips.
- Novelty: the paper is not claiming a larger stereo backbone; it contributes failure-aware online refinement and safety evaluation.
"""
    qualitative = """
# Qualitative Figure Plan

Build these next from existing selected clips and EGBM predictions:

1. Pathological temporal flicker panel:
   - raw disparity, EGBM refined, GT, oracle, raw error, refined error, damping map, improvement map.
   - Use the `high_temporal_flicker` clip.

2. Boundary overcorrection panel:
   - focus on tissue/tool boundary where naive correction creates Bad-3.
   - Show EGBM damping suppressing hard negatives.

3. Clean-core safety panel:
   - show that EGBM leaves clean regions mostly unchanged or harmlessly refined.

4. Temporal strip:
   - same row/patch over 8-12 frames showing raw flicker and EGBM stabilization.

5. 3D surface/wound phantom visualization:
   - render point-cloud surfaces before/after for one pathological frame.
   - Keep it as supplementary unless the surface makes the clinical geometry clearer.

Recommended examples:
- `high_temporal_flicker`: demonstrates temporal benefit and weaker but positive damping separation.
- `high_boundary_error`: strongest hard-negative damping separation.
"""
    ablation = """
# Minimal Ablation Plan

No training was run for this package. These are the smallest paper ablations worth running next.

| Ablation | Question | Expected runtime | Risk | Success criterion |
|---|---|---:|---|---|
| No temporal memory | Does memory drive selected gains? | similar or slightly lower | may collapse to v4-like safety-only behavior | selected MAE worsens while safety remains |
| No boundary branch | Is boundary handling responsible for patho safety? | lower | high_boundary_error new Bad-3 increases | worse patho new-Bad3 |
| No dynamic abstention | Is gating necessary? | similar | overcorrection returns | new Bad-3 rises sharply |
| Single residual / no mixture experts | Is specialization useful? | lower | loses oracle-gap recovery | lower gap with similar runtime |
| No damping | Does damping explain safety? | similar | patho new-Bad3 spikes | patho new-Bad3 worsens |
| No hard-negative damping supervision | Are hard negatives explicitly learned? | similar | hard-negative damping separation weakens | hard-neg damping approaches hard-pos damping |

Stop after these. Do not add architecture variants until these answer which component is doing the work.
"""
    risks = """
# Reviewer Risk Notes

## Is this just overfitting selected oracle clips?

Not by the current evidence. EGBM improves the independent full-GT held-out test split from 4.6690 to 4.5226 px MAE and improves Bad-3 from 33.536% to 32.941%. The selected clips show the larger oracle-headroom effect, but full-GT generalization stays positive.

## Why is modifying ~62% of selected pixels safe?

Modification rate alone is the wrong safety metric. EGBM modifies many pixels in high-error regions, but global frame-mean new Bad-3 is 0.96%, pixel-weighted new Bad-3 is 0.21%, and pathological new Bad-3 is 1.30%. The transition analysis shows many more fixed Bad-3 pixels than newly introduced ones.

## How is this robotics and not pure CV?

The system is designed around online constraints and failure safety: S2M2+EGBM is about 68.25 ms/frame, below a 100 ms perception budget, and the evaluation penalizes newly introduced geometric failures that matter for control and scene understanding.

## What is novel versus SAV/TCSM/StereoDiffusion?

Those methods target stronger temporal/deep stereo estimates, often offline or heavier. EGBM is an online-compatible refinement layer for a frozen stereo matcher, trained to decide when corrections are unsafe. % cite SAV here
% cite Temporally Consistent Stereo Matching here
% cite StereoDiffusion MICCAI 2024 here

## Why not use SAV directly online?

SAV is a valuable teacher/oracle candidate, but not the online target in this system. EGBM captures part of that headroom at about 6.25 ms/frame without running SAV during deployment.

## What does online-compatible mean?

The refiner consumes existing S2M2 outputs and local temporal/boundary state, avoids heavy teacher inference, and keeps total estimated runtime under the system budget.
"""
    return {"README.md": readme, "paper_outline.md": outline, "qualitative_plan.md": qualitative, "ablation_plan.md": ablation, "reviewer_risk_notes.md": risks}


def make_latex(summary: dict, comparison: list[dict[str, str]], damping: list[dict[str, str]]) -> dict[str, str]:
    e = summary["selected"]["all"]
    p = summary["selected"]["pathological"]
    c = summary["selected"]["clean"]
    t = summary["full_gt_test"]
    method = r"""
\section{Method: Event-Gated Boundary-Memory Refinement}

We refine frozen S2M2 disparity predictions with an online-compatible event-gated boundary-memory (EGBM) module. The refiner predicts bounded corrections and a damping policy that suppresses corrections when they are likely to turn raw-good pixels into geometric failures. The design targets robotic deployment: expensive teachers such as SAV or RAFT are not run online, and the refiner adds only a small latency overhead to S2M2.

EGBM is trained with full-GT supervision and selected failure clips that expose temporal flicker and boundary overcorrection. Its policy is evaluated with both accuracy metrics and safety metrics, especially new Bad-3 errors introduced from raw-good regions.

% cite edge-aware stereo refinement here
% cite SAV here
% cite Temporally Consistent Stereo Matching here
% cite StereoDiffusion MICCAI 2024 here
"""
    results = f"""
\\section{{Results}}

EGBM improves the selected oracle clips from {e['raw_mae']:.4f} px to {e['refined_mae']:.4f} px MAE and recovers {e['oracle_gap_recovered_pct']:.2f}\\% of the oracle/SAV headroom. On pathological clips, EGBM keeps new Bad-3 to {p['new_bad3_frame_mean_pct']:.2f}\\%, compared with 15.77\\% for the previous v3.2c baseline. On clean clips, new Bad-3 remains {c['new_bad3_frame_mean_pct']:.2f}\\%.

On the held-out full-GT test split, EGBM improves raw S2M2 from {t['raw_mae']:.4f} to {t['refined_mae']:.4f} px MAE and reduces Bad-3 from {t['raw_bad3']:.3f}\\% to {t['refined_bad3']:.3f}\\%. The refiner runs in 6.25 ms/frame, giving an estimated S2M2+EGBM runtime of 68.25 ms/frame under a 100 ms online budget.
"""
    bullets = r"""
\begin{itemize}
  \item We introduce a failure-aware online refiner for frozen S2M2 surgical stereo predictions.
  \item We show that event-gated damping suppresses harmful corrections while preserving oracle-headroom gains.
  \item We evaluate safety with new Bad-3 transitions, not only aggregate MAE.
  \item We demonstrate deployment feasibility at an estimated 68.25 ms/frame for S2M2+EGBM.
\end{itemize}
"""
    abstract = r"""
Accurate depth perception is central to surgical robotic autonomy, but high-quality temporal stereo methods are often too slow for online deployment. We study online refinement of frozen S2M2 disparity predictions using rectified temporal ground truth from SCARED. We propose EGBM, an event-gated boundary-memory refiner that learns when temporal/boundary corrections are helpful and damps corrections that would introduce new geometric failures. EGBM recovers 20.37\% of selected oracle/SAV headroom, reduces held-out full-GT MAE from 4.6690 to 4.5226 px, and keeps pathological new Bad-3 to 1.30\%, while adding 6.25 ms/frame to the pipeline. These results support failure-aware refinement as a practical route to safer online stereo perception in robotic surgery.
"""
    final_table = latex_table(
        numeric_table_rows(comparison),
        [
            ("method", "Method"),
            ("selected_mae", "Selected MAE"),
            ("gap", "Gap \\%"),
            ("patho_new_bad3", "Patho new Bad-3"),
            ("clean_new_bad3", "Clean new Bad-3"),
            ("full_gt_test_mae", "Test MAE"),
            ("runtime", "ms/frame"),
        ],
        "Final refiner comparison. Lower is better for MAE and new Bad-3; higher is better for oracle gap recovered.",
        "tab:egbm-final-comparison",
    )
    damp_rows = []
    for clip in sorted({r["clip_id"] for r in damping}):
        hn = next((r for r in damping if r["clip_id"] == clip and r["group"] == "hard_neg"), None)
        hp = next((r for r in damping if r["clip_id"] == clip and r["group"] == "hard_pos"), None)
        valid = next((r for r in damping if r["clip_id"] == clip and r["group"] == "valid"), None)
        if hn and hp and valid:
            damp_rows.append(
                {
                    "mode": hn["failure_mode"],
                    "valid": fmt(f(valid, "damping_mean"), 3),
                    "hard_neg": fmt(f(hn, "damping_mean"), 3),
                    "hard_pos": fmt(f(hp, "damping_mean"), 3),
                    "sep": fmt(f(hp, "damping_mean") - f(hn, "damping_mean"), 3),
                }
            )
    damping_table = latex_table(
        damp_rows,
        [("mode", "Failure mode"), ("valid", "Valid"), ("hard_neg", "Hard neg."), ("hard_pos", "Hard pos."), ("sep", "Sep.")],
        "EGBM damping is lower on hard negatives than hard positives, especially for boundary failures.",
        "tab:egbm-damping",
    )
    return {
        "egbm_method_section.tex": method,
        "egbm_results_section.tex": results,
        "contribution_bullets.tex": bullets,
        "abstract_draft.tex": abstract,
        "final_comparison_table.tex": final_table,
        "damping_analysis_table.tex": damping_table,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=SRC)
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_root.exists() and not args.overwrite:
        raise SystemExit(f"{args.output_root} exists; pass --overwrite")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "latex").mkdir(exist_ok=True)
    (args.output_root / "figures").mkdir(exist_ok=True)

    summary = json.loads((args.source_root / "aggregate_summary.json").read_text())
    comparison = read_csv(args.source_root / "final_comparison_table.csv")
    damping = read_csv(args.source_root / "damping_analysis.csv")
    threshold = read_csv(args.source_root / "threshold_sweep.csv")
    runtime = json.loads((args.source_root / "runtime_summary.json").read_text())

    shutil.copy2(args.source_root / "final_comparison_table.csv", args.output_root / "final_comparison_table.csv")
    shutil.copy2(args.source_root / "final_comparison_table_latex.tex", args.output_root / "latex" / "source_final_comparison_table.tex")

    for name, text in make_markdown(summary, comparison).items():
        write(args.output_root / name, text)
    for name, text in make_latex(summary, comparison, damping).items():
        write(args.output_root / "latex" / name, text)

    make_figures(args.output_root / "figures", args.source_root, comparison, damping, threshold, runtime)

    log = {
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "inputs": [
            "aggregate_summary.json",
            "final_comparison_table.csv",
            "damping_analysis.csv",
            "threshold_sweep.csv",
            "runtime_summary.json",
        ],
        "no_training": True,
        "no_teacher_inference": True,
        "package_files": sorted(str(p.relative_to(args.output_root)) for p in args.output_root.rglob("*") if p.is_file()),
    }
    write(args.output_root / "run.log", json.dumps(log, indent=2))
    print(f"wrote {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
