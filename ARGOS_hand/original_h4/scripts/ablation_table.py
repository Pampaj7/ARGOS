#!/usr/bin/env python3
"""Emit the ablation table from finished evaluation runs, as LaTeX and as text.

Hand-copying twenty numbers out of JSON into a table is how a paper acquires a
transcription error that nobody finds. This reads the same reports the tables elsewhere
read and prints the row exactly as it should appear.

The pre-registered reporting rule is applied here rather than left to the writer: a
variant whose held-out Bad1 differs from canonical by less than the three-seed spread is
marked indistinguishable, not better or worse.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT.parent / "results"
PREREGISTER = ROOT / "model_design/ablation_preregister.json"
CANONICAL = RESULTS / "scared_masked"
VARIANTS = {
    "a1": ("A1", "no appearance channels", "$142\\rightarrow78$ ch"),
    "a2": ("A2", "no learned evidence", "$142\\rightarrow38$ ch"),
    "a3": ("A3", "single resolution", "same $177$k"),
    "a4": ("A4", "relaxed convexity", "$+49$ par."),
}
METRICS = ("EPE", "Bad1", "Bad3", "RMSE")
# The three-seed pooled Bad1 spread, from the seed study. Differences below this are
# not interpreted, as declared before any ablation was launched.
SEED_SPREAD_PP = 0.37


def reduction(root: Path, split: str, metric: str) -> float | None:
    """Relative reduction (%) of the metric on the split, macro-sequence over all runs."""
    raw, refined = [], []
    for path in glob.glob(str(root / "**/reports/*/*.json"), recursive=True):
        if f"/{split}/" not in path:
            continue
        family = json.loads(Path(path).read_text())["aggregate"]["disparity_px"]
        if metric not in family["raw"]:
            continue
        raw.append(family["raw"][metric]["macro_sequence"])
        refined.append(family["refined"][metric]["macro_sequence"])
    if not raw:
        return None
    a, b = float(np.mean(raw)), float(np.mean(refined))
    return 100.0 * (a - b) / a


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-root", type=Path, default=RESULTS / "ablation_eval")
    parser.add_argument("--require-all", action="store_true",
                        help="exit non-zero unless every pre-registered variant has landed")
    args = parser.parse_args()

    rows, missing = {}, []
    rows["canonical"] = {(s, m): reduction(CANONICAL, s, m)
                         for s in ("scared-d2", "scared-d7") for m in METRICS}
    for key in VARIANTS:
        root = args.eval_root / key
        value = {(s, m): reduction(root, s, m) for s in ("scared-d2", "scared-d7") for m in METRICS}
        if value[("scared-d7", "Bad1")] is None:
            missing.append(key)
        else:
            rows[key] = value

    print(f"{'variant':<26}" + "".join(f"{s.split('-')[1].upper()} {m:<6}" for s in ("scared-d2", "scared-d7") for m in METRICS))
    for key, value in rows.items():
        label = "canonical" if key == "canonical" else f"{VARIANTS[key][0]} {VARIANTS[key][1]}"
        print(f"{label:<26}" + "".join(
            f"{value[(s, m)]:>9.2f}" if value[(s, m)] is not None else f"{'--':>9}"
            for s in ("scared-d2", "scared-d7") for m in METRICS))
    if missing:
        print(f"\nnot yet landed: {', '.join(missing)}")
        if args.require_all:
            raise SystemExit(1)

    base = rows["canonical"][("scared-d7", "Bad1")]
    print(f"\nheld-out Bad1 verdicts (three-seed spread {SEED_SPREAD_PP} pp):")
    for key in VARIANTS:
        if key not in rows:
            continue
        delta = rows[key][("scared-d7", "Bad1")] - base
        verdict = ("indistinguishable" if abs(delta) < SEED_SPREAD_PP
                   else "better" if delta > 0 else "worse")
        print(f"  {VARIANTS[key][0]}  {delta:+.2f} pp  -> {verdict}")

    print("\n% ---- LaTeX ----")
    print("\\begin{tabular}{llcccc}")
    print("\\toprule")
    print("& & \\multicolumn{2}{c}{D2 dev.} & \\multicolumn{2}{c}{D7 held out} \\\\")
    print("\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}")
    print("Variant & Change & EPE & Bad1 & EPE & Bad1 \\\\")
    print("\\midrule")
    c = rows["canonical"]
    print(f"Canonical & --- & ${c[('scared-d2','EPE')]:.2f}$ & ${c[('scared-d2','Bad1')]:.2f}$ "
          f"& $\\mathbf{{{c[('scared-d7','EPE')]:.2f}}}$ & ${c[('scared-d7','Bad1')]:.2f}$ \\\\")
    for key in VARIANTS:
        if key not in rows:
            continue
        v = rows[key]
        name, _, change = VARIANTS[key]
        print(f"{name} & {change} & ${v[('scared-d2','EPE')]:.2f}$ & ${v[('scared-d2','Bad1')]:.2f}$ "
              f"& ${v[('scared-d7','EPE')]:.2f}$ & ${v[('scared-d7','Bad1')]:.2f}$ \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


if __name__ == "__main__":
    main()
