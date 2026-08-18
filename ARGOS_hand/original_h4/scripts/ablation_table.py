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
# The canonical model has three seeds. Comparing an ablation against seed 0 alone is
# what we did first and it was misleading: seed 0 is the WEAKEST of the three on held-out
# D7, so every variant looked better than it really is. The reference is the three-seed
# mean, and the spread is what decides whether a difference means anything.
CANONICAL_SEEDS = {
    "seed 0": RESULTS / "scared_masked",
    "seed 1": RESULTS / "seed_eval/seed_1",
    "seed 2": RESULTS / "seed_eval/seed_2",
}
CANONICAL = RESULTS / "scared_masked"
# A2 was given the same three-seed treatment as the canonical model, under a rule written
# before the extra seeds were trained. Reading it as a single run -- which this script did
# until the seeds were noticed sitting unused on disk -- reproduces exactly the mistake the
# comment above records, with the roles reversed.
VARIANT_SEEDS = {"a2": ("a2", "a2_seed1", "a2_seed2")}
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
    seeds = {name: {(s, m): reduction(root, s, m)
                    for s in ("scared-d2", "scared-d7") for m in METRICS}
             for name, root in CANONICAL_SEEDS.items() if (root / "runs").is_dir()}
    rows["canonical (3 seeds)"] = {
        key: float(np.mean([v[key] for v in seeds.values() if v[key] is not None]))
        for key in seeds["seed 0"]}
    spread = {key: float(np.std([v[key] for v in seeds.values() if v[key] is not None], ddof=1))
              for key in seeds["seed 0"]}
    variant_spread = {}
    for key in VARIANTS:
        runs = [args.eval_root / name for name in VARIANT_SEEDS.get(key, (key,))]
        per_seed = [{(s, m): reduction(root, s, m) for s in ("scared-d2", "scared-d7") for m in METRICS}
                    for root in runs if root.is_dir()]
        per_seed = [v for v in per_seed if v[("scared-d7", "Bad1")] is not None]
        if not per_seed:
            missing.append(key)
            continue
        rows[key] = {k: float(np.mean([v[k] for v in per_seed])) for k in per_seed[0]}
        if len(per_seed) > 1:
            variant_spread[key] = ({k: float(np.std([v[k] for v in per_seed], ddof=1)) for k in per_seed[0]},
                                   len(per_seed), per_seed)

    print(f"{'variant':<26}" + "".join(f"{s.split('-')[1].upper()} {m:<6}" for s in ("scared-d2", "scared-d7") for m in METRICS))
    for key, value in rows.items():
        label = key if key.startswith("canonical") else f"{VARIANTS[key][0]} {VARIANTS[key][1]}"
        print(f"{label:<26}" + "".join(
            f"{value[(s, m)]:>9.2f}" if value[(s, m)] is not None else f"{'--':>9}"
            for s in ("scared-d2", "scared-d7") for m in METRICS))
    if missing:
        print(f"\nnot yet landed: {', '.join(missing)}")
        if args.require_all:
            raise SystemExit(1)

    base = rows["canonical (3 seeds)"][("scared-d7", "Bad1")]
    sigma = spread[("scared-d7", "Bad1")]
    values = [v[("scared-d7", "Bad1")] for v in seeds.values()]
    print(f"\nheld-out Bad1: canonical seeds {', '.join(f'{v:.2f}' for v in sorted(values))} "
          f"-> mean {base:.2f}, std {sigma:.2f}")
    print("a margin inside the seed range is not a result:")
    for key in VARIANTS:
        if key not in rows:
            continue
        value = rows[key][("scared-d7", "Bad1")]
        delta = value - base
        inside = min(values) <= value <= max(values)
        count = variant_spread[key][1] if key in variant_spread else 1
        verdict = ("indistinguishable" if abs(delta) < SEED_SPREAD_PP
                   else "inside the canonical seed range" if inside
                   else f"outside it, {delta / sigma:+.1f} sigma, "
                        f"{f'{count} seeds' if count > 1 else 'one seed'}")
        print(f"  {VARIANTS[key][0]}  {value:.2f}  ({delta:+.2f} pp)  -> {verdict}")

    # The promotion rule, quoted from ablation_preregister.json and evaluated here so that
    # it cannot be re-read favourably after the fact. The decision split is D2 alone; D7 is
    # reported afterwards and takes no part in the choice.
    for key, (sd, count, per_seed) in variant_spread.items():
        canonical_d2 = rows["canonical (3 seeds)"][("scared-d2", "Bad1")]
        value = rows[key][("scared-d2", "Bad1")]
        each = sorted(v[("scared-d2", "Bad1")] for v in per_seed)
        print(f"\npre-registered promotion rule for {VARIANTS[key][0]} -- decision split: D2 ONLY")
        print(f"  D2 Bad1 seeds {', '.join(f'{v:.2f}' for v in each)} -> mean {value:.2f}, "
              f"std {sd[('scared-d2', 'Bad1')]:.2f} over {count} seeds")
        print(f"  canonical three-seed mean {canonical_d2:.2f}, spread {spread[('scared-d2', 'Bad1')]:.2f}")
        exceeds = value - canonical_d2
        comparable = sd[("scared-d2", "Bad1")] <= 2 * spread[("scared-d2", "Bad1")]
        print(f"  exceeds by {exceeds:+.2f} pp; spread {'comparable' if comparable else 'NOT comparable'}")
        print(f"  -> {'PROMOTE' if exceeds > 0 and comparable else 'stays an ablation'}")

    print("\n% ---- LaTeX ----")
    print("\\begin{tabular}{llcccc}")
    print("\\toprule")
    print("& & \\multicolumn{2}{c}{D2 dev.} & \\multicolumn{2}{c}{D7 held out} \\\\")
    print("\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}")
    print("Variant & Change & EPE & Bad1 & EPE & Bad1 \\\\")
    print("\\midrule")
    c = rows["canonical (3 seeds)"]
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
