#!/usr/bin/env python3
"""Two questions Table I's caption cannot answer by itself.

(1) Where does seed 0 fall among the three, ARENA BY ARENA? The closure caption
    says seed 0 is the weakest of its three, but that is measured on D2 EPE, and
    Table I shows D7 and DRENDS. Sec. V-J already records that the seed's rank
    changes with the configuration -- seed 0 is the low tail for the shipped head
    and the high tail for the 142-channel base -- so a rank measured on one split
    is not evidence about another. Writing "seed 0 understates" in the caption of
    the main table without checking every arena it covers would be a guess.

(2) Does the SIGN of every cell survive the seed? This is the stronger answer to
    the reviewer, and it does not depend on (1) at all: if all cells improve at
    seeds 1 and 2 as well, contribution (i) is seed-invariant in its claim, and
    where seed 0 happens to fall stops mattering.

Reads finished reports only. A cell is (arena, backbone), pooled over that
backbone's sequences or recordings the way Table I pools them: macro over
sequences, ratio taken after pooling.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT.parent / "results"
METRICS = ("EPE", "Bad1")

# The shipped 38-channel head at the three pre-registered seeds. DRENDS carries
# three seeds on three backbones only; the two missing cells are reported as such
# rather than silently dropped from the count.
RUNS = {
    "D2": {0: RESULTS / "ablation_eval/a2/runs/scared-d2",
           1: RESULTS / "ablation_eval/a2_seed1/runs/scared-d2",
           2: RESULTS / "ablation_eval/a2_seed2/runs/scared-d2"},
    "D7": {0: RESULTS / "ablation_eval/a2/runs/scared-d7",
           1: RESULTS / "ablation_eval/a2_seed1/runs/scared-d7",
           2: RESULTS / "ablation_eval/a2_seed2/runs/scared-d7"},
    "DRENDS": {0: RESULTS / "drends_a2/seed0",
               1: RESULTS / "drends_a2/seed1",
               2: RESULTS / "drends_a2/seed2"},
}


def cells(root: Path, metric: str) -> dict[str, dict[str, tuple[float, float]]]:
    """{backbone: {unit: (raw, refined)}} from every report below `root`."""
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for report in sorted(root.rglob("reports/*/*.json")):
        aggregate = json.loads(report.read_text())["aggregate"]["disparity_px"]
        if metric not in aggregate.get("raw", {}) or metric not in aggregate.get("refined", {}):
            continue
        out.setdefault(report.parent.name, {})[report.stem] = (
            aggregate["raw"][metric]["macro_sequence"],
            aggregate["refined"][metric]["macro_sequence"])
    return out


def reduction(pairs: dict[str, tuple[float, float]]) -> float:
    raw = sum(p[0] for p in pairs.values()) / len(pairs)
    refined = sum(p[1] for p in pairs.values()) / len(pairs)
    return 100.0 * (raw - refined) / raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=RESULTS / "seed_sign_invariance")
    args = parser.parse_args()

    record: dict = {"cells": [], "arena_rank": [], "unit_level": []}
    for metric in METRICS:
        for arena, seeds in RUNS.items():
            per_seed = {s: cells(p, metric) for s, p in seeds.items() if p.exists()}
            common = sorted(set.intersection(*(set(c) for c in per_seed.values())))
            missing = sorted(set(per_seed[0]) - set(common))

            for backbone in common:
                units = set.intersection(*(set(per_seed[s][backbone]) for s in per_seed))
                values = {s: reduction({u: per_seed[s][backbone][u] for u in units})
                          for s in sorted(per_seed)}
                record["cells"].append({
                    "metric": metric, "arena": arena, "backbone": backbone,
                    "units": len(units),
                    **{f"seed{s}": values[s] for s in values},
                    "improves_every_seed": all(v > 0 for v in values.values()),
                    "seed0_rank_of_3": sorted(values.values()).index(values[0]) + 1,
                })
                # Sequence level, where a sign can hide inside a pooled cell.
                for unit in sorted(units):
                    per_unit = {s: reduction({unit: per_seed[s][backbone][unit]})
                                for s in sorted(per_seed)}
                    record["unit_level"].append({
                        "metric": metric, "arena": arena, "backbone": backbone, "unit": unit,
                        **{f"seed{s}": per_unit[s] for s in per_unit},
                        "improves_every_seed": all(v > 0 for v in per_unit.values()),
                    })

            # Arena pooled over its backbones, which is what a caption would claim.
            pooled = {}
            for s in sorted(per_seed):
                merged = {f"{b}/{u}": v for b in common
                          for u, v in per_seed[s][b].items()}
                pooled[s] = reduction(merged)
            record["arena_rank"].append({
                "metric": metric, "arena": arena, "backbones": len(common),
                "backbones_seed0_only": missing,
                **{f"seed{s}": pooled[s] for s in pooled},
                "seed0_rank_of_3": sorted(pooled.values()).index(pooled[0]) + 1,
                "seed0_understates": pooled[0] == min(pooled.values()),
            })

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "seed_sign_invariance.json").write_text(json.dumps(record, indent=2) + "\n")

    print("ARENA POOLED -- where seed 0 falls (relative reduction %, higher is better)")
    for r in record["arena_rank"]:
        rank = {1: "LOWEST", 2: "middle", 3: "HIGHEST"}[r["seed0_rank_of_3"]]
        note = f"  [{len(r['backbones_seed0_only'])} backbone(s) seed-0 only]" if r["backbones_seed0_only"] else ""
        print(f"  {r['metric']:<5} {r['arena']:<7} {r['backbones']}bb  "
              f"{r['seed0']:6.2f} {r['seed1']:6.2f} {r['seed2']:6.2f}   seed0 {rank}{note}")

    print("\nCELL SIGN INVARIANCE (arena x backbone, pooled)")
    for metric in METRICS:
        rows = [c for c in record["cells"] if c["metric"] == metric]
        bad = [c for c in rows if not c["improves_every_seed"]]
        print(f"  {metric:<5} {len(rows) - len(bad)}/{len(rows)} cells improve at ALL THREE seeds")
        for c in bad:
            print(f"    NOT INVARIANT: {c['arena']}/{c['backbone']} "
                  f"{c['seed0']:.2f} {c['seed1']:.2f} {c['seed2']:.2f}")

    print("\nSEQUENCE/RECORDING SIGN INVARIANCE")
    for metric in METRICS:
        rows = [u for u in record["unit_level"] if u["metric"] == metric]
        bad = [u for u in rows if not u["improves_every_seed"]]
        print(f"  {metric:<5} {len(rows) - len(bad)}/{len(rows)} units improve at ALL THREE seeds")
        for u in bad:
            print(f"    exception: {u['arena']}/{u['backbone']}/{u['unit']} "
                  f"{u['seed0']:.2f} {u['seed1']:.2f} {u['seed2']:.2f}")


if __name__ == "__main__":
    main()
