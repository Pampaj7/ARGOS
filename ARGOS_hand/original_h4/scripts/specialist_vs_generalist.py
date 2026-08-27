#!/usr/bin/env python3
"""Specialist against the shipped generalist, on the cells the declaration names.

Both sides are seed 0 and both are read from finished definitive-evaluation reports
under the same protocol, so the only difference between the two columns is which
training set produced the weights.

The reading rule is the declaration's, not a fresh one: the shipped head's three-seed
D7 Bad1 spread is 0.22 points and its registered threshold is 0.37, so a gap below
0.37 is not read at all and a gap above it is a bound pending seeds, since these arms
are single-seed. D2 is reported because it is the split the specialist selected its
own checkpoint on -- the arena where it is most flattered -- and hiding that would be
the selection optimism this project keeps finding in other people's numbers.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "results"
GENERALIST = RESULTS / "ablation_eval/a2"
SPECIALISTS = RESULTS / "specialist_eval"
THRESHOLD = 0.37


def reductions(root: Path, backbone: str) -> dict[str, tuple[float, float, int]]:
    """{split: (EPE reduction %, Bad1 reduction %, cells)} for one backbone."""
    per_split: dict[str, dict] = {}
    for path in glob.glob(f"{root}/runs/*/*/reports/{backbone}/*.json"):
        split = path.split("/runs/")[1].split("/")[0]
        agg = json.loads(Path(path).read_text())["aggregate"]["disparity_px"]
        per_split.setdefault(split, {})[Path(path).stem] = (
            agg["raw"]["EPE"]["macro_sequence"], agg["refined"]["EPE"]["macro_sequence"],
            agg["raw"]["Bad1"]["macro_sequence"], agg["refined"]["Bad1"]["macro_sequence"])
    out = {}
    for split, cells in per_split.items():
        n = len(cells)
        cols = [sum(v[i] for v in cells.values()) / n for i in range(4)]
        out[split] = (100 * (cols[0] - cols[1]) / cols[0],
                      100 * (cols[2] - cols[3]) / cols[2], n)
    return out


def main() -> None:
    arms = sorted(p.name for p in SPECIALISTS.iterdir()
                  if p.is_dir() and (p / "definitive_table.csv").is_file())
    if not arms:
        print(f"no finished specialist evaluations under {SPECIALISTS}")
        return
    print(f"{len(arms)} arm(s) finished. Reduction %, higher is better; "
          f"gaps below {THRESHOLD} points are not read.\n")
    header = f"{'backbone':<22} {'split':<10} {'general':>9} {'special':>9} {'gap':>8}  {'':<6}"
    for metric, index in (("EPE", 0), ("Bad1", 1)):
        print(f"--- {metric} ---")
        print(header)
        for backbone in arms:
            spec = reductions(SPECIALISTS / backbone, backbone)
            gen = reductions(GENERALIST, backbone)
            for split in ("scared-d2", "scared-d7"):
                if split not in spec or split not in gen:
                    continue
                g, s = gen[split][index], spec[split][index]
                gap = s - g
                verdict = "--" if abs(gap) < THRESHOLD else ("SPEC" if gap > 0 else "GEN")
                print(f"{backbone:<22} {split:<10} {g:9.2f} {s:9.2f} {gap:+8.2f}  {verdict:<6}")
        print()
    print("SPEC/GEN mark which side is ahead by more than the threshold; -- means the\n"
          "difference is inside the spread the registration says not to read.")


if __name__ == "__main__":
    main()
