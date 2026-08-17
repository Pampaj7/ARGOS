#!/usr/bin/env python3
"""Compare A2 against the canonical model on DRENDS, per seed and per backbone.

DRENDS is the out-of-domain split, so this is the axis the paper's transfer claim
actually rests on. The comparison is deliberately awkward: A2 has three seeds here and
the canonical model has only one (`drends_masked`, seed 0), because the seed study was
run on SCARED and never extended to DRENDS.

Reporting a three-seed A2 mean against a one-seed canonical number is the same mistake
we already made once on the ablations, where seed 0 happened to be the weakest of the
three and flattered every variant. So the headline here is the SEED-0 vs SEED-0 pairing,
which is matched, and the A2 seed spread is printed beside it as context rather than
folded into the comparison.

Both sides are checked to share the same raw predictions before any delta is believed:
if the frozen backbone output differed, the comparison would be meaningless.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT.parent / "results"
CANONICAL = RESULTS / "drends_masked"          # seed 0 only; no DRENDS seed study exists
A2 = RESULTS / "drends_a2"
BACKBONES = ("CREStereo", "Fast-FoundationStereo", "RAFT-Stereo")
METRICS = ("EPE", "Bad1", "Bad3", "RMSE")


def sequences(root: Path, backbone: str) -> dict[str, dict]:
    """Every scored sequence of one backbone, keyed by sequence name."""
    out = {}
    for path in glob.glob(str(root / backbone / "reports/*/*.json")):
        out[Path(path).stem] = json.loads(Path(path).read_text())["aggregate"]["disparity_px"]
    return out


def macro(reports: dict[str, dict], family: str, metric: str) -> float | None:
    values = [r[family][metric]["macro_sequence"] for r in reports.values() if metric in r[family]]
    return float(np.mean(values)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metric", default="Bad1")
    args = parser.parse_args()

    seeds = sorted(p.name for p in A2.glob("seed*") if p.is_dir())
    print(f"canonical: {CANONICAL.name} (seed 0 only)   A2 seeds present: {', '.join(seeds)}\n")

    rows = {}
    for backbone in BACKBONES:
        base = sequences(CANONICAL, backbone)
        if not base:
            continue
        entry = {"canonical": macro(base, "refined", args.metric), "raw": macro(base, "raw", args.metric)}
        for seed in seeds:
            variant = sequences(A2 / seed, backbone)
            if not variant:
                continue
            # The raw predictions come from the same frozen backbone on the same frames, so
            # they must agree exactly. A mismatch means the two runs are not comparable.
            shared = sorted(set(base) & set(variant))
            a = np.array([base[s]["raw"][args.metric]["macro_sequence"] for s in shared])
            b = np.array([variant[s]["raw"][args.metric]["macro_sequence"] for s in shared])
            if not np.allclose(a, b, atol=1e-6):
                raise RuntimeError(f"{backbone}/{seed}: raw predictions differ from canonical; "
                                   "the comparison would be meaningless")
            entry[seed] = macro(variant, "refined", args.metric)
            entry[f"{seed}_n"] = len(shared)
        rows[backbone] = entry

    header = f"{'backbone':<24}{'raw':>9}{'canon(s0)':>11}" + "".join(f"{s:>9}" for s in seeds)
    print(header)
    for backbone, entry in rows.items():
        line = f"{backbone:<24}{entry['raw']:>9.3f}{entry['canonical']:>11.3f}"
        line += "".join(f"{entry[s]:>9.3f}" if entry.get(s) is not None else f"{'--':>9}" for s in seeds)
        print(line)

    print(f"\nmatched comparison, seed 0 vs seed 0 ({args.metric}, lower is better):")
    for backbone, entry in rows.items():
        if entry.get("seed0") is None:
            continue
        delta = entry["seed0"] - entry["canonical"]
        print(f"  {backbone:<24} canonical {entry['canonical']:.3f}  A2 {entry['seed0']:.3f}  "
              f"({delta:+.3f} {'A2 better' if delta < 0 else 'canonical better'})")

    landed = [s for s in seeds if all(rows[b].get(s) is not None for b in rows)]
    if len(landed) > 1:
        print(f"\nA2 seed spread over {len(landed)} complete seeds (context, not a comparison):")
        for backbone, entry in rows.items():
            values = [entry[s] for s in landed]
            print(f"  {backbone:<24} {', '.join(f'{v:.3f}' for v in values)}  "
                  f"-> mean {np.mean(values):.3f}, std {np.std(values, ddof=1):.3f}")
        print("\nThe canonical model has no DRENDS seed study, so a three-seed A2 mean has no\n"
              "matched counterpart. Either run canonical seeds 1-2 on DRENDS, or report the\n"
              "seed-0 pairing above and say so.")


if __name__ == "__main__":
    main()
