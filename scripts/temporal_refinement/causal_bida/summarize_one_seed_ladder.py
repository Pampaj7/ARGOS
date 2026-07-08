#!/usr/bin/env python3
"""Summarise ARGOS v2 one-seed ladder outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ORDER = [
    "raw_s2m2",
    "current_only",
    "aligned_local_faithful",
    "faithful_causal_bida",
    "faithful_causal_bida_state_reset",
    "faithful_causal_bida_shuffled_history",
    "safe_causal_bida",
]


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("")
        return
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    for cfg in ORDER:
        p = args.root / f"{cfg}_seed0" / "aggregate_metrics.json"
        if not p.exists():
            rows.append({"config": cfg, "status": "missing"})
            continue
        d = json.loads(p.read_text())
        row = {"config": cfg, "status": "complete", "params": d.get("params"), **d.get("val", {})}
        rows.append(row)
    raw = next((r for r in rows if r["config"] == "raw_s2m2" and r["status"] == "complete"), None)
    deltas = []
    if raw:
        for r in rows:
            if r["status"] != "complete":
                continue
            deltas.append({
                "config": r["config"],
                "mae_delta_vs_raw": raw["refined_mae"] - r["refined_mae"],
                "bad3_delta_vs_raw": raw["refined_bad3"] - r["refined_bad3"],
                "new_bad3": r.get("new_bad3"),
                "modified_pixel_ratio": r.get("modified_pixel_ratio"),
            })
    by = {r["config"]: r for r in rows if r["status"] == "complete"}
    gates = {
        "alignment_gate": by.get("aligned_local_faithful", {}).get("refined_mae", float("inf")) < by.get("current_only", {}).get("refined_mae", -float("inf")),
        "propagation_gate": by.get("faithful_causal_bida", {}).get("refined_mae", float("inf")) < by.get("aligned_local_faithful", {}).get("refined_mae", -float("inf")),
        "state_gate": by.get("faithful_causal_bida", {}).get("refined_mae", float("inf")) < by.get("faithful_causal_bida_state_reset", {}).get("refined_mae", -float("inf")),
        "history_gate": by.get("faithful_causal_bida", {}).get("refined_mae", float("inf")) < by.get("faithful_causal_bida_shuffled_history", {}).get("refined_mae", -float("inf")),
        "safety_gate": by.get("safe_causal_bida", {}).get("new_bad3", float("inf")) <= by.get("faithful_causal_bida", {}).get("new_bad3", -float("inf")),
    }
    complete = all(by.get(c) for c in ORDER)
    if not complete:
        classification = "INVALID_OR_INCOMPLETE"
    elif gates["alignment_gate"] and gates["propagation_gate"] and gates["state_gate"] and gates["history_gate"]:
        classification = "ALIGNMENT_AND_PROPAGATION_CONFIRMED"
    elif gates["alignment_gate"]:
        classification = "ALIGNMENT_CONFIRMED_PROPAGATION_NOT_CONFIRMED"
    else:
        classification = "NO_HISTORY_SIGNAL"
    args.root.mkdir(parents=True, exist_ok=True)
    write_csv(args.root / "aggregate_metrics.csv", rows)
    write_csv(args.root / "comparison_deltas.csv", deltas)
    (args.root / "decision_gate.json").write_text(json.dumps({"classification": classification, "gates": gates, "complete": complete}, indent=2) + "\n")
    (args.root / "findings.md").write_text("# ARGOS v2 One-Seed Ladder Findings\n\n" + json.dumps({"classification": classification, "gates": gates}, indent=2) + "\n")
    (args.root / "limitations.md").write_text("# ARGOS v2 One-Seed Ladder Limitations\n\nOne seed only. No D4D/SERV-CT. Checkpoints selected on validation MAE.\n")
    (args.root / "README.md").write_text("# ARGOS v2 One-Seed Ladder\n\nSee `aggregate_metrics.csv`, `comparison_deltas.csv`, and `decision_gate.json`.\n")
    print(json.dumps({"classification": classification, "gates": gates, "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
