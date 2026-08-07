#!/usr/bin/env python3
"""Compact, count-weighted post-hoc aggregation of the D2 temporal sidecar."""
from __future__ import annotations
import argparse, csv, math
from collections import defaultdict
from pathlib import Path

EXP = Path("/dtu/p1/leopam/ARGOS/ARGOS_FREEZED/experiments/04_cross_dataset_scaling")
INPUT = EXP / "scared_geometry_temporal/frame_metrics.csv"
OUTPUT = EXP / "metrics"
METHODS = ("raw", "h4", "immutable")
METRICS = (("epe", "epe_count"), ("gt_tce", "gt_tce_count"), ("rgt_tce", "gt_tce_count"), ("nr_tce", "nr_tce_count"), ("stereo_photo", "stereo_photo_count"), ("temporal_photo", "temporal_photo_count"))


def number(value: str) -> float:
    try: return float(value)
    except (TypeError, ValueError): return float("nan")


def weighted(rows: list[dict], value: str, count: str) -> float | None:
    values = [(number(row[value]), number(row[count])) for row in rows]
    values = [(v, n) for v, n in values if math.isfinite(v) and n > 0]
    return sum(v * n for v, n in values) / sum(n for _, n in values) if values else None


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle: return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def aggregate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    geometry = [row for row in rows if int(row["age_frames"]) == 0]
    temporal = [row for row in rows if int(row["age_frames"]) in (1, 2, 4, 8)]
    geo_by_frame = {(r["backbone"], r["sequence"], r["frame_id"], r["method"]): number(r["epe"]) for r in geometry}
    scopes = sorted({r["backbone"] for r in rows}) + ["ALL"]
    result, pareto = [], []
    for scope in scopes:
        pick = lambda values: values if scope == "ALL" else [r for r in values if r["backbone"] == scope]
        geo, temp = pick(geometry), pick(temporal)
        raw_epe = weighted([r for r in geo if r["method"] == "raw"], "epe", "epe_count")
        for method in METHODS:
            method_geo = [r for r in geo if r["method"] == method]
            frame_deltas = []
            sequence = defaultdict(lambda: defaultdict(list))
            for row in method_geo:
                key = (row["backbone"], row["sequence"], row["frame_id"]); raw = geo_by_frame.get((*key, "raw")); current = number(row["epe"])
                if raw is not None and math.isfinite(current): frame_deltas.append(raw - current); sequence[key[:2]][method].append(current); sequence[key[:2]]["raw"].append(raw)
            win = [v > 0 for v in frame_deltas]
            seq_win = [sum(values[method]) / len(values[method]) < sum(values["raw"]) / len(values["raw"]) for values in sequence.values() if method != "raw" and values[method] and values["raw"]]
            for age in (0, 1, 2, 4, 8):
                method_temp = [r for r in temp if r["method"] == method and int(r["age_frames"]) == age]
                row = {"scope": scope, "backbone": scope, "method": method, "age_frames": age,
                       "epe": weighted(method_geo, "epe", "epe_count"), "gain_vs_raw_epe": (raw_epe - weighted(method_geo, "epe", "epe_count")) if raw_epe is not None and weighted(method_geo, "epe", "epe_count") is not None else None,
                       "frame_win_rate_vs_raw": (sum(win) / len(win)) if method != "raw" and win else (0.0 if method == "raw" else None),
                       "sequence_win_rate_vs_raw": (sum(seq_win) / len(seq_win)) if method != "raw" and seq_win else (0.0 if method == "raw" else None),
                       "worst_frame_delta_vs_raw": min(frame_deltas) if method != "raw" and frame_deltas else (0.0 if method == "raw" else None),
                       "track_jitter": "N/A", "clean_pixel_degradation": "N/A", "update_magnitude": "N/A",
                       "na_reason": "prediction tensors/raw-good masks/long-track support are not saved by the compact sidecar"}
                for value, count in METRICS[1:]: row[value] = weighted(method_temp, value, count)
                result.append(row)
        for age in (1, 2, 4, 8):
            raw = next((r for r in result if r["scope"] == scope and r["method"] == "raw" and r["age_frames"] == age), None)
            if not raw: continue
            for method in ("h4", "immutable"):
                current = next((r for r in result if r["scope"] == scope and r["method"] == method and r["age_frames"] == age), None)
                if current:
                    for metric in ("gt_tce", "rgt_tce", "nr_tce"):
                        if raw[metric] is not None and current[metric] is not None:
                            pareto.append({"scope": scope, "backbone": scope, "method": method, "age_frames": age, "temporal_metric": metric, "geometry_gain_epe": current["gain_vs_raw_epe"], "temporal_gain": raw[metric] - current[metric]})
    return result, pareto


def svg(path: Path, rows: list[dict]) -> None:
    points = [r for r in rows if r["temporal_metric"] == "gt_tce" and r["geometry_gain_epe"] is not None]
    width, height, margin = 640, 420, 48
    xs = [r["geometry_gain_epe"] for r in points] or [0.0]; ys = [r["temporal_gain"] for r in points] or [0.0]
    lo_x, hi_x = min(xs + [0.0]), max(xs + [0.0]); lo_y, hi_y = min(ys + [0.0]), max(ys + [0.0])
    scale = lambda v, lo, hi, size: margin + (v - lo) / (hi - lo or 1.0) * size
    circles = "\n".join(f'<circle cx="{scale(r["geometry_gain_epe"],lo_x,hi_x,width-2*margin):.1f}" cy="{height-scale(r["temporal_gain"],lo_y,hi_y,height-2*margin):.1f}" r="4"><title>{r["scope"]} {r["method"]} CS{r["age_frames"]}</title></circle>' for r in points)
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><style>text{{font:12px sans-serif}}circle{{fill:#2563eb}}</style><line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="black"/><line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="black"/><text x="{margin}" y="20">GT-TCE gain vs geometry gain (D2)</text><text x="{width/2-70}" y="{height-10}">EPE gain vs raw</text><text x="4" y="{height/2}">GT-TCE gain</text>{circles}</svg>\n')


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--input", type=Path, default=INPUT); parser.add_argument("--output", type=Path, default=OUTPUT); args = parser.parse_args()
    if not args.input.is_file(): raise FileNotFoundError(args.input)
    args.output.mkdir(parents=True, exist_ok=True); summary, pareto = aggregate(read_rows(args.input)); write_csv(args.output / "temporal_aggregate.csv", summary); write_csv(args.output / "pareto_geometry_temporal.csv", pareto); svg(args.output / "pareto_geometry_temporal.svg", pareto)
    (args.output / "AGGREGATION_README.md").write_text("# D2 temporal aggregation\n\nMeans are weighted by the metric-specific valid count from strict-existing support. Geometry uses age-0 rows; temporal rows use CS1/2/4/8 only. Track jitter, clean-pixel degradation, and update magnitude are N/A because compact frame CSVs do not retain prediction tensors, raw-good masks, or long tracks.\n")


if __name__ == "__main__": main()
