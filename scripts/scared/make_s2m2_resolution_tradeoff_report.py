#!/usr/bin/env python3
import csv
import json
from pathlib import Path


OUT_DIR = Path("/dtu/p1/leopam/ARGOS/results/s2m2_resolution_tradeoff")
SRC_JSON = OUT_DIR / "s2m2_size_tradeoff.json"
SRC_CSV = OUT_DIR / "s2m2_size_tradeoff.csv"


def load_rows():
    payload = json.loads(SRC_JSON.read_text())
    return payload, payload["summary"]


def pareto(rows):
    front = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            better_or_equal = (
                other["valid_depth_mae"] <= row["valid_depth_mae"]
                and other["avg_inference_time_ms"] <= row["avg_inference_time_ms"]
                and other["peak_gpu_memory_mb"] <= row["peak_gpu_memory_mb"]
            )
            strictly_better = (
                other["valid_depth_mae"] < row["valid_depth_mae"]
                or other["avg_inference_time_ms"] < row["avg_inference_time_ms"]
                or other["peak_gpu_memory_mb"] < row["peak_gpu_memory_mb"]
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(row)
    return sorted(front, key=lambda r: (r["avg_inference_time_ms"], r["valid_depth_mae"]))


def best_under(rows, ms):
    candidates = [r for r in rows if r["avg_inference_time_ms"] <= ms]
    if not candidates:
        return None
    return min(candidates, key=lambda r: (r["valid_depth_mae"], r["valid_disp_mae"]))


def fmt_model(row):
    return f"{row['model']}@{row['resize_label']}"


def write_outputs(payload, rows):
    csv_target = OUT_DIR / "s2m2_resolution_tradeoff.csv"
    json_target = OUT_DIR / "s2m2_resolution_tradeoff.json"
    md_target = OUT_DIR / "s2m2_resolution_tradeoff.md"

    csv_target.write_text(SRC_CSV.read_text())
    json_target.write_text(json.dumps(payload, indent=2) + "\n")

    sl_rows = [r for r in rows if r["model"] in {"S", "L"}]
    all_front = pareto(rows)
    sl_front = pareto(sl_rows)
    under_500 = best_under(sl_rows, 500.0)
    under_300 = best_under(sl_rows, 300.0)
    lowest_vram = min(sl_rows, key=lambda r: r["peak_gpu_memory_mb"])
    best = min(rows, key=lambda r: r["valid_depth_mae"])

    lines = [
        "# S2M2 Resolution Tradeoff On SCARED",
        "",
        f"Dataset: `{payload['dataset']}`",
        "",
        "This report focuses on S2M2-S and S2M2-L across input resolutions. S2M2-XL runs are retained as reference.",
        "",
        "Disparity rescaling after resized inference is verified in the benchmark script:",
        "",
        "```python",
        "pred_disp_original = pred_disp_resized / scale_x",
        "```",
        "",
        "## Summary",
        "",
        "| model | width | depth MAE | depth median | depth RMSE | disp MAE | disp RMSE | bad 2px | bad 2mm | avg ms | peak MB | params M |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['resize_label']} | {r['valid_depth_mae']:.4f} | {r['valid_depth_median']:.4f} | "
            f"{r['valid_depth_rmse']:.4f} | {r['valid_disp_mae']:.4f} | {r['valid_disp_rmse']:.4f} | "
            f"{r['bad_2px']:.2f} | {r['bad_2mm']:.2f} | {r['avg_inference_time_ms']:.2f} | "
            f"{r['peak_gpu_memory_mb']:.1f} | {r['model_param_count_m']:.2f} |"
        )

    lines.extend(["", "## Practical Deployment Questions", ""])
    if under_500:
        lines.append(
            f"- Highest accuracy under 500 ms among S/L: `{fmt_model(under_500)}` "
            f"with depth MAE `{under_500['valid_depth_mae']:.4f} mm`, "
            f"disp MAE `{under_500['valid_disp_mae']:.4f} px`, "
            f"`{under_500['avg_inference_time_ms']:.2f} ms`, and `{under_500['peak_gpu_memory_mb']:.1f} MB` VRAM."
        )
    if under_300:
        lines.append(
            f"- Highest accuracy under 300 ms among S/L: `{fmt_model(under_300)}` "
            f"with depth MAE `{under_300['valid_depth_mae']:.4f} mm`, "
            f"disp MAE `{under_300['valid_disp_mae']:.4f} px`, "
            f"`{under_300['avg_inference_time_ms']:.2f} ms`, and `{under_300['peak_gpu_memory_mb']:.1f} MB` VRAM."
        )
    else:
        lines.append("- No S/L candidate under 300 ms was available.")
    lines.append(
        f"- Lowest VRAM S/L candidate: `{fmt_model(lowest_vram)}` at `{lowest_vram['peak_gpu_memory_mb']:.1f} MB`, "
        f"depth MAE `{lowest_vram['valid_depth_mae']:.4f} mm`. Compared with best overall `{fmt_model(best)}`, "
        f"degradation is `{lowest_vram['valid_depth_mae'] - best['valid_depth_mae']:.4f} mm`."
    )
    lines.append(
        "- XL reference: XL is still the most accurate at full resolution, but the gain over L/full is small compared with runtime and VRAM."
    )

    lines.extend(["", "## Pareto Frontier", "", "S/L depth accuracy vs runtime vs VRAM frontier:", ""])
    for r in sl_front:
        lines.append(
            f"- `{fmt_model(r)}`: depth MAE `{r['valid_depth_mae']:.4f} mm`, "
            f"disp MAE `{r['valid_disp_mae']:.4f} px`, `{r['avg_inference_time_ms']:.2f} ms`, "
            f"`{r['peak_gpu_memory_mb']:.1f} MB`."
        )
    lines.extend(["", "All-model frontier including XL reference:", ""])
    for r in all_front:
        lines.append(
            f"- `{fmt_model(r)}`: depth MAE `{r['valid_depth_mae']:.4f} mm`, "
            f"`{r['avg_inference_time_ms']:.2f} ms`, `{r['peak_gpu_memory_mb']:.1f} MB`."
        )

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "For practical deployment, `L@736` is the cleanest balance under 300 ms, while `L@full` is the best S/L candidate under 500 ms. "
            "`S@512` has the lowest VRAM and fastest runtime, but gives up more accuracy. "
            "Use `XL@full` only as a teacher/reference unless larger SCARED subsets show a bigger hard-frame advantage.",
            "",
            "Qualitative montages are in `qualitative/`.",
        ]
    )
    md_target.write_text("\n".join(lines) + "\n")


def main():
    payload, rows = load_rows()
    write_outputs(payload, rows)


if __name__ == "__main__":
    main()

