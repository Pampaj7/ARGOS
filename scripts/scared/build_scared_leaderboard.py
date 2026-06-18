import csv
import json
from pathlib import Path


RUNS = [
    {
        "model": "SGBM",
        "setting": "default",
        "result_dir": "results/scared_sgbm_float_eval",
        "notes": "OpenCV SGBM max_disp=320 block=3; raw mask keeps historical pred>0 filter",
    },
    {
        "model": "S2M2-L",
        "setting": "resize1024",
        "result_dir": "results/01_frame_stereo/SCARED/scared_s2m2_L_float_eval",
        "notes": "S2M2-L, max_width=1024",
    },
    {
        "model": "S2M2-L",
        "setting": "full-res",
        "result_dir": "results/01_frame_stereo/SCARED/scared_s2m2_L_full_float_eval",
        "notes": "S2M2-L, original 1024x1280",
    },
    {
        "model": "S2M2-XL",
        "setting": "resize1024",
        "result_dir": "results/01_frame_stereo/SCARED/scared_s2m2_XL_float_eval",
        "notes": "S2M2-XL, max_width=1024",
    },
    {
        "model": "S2M2-XL",
        "setting": "full-res",
        "result_dir": "results/01_frame_stereo/SCARED/scared_s2m2_XL_full_float_eval",
        "notes": "S2M2-XL, original 1024x1280",
    },
    {
        "model": "Fast-FoundationStereo",
        "setting": "ONNX 320x736",
        "result_dir": "results/scared_fast_foundationstereo_onnx_eval",
        "notes": "20_30_48 iters=4 ONNX; fixed 320x736 input, disparity rescaled to GT resolution",
    },
]


COLUMNS = [
    "model",
    "setting",
    "raw_depth_mae_mm",
    "raw_depth_rmse_mm",
    "valid_disp_depth_mae_mm",
    "valid_disp_depth_rmse_mm",
    "depth_mae_cap100_mm",
    "depth_rmse_cap100_mm",
    "depth_mae_cap200_mm",
    "depth_rmse_cap200_mm",
    "depth_median_abs_error_mm",
    "depth_abs_error_p95_mm",
    "depth_abs_error_p99_mm",
    "bad_1mm",
    "bad_2mm",
    "bad_5mm",
    "pred_disp_le_0_1_ratio",
    "pred_disp_le_0_5_ratio",
    "runtime_ms_mean",
    "notes",
]


def load_summary(result_dir):
    path = Path(result_dir) / "summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def row_from_summary(run):
    s = load_summary(run["result_dir"])
    return {
        "model": run["model"],
        "setting": run["setting"],
        "raw_depth_mae_mm": s["depth_mae_mm"],
        "raw_depth_rmse_mm": s["depth_rmse_mm"],
        "valid_disp_depth_mae_mm": s["valid_disp_depth_mae_mm"],
        "valid_disp_depth_rmse_mm": s["valid_disp_depth_rmse_mm"],
        "depth_mae_cap100_mm": s["depth_mae_cap100_mm"],
        "depth_rmse_cap100_mm": s["depth_rmse_cap100_mm"],
        "depth_mae_cap200_mm": s["depth_mae_cap200_mm"],
        "depth_rmse_cap200_mm": s["depth_rmse_cap200_mm"],
        "depth_median_abs_error_mm": s["depth_median_abs_error_mm"],
        "depth_abs_error_p95_mm": s["abs_depth_error_p95"],
        "depth_abs_error_p99_mm": s["abs_depth_error_p99"],
        "bad_1mm": s["depth_bad1mm_pct"],
        "bad_2mm": s["depth_bad2mm_pct"],
        "bad_5mm": s["depth_bad5mm_pct"],
        "pred_disp_le_0_1_ratio": s["pred_disp_le_0_1_ratio"],
        "pred_disp_le_0_5_ratio": s["pred_disp_le_0_5_ratio"],
        "runtime_ms_mean": s.get("runtime_ms", ""),
        "notes": run["notes"],
    }


def fmt(v):
    if isinstance(v, float):
        if abs(v) >= 1000:
            return f"{v:.1f}"
        return f"{v:.4f}"
    return str(v)


def write_markdown(path, rows):
    headers = COLUMNS
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row[h]) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n")


def main():
    rows = [row_from_summary(run) for run in RUNS]
    out_csv = Path("results/scared_leaderboard.csv")
    out_json = Path("results/scared_leaderboard.json")
    out_md = Path("results/scared_leaderboard.md")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    out_json.write_text(json.dumps(rows, indent=2))
    write_markdown(out_md, rows)
    print(out_md.read_text())


if __name__ == "__main__":
    main()
