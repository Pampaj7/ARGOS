import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/home/pampaj/Desktop/stereo")
OUT = ROOT / "argos_baselines"

RUNS = [
    ("SGBM", ROOT / "Fast-FoundationStereo/output_servct_eval_sgbm/summary.json", "tested", "classical OpenCV baseline"),
    ("Fast-FoundationStereo ONNX", ROOT / "Fast-FoundationStereo/output_servct_eval/summary.json", "tested", "NVLabs real-time foundation baseline"),
    ("S2M2-S zero-shot", ROOT / "s2m2/output_servct_eval_s2m2_S/summary.json", "tested", "pretrained S checkpoint"),
    ("S2M2-M zero-shot", ROOT / "s2m2/output_servct_eval_s2m2_M/summary.json", "tested", "pretrained M checkpoint"),
    ("S2M2-S fine-tuned all-surgical", ROOT / "s2m2/output_servct_eval_s2m2_S_all_surgical_checkpoint/summary.json", "tested", "SERV-CT adapted checkpoint; not independent holdout"),
    ("Stereo Anywhere VIT-L", ROOT / "stereoanywhere/output_servct_eval_stereoanywhere_vitl/summary.json", "tested", "Depth Anything V2-L prior"),
    ("RT-MonSter++ zero-shot", ROOT / "MonSter-plusplus/RT-MonSter++/output_servct_eval_rtmonster_zeroshot/summary.json", "tested", "RT zero-shot checkpoint"),
    ("MonSter++ MixAll large i16", ROOT / "MonSter-plusplus/MonSter++/output_servct_eval_monsterpp_mixall_i16/summary.json", "tested", "large MixAll checkpoint, 16 iterations"),
    ("CREStereo", ROOT / "stereo_matching_crestereo/output_servct_eval_crestereo/summary.json", "tested", "bundled epoch-570 checkpoint"),
    ("RAFT-Stereo RVC", ROOT / "RAFT-Stereo/output_servct_eval_raft_rvc/summary.json", "tested", "iRAFT RVC checkpoint, context_norm=instance"),
    ("RAFT-Stereo Middlebury", ROOT / "RAFT-Stereo/output_servct_eval_raft_middlebury/summary.json", "tested", "Middlebury checkpoint"),
]

REPO_STATUS = [
    ("RAFT-Stereo", "cloned and tested; Dropbox models downloaded", "historical reviewer baseline"),
    ("IGEV++", "cloned; weights are Google Drive gated/manual", "strong pure stereo baseline"),
    ("Selective-Stereo", "cloned; weights are Google Drive gated/manual", "CVPR 2024 Highlight, detail/frequency baseline"),
    ("CREStereo", "cloned and tested", "practical robust stereo baseline"),
    ("MonSter++", "cloned; RT and large tested", "monodepth-prior foundation stereo"),
    ("DEFOM-Stereo", "cloned; checkpoint download running in screen `argos_defom_download`", "depth-foundation stereo baseline"),
]


def load_rows():
    rows = []
    for name, path, status, notes in RUNS:
        row = {"model": name, "status": status, "notes": notes, "summary_path": str(path)}
        if path.exists():
            data = json.loads(path.read_text())
            row.update(data)
        else:
            row.update({"mae_px": None, "rmse_px": None, "bad2_pct": None, "depth_mae_mm": None, "depth_rmse_mm": None, "frames": None})
            row["status"] = "pending"
        rows.append(row)
    return rows


def write_markdown(df):
    tested = df[df["mae_px"].notna()].sort_values("depth_mae_mm")
    lines = [
        "# ARGOS SERV-CT Baseline Scoreboard",
        "",
        "Common benchmark: SERV-CT Reference_CT, 16 rectified stereo frames unless noted.",
        "Metrics are disparity-space and metric-depth errors against GT disparity/depth.",
        "",
        "## Scores",
        "",
        "| Rank | Model | Disp MAE px | Disp RMSE px | Bad-2 % | Depth MAE mm | Depth RMSE mm | Frames | Notes |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(tested.to_dict("records"), start=1):
        lines.append(
            f"| {rank} | {row['model']} | {row['mae_px']:.3f} | {row['rmse_px']:.3f} | "
            f"{row['bad2_pct']:.2f} | {row['depth_mae_mm']:.3f} | {row['depth_rmse_mm']:.3f} | "
            f"{int(row['frames'])} | {row['notes']} |"
        )
    lines += [
        "",
        "## Repository Status",
        "",
        "| Repo | Status | Why it matters |",
        "|---|---|---|",
    ]
    for repo, status, why in REPO_STATUS:
        lines.append(f"| {repo} | {status} | {why} |")
    lines += [
        "",
        "## Local Modifications",
        "",
        "- `stereo_matching_crestereo/stereo_matching_crestereo/stereo_matching.py`: patched resize guard so `input_hw=None` does not trigger `boxx.resize` with NumPy 2.x.",
        "- `stereo_matching_crestereo/scripts_eval_servct_crestereo.py`: added SERV-CT evaluator and montage writer.",
        "- `MonSter-plusplus/MonSter++/core/monster.py` and `RT-MonSter++/core/monster.py`: patched DepthAnything checkpoint path to local `checkpoints/depth_anything_v2_{encoder}.pth`.",
        "- `MonSter-plusplus/*/scripts_eval_servct_monster.py`: added SERV-CT evaluator for RT and large checkpoints.",
        "- `argos_baselines/scripts/make_servct_scoreboard.py`: creates this report and PNG ranking.",
        "",
        "## Next Baselines",
        "",
        "- Add S2M2-L/XL once the queued checkpoint download starts after SCARED.",
        "- Add IGEV++ once weights are available from Google Drive or a usable mirror.",
        "- Add Selective-Stereo once weights are available.",
        "- Finish DEFOM-Stereo checkpoint download and add its SERV-CT evaluator.",
    ]
    (OUT / "docs/servct_scoreboard.md").write_text("\n".join(lines) + "\n")


def write_plot(df):
    tested = df[df["depth_mae_mm"].notna()].sort_values("depth_mae_mm")
    plt.figure(figsize=(11, 6))
    bars = plt.barh(tested["model"], tested["depth_mae_mm"], color="#2c7fb8")
    plt.gca().invert_yaxis()
    plt.xlabel("Depth MAE (mm), lower is better")
    plt.title("ARGOS SERV-CT Surgical Stereo Baselines")
    for bar, value in zip(bars, tested["depth_mae_mm"]):
        plt.text(value + 0.03, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / "images/servct_depth_mae_scoreboard.png", dpi=180)
    plt.close()


def main():
    (OUT / "docs").mkdir(parents=True, exist_ok=True)
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "metrics").mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "metrics/servct_scoreboard.csv", index=False)
    write_markdown(df)
    write_plot(df)
    print(OUT / "docs/servct_scoreboard.md")
    print(OUT / "images/servct_depth_mae_scoreboard.png")


if __name__ == "__main__":
    main()
