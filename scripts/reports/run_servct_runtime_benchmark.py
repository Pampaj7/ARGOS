#!/usr/bin/env python3
"""Run a unified adapter-level runtime/VRAM benchmark for SERV-CT methods.

The existing SERV-CT adapters are script-oriented and do not all expose a pure
predict function. This harness therefore measures the same adapter command for
each method, with model load and file I/O included, and labels the protocol
accordingly. It never uses paper numbers or old unknown-hardware timings.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path.cwd()
OUT = Path("results/servct_unified_frame_benchmark_v1")
RUNTIME_TMP = OUT / "runtime_tmp"
STEREO = Path("../external")
SERVCT_ROOT = Path("Fast-FoundationStereo/data/surgical_stereo/servct/SERV-CT")
FF_PY = Path(".miniconda/envs/argos/bin/python")
AI_PY = "python"
FRAMES = 16


@dataclass
class RuntimeJob:
    method: str
    cwd: Path
    command: list[str]
    framework: str
    device: str
    precision: str
    input_resolution: str = "native/legacy_eval"
    notes: str = ""


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def method_slug(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def s2m2_job(method: str, model_type: str, out: str, checkpoint: str | None = None) -> RuntimeJob:
    cmd = [
        str(Path("../../ARGOS/scripts/s2m2/eval_servct_s2m2.py")),
        "--servct_root",
        "../Fast-FoundationStereo/data/surgical_stereo/servct/SERV-CT",
        "--weights_dir",
        "weights/pretrain_weights",
        "--model_type",
        model_type,
        "--out_dir",
        out,
    ]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]
    return RuntimeJob(method, STEREO / "s2m2", [AI_PY] + cmd, "PyTorch", "cuda", "amp_fp16")


def jobs() -> list[RuntimeJob]:
    return [
        s2m2_job("S2M2-S all-surgical finetuned", "S", "runtime_s2m2_s_all_surgical", "output_servct_finetune_all_surgical_S/s2m2_servct_finetuned.pth"),
        s2m2_job("S2M2-S honest finetuned", "S", "runtime_s2m2_s_honest", "output_servct_finetune_honest_S/s2m2_servct_finetuned.pth"),
        s2m2_job("S2M2-S pretrained", "S", "runtime_s2m2_s"),
        s2m2_job("S2M2-M pretrained", "M", "runtime_s2m2_m"),
        s2m2_job("S2M2-L pretrained", "L", "runtime_s2m2_l"),
        s2m2_job("S2M2-XL pretrained", "XL", "runtime_s2m2_xl"),
        RuntimeJob(
            "Fast-FoundationStereo ONNX",
            STEREO / "Fast-FoundationStereo",
            [".conda/bin/python", "scripts/eval_servct_onnx.py", "--servct_root", "data/surgical_stereo/servct/SERV-CT", "--out_dir", "runtime_fast_foundation_onnx"],
            "ONNX Runtime CUDA",
            "cuda",
            "fp32",
            "320x736",
        ),
        RuntimeJob(
            "SGBM",
            STEREO / "Fast-FoundationStereo",
            [".conda/bin/python", "../../ARGOS/scripts/fast_foundationstereo/eval_servct_sgbm.py", "--servct_root", "data/surgical_stereo/servct/SERV-CT", "--out_dir", "runtime_sgbm"],
            "OpenCV",
            "cpu",
            "int/float32",
            "original",
        ),
        RuntimeJob("StereoAnywhere ViT-L", STEREO / "stereoanywhere", [str(Path("../.miniconda/envs/argos/bin/python")), "../../ARGOS/scripts/stereoanywhere/eval_servct_stereoanywhere.py", "--servct_root", "../Fast-FoundationStereo/data/surgical_stereo/servct/SERV-CT", "--weights", "weights/stereoanywhere_sceneflow.pth", "--loadmonomodel", "weights/depth_anything_v2_vitl.pth", "--vit_encoder", "vitl", "--out_dir", "runtime_stereoanywhere_vitl"], "PyTorch", "cuda", "fp32/adapter_default"),
        RuntimeJob("StereoAnywhere", STEREO / "stereoanywhere", [str(Path("../.miniconda/envs/argos/bin/python")), "../../ARGOS/scripts/stereoanywhere/eval_servct_stereoanywhere.py", "--servct_root", "../Fast-FoundationStereo/data/surgical_stereo/servct/SERV-CT", "--weights", "weights/stereoanywhere_sceneflow.pth", "--loadmonomodel", "weights/depth_anything_v2_vits.pth", "--vit_encoder", "vits", "--out_dir", "runtime_stereoanywhere"], "PyTorch", "cuda", "fp32/adapter_default"),
        RuntimeJob("RAFT-Stereo RVC", STEREO / "RAFT-Stereo", [str(Path("../.miniconda/envs/argos/bin/python")), "../../ARGOS/scripts/raft_stereo/eval_servct_raft.py", "--servct_root", "../Fast-FoundationStereo/data/surgical_stereo/servct/SERV-CT", "--restore_ckpt", "models/iraftstereo_rvc.pth", "--out_dir", "runtime_raft_rvc", "--valid_iters", "32", "--corr_implementation", "reg"], "PyTorch", "cuda", "fp32"),
        RuntimeJob("RAFT-Stereo Middlebury", STEREO / "RAFT-Stereo", [str(Path("../.miniconda/envs/argos/bin/python")), "../../ARGOS/scripts/raft_stereo/eval_servct_raft.py", "--servct_root", "../Fast-FoundationStereo/data/surgical_stereo/servct/SERV-CT", "--restore_ckpt", "models/raftstereo-middlebury.pth", "--out_dir", "runtime_raft_middlebury", "--valid_iters", "32", "--corr_implementation", "reg"], "PyTorch", "cuda", "fp32"),
        RuntimeJob("RAFT-Stereo SceneFlow", STEREO / "RAFT-Stereo", [str(Path("../.miniconda/envs/argos/bin/python")), "../../ARGOS/scripts/raft_stereo/eval_servct_raft.py", "--servct_root", "../Fast-FoundationStereo/data/surgical_stereo/servct/SERV-CT", "--restore_ckpt", "models/raftstereo-sceneflow.pth", "--out_dir", "runtime_raft_sceneflow", "--valid_iters", "32", "--corr_implementation", "reg"], "PyTorch", "cuda", "fp32"),
        RuntimeJob("RAFT-Stereo ETH3D", STEREO / "RAFT-Stereo", [str(Path("../.miniconda/envs/argos/bin/python")), "../../ARGOS/scripts/raft_stereo/eval_servct_raft.py", "--servct_root", "../Fast-FoundationStereo/data/surgical_stereo/servct/SERV-CT", "--restore_ckpt", "models/raftstereo-eth3d.pth", "--out_dir", "runtime_raft_eth3d", "--valid_iters", "32", "--corr_implementation", "reg"], "PyTorch", "cuda", "fp32"),
        RuntimeJob("CREStereo", STEREO / "stereo_matching_crestereo", [str(Path("../.miniconda/envs/argos/bin/python")), "../../ARGOS/scripts/crestereo/eval_servct_crestereo.py", "--servct_root", "../Fast-FoundationStereo/data/surgical_stereo/servct/SERV-CT", "--out_dir", "runtime_crestereo"], "PyTorch", "cuda", "fp32"),
        RuntimeJob("DEFOM-Stereo ViT-L Middlebury", STEREO / "DEFOM-Stereo", [str(Path("../.miniconda/envs/argos/bin/python")), "../../ARGOS/scripts/defom_stereo/eval_servct_defom.py", "--servct_root", "../Fast-FoundationStereo/data/surgical_stereo/servct/SERV-CT", "--restore_ckpt", "checkpoints/defomstereo_vitl_middlebury.pth", "--out_dir", "runtime_defom_vitl_middlebury", "--dinov2_encoder", "vitl", "--valid_iters", "16", "--corr_implementation", "reg"], "PyTorch", "cuda", "fp32"),
        RuntimeJob("DEFOM-Stereo ViT-L ETH3D", STEREO / "DEFOM-Stereo", [str(Path("../.miniconda/envs/argos/bin/python")), "../../ARGOS/scripts/defom_stereo/eval_servct_defom.py", "--servct_root", "../Fast-FoundationStereo/data/surgical_stereo/servct/SERV-CT", "--restore_ckpt", "checkpoints/defomstereo_vitl_eth3d.pth", "--out_dir", "runtime_defom_vitl_eth3d", "--dinov2_encoder", "vitl", "--valid_iters", "16", "--corr_implementation", "reg"], "PyTorch", "cuda", "fp32"),
        RuntimeJob("DEFOM-Stereo ViT-L KITTI", STEREO / "DEFOM-Stereo", [str(Path("../.miniconda/envs/argos/bin/python")), "../../ARGOS/scripts/defom_stereo/eval_servct_defom.py", "--servct_root", "../Fast-FoundationStereo/data/surgical_stereo/servct/SERV-CT", "--restore_ckpt", "checkpoints/defomstereo_vitl_kitti.pth", "--out_dir", "runtime_defom_vitl_kitti", "--dinov2_encoder", "vitl", "--valid_iters", "16", "--corr_implementation", "reg"], "PyTorch", "cuda", "fp32"),
        RuntimeJob("DEFOM-Stereo ViT-S RVC", STEREO / "DEFOM-Stereo", [str(Path("../.miniconda/envs/argos/bin/python")), "../../ARGOS/scripts/defom_stereo/eval_servct_defom.py", "--servct_root", "../Fast-FoundationStereo/data/surgical_stereo/servct/SERV-CT", "--restore_ckpt", "checkpoints/defomstereo_vits_rvc.pth", "--out_dir", "runtime_defom_vits_rvc", "--dinov2_encoder", "vits", "--valid_iters", "16", "--corr_implementation", "reg"], "PyTorch", "cuda", "fp32"),
        RuntimeJob("DEFOM-Stereo ViT-S SceneFlow", STEREO / "DEFOM-Stereo", [str(Path("../.miniconda/envs/argos/bin/python")), "../../ARGOS/scripts/defom_stereo/eval_servct_defom.py", "--servct_root", "../Fast-FoundationStereo/data/surgical_stereo/servct/SERV-CT", "--restore_ckpt", "checkpoints/defomstereo_vits_sceneflow.pth", "--out_dir", "runtime_defom_vits_sceneflow", "--dinov2_encoder", "vits", "--valid_iters", "16", "--corr_implementation", "reg"], "PyTorch", "cuda", "fp32"),
        RuntimeJob("DEFOM-Stereo ViT-L SceneFlow", STEREO / "DEFOM-Stereo", [str(Path("../.miniconda/envs/argos/bin/python")), "../../ARGOS/scripts/defom_stereo/eval_servct_defom.py", "--servct_root", "../Fast-FoundationStereo/data/surgical_stereo/servct/SERV-CT", "--restore_ckpt", "checkpoints/defomstereo_vitl_sceneflow.pth", "--out_dir", "runtime_defom_vitl_sceneflow", "--dinov2_encoder", "vitl", "--valid_iters", "16", "--corr_implementation", "reg"], "PyTorch", "cuda", "fp32"),
        RuntimeJob("MonSter++ MixAll i16", STEREO / "MonSter-plusplus/MonSter++", [str(Path("../../.miniconda/envs/argos/bin/python")), "scripts_eval_servct_monster.py", "--servct_root", "../../Fast-FoundationStereo/data/surgical_stereo/servct/SERV-CT", "--restore_ckpt", "checkpoints/Mix_all_large.pth", "--out_dir", "runtime_monsterpp_mixall_i16", "--valid_iters", "16", "--encoder", "vitl", "--hidden_dims", "128", "128", "128"], "PyTorch", "cuda", "fp32"),
        RuntimeJob("MonSter++ MixAll", STEREO / "MonSter-plusplus/MonSter++", [str(Path("../../.miniconda/envs/argos/bin/python")), "scripts_eval_servct_monster.py", "--servct_root", "../../Fast-FoundationStereo/data/surgical_stereo/servct/SERV-CT", "--restore_ckpt", "checkpoints/Mix_all_large.pth", "--out_dir", "runtime_monsterpp_mixall", "--valid_iters", "4", "--encoder", "vitl", "--hidden_dims", "128", "128", "128"], "PyTorch", "cuda", "fp32"),
        RuntimeJob("RT-MonSter++ zero-shot", STEREO / "MonSter-plusplus/RT-MonSter++", [str(Path("../../.miniconda/envs/argos/bin/python")), "../../../ARGOS/scripts/monsterplusplus/eval_servct_rtmonsterplusplus.py", "--servct_root", "../../Fast-FoundationStereo/data/surgical_stereo/servct/SERV-CT", "--restore_ckpt", "checkpoints/Zero_shot.pth", "--out_dir", "runtime_rtmonsterpp", "--valid_iters", "4"], "PyTorch", "cuda", "fp32"),
    ]


def query_gpu_apps() -> dict[int, int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return {}
    apps: dict[int, int] = {}
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                apps[int(parts[0])] = int(parts[1])
            except ValueError:
                pass
    return apps


def monitor_gpu(proc: subprocess.Popen[Any], samples: list[dict[str, Any]], stop: threading.Event) -> None:
    while not stop.is_set() and proc.poll() is None:
        apps = query_gpu_apps()
        samples.append({"time": time.time(), "total_compute_vram_mb": sum(apps.values()), "apps": apps})
        time.sleep(0.1)


def env_info() -> dict[str, Any]:
    info: dict[str, Any] = {"protocol": "adapter_end_to_end_per_frame; model load, disk I/O, metric/montage file writing included because legacy adapters are script-oriented"}
    for key, cmd in {
        "gpu": ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        "ai_python": ["conda", "run", "-n", "ai", "python", "-c", "import sys,torch; print(sys.executable); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"],
        "ff_python": [str(STEREO / FF_PY), "-c", "import sys,torch,onnxruntime as ort; print(sys.executable); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(ort.__version__); print(ort.get_available_providers())"],
    }.items():
        try:
            info[key] = subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
        except Exception as exc:
            info[key] = f"ERROR: {exc}"
    return info


def run_job(job: RuntimeJob, repeat: int) -> tuple[dict[str, Any], str | None]:
    slug = method_slug(job.method)
    stdout_path = OUT / f"runtime_{slug}_repeat{repeat}.log"
    start_apps = query_gpu_apps()
    t0 = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONPATH"] = ".:src" if job.cwd.name == "s2m2" else "."
    proc = subprocess.Popen(
        job.command,
        cwd=job.cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    samples: list[dict[str, Any]] = []
    stop = threading.Event()
    thread = threading.Thread(target=monitor_gpu, args=(proc, samples, stop), daemon=True)
    thread.start()
    output, _ = proc.communicate()
    stop.set()
    thread.join(timeout=1.0)
    elapsed = time.perf_counter() - t0
    stdout_path.write_text(output)
    peak_total = max([s["total_compute_vram_mb"] for s in samples], default=0)
    baseline = sum(start_apps.values())
    peak_delta = max(0, peak_total - baseline)
    row = {
        "method": job.method,
        "repeat": repeat,
        "status": "success" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "frames": FRAMES,
        "runtime_ms_per_frame": elapsed * 1000.0 / FRAMES,
        "fps": FRAMES / elapsed if elapsed > 0 else math.nan,
        "peak_total_compute_vram_mb": peak_total,
        "peak_delta_vram_mb": peak_delta,
        "framework": job.framework,
        "device": job.device,
        "precision": job.precision,
        "model_input_resolution": job.input_resolution,
        "original_SERVCT_resolution": "720x576",
        "evaluation_resolution": "legacy adapter output",
        "resize_or_padding_policy": "adapter-specific; same command/config as accuracy evaluation",
        "stdout_log": str(stdout_path),
        "command": " ".join(job.command),
        "cwd": str(job.cwd),
        "runtime_protocol": "adapter_end_to_end_per_frame",
        "notes": job.notes,
    }
    err = None if proc.returncode == 0 else output[-2000:]
    return row, err


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, grp in raw[raw["status"] == "success"].groupby("method"):
        rows.append(
            {
                "method": method,
                "runtime_mean_ms": grp["runtime_ms_per_frame"].mean(),
                "runtime_median_ms": grp["runtime_ms_per_frame"].median(),
                "runtime_std_ms": grp["runtime_ms_per_frame"].std(ddof=0),
                "runtime_p5_ms": grp["runtime_ms_per_frame"].quantile(0.05),
                "runtime_p95_ms": grp["runtime_ms_per_frame"].quantile(0.95),
                "fps": 1000.0 / grp["runtime_ms_per_frame"].mean(),
                "peak_vram_mb": grp["peak_delta_vram_mb"].max(),
                "peak_vram_gb": grp["peak_delta_vram_mb"].max() / 1024.0,
                "device": grp["device"].iloc[0],
                "framework": grp["framework"].iloc[0],
                "precision": grp["precision"].iloc[0],
                "runtime_protocol": grp["runtime_protocol"].iloc[0],
                "repeats": len(grp),
            }
        )
    return pd.DataFrame(rows)


def write_md_table(df: pd.DataFrame, path: Path) -> None:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            vals.append("" if pd.isna(v) else (f"{v:.3f}" if isinstance(v, float) else str(v)))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n")


def update_tables(runtime_summary: pd.DataFrame) -> None:
    full = pd.read_csv(OUT / "servct_benchmark_full.csv")
    merged = full.drop(columns=[c for c in ["runtime_mean_ms", "runtime_median_ms", "runtime_std_ms", "fps", "peak_vram_mb", "precision"] if c in full], errors="ignore").merge(
        runtime_summary[["method", "runtime_mean_ms", "runtime_median_ms", "runtime_std_ms", "fps", "peak_vram_mb", "precision", "device", "framework", "runtime_protocol"]],
        on="method",
        how="left",
    )
    merged.to_csv(OUT / "servct_benchmark_full_with_runtime.csv", index=False)
    slide = pd.DataFrame(
        {
            "Method": merged["method"],
            "Training": merged["checkpoint"],
            "Depth MAE [mm]": merged["depth_mae_mm"],
            "Bad-2 mm [%]": merged["bad_2mm_percent"],
            "Disp. MAE [px]": merged["disparity_mae_px"],
            "Runtime [ms]": merged["runtime_mean_ms"],
            "FPS": merged["fps"],
            "Peak VRAM [GB]": merged["peak_vram_mb"] / 1024.0,
            "Device": merged["device"],
        }
    ).sort_values("Depth MAE [mm]", na_position="last")
    slide.to_csv(OUT / "servct_benchmark_slide_ready_with_runtime.csv", index=False)
    write_md_table(slide, OUT / "servct_benchmark_slide_ready_with_runtime.md")


def plot_joined(runtime_summary: pd.DataFrame) -> None:
    full = pd.read_csv(OUT / "servct_benchmark_full.csv")
    full = full.drop(columns=[c for c in ["runtime_mean_ms", "runtime_median_ms", "runtime_std_ms", "fps", "peak_vram_mb", "precision"] if c in full], errors="ignore")
    df = full.merge(runtime_summary, on="method", how="inner")
    specs = [
        ("runtime_mean_ms", "depth_mae_mm", "runtime_vs_depth_mae"),
        ("peak_vram_mb", "depth_mae_mm", "vram_vs_depth_mae"),
        ("runtime_mean_ms", "peak_vram_mb", "runtime_vram_pareto"),
    ]
    for x, y, name in specs:
        fig, ax = plt.subplots(figsize=(9, 6), dpi=160)
        clean = df.dropna(subset=[x, y])
        ax.scatter(clean[x], clean[y], s=65)
        for _, r in clean.iterrows():
            ax.annotate(str(r["method"])[:24], (r[x], r[y]), fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(True, alpha=0.25)
        ax.set_title(name.replace("_", " "))
        fig.tight_layout()
        fig.savefig(OUT / "plots" / f"{name}.png")
        fig.savefig(OUT / "plots" / f"{name}.pdf")
        plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plots").mkdir(exist_ok=True)
    RUNTIME_TMP.mkdir(exist_ok=True)
    (OUT / "hardware_software_environment.json").write_text(json.dumps(env_info(), indent=2))

    all_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    repeat_count = int(os.environ.get("ARGOS_RUNTIME_REPEATS", "1"))
    job_list = jobs()
    only = os.environ.get("ARGOS_RUNTIME_ONLY")
    if only:
        wanted = {x.strip() for x in only.split("|") if x.strip()}
        job_list = [j for j in job_list if j.method in wanted]
    print(f"Runtime benchmark protocol: adapter_end_to_end_per_frame, repeats={repeat_count}")
    print(f"Methods: {len(job_list)}")
    for job in job_list:
        print(f"\n### {job.method}")
        for rep in range(1, repeat_count + 1):
            row, err = run_job(job, rep)
            all_rows.append(row)
            print(f"repeat {rep}: {row['status']} {row['runtime_ms_per_frame']:.1f} ms/frame peak_delta={row['peak_delta_vram_mb']} MB")
            if err:
                failures.append(
                    {
                        "method": job.method,
                        "repeat": rep,
                        "stage": "adapter_command",
                        "error_summary": err.replace("\n", " ")[:1000],
                        "attempted_fix": "No invasive adapter changes during runtime benchmark.",
                        "final_status": "failed",
                        "stdout_log": row["stdout_log"],
                    }
                )

    raw = pd.DataFrame(all_rows)
    if only and (OUT / "servct_runtime_raw.csv").exists():
        previous = pd.read_csv(OUT / "servct_runtime_raw.csv")
        previous = previous[~previous["method"].isin({j.method for j in job_list})]
        raw = pd.concat([previous, raw], ignore_index=True)
    raw.to_csv(OUT / "servct_runtime_raw.csv", index=False)
    summary = summarize(raw)
    summary.to_csv(OUT / "servct_runtime_summary.csv", index=False)
    fail_df = pd.DataFrame(failures)
    if only and (OUT / "servct_runtime_failures.csv").exists():
        previous_fail = pd.read_csv(OUT / "servct_runtime_failures.csv")
        previous_fail = previous_fail[~previous_fail["method"].isin({j.method for j in job_list})]
        fail_df = pd.concat([previous_fail, fail_df], ignore_index=True)
    fail_df.to_csv(OUT / "servct_runtime_failures.csv", index=False)
    update_tables(summary)
    plot_joined(summary)

    evidence = []
    for _, r in summary.iterrows():
        evidence.append(
            {
                "asset": "servct_benchmark_full_with_runtime.csv",
                "asset_type": "runtime_table",
                "claim_or_metric": f"{r['method']} adapter runtime and peak observed compute VRAM",
                "source_file": "servct_runtime_raw.csv",
                "source_row_or_checkpoint": r["method"],
                "evaluation_protocol": r["runtime_protocol"],
                "notes": "Adapter-level measurement; model load, disk I/O, metric/montage writing included.",
            }
        )
    pd.DataFrame(evidence).to_csv(OUT / "runtime_evidence_manifest.csv", index=False)
    print("\nSummary:")
    print(summary.sort_values("runtime_mean_ms").to_string(index=False))


if __name__ == "__main__":
    main()
