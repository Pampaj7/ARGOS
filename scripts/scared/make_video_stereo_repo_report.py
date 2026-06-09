#!/usr/bin/env python3
import csv
import json
import subprocess
from pathlib import Path


ARGOS = Path("/home/pampaj/Desktop/ARGOS")
REPOS = ARGOS / "external/video_stereo_repos"
OUT = ARGOS / "results/video_stereo_repos"


def sh(cmd, cwd=None):
    try:
        return subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def log_tail(model, name):
    path = OUT / model / name
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-8:])


def commit(repo_name):
    return sh(["git", "-C", str(REPOS / repo_name), "rev-parse", "HEAD"]) or "unknown"


def env_notes(model, text):
    d = OUT / model
    d.mkdir(parents=True, exist_ok=True)
    (d / "environment_notes.txt").write_text(text + "\n")


def main():
    rows = [
        {
            "model_name": "TemporalStereo",
            "repo_url": "https://github.com/youmi-zym/TemporalStereo",
            "repo_dir": str(REPOS / "TemporalStereo"),
            "commit_hash": commit("TemporalStereo"),
            "license": "Apache-2.0",
            "checkpoint_available": "manual Google Drive",
            "checkpoint_download_instructions": "README links Google Drive folder for pretrained checkpoints; not present locally.",
            "expected_input_format": "Dataset loaders / annotation JSON; video_inference expects image folders plus camera pose/intrinsics style inputs.",
            "expected_output_format": "Disparity visualizations / demo outputs; exact custom export needs wrapper.",
            "supports_custom_left_right_sequence": "possible with wrapper",
            "requires_camera_pose": "yes for video_inference path; maybe no for dataset demo",
            "requires_optical_flow": "no documented requirement",
            "requires_intrinsics_or_baseline": "yes/likely for video mode",
            "temporal_window_length": "multi-frame model; configurable through project scripts, not cleanly exposed for ARGOS yet",
            "resolution_constraints": "trained/eval examples use benchmark-specific shapes; max disparity 192 appears in demo metrics",
            "documented_runtime": "not clearly documented in README",
            "documented_memory": "not clearly documented in README",
            "integration_difficulty": "high",
            "main_blockers": "No local checkpoint; dependencies include Python 3.8, PyTorch 1.10/CUDA 11.3, Apex, Detectron2, Cupy; current smoke fails before argparse on matplotlib seaborn-whitegrid style in Python 3.12 env.",
            "recommended_next_action": "Create isolated temporalstereo conda env; download checkpoint; patch matplotlib style or install compatible seaborn; then write ARGOS folder-sequence wrapper.",
            "demo_runs": "no",
            "smoke_status": "failed: matplotlib style/env mismatch",
        },
        {
            "model_name": "TC-Stereo",
            "repo_url": "https://github.com/jiaxiZeng/Temporally-Consistent-Stereo-Matching",
            "repo_dir": str(REPOS / "Temporally-Consistent-Stereo-Matching"),
            "commit_hash": commit("Temporally-Consistent-Stereo-Matching"),
            "license": "MIT",
            "checkpoint_available": "manual Dropbox",
            "checkpoint_download_instructions": "README points to Dropbox checkpoints for TartanAir, SceneFlow, and KITTI_raw; not present locally.",
            "expected_input_format": "TartanAir/SceneFlow/KITTI_raw dataset layouts with image_left/image_right/disparity/pose.",
            "expected_output_format": "Evaluation metrics/visualization from evaluate_stereo.py; custom export needs wrapper.",
            "supports_custom_left_right_sequence": "possible with custom dataset wrapper",
            "requires_camera_pose": "yes for temporal consistency training/eval datasets",
            "requires_optical_flow": "no documented requirement",
            "requires_intrinsics_or_baseline": "likely for depth/pose-aware evaluation; disparity inference itself may not",
            "temporal_window_length": "sequence model; exact window controlled by dataset/config",
            "resolution_constraints": "benchmark-specific; not cleanly documented",
            "documented_runtime": "not documented",
            "documented_memory": "not documented",
            "integration_difficulty": "medium-high",
            "main_blockers": "No local checkpoint; current shared env lacks wandb and exact torch/cupy stack; custom ARGOS loader needed.",
            "recommended_next_action": "Download Dropbox checkpoint and build minimal ARGOS eval_stereo wrapper; likely best first non-SAV integration if checkpoint retrieval succeeds.",
            "demo_runs": "no",
            "smoke_status": "failed: missing wandb in shared env",
        },
        {
            "model_name": "DynamicStereo",
            "repo_url": "https://github.com/facebookresearch/dynamic_stereo",
            "repo_dir": str(REPOS / "dynamic_stereo"),
            "commit_hash": commit("dynamic_stereo"),
            "license": "CC-BY-NC-4.0",
            "checkpoint_available": "automatic/manual from README",
            "checkpoint_download_instructions": "README provides wget URLs/scripts for checkpoints into ./checkpoints.",
            "expected_input_format": "Dynamic Replica / Sintel configs; real-data eval config exists.",
            "expected_output_format": "Depth/reconstruction evaluation outputs under exp_dir.",
            "supports_custom_left_right_sequence": "possible through real-data config or wrapper",
            "requires_camera_pose": "yes/likely; dataset provides intrinsics/extrinsics and depth",
            "requires_optical_flow": "no for inference, but dataset includes flow/trajectories",
            "requires_intrinsics_or_baseline": "yes for depth and Dynamic Replica format",
            "temporal_window_length": "configs include 40-frame and 150-frame eval modes",
            "resolution_constraints": "heavy video transformer; README examples target Dynamic Replica/Sintel",
            "documented_runtime": "not clearly documented",
            "documented_memory": "Dynamic Replica dataset memory documented; model eval memory not clearly documented",
            "integration_difficulty": "medium-high",
            "main_blockers": "Hydra/PyTorch3D/PyTorch 1.12 CUDA 11.3 env required; no checkpoint downloaded yet; non-commercial license.",
            "recommended_next_action": "Create isolated dynamicstereo env and download real-data checkpoint; evaluate real config on ARGOS sequence if intrinsics format can be matched.",
            "demo_runs": "no",
            "smoke_status": "failed: missing hydra in shared env",
        },
        {
            "model_name": "BiDAStereo",
            "repo_url": "https://github.com/TomTomTommi/bidastereo",
            "repo_dir": str(REPOS / "bidastereo"),
            "commit_hash": commit("bidastereo"),
            "license": "MIT",
            "checkpoint_available": "manual GitHub release",
            "checkpoint_download_instructions": "README links release v0.0 checkpoints; copy to ./bidastereo/checkpoints/.",
            "expected_input_format": "SceneFlow/Sintel/Dynamic Replica plus real-data eval config.",
            "expected_output_format": "Evaluation outputs and visualizations from evaluation/evaluate.py.",
            "supports_custom_left_right_sequence": "possible through real-data config/wrapper",
            "requires_camera_pose": "yes/likely for Dynamic Replica; real-data may be less strict",
            "requires_optical_flow": "alignment method may use RAFT/flow components; not a simple frame-only stereo CLI",
            "requires_intrinsics_or_baseline": "likely for depth/eval; disparity model itself can operate on stereo pairs",
            "temporal_window_length": "sample_len/kernel_size; README mentions reducing kernel_size 20 to 10 for memory",
            "resolution_constraints": "evaluated on A6000 48GB; Dynamic Replica eval requires about 32GB unless kernel size reduced",
            "documented_runtime": "A6000 48GB noted, no speed number",
            "documented_memory": "32GB for Dynamic Replica eval; training 8 V100 32GB or 4 A100 80GB",
            "integration_difficulty": "high",
            "main_blockers": "No local checkpoint; PyTorch3D/PyTorch 1.12 CUDA 11.3 env required; high memory expectation.",
            "recommended_next_action": "Defer until TC-Stereo/DynamicStereo are sorted; useful as quality-oriented recent baseline, not deployment first.",
            "demo_runs": "no",
            "smoke_status": "failed: missing hydra in shared env",
        },
        {
            "model_name": "StereoAnyVideo",
            "repo_url": "https://github.com/TomTomTommi/stereoanyvideo",
            "repo_dir": str(REPOS / "stereoanyvideo"),
            "commit_hash": commit("stereoanyvideo"),
            "license": "Apache-2.0",
            "checkpoint_available": "yes, local",
            "checkpoint_download_instructions": "README Google Drive; local checkpoint found at checkpoints/StereoAnyVideo_MIX.pth.",
            "expected_input_format": "Folder with left/*.png and right/*.png stereo video frames.",
            "expected_output_format": "disparity.npy, normalized disparity PNGs, disparity_norm.mp4.",
            "supports_custom_left_right_sequence": "yes",
            "requires_camera_pose": "no",
            "requires_optical_flow": "no explicit external flow",
            "requires_intrinsics_or_baseline": "no for disparity inference; needed only to convert to metric depth",
            "temporal_window_length": "frame_size default 150; ARGOS smoke used 5 frames and iters 6",
            "resolution_constraints": "heavy; ARGOS smoke used 384x640 for memory/speed",
            "documented_runtime": "not documented",
            "documented_memory": "not documented; expected heavy",
            "integration_difficulty": "low-medium",
            "main_blockers": "Runtime/VRAM instrumentation still needed; model is heavy at high resolution; Google Drive checkpoint for fresh setup.",
            "recommended_next_action": "Integrate first as video quality upper bound and add timed wrapper at 384x640/512x736/720x1280.",
            "demo_runs": "yes",
            "smoke_status": "success on ARGOS/SCARED test_sequence using CUDA",
        },
    ]

    for row in rows:
        env_notes(row["model_name"], "\n".join([f"{k}: {v}" for k, v in row.items()]))

    smoke_metrics = json.loads((OUT / "smoke_metrics.json").read_text())
    payload = {"repos": rows, "smoke_metrics": smoke_metrics}
    (OUT / "report.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (OUT / "report.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Video Stereo Repository Scouting",
        "",
        "Repos are isolated under `external/video_stereo_repos/`. No global dependency installation was performed.",
        "",
        "Common smoke-test sequence: `results/video_stereo_repos/test_sequence/` with 5 rectified SCARED dataset_8 keyframes, left/right images, GT disparity, GT depth, valid masks, and metadata.",
        "",
        "Important caveat: these 5 clean keyframes are not guaranteed to be a temporally consecutive video clip. The smoke test checks integration and custom input handling first; true temporal behavior still needs a consecutive SCARED warped/clean sequence.",
        "",
        "## Repo Table",
        "",
        "| model | commit | license | checkpoint | custom sequence | pose | flow | intrinsics | difficulty | smoke | next action |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model_name']} | `{r['commit_hash'][:8]}` | {r['license']} | {r['checkpoint_available']} | "
            f"{r['supports_custom_left_right_sequence']} | {r['requires_camera_pose']} | {r['requires_optical_flow']} | "
            f"{r['requires_intrinsics_or_baseline']} | {r['integration_difficulty']} | {r['smoke_status']} | {r['recommended_next_action']} |"
        )

    lines.extend(
        [
            "",
            "## Smoke Test Results",
            "",
            "| model | disp MAE | depth MAE | depth median | bad 2px | bad 2mm | failure <=0.5 | temporal pred diff | temporal error variation | runtime ms | peak MB |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for r in smoke_metrics["summary"]:
        lines.append(
            f"| {r['model_name']} | {r['valid_disp_mae']:.4f} | {r['valid_depth_mae']:.4f} | {r['valid_depth_median']:.4f} | "
            f"{r['bad_2px']:.2f} | {r['bad_2mm']:.2f} | {r['pred_disp_le_0_5_ratio']:.4f} | "
            f"{r['mean_abs_consecutive_pred_disp_diff']:.4f} | {r['temporal_error_variation']:.4f} | "
            f"{'' if r['avg_runtime_ms'] is None else f'{r['avg_runtime_ms']:.2f}'} | "
            f"{'' if r['peak_gpu_memory_mb'] is None else f'{r['peak_gpu_memory_mb']:.1f}'} |"
        )

    lines.extend(
        [
            "",
            "StereoAnyVideo smoke command and logs are in `results/video_stereo_repos/StereoAnyVideo/`. Other repos have failed smoke logs in their respective folders; failures are dependency/checkpoint integration findings, not model-quality conclusions.",
            "",
            "## Answers",
            "",
            "1. Runnable now: StereoAnyVideo is runnable on the ARGOS/SCARED custom sequence with the local checkpoint. TemporalStereo, TC-Stereo, DynamicStereo, and BiDAStereo are cloned but not runnable in the shared env without isolated dependency setup and/or checkpoints.",
            "2. Minimal custom rectified sequence support: StereoAnyVideo accepts `left/` and `right/` image folders directly. DynamicStereo and BiDAStereo have real-data configs and should be adaptable. TC-Stereo and TemporalStereo need custom dataset/wrapper work and likely pose/intrinsics handling.",
            "3. Usable pretrained checkpoints: StereoAnyVideo has a local checkpoint. DynamicStereo has scripted/manual checkpoint download. TC-Stereo and TemporalStereo provide Dropbox/Google Drive checkpoint links. BiDAStereo provides GitHub release checkpoints.",
            "4. First ARGOS integration: StereoAnyVideo first, because it already runs on custom stereo folders and gives a quality upper-bound style temporal baseline. Next should be TC-Stereo if its Dropbox checkpoint is easy to retrieve; otherwise DynamicStereo real-data config.",
            "5. Model roles:",
            "   - video quality upper bound: StereoAnyVideo;",
            "   - efficient temporal baseline: TC-Stereo candidate once checkpoint/env are ready; TemporalStereo is older but efficient on paper;",
            "   - deployment candidate: none proven yet; compare TC-Stereo/TemporalStereo against S2M2-L@736 after real smoke runs;",
            "   - temporal distillation teacher: StereoAnyVideo now, possibly BiDAStereo/DynamicStereo after successful checkpoint setup.",
            "",
            "## Main Blockers",
            "",
            "- TemporalStereo: Python 3.8/PyTorch 1.10/CUDA 11.3 plus Apex, Detectron2, Cupy; smoke currently fails on matplotlib style before deeper imports.",
            "- TC-Stereo: missing `wandb` in shared env and no checkpoint present; likely manageable in a separate env.",
            "- DynamicStereo: missing Hydra/PyTorch3D stack and checkpoint; CC-BY-NC-4.0 license should be noted for downstream use.",
            "- BiDAStereo: missing Hydra/PyTorch3D/checkpoint and documented high VRAM expectation.",
            "- StereoAnyVideo: runs, but needs timed/VRAM-instrumented wrapper and a true consecutive SCARED video sequence for temporal claims.",
        ]
    )
    (OUT / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

