"""Uniform build_predictor(name, device) -> (method, checkpoint, predict_fn) for the 5
ARGOS-V2 pilot backbones. Reuses existing ARGOS wrappers (scripts/scared/) instead of
reimplementing model loading. Each backbone is expected to run in its own OS subprocess
(see run_backbone_cache.py) — avoids the sys.modules['core']/'utils' collision seen when
multiple external stereo repos are imported into one process (see ARGOS memory:
argos-v3-2 / SCARED unified_keyframes session notes).

predict_fn(left_rgb, right_rgb) -> (disp [H,W] float32 native resolution, runtime_ms float)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from argos_v2.paths import ARGOS_ROOT

SCARED_SCRIPTS = ARGOS_ROOT / "scripts/scared"
TEMPORAL_DATA_PREP = ARGOS_ROOT / "scripts/temporal_refinement/data_prep"

BACKBONE_NAMES = ["S2M2-S", "RAFT-Stereo", "StereoAnywhere", "CREStereo", "Fast-FoundationStereo"]


def build_predictor(name: str, device: torch.device):
    if str(ARGOS_ROOT) not in sys.path:
        sys.path.insert(0, str(ARGOS_ROOT))  # predict_s2m2_long_sequences does `from scripts.argos_paths import ...`
    if str(SCARED_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCARED_SCRIPTS))
    if str(TEMPORAL_DATA_PREP) not in sys.path:
        sys.path.insert(0, str(TEMPORAL_DATA_PREP))

    if name == "S2M2-S":
        from predict_s2m2_long_sequences import build_model, infer

        model = build_model(device, "S")
        width = 512  # matched ARGOS convention, see run_unified_keyframes_ablation.py

        def predict(left, right):
            pred, ms, _scale = infer(model, left, right, device, width)
            return pred, ms

        return "S2M2-S", "CH128NTR1.pth", predict

    if name == "RAFT-Stereo":
        from eval_scared_external_native import eval_raft

        method, checkpoint, _res, predict = eval_raft([], None, device)
        return method, checkpoint, predict

    if name == "StereoAnywhere":
        from eval_scared_external_native import eval_stereoanywhere

        method, checkpoint, _res, predict = eval_stereoanywhere([], None, device)
        return method, checkpoint, predict

    if name == "CREStereo":
        from eval_scared_external_native import eval_crestereo

        method, checkpoint, _res, predict = eval_crestereo([], None, device)
        return method, checkpoint, predict

    if name == "Fast-FoundationStereo":
        from run_unified_keyframes_ablation import make_fast_foundationstereo_predict

        predict = make_fast_foundationstereo_predict(device)
        return "Fast-FoundationStereo_ONNX", "20_30_48_iters_4_res_320x736.onnx", predict

    raise ValueError(f"unknown backbone: {name}")


def inference_resolution(name: str, native_h: int, native_w: int) -> tuple[int, int, str]:
    """(height, width, note) actually fed to the network. Metadata-only, does not affect
    the predict path (each wrapper handles its own resize/pad internally)."""
    if name == "S2M2-S":
        width = 512
        height = round(native_h * width / native_w)
        return height, width, "resized to width=512, aspect-preserved"
    if name == "Fast-FoundationStereo":
        import yaml
        from argos_v2.paths import FRAME_STEREO_REPOS_DIR
        onnx_path = FRAME_STEREO_REPOS_DIR / "Fast-FoundationStereo/weights/onnx/20_30_48/320x736/20_30_48_iters_4_res_320x736.onnx"
        cfg = yaml.safe_load(onnx_path.with_suffix(".yaml").read_text())
        h, w = cfg["image_size"]
        return h, w, "fixed ONNX input resolution"
    return native_h, native_w, "native resolution, padded internally to a multiple of 32"
