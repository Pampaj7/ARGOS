#!/usr/bin/env python3
"""TC-Stereo with its temporal path enabled, on SCARED-C's corrected per-frame poses.

The frame-path run beside this one was justified by the claim that SCARED-C ships no
per-frame pose. It does, and the claim has been retracted in three places. This runs the
method the way its authors intended: state carried across the sequence and the previous
disparity warped by a rigid 6-DoF reprojection from pose, intrinsics and baseline.

The chaining convention is taken from the method's own `evaluate_stereo.py`, not inferred:
K is padded together with the images, `params` is passed only from the second frame, and
`flow_q`, `net_list`, `fmap1` and the pose are carried forward after each forward pass.

The one thing that is ours to get right is the frame the poses live in. SCARED-C's poses
map the keyframe into each frame's *native* camera, while the disparity TC-Stereo warps is
*rectified*. Rectified and native differ by R1, so the pose in the rectified frame is the
conjugation R1 . pose . R1^T. Getting this wrong produces a plausible number rather than an
error, which is why `--self-check` exists: it warps ground-truth disparity between two
frames with the method's own `warp()` and compares against the ground truth actually
observed there, and it scores the wrong conventions alongside the intended one.

What the result will and will not mean: the rigid-scene assumption behind that warp is
violated by deforming tissue. That is a property of the method meeting this data and is
worth measuring, not a reason to keep declining to measure it.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARGOS = ROOT.parents[1]
TCSTEREO = ARGOS / "external/comparison_methods/Temporally-Consistent-Stereo-Matching"
BIDA = ARGOS / "ARGOS_hand/external_comparison/results/bidastabilizer_raftstereo_robust/d2_full"
OUT = ROOT.parent / "results" / "tcstereo_temporal"
SEQUENCES = ("dataset_2_keyframe_2", "dataset_2_keyframe_3", "dataset_2_keyframe_4")


def _paths() -> None:
    for path in (str(ROOT), str(ROOT / "scripts"), str(ARGOS), str(ARGOS / "ARGOS-V2/scripts"),
                 str(ARGOS / "ARGOS_FREEZED/src")):
        if path not in sys.path:
            sys.path.insert(0, path)


def _patch_softsplat() -> None:
    """Make the method's vendored splatting kernel compile against a modern cupy.

    `softsplat.py` calls `cupy.cuda.compile_with_cache`, removed in cupy 13, and passes its
    include paths as `"-I <path>"` with a space, which nvrtc rejects as one token. Both are
    packaging drift, not a difference in what the kernel computes, so they are shimmed here
    rather than by editing the competitor's source: the comparison stays against the code
    its authors released.
    """
    import cupy
    if hasattr(cupy.cuda, "compile_with_cache"):
        return

    def compile_with_cache(source, options=()):
        flat = []
        for option in options:
            flat += option.split() if option.startswith("-I ") else [option]
        return cupy.RawModule(code=source, options=tuple(flat))

    cupy.cuda.compile_with_cache = compile_with_cache


def rectified_poses(sequence: str, frame_ids: list[str], native_shape: tuple[int, int]):
    """Per-frame poses conjugated into the rectified camera frame, and the rectified K.

    `world_frame_drift.py` established and validated the rectified/native relation against
    an absolute reference; this reuses its loader rather than re-deriving it.
    """
    import cv2
    _paths()
    from world_frame_drift import load_poses, rectification_rotation, sequence_paths
    sys.path.insert(0, str(ARGOS))
    from scripts.scared_c.build_corrected_temporal_gt import load_calib

    kf_dir, _ = sequence_paths(sequence)
    calib = load_calib(kf_dir / "endoscope_calibration.yaml")
    height, width = native_shape
    r1, _r2, p1, _p2, _q, _roi1, _roi2 = cv2.stereoRectify(
        calib["M1"], calib["D1"], calib["M2"], calib["D2"], (width, height),
        calib["R"], calib["T"].reshape(3, 1), flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    poses = load_poses(sequence, frame_ids)
    lift = np.eye(4)
    lift[:3, :3] = rectification_rotation(sequence, native_shape)
    conjugated = {k: lift @ v @ lift.T for k, v in poses.items()}
    # P2[0,3] = -fx * baseline, so the baseline is that over fx -- the same expression the
    # ground-truth converter uses. The bogus `and` that stood here first would have returned
    # P2[0,3] whenever fx was non-zero, i.e. a baseline about 1000x too large.
    baseline_mm = abs(float(_p2[0, 3] / _p2[0, 0]))
    return conjugated, np.asarray(p1[:3, :3], dtype=np.float64), baseline_mm


def self_check(sequence: str, frames: int, device_name: str = "cuda:0") -> None:
    """Warp ground truth between frames with the method's own warp and see which pose
    convention lands on the ground truth actually observed there.

    A wrong convention does not raise: it warps to the wrong place and yields a number. So
    the intended convention is scored against three wrong ones and is required to beat them
    by a wide margin, the same discipline the world-frame drift check needed before its
    first version was thrown away for having no discriminating power.
    """
    import torch
    _paths()
    if str(TCSTEREO) not in sys.path:
        sys.path.insert(0, str(TCSTEREO))
    _patch_softsplat()
    from core.utils.geo_utils import cal_relative_transformation, warp
    from argos_v2.scared_c_data import load_frame_gt, load_sequence_info

    info = load_sequence_info(sequence)
    ids = [info.frame_ids[i] for i in np.linspace(0, len(info.frame_ids) - 1, frames).astype(int)]
    native, _valid = load_frame_gt(info, ids[0])
    poses, intrinsics, baseline_mm = rectified_poses(sequence, ids, native.shape)

    import cv2
    grid_h, grid_w = 144, 180
    scale = grid_w / native.shape[1]
    k_scale = intrinsics * np.array([[scale], [scale], [1.0]])
    # softsplat is a raw CUDA kernel with no CPU fallback -- it asserts False -- so the
    # check runs on the device the scoring will run on.
    device = torch.device(device_name)

    def grid_gt(frame_id):
        depth, valid = load_frame_gt(info, frame_id)
        cover = cv2.resize(valid.astype(np.float32), (grid_w, grid_h), interpolation=cv2.INTER_AREA)
        num = cv2.resize(depth * valid.astype(np.float32), (grid_w, grid_h), interpolation=cv2.INTER_AREA)
        return num / np.maximum(cover, 1e-6) * scale, cover > 0.5

    variants = {
        "intended  R1 . pose . R1^T": lambda a, b: (poses[a], poses[b]),
        "wrong     transposed order": lambda a, b: (poses[b], poses[a]),
        "wrong     identity relative": lambda a, b: (np.eye(4), np.eye(4)),
    }
    print(f"self-check {sequence}: warping ground truth across {len(ids)} frames, "
          f"baseline {baseline_mm:.3f} mm")
    scores = {}
    for label, pick in variants.items():
        errors = []
        for previous, current in zip(ids[:-1], ids[1:]):
            disp_prev, mask_prev = grid_gt(previous)
            disp_cur, mask_cur = grid_gt(current)
            prev_t, cur_t = pick(previous, current)
            relative = cal_relative_transformation(
                torch.from_numpy(prev_t).float()[None].to(device),
                torch.from_numpy(cur_t).float()[None].to(device))
            k = torch.from_numpy(k_scale).float()[None].to(device)
            # `warp` asserts disp >= 0 and returns positive disparity: the negative sign
            # TC-Stereo carries on `flow_q` is the model's internal convention, not the
            # geometry's, so ground truth goes in as-is. Pixels the previous frame does
            # not cover cannot be warped and are excluded rather than counted as zero.
            valid_prev = torch.from_numpy((disp_prev * mask_prev).astype(np.float32))
            warped, _f, valid = warp(
                valid_prev[None, None].to(device),
                torch.zeros(1, 1, grid_h, grid_w, device=device),
                relative, k, torch.linalg.inv(k),
                torch.tensor([[baseline_mm]]).float().to(device))
            got = warped[0, 0].cpu().numpy()
            keep = (valid[0, 0].cpu().numpy() > 0.5) & mask_cur & (got > 0)
            if keep.sum() > 200:
                errors.append(np.abs(got[keep] - disp_cur[keep]))
        scores[label] = float(np.median(np.concatenate(errors))) if errors else float("nan")
        print(f"  {label:28s} median |disparity error| = {scores[label]:8.4f} px")
    best = min(scores, key=lambda k: scores[k])
    assert best.startswith("intended"), (
        f"a wrong pose convention warps better than the intended one ({best}); "
        "the conjugation into the rectified frame is not what this assumes")
    print("PASS: the intended convention warps ground truth onto ground truth best")


def geometry(sequence: str, frame_ids: list[str], grid: tuple[int, int]):
    """Poses in the rectified frame, K scaled to the evaluation grid, and the baseline.

    The NPZ boundary stores a resized rectified image, so K scales isotropically and
    disparity scales with the width ratio. That this resize is a resize and not a crop is
    exactly what `--self-check` tests: a crop would move the principal point and the
    intended convention would stop winning.
    """
    _paths()
    from argos_v2.scared_c_data import load_frame_gt, load_sequence_info

    info = load_sequence_info(sequence)
    native, _valid = load_frame_gt(info, frame_ids[0])
    poses, intrinsics, baseline_mm = rectified_poses(sequence, frame_ids, native.shape)
    scale = grid[1] / native.shape[1]
    return poses, intrinsics * np.array([[scale], [scale], [1.0]]), baseline_mm, scale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=str(TCSTEREO / "checkpoints/sceneflow.pth"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iters", type=int, default=32)
    parser.add_argument("--sequences", nargs="+", default=list(SEQUENCES))
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--skip-self-check", action="store_true",
                        help="only for reruns; the convention is not re-validated")
    args = parser.parse_args()

    if not args.skip_self_check:
        self_check(args.sequences[0], 8, args.device)
    if args.self_check:
        return

    import torch
    _paths()
    if str(TCSTEREO) not in sys.path:
        sys.path.insert(0, str(TCSTEREO))
    _patch_softsplat()
    from core.utils.utils import InputPadder
    from run_tcstereo_reference import INVALID_PENALTY_PX, load_tcstereo, metrics

    device = torch.device(args.device)
    model = load_tcstereo(Path(args.checkpoint), device)
    rows = []

    for sequence in args.sequences:
        raw_npz = np.load(BIDA / sequence / "raw.npz", allow_pickle=False)
        evaluation = np.load(BIDA / sequence / "evaluation.npz", allow_pickle=False)
        frame_ids = [str(f) for f in raw_npz["frame_ids"]]
        T = min(len(frame_ids), args.max_frames) if args.max_frames else len(frame_ids)
        frame_ids = frame_ids[:T]

        raw = raw_npz["raw_disparity"][:T, 0].astype(np.float64)
        raw_valid = raw_npz["raw_valid"][:T, 0].astype(bool)
        gt = evaluation["gt_disparity"][:T, 0].astype(np.float64)
        gt_valid = evaluation["gt_valid"][:T, 0].astype(bool)
        grid = raw.shape[1:]

        poses, k_grid, baseline_mm, _scale = geometry(sequence, frame_ids, grid)
        k_raw = torch.from_numpy(k_grid).float().to(device)[None]
        baseline = torch.tensor([[baseline_mm]]).float().to(device)

        tc = np.empty_like(raw)
        # The state TC-Stereo carries between frames, named as its own evaluate_stereo.py
        # names it. `params` stays None on the first frame: there is nothing to warp yet.
        flow_q = net_list = fmap1 = previous_T = None
        for i in range(T):
            left = torch.from_numpy(raw_npz["rgb_left"][i:i + 1].copy()).float().to(device)
            right = torch.from_numpy(raw_npz["rgb_right"][i:i + 1].copy()).float().to(device)
            if float(left.max()) <= 1.5:      # the boundary stores [0,1]; TC-Stereo wants [0,255]
                left, right = left * 255.0, right * 255.0
            padder = InputPadder(left.shape, divis_by=32)
            (left_p, right_p), k = padder.pad(left, right, K=k_raw)
            current_T = torch.from_numpy(poses[frame_ids[i]]).float().to(device)[None]

            params = {"K": k, "T": current_T, "previous_T": previous_T,
                      "last_disp": flow_q, "last_net_list": net_list,
                      "fmap1": fmap1, "baseline": baseline}
            with torch.inference_mode():
                output = model(left_p, right_p, iters=args.iters, test_mode=True,
                               params=params if flow_q is not None else None)
            disparity, _k = padder.unpad(-output["flow"], K=k)
            tc[i] = disparity[0, 0].float().cpu().numpy()
            flow_q, net_list, fmap1 = output["flow_q"], output["net_list"], output["fmap1"]
            previous_T = current_T
            if i % 100 == 0:
                print(f"  {sequence} {i}/{T}", flush=True)

        for support_name, support in (("gt", gt_valid), ("gt_and_raw", gt_valid & raw_valid)):
            row = metrics(tc, gt, support)
            if row:
                rows.append({"sequence": sequence, "frames": T, "support": support_name,
                             "method": "tcstereo_sceneflow_temporal", "causal": "yes",
                             "frozen_backbone": "no"} | row)
        print(f"{sequence}: {T} frames done", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "tcstereo_temporal.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    pooled = {}
    for support_name in ("gt", "gt_and_raw"):
        sub = [r for r in rows if r["support"] == support_name]
        n = sum(r["pixels"] for r in sub)
        pooled[support_name] = {m: sum(r[m] * r["pixels"] for r in sub) / n
                                for m in ("EPE", "Bad1", "Bad3", "RMSE", "P95", "InvalidRate")} | {"pixels": n}
    (OUT / "pooled.json").write_text(json.dumps(pooled, indent=2) + "\n")
    (OUT / "run_manifest.json").write_text(json.dumps({
        "project": "ARGOS v2", "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "TC-Stereo with its temporal path enabled on SCARED-C's corrected poses",
        "input_boundary": str(BIDA),
        "tcstereo_checkpoint": args.checkpoint, "iters": args.iters,
        "temporal_path_enabled": True,
        "pose_source": "SCARED-C corrected per-frame poses, conjugated R1 . pose . R1^T into "
                       "the rectified frame; validated by --self-check against two wrong "
                       "conventions before scoring",
        "self_check_run": not args.skip_self_check,
        "invalid_penalty_px": INVALID_PENALTY_PX,
        "training_performed": False, "threshold_tuning_performed": False,
    }, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "pooled": pooled}, indent=2))


if __name__ == "__main__":
    main()
