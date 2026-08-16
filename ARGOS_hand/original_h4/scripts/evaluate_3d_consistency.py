#!/usr/bin/env python3
"""Frame-to-frame 3D surface consistency (point-to-point drift, normal stability) on SCARED-C.

Per-frame disparity error says nothing about whether accumulated geometry is *stable*: a
refiner can leave EPE unchanged while removing the frame-to-frame surface jitter that
corrupts downstream mapping/servoing. This script back-projects each frame's disparity
(raw / refined / GT) to a metric point cloud in the left-camera frame on the 144x180 cache
grid, pulls the previous frame's cloud onto the current grid with the same SEA-RAFT
flow-correspondence convention used by evaluate_temporal_corrected.py, and measures:

  p2p_mm      Euclidean distance (mm) between each point and its flow-corresponded point
              from the previous frame, both expressed in their own left-camera frames.
  normal_deg  angle (deg) between the per-pixel surface normal and the flow-corresponded
              previous-frame normal (normals from central differences of the XYZ map,
              oriented toward the camera; warped with nearest sampling so no blending).

Support is the prediction-independent protocol used elsewhere -- GT coverage > 0.5 AND raw
backbone validity -- at both frames of a pair (previous-frame support pulled through the
warp), intersected with SEA-RAFT forward-backward cycle consistency at library-default
thresholds. Identical support for raw, refined and GT within a backbone. The GT rows are
the irreducible floor: real scene + camera motion between frames shows up in GT exactly as
in the predictions, so the claimable quantity is the *excess* of raw/refined over GT.

What this can and cannot claim
------------------------------
SCARED-C ships no per-frame camera pose in this repo, and none is invented here. Both
clouds of a pair live in their *own* camera frames, so p2p_mm mixes true surface change
with rigid camera motion; the GT floor absorbs that shared component. This is
FRAME-TO-FRAME SURFACE STABILITY, NOT global reconstruction drift: no windowed rigid
alignment, no accumulated trajectory, no loop metric. The sliding window (default 8
frames) only pools consecutive-pair statistics; it does not chain transforms.

Assumptions a reviewer could challenge: principal point at image center and fy = fx
(rectified, near-isotropic grid rescale -- asserted at runtime); disparity clamped at
0.1 px before back-projection to bound depth; flow correspondence stands in for true
scene-point correspondence (gated by forward-backward consistency, not tuned).

`run_comparison.py` and `definitive_evaluation.py` are pinned by freeze manifests and are
imported unchanged. Nothing is trained and no threshold is tuned.

Smoke:        python evaluate_3d_consistency.py --self-check
GPU smoke:    python evaluate_3d_consistency.py --sequences dataset_2_keyframe_2 \
                  --backbones S2M2-S --max-frames 12
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARGOS = ROOT.parents[1]
OUT = ROOT.parent / "results" / "three_d_consistency"
D2 = ("dataset_2_keyframe_2", "dataset_2_keyframe_3", "dataset_2_keyframe_4")
D7 = ("dataset_7_keyframe_1", "dataset_7_keyframe_2", "dataset_7_keyframe_3", "dataset_7_keyframe_4")
BACKBONES = ("S2M2-S", "RAFT-Stereo", "StereoAnywhere", "CREStereo", "Fast-FoundationStereo")
GRID_W, GRID_H = 180, 144
MIN_DISP_PX = 0.1  # depth cap fx*b/0.1; prevents infinities without masking any pixel


def _paths() -> None:
    for path in (str(ROOT), str(ROOT / "scripts"), str(ARGOS / "ARGOS_FREEZED/src"),
                 str(ARGOS / "ARGOS-V2/scripts"),
                 str(ARGOS / "ARGOS_FREEZED/experiments/02_massive_training/scripts")):
        if path not in sys.path:
            sys.path.insert(0, path)


def backproject(disp, fx: float, baseline_mm: float, cx: float, cy: float):
    """Disparity [H,W] (grid px, grid-scaled fx) -> XYZ [3,H,W] in mm, left-camera frame,
    +Z forward. Assumes fy == fx and principal point (cx, cy)."""
    import torch
    h, w = disp.shape
    z = fx * baseline_mm / disp.clamp(min=MIN_DISP_PX)
    ys, xs = torch.meshgrid(torch.arange(h, device=disp.device, dtype=disp.dtype),
                            torch.arange(w, device=disp.device, dtype=disp.dtype), indexing="ij")
    return torch.stack(((xs - cx) * z / fx, (ys - cy) * z / fx, z))


def normals_from_xyz(xyz):
    """XYZ [3,H,W] -> unit normals [3,H,W] from central differences, oriented toward the
    camera (dot(n, p) <= 0)."""
    import torch
    gy, gx = torch.gradient(xyz, dim=(1, 2))
    n = torch.cross(gx, gy, dim=0)
    n = n / torch.linalg.vector_norm(n, dim=0, keepdim=True).clamp_min(1e-9)
    flip = (n * xyz).sum(0, keepdim=True) > 0
    return torch.where(flip, -n, n)


def erode(mask):
    """[B,1,H,W] bool -> 3x3-eroded bool (normals use +/-1 px neighbours)."""
    import torch.nn.functional as F
    return F.max_pool2d((~mask).float(), 3, stride=1, padding=1) < 0.5


def _stats(values: np.ndarray) -> dict[str, float]:
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95))}


def self_check() -> None:
    """Analytic validation of back-projection, normals and warp correspondence. CPU-only,
    no dataset, no GPU."""
    import torch
    _paths()
    from argos_freezed.alignment.bida_pull_warp import causal_warp

    h, w, fx, b = 48, 64, 100.0, 5.0
    cx, cy = (w - 1) / 2, (h - 1) / 2
    us = torch.arange(w, dtype=torch.float32).expand(h, w)

    # Fronto-parallel plane at Z0: exact depth, exact X, normal (0,0,-1) toward camera.
    z0 = 80.0
    disp = torch.full((h, w), fx * b / z0)
    xyz = backproject(disp, fx, b, cx, cy)
    assert torch.allclose(xyz[2], torch.full((h, w), z0), atol=1e-4), "depth mismatch"
    assert torch.allclose(xyz[0], (us - cx) * z0 / fx, atol=1e-4), "X back-projection mismatch"
    n = normals_from_xyz(xyz)
    assert torch.allclose(n[2], torch.full((h, w), -1.0), atol=1e-5), "fronto normal wrong"

    # Tilted plane Z = Z0 / (1 - s*(u-cx)/fx): analytic normal (s,0,-1)/norm toward camera.
    s = 0.2
    z_tilt = z0 / (1 - s * (us - cx) / fx)
    xyz_t = backproject(fx * b / z_tilt, fx, b, cx, cy)
    n_t = normals_from_xyz(xyz_t)
    expected = torch.tensor([s, 0.0, -1.0])
    expected = expected / expected.norm()
    dots = (n_t[:, 2:-2, 2:-2] * expected.view(3, 1, 1)).sum(0)
    assert float(dots.min()) > float(np.cos(np.deg2rad(0.5))), "tilted-plane normal off by >0.5 deg"

    # Zero flow: identical frame -> zero p2p drift.
    warp0 = causal_warp(xyz_t[None], torch.zeros(1, 2, h, w))
    drift0 = torch.linalg.vector_norm(warp0.warped[0] - xyz_t, dim=0)[warp0.valid[0, 0]]
    assert float(drift0.max()) < 1e-3, "zero-flow drift nonzero"

    # +1 px x-flow on the fronto plane: corresponded point sits Z0/fx mm away in X.
    flow = torch.zeros(1, 2, h, w)
    flow[:, 0] = 1.0
    warp1 = causal_warp(xyz[None], flow)
    drift1 = torch.linalg.vector_norm(xyz - warp1.warped[0], dim=0)[warp1.valid[0, 0]]
    assert abs(float(drift1.mean()) - z0 / fx) < 1e-3, "1-px correspondence distance wrong"

    # Nearest-mode normal warp on the tilted plane keeps unit, camera-facing normals.
    warpn = causal_warp(n_t[None], flow, mode="nearest")
    norms = torch.linalg.vector_norm(warpn.warped[0], dim=0)[warpn.valid[0, 0]]
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "nearest normal warp blended"

    print("self-check PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("d2", "d7"), default="d2")
    parser.add_argument("--backbones", nargs="+", default=list(BACKBONES))
    parser.add_argument("--sequences", nargs="+")
    parser.add_argument("--module", default="model_design.comparison.canonical_h4_masked:factory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--flow-batch-size", type=int, default=32)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--window", type=int, default=8, help="sliding-window length in frames")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return

    import cv2
    import torch
    _paths()
    from argos_freezed.alignment.bida_pull_warp import causal_warp, forward_backward_consistency
    from argos_freezed.alignment.sea_raft_adapter import SEARAFTFlowAdapter
    from argos_v2.cache_io import load_sequence_cache
    from argos_v2.scared_c_data import load_frame_gt, load_sequence_info, read_rgb
    from model_design.comparison.run_comparison import drive, load_factory

    device = torch.device(args.device)
    adapter = load_factory(args.module)(device=args.device)
    flow_model = SEARAFTFlowAdapter(device=device)
    sequences = tuple(args.sequences or (D2 if args.split == "d2" else D7))
    cx, cy = (GRID_W - 1) / 2, (GRID_H - 1) / 2
    rows = []

    def rgb(path: Path):
        value = cv2.resize(read_rgb(path), (GRID_W, GRID_H), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(np.ascontiguousarray(value)).permute(2, 0, 1).float().to(device)[None]

    for sequence in sequences:
        info = load_sequence_info(sequence)
        ids = info.frame_ids[:args.max_frames] if args.max_frames else info.frame_ids
        images = [rgb(info.seq_dir / "left" / f"{v}.png") for v in ids]
        right = [rgb(info.seq_dir / "right" / f"{v}.png") for v in ids]
        gts, covers = [], []
        native_shape = None
        for frame_id in ids:
            native, valid = load_frame_gt(info, frame_id)
            native_shape = native.shape
            coverage = cv2.resize(valid.astype(np.float32), (GRID_W, GRID_H), interpolation=cv2.INTER_AREA)
            numerator = cv2.resize(native * valid.astype(np.float32), (GRID_W, GRID_H), interpolation=cv2.INTER_AREA)
            gts.append(numerator / np.maximum(coverage, 1e-6) * (GRID_W / native.shape[1]))
            covers.append(coverage)
        gts, covers = np.stack(gts), np.stack(covers)
        T = len(ids)
        # fy = fx is only sound if the grid rescale is isotropic.
        assert abs(GRID_W / native_shape[1] - GRID_H / native_shape[0]) < 1e-3, \
            f"anisotropic grid rescale for {sequence}: native {native_shape}"
        fx_grid = info.fx * GRID_W / native_shape[1]

        # Pair flows: flow[t-1] lives on frame t and pulls frame t-1 (= infer(t, t-1));
        # forward-backward cycle consistency at library defaults gates correspondences.
        flow_list, fbv_list = [], []
        with torch.inference_mode():
            for start in range(1, T, args.flow_batch_size):
                idx = list(range(start, min(start + args.flow_batch_size, T)))
                cur = torch.cat([images[i] for i in idx])
                prev = torch.cat([images[i - 1] for i in idx])
                onto_cur = flow_model.infer(cur, prev)
                onto_prev = flow_model.infer(prev, cur)
                fb = forward_backward_consistency(onto_cur, onto_prev)
                flow_list.append(onto_cur)
                fbv_list.append(fb.valid)
            pair_flow = torch.cat(flow_list)   # [T-1, 2, H, W]
            fb_valid = torch.cat(fbv_list)     # [T-1, 1, H, W] bool
        print(f"{sequence}: {T - 1} pair flows ready", flush=True)

        for backbone in args.backbones:
            disparity, validity, cache_ids, _ = load_sequence_cache(backbone, sequence)
            if [str(v) for v in cache_ids[:T]] != ids:
                raise RuntimeError(f"frame-ID mismatch: {backbone}/{sequence}")
            frames = [{"index": i,
                       "raw": torch.from_numpy(np.asarray(disparity[i:i + 1], np.float32))[:, None].to(device),
                       "raw_valid": torch.from_numpy(np.asarray(validity[i:i + 1]) > 0)[:, None].to(device),
                       "rgb": images[i], "right_rgb": right[i]} for i in range(T)]

            def flow_pair(current, past):
                a, b = current["index"], past["index"]
                return flow_model.infer(images[a], images[b]), flow_model.infer(images[b], images[a])

            with torch.inference_mode():
                outputs = dict(drive(adapter, frames, flow_pair))
                refined = torch.stack([outputs[i]["disparity"][0, 0].detach().float() for i in range(T)]).to(device)
                raw = torch.from_numpy(np.asarray(disparity[:T], np.float32)).to(device)
                gt = torch.from_numpy(gts).to(device)

                # Prediction-independent protocol: GT coverage > 0.5 AND raw validity.
                proto = (torch.from_numpy(covers > .5).to(device)
                         & (torch.from_numpy(np.asarray(validity[:T]) > 0).to(device)))[:, None]
                proto_n = erode(proto)  # normals need a valid 3x3 neighbourhood

                samples = {m: {"p2p_mm": {}, "normal_deg": {}} for m in ("raw", "refined", "gt")}
                for method, disp_seq in (("raw", raw), ("refined", refined), ("gt", gt)):
                    xyz = torch.stack([backproject(disp_seq[t], fx_grid, info.baseline_mm, cx, cy)
                                       for t in range(T)])
                    nrm = torch.stack([normals_from_xyz(xyz[t]) for t in range(T)])
                    for t in range(1, T):
                        flow = pair_flow[t - 1:t]
                        wxyz = causal_warp(xyz[t - 1:t], flow, source_valid=proto[t - 1:t].float())
                        wn = causal_warp(nrm[t - 1:t], flow, source_valid=proto_n[t - 1:t].float(),
                                         mode="nearest")
                        m_p2p = proto[t:t + 1] & wxyz.valid & fb_valid[t - 1:t]
                        m_nrm = proto_n[t:t + 1] & wn.valid & fb_valid[t - 1:t]
                        d3 = torch.linalg.vector_norm(xyz[t] - wxyz.warped[0], dim=0)[m_p2p[0, 0]]
                        dot = (nrm[t] * wn.warped[0]).sum(0).clamp(-1.0, 1.0)[m_nrm[0, 0]]
                        samples[method]["p2p_mm"][t] = d3.cpu().numpy()
                        samples[method]["normal_deg"][t] = np.degrees(np.arccos(dot.cpu().numpy()))

            for method in ("raw", "refined", "gt"):
                for metric in ("p2p_mm", "normal_deg"):
                    per_pair = samples[method][metric]
                    pooled = np.concatenate([per_pair[t] for t in sorted(per_pair)]) \
                        if per_pair else np.zeros(0, np.float32)
                    if pooled.size:
                        for stat, value in _stats(pooled).items():
                            rows.append({"split": args.split, "sequence": sequence, "backbone": backbone,
                                         "method": method, "metric": metric, "stat": stat,
                                         "aggregation": "pooled", "value": value, "support": int(pooled.size)})
                    # Sliding window of N frames = N-1 consecutive pairs, macro-averaged.
                    win_stats, win_support = [], []
                    for w0 in range(0, T - args.window + 1):
                        chunk = [per_pair[t] for t in range(w0 + 1, w0 + args.window) if per_pair.get(t, np.zeros(0)).size]
                        if not chunk:
                            continue
                        win = np.concatenate(chunk)
                        win_stats.append(_stats(win))
                        win_support.append(win.size)
                    if win_stats:
                        for stat in ("mean", "median", "p95"):
                            rows.append({"split": args.split, "sequence": sequence, "backbone": backbone,
                                         "method": method, "metric": metric, "stat": stat,
                                         "aggregation": "window_macro",
                                         "value": float(np.mean([s[stat] for s in win_stats])),
                                         "support": int(np.mean(win_support))})
            print(f"  {backbone}: {len(rows)} rows", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    import csv
    target = OUT / f"three_d_consistency_{args.split}.csv"
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]) if rows else
                                ["split", "sequence", "backbone", "method", "metric", "stat",
                                 "aggregation", "value", "support"])
        writer.writeheader()
        writer.writerows(rows)
    (OUT / f"run_manifest_{args.split}.json").write_text(json.dumps({
        "project": "ARGOS v2", "generated_at": datetime.now(timezone.utc).isoformat(),
        "module": args.module, "module_provenance": adapter.describe(),
        "split": args.split, "sequences": list(sequences), "backbones": list(args.backbones),
        "window_frames": args.window, "grid": [GRID_H, GRID_W], "min_disp_px": MIN_DISP_PX,
        "metrics": {
            "p2p_mm": "|| X_t(u,v) - warp(X_{t-1})(u,v) ||_2 in mm; clouds in their own "
                      "left-camera frames; frame-to-frame surface stability, NOT global drift",
            "normal_deg": "angle between camera-facing central-difference normals of "
                          "corresponding points (previous-frame normals warped nearest)"},
        "protocol": "GT coverage > 0.5 AND raw validity at both frames (previous pulled "
                    "through warp) AND SEA-RAFT forward-backward validity (library defaults); "
                    "identical support for raw/refined/gt; normals additionally 3x3-eroded",
        "alignment": "pair_flow[t-1] = SEA-RAFT infer(frame_t, frame_{t-1}) pulls t-1 onto t",
        "assumptions": ["principal point at image center", "fy == fx (isotropic rescale asserted)",
                        "disparity clamped at 0.1 px before back-projection",
                        "no camera pose used or invented; GT rows are the motion floor"],
        "training_performed": False, "threshold_tuning_performed": False,
        "rows": len(rows),
    }, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "rows": len(rows), "csv": str(target)}, indent=2))


if __name__ == "__main__":
    main()
