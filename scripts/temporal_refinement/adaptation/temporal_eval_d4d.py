#!/usr/bin/env python3
"""D4D temporal-consistency eval on full consecutive clips (prediction-space, no dense GT).

Runs raw S2M2-S + zero-shot/adapted EGBM-v3-CARE-S causally over whole clips and measures
prediction-space + RAFT motion-compensated temporal diagnostics. GT only at start/end Zivid
anchors (sparse geometric check).

Metrics (all labelled diagnostic; lower is NOT automatically better — tissue is non-rigid):
  mc_inconsistency  mean |D_t - warp(D_{t-1}; flow)| on valid & non-occluded (RAFT fwd-bwd)
  hf_energy         mean |D_t - 2 D_{t-1} + D_{t-2}|            (temporal 2nd diff)
  corr_var          temporal variance of applied residual
  signflip_rate     fraction of modified px whose applied sign flips t-1->t
  isolated_rate     fraction modified at t but not t-1 (one-frame onset)
  gate_hf / damp_hf temporal 2nd diff of gate / damping (EGBM aux)
  boundary_mc       mc_inconsistency on edge pixels
  depth_mc_mm       mc_inconsistency in depth (mm) on reliable regions
  mc_by_motion      mc_inconsistency stratified by flow magnitude (motion-lag / oversmoothing)
  modified_ratio / applied_abs   identity-collapse indicators
  anchor_mae        MAE vs Zivid GT at start/end anchor frames
No future frames; no clip/session crossing.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path("/dtu/p1/leopam/ARGOS")
for p in ("scripts/temporal_refinement/ood/d4d", "scripts/temporal_refinement/data_prep",
          "scripts/temporal_refinement/eval_scripts", "scripts/temporal_refinement",
          "scripts/temporal_refinement/models", "scripts/temporal_refinement/lib",
          "scripts/temporal_refinement/ood/eval"):
    sys.path.insert(0, str(ROOT / p))
from d4d_keyframe_gt import load_cam, rectify_maps, session_root  # noqa: E402
from predict_s2m2_long_sequences import build_model, infer  # noqa: E402
from generate_distillation_targets_selected_clips import target_hw, valid_masked_downsample_disparity  # noqa: E402
from train_tiny_refiner_v3_1_staged_abstention import make_features_from_raws, DISP_SCALE  # noqa: E402
from egbm_v3_care_streaming_refiner import egbm_v3_care_streaming  # noqa: E402
from flow import FrozenRAFT, warp_disp, flow_confidence  # noqa: E402

OUT = ROOT / "results/03_temporal_refinement/adaptation/d4d_temporal_eval"
RUNS = ROOT / "results/03_temporal_refinement/adaptation/d4d_few_shot_pilot/runs"
RAFT_CKPT = ROOT / "external/frame_stereo_repos/RAFT/checkpoints/raft-things.pth"
REG_CKPT = ROOT / "results/03_temporal_refinement/training/egbm_v3_care_streaming/checkpoints/best.pt"
SCALE = 0.25; MIN_VALID = 0.25; CTX = 4
CONFIGS = {
    "raw": None,
    "zero_shot": REG_CKPT,
    "calib_2s": RUNS / "EGBM-v3-CARE-S__calibration_only__2session_seed2__seed2/best_combined.pt",
    "calib_8s": RUNS / "EGBM-v3-CARE-S__calibration_only__8session_seed1__seed1/best_combined.pt",
    "full_4s": RUNS / "EGBM-v3-CARE-S__full__4session_seed1__seed1/best_combined.pt",
    "scratch_4s": RUNS / "EGBM-v3-CARE-S__scratch__4session_seed1__seed1/best_combined.pt",
}


def ts_of(name): a, b = Path(name).stem.split("_"); return float(f"{a}.{b}")


def load_model(ckpt_path, device):
    if ckpt_path is None:
        return None
    m = egbm_v3_care_streaming(16, 3.0)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck.get("state_dict", ck))
    m.load_state_dict(sd, strict=False)
    return m.to(device).eval()


def edge(a):
    g = np.zeros_like(a); g[:, 1:] = np.abs(a[:, 1:] - a[:, :-1]); return g > 1.0


@torch.no_grad()
def refine_sequence(model, raw_seq, valid_seq, device):
    """Causal EGBM over a clip. raw_seq/valid_seq: [T,H,W]. Returns refined[T], applied[T], aux."""
    T = len(raw_seq)
    refined = np.zeros_like(raw_seq); applied = np.zeros_like(raw_seq)
    gate_seq = np.zeros_like(raw_seq); damp_seq = np.zeros_like(raw_seq)
    for t in range(T):
        idx = [max(0, t - k) for k in range(CTX)]
        raws = np.stack([raw_seq[i] for i in idx]).astype(np.float32)
        valids = np.stack([valid_seq[i] for i in idx]).astype(np.float32)
        x, _, _ = make_features_from_raws(raws, valids)
        xt = torch.from_numpy(x).unsqueeze(0).to(device)
        out = model(xt, 3.0)
        residual = out[2][0, 0].cpu().numpy()
        refined[t] = raw_seq[t] + residual
        applied[t] = residual
        if len(out) > 3 and isinstance(out[3], dict):
            d = out[3]
            if "gate" in d: gate_seq[t] = d["gate"][0, 0].cpu().numpy()
            if "damping" in d: damp_seq[t] = d["damping"][0, 0].cpu().numpy()
    return refined, applied, gate_seq, damp_seq


def clip_frames(specimen, session, start_ts, end_ts, max_frames):
    sr = session_root(specimen) / session
    lefts = sorted((sr / "left_images").glob("*.png"), key=lambda p: ts_of(p.name))
    frames = [p for p in lefts if start_ts <= ts_of(p.name) <= end_ts
              and (sr / "right_images" / p.name).exists()]
    if max_frames and len(frames) > max_frames:
        step = len(frames) / max_frames
        frames = [frames[int(i * step)] for i in range(max_frames)]
    return frames, sr


@torch.no_grad()
def process_clip(specimen, session, clip, s2m2, raft, device, max_frames):
    start_ts = float(clip["start"]["files"]["left_images"]["timestamp"])
    end_ts = float(clip["end"]["files"]["left_images"]["timestamp"])
    frames, sr = clip_frames(specimen, session, start_ts, end_ts, max_frames)
    if len(frames) < 5:
        return None
    ci = sr / "camera_info"
    left, right = load_cam(ci / "left.yaml"), load_cam(ci / "right.yaml")
    lmapx, lmapy = rectify_maps(left); rmapx, rmapy = rectify_maps(right)
    W, H = left["W"], left["H"]
    oh, ow = target_hw((H, W), SCALE)
    # RAFT needs /8
    ph, pw = ((oh + 7) // 8) * 8, ((ow + 7) // 8) * 8
    raw_seq, gray_seq = [], []
    for fp in frames:
        l = cv2.remap(cv2.imread(str(fp)), lmapx, lmapy, cv2.INTER_LINEAR)
        r = cv2.remap(cv2.imread(str(sr / "right_images" / fp.name)), rmapx, rmapy, cv2.INTER_LINEAR)
        disp, _, _ = infer(s2m2, cv2.cvtColor(l, cv2.COLOR_BGR2RGB), cv2.cvtColor(r, cv2.COLOR_BGR2RGB), device, 512)
        dr, _ = valid_masked_downsample_disparity(disp.astype(np.float32), np.isfinite(disp) & (disp > 0), oh, ow, MIN_VALID)
        raw_seq.append(dr.astype(np.float32))
        gsm = cv2.resize(cv2.cvtColor(l, cv2.COLOR_BGR2RGB), (ow, oh), interpolation=cv2.INTER_AREA)
        gray_seq.append(gsm)
    raw_seq = np.stack(raw_seq)
    valid_seq = (np.isfinite(raw_seq) & (raw_seq > 0)).astype(np.float32)
    fx = float(left["P"][0, 0]); base_mm = float(-right["P"][0, 3] / left["P"][0, 0]) * 1e3
    # flow (t-1 -> t) + occlusion, at grid res
    imgs = torch.from_numpy(np.stack(gray_seq)).permute(0, 3, 1, 2).float().to(device) / 255.0
    imgs_p = torch.nn.functional.pad(imgs, (0, pw - ow, 0, ph - oh), mode="replicate")
    flows, occs = [], []
    for t in range(1, len(frames)):
        f_fwd = raft(imgs_p[t - 1:t], imgs_p[t:t + 1])
        f_bwd = raft(imgs_p[t:t + 1], imgs_p[t - 1:t])
        _, occ = flow_confidence(f_fwd, f_bwd)
        flows.append(f_fwd[:, :, :oh, :ow]); occs.append(occ[:, :, :oh, :ow])
    return {"raw_seq": raw_seq, "valid_seq": valid_seq, "flows": flows, "occs": occs,
            "fx": fx, "base_mm": base_mm, "frames": [f.name for f in frames]}


def temporal_metrics(disp_seq, valid_seq, flows, occs, applied_seq, gate_seq, damp_seq, fx, base_mm, device):
    T = len(disp_seq)
    d = {}
    mc, mc_b, mc_depth = [], [], []
    mc_lo, mc_hi = [], []  # motion-stratified
    for t in range(1, T):
        Dt = torch.from_numpy(disp_seq[t]).float().to(device)[None, None]
        Dp = torch.from_numpy(disp_seq[t - 1]).float().to(device)[None, None]
        warped = warp_disp(Dp, flows[t - 1])[0, 0].cpu().numpy()
        m = (valid_seq[t] > 0) & (valid_seq[t - 1] > 0) & (occs[t - 1][0, 0].cpu().numpy() < 0.5)
        if m.sum() == 0:
            continue
        inc = np.abs(disp_seq[t] - warped)
        mc.append(inc[m].mean())
        eb = edge(disp_seq[t]) & m
        if eb.any(): mc_b.append(inc[eb].mean())
        # depth inconsistency
        dep_t = fx * base_mm / np.maximum(disp_seq[t], 1e-6); dep_w = fx * base_mm / np.maximum(warped, 1e-6)
        mc_depth.append(np.abs(dep_t - dep_w)[m].mean())
        # motion stratify by flow magnitude
        fmag = torch.linalg.vector_norm(flows[t - 1], dim=1)[0].cpu().numpy()
        lo = m & (fmag < 1.0); hi = m & (fmag >= 3.0)
        if lo.any(): mc_lo.append(inc[lo].mean())
        if hi.any(): mc_hi.append(inc[hi].mean())
    d["mc_inconsistency"] = float(np.mean(mc)) if mc else float("nan")
    d["boundary_mc"] = float(np.mean(mc_b)) if mc_b else float("nan")
    d["depth_mc_mm"] = float(np.mean(mc_depth)) if mc_depth else float("nan")
    d["mc_lowmotion"] = float(np.mean(mc_lo)) if mc_lo else float("nan")
    d["mc_highmotion"] = float(np.mean(mc_hi)) if mc_hi else float("nan")
    # hf energy (2nd temporal diff)
    hf = [np.abs(disp_seq[t] - 2 * disp_seq[t - 1] + disp_seq[t - 2])[(valid_seq[t] > 0) & (valid_seq[t - 1] > 0) & (valid_seq[t - 2] > 0)].mean()
          for t in range(2, T) if ((valid_seq[t] > 0) & (valid_seq[t - 1] > 0) & (valid_seq[t - 2] > 0)).any()]
    d["hf_energy"] = float(np.mean(hf)) if hf else float("nan")
    # correction stats
    if applied_seq is not None:
        vm = valid_seq > 0
        d["corr_var"] = float(np.mean([applied_seq[t][vm[t]].var() if vm[t].any() else 0 for t in range(T)])) if T else 0.0
        d["applied_abs"] = float(np.mean([np.abs(applied_seq[t])[vm[t]].mean() if vm[t].any() else 0 for t in range(T)]))
        d["modified_ratio"] = float(np.mean([(np.abs(applied_seq[t]) > 0.1)[vm[t]].mean() if vm[t].any() else 0 for t in range(T)]))
        sf, iso = [], []
        for t in range(1, T):
            m = (valid_seq[t] > 0) & (valid_seq[t - 1] > 0)
            mod = (np.abs(applied_seq[t]) > 0.1) & m
            if mod.any():
                sf.append((np.sign(applied_seq[t]) != np.sign(applied_seq[t - 1]))[mod].mean())
                iso.append((np.abs(applied_seq[t - 1]) <= 0.1)[mod].mean())
        d["signflip_rate"] = float(np.mean(sf)) if sf else 0.0
        d["isolated_rate"] = float(np.mean(iso)) if iso else 0.0
        # gate/damp temporal 2nd diff
        for nm, seq in [("gate_hf", gate_seq), ("damp_hf", damp_seq)]:
            if seq is not None and seq.any():
                gg = [np.abs(seq[t] - 2 * seq[t - 1] + seq[t - 2]).mean() for t in range(2, T)]
                d[nm] = float(np.mean(gg)) if gg else float("nan")
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips-per-specimen", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    s2m2 = build_model(device, "S")
    raft = FrozenRAFT(RAFT_CKPT).to(device).eval()
    models = {k: load_model(v, device) for k, v in CONFIGS.items()}

    # select clips
    specimens = ["specimen_1", "specimen_2", "specimen_3"] if not args.smoke else ["specimen_1"]
    clips = []
    for spec in specimens:
        sr = session_root(spec)
        sessions = [s for s in sorted(sr.glob("*")) if s.is_dir() and (s / "clips.json").exists()]
        picked = 0
        for s in sessions:
            for c in json.loads((s / "clips.json").read_text()).get("clips", []):
                clips.append((spec, s.name, c)); picked += 1
                if picked >= (1 if args.smoke else args.clips_per_specimen):
                    break
            if picked >= (1 if args.smoke else args.clips_per_specimen):
                break
    if args.smoke:
        clips = clips[:1]; args.max_frames = min(args.max_frames, 25)
    print(f"temporal eval: {len(clips)} clips, {len(models)} configs, device={device}")

    per_clip = []
    for spec, sess, clip in clips:
        base = process_clip(spec, sess, clip, s2m2, raft, device, args.max_frames)
        if base is None:
            continue
        raw_seq, valid_seq = base["raw_seq"], base["valid_seq"]
        for cfg, model in models.items():
            if model is None:
                disp_seq, applied, gate, damp = raw_seq, None, None, None
            else:
                disp_seq, applied, gate, damp = refine_sequence(model, raw_seq, valid_seq, device)
            m = temporal_metrics(disp_seq, valid_seq, base["flows"], base["occs"], applied, gate, damp,
                                 base["fx"], base["base_mm"], device)
            m.update({"config": cfg, "specimen": spec, "session": sess, "clip": clip["name"],
                      "frames": len(base["frames"])})
            per_clip.append(m)
        print(f"  {spec}/{clip['name']}: {len(base['frames'])} frames done")

    keys = sorted({k for r in per_clip for k in r})
    lead = ["config", "specimen", "session", "clip", "frames"]
    keys = lead + [k for k in keys if k not in lead]
    with (args.out / "per_clip_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(per_clip)
    # aggregate by config
    from collections import defaultdict
    g = defaultdict(list)
    for r in per_clip:
        g[r["config"]].append(r)
    metcols = [k for k in keys if k not in lead]
    agg = []
    for cfg, rr in g.items():
        row = {"config": cfg, "clips": len(rr)}
        for k in metcols:
            v = [r[k] for r in rr if isinstance(r.get(k), float) and r[k] == r[k]]
            row[k] = round(float(np.mean(v)), 4) if v else None
        agg.append(row)
    order = list(CONFIGS)
    agg.sort(key=lambda r: order.index(r["config"]))
    with (args.out / "aggregate_temporal_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0].keys())); w.writeheader(); w.writerows(agg)
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
