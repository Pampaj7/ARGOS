#!/usr/bin/env python3
"""Empirical flow-direction + disparity-warp validation for NVDS-lite motion-aware supervision.

warp_disp(x, flow) samples x at p+flow(p) (backward sampling). To warp frame (t-1) content into
frame t coordinates we therefore need flow that, evaluated at a pixel of frame t, points back to
where that pixel came from in t-1 -> i.e. BACKWARD flow (t -> t-1). This script settles the
direction empirically on real SCARED pairs by measuring masked photometric + disparity
reconstruction error for both directions and a no-warp baseline, and validates the warp-support
mask construction (warped previous-valid, in-bounds, occlusion) used by the training warp loss.

Outputs (results/03_temporal_refinement/nvds_lite_causal_pilot/validation/):
  flow_mask_validation.json  + contact_sheet_*.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/dtu/p1/leopam/ARGOS")
sys.path.insert(0, str(ROOT / "scripts/temporal_refinement/lib"))
from flow import FrozenRAFT, warp_disp, flow_confidence  # noqa: E402

TARGETS = ROOT / "results/03_temporal_refinement/evaluation/s2m2_gt_refiner_targets_full"
AUX = ROOT / "results/03_temporal_refinement/nvds_lite_causal_pilot/aux_cache"
RAFT_CKPT = ROOT / "external/frame_stereo_repos/RAFT/checkpoints/raft-things.pth"
OUT = ROOT / "results/03_temporal_refinement/nvds_lite_causal_pilot/validation"
SEQ = "dataset_1_keyframe_1"
N_PAIRS = 24
H, W = 256, 320


def warp_generic(x, flow):
    """warp_disp generalized to C channels: x[p] <- x[p+flow(p)]."""
    B, C, h, w = x.shape
    gy, gx = torch.meshgrid(torch.arange(h, device=x.device, dtype=torch.float32),
                            torch.arange(w, device=x.device, dtype=torch.float32), indexing="ij")
    grid = torch.stack([gx, gy], 0).unsqueeze(0).expand(B, -1, -1, -1) + flow
    nx = 2 * grid[:, 0] / (w - 1) - 1
    ny = 2 * grid[:, 1] / (h - 1) - 1
    gn = torch.stack([nx, ny], -1)
    inb = (nx.abs() <= 1) & (ny.abs() <= 1)
    return F.grid_sample(x, gn, mode="bilinear", padding_mode="border", align_corners=True), inb.unsqueeze(1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raft = FrozenRAFT(RAFT_CKPT).to(device).eval()

    z = np.load(TARGETS / "targets" / f"{SEQ}.npz")
    aux = np.load(AUX / f"{SEQ}.npz")
    rgb = aux["rgb"].astype(np.float32) / 255.0  # T,H,W,3
    raw = z["raw_disp"].astype(np.float32)
    valid = (z["valid_mask"] > 0).astype(np.float32)
    T = min(N_PAIRS + 1, rgb.shape[0])

    imgs = torch.from_numpy(rgb[:T]).permute(0, 3, 1, 2).to(device)  # T,3,H,W

    res = {"flow_convention_in_code": "warp_disp(x, flow) = x[p + flow(p)] (backward grid-sample)",
           "cache_flow_stored": "flow_fwd = RAFT(img[t-1] -> img[t]) (forward, t-1->t)",
           "seq": SEQ, "n_pairs": T - 1}
    photo = {"no_warp": [], "fwd_flow": [], "bwd_flow": []}
    dispmc = {"no_warp": [], "fwd_flow": [], "bwd_flow": []}
    support = {"fwd_flow": [], "bwd_flow": []}
    fb_stats = []
    nan_inf = 0

    for t in range(1, T):
        i_prev, i_cur = imgs[t - 1:t], imgs[t:t + 1]
        f_fwd = raft(i_prev, i_cur)   # t-1 -> t
        f_bwd = raft(i_cur, i_prev)   # t   -> t-1
        conf, occ = flow_confidence(f_fwd, f_bwd)
        fb_err = torch.linalg.vector_norm(f_fwd + warp_generic(f_bwd, f_fwd)[0], dim=1).mean().item()
        fb_stats.append(fb_err)

        Dp = torch.from_numpy(raw[t - 1]).to(device)[None, None]
        Dc = raw[t]
        vc = valid[t] > 0.5
        vp = torch.from_numpy(valid[t - 1]).to(device)[None, None]

        for name, fl in (("fwd_flow", f_fwd), ("bwd_flow", f_bwd)):
            # photometric: warp previous RGB into current with this flow
            wp, inb = warp_generic(i_prev, fl)
            vp_w, _ = warp_generic(vp, fl)
            m = vc & (vp_w[0, 0].cpu().numpy() > 0.5) & (inb[0, 0].cpu().numpy())
            if m.sum() == 0:
                continue
            perr = np.abs(wp[0].permute(1, 2, 0).cpu().numpy() - rgb[t])[m].mean()
            photo[name].append(float(perr))
            # disparity mc: warp previous disparity, compare current
            Dw = warp_disp(Dp, fl)[0, 0].cpu().numpy()
            if not np.isfinite(Dw).all():
                nan_inf += int((~np.isfinite(Dw)).sum())
            dispmc[name].append(float(np.abs(Dc - Dw)[m].mean()))
            support[name].append(float(m.mean()))
        # no-warp baselines
        m0 = vc & (valid[t - 1] > 0.5)
        photo["no_warp"].append(float(np.abs(rgb[t - 1] - rgb[t])[m0].mean()))
        dispmc["no_warp"].append(float(np.abs(raw[t - 1] - Dc)[m0].mean()))

        if t <= 4:  # contact sheets for first few pairs
            wf, _ = warp_generic(i_prev, f_fwd)
            wb, _ = warp_generic(i_prev, f_bwd)
            row = np.concatenate([
                (rgb[t - 1] * 255), (rgb[t] * 255),
                (wf[0].permute(1, 2, 0).cpu().numpy() * 255),
                (wb[0].permute(1, 2, 0).cpu().numpy() * 255),
            ], axis=1).astype(np.uint8)
            cv2.imwrite(str(OUT / f"contact_sheet_pair{t}.png"), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))

    def stat(d):
        return {k: {"mean": float(np.mean(v)), "median": float(np.median(v))} if v else None for k, v in d.items()}

    res["photometric_error_masked"] = stat(photo)
    res["disparity_mc_inconsistency_px"] = stat(dispmc)
    res["warp_support_fraction"] = stat(support)
    res["fb_consistency_px"] = {"mean": float(np.mean(fb_stats)), "median": float(np.median(fb_stats))}
    res["nan_inf_in_warped_disp"] = nan_inf
    # decision: pick the flow direction with lower masked photometric error (and cross-check disp mc)
    pf = np.mean(photo["fwd_flow"]); pb = np.mean(photo["bwd_flow"]); p0 = np.mean(photo["no_warp"])
    df = np.mean(dispmc["fwd_flow"]); db = np.mean(dispmc["bwd_flow"]); d0 = np.mean(dispmc["no_warp"])
    best = "bwd_flow" if pb < pf else "fwd_flow"
    res["contact_sheet_layout"] = "columns: prev | current | warp(prev, fwd_flow) | warp(prev, bwd_flow)"
    res["decision"] = {
        "photometric": {"no_warp": float(p0), "fwd_flow": float(pf), "bwd_flow": float(pb)},
        "disparity_mc": {"no_warp": float(d0), "fwd_flow": float(df), "bwd_flow": float(db)},
        "chosen_direction": best,
        "warp_helps_vs_nowarp": bool(min(pf, pb) < p0),
        "note": ("warp_disp needs flow(t->t-1) to pull frame t-1 into frame t. RAFT(t-1->t) is "
                 "'fwd_flow'; RAFT(t->t-1) is 'bwd_flow'. The direction with LOWER masked "
                 "photometric error is the correct one to feed warp_disp for the warp loss."),
    }
    (OUT / "flow_mask_validation.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res["decision"], indent=2))
    print("wrote", OUT / "flow_mask_validation.json")


if __name__ == "__main__":
    main()
