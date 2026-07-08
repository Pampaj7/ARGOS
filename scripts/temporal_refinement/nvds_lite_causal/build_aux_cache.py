#!/usr/bin/env python3
"""Precompute per-sequence RGB (256x320) + the RAFT warp flow and target-frame occlusion mask
for consecutive SCARED frames, cached once for NVDS-lite causal training (avoids repeated RAFT
passes in the training loop and keeps RAFT out of the training graph entirely).

Flow convention (validated empirically in validate_flow.py):
  warp_disp(x, flow) samples x at p+flow(p). To pull frame (t-1) content into frame t we need
  BACKWARD flow (t -> t-1). So we store, at index t-1:
      warp_flow[t-1] = RAFT(img[t] -> img[t-1])           # warps frame t-1 into frame t coords
      occ[t-1]       = occlusion in frame t (fwd-bwd consistency, anchored at frame t)
Read-only on raw dataset; writes only under results/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path("/dtu/p1/leopam/ARGOS")
sys.path.insert(0, str(ROOT / "scripts/temporal_refinement/lib"))
from flow import FrozenRAFT, flow_confidence  # noqa: E402

TARGETS = ROOT / "results/03_temporal_refinement/evaluation/s2m2_gt_refiner_targets_full"
DATASET_ROOT = ROOT / "dataset/SCARED/curated/temporal_gt_rectified"
RAFT_CKPT = ROOT / "external/frame_stereo_repos/RAFT/checkpoints/raft-things.pth"
OUT = ROOT / "results/03_temporal_refinement/nvds_lite_causal_pilot/aux_cache"
H, W = 256, 320


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raft = FrozenRAFT(RAFT_CKPT).to(device).eval()
    shard_files = sorted(TARGETS.glob("targets/*.npz"))
    for sf in shard_files:
        seq = sf.stem
        out_f = OUT / f"{seq}.npz"
        if out_f.exists():
            # validate it has the new field; if it's an old fwd-flow cache, rebuild
            try:
                if "warp_flow" in np.load(out_f).files:
                    print("skip", seq, flush=True)
                    continue
            except Exception:
                pass
        z = np.load(sf)
        frame_ids = z["frame_id"]
        T = len(frame_ids)
        imgs = []
        for fid in frame_ids:
            p = DATASET_ROOT / seq / "left" / f"{fid}.png"
            img = cv2.imread(str(p))
            if img is None:
                raise RuntimeError(f"missing frame {p}")
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        rgb = np.stack(imgs).astype(np.uint8)  # T,H,W,3
        imgt = torch.from_numpy(rgb).permute(0, 3, 1, 2).float().to(device) / 255.0
        warp_flow = np.zeros((max(T - 1, 0), 2, H, W), dtype=np.float16)
        occ = np.zeros((max(T - 1, 0), H, W), dtype=np.uint8)
        with torch.no_grad():
            for t in range(1, T):
                f_warp = raft(imgt[t:t + 1], imgt[t - 1:t])   # t -> t-1 (warps frame t-1 into t)
                f_other = raft(imgt[t - 1:t], imgt[t:t + 1])  # t-1 -> t
                _, o = flow_confidence(f_warp, f_other)        # occlusion in frame t
                warp_flow[t - 1] = f_warp[0].cpu().numpy().astype(np.float16)
                occ[t - 1] = (o[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
        np.savez_compressed(out_f, rgb=rgb, warp_flow=warp_flow, occ=occ)
        print(seq, "done", T, "frames", flush=True)
    print("AUX_CACHE_DONE")


if __name__ == "__main__":
    raise SystemExit(main())
