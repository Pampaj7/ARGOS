#!/usr/bin/env python3
"""Zero-shot D4D eval for the VDPP causal model: sparse Zivid-anchor MAE + full-clip
prediction-space temporal diagnostics. Reuses temporal_eval_d4d building blocks (S2M2 + RAFT).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/dtu/p1/leopam/ARGOS")
for p in ("scripts/temporal_refinement/adaptation", "scripts/temporal_refinement/vdpp_style_causal",
          "scripts/temporal_refinement/ood/eval", "scripts/temporal_refinement/ood/d4d"):
    sys.path.insert(0, str(ROOT / p))
import temporal_eval_d4d as TE  # noqa: E402
from train_vdpp_causal import VDPPCausal  # noqa: E402
from evaluate_ood_refiners import frame_metrics, edge_map, load_samples_and_shards  # noqa: E402
from train_tiny_refiner_v3_1_staged_abstention import FullFrameDataset  # noqa: E402

OUT = ROOT / "results/03_temporal_refinement/vdpp_style_causal_pilot"


def load_vdpp(ckpt, device):
    m = VDPPCausal()
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model_state_dict"]); return m.to(device).eval()


@torch.no_grad()
def vdpp_causal_clip(model, raw_seq, valid_seq, device, temporal_mode="full_history", clen=8):
    """Causal refine of a clip in non-overlapping windows of clen (bounds shuffled cost)."""
    T = raw_seq.shape[0]
    out = np.zeros_like(raw_seq)
    for st in range(0, T, clen):
        sl = slice(st, min(st + clen, T))
        rt = torch.from_numpy(raw_seq[sl]).to(device)[None, :, None]
        vt = torch.from_numpy(valid_seq[sl]).to(device)[None, :, None]
        out[sl] = model(rt, vt, temporal_mode=temporal_mode)[0, :, 0].cpu().numpy()
    return out, out - raw_seq


@torch.no_grad()
def anchor_eval(model, device, tmode="full_history"):
    samples, shards, idx = load_samples_and_shards(ROOT / "results/03_temporal_refinement/ood/d4d_s2m2_zero_shot/d4d_index.csv")
    meta = {r["sequence_id"]: r for r in idx}
    ds = FullFrameDataset(samples, shards, 4)
    rows = []
    for i in range(len(ds)):
        it = ds[i]
        raw = it["raw"][0].numpy().astype(np.float32); gt = it["gt"][0].numpy().astype(np.float32)
        valid = it["valid"][0].numpy() > 0.5
        if valid.sum() == 0:
            continue
        # build 4-frame causal clip from the shard raw window (already temporal order in shard)
        z = shards[samples[i].target_path]
        raw_seq = z["raw_disp"][:samples[i].offset + 1].astype(np.float32)
        vseq = (z["valid_mask"][:samples[i].offset + 1] > 0).astype(np.float32)
        vseq = (np.isfinite(raw_seq) & (raw_seq > 0)).astype(np.float32)
        refined_seq, _ = vdpp_causal_clip(model, raw_seq, vseq, device, tmode)
        refined = refined_seq[samples[i].offset]
        d = frame_metrics(raw, refined, gt, valid, edge_map(raw))
        d["specimen"] = meta[samples[i].sequence_id]["specimen"]
        rows.append(d)
    def wm(k):
        v=[r[k] for r in rows if k in r and r[k]==r[k]]; return float(np.mean(v)) if v else float("nan")
    return {"anchors": len(rows), "mae": wm("refined_mae"), "raw_mae": wm("raw_mae"),
            "delta_mae": wm("delta_mae"), "new_bad3": wm("new_bad3_pct_of_rawgood"),
            "harmful": wm("harmful_rate"), "modified": wm("modified_pixel_ratio")}, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--temporal-mode", default="full_history")
    ap.add_argument("--clips-per-specimen", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_vdpp(args.ckpt, device)
    s2m2 = TE.build_model(device, "S")
    raft = TE.FrozenRAFT(TE.RAFT_CKPT).to(device).eval()

    # sparse anchor geometric
    tmode = args.temporal_mode
    ag, arows = anchor_eval(model, device, tmode)
    (args.out / "d4d_anchor_metrics.csv").parent.mkdir(parents=True, exist_ok=True)
    with (args.out / "d4d_anchor_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in arows for k in r})); w.writeheader(); w.writerows(arows)

    # full-clip temporal (reuse TE.process_clip + TE.temporal_metrics)
    clips = []
    for spec in ["specimen_1", "specimen_2", "specimen_3"]:
        sr = TE.session_root(spec)
        picked = 0
        for s in [x for x in sorted(sr.glob("*")) if x.is_dir() and (x / "clips.json").exists()]:
            for c in json.loads((s / "clips.json").read_text()).get("clips", []):
                clips.append((spec, s.name, c)); picked += 1
                if picked >= args.clips_per_specimen:
                    break
            if picked >= args.clips_per_specimen:
                break
    trows = []
    for spec, sess, clip in clips:
        base = TE.process_clip(spec, sess, clip, s2m2, raft, device, args.max_frames)
        if base is None:
            continue
        for cfg_name, do_refine in [("raw", False), ("vdpp_tgm", True)]:
            if do_refine:
                disp_seq, applied = vdpp_causal_clip(model, base["raw_seq"], base["valid_seq"], device, tmode)
                gate = damp = None
            else:
                disp_seq, applied, gate, damp = base["raw_seq"], None, None, None
            m = TE.temporal_metrics(disp_seq, base["valid_seq"], base["flows"], base["occs"], applied,
                                    gate, damp, base["fx"], base["base_mm"], device)
            m.update({"config": cfg_name, "specimen": spec, "clip": clip["name"], "frames": len(base["frames"])})
            trows.append(m)
        print(f"  {spec}/{clip['name']} done", flush=True)
    with (args.out / "d4d_temporal_metrics.csv").open("w", newline="") as f:
        keys = sorted({k for r in trows for k in r})
        w = csv.DictWriter(f, fieldnames=["config", "specimen", "clip", "frames"] + [k for k in keys if k not in ("config", "specimen", "clip", "frames")])
        w.writeheader(); w.writerows(trows)
    # aggregate temporal by config
    from collections import defaultdict
    g = defaultdict(list)
    for r in trows:
        g[r["config"]].append(r)
    tagg = {}
    for cfg, rr in g.items():
        tagg[cfg] = {k: round(float(np.mean([r[k] for r in rr if isinstance(r.get(k), float) and r[k] == r[k]])), 4)
                     for k in ("mc_inconsistency", "hf_energy", "depth_mc_mm", "boundary_mc", "modified_ratio")}
    (args.out / "d4d_vdpp_summary.json").write_text(json.dumps({"anchor": ag, "temporal": tagg}, indent=2, default=float) + "\n")
    print(json.dumps({"anchor": ag, "temporal": tagg}, indent=2, default=float))


if __name__ == "__main__":
    raise SystemExit(main())
