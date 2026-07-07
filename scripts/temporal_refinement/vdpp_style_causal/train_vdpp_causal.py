#!/usr/bin/env python3
"""Minimal VDPP-style causal residual refiner with explicit Temporal Gradient Matching (TGM).

Question: was EGBM failing mainly for lack of dense temporal supervision? Train a small causal
refiner on SCARED consecutive clips with per-frame GT and an explicit TGM loss, plus mandatory
current-frame-only and shuffled-history ablations. S2M2 frozen; disparity-only; no optical flow.

Model: per-frame 5-ch geometric input (raw, gx, gy, edge, valid; ÷DISP_SCALE) -> shared conv
encoder -> causal ConvGRU over the clip -> bounded residual (scale·tanh). refined = raw + res.

Two independent axes (decoupled to avoid confounding loss supervision with temporal structure):
  --temporal-input-mode  full_history | current_frame_only | shuffled_history
  --loss-mode            spatial_only | spatial_plus_tgm
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import sys
ROOT = Path("/dtu/p1/leopam/ARGOS")
for p in ("scripts/temporal_refinement", "scripts/temporal_refinement/ood/eval"):
    sys.path.insert(0, str(ROOT / p))
from evaluate_ood_refiners import frame_metrics, edge_map, EPS  # noqa: E402

DISP_SCALE = 64.0
TARGETS = ROOT / "results/03_temporal_refinement/evaluation/s2m2_gt_refiner_targets_full"
SPLIT = ROOT / "results/03_temporal_refinement/training/refiner_failure_analysis/proposed_balanced_split.json"
OUT = ROOT / "results/03_temporal_refinement/vdpp_style_causal_pilot"


class ConvGRUCell(nn.Module):
    def __init__(self, ch, hid):
        super().__init__()
        self.z = nn.Conv2d(ch + hid, hid, 3, padding=1)
        self.r = nn.Conv2d(ch + hid, hid, 3, padding=1)
        self.h = nn.Conv2d(ch + hid, hid, 3, padding=1)

    def forward(self, x, h):
        zr = torch.cat([x, h], 1)
        z = torch.sigmoid(self.z(zr)); r = torch.sigmoid(self.r(zr))
        q = torch.tanh(self.h(torch.cat([x, r * h], 1)))
        return (1 - z) * h + z * q


class VDPPCausal(nn.Module):
    def __init__(self, hid=96, res_scale=3.0):
        super().__init__()
        self.res_scale = res_scale
        self.enc = nn.Sequential(
            nn.Conv2d(5, hid, 3, padding=1), nn.GroupNorm(8, hid), nn.SiLU(True),
            nn.Conv2d(hid, hid, 3, padding=1), nn.GroupNorm(8, hid), nn.SiLU(True),
            nn.Conv2d(hid, hid, 3, padding=1), nn.GroupNorm(8, hid), nn.SiLU(True),
        )
        self.gru = ConvGRUCell(hid, hid)
        self.head = nn.Sequential(nn.Conv2d(hid, hid, 3, padding=1), nn.SiLU(True), nn.Conv2d(hid, 1, 1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def feat(self, raw, valid):
        gx = torch.zeros_like(raw); gy = torch.zeros_like(raw)
        gx[..., 1:] = raw[..., 1:] - raw[..., :-1]
        gy[..., 1:, :] = raw[..., 1:, :] - raw[..., :-1, :]
        edge = torch.sqrt(gx * gx + gy * gy)
        x = torch.cat([raw / DISP_SCALE, gx / DISP_SCALE, gy / DISP_SCALE, edge / DISP_SCALE, valid], 1)
        return self.enc(x)

    def forward(self, raw_clip, valid_clip, temporal_mode="full_history"):
        """raw_clip/valid_clip: [B,T,1,H,W]. Returns refined [B,T,1,H,W].

        temporal_mode decouples HOW causal history reaches each output frame t (loss is
        applied separately). Target frame t is always the LAST encoded input and its output
        position is fixed -> GT[t] and GT ordering are preserved; no future frames are used.
          full_history        streaming GRU over 0..t (true order)
          current_frame_only  hidden reset each frame -> output uses only frame t (no memory)
          shuffled_history    history frames 0..t-1 permuted, current frame t applied last
        """
        B, T = raw_clip.shape[:2]
        feats = [self.feat(raw_clip[:, t], valid_clip[:, t]) for t in range(T)]
        refined = [None] * T
        if temporal_mode == "full_history":
            h = torch.zeros_like(feats[0])
            for t in range(T):
                h = self.gru(feats[t], h)
                refined[t] = raw_clip[:, t] + self.res_scale * torch.tanh(self.head(h))
        elif temporal_mode == "current_frame_only":
            for t in range(T):
                h = self.gru(feats[t], torch.zeros_like(feats[t]))
                refined[t] = raw_clip[:, t] + self.res_scale * torch.tanh(self.head(h))
        elif temporal_mode == "shuffled_history":
            for t in range(T):
                hist = list(range(t))
                np.random.shuffle(hist)                      # corrupt only history order
                h = torch.zeros_like(feats[t])
                for j in hist:
                    h = self.gru(feats[j], h)
                h = self.gru(feats[t], h)                    # current frame applied last
                refined[t] = raw_clip[:, t] + self.res_scale * torch.tanh(self.head(h))
        else:
            raise ValueError(temporal_mode)
        return torch.stack(refined, 1)


# ----------------------------- data -----------------------------
def load_split_shards(split_name):
    idx = list(csv.DictReader((TARGETS / "frame_targets_index.csv").open()))
    sp = json.loads(SPLIT.read_text())
    seqs = [s for s in sp[split_name]]
    shards = {}
    for s in set(r["sequence_id"] for r in idx if r["sequence_id"] in seqs):
        z = np.load(TARGETS / "targets" / f"{s}.npz")
        shards[s] = {k: z[k] for k in ("raw_disp", "gt_disp", "valid_mask")}
    return shards


def sample_clip(shards, clen, rng):
    s = rng.choice(list(shards.keys()))
    T = shards[s]["raw_disp"].shape[0]
    if T < clen:
        start = 0; clen = T
    else:
        start = rng.integers(0, T - clen + 1)
    sl = slice(start, start + clen)
    raw = shards[s]["raw_disp"][sl].astype(np.float32)
    gt = shards[s]["gt_disp"][sl].astype(np.float32)
    v = (shards[s]["valid_mask"][sl] > 0).astype(np.float32)
    return raw, gt, v


# ----------------------------- losses ---------------------------
def clip_losses(refined, raw, gt, valid, lam_tgm, lam_good, use_tgm):
    m = valid > 0.5
    l_fit = (refined - gt).abs()[m].mean() if m.any() else torch.zeros((), device=refined.device)
    applied = refined - raw
    good = m & (((raw - gt).abs()) < 1.0)
    l_good = applied.abs()[good].mean() if good.any() else torch.zeros((), device=refined.device)
    loss = l_fit + lam_good * l_good
    l_tgm = torch.zeros((), device=refined.device)
    if use_tgm:
        T = refined.shape[1]
        terms = []
        for t in range(1, T):
            mm = (valid[:, t] > 0.5) & (valid[:, t - 1] > 0.5)
            if mm.any():
                dref = refined[:, t] - refined[:, t - 1]
                dgt = gt[:, t] - gt[:, t - 1]
                terms.append((dref - dgt).abs()[mm].mean())
        if terms:
            l_tgm = torch.stack(terms).mean()
        loss = loss + lam_tgm * l_tgm
    return loss, {"fit": float(l_fit), "good": float(l_good), "tgm": float(l_tgm)}


# ----------------------------- eval -----------------------------
@torch.no_grad()
def eval_sequences(model, shards, device, temporal_mode, clen=8):
    """Causal eval over full sequences (streaming hidden). Geometric + temporal metrics."""
    geo, temporal = [], []
    # window-based causal eval (clip length clen) so shuffled_history stays O(clen^2), not O(T^2);
    # matches the training clip distribution. Non-overlapping windows -> each frame scored once.
    for s, sh in shards.items():
        raw = sh["raw_disp"].astype(np.float32); gt = sh["gt_disp"].astype(np.float32)
        v = (sh["valid_mask"] > 0).astype(np.float32)
        T = raw.shape[0]
        for st in range(0, T, clen):
            sl = slice(st, min(st + clen, T))
            if sl.stop - sl.start < 2:
                continue
            rr = raw[sl]; gg = gt[sl]; vv = v[sl]; L = rr.shape[0]
            rt = torch.from_numpy(rr).to(device)[None, :, None]
            vt = torch.from_numpy(vv).to(device)[None, :, None]
            ref = model(rt, vt, temporal_mode=temporal_mode)[0, :, 0].cpu().numpy()
            for t in range(L):
                m = vv[t] > 0.5
                if m.sum():
                    geo.append(frame_metrics(rr[t], ref[t], gg[t], m > 0, edge_map(rr[t])))
            for t in range(1, L):
                m = (vv[t] > 0.5) & (vv[t - 1] > 0.5)
                if m.sum() == 0:
                    continue
                er = np.abs(ref[t] - gg[t]); erp = np.abs(ref[t - 1] - gg[t - 1])
                dref = ref[t] - ref[t - 1]; dgt = gg[t] - gg[t - 1]
                tr = {"tgm_error": float(np.abs(dref - dgt)[m].mean()),
                      "terr_jitter": float(np.abs(er - erp)[m].mean())}
                eb = (edge_map(ref[t]) > 1.0) & m
                if eb.any():
                    tr["boundary_tgm"] = float(np.abs(dref - dgt)[eb].mean())
                temporal.append(tr)
            for t in range(2, L):
                m = (vv[t] > 0.5) & (vv[t - 1] > 0.5) & (vv[t - 2] > 0.5)
                if m.any():
                    er = np.abs(ref[t] - gg[t]); erp = np.abs(ref[t - 1] - gg[t - 1]); erpp = np.abs(ref[t - 2] - gg[t - 2])
                    temporal.append({"hf_error_energy": float(np.abs(er - 2 * erp + erpp)[m].mean())})
    def agg(rows, keys):
        out = {}
        for k in keys:
            vals = [r[k] for r in rows if k in r and r[k] == r[k]]
            out[k] = float(np.mean(vals)) if vals else float("nan")
        return out
    g = agg(geo, ["refined_mae", "raw_mae", "delta_mae", "refined_bad1", "refined_bad3", "refined_bad5",
                  "new_bad3_pct_of_rawgood", "harmful_rate", "modified_pixel_ratio", "beneficial_rate"])
    tm = agg(temporal, ["tgm_error", "terr_jitter", "boundary_tgm", "hf_error_energy"])
    return g, tm


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--temporal-input-mode", default="full_history",
                    choices=["full_history", "current_frame_only", "shuffled_history"])
    ap.add_argument("--loss-mode", default="spatial_plus_tgm",
                    choices=["spatial_only", "spatial_plus_tgm"])
    ap.add_argument("--clip-len", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lam-tgm", type=float, default=1.0)
    ap.add_argument("--lam-good", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--out", type=Path, default=OUT / "runs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    rng = np.random.default_rng(args.seed)
    if args.smoke:
        args.steps = 20; args.eval_every = 20

    train = load_split_shards("train"); val = load_split_shards("val"); test = load_split_shards("test")
    model = VDPPCausal().to(device)
    params = sum(p.numel() for p in model.parameters())
    use_tgm = args.loss_mode == "spatial_plus_tgm"
    tmode = args.temporal_input_mode
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    run_id = f"{tmode}__{args.loss_mode}__lam{args.lam_tgm}__clip{args.clip_len}__seed{args.seed}"
    out = args.out / run_id; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); log = []
    best = (1e9, None)
    for step in range(1, args.steps + 1):
        model.train()
        batch = [sample_clip(train, args.clip_len, rng) for _ in range(args.batch)]
        raw = torch.from_numpy(np.stack([b[0] for b in batch]))[:, :, None].to(device)
        gt = torch.from_numpy(np.stack([b[1] for b in batch]))[:, :, None].to(device)
        v = torch.from_numpy(np.stack([b[2] for b in batch]))[:, :, None].to(device)
        with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            refined = model(raw, v, temporal_mode=tmode)
            loss, parts = clip_losses(refined, raw, gt, v, args.lam_tgm, args.lam_good, use_tgm)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            g, tm = eval_sequences(model, val, device, tmode, args.clip_len)
            score = g["refined_mae"] + 0.02 * g["new_bad3_pct_of_rawgood"] + 0.5 * g["harmful_rate"]
            log.append({"step": step, "loss": float(loss), **parts, "val_mae": g["refined_mae"],
                        "val_tgm": tm["tgm_error"], "val_newbad3": g["new_bad3_pct_of_rawgood"], "val_score": score})
            print(f"[{run_id}] step{step} loss={float(loss):.4f} val_mae={g['refined_mae']:.4f} "
                  f"val_tgm={tm['tgm_error']:.4f} score={score:.4f}", flush=True)
            if score < best[0]:
                best = (score, {k: v.detach().clone() for k, v in model.state_dict().items()})
    if best[1] is not None:
        model.load_state_dict(best[1])
        torch.save({"model_state_dict": best[1], "args": vars(args), "params": params}, out / "best.pt")
    model.eval()
    gt_g, gt_tm = eval_sequences(model, test, device, tmode, args.clip_len)
    cfg = {"run_id": run_id, "temporal_input_mode": tmode, "loss_mode": args.loss_mode, "lam_tgm": args.lam_tgm, "clip_len": args.clip_len, "params": params,
           "steps": args.steps, "lam_tgm": args.lam_tgm, "wall_time_s": round(time.time() - t0, 1),
           "test_geometric": {k: round(v, 4) for k, v in gt_g.items()},
           "test_temporal": {k: round(v, 4) for k, v in gt_tm.items()}}
    (out / "config.json").write_text(json.dumps(cfg, indent=2, default=float) + "\n")
    with (out / "train_log.csv").open("w", newline="") as f:
        if log:
            w = csv.DictWriter(f, fieldnames=list(log[0].keys())); w.writeheader(); w.writerows(log)
    print(json.dumps({"run_id": run_id, "params": params, **cfg["test_geometric"],
                      "test_tgm_error": gt_tm["tgm_error"], "test_terr_jitter": gt_tm["terr_jitter"]},
                     indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
