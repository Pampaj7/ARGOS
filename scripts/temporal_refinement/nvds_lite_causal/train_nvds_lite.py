#!/usr/bin/env python3
"""Causal NVDS-lite pilot: explicit target-to-history local-correlation matching + RGB-D
features + motion-aware (RAFT-warp) supervision, testing whether genuine causal temporal
refinement (as opposed to implicit recurrent memory over disparity-only input) transfers OOD.

Prior findings this continues from (see results/03_temporal_refinement/vdpp_style_causal_confirmation
and .../ood/): unrestricted residuals overfit SCARED; calibration recovers safety by collapsing
toward identity; TGM regularizes but a disparity-only ConvGRU does not robustly exploit causal
history; D4D temporal transfer stays flat. New hypothesis: genuine temporal refinement needs
explicit current-to-history correspondence, not implicit recurrent state.

Six fixed configs (see CONFIGS below), 3 seeds each, clip_len=8, on frozen S2M2-S raw
disparity + RAFT flow/occlusion cached ahead of time (build_aux_cache.py). S2M2 and RAFT are
never in the training graph (flow/rgb/disp are precomputed/cached to disk); only this refiner
is trained.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/dtu/p1/leopam/ARGOS")
for p in ("scripts/temporal_refinement", "scripts/temporal_refinement/ood/eval",
          "scripts/temporal_refinement/lib", "scripts/temporal_refinement/nvds_lite_causal"):
    sys.path.insert(0, str(ROOT / p))
from evaluate_ood_refiners import frame_metrics, edge_map  # noqa: E402
from model import build_model, DISP_SCALE  # noqa: E402

TARGETS = ROOT / "results/03_temporal_refinement/evaluation/s2m2_gt_refiner_targets_full"
SPLIT = ROOT / "results/03_temporal_refinement/training/refiner_failure_analysis/proposed_balanced_split.json"
AUX_CACHE = ROOT / "results/03_temporal_refinement/nvds_lite_causal_pilot/aux_cache"
OUT = ROOT / "results/03_temporal_refinement/nvds_lite_causal_pilot"

# A-F pilot matrix. Fixed lambdas shared across configs where the loss is mathematically
# applicable, so any performance gap is attributable to the swept factor (architecture /
# RGB / temporal-input-mode / warp-loss), not to a different loss recipe.
LAM_DEFAULT = dict(lam_tgm=1.0, lam_warp=1.0, lam_safe=0.2, lam_sparse=0.02)
CONFIGS = {
    "A": dict(model="nvds_lite", use_rgb=True, temporal_mode="full_history", **LAM_DEFAULT),
    "B": dict(model="nvds_lite", use_rgb=True, temporal_mode="full_history", **{**LAM_DEFAULT, "lam_warp": 0.0}),
    "C": dict(model="nvds_lite", use_rgb=False, temporal_mode="full_history", **LAM_DEFAULT),
    "D": dict(model="nvds_lite", use_rgb=True, temporal_mode="current_frame_only", **LAM_DEFAULT),
    "E": dict(model="nvds_lite", use_rgb=True, temporal_mode="shuffled_history", **LAM_DEFAULT),
    "F": dict(model="concat_baseline", use_rgb=True, temporal_mode="full_history", **LAM_DEFAULT),
}


# ----------------------------- data -----------------------------
def load_split_shards(split_name):
    idx = list(csv.DictReader((TARGETS / "frame_targets_index.csv").open()))
    sp = json.loads(SPLIT.read_text())
    seqs = [s for s in sp[split_name]]
    shards = {}
    for s in set(r["sequence_id"] for r in idx if r["sequence_id"] in seqs):
        z = np.load(TARGETS / "targets" / f"{s}.npz")
        aux = np.load(AUX_CACHE / f"{s}.npz")
        shards[s] = {
            "raw_disp": z["raw_disp"], "gt_disp": z["gt_disp"], "valid_mask": z["valid_mask"],
            "rgb": aux["rgb"], "warp_flow": aux["warp_flow"], "occ": aux["occ"],
        }
    return shards


def sample_clip(shards, clen, rng):
    s = rng.choice(list(shards.keys()))
    sh = shards[s]
    T = sh["raw_disp"].shape[0]
    L = min(clen, T)
    start = rng.integers(0, T - L + 1) if T > L else 0
    sl = slice(start, start + L)
    raw = sh["raw_disp"][sl].astype(np.float32)
    gt = sh["gt_disp"][sl].astype(np.float32)
    v = (sh["valid_mask"][sl] > 0).astype(np.float32)
    rgb = sh["rgb"][sl].astype(np.float32) / 255.0
    flow = sh["warp_flow"][start:start + L - 1].astype(np.float32) if L > 1 else np.zeros((0, 2) + raw.shape[1:], np.float32)
    occ = sh["occ"][start:start + L - 1].astype(np.float32) if L > 1 else np.zeros((0,) + raw.shape[1:], np.float32)
    return raw, gt, v, rgb, flow, occ


def collate(batch, device):
    raw = torch.from_numpy(np.stack([b[0] for b in batch]))[:, :, None].to(device)
    gt = torch.from_numpy(np.stack([b[1] for b in batch]))[:, :, None].to(device)
    v = torch.from_numpy(np.stack([b[2] for b in batch]))[:, :, None].to(device)
    rgb = torch.from_numpy(np.stack([b[3] for b in batch])).permute(0, 1, 4, 2, 3).to(device)
    flow = torch.from_numpy(np.stack([b[4] for b in batch])).to(device)
    occ = torch.from_numpy(np.stack([b[5] for b in batch]))[:, :, None].to(device)
    return raw, gt, v, rgb, flow, occ


# ----------------------------- losses ---------------------------
BORDER = 4  # px margin excluded from the warp-consistency loss (unreliable near frame edges)


def warp_with_support(x, flow):
    """x[p] <- x[p + flow(p)] (matches lib.flow.warp_disp). flow is the backward warp flow
    (target t -> source t-1) so this pulls frame t-1 content into frame t. Returns
    (warped, inbounds[B,1,H,W]) where inbounds marks pixels whose sample stayed in-frame."""
    B, C, h, w = x.shape
    gy, gx = torch.meshgrid(torch.arange(h, device=x.device, dtype=torch.float32),
                            torch.arange(w, device=x.device, dtype=torch.float32), indexing="ij")
    grid = torch.stack([gx, gy], 0).unsqueeze(0).expand(B, -1, -1, -1) + flow
    nx = 2 * grid[:, 0] / (w - 1) - 1
    ny = 2 * grid[:, 1] / (h - 1) - 1
    gn = torch.stack([nx, ny], -1)
    inb = ((nx.abs() <= 1) & (ny.abs() <= 1)).unsqueeze(1).float()
    return F.grid_sample(x, gn, mode="bilinear", padding_mode="border", align_corners=True), inb


def clip_losses(refined, raw, gt, valid, flow, occ, lam):
    B, T, _, H, W = refined.shape
    m = valid > 0.5
    l_spatial = (refined - gt).abs()[m].mean() if m.any() else refined.new_zeros(())
    applied = refined - raw
    good = m & ((raw - gt).abs() < 1.0)
    l_safe = applied.abs()[good].mean() if good.any() else refined.new_zeros(())
    l_sparse = applied.abs()[m].mean() if m.any() else refined.new_zeros(())

    l_tgm = refined.new_zeros(())
    tgm_terms = []
    for t in range(1, T):
        mm = (valid[:, t] > 0.5) & (valid[:, t - 1] > 0.5)
        if mm.any():
            dref = refined[:, t] - refined[:, t - 1]
            dgt = gt[:, t] - gt[:, t - 1]
            tgm_terms.append((dref - dgt).abs()[mm].mean())
    if tgm_terms:
        l_tgm = torch.stack(tgm_terms).mean()

    # Motion-aware warp consistency. flow is the backward warp flow (t -> t-1); warp_with_support
    # pulls frame t-1 into frame t. Support mask uses: current valid, the WARPED previous valid
    # (never the raw unwarped valid[t-1]), the in-bounds mask, target-frame occlusion, and a border
    # margin. A naive image-flow disparity warp is only approximate under viewpoint change, so the
    # dense GT temporal loss (L_tgm) is the primary supervised temporal signal; this term is a
    # self-consistency regularizer masked to reliable regions.
    l_warp = refined.new_zeros(())
    warp_terms = []
    border_mask = torch.zeros(1, 1, H, W, device=refined.device)
    border_mask[:, :, BORDER:H - BORDER, BORDER:W - BORDER] = 1.0
    for t in range(1, T):
        warped_prev, inb = warp_with_support(refined[:, t - 1], flow[:, t - 1])
        vp_warped, _ = warp_with_support(valid[:, t - 1], flow[:, t - 1])
        support = ((valid[:, t] > 0.5) & (vp_warped > 0.5) & (inb > 0.5)
                   & (occ[:, t - 1] < 0.5) & (border_mask > 0.5))
        if support.any():
            warp_terms.append((refined[:, t] - warped_prev).abs()[support].mean())
    if warp_terms:
        l_warp = torch.stack(warp_terms).mean()

    loss = (l_spatial + lam["lam_tgm"] * l_tgm + lam["lam_warp"] * l_warp
            + lam["lam_safe"] * l_safe + lam["lam_sparse"] * l_sparse)
    parts = {"spatial": float(l_spatial), "tgm": float(l_tgm), "warp": float(l_warp),
              "safe": float(l_safe), "sparse": float(l_sparse)}
    return loss, parts


# ----------------------------- eval -----------------------------
@torch.no_grad()
def eval_sequences(model, shards, device, temporal_mode, rng, clen=8):
    """Windowed causal eval (non-overlapping clen windows; each frame scored once)."""
    geo, bnd, temporal, corr_diag, mc_diag = [], [], [], [], []
    for s, sh in shards.items():
        raw = sh["raw_disp"].astype(np.float32); gt = sh["gt_disp"].astype(np.float32)
        v = (sh["valid_mask"] > 0).astype(np.float32)
        rgb = sh["rgb"].astype(np.float32) / 255.0
        flow_full = sh["warp_flow"].astype(np.float32); occ_full = sh["occ"].astype(np.float32)
        T = raw.shape[0]
        for st in range(0, T, clen):
            sl = slice(st, min(st + clen, T))
            L = sl.stop - sl.start
            if L < 2:
                continue
            rr = raw[sl]; gg = gt[sl]; vv = v[sl]; rgbw = rgb[sl]
            flow_w = flow_full[st:st + L - 1]; occ_w = occ_full[st:st + L - 1]
            rt = torch.from_numpy(rr).to(device)[None, :, None]
            vt = torch.from_numpy(vv).to(device)[None, :, None]
            rgbt = torch.from_numpy(rgbw).to(device)[None].permute(0, 1, 4, 2, 3)
            ref, _ = model(rt, vt, rgbt, temporal_mode, rng)
            ref = ref[0, :, 0].cpu().numpy()
            applied = ref - rr
            for t in range(L):
                m = vv[t] > 0.5
                if m.sum():
                    geo.append(frame_metrics(rr[t], ref[t], gg[t], m > 0, edge_map(rr[t])))
                    eb = (edge_map(ref[t]) > 1.0) & m
                    if eb.sum():
                        bnd.append(frame_metrics(rr[t], ref[t], gg[t], eb, edge_map(rr[t])))
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
                # correction-flicker diagnostics
                mod_t = np.abs(applied[t]) > 0.1
                if mod_t.any():
                    corr_diag.append({
                        "signflip_rate": float((np.sign(applied[t]) != np.sign(applied[t - 1]))[mod_t & m].mean()) if (mod_t & m).any() else float("nan"),
                        "isolated_rate": float((np.abs(applied[t - 1]) <= 0.1)[mod_t & m].mean()) if (mod_t & m).any() else float("nan"),
                    })
                # motion-compensated self-consistency (prediction-space, no GT; cached backward
                # warp flow t->t-1, in-bounds + occlusion masked). NOTE: on SCARED inter-frame
                # motion is tiny, so even a correct warp adds resampling noise that can exceed the
                # true temporal change -> this is a weak diagnostic here (see flow_mask_validation).
                fl = torch.from_numpy(flow_w[t - 1]).to(device)[None]
                Dp = torch.from_numpy(ref[t - 1]).to(device)[None, None]
                warped, inb = warp_with_support(Dp, fl)
                warped = warped[0, 0].cpu().numpy(); inb = inb[0, 0].cpu().numpy() > 0.5
                mcm = m & (occ_w[t - 1] < 0.5) & inb
                if mcm.any():
                    mc_diag.append({"mc_inconsistency": float(np.abs(ref[t] - warped)[mcm].mean())})
            for t in range(2, L):
                m = (vv[t] > 0.5) & (vv[t - 1] > 0.5) & (vv[t - 2] > 0.5)
                if m.any():
                    er = np.abs(ref[t] - gg[t]); erp = np.abs(ref[t - 1] - gg[t - 1]); erpp = np.abs(ref[t - 2] - gg[t - 2])
                    temporal.append({"hf_error_energy": float(np.abs(er - 2 * erp + erpp)[m].mean())})

    def agg(rows, keys):
        return {k: (float(np.nanmean([r[k] for r in rows if k in r])) if rows else float("nan")) for k in keys}

    g = agg(geo, ["refined_mae", "raw_mae", "delta_mae", "refined_bad1", "refined_bad3", "refined_bad5",
                  "new_bad3_pct_of_rawgood", "harmful_rate", "beneficial_rate", "modified_pixel_ratio"])
    g["boundary_mae"] = float(np.nanmean([r["refined_mae"] for r in bnd])) if bnd else float("nan")
    tm = agg(temporal, ["tgm_error", "terr_jitter", "boundary_tgm", "hf_error_energy"])
    tm.update(agg(corr_diag, ["signflip_rate", "isolated_rate"]))
    tm.update(agg(mc_diag, ["mc_inconsistency"]))
    return g, tm


def run_id_for(config, clip_len, seed):
    cfg = CONFIGS[config]
    return f"{config}__{cfg['model']}__rgb{int(cfg['use_rgb'])}__{cfg['temporal_mode']}__clip{clip_len}__seed{seed}"


def train_run(args, train, val, test, device):
    """One training run given preloaded split shards (so a matrix driver can load the cache once
    and reuse it across all runs instead of re-decompressing 7.9GB per run)."""
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if args.smoke:
        args.steps = 20; args.eval_every = 20; args.batch = 2

    cfg = CONFIGS[args.config]
    run_id = run_id_for(args.config, args.clip_len, args.seed)
    out = args.out / run_id; out.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg["model"], use_rgb=cfg["use_rgb"]).to(device)
    params = sum(p.numel() for p in model.parameters())
    assert all(p.requires_grad for p in model.parameters())  # refiner itself is trainable; S2M2/RAFT never enter this graph
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    t0 = time.time(); log = []; eval_time = 0.0
    best = (1e9, None)
    for step in range(1, args.steps + 1):
        model.train()
        batch = [sample_clip(train, args.clip_len, rng) for _ in range(args.batch)]
        raw, gt, v, rgb, flow, occ = collate(batch, device)
        with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            refined, _ = model(raw, v, rgb, cfg["temporal_mode"], rng)
            loss, parts = clip_losses(refined, raw, gt, v, flow, occ, cfg)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            _te = time.time()
            g, tm = eval_sequences(model, val, device, cfg["temporal_mode"], rng, args.clip_len)
            eval_time += time.time() - _te
            score = g["refined_mae"] + 0.02 * g["new_bad3_pct_of_rawgood"] + 0.5 * g["harmful_rate"]
            log.append({"step": step, "loss": float(loss), **parts, "val_mae": g["refined_mae"],
                        "val_tgm": tm["tgm_error"], "val_newbad3": g["new_bad3_pct_of_rawgood"],
                        "val_modified": g["modified_pixel_ratio"], "val_harmful": g["harmful_rate"],
                        "val_score": score})
            print(f"[{run_id}] step{step} loss={float(loss):.4f} val_mae={g['refined_mae']:.4f} "
                  f"val_mod={g['modified_pixel_ratio']:.5f} val_tgm={tm['tgm_error']:.4f} "
                  f"warp={parts['warp']:.4f} score={score:.4f}", flush=True)
            if score == score and score < best[0]:
                best = (score, {k: w.detach().clone() for k, w in model.state_dict().items()})
    if best[1] is not None:
        model.load_state_dict(best[1])
        torch.save({"model_state_dict": best[1], "args": vars(args), "config": cfg, "params": params}, out / "best.pt")
    model.eval()
    _te = time.time()
    gt_g, gt_tm = eval_sequences(model, test, device, cfg["temporal_mode"], rng, args.clip_len)
    eval_time += time.time() - _te
    total = time.time() - t0
    print(f"[{run_id}] TIMING total={total:.1f}s eval={eval_time:.1f}s train={total - eval_time:.1f}s "
          f"({args.steps} steps)", flush=True)
    out_cfg = {"run_id": run_id, "letter": args.config, **cfg, "clip_len": args.clip_len, "params": params,
               "steps": args.steps, "seed": args.seed, "wall_time_s": round(total, 1),
               "eval_time_s": round(eval_time, 1),
               "test_geometric": {k: round(v, 4) for k, v in gt_g.items()},
               "test_temporal": {k: round(v, 4) for k, v in gt_tm.items()}}
    (out / "config.json").write_text(json.dumps(out_cfg, indent=2, default=float) + "\n")
    with (out / "train_log.csv").open("w", newline="") as f:
        if log:
            w = csv.DictWriter(f, fieldnames=list(log[0].keys())); w.writeheader(); w.writerows(log)
    print(json.dumps(out_cfg, indent=2, default=float))
    return out_cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, choices=list(CONFIGS))
    ap.add_argument("--clip-len", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=1200)
    ap.add_argument("--out", type=Path, default=OUT / "runs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    train = load_split_shards("train"); val = load_split_shards("val"); test = load_split_shards("test")
    train_run(args, train, val, test, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
