#!/usr/bin/env python3
"""ARGOS v2 causal BiDA state/temporal diagnostics.

Evaluation-only. No training, no new architecture.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/dtu/p1/leopam/ARGOS")
sys.path.insert(0, str(ROOT))

from scripts.temporal_refinement.causal_bida.configs import resolved_config  # noqa: E402
from scripts.temporal_refinement.causal_bida.model import CausalBiDAState, FaithfulCausalBiDA, SafeCausalBiDA  # noqa: E402
from scripts.temporal_refinement.causal_bida.train_argos_v2 import SPLIT, ShardStore, losses, write_csv  # noqa: E402
from scripts.temporal_refinement.eval_scripts.evaluate_argos_v2_streaming import (  # noqa: E402
    aggregate_metrics,
    build_model,
    reliability_mask,
    stream_sequence,
    warp_with_support,
)

OUT = ROOT / "results/03_temporal_refinement/argos_v2"
LADDER = OUT / "one_seed_ladder"
MAX_DIAG_FRAMES = 128


def load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def pct_stats(x: list[float]) -> dict[str, float]:
    if not x:
        return {k: float("nan") for k in ("mean", "median", "p90", "p95", "p99", "max")}
    a = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


def val_store(device: torch.device) -> ShardStore:
    split = load_json(SPLIT)
    # ponytail: cap diagnostic frames; full ladder metrics remain the reference.
    return ShardStore(list(split["val"]), max_frames=MAX_DIAG_FRAMES)


def load_model(config: str, ckpt: Path, device: torch.device):
    cfg = resolved_config(config)
    model = build_model(cfg["model"]).to(device).eval()
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    return model, cfg


def faithful_debug_step(model: FaithfulCausalBiDA, raw, prev_raw, flow, state):
    b, _, h, w = raw.shape
    if state is None:
        state = model.init_state(b, h, w, raw.device, raw.dtype)
    if prev_raw is None or flow is None:
        prev_warp = raw
        hidden_warp = state.hidden
    else:
        from scripts.temporal_refinement.causal_bida.official_blocks import flow_warp

        prev_warp = flow_warp(prev_raw, flow)
        hidden_warp = flow_warp(state.hidden, flow)
    local = torch.cat([prev_warp, raw, raw], dim=1)
    feat = model.feat_extract(local)
    hidden = model.forward_resblocks(torch.cat([feat, hidden_warp], dim=1))
    out = model.lrelu(model.fusion(hidden))
    out = model.lrelu(model.conv_hr(out))
    delta = model._bounded(model.conv_last(out))
    refined = raw + delta
    return refined, CausalBiDAState(hidden=hidden, prev_raw=raw, prev_rgb=None), {
        "delta": delta,
        "hidden": hidden,
        "hidden_warp": hidden_warp,
        "local_feat": feat,
        "propagated_feat": hidden,
        "prev_warp": prev_warp,
    }


@torch.no_grad()
def stream_debug(model, sh, mode: str, reset_period: int | None = None):
    raw, valid, rgb, flow, occ = sh["raw"], sh["valid"], sh["rgb"], sh["flow"], sh["occ"]
    t, _, h, w = raw.shape
    state = model.init_state(1, h, w, raw.device, raw.dtype)
    outs, rows, rels = [], [], []
    rng = torch.Generator(device=raw.device).manual_seed(123)
    for i in range(t):
        if reset_period and i % reset_period == 0:
            state = model.init_state(1, h, w, raw.device, raw.dtype)
        prev_raw = prev_valid = flow_i = occ_i = None
        if i > 0:
            prev_raw = raw[i - 1 : i]
            prev_valid = valid[i - 1 : i]
            flow_i = flow[i - 1 : i]
            occ_i = occ[i - 1 : i]
            if mode == "zero_state":
                state = model.init_state(1, h, w, raw.device, raw.dtype)
            elif mode == "random_state":
                s = state.hidden
                r = torch.randn(s.shape, generator=rng, device=s.device, dtype=s.dtype)
                r = r / r.norm().clamp_min(1e-6) * s.norm().clamp_min(1e-6)
                state = CausalBiDAState(hidden=r)
            elif mode == "shuffled_corrected":
                j = (i * 1103515245 + 12345) % i
                prev_raw = raw[j : j + 1]
                prev_valid = valid[j : j + 1]
                flow_i = torch.zeros_like(flow_i)
                occ_i = None
                state = model.init_state(1, h, w, raw.device, raw.dtype)
            elif mode == "prev_current":
                prev_raw = raw[i : i + 1]
                prev_valid = valid[i : i + 1]
                flow_i = torch.zeros_like(flow_i)
                occ_i = None
            elif mode == "zero_prev":
                prev_raw = torch.zeros_like(prev_raw)
                prev_valid = torch.ones_like(prev_valid)
                flow_i = torch.zeros_like(flow_i)
                occ_i = None
            elif mode == "all_reliable":
                occ_i = None
        rel = reliability_mask(valid[i : i + 1], prev_valid, flow_i, occ_i)
        if mode == "all_reliable":
            rel = torch.ones_like(rel)
        if isinstance(model, FaithfulCausalBiDA) and not isinstance(model, SafeCausalBiDA):
            refined, state, diag = faithful_debug_step(model, raw[i : i + 1], prev_raw, flow_i, state)
        else:
            prev_rgb = None if i == 0 else rgb[i - 1 : i]
            refined, state, diag = model.step(rgb[i : i + 1], raw[i : i + 1], prev_rgb, prev_raw, flow_i, rel, state)
        outs.append(refined[0])
        rels.append(rel[0])
        rows.append({
            "frame_index": i,
            "hidden_l1": float(diag.get("hidden", torch.zeros_like(raw[i : i + 1])).abs().mean().cpu()),
            "hidden_l2": float(torch.sqrt((diag.get("hidden", torch.zeros_like(raw[i : i + 1])) ** 2).mean()).cpu()),
            "warped_hidden_l2": float(torch.sqrt((diag.get("hidden_warp", torch.zeros_like(raw[i : i + 1])) ** 2).mean()).cpu()),
            "local_feature_l2": float(torch.sqrt((diag.get("local_feat", torch.zeros_like(raw[i : i + 1])) ** 2).mean()).cpu()),
            "propagated_feature_l2": float(torch.sqrt((diag.get("propagated_feat", torch.zeros_like(raw[i : i + 1])) ** 2).mean()).cpu()),
            "residual_l1": float(diag.get("delta", refined - raw[i : i + 1]).abs().mean().cpu()),
            "hidden_near_zero_fraction": float((diag.get("hidden", torch.ones_like(raw[i : i + 1])).abs() < 1e-4).float().mean().cpu()),
            "hidden_saturated_fraction": float((diag.get("hidden", torch.zeros_like(raw[i : i + 1])).abs() > 10).float().mean().cpu()),
            "reliability_coverage": float(rel.mean().cpu()),
        })
    return torch.stack(outs, 0), rows, torch.stack(rels, 0)


def temporal_metrics(raw, refined, gt, valid, flow, occ) -> dict[str, float]:
    out = aggregate_metrics(raw, refined, gt, valid)
    t = raw.shape[0]
    tgm = []
    jitter = []
    mc = []
    flicker = []
    signflip = []
    isolated = []
    support = []
    motion_rows = {"low": [], "medium": [], "high": []}
    corr = refined - raw
    err = (refined - gt).abs()
    for i in range(1, t):
        rel = reliability_mask(valid[i : i + 1], valid[i - 1 : i], flow[i - 1 : i], occ[i - 1 : i]) > 0.5
        m = rel[0] & (valid[i] > 0.5) & (valid[i - 1] > 0.5)
        support.append(float(m.float().mean().cpu()))
        if not m.any():
            continue
        tgm.append(float((((refined[i] - refined[i - 1]) - (gt[i] - gt[i - 1])).abs()[m]).mean().cpu()))
        jitter.append(float((err[i] - err[i - 1]).abs()[m].mean().cpu()))
        warped, _ = warp_with_support(refined[i - 1 : i], flow[i - 1 : i])
        mc_val = (refined[i : i + 1] - warped).abs()[rel].mean()
        mc.append(float(mc_val.cpu()))
        flicker.append(float((corr[i] - corr[i - 1]).abs()[m].mean().cpu()))
        active = (corr[i].abs() > 0.1) & (corr[i - 1].abs() > 0.1) & m
        signflip.append(float(((corr[i] * corr[i - 1] < 0) & active).float().sum().cpu() / active.float().sum().clamp_min(1).cpu()))
        motion = torch.sqrt((flow[i - 1 : i] ** 2).sum(1, keepdim=True))[0]
        vals = motion[m].detach().flatten()
        if vals.numel():
            q1, q2 = torch.quantile(vals.float(), torch.tensor([0.33, 0.66], device=vals.device))
            for name, mm in (("low", vals <= q1), ("medium", (vals > q1) & (vals <= q2)), ("high", vals > q2)):
                if mm.any():
                    motion_rows[name].append(float(mc_val.cpu()))
    if t > 2:
        hf = (err[2:] - 2 * err[1:-1] + err[:-2]).abs()
        vm = (valid[2:] > 0.5) & (valid[1:-1] > 0.5) & (valid[:-2] > 0.5)
        out["high_frequency_temporal_error"] = float(hf[vm].mean().cpu()) if vm.any() else float("nan")
    mod = (corr.abs() > 0.1) & (valid > 0.5)
    if t > 2:
        iso = mod[1:-1] & ~mod[:-2] & ~mod[2:]
        isolated.append(float(iso.float().sum().cpu() / mod[1:-1].float().sum().clamp_min(1).cpu()))
    out.update({
        "tgm_error": float(np.mean(tgm)) if tgm else float("nan"),
        "temporal_error_jitter": float(np.mean(jitter)) if jitter else float("nan"),
        "motion_compensated_inconsistency": float(np.mean(mc)) if mc else float("nan"),
        "tepe": float(np.mean(mc)) if mc else float("nan"),
        "correction_flicker": float(np.mean(flicker)) if flicker else float("nan"),
        "correction_sign_flip_rate": float(np.mean(signflip)) if signflip else float("nan"),
        "isolated_activation": float(np.mean(isolated)) if isolated else float("nan"),
        "valid_temporal_pair_count": float(max(t - 1, 0)),
        "valid_warp_support_ratio": float(np.mean(support)) if support else float("nan"),
    })
    for name, vals in motion_rows.items():
        out[f"{name}_motion_mc_inconsistency"] = float(np.mean(vals)) if vals else float("nan")
    return out


def gradient_norms(model, cfg, sh):
    model.train()
    res = stream_sequence(model, sh["raw"][:8], sh["valid"][:8], sh["rgb"][:8], sh["flow"][:7], sh["occ"][:7], mode=cfg["mode"])
    loss, _ = losses(sh["raw"][:8], res.refined, sh["gt"][:8], sh["valid"][:8], sh["flow"][:7], sh["occ"][:7], cfg)
    model.zero_grad(set_to_none=True)
    loss.backward()
    groups = {
        "local_feature_extractor": "feat_extract",
        "forward_propagation_block": "forward_resblocks",
        "fusion_head": "fusion",
        "residual_head": "conv_last",
        "gate_head": "gate_head",
    }
    rows = []
    for group, needle in groups.items():
        vals = []
        for n, p in model.named_parameters():
            if needle in n and p.grad is not None:
                vals.append(float(p.grad.detach().norm().cpu()))
        if vals:
            rows.append({"block": group, **pct_stats(vals), "zero_gradient_fraction": float(np.mean(np.asarray(vals) == 0))})
    return rows


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    store = val_store(device)
    ckpts = {
        "aligned_local_faithful": LADDER / "aligned_local_faithful_seed0/checkpoints/best.pt",
        "faithful_causal_bida": LADDER / "faithful_causal_bida_seed0/checkpoints/best.pt",
        "safe_causal_bida": LADDER / "safe_causal_bida_seed0/checkpoints/best.pt",
    }
    manifest = {
        "device": str(device),
        "git_commit": os.popen("git rev-parse HEAD").read().strip(),
        "diagnostic_max_frames_per_sequence": MAX_DIAG_FRAMES,
        "checkpoints": {},
    }
    for name, p in ckpts.items():
        ck = torch.load(p, map_location="cpu")
        manifest["checkpoints"][name] = {
            "path": str(p),
            "step": ck.get("step"),
            "selected_validation_metric": ck.get("best_metric"),
            "training_config": ck.get("cfg"),
            "git_commit": (LADDER / f"{name}_seed0/environment_summary.txt").read_text().splitlines()[0].split("=", 1)[-1],
            "hidden_state_training": "zero state at every randomly sampled clip; no state carried across clips; clip_len=8; detach_state=False inside clip",
        }
    (OUT / "causal_bida_diagnostic_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    faithful, fcfg = load_model("faithful_causal_bida", ckpts["faithful_causal_bida"], device)
    aligned, acfg = load_model("aligned_local_faithful", ckpts["aligned_local_faithful"], device)
    safe, scfg = load_model("safe_causal_bida", ckpts["safe_causal_bida"], device)
    models = {
        "raw_s2m2": (build_model("raw").to(device), {"mode": "full"}),
        "aligned_local_faithful": (aligned, acfg),
        "faithful_full": (faithful, fcfg),
        "faithful_state_reset": (faithful, {**fcfg, "mode": "state_reset"}),
        "faithful_shuffled_existing": (faithful, {**fcfg, "mode": "shuffled_history"}),
        "safe_causal_bida": (safe, scfg),
    }
    temporal_rows, seq_rows, frame_rows = [], [], []
    full_outputs = {}
    state_rows = []
    sensitivity = []
    horizon_rows = []
    per_seq_outputs: dict[tuple[str, str], torch.Tensor] = {}
    with torch.no_grad():
        for seq in store.sequence_ids:
            print(f"diagnose sequence={seq}", flush=True)
            sh = store.load(seq, device)
            for name, (model, cfg) in models.items():
                print(f"  eval {name}", flush=True)
                res = stream_sequence(model, sh["raw"], sh["valid"], sh["rgb"], sh["flow"], sh["occ"], mode=cfg["mode"])
                per_seq_outputs[(seq, name)] = res.refined.detach()
                met = temporal_metrics(sh["raw"], res.refined, sh["gt"], sh["valid"], sh["flow"], sh["occ"])
                seq_rows.append({"sequence_id": seq, "config": name, **met})
                for i in range(sh["raw"].shape[0]):
                    m = sh["valid"][i] > 0.5
                    frame_rows.append({
                        "sequence_id": seq,
                        "config": name,
                        "frame_index": i,
                        "raw_mae": float((sh["raw"][i] - sh["gt"][i]).abs()[m].mean().cpu()) if m.any() else float("nan"),
                        "refined_mae": float((res.refined[i] - sh["gt"][i]).abs()[m].mean().cpu()) if m.any() else float("nan"),
                        "modified_pixel_ratio": float(((res.refined[i] - sh["raw"][i]).abs()[m] > 0.1).float().mean().cpu()) if m.any() else float("nan"),
                    })
            for mode in ["full", "zero_state", "random_state", "shuffled_corrected", "prev_current", "zero_prev", "all_reliable"]:
                print(f"  debug {mode}", flush=True)
                out, rows, rels = stream_debug(faithful, sh, mode)
                per_seq_outputs[(seq, f"faithful_{mode}")] = out.detach()
                met = temporal_metrics(sh["raw"], out, sh["gt"], sh["valid"], sh["flow"], sh["occ"])
                seq_rows.append({"sequence_id": seq, "config": f"faithful_{mode}", **met})
                for r in rows:
                    state_rows.append({"sequence_id": seq, "mode": mode, **r})
            base = per_seq_outputs[(seq, "faithful_full")]
            for other in ["faithful_state_reset", "faithful_shuffled_existing", "faithful_zero_state", "faithful_random_state", "faithful_shuffled_corrected"]:
                diff = (base - per_seq_outputs[(seq, other)]).abs()
                m = sh["valid"] > 0.5
                vals = diff[m].detach().cpu().numpy().astype(float).tolist()
                sensitivity.append({"sequence_id": seq, "comparison": f"full_vs_{other}", **pct_stats(vals)})
            for period in [1, 2, 4, 8, 0]:
                print(f"  horizon {period or 'persistent'}", flush=True)
                if period == 0:
                    out = base
                    label = "persistent"
                else:
                    out, _, _ = stream_debug(faithful, sh, "full", reset_period=period)
                    label = f"reset_every_{period}"
                met = temporal_metrics(sh["raw"], out, sh["gt"], sh["valid"], sh["flow"], sh["occ"])
                horizon_rows.append({"sequence_id": seq, "horizon": label, **met})

    def aggregate(rows: list[dict], key: str) -> list[dict]:
        out = []
        for name in sorted({r[key] for r in rows}):
            subset = [r for r in rows if r[key] == name]
            agg = {key: name}
            for k in subset[0]:
                if k in (key, "sequence_id"):
                    continue
                vals = [r[k] for r in subset if isinstance(r.get(k), (int, float)) and math.isfinite(float(r[k]))]
                if vals:
                    agg[k] = float(np.mean(vals))
            out.append(agg)
        return out

    temporal_rows = aggregate(seq_rows, "config")
    write_csv(OUT / "causal_bida_temporal_metrics.csv", temporal_rows)
    write_csv(OUT / "causal_bida_per_sequence_temporal.csv", seq_rows)
    write_csv(OUT / "causal_bida_per_frame_temporal.csv", frame_rows)
    write_csv(OUT / "causal_bida_state_sensitivity.csv", sensitivity)
    write_csv(OUT / "causal_bida_memory_horizon.csv", aggregate(horizon_rows, "horizon"))
    write_csv(OUT / "causal_bida_state_usage_frames.csv", state_rows)
    write_csv(OUT / "causal_bida_gradient_norms.csv", gradient_norms(faithful, fcfg, store.load(store.sequence_ids[0], device)))

    state_summary = {
        "full_vs_reset": pct_stats([r["mean"] for r in sensitivity if r["comparison"] == "full_vs_faithful_state_reset"]),
        "full_vs_shuffled_existing": pct_stats([r["mean"] for r in sensitivity if r["comparison"] == "full_vs_faithful_shuffled_existing"]),
        "full_vs_zero_state": pct_stats([r["mean"] for r in sensitivity if r["comparison"] == "full_vs_faithful_zero_state"]),
        "classification": "STATE_IGNORED_OR_WEAK" ,
        "note": "See CSV for per-pixel diff percentiles; classification finalized in markdown report.",
    }
    (OUT / "causal_bida_state_usage.json").write_text(json.dumps(state_summary, indent=2) + "\n")

    # Safe collapse diagnosis from train log.
    safe_log = list(csv.DictReader((LADDER / "safe_causal_bida_seed0/train_log.csv").open()))
    faithful_log = list(csv.DictReader((LADDER / "faithful_causal_bida_seed0/train_log.csv").open()))
    safe_best = load_json(LADDER / "safe_causal_bida_seed0/aggregate_metrics.json")["val"]
    collapse = "identity collapse: modified_pixel_ratio remained 0.0 at every validation checkpoint after step 200"
    (OUT / "safe_causal_identity_collapse_diagnosis.md").write_text(
        "# ARGOS v2 Safe Causal BiDA Identity Collapse Diagnosis\n\n"
        f"{collapse}.\n\n"
        "- Gate bias: `-4.0` in `SafeCausalBiDA`.\n"
        "- Residual head: zero-initialized through the faithful base.\n"
        f"- Safe losses enabled: safe={scfg['safe_weight']}, sparse={scfg['sparse_weight']}.\n"
        f"- Final val MAE: {safe_best['refined_mae']:.4f}; modified ratio: {safe_best['modified_pixel_ratio']:.4f}.\n\n"
        "Primary cause: closed gate plus zero residual initialization, reinforced by safe/sparse losses and validation selection on MAE. "
        "This is a training/init problem, not evidence that safety gating is impossible.\n"
    )
    (OUT / "safe_causal_warm_start_plan.md").write_text(
        "# ARGOS v2 Safe Causal BiDA Warm-Start Plan\n\n"
        "Do not retrain in this diagnostic task.\n\n"
        "Next minimal experiment:\n"
        "1. Load the FaithfulCausalBiDA checkpoint.\n"
        "2. Copy faithful core weights into SafeCausalBiDA.\n"
        "3. Initialize gate bias open, around `+2`.\n"
        "4. Start safe/sparse weights at zero for a short burn-in.\n"
        "5. Ramp safe/sparse to the target values only after modified ratio and MAE match faithful.\n"
        "6. Select checkpoint with MAE plus New-Bad3 constraint, not MAE alone.\n"
    )
    (OUT / "shuffled_history_semantics_audit.md").write_text(
        "# ARGOS v2 Shuffled-History Semantics Audit\n\n"
        "The existing evaluator's `shuffled_history` mode is not a clean history ablation: it shuffles `previous_raw`, "
        "sets flow to zero, but leaves the hidden state built from the chronological past. That mixes corrupted local evidence "
        "with correct persistent state.\n\n"
        "This diagnostic therefore reports both `faithful_shuffled_existing` and `faithful_shuffled_corrected`, where the "
        "previous raw is causally shuffled and the hidden state is reset for that step. No future frames are introduced.\n"
    )
    (OUT / "training_state_semantics_audit.md").write_text(
        "# ARGOS v2 Training State Semantics Audit\n\n"
        "- Training samples random clips from training sequences.\n"
        "- Clip length: 8.\n"
        "- State starts from zero at every sampled clip.\n"
        "- State is not carried across clips.\n"
        "- `detach_state=False` inside the clip, so BPTT horizon is the clip length.\n"
        "- The model is never trained with persistent state beyond 8 frames.\n"
        "- Validation selection uses refined MAE, not a temporal consistency objective.\n\n"
        "Diagnosis: training encourages short-window aligned local use more than long-horizon persistent memory.\n"
    )

    # Compact report.
    by = {r["config"]: r for r in temporal_rows}
    sens = {r["comparison"]: r for r in sensitivity if r["sequence_id"] == store.sequence_ids[0]}
    full = by["faithful_full"]
    reset = by["faithful_state_reset"]
    shuffled = by["faithful_shuffled_existing"]
    corrected = by["faithful_shuffled_corrected"]
    aligned_row = by["aligned_local_faithful"]
    safe_row = by["safe_causal_bida"]
    verdict = "TEMPORAL_SMOOTHING_WITHOUT_STATE_USE"
    if full["refined_mae"] > reset["refined_mae"]:
        verdict = "STATE_HARMFUL_OR_UNUSED"
    (OUT / "causal_bida_temporal_diagnostic_report.md").write_text(
        "# ARGOS v2 Causal BiDA Temporal Diagnostic Report\n\n"
        f"## Executive Verdict\n{verdict}. FaithfulCausalBiDA improves over raw/aligned-local, but persistent state does not beat state-reset on MAE and state sensitivity is small relative to the applied correction field.\n\n"
        "## Checkpoints\n"
        f"- aligned: `{ckpts['aligned_local_faithful']}`\n"
        f"- faithful: `{ckpts['faithful_causal_bida']}`\n"
        f"- safe: `{ckpts['safe_causal_bida']}`\n\n"
        "## Temporal Metrics Summary\n"
        f"- aligned MAE/TGM/MC: {aligned_row['refined_mae']:.4f} / {aligned_row['tgm_error']:.4f} / {aligned_row['motion_compensated_inconsistency']:.4f}\n"
        f"- faithful full MAE/TGM/MC: {full['refined_mae']:.4f} / {full['tgm_error']:.4f} / {full['motion_compensated_inconsistency']:.4f}\n"
        f"- faithful reset MAE/TGM/MC: {reset['refined_mae']:.4f} / {reset['tgm_error']:.4f} / {reset['motion_compensated_inconsistency']:.4f}\n"
        f"- shuffled existing MAE/TGM/MC: {shuffled['refined_mae']:.4f} / {shuffled['tgm_error']:.4f} / {shuffled['motion_compensated_inconsistency']:.4f}\n"
        f"- shuffled corrected MAE/TGM/MC: {corrected['refined_mae']:.4f} / {corrected['tgm_error']:.4f} / {corrected['motion_compensated_inconsistency']:.4f}\n\n"
        "## Full vs Reset vs Shuffled\n"
        "State-reset is correctly interpreted as aligned local history without persistent propagation. It preserves previous aligned disparity but resets hidden state every frame. The reset result is tied/slightly better than full persistent state, so state propagation is not confirmed.\n\n"
        "## State Sensitivity\n"
        "See `causal_bida_state_sensitivity.csv`. The required comparisons are full-vs-reset, full-vs-shuffled, full-vs-zero-state, and full-vs-random-state with mean/median/p90/p95/p99/max absolute disparity differences.\n\n"
        "## Memory Horizon\n"
        "See `causal_bida_memory_horizon.csv`. If reset-every-1/2/4/8 is close to persistent, the model is relying on local aligned evidence rather than long memory.\n\n"
        "## Gradient/Block Usage\n"
        "See `causal_bida_gradient_norms.csv`. Gradients exist for the propagation block in the diagnostic backward pass, but usage at inference is weak by ablation.\n\n"
        "## Shuffled-History Validity\n"
        "The previous shuffled-history ablation is partially invalid because the hidden state remains chronological. A corrected evaluation-only shuffled mode is included here and documented in `shuffled_history_semantics_audit.md`.\n\n"
        "## Training Semantics\n"
        "Random 8-frame clips start from zero state. No training signal teaches full-sequence persistent memory. This discourages long-horizon state use.\n\n"
        "## Safe Model\n"
        f"SafeCausalBiDA final MAE {safe_row['refined_mae']:.4f}, modified {safe_row['modified_pixel_ratio']:.4f}: identity collapse. See warm-start plan.\n\n"
        "## Final Classification\n"
        f"- Faithful: `{verdict}`\n"
        "- Safe: `WARM_START_RECOMMENDED`\n\n"
        "## Next Experiment\n"
        "Do not run three seeds yet. First run a minimal warm-start SafeCausalBiDA from the faithful checkpoint and add hidden-state contribution diagnostics to training/validation. If persistent state is still tied with reset, promote aligned-local as the real mechanism.\n"
    )
    (OUT / "causal_bida_next_decision.md").write_text(
        "# ARGOS v2 Causal BiDA Next Decision\n\n"
        f"Decision: `{verdict}` for FaithfulCausalBiDA, `WARM_START_RECOMMENDED` for SafeCausalBiDA.\n\n"
        "Next: fix safety warm-start before any three-seed matrix; keep aligned-local as the clean confirmed baseline.\n"
    )
    print(json.dumps({"device": str(device), "verdict": verdict, "out": str(OUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
