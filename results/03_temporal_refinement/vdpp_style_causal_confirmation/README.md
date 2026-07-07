# VDPP-style causal temporal-usage confirmation study

Rigorous re-run of the VDPP causal pilot (`8274259`) after finding its ablation confounded loss
supervision with temporal-input structure. See `findings.md` for the result (**MARGINAL — not
confirmed**) and `limitations.md` for caveats.

## Design
`temporal_input_mode` (full_history / current_frame_only / shuffled_history) is now decoupled
from `loss_mode` (spatial_only / spatial_plus_tgm) as independent flags in
`train_vdpp_causal.py`. Core factorial: 4 configs × 3 seeds, clip_len=8, λ_tgm=1.0 (12 runs).
TGM-weight sweep: full_history+TGM at λ ∈ {0.2, 0.5, 1.0} × 3 seeds (6 runs, 3 reused from the
factorial). D4D zero-shot re-eval: all 3 temporal_input_mode variants × 3 seeds (9 eval runs).

## Layout
- `runs/` — 18 training runs (`config.json` + `train_log.csv` + gitignored `best.pt`).
- `d4d/<mode>__seed<N>/` — per-(mode,seed) D4D zero-shot eval (`d4d_anchor_metrics.csv`,
  `d4d_temporal_metrics.csv`, `d4d_vdpp_summary.json`).
- `per_seed_scared_metrics.csv`, `scared_mean_std.csv`, `factorial_ablation.csv` — SCARED eval,
  raw per-seed and aggregated.
- `tgm_weight_sweep.csv` — λ sweep for full_history+TGM.
- `d4d_per_clip_metrics.csv`, `d4d_per_specimen_metrics.csv`, `d4d_bootstrap_ci.json` — paired
  D4D diagnostics (vdpp − raw) vs raw, with bootstrap CIs.
- `temporal_usage_decision.json` — the 5-gate decision output (verdict: MARGINAL).
- `run_manifest.csv` — all 18 runs with key config + result fields.
- `findings.md`, `limitations.md`, `changed_files.txt` — this study's report.

## Reproduce
```bash
# factorial (per seed, resumable):
python scripts/temporal_refinement/vdpp_style_causal/train_vdpp_causal.py \
  --temporal-input-mode {full_history,current_frame_only,shuffled_history} \
  --loss-mode {spatial_only,spatial_plus_tgm} --lam-tgm 1.0 --clip-len 8 --steps 1200 \
  --seed {0,1,2} --out results/03_temporal_refinement/vdpp_style_causal_confirmation/runs

# D4D eval (per mode,seed):
python scripts/temporal_refinement/vdpp_style_causal/eval_vdpp_d4d.py \
  --ckpt <run>/best.pt --temporal-mode <mode> --clips-per-specimen 3 --max-frames 100 \
  --out results/03_temporal_refinement/vdpp_style_causal_confirmation/d4d/<mode>__seed<N>

# aggregate + decision gate:
python scripts/temporal_refinement/vdpp_style_causal/summarize_confirmation.py
```
Requires `p1i` H100 GPU (see `[[argos-node-access]]` memory for the LSF launch pattern — use
`-app h100app` with `ESUB_BYPASS=1 ESUB_QUIET=1` for long unattended runs, plain `bsub -I -q p1i`
defaults to a 15-min walltime).
