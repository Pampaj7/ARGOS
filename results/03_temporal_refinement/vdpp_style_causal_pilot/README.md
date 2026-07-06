# VDPP-style causal pilot (Temporal Gradient Matching)

Tests whether EGBM failed for lack of explicit dense temporal supervision. Trains a minimal
752k causal ConvGRU residual refiner on SCARED consecutive clips with a TGM loss; S2M2 frozen;
no optical flow. Mandatory current-frame / shuffled-history ablations.

## Result: TGM supervision works — decision gate PASS
SCARED temporal error 1.712→1.550 (−9.5%) vs spatial-only; shuffled-history is worst (uses real
time); D4D zero-shot slightly improves anchor MAE (+0.05, safe) AND all temporal diagnostics
(mc 0.390→0.382, HF 0.507→0.496). First ARGOS temporal model to transfer a real temporal gain
to D4D without false-activation. Full analysis: `findings.md`, `decision_gate.json`.

## Files
`train_vdpp_causal.py` (model+trainer), `eval_vdpp_d4d.py` (D4D transfer), `runs/<mode>/`
(config.json, best.pt, train_log), `ablation_table.csv`, `scared_{geometric,temporal}_metrics.csv`,
`d4d_{anchor,temporal}_metrics.csv`, `run_manifest.csv`, `model_summary.json`, `code_audit.md`,
`methodology.md`, `findings.md`, `limitations.md`, `decision_gate.json`.

## Reproduce
```bash
for M in spatial tgm current_frame shuffled; do
  python scripts/temporal_refinement/vdpp_style_causal/train_vdpp_causal.py --mode $M --clip-len 8 --steps 1500 --seed 0
done
python scripts/temporal_refinement/vdpp_style_causal/eval_vdpp_d4d.py --ckpt runs/tgm__clip8__seed0/best.pt
```

## Recommended next step
Motion-aligned (flow-warped) temporal refinement + a selectivity/safety term, to amplify the
confirmed-but-small D4D temporal benefit. Test clip-len 16 and TGM-weight sweep.
