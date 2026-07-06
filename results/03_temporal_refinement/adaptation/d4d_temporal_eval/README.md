# D4D temporal-consistency evaluation

Tests whether D4D-adapted / zero-shot EGBM-v3-CARE-S improve temporal consistency on full
consecutive D4D clips (causal inference, RAFT motion-compensated diagnostics), beyond sparse
Zivid-anchor geometric safety. **No retraining.**

## Result: temporal axis is flat — no config helps
MC temporal inconsistency 0.390 (raw) → 0.39–0.40 (all refiners); HF energy 0.507 → 0.52.
Refiners' corrections are negligible vs real inter-frame motion; they add slight jitter, not
stability. Identity-collapse confirmed (calib halves correction; scratch = identity).
**SCARED temporal pretraining gives no temporal advantage on D4D.** See `findings.md`.

## Files
`temporal_eval_d4d.py` (pipeline), `per_clip_metrics.csv`, `aggregate_temporal_metrics.csv`,
`temporal_vs_geometric_pareto.{csv,json}`, `figures/`, `findings.md`, `limitations.md`.

## Reproduce
```bash
bsub -I -q p1i -gpu "num=1:mode=shared" -n 8 bash -lc '... temporal_eval_d4d.py \
  --clips-per-specimen 2 --max-frames 120 --out <OUT>'
```
Metrics: MC-inconsistency (flow-warp), HF energy (2nd temporal diff), correction variance,
sign-flip/isolated activation, gate/damping stability, boundary/depth jitter, motion-stratified
MC, identity-collapse (modified ratio / |applied|). All diagnostic (no dense temporal GT).
