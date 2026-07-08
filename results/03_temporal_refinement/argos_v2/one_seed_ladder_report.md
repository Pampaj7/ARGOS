# ARGOS v2 One-Seed Ladder Report

## Executive Conclusion
One-seed ladder completed on H100. The result is **ALIGNMENT_CONFIRMED_PROPAGATION_NOT_CONFIRMED**. Aligned-local gives a small improvement over raw/current-only. Faithful causal BiDA improves further, but its state-reset evaluation is essentially tied or slightly better, so this run does not prove persistent hidden-state value. Safe causal BiDA collapses to identity.

## Run Table
| Run | Config | Refined MAE | Delta MAE | Refined Bad3 | New Bad3 | Modified | Params |
|---|---|---:|---:|---:|---:|---:|---:|
| R0 | raw_s2m2 | 5.9741 | 0.0000 | 44.6198 | 0.0000 | 0.00% | 0 |
| R1 | current_only | 5.9741 | 0.0000 | 44.6191 | 0.0000 | 0.00% | 522082 |
| R2 | aligned_local_faithful | 5.9689 | 0.0053 | 43.9559 | 2.6997 | 51.38% | 239393 |
| R3 | faithful_causal_bida | 5.9595 | 0.0147 | 43.9300 | 2.1416 | 98.39% | 489185 |
| R4 | state_reset eval | 5.9593 | 0.0148 | 43.9174 | 2.2577 | 98.87% | 489185 |
| R5 | shuffled_history eval | 5.9622 | 0.0119 | 43.9870 | 2.0102 | 96.56% | 489185 |
| R6 | safe_causal_bida | 5.9741 | 0.0000 | 44.6198 | 0.0000 | 0.00% | 522082 |

## Training Stability
All trainable configs completed 1200 steps and wrote best/latest checkpoints. No NaN/OOM was observed in stderr; only a PyTorch AMP deprecation warning was emitted.

## Geometry Comparison
- Raw/current-only are identical within metric precision. Current-only effectively did not activate.
- Aligned-local improves MAE by 0.0053.
- Faithful causal BiDA improves MAE by 0.0147, the best main trained row.

## Temporal And History Comparison
- Faithful vs aligned-local: small additional MAE gain of -0.0094 MAE relative to aligned-local, i.e. better by 0.0094.
- State-reset eval: 5.9593, slightly better than persistent faithful 5.9595; state signal is therefore not confirmed.
- Shuffled-history eval: 5.9622; history corruption hurts versus faithful/state-reset but the difference is small.

## Safety Comparison
Faithful and aligned-local both modify many pixels and introduce New-Bad3 around 2-3 percentage points on validation. Safe causal BiDA avoids New-Bad3 but collapses to identity, so it does not preserve temporal benefit.

## Runtime Comparison
Training/evaluation ran in LSF p1i job `28884920` on `n-62-12-83` with two H100s. Parameter counts are in the run table. Per-frame latency and peak VRAM were not re-benchmarked in this ladder wrapper.

## Outcome Classification
`ALIGNMENT_CONFIRMED_PROPAGATION_NOT_CONFIRMED`

## Recommendation
Promote aligned-local and faithful causal BiDA only if the next check focuses on why state reset matches persistent state. Do not promote SafeCausalBiDA as configured; first fix identity collapse/gate initialization or safety loss balance.

## Provenance
- LSF job: `28884920`
- Node: `n-62-12-83`
- Git commit: `b1b80a41d185775cb75e56ad3d707ed41e727790`
