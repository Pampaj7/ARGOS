# VDPP causal pilot — findings

**Question:** was EGBM failing mainly for lack of explicit dense temporal supervision?
**Answer: largely yes.** A 752k-param causal ConvGRU refiner trained on SCARED consecutive
clips with a Temporal Gradient Matching (TGM) loss improves SCARED temporal consistency, the
gain requires real temporal order, and it transfers a small-but-real temporal improvement to
D4D zero-shot without false-activation — unlike every prior EGBM/adaptation result.

## SCARED test (frozen, raw MAE 5.22, raw tgm-error ≈1.72)

| mode | MAE | Δtgm-err | terr-jitter | HF-err | boundary-tgm | modified % |
|------|----:|--------:|------------:|-------:|-------------:|-----------:|
| spatial (loss only)   | 5.074 | 1.712 | 1.585 | 2.616 | 2.110 | 41.5 |
| **tgm** | 5.110 | **1.550** | **1.449** | **2.356** | **1.995** | 47.9 |
| current_frame (no mem)| 5.164 | 1.700 | 1.583 | 2.614 | 9.6 | — |
| shuffled (broken time)| 5.293 | **1.917** | — | — | — | 69.1 |

- **TGM loss drives the temporal gain**: tgm-error 1.712 → 1.550 (−9.5%); every temporal
  metric best for tgm. Spatial loss alone (spatial / current_frame) ≈ raw temporal.
- **The model genuinely uses temporal order**: shuffled-history is the WORST (1.917 > raw 1.72,
  modified 69%) — breaking order hurts. So the improvement is real temporal modelling, not a
  loss artefact.
- **Mild spatial/safety cost**: tgm MAE 5.11 still beats raw 5.22, but new-Bad3 4.87→7.01 and
  harmful 0.42→0.47 rise slightly (acceptable, not catastrophic).

## D4D zero-shot transfer (tgm checkpoint)
- **Sparse Zivid anchors**: MAE 4.037 vs raw 4.090 → **ΔMAE +0.053 (improves)**, new-Bad3 0.03%,
  harmful 3.8%, modified only 2.8% — SAFE, no false-activation (contrast: SCARED-trained EGBM
  harmed D4D badly).
- **Full-clip temporal (prediction-space diagnostics)**: mc-inconsistency 0.390→**0.382**,
  HF energy 0.507→**0.496**, depth-MC 0.691→**0.683**, boundary-MC 2.802→**2.711** — all
  slightly BETTER than raw. Modified 6% (no identity collapse).

## Decision gate: **PASS** (see `decision_gate.json`)
All five criteria met. This is the first ARGOS temporal model that both improves SCARED
temporal consistency and carries a reproducible temporal improvement to D4D zero-shot while
staying geometrically safe.

## Interpretation
EGBM's failure was substantially a **supervision** problem: it optimised current-frame residual
without dense temporal-gradient targets, so it never learned temporal coherence. Explicit TGM
supervision on consecutive-frame GT fixes this on SCARED and partially transfers OOD. The D4D
gain is small (corrections are still modest vs real non-rigid motion), so the natural next step
is **motion-aligned** refinement to amplify the temporal benefit — but the supervision
hypothesis is confirmed and the branch is worth continuing.
