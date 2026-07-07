# VDPP causal temporal-usage confirmation — findings

## Question
Commit `8274259`'s pilot claimed "explicit temporal supervision (TGM) helps, gate PASS" and
"shuffled history is much worse", but the ablation was **confounded**: only `mode=tgm` used the
TGM loss; `current_frame` and `shuffled` were trained *without* TGM. So the previous result could
not distinguish "the architecture uses causal history" from "TGM loss alone regularizes the
output, with or without meaningful history."

This study decouples `temporal_input_mode` (full_history / current_frame_only / shuffled_history)
from `loss_mode` (spatial_only / spatial_plus_tgm) and re-runs the confirmatory factorial with 3
seeds per cell, plus a small TGM-weight sweep and a full D4D zero-shot re-evaluation.

## Headline
**MARGINAL — not confirmed.** Group means point the expected direction (full_history+TGM has the
lowest mean TGM error), but per-seed distributions **overlap heavily** with both ablations: full
history's worst seed (1.721) is worse than shuffled_history's best seed (1.638), and worse than
every current_frame_only seed. The original pilot's clean separation (1.550 vs 1.917, ~19%) was
mostly an artefact of the TGM-loss confound, not evidence of causal history exploitation. On D4D,
the full_history model shows **no stable temporal benefit** — sign of the mc_inconsistency
improvement is not stable across seeds (2/3 seeds ≈0 or worse), and the ablations show effects of
similar or larger magnitude.

## Factorial (SCARED, `tmp_tgm_error`, lower=better, n=3 seeds, clip8, λ=1.0)
| config | mean | std | seeds |
|---|---|---|---|
| full_history + spatial_only | 1.7023 | 0.0117 | 1.6934, 1.6946, 1.7189 |
| full_history + spatial_plus_tgm | **1.6456** | 0.0534 | 1.6121, 1.7210, 1.6038 |
| current_frame_only + spatial_plus_tgm | 1.6816 | 0.0033 | 1.6781, 1.6861, 1.6806 |
| shuffled_history + spatial_plus_tgm | 1.6501 | 0.0107 | 1.6380, 1.6641, 1.6482 |

- **Gate 1** (full+TGM < full+spatial on mean): **PASS** (1.6456 < 1.7023).
- **Gate 2** (full+TGM < current_frame+TGM on mean): pass on mean (1.6456 < 1.6816), but **not
  robust** — full's seed range [1.60, 1.72] fully contains current_frame's range [1.68, 1.69].
- **Gate 3** (full+TGM < shuffled+TGM on mean): pass on mean (1.6456 < 1.6501, ~0.3% margin), but
  **not robust** — full's worst seed (1.721) is worse than shuffled's best seed (1.638).
- Geometric metrics (MAE, harmful_rate) show no consistent separation either: full_history+TGM
  MAE 5.130±0.079 vs shuffled 5.150±0.023 vs current_frame 5.131±0.019 — all within noise of each
  other. harmful_rate is high-variance across every config (0.0–0.54 across seeds), with no
  config reliably safer.

`full_history+TGM`'s seed-1 run is the outlier driving the high std (1.721 vs 1.60–1.61 for the
other two seeds) — one bad initialization/seed is enough to erase the group's mean advantage.

## TGM-weight sweep (full_history, λ ∈ {0.2, 0.5, 1.0}, n=3 seeds each)
| λ | tgm_error mean±std | MAE mean±std | harmful_rate mean±std |
|---|---|---|---|
| 0.2 | 1.6887 ± 0.0147 | 5.118 ± 0.009 | 0.303 ± 0.068 |
| 0.5 | 1.6619 ± 0.0081 | 5.137 ± 0.050 | 0.327 ± 0.042 |
| 1.0 | 1.6456 ± 0.0534 | 5.130 ± 0.079 | 0.234 ± 0.167 |

Lower λ gives *more stable* (lower-variance) TGM error and MAE at a similar or better mean, at the
cost of a smaller nominal temporal gain. λ=1.0's better mean is bought entirely by higher variance
(one lucky-ish pair of seeds) — not a free win.

## D4D zero-shot re-evaluation (all 3 TGM variants, 3 seeds, 9 clips/config)
`mc_inconsistency` (motion-compensated disparity inconsistency), vdpp − raw, paired per-clip
bootstrap (2000 iters), 4 clips/seed with valid RAFT flow:

| variant | seed0 | seed1 | seed2 | sign stable? |
|---|---|---|---|---|
| full_history | +0.0001 [-0.0002,0.0005] | 0.0000 [0,0] | +0.0029 [-0.0,0.0088] | **No** — never clearly negative |
| current_frame_only | −0.0020 [-0.0058,-0.0] | −0.0020 [-0.0061,0.0] | −0.0038 [-0.0062,-0.0017] | Yes (all improve) |
| shuffled_history | −0.0006 [-0.0019,0.0002] | +0.0018 [-0.0001,0.0057] | +0.0001 [-0.0,0.0003] | No |

**Gate 4 fails**: full_history's D4D temporal effect is not stable in sign, and is essentially
zero-to-slightly-worse. Ironically `current_frame_only` — the ablation with no memory — shows the
most consistent (small) D4D improvement, which is further evidence the "temporal gain" measured on
SCARED does not reflect the model exploiting causal history on out-of-distribution data.

Sparse Zivid-anchor MAE (137 anchors, zero-shot, no identity collapse):
| variant | seed | ΔMAE (vs raw 4.090) | harmful | modified |
|---|---|---|---|---|
| full_history | 0 | +0.033 | 0.010 | 0.013 |
| full_history | 1 | −0.000 (no-op) | 0.000 | 0.000 |
| full_history | 2 | +0.038 | 0.018 | 0.016 |
| current_frame_only | 0–2 | +0.040…+0.047 | 0.000–0.006 | 0.014–0.018 |
| shuffled_history | 0–2 | +0.028…+0.051 | 0.014–0.080 | 0.014–0.016 |

**Gate 5 passes**: no identity collapse, no unsafe false activation; harmful rates are all low
(≤0.08) and comparable across variants (shuffled_history seed0 is the highest at 0.08, still
small). All three variants give a small, similar, safe zero-shot improvement on D4D anchors —
consistent with the refiner learning a generically-useful conservative correction, not something
specific to causal temporal structure.

## Decision (5-gate rule)
1. full+TGM beats full+spatial (SCARED): **PASS**
2. full+TGM beats current_frame+TGM: pass on mean, **fails robustness** (seed overlap)
3. full+TGM beats shuffled+TGM: pass on mean, **fails robustness** (seed overlap)
4. D4D sign stability: **FAIL**
5. No identity collapse / unsafe activation: **PASS**

**Verdict: MARGINAL.** Do not claim confirmed temporal usage at this pilot scale. TGM loss
regularizes the output (gate 1 genuinely passes, robustly — spatial_only is consistently worse),
but the evidence that the *causal-history mechanism itself* is exploited (gates 2, 3, 4) is weak
and not robust to seed variance, and shows no reproducible benefit on the out-of-distribution D4D
data. The original pilot's strong claim was likely driven by the loss confound, not genuine
temporal-order usage.
