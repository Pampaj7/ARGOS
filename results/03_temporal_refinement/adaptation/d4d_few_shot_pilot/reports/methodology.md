# Methodology — D4D few-shot adaptation pilot

**Hypothesis**: SCARED-trained refiners retain large-error correction skill on D4D; their
activation policy is miscalibrated (false-activates on raw-good pixels).

**Data**: 4-frame causal shards from the zero-shot pipeline (GT only at anchor; pixel-level
supervision, ~40k valid px/anchor). Train sessions = existing `few_shot/*` split train MINUS
frozen-overlap; **frozen eval** = `session_disjoint` validation (22 anchors, selection) and
test (30 anchors, reporting) for ALL runs. Skips + starvation documented (1-2-session splits
hold 0-2 anchors).

**Modes / trainable params** (audit/model_parameter_groups.csv): v3.2c calib=65 / head=130 /
full=194,818; EGBM-v3-CARE-S calib=10,856 (bad/damping/threshold/router/boundary_atten) /
head=94,577 (+experts/boundary/care_head) / full=4,420,122. Scratch = same arch random init.
S2M2 backbone frozen everywhere. Gradient isolation verified bitwise per run.

**Loss** (predefined): L1(refined,gt|valid) + 1.0·|applied| on raw-good(<1px) + 0.05·|applied|.
**Selection** (predefined, val only): MAE + 0.02·newBad3% + 0.5·harmful_rate.
**Budget**: ≤50 epochs, patience 8, AdamW (lr 1e-3/3e-4/3e-5/3e-4 calib/head/full/scratch),
clip 1.0, batch 4, deterministic seeds, fp32.
**Selectivity** = pooled ΔMAE(raw err>6px) − max(0, −ΔMAE(raw err<1px)).
