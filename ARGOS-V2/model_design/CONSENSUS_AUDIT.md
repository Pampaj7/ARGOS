# Cross-Memory Consensus Correction (CMC) — design and predeclared gates

## What this is

An ARGOS-original, zero-parameter causal disparity correction. It is not adapted
from an external repository; it is a new combination motivated directly by the
measured failure modes of the four prior studies. This document is written
before the validation runs; the promotion gates below are predeclared.

## Evidence base (all numbers from committed results, coverage 0.50)

1. **Per-age relative selection is nearly unlearnable.** The learned t-1
   selector answering "is this aligned memory better than raw?" reached AUROC
   0.549-0.554 with recall 0.9-2.0% (`results/learned_t1_refiner/
   final_seen/calibration_metrics.json`). The PPM learned long-memory selector
   was worse than the t-1 refiner on every backbone with false-update 42%
   (`results/ppmstereo_validation/aggregate_summary.json`).
2. **Absolute raw-error detection is easy.** The same run's `raw_error_gate`
   head is well calibrated: Brier 0.027-0.035, ECE 0.02-0.04. Detecting "raw is
   wrong here" is tractable; picking which single memory beats it is not.
3. **The four memory ages are individually weak but collectively informative.**
   Oracle best-age distribution is nearly uniform (raw best 26.7-28.3%, each
   age best 17.0-19.4%; `memory_oracle_summary.json`), and each added age
   contributes a repeatable incremental oracle gain (t-2: +0.046, t-4: +0.043,
   t-8: +0.039 px on RAFT-Stereo). No prior study used the agreement *among*
   the memories; every prior gate scored each memory against raw in isolation.
4. **The binding constraint is safety, not accuracy.** Every learned approach
   improved aggregate EPE but failed clean-pixel safety (clean degradation
   ~30-37%, false-update ~28-42%).

## Formulation

Given BiDA-aligned past disparities `m_k`, ages `k ∈ {1,2,4,8}`, with per-pixel
validity `v_k` (aligned validity ∧ warp support), define per pixel:

```
n      = Σ_k v_k                        (valid witness count)
median = median({m_k : v_k})            (consensus estimate)
spread = median({|m_k − median| : v_k}) (MAD, consensus tightness)
d      = |raw − median|                 (raw's disagreement with consensus)
gate   = (n ≥ n_min) ∧ (spread ≤ τ_s) ∧ (d ≥ τ_d + κ·spread)
refined = raw + gate · clip(median − raw, ±B)
```

`B = 3 px` (the validated bound of the t-1 A2 refiner). The statistical claim:
several independently-estimated, independently-warped past disparities agreeing
tightly with each other while raw sits far outside their consensus is evidence
that raw — not the memories — is the outlier. This converts the unlearnable
relative question (fact 1) into the tractable outlier question (fact 2), using
the collective information (fact 3), with safety by construction (fact 4): on
clean pixels raw agrees with the consensus, `d` is small, and the gate stays
closed.

Known failure mode, accepted and measured rather than hidden: correlated memory
error. If a past error persists across all four ages (static artifact, e.g. a
specular patch fixed in image space while `raw` is momentarily correct), the
consensus is wrong and confidently so. `κ·spread` does not protect against
this; only the empirical false-update rate can say how often it happens.

## Relation to state of the art

- PPMStereo picks top-K memories by learned quality/similarity — per-memory
  scores, no inter-memory consensus. Its released selector is non-causal.
- BiDAStabilizer fuses one past + one future frame convolutionally.
- Robust multi-frame median/MAD is classical (e.g. TSDF fusion, burst
  photography), but we found no stereo-video work using inter-memory MAD
  agreement as the *gate* for overriding the current frame; novelty here is the
  combination and the safety framing, not the median itself.

## Predeclared validation protocol (mirrors prior studies)

Namespace `cache-grid-from-cached-predictions`, pixel-count-weighted metrics,
common mask = GT coverage ∧ raw-valid ∧ aligned-t1-valid ∧ t1-warp-support,
primary coverage 0.50, bound 3 px, ages (1,2,4,8), SEA-RAFT flow.

- **Stage 1 (sweep, train split only).** Grid over
  `n_min ∈ {3,4}`, `τ_s ∈ {0.25,0.5,1.0}`, `τ_d ∈ {0.5,1.0,2.0}`,
  `κ ∈ {0,1}` on 3 train-split sequences, 3 seen backbones. Also compute the
  mechanism oracle (apply correction only where it truly helps) as ceiling.
  Gate to stage 2: at least one config with aggregate gain ≥ 0.005 px AND
  false-update rate ≤ 0.20 AND clean1 degradation ≤ 0.15 (i.e., materially
  safer than every learned predecessor, not merely non-worse).
- **Stage 2 (frozen config, held-out).** The single best stage-1 config,
  frozen, on dataset_7_keyframe_1..4, 3 seen backbones, 300 frames.
  Gate to stage 3: gain ≥ 0.005 px on ≥ 2/3 backbones AND stage-1 safety
  bounds hold on held-out data.
- **Stage 3 (unseen backbone).** Fast-FoundationStereo, never used for
  selection or tuning. Reported whatever the outcome.

If stage 1 fails, stages 2-3 are not run and the verdict is NO-GO with the
sweep table as evidence. Hyperparameters are selected on train sequences only.

## Cost

No training. Consensus math is O(K) numpy per pixel. Runtime cost remains the
four-age flow inference already measured at ~10.4 ms/frame total in the PPM
study; CMC adds only element-wise ops.

## Outcome (added after the stage-1 run; nothing above was edited)

**NO-GO — stage-1 gate failed on the train sweep; stages 2-3 not run.**
Best config gain +0.0013 px (gate: >= 0.005) with false-update 22.8%
(gate: <= 20%); aggressive configs reach 86% false-update. The predeclared
correlated-memory-error failure mode is dominant, not exceptional, and the
mechanism ceiling is only 0.0297 px (30% of the multi-memory oracle). Full
numbers and interpretation: `results/consensus_validation/README.md`.
