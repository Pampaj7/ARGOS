# ARGOS v2 Minimal NVDS-Lite Closure Plan

## Goal

Close NVDS-lite as a baseline, not as the next main ARGOS v2 architecture.

The single scientific question is:

```text
Does corrected explicit causal history provide real benefit over current-only, shuffled-history, and naive concatenation while preserving geometry and safety?
```

## Precondition

Before running the closure, fix or add evaluation support for proper causal streaming/sliding evaluation. The current `eval_sequences` resets every non-overlapping `clip_len=8` window and is not adequate for the final decision.

Training can still use `clip_len=8`; final evaluation should not reset history at arbitrary non-overlapping boundaries.

## Exact Configurations Still Needed

One seed first, not 18 runs:

1. Raw S2M2 baseline from the same validation/test split.
2. `A`: NVDS-lite, RGB-D, full causal history, warp loss enabled.
3. `D`: NVDS-lite, RGB-D, current-frame-only.
4. `E`: NVDS-lite, RGB-D, shuffled causal history.
5. `F`: naive causal multi-frame concat baseline.
6. Optional `B`: same as A but without warp loss.

Only if A clearly beats D/E/F without safety regression should the matrix be promoted to three seeds and optional C/B ablations.

## Existing Reusable Checkpoints

No checkpoint is reusable for the closure conclusion.

Reusable assets:

- auxiliary RGB/flow/occlusion cache under `results/03_temporal_refinement/nvds_lite_causal_pilot/aux_cache`;
- flow/mask/causality validation outputs;
- current low-memory matcher code;
- current corrected loss weights and gate initialisation.

## Validation Split

Use the existing sequence-disjoint split from:

`results/03_temporal_refinement/training/refiner_failure_analysis/proposed_balanced_split.json`

Do not change splits between configs.

## Metrics Required

Geometric:

- raw/refined MAE;
- Bad1/Bad3/Bad5;
- New-Bad3;
- harmful/beneficial correction rate;
- raw-good preservation;
- boundary MAE;
- modified-pixel ratio;
- correction magnitude distribution.

Temporal:

- TGM error;
- temporal error jitter;
- high-frequency error energy;
- motion-compensated inconsistency;
- correction flicker;
- sign-flip and isolated activation rates.

Behaviour/runtime:

- gate mean/std/p5/p50/p95;
- residual percentiles and saturation fraction;
- inference ms/frame;
- peak VRAM;
- parameter count.

## Expected Runtime

Observed diagnostics:

- A config batch 4 train step: about `1.2 s/step`;
- validation pass: about `99 s`;
- peak VRAM: about `8.8 GB` on H100;
- 100-step timing run total: about `251 s` including evaluation.

A one-seed five-config closure is feasible on one or two H100s, but should be launched through the correct LSF/compute-node pattern, not from the login host as a detached local process.

## Stop/Go Decision Rules

Declare NVDS-lite `NOT CONFIRMED` if A does not beat both D and E on temporal metrics, or if gains come with material geometry/New-Bad3 degradation.

Declare `MARGINAL` if A improves temporal metrics but distributions overlap strongly or only one metric improves.

Promote to three seeds only if A clearly:

- beats D current-only;
- beats E shuffled-history;
- beats F concat;
- avoids identity collapse;
- preserves geometry and New-Bad3 safety;
- shows interpretable difference from optional B no-warp.

## Explicit Scope

This is a baseline closure. It is not the main ARGOS v2 architecture. The SOTA document points next to aligned-local-only and causal forward BiDA-style propagation.
