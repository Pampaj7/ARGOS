# ARGOS v2 — Stereo-photometric causal-memory audit

## Question

The large-scale BiDA audit proves that the causally warped t-1 disparity has
positive raw-or-memory oracle gain, but all selectors based only on temporal
disagreement, flow and local disparity maps leave much of that signal unused.
This audit asks a smaller, falsifiable question: **does the current rectified
stereo pair provide backbone-agnostic evidence for which of raw and aligned
memory is geometrically better?**

It is a quality cue only.  It does not change SEA-RAFT, BiDA, a cached stereo
prediction, or synthesize a new disparity.

## Existing contract

SCARED-C cache disparity is positive left disparity on the 144x180 cache grid.
For a candidate ``d`` at left coordinate ``x``, the corresponding right-image
coordinate is therefore ``x - d``.  `stereo_photometric.py` uses bilinear
`grid_sample`, `align_corners=True`, zero padding and the same `(W-1,H-1)`
normalization convention as the canonical BiDA warp.  It returns explicit
right-image in-bounds support.  Candidate comparison always uses the common
mask:

```text
GT coverage > threshold
& raw prediction valid
& causally sampled memory valid
& BiDA warp support
& raw right-image support
& memory right-image support
```

Thus raw, memory, oracle and any photometric selection have identical support.
The candidate memory remains causal: it is obtained only from t-1 via the
validated current-to-past SEA-RAFT flow and canonical BiDA warp.

## Deterministic cues

For raw and aligned-memory disparity independently the audit computes:

1. RGB L1 reprojection residual;
2. local-mean RGB L1 over an odd validation-selected window;
3. luminance ZNCC cost over the same window.

Local ZNCC is included because surgical imagery contains illumination changes
and specularity; it is not presumed reliable.  The direct policy is also
deliberately minimal: choose memory only if its candidate cost is lower than
raw by a validation-selected margin.  There is no learned threshold, no RGB
encoder, no cost volume and no target-domain adaptation.

## Why this is scientifically distinct from prior failed selectors

Prior universal selectors saw temporal evidence but not the stereo-image
evidence of whether either candidate actually explains the *current* right
view.  Classical stereo confidence relies on matching ambiguity and local
consistency; examples include Haeusler et al., CVPR 2013 and Poggi & Mattoccia,
CVPR 2017.  Surgical stereo work also explicitly notes that photometric
consistency is informative but fails under specularity/texturelessness (Song
et al., 2021/2022 BDIS).  Therefore this audit must quantify both utility and
failure modes, rather than treating a low residual as ground truth.

The method remains compatible with ARGOS's frozen-backbone principle because
it uses only left/right RGB, externally cached disparity candidates, and the
already validated causal alignment.

## Selection protocol and leakage boundary

1. Run a small smoke only to validate sign, masks and finite metrics; delete
   its output after success.
2. On SCARED-C `dataset_7_keyframe_1/2`, sweep only the preregistered local
   windows `{15,21,31}` and residual margins `{0,.002,.005,.01,.02}` for RGB
   L1 and ZNCC.
3. Freeze the best positive-gain candidate subject to false-update <2%, clean
   degradation <1% and nonzero coverage.  If no such point exists, record
   infeasibility; do not relax it using the final test.
4. Evaluate the frozen point once on `dataset_7_keyframe_3/4` for the three
   seen backbones.  Only a passing seen result permits untouched
   Fast-FoundationStereo and CREStereo evaluation.

This historical keyframe split is retained only to compare directly with the
existing selector work.  It is not sufficient for a paper-level cross-session
claim: any apparent pass must subsequently undergo leave-one-dataset-ID-out
validation, with calibration and test dataset IDs disjoint.

## Initial non-binding probe correction

An early ten-pair notebook-style calculation suggested 25.7% recovery for
pixel RGB residual and 46.3% for 31x31 local RGB L1.  It did **not** intersect
the memory candidate's right-image support, and is therefore not a valid
paired comparison.  The executable smoke now applies the full common mask and
is the sole sign/mask check retained by this study.  No candidate was selected
from either smoke calculation.

## Promotion logic

The cue is useful only if the frozen held-out result improves geometric EPE
with a positive sequence-level interval, preserves clean predictions, and
survives backbone transfer.  Better temporal smoothness alone is never a
promotion criterion.  An OOD/cross-domain claim additionally requires each
domain to provide valid rectified stereo pairs and valid geometry; no such
claim is made by this SCARED-C audit alone.

## Validation result: deterministic policy

The completed validation sweep (`results/stereo_photometric_selector/validation`)
processed all 3,819 held-out keyframe-1/2 pairs across the three seen
backbones.  Under the preregistered safety constraint, its chosen policy is
ZNCC, 21x21, margin .002.  It has positive gain (+.000237 px) but recovers
only 1.01% of the `.02338 px` oracle gain, at 1.69% coverage.  The higher
coverage L1 policies recover at most 12.1% but violate clean-degradation and
false-update limits by a large margin.  Consequently direct photometric
selection is **NO-GO** and is not sent to final-test/unseen evaluation.

The next controlled use of this evidence is a fixed-input ablation in the
already existing utility-selector CNN: it receives raw/memory local L1 and
ZNCC maps while retaining the same causal t-1 task, frozen BiDA/SEA-RAFT, and
raw-by-default decision.  This is justified only because the cost cue has
some unsafely distributed utility; it is not a new refiner or a replacement
for photometric validation.

### One-epoch feature pilot (not a promotion result)

With the same seed, data, 303k-parameter local CNN, batch schedule and legacy
utility objective, adding the five frozen photometric channels changes
validation AUROC/AUPRC from `.94565/.42652` to `.94921/.43207` and utility MAE
from `.04544` to `.04509`.  Decisive helpful-vs-harmful AUROC remains near
chance (`.47415 -> .47641`) and the 25%-harm-cost policy constraint is
infeasible for both.  Thus this is only enough evidence to run one complete
seed; it is explicitly not evidence that photometry solves safe selection.

### Full-seed early-stop decision

The subsequent 12-epoch photometric seed was stopped after epoch 4, before any
test or unseen data were opened. Validation loss plateaued (`.32566`,
`.31941`, `.31889`, `.31861`), decisive AUROC was `.463`, `.467`, `.466`,
`.431`, and the predeclared constrained policy remained infeasible at every
checkpoint (harm-cost fraction `.925`). Continuing the identical legacy
objective would not answer a new scientific question and would delay the
independent LRC audit. This run is consequently a **negative pilot**, not a
completed seed or a basis for a performance comparison.
