# ARGOS v2 — frozen stereo left-right consistency audit

This directory contains a calibration-only audit of a universal current-frame
stereo confidence cue.  It does **not** contain a trained selector or refined
disparity.

## Scope and leakage boundary

- Sequences: `dataset_7_keyframe_1`, `dataset_7_keyframe_2` only.
- Backbones: `S2M2-S`, `RAFT-Stereo`, `StereoAnywhere` only.
- No final held-out keyframe sequence, unseen backbone, SERV-CT, D4D, or
  StereoMIS input is loaded for this calibration audit.
- Right-reference predictions are separate `_rightref_<backbone>` cache
  namespaces.  They are obtained from the frozen original backbone on
  `flip(right), flip(left)` and are horizontally unflipped before use.

## Metric contracts

All disparity values use cache-grid pixels at 144x180.  For a current left
candidate `d_L`, LRC is evaluated on the left grid as:

```text
abs(d_L(x) - d_R(x - d_L(x))).
```

Every LRC residual retains an explicit in-bounds support and a bilinearly
sampled right-valid mask.  Raw-vs-memory LRC comparison uses the paired mask:

```text
GT valid at coverage > .50
& raw prediction valid
& BiDA aligned memory valid
& BiDA warp support
& raw LRC valid
& memory LRC valid.
```

The main diagnostic is whether `LRC(raw) - LRC(memory)` predicts true
raw-minus-memory utility.  This is reported as a threshold-free AUROC and
correlation, separately per backbone and sequence, followed by a
sequence-unit bootstrap summary.  It is not a selected policy.

## Promotion gate

LRC may be supplied as a frozen input to exactly one controlled selector
ablation only if the calibration audit shows positive, consistent
raw-vs-memory utility ranking across the three seen backbones.  Any threshold,
architecture, checkpoint and policy must still be selected only on these two
sequences before opening final seen, unseen-backbone, or cross-domain data.
