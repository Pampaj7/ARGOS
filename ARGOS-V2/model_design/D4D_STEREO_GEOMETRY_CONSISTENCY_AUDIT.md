# ARGOS v2 — D4D stereo-geometry consistency audit

## Trigger

The existing D4D cache audit correctly reported an apparently global
RAFT-Stereo scale error against the curated Zivid disparity: on specimen 2,
the raw/GT correlation was `.953` but the cache-grid raw and GT medians were
approximately `3.95` and `16.27` pixels.  That observation alone cannot tell
whether the frozen backbone, the cache conversion, or the reference geometry
is responsible.

A forward-only anchor check exposes a material consistency issue.  On anchor
`specimen_2 / 2025_04_10-16_07_15 / 1744294061_039222002`, all maps use the
same rectified 894x714 left/right pair and are converted to the canonical
144x180 grid.  Over the same Zivid-valid pixels:

| candidate | median disparity | census cost | local RGB-L1 |
|---|---:|---:|---:|
| curated Zivid GT | 16.15 | .2130 | .0431 |
| curated Zivid GT × .25 | 4.04 | .0490 | .0160 |
| RAFT-Stereo cached raw | 3.88 | .0274 | .0042 |

Thus the direct current stereo image correspondence favours roughly one
quarter of the curated-Zivid disparity, whereas the stored curated disparity
is four times larger.  S2M2-S and StereoAnywhere happen to be numerically
near the curated GT but also have much worse left-right photometric matching
on this anchor.  This is not evidence that photometry alone is a replacement
for structured-light geometry; it is evidence that the two measurements are
not yet demonstrably in the same rectified stereo-disparity convention.

## What this audit does and does not do

The next deterministic audit covers all valid D4D anchors and reuses the
existing left/right rectification and cache-grid contracts.  It measures
current-frame stereo reprojection cost for:

1. curated Zivid disparity;
2. fixed, preregistered scaled copies of it;
3. frozen cached disparities from S2M2-S, RAFT-Stereo and StereoAnywhere.

It also runs a fixed SCARED-C control, where existing GT and rectified stereo
must agree under the same implementation.  It creates no prediction or flow
cache, trains no network, modifies no calibration, and never uses the result
to rescale D4D labels or model predictions.

The question is solely a data-validity one:

> Does the curated D4D disparity reproduce the observed rectified left/right
> correspondence at the declared pixel grid?

If a near-constant non-unit image-optimal scale persists across anchors and is
absent in the SCARED control, D4D must be labelled **geometry-contract
inconsistent** for frozen-stereo EPE claims.  Existing D4D outcomes then remain
diagnostic only; they cannot falsify or promote a BiDA refinement.  The source
of the discrepancy must be repaired at the calibrated camera/pose/GT level,
not hidden by a learned temporal module or GT-derived scale correction.

## Completed frozen audit — 2026-07-19

`scripts/run_d4d_stereo_geometry_audit.py` evaluated all 156 available D4D
anchors and a fixed 25-frame SCARED-C rectified-GT control at the canonical
144x180 grid.  It used no learned module, no GT correction and no newly written
prediction/flow cache.  Candidate costs were evaluated over the same
GT/candidate/right/census support within each comparison.

| data / fixed candidate family | local RGB-L1 winner | ternary-census winner |
|---|---:|---:|
| D4D curated Zivid disparity × fixed scales | **0.25** | **0.25** |
| SCARED-C established GT × fixed scales | **1.0** | **1.0** |

Among the 112 D4D anchors with nonempty census-supported Zivid evaluation,
scale `.25` was the per-anchor local-L1 winner for 83 and the census winner
for 72.  For the same support, mean local RGB-L1/census were `.0420/.1572` at
Zivid × `.25` versus `.0471/.2455` at the stored scale `1.0`.  The effect is
large enough to be material, but the median valid support is only 64.5 cache
pixels per finite D4D anchor; it is therefore a **data-contract diagnostic**,
not a replacement for dense geometric validation.

The SCARED-C control selected scale `1.0` for both costs in all 25/25 frames
(mean local RGB-L1 `.0264`, census `.0995` at scale `1.0`).  This rules out a
generic sign, resize, or stereo-warp convention error in the audit itself.

As a separate provenance check, each of the 156 anchor images was re-read
from the causal context frame, rectified with its recorded camera maps, and
compared to the `left_rectified.png` and `right_rectified.png` stored next to
the curated GT.  All 312 comparisons were bit-exact (maximum uint8 absolute
difference `0`).  The scale result is therefore not explained by a wrong
frame ID, temporal offset, or left/right image mismatch in this audit.

The full compact artifacts are in
`results/d4d_stereo_geometry_audit/`.  No D4D GT or model output has been
rescaled.  Until the calibrated D4D reference contract is repaired upstream,
D4D may be retained for no-reference temporal diagnostics only and cannot be
used to claim frozen-stereo EPE, to tune a selector, or to falsify a
backbone-agnostic BiDA result.
