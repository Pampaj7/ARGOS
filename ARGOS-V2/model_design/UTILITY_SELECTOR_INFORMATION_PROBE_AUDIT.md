# ARGOS v2 — Candidate-conditioned stereo-evidence information probe

## Question and frozen comparison

The strict session-disjoint 128-channel, 8-residual-block utility selector
recovers only 26.96% (+/- 4.63% over three seeds) of the cache-grid
raw-versus-causally-aligned-t-1 oracle gain on dataset 7.  Increasing the
selector from 303,747 parameters / 19x19 receptive field to 2,379,011
parameters / 35x35 receptive field therefore did not cross the predeclared
50% promotion gate.  This probe holds that 128x8 selector, the utility target,
dead band, legacy objective, causal BiDA warp, optimizer, crop, sampler,
calibration and hard abstention policy fixed.  It changes only its observable
evidence.

The control sees the validated 13 universal maps.  The treatment appends
candidate-conditioned current-frame stereo correspondence evidence for the
two already-fixed candidates: raw `d_t` and BiDA-aligned `d_{t-1 -> t}`.
Neither candidate, the geometric validity mask, nor the decision rule changes.

## Why this is a distinct test from the earlier photometric/census audit

The previous deterministic census selector evaluated a cost at each candidate
as a direct decision score and was a NO-GO.  It did not supply a learned
selector with the *local horizontal cost shape around both candidates*.
Consequently it did not test whether candidate support, local ambiguity and
the raw-versus-memory difference make relative utility more observable.  This
probe is not a retry of RGB-L1 gating and does not use RGB-L1 as its main cue.

## Fixed correspondence representation

At cache-grid resolution, for each candidate and fixed offsets
`[-4, -2, -1, 0, +1, +2, +4]` pixels, the implementation:

1. samples the current rectified right image at `x_right = x_left - (d+o)`
   with the already validated positive-left disparity convention and
   `align_corners=True`;
2. computes a 5x5 ternary census mismatch against the current left image;
3. carries exact right-image and census-window support without changing the
   existing BiDA/GT evaluation mask.

The full treatment receives both seven-point curves, candidate cost, local
minimum cost, offset to that minimum, best/second-best margin, local curvature,
normalised cost-curve sharpness, per-candidate local support fraction, direct
raw-minus-memory statistic differences, candidate support, and a fixed image
boundary flag.  Census costs and margins lie in `[0,1]`; offsets are divided by
4; curvature is divided by 2 and clipped to `[-1,1]`; sharpness and supports
lie in `[0,1]`.  These are fixed backbone-independent normalisations.  An
empirical summary is collected only from training batches and written beside
each seed; it is descriptive and is never fitted on validation or test data.

## Controlled variants and selection protocol

* **A / control:** no matching channels (13 universal channels).
* **B1 / cost:** candidate census costs and support only.
* **B2 / shape:** B1 plus local cost statistics and their differences.
* **B3 / full:** B2 plus the two explicit seven-point cost curves.

The main three-seed campaign compares A with B3.  B1/B2 are training-domain
ablations only: a single seed each, selected using dataset 2 validation only,
and are never ranked using dataset 7.  Dataset 7 remains unread until the
method, normalisation and validation-selected policy are frozen.

The exact strict split is: train datasets 1, 3 and 6; validation/calibration
dataset 2; final test dataset 7.  Training uses S2M2-S, RAFT-Stereo and
StereoAnywhere with the existing balanced sequence sampler.  Fast-
FoundationStereo, CREStereo and all OOD data remain prohibited unless the
strict seen promotion gate passes.

## Relation to CODD and scope

CODD uses internal stereo left/right features, local confidence perturbations,
pixel/patch correlations, semantic features and learned scene flow.  This
probe deliberately tests only the smallest universal analogue of its
candidate-conditioned stereo correspondence cue: deterministic census curves
from current rectified RGB, with no internal cost volume, semantics or learned
motion.  Thus a negative result is evidence about this compact observable
information, not a claim that all correspondence information is useless.

## Predeclared interpretation

* Strong train fit but test recovery near 27%: the extra evidence does not
  generalise across sequences.
* No train/overfit improvement: verify extraction, support, normalisation and
  target alignment before drawing an informational conclusion.
* Better ranking but unchanged realised gain: calibration/cost-sensitive action
  selection, not representation ranking, remains limiting.
* Only a test recovery above 50% with acceptable safety would promote the
  treatment beyond the strict seen protocol.

## Completed outcome (2026-07-21)

The B3/full treatment passed its tiny overfit and end-to-end smoke contracts,
then completed three strict seeds.  Its dataset-7 mean oracle recovery was
27.235% (sample standard deviation 1.852%), versus 26.963% (4.629%) for the
frozen 13-map 128x8 control.  Mean selector EPE changed from 0.519577 to
0.519434 px: a 0.000142 px difference.  The extra channels instead raised
coverage from 0.989% to 1.872%, false-update rate from 0.548% to 1.287%, and
clean-pixel degradation from 0.286% to 0.543%.

Thus neither the existing capacity limit nor this compact external census
representation is the dominant bottleneck under the strict sequence-disjoint
protocol.  The result does not show that all stereo-matching evidence is
uninformative: it rules out this minimal, backbone-independent, local
census-cost representation as a material solution.  The predeclared >50%
recovery promotion gate failed in every seed, so Fast-FoundationStereo,
CREStereo, and OOD datasets were not evaluated.
