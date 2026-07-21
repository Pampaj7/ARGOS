# ARGOS v2 — Multi-domain proposal-applicability control

## Motivation

The M1 raw-error detector control failed without loading any final-only data:
it retained only 29.96% of the new A2 gain on SCARED-C validation and authorized
0% of D4D specimen-2.  The failure is expected from a target mismatch.  A
decision to apply a frozen A2 update requires estimating

`u = |d_raw - d_gt| - |d_A2 - d_gt|`,

not merely whether `d_raw` has error.

## Fixed components

- Stereo caches, SEA-RAFT, canonical causal BiDA warp and all masks.
- The selected multi-domain A2 D2 checkpoint:
  `results/multidomain_a2_proposal_v2/M1/frozen/checkpoints/best_validation.pt`.
- The existing P4 `ProposalApplicabilityDetector` architecture (15,533
  parameters), 23 universal evidence maps and unchanged P4 losses.

Only P4 parameters are trainable.  No RGB semantic encoder, identity,
backbone label, future frame, cost volume or stereo hidden feature is exposed.

## Domain and leakage protocol

| Role | SCARED-C | D4D |
|---|---|---|
| Train | validated non-dataset-7 sequences | specimen_1 Zivid anchors |
| Selection/calibration | dataset_7 keyframes 1/2 | specimen_2 Zivid anchors |
| Final-only | dataset_7 keyframes 3/4 | specimen_3 |

SCARED-C is balanced over S2M2-S, RAFT-Stereo and StereoAnywhere.  D4D is
restricted to S2M2-S because the cache audit measured a global disparity-scale
mismatch for the RAFT-Stereo and StereoAnywhere D4D predictions.  SERV-CT,
StereoMIS, Fast-FoundationStereo, CREStereo and D4D specimen_3 are forbidden
before a frozen seen-domain gate.

## Target sanity check before training

On D4D specimen_2 (not a training specimen), the frozen D2 A2 proposal had
mean utility `+0.3326 px` over 8,933 paired-valid pixels.  With a 0.10 px
margin, 65.34% were helpful and 29.16% harmful.  Therefore direct universal
proposal-utility prediction is nontrivial: the safe policy cannot authorize
everything despite the aggregate A2 gain.

## Promotion gate

Calibration selects only utility margin, uncertainty ceiling and P4 helpful
class on SCARED validation plus D4D specimen_2.  Before final-only data are
opened, both domains must have nonzero coverage, safe false-update/clean
degradation rates and non-negative geometry gain.  This is a controlled
target-formulation test, not a new refiner or a claim about unseen domains.

## First frozen result (P4 D2)

The first policy selected on SCARED validation plus D4D specimen_2 used
continuous predicted utility, `sigma < 0.5`, and no hard helpful-class veto.
It was eligible on those selection domains.  On the one-shot held-out
evaluation it improved geometry on all opened sources:

| Final source | Raw EPE | P4 EPE | Gain | Coverage | Clean degradation |
|---|---:|---:|---:|---:|---:|
| SCARED-C, all seen backbones | 0.961605 | 0.955073 | +0.006532 | 2.68% | 0.79% |
| D4D specimen_3, S2M2-S | 7.300630 | 6.995149 | +0.305480 | 34.07% | 13.86% |

All three SCARED-C backbones improved and its safety gate passed.  D4D did
not: among its 101 raw-clean pixels in the common mask, 14 were degraded.  The
final D4D clean-degradation rate is therefore 13.86%, above the predeclared
10% limit.  This configuration is **NO-GO**; Fast-FoundationStereo,
CREStereo, SERV-CT and StereoMIS remain unloaded.  The result motivates a
separate exploratory cross-specimen data-efficiency control, not a relaxation
of the frozen-policy gate.

## Exploratory cross-specimen result

P4 was then given D4D specimen_1 and specimen_2 labels, while A2 stayed
frozen and specimen_3 stayed unseen.  To avoid selecting on a specimen that
had entered P4 training, its policy/checkpoint were selected on SCARED-C
validation only.  This exploratory protocol also failed: specimen_3 EPE
improved from 7.300630 to 6.479353 px, but the policy authorized 95.19% of
paired-valid pixels and produced 83.17% clean degradation.  SCARED-C test
also exceeded its safety limit (3.10% clean degradation).  More D4D
supervision therefore increased raw gain but did not produce a safe invariant
authorization rule.  This is another **NO-GO**, not a basis for unseen/OOD
evaluation.
