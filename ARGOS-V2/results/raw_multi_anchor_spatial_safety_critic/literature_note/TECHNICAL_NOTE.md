# ARGOS v2 — Spatial Comparative Error Critic: pre-implementation technical note

Date: 2026-07-23. Project: ARGOS v2. Experiment: `raw_multi_anchor_spatial_safety_critic`.

## Question

Can local spatial, temporal-alignment, and stereo-consistency evidence identify when the
already-frozen multi-anchor proposal is safer and more accurate than the current raw
disparity — accepting the frozen proposal or returning raw bit-exact?

## Why the prior scalar gate failed

The `raw_multi_anchor_selective_gate` experiment trained pointwise (1x1) gates over 20
scalar per-pixel statistics of the frozen proposal. Frozen dataset-7 outcome: harm AUROC
~0.57 (logistic 0.570, MLP 0.567, joint 0.566), i.e. barely above chance; the only
policies satisfying the harm<=10% constraint on validation collapsed to ~1% coverage and
lost the fixed-H=4 advantage (gain over H=4: -0.00032). Three structural reasons:

1. **No spatial context.** Harm concentrates at disparity boundaries, occlusions, thin
   instruments, and specular regions. A pixel's scalar statistics (score, update
   magnitude, temporal MAD, fb-confidence) are nearly identical on the two sides of a
   depth edge, while the true error sign flips. The 5x5 local mean/variance channels
   were precomputed statistics of *one* map, not learnable spatial filters; the gate
   could not see edge geometry, boundary orientation, or coherent misalignment patterns.
2. **Information already consumed.** The frozen refiner selected its proposal using
   essentially the same scalar family; pixels surviving its thresholds are the ones
   where those scalars are least informative. The residual harm signal lives in evidence
   families the refiner never saw: image-space alignment residuals and stereo
   photometric consistency of the competing hypotheses.
3. **Wrong target granularity.** A single harm probability collapses the comparative
   question (is e_proposed > e_raw?) into a binary label at one margin, discarding the
   magnitude information needed for a risk-sensitive operating point.

## Why the next model must use spatial evidence

The failure hypothesis (supported by the gate's per-bin analyses: harm concentrates in
high update-magnitude and low agreement bins but AUROC within bins stays near chance) is
that harm is identifiable from *local spatial structure*, not pointwise statistics.
CODD's fusion network — the validated reference for this pipeline — is itself a small
spatial CNN over both candidates, and the stereo-confidence literature (Poggi et al.'s
reviews; Lee et al. 2024 plane-sweep confidence) consistently uses local spatial
windows. A fully convolutional critic with ~31–63 px receptive field can see disparity
boundaries, occlusion fringes, thin structures, and coherent flow-failure blobs — the
structures that decide whether a temporal correction is safe.

## Why raw-error and proposal-error should be predicted separately

The decision object is the *difference* delta = e_raw - e_proposed (Okati et al. 2021:
optimal triage thresholds the predicted error difference; Jitkrittum et al. 2023:
confidence-only deferral is provably suboptimal when the second model is a specialist —
the temporal proposal is a textbook specialist). Predicting the two errors with separate
heteroscedastic heads (Kendall & Gal 2017) rather than delta directly:

- gives calibrated per-hypothesis aleatoric scales, whose combination
  sigma_delta = sqrt(sigma_raw^2 + sigma_proposed^2) yields a principled lower
  confidence bound LCB = delta_hat - lambda*sigma_delta for conservative acceptance
  (ConfidNet/Corbière et al.: learn failure-specific confidence, not generic softmax);
- keeps supervision dense and well-scaled on both branches (e_raw is supervised on all
  valid pixels, not only where the refiner proposed);
- lets diagnostics attribute failure: is the critic wrong about the raw map or about
  the proposal?

## Why generic confidence is not the correct target

A generic stereo-confidence map answers "is d_raw correct?". The veto decision needs
"is d_proposed *better than* d_raw here?". These disagree exactly where the decision
matters: at pixels where raw is bad, generic confidence is low, yet the proposal may be
worse (stale anchor, misalignment); at pixels where raw is good, the proposal can still
be accepted if nearly identical. SelectiveNet-style selective prediction optimizes
risk-coverage for a *decision*, and the decision-theoretic results above show the
sufficient statistic is the pairwise expected-error difference plus its uncertainty —
hence mu_raw, mu_proposed, sigma's, and an explicit harm-probability head trained with
asymmetric cost (false-safe weighted above false-reject), evaluated by risk-coverage
curves rather than AUROC alone (Jaeger et al. 2023; the gate's 0.57 AUROC with negative
net utility is the classic "good-looking score, negative net benefit" trap).

## Why this remains a frozen-backbone post-hoc adapter

ARGOS v1 refiners that retrained the correction path learned backbone+dataset error
signatures and catastrophically degraded OOD (SERV-CT: raw ~1.28 px -> refined ~6.5 px).
The multi-anchor geometry is validated GO (EPE 0.51047, +0.00679 over fixed H=4, all
backbones/sequences positive); only its *authorization* failed. Retraining the refiner
would confound geometry and safety and reopen a validated result. The critic is
therefore a veto-only adapter over completely frozen S2M2-S / RAFT-Stereo /
StereoAnywhere / SEA-RAFT / BiDA alignment / anchor construction / refiner checkpoint
(sha256 40526a32…): it may only replace d_proposed by d_raw, bit-exact via torch.where,
may not reopen rejected interventions, and never writes to memory. This isolates the
scientific variable — the evidence family needed for safe authorization — and keeps the
failure mode bounded by construction (worst case = frozen ungated proposal; best
case = oracle rejection ceiling).

## Evidence families staged (configurations C→E)

C: spatial geometry (hypotheses, updates, temporal consensus, support, fb-confidence,
gradients) — isolates the value of spatial context over pointwise scalars.
D: + temporal photometric residual for the selected anchor age (exposes flow failure,
occlusion, deformation, appearance change — BiDA alignment quality made observable).
E (primary): + current-stereo photometric consistency of d_raw / d_anchor / d_proposed
and their pairwise differences (external, cost-volume-free stereo evidence; comparative
by construction; treated as evidence, not a hard rule, because specular surgical scenes
violate brightness constancy).
F (controlled ablation, after C–E): external plane-sweep-style perturbation confidence
around d_raw and d_proposed, approximating Lee et al. 2024 without cost-volume access.

## References

- Geifman & El-Yaniv, SelectiveNet (ICML 2019). Selective prediction, risk-coverage.
- Corbière et al., Addressing Failure Prediction by Learning Model Confidence
  (NeurIPS 2019). Failure-specific confidence target.
- Kendall & Gal (NeurIPS 2017). Heteroscedastic aleatoric + epistemic uncertainty.
- Lee et al. (2024). Stereo confidence via disparity plane sweep, external to backbone.
- Wang et al. Evidential uncertainty for stereo matching (used as motivation only; no
  evidential regularizer in the primary loss — YAGNI).
- Li et al., CODD (WACV 2023). Comparative-error supervision of temporal fusion.
- Okati et al. (NeurIPS 2021); Jitkrittum et al. (NeurIPS 2023). Optimal deferral
  thresholds the predicted error difference; confidence-only deferral suboptimal.
- Jaeger et al. (ICLR 2023). AURC / risk-coverage over fragmented AUROC.
