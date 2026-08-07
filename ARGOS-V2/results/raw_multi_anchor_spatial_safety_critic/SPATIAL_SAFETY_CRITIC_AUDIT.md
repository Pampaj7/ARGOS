# ARGOS v2 — Raw Multi-Anchor Spatial Safety Critic: Freeze & Dataset-7 Audit

**Project:** ARGOS v2. **Experiment:** `raw_multi_anchor_spatial_safety_critic`.
**Frozen:** dataset 2 (validation) only. **Test:** dataset 7 (opened once, post-freeze).
**Seed:** 20260722. **Repo commit:** `998d77156257a674b0b3beda0e5ce67d1e1c796e`.
**Freeze manifest SHA-256:** `c620cedb84f20751270c0e9ff5ed590b96eaaa512dd17a740831c23dbe2d5d24`.

## Question

Can a lightweight spatial comparative error critic preserve the verified multi-anchor
advantage over bounded CODD-style H=4 fusion while transferring a safe and useful
intervention policy from dataset 2 to frozen dataset 7?

## Verdict

| Axis | Verdict |
|---|---|
| Geometry (multi-anchor > H4) | **GO** |
| Safety (harm ≤ 10% at useful coverage) | **CONDITIONAL** |
| **Overall** | **CONDITIONAL GO** |

13/14 decision checks pass; the sole failure is the harmful-update bound
(24.7% > 10%). The critic strictly dominates the ungated multi-anchor on every
safety axis, preserves the H4 advantage, and transfers from D2 to D7 without
coverage collapse — but no operating point on the frozen checkpoints reaches
≤10% harm at coverage ≥ 0.5%.

## Method (unchanged, veto-only)

`SpatialErrorCritic`: fully-convolutional CNN, GroupNorm(8)+SiLU, dilations
[1,2,4,2,1], receptive field 43 px, ~0.40M params. Predicts μ_raw, μ_proposed,
their aleatoric log-scales, and a harm logit from local spatial evidence.
Acceptance is the conservative LCB rule (Okati 2021 / Jitkrittum 2023: threshold
the predicted error *difference*, not confidence):

```
LCB = δ̂ − λ·σ_δ ,  δ̂ = μ_raw − μ_proposed ,  σ_δ = sqrt(exp(2 s_raw)+exp(2 s_proposed))
accept = eligible ∧ (LCB > τ_gain) ∧ (p_harm < τ_harm)   →  d_proposed, else d_raw bit-exact
```

The critic never changes the frozen anchor/weight/proposal, never reopens a
rejected pixel, and never enters the anchor bank. No GT, no future frames, no
backbone identity, no internal cost volumes (verified: `protocol_audit.json`,
`test_spatial_error_critic.py`, `test_spatial_critic_freeze_guard.py`).

Four cumulative evidence families were trained (seed 20260722, 12 epochs each):
geometry (28 ch), temporal (32), **stereo (45, predeclared primary)**,
plane_sweep (51, controlled ablation — an external small-perturbation
plane-sweep approximation, NOT a full disparity volume). Checkpoint selection
uses the unconstrained net-utility score (bug fix confirmed: selected epochs
7/8/8/8, none frozen at epoch 1); the harm-constrained point is a separate
diagnostic only.

## Dataset-2 calibration (validation only)

Full risk-coverage grid over (λ, τ_gain, τ_harm). Harm detection at margin 0.10:

| Family | harm AUROC | harm AUPRC | δ Spearman | best gain | cov | harm | feasible point? |
|---|---|---|---|---|---|---|---|
| geometry | 0.708 | 0.636 | 0.375 | 0.0056 | 3.0% | 25.4% | no |
| temporal | 0.747 | 0.689 | 0.437 | 0.0061 | 2.8% | 25.5% | no |
| **stereo** | 0.743 | 0.700 | 0.482 | 0.0063 | 3.2% | 26.3% | no |
| plane_sweep | 0.750 | 0.725 | 0.503 | 0.0065 | 3.0% | 25.7% | no |

**No family reaches harm ≤ 10% at coverage ≥ 0.5%.** To hold harm ≤ 10% the
critic must restrict to the top ~1% of *eligible* pixels (< 0.5% of valid) — the
same coverage-collapse regime that sank the scalar gate. The four families are
statistically indistinguishable in net gain (0.0056–0.0065); the plane_sweep
gain edge over stereo (~0.0002) is within noise. **Ablation conclusion:** the
extra plane-sweep evidence does not add *useful* information (marginally best
separability, no feasibility unlock, +12 warps/pixel of compute) — the
predeclared **stereo** primary is retained.

## Freeze (dataset-2 only)

Primary = stereo, net-utility-maximizing point: λ=0, τ_gain=0.025, τ_harm=0.7.
Predeclared safety-oriented secondary: λ=0, τ_gain=0.2, τ_harm=0.2 (D2 harm
18.7% @ 0.85% cov, clean-deg 0.33%, precision 74%). Manifest hashed; dataset 7
confirmed closed at freeze.

## Dataset-7 frozen evaluation (opened once)

Raw EPE **0.54793**, bounded H4 EPE (common support) **0.51726**.

| Policy | out EPE | gain vs raw | **gain vs H4** | cov | **harm (acc)** | harm (all-valid) | precision | clean-deg | degr-frame | ungated gain retained |
|---|---|---|---|---|---|---|---|---|---|---|
| ungated multi-anchor | 0.51048 | 0.03745 | +0.00678 | 15.5% | 35.0% | 5.42% | 49.0% | 4.86% | 26.2% | 100% |
| **critic (stereo)** | **0.50718** | **0.04074** | **+0.01008** | 7.9% | **24.7%** | 1.95% | 61.8% | **2.10%** | **16.7%** | **108.8%** |
| prior scalar gate (best) | 0.51713 | 0.03080 | +0.00013 | 1.1% | 12.1% | — | — | 0.07% | — | — |

- **Beats raw and H4.** Critic gain-over-H4 (+0.0101) exceeds ungated (+0.0068).
- **Dominates ungated on every safety axis:** higher net gain, harm 35→25%,
  precision 49→62%, clean-deg 4.9→2.1% (≤3% ✓), degraded-frame 26→17% (≤25% ✓),
  and retains 108.8% of ungated gain (it removes net-harmful interventions).
- **No coverage collapse:** 7.9% vs the scalar gate's 1.1%. The scalar gate
  reaches low harm only by near-total abstention, losing the H4 advantage
  (+0.0001); the spatial critic keeps 7× the coverage and a real H4 margin.
- **Harm bound unmet:** 24.7% ≫ 10% at the frozen point.

**Per-backbone (all three seen backbones improve):** S2M2-S gain 0.0264 / harm
25.1%; RAFT-Stereo 0.0531 / 25.9%; StereoAnywhere 0.0427 / 23.1%. No single
backbone carries the result.

**Per-sequence (all four improve):** kf1 0.0048/26.4%, kf2 0.0238/41.9%,
kf3 0.1040/22.1%, kf4 0.0247/24.8%. kf3 (raw EPE 1.23) contributes most absolute
gain; kf2 has the highest harm (41.9%).

**Distant-anchor authorization retained:** accepted-age fractions CS1 0.231,
CS2 0.257, CS4 0.262, CS8 0.250 — the critic still authorizes ~51% CS4/CS8
interventions (no distant-anchor suppression).

**Harm detection on D7:** AUROC 0.6915, AUPRC 0.5727, Brier 0.2716, ECE 0.2223,
δ Spearman 0.406. Substantially above the scalar gate (0.57) but below D2 (0.743)
— modest separability drift, poor probability calibration.

**Oracle ceiling (D7):** reject-all-harmful oracle → gain 0.0579 @ 9.8% cov, harm
0%. The critic recovers 71% of that harm-free gain but at 24.7% harm — the gap is
the harm it fails to reject.

**Transfer D2 → D7 (frozen stereo point):** coverage 3.2% → 7.9%, harm 26.3% →
24.7%, gain 0.0063 → 0.0407, AUROC 0.743 → 0.691. Harm rate and coverage
transfer **stably** (no worse on test) — the scalar gate's failure mode
(transfer instability + coverage collapse) is materially fixed.

**Runtime overhead:** 1.02 ms/frame, peak 714.6 MB GPU. Parameter overhead
~0.40M (< 1.5M budget).

## Interpretation

- **Geometry (GO):** immutable multi-age anchors beat bounded H4 on D7 across all
  backbones and sequences; the critic slightly extends the margin.
- **Safety (CONDITIONAL):** the spatial critic is a real, transferable improvement
  — it dominates the ungated proposal, beats H4, avoids coverage collapse, and
  triples-plus the usable coverage of the scalar gate while keeping the H4
  advantage. But it does not reach the ≤10% harmful-update target at any useful
  operating point. The binding limitation is **harm separability/calibration**
  (test AUROC 0.69, ECE 0.22), not transfer (stable), coverage (7.9%), geometry
  (GO), or distant-anchor authorization (intact).

## Next controlled experiment (CONDITIONAL GO branch)

Bottleneck = harm calibration/separability. Smallest next step: hard-negative
mining at occlusion/motion boundaries (where harm concentrates), post-hoc
calibration of the harm head (temperature/Platt), and a predeclared
harm-constrained operating point. If separability plateaus, move to **joint
training** of candidate utility + pairwise fusion + spatial harm prediction +
exact abstention (multi-seed), rather than a frozen post-hoc veto. Keep bounded
CODD-style H4 as the strong baseline/fallback.

## Scope / non-claims

Seen backbones only (S2M2-S, RAFT-Stereo, StereoAnywhere); D7 is in-distribution
test, not OOD. No claim of OOD generalization, unseen-backbone safety transfer,
clinical safety, conformal guarantees, or backbone-agnosticism.

## Artifacts

`calibration/freeze_manifest.json` (+`.sha256`), `calibration/frozen_policy.json`,
`aggregate_summary.json`, `verdicts.json`, `validation_summary.csv`,
`risk_coverage.csv`, `harm_detection_metrics.csv`, `fixed_risk_operating_points.csv`,
`checkpoint_selection.csv`, `protocol_audit/{existing_family_audit,checkpoint_hashes,dataset_access_audit,test_opened}.json`,
`frozen_test/{summary.json,summary.csv,frame_metrics.csv,per_backbone_metrics.csv,per_sequence_metrics.csv,harm_detection_metrics.csv,risk_coverage.csv,oracle_rejection_ceiling.csv,per_age_metrics.csv,residual_bin_analysis.csv,update_magnitude_analysis.csv}`,
`paper_ready_tables.tex`.
