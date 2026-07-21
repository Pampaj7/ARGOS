# ARGOS v2 — Utility-risk causal memory selector audit

Date: 2026-07-19

## Scientific status before this run

The full causal BiDA signal audit establishes a real raw-or-aligned-t-1
ceiling.  The first 303,747-parameter selector improves the cache-grid EPE on
both final-test sequences in all three seeds, but recovers only
31.30% +/- 3.21% of the oracle gain.  Its mean EPE is 0.702969 versus 0.723909
raw (coverage threshold 0.50).  Every seed therefore fails the preregistered
50% oracle-recovery promotion gate.

The corrected authoritative summaries are:

- `results/utility_memory_selector/aggregate_summary.json`
- `results/utility_memory_selector/per_seed_summary.csv`
- `results/utility_memory_selector/seed_{0,1,2}/frame_metrics.csv`

`dataset_7_keyframe_3` and `dataset_7_keyframe_4` are held-out keyframe
groups for the historical comparison, not independent acquisition sessions.
Backbone results on the same video are repeated measurements, not additional
independent samples.

There is a second hierarchy above keyframe groups: the 17 accepted groups come
from only five SCARED dataset/session IDs (`1`, `2`, `3`, `6`, `7`).  The
historical validation (`dataset_7` keyframes 1/2) and test (`dataset_7`
keyframes 3/4) therefore share the same higher-level acquisition.  The current
pilot is retained only for an exact baseline comparison.  A paper claim needs
leave-one-dataset-ID-out cross-fitting, with calibration and test dataset IDs
disjoint, and a dataset/session-unit uncertainty analysis.

Re-aggregating the frozen oracle audit at that stricter acquisition unit still
supports the foundational signal.  Equal-weight oracle gains for dataset IDs
1/2/3/6/7 are respectively 0.01025, 0.02846, 0.01845, 0.19899, and 0.05069
cache pixels.  All five are positive; the equal-dataset mean is 0.06137 and a
100,000-resample dataset-unit bootstrap CI95 is [0.01717, 0.13158].  Dataset 6
is much larger, so both per-dataset reporting and the grouped interval remain
mandatory.

## Failure found in the first selector

1. `BalancedSequenceSampler` equalizes backbone/sequence groups, but the loss
   remains dominated by the natural pixel distribution.
2. Pixels inside the +/-0.10 px indifference interval are assigned to the
   raw-better class by the legacy BCE.
3. Both magnitude heads are regressed to zero on classes for which their
   magnitude is undefined.  Their difference is therefore not a conditional
   expected action utility.
4. Three independently trained heads are joined by three post-hoc thresholds;
   the training objective does not optimize the realized raw-versus-memory
   decision.
5. The old calibration constraint found no feasible point and silently fell
   back to points with 23--32% harmful acceptance on validation.  New runs must
   record infeasibility explicitly.

## Evidence-backed minimal intervention

The CNN, receptive field, inputs, BiDA convention, SEA-RAFT, data, and target
utility are unchanged.  Only the objective and the interpretation of existing
heads change.

For helpful probability `p`, conditional helpful magnitude `g`, and
conditional harmful magnitude `h`, the decision score is

```
U_hat = p * g - (1 - p) * h
```

The initially proposed direct policy-risk objective was rejected from
training/validation alone: after one full epoch it had validation AUROC 0.439,
selected 100% of sampled pixels at its best score threshold, and had accepted
harmful/helpful magnitude 91.7%.  It therefore does not enter the seed study.

The retained `utility_calibrated` objective uses:

- BCE over all valid pixels, with indifferent pixels explicitly assigned to
  the default raw/abstain class;
- gain regression on helpful pixels only;
- harm regression on harmful pixels only;
- signed-utility regression;
- no direct policy gradient; authorization is selected later on validation;
- a fixed higher classification weight for harmful examples, as in the
  validated legacy selector.

This implements the selective-prediction principle of ordering conditional
risk and maximizing useful coverage under controlled error.  Relevant primary
references are Franc and Prusa, ICML 2019, *On discriminative learning of
prediction uncertainty*; Gangrade et al., AISTATS 2021, *Selective
Classification via One-Sided Prediction*; and Jeong et al., CVPRW 2025,
*Improving Optical Flow and Stereo Depth Estimation by Leveraging
Uncertainty-Based Learning Difficulties*.

Primary-source links:

- https://proceedings.mlr.press/v97/franc19a.html
- https://proceedings.mlr.press/v130/gangrade21a.html
- https://openaccess.thecvf.com/content/CVPR2025W/UNCV/html/Jeong_Improving_Optical_Flow_and_Stereo_Depth_Estimation_by_Leveraging_Uncertainty-Based_CVPRW_2025_paper.html
- https://openaccess.thecvf.com/content/ICCV2025/html/Jing_Stereo_Any_Video_Temporally_Consistent_Stereo_Matching_ICCV_2025_paper.html
- https://arxiv.org/abs/2510.20178
- https://openaccess.thecvf.com/content/CVPR2026/html/Zheng_Pip-Stereo_Progressive_Iterations_Pruner_for_Iterative_Optimization_based_Stereo_Matching_CVPR_2026_paper.html

Recent video-stereo systems such as Stereo Any Video and PPMStereo improve
temporal aggregation with rich cost-volume/internal representations.  Those
mechanisms are not portable across frozen heterogeneous backbones, and the
completed ARGOS universal PPMStereo adapter did not exploit its long-memory
oracle safely.  Their results support the value of temporal evidence but do
not supply a valid backbone-agnostic selector.  The selective-risk formulation
is therefore the smallest literature-supported change consistent with the
ARGOS contract; it does not reintroduce cost volumes, semantic encoders, or
hidden state.

Pip-Stereo (CVPR 2026) independently reports that iterative disparity updates
are spatially sparse and temporally redundant.  Although its pruner is tied to
an iterative backbone, this supports ARGOS's raw-by-default, sparse temporal
intervention policy rather than dense blending.

## Frozen tensor contract

Inputs remain 13 universal channels at `[B,C,144,180]`:

1. raw disparity;
2. aligned t-1 disparity;
3. signed disagreement;
4. absolute disagreement;
5. current-to-past flow x;
6. current-to-past flow y;
7. flow magnitude;
8. forward-backward confidence;
9. warp support;
10. aligned validity;
11. raw validity;
12. raw disparity gradient magnitude;
13. aligned-memory gradient magnitude.

There is no RGB, backbone identity, confidence internal to a stereo backbone,
cost volume, future frame, recurrent state, or long memory.

## Split and leakage policy

- Train: 13 accepted non-`dataset_7` sequences, three seen backbones.
- Validation/calibration: `dataset_7_keyframe_1`, `dataset_7_keyframe_2`.
- Frozen seen test: `dataset_7_keyframe_3`, `dataset_7_keyframe_4`.
- Unseen backbones: Fast-FoundationStereo and CREStereo, loaded only if the
  frozen seen promotion gate passes.

The present split is retained for direct comparison.  If this objective passes
the initial gate, a grouped five-fold evaluation over dataset IDs 1/2/3/6/7 is
required before a paper claim.  Keyframes from one dataset ID must never be
split across train, calibration, and test within a fold.

## Why raw-error stratification is necessary

The full 17-sequence audit shows that the oracle is highly non-uniform:

| deterministic region | oracle gain (px) | memory-better fraction |
|---|---:|---:|
| raw error <= 1 px | 0.02173 | 49.5% |
| raw error 1--3 px | 0.20358 | 61.6% |
| raw error > 3 px | 2.69568 | 56.9% |
| low motion < 1 px | 0.03654 | 49.9% |
| high motion >= 1 px | 0.07461 | 49.6% |
| non-boundary | 0.02932 | 49.8% |
| boundary | 0.05469 | 49.8% |

The useful signal is therefore not primarily a higher win frequency.  It is
the large positive magnitude available on relatively rare current-frame
stereo failures.  Training on natural pixels without raw-error strata
underweights precisely the events that determine net geometric utility.

## Calibration

The `utility_calibrated` policy thresholds the single `U_hat` score.  Candidate
thresholds are validation-score quantiles.  The selected point maximizes mean
realized validation utility subject to:

- coverage >= 0.2%;
- accepted harmful magnitude / accepted helpful magnitude <= 25%.

If no candidate is feasible, the artifact records
`calibration_constraint_feasible=false`; this is a failed calibration, not a
silent success.

## Promotion rule

The existing seen gate remains binding: positive paired EPE gain, positive
sequence-unit bootstrap lower bound, at least 80% positive sequences, false
updates below 2%, clean degradation below 1%, and more than 50% of the
raw-or-memory oracle gain recovered.  Unseen-backbone and cross-domain work is
conditional on passing this seen gate.

## Commands

Smoke (passed; output deleted before any full-seed evaluation):

```

The final loss-scale smoke reduced the total loss from 0.92776 to 0.77028
(16.97%) with finite outputs and gradients.  An earlier scale-probe smoke also
passed but was superseded before any full evaluation.
python scripts/run_utility_memory_selector.py --mode smoke \
  --objective utility_calibrated --device cuda:0 --workers 8 --batch-size 16 \
  --output results/utility_memory_selector_utility_calibrated/_smoke_v2
```

Full seed template:

```
python scripts/run_utility_memory_selector.py --mode train \
  --objective utility_calibrated --device cuda:0 --workers 32 --preload-workers 32 \
  --batch-size 64 --epochs 24 --seed SEED \
  --output results/utility_memory_selector_utility_risk/seed_SEED
```

The RAM preload keeps resized RGB/GT identical bit-for-bit to the validated
on-demand loader.  On `dataset_1_keyframe_2`, all returned tensors matched
exactly; the full training split occupies 3,668,068,800 bytes for 12,865
universal frames.  Stereo disparity arrays remain read-only mmaps.

## Rejected full-data policy-risk pilots

The first full-data epoch deliberately tested a 10x policy-risk multiplier.
It was rejected before further training: validation AUROC was 0.4754, the
best threshold selected 100% of pixels, and accepted harmful/helpful magnitude
was 91.7%.  The policy term dominated the BCE and learned the aggregate class
payoff instead of a useful ordering.  The compact rejected-pilot row is kept at
`results/utility_memory_selector_utility_risk/pilot_overweighted_policy_epoch1.csv`.

The `utility_calibrated` replacement restores the all-valid abstention label,
keeps conditional magnitude regression, uses unit signed-utility weight, and
removes the direct policy term.  This choice was made from training and
validation only, before any final-test or unseen evaluation.

## Cost-sensitive probability pilot

The full one-epoch `utility_calibrated` pilot restored validation probability
AUROC to 0.9546 but its conditional magnitude heads were nearly constant:
their utility score had Spearman correlation -0.007 with true utility and no
feasible 25% harm-cost operating point.  It is therefore not promoted.

The next, still frozen-data and architecture-identical, pilot is
`utility_weighted`.  It calibrates the probability head itself with bounded
utility-aware BCE weights: helpful pixels receive `1 + gain`, harmful pixels
receive `1 + 4 * harm`, and indifferent pixels receive weight 0.25.  The
probability score—not the failed magnitude difference—is selected on validation
under the same harm-cost/coverage constraint.  This is a single
cost-sensitive-risk check; no test, unseen backbone, or OOD dataset is loaded.

## Regional-utility ceiling and patch-9 pilot

The cost-sensitive pixel pilot still had no feasible 25% harm-cost point.
However, a calibration-only patch oracle establishes that the information is
spatially coherent: ranking pixels by their true 9x9 local mean utility yields
harm-cost 0.010 at 2% coverage and 0.026 at 20% coverage.  The current model's
9x9-smoothed probability does **not** recover this (`0.53` harm cost at 0.1%
coverage), so it is a prediction target rather than a post-processing win.

The next pilot uses the same 303,747-parameter CNN and the exact same 13
causal channels, with a 9x9 GT-only local-mean utility supervision target.
Per-pixel utility remains unchanged for all validation/test metrics.  This is
the smallest controlled test of whether the universal evidence can predict the
regional causal signal shown by the oracle; it adds neither state, extra memory,
RGB, backbone identity, nor a new flow/stereo module.

## Coverage-constrained selective-utility pilot

The first regional-label pilot retained high pixel AUROC but did not recover a
safe regional operating point.  The next objective keeps the same probability
head and CNN, but replaces the failed unnormalized action term with a
SelectiveNet-style normalized selective utility risk,

```
R = - sum(g * U_local) / sum(g),
L = L_weighted_BCE + 10 R + 32 max(0, 0.02 - mean(g))^2,
```

where `g` is the soft authorization and `U_local` is only the 9x9 GT-local
utility supervision.  Normalizing by selected mass prevents the former
all-pixel optimum.  The 2% coverage term prevents a trivial empty selection.
Both constants were fixed from training-loss scale and the patch-oracle
coverage study, before any final test or unseen evaluation.

### Result: rejected before final evaluation

The one-epoch, full-train-split pilot completed on 19 July 2026 (1,288
balanced backbone/sequence batches; validation remained the two frozen
`dataset_7_keyframe_{1,2}` groups).  It is a negative result, not a candidate
checkpoint:

| validation diagnostic | result |
| --- | ---: |
| per-pixel helpfulness AUROC | 0.4979 |
| per-pixel helpfulness AUPRC | 0.1476 |
| best sampled net utility | 0.00198 px |
| feasible policy at harm-cost <= 0.25 and coverage >= 0.2% | no |
| probability top 0.1% harm cost, kernel 1 | 0.325 |
| probability top 0.2% harm cost, kernel 9 | 0.326 |
| oracle-local-utility top 0.2% harm cost, kernel 9 | 0.0042 |

The validation probability collapsed near zero outside the training support
and did not rank true per-pixel utility (nor did 3x3--31x31 probability
smoothing repair it).  This is substantially below the regional oracle
ceiling, so further epochs, three seeds, final held-out sequences, unseen
backbones, and OOD datasets are not justified.  The output is retained as a
compact rejected pilot at
`results/utility_memory_selector_selective_patch9/pilot_epoch1/`; no test or
unseen data were loaded for this decision.
