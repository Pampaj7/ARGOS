# ARGOS v2 — frozen left-right consistency (LRC) audit plan

## Motivation and scope

The large-scale causal BiDA signal audit establishes a positive t-1 oracle,
while the current utility selector cannot reliably tell which of the raw and
temporally aligned candidates is geometrically better.  Current-frame
stereo-left/right consistency is a distinct, externally observable confidence
cue: it assesses whether a candidate agrees with an independently computed
right-reference disparity.  This is **not** a new stereo network, internal
cost volume, or temporal refiner.

Classical LRC is a standard stereo confidence cue, and recent universal
stereo-confidence work continues to combine it with other external evidence:
see [Semi-Stereo, CVPRW 2024](https://openaccess.thecvf.com/content/CVPR2024W/3DMV/html/Yue_Semi-Stereo_A_Universal_Stereo_Matching_Framework_for_Imperfect_Data_via_CVPRW_2024_paper.html).
The reference additionally supports the need to evaluate local spatial
confidence rather than raw pixel scores alone: [Poggi & Mattoccia, CVPR
2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Poggi_Learning_to_Predict_CVPR_2017_paper.html).

## Exact black-box geometry

ARGOS cached disparities are positive left disparities on the 144x180 grid:

```text
x_R = x_L - d_L(x_L).
```

Most frozen backbones expect this positive ordering.  Therefore a
right-reference prediction is produced, for one unchanged frame, by calling
the same frozen backbone with:

```text
(left', right') = (horizontal_flip(right), horizontal_flip(left)).
```

The resulting positive map is horizontally unflipped to produce `d_R` on the
original right grid.  For an original physical match, the flipped disparity
magnitude remains positive.  The canonical LRC residual on the left grid is:

```text
r_LR(x_L) = |d_L(x_L) - d_R(x_L - d_L(x_L))|.
```

`external_components/stereo_lr_consistency.py` uses bilinear `grid_sample`,
`align_corners=True`, zero padding, explicit in-bounds support, and an
explicit sampled-right-valid mask.  Neither an out-of-bounds sample nor an
invalid reverse prediction is assigned a residual of zero.

## Frozen audit ladder

Before adding LRC to any selector, run exactly this ladder:

1. **Reverse-inference smoke**: one SCARED-C sequence, ten frames, each seen
   backbone.  Check finite positive reverse maps, reflection/unreflection
   orientation, unit equality at cache width 180, and LRC support.
2. **Confidence audit on only calibration sequences** `dataset_7_keyframe_1`
   and `_2`: measure raw-error correlation/AUROC for raw LRC, aligned-memory
   LRC, their signed difference, and the existing temporal/photo cues.
3. Only if candidate-level ranking or calibrated utility is materially better
   than the existing temporal/photo evidence, run one fixed-input selector
   ablation.  The split, frozen BiDA path, loss, and network must otherwise be
   unchanged.
4. Freeze any operating point on calibration sequences; evaluate final seen,
   then Fast-FoundationStereo and CREStereo.  SERV-CT/D4D/StereoMIS remain
   untouched until those gates pass.

No OOD data, unseen backbone, final-test sequence, or right-reference feature
may be used for method/threshold/checkpoint selection.

## Conditional follow-up: raw-consistency safety veto

This is deliberately **not** a second selector and is not enabled by the
current audit.  It is pre-registered only because preliminary calibration
shards show a distinction that the direct candidate score does not capture:
raw LRC can identify raw stereo failures, while `LRC(raw)-LRC(memory)` does
not consistently rank the temporally warped candidate.

If, and only if, the completed three-backbone calibration audit establishes
that raw-LRC Bad1 AUROC is above chance on every backbone, the smallest next
test is a frozen logical veto on the existing utility selector:

```text
authorize = frozen_utility_selector_authorize
            AND raw_LRC is above its frame-relative LRC quantile
            AND raw_LRC_support.
```

The interpretation is safety-only: a low LRC raw prediction is externally
stereo-consistent and is preserved rather than replaced by temporal memory.
LRC never opens a temporal update and never chooses memory by itself.  Raw LRC
can have a backbone-dependent absolute scale even when its within-frame error
ranking is valid.  Therefore the primary policy is a fixed *frame-relative*
quantile, computed from valid raw-LRC pixels in the current frame, not an
absolute LRC threshold and not a backbone-conditioned normalization.  A small
predeclared grid of quantiles is selected on `dataset_7_keyframe_1/2` pooled
across the three seen backbones; checkpoint, test sequences, unseen backbones
and all OOD data remain unavailable during selection.  The required comparison is against the
same frozen selector at matched/recorded coverage, with paired masks and the
same cache-grid geometry metrics.  Promotion requires retained geometry gain
and a reduction in clean degradation/false updates without near-zero coverage.

The only candidate frame quantiles are `.50`, `.75`, `.90` and `.95`; a policy
must retain at least 70% of the frozen selector gain while lowering both clean
degradation and false updates.  This keeps the safety-veto search small and
prevents a post-hoc per-backbone threshold.

### Implementation smoke

The parameter-free implementation is
`models/lrc_safety_veto.py`.  Its synthetic contracts are covered by
`tests/test_lrc_safety_veto.py`.  A 32-pair real S2M2-S calibration smoke with
the frozen seed-0 selector completed without gradients: base authorization was
`0.6366%`, the `.75` frame-relative veto retained `0.3056%`, it opened no new
update, and every rejected output was bit-exactly raw.  This verifies only the
one-way tensor contract; it is not a geometry result or an operating point.

If raw-LRC is not stable on all three backbones, or if the veto merely removes
all useful updates, this branch is closed without training an LRC model.

## Scope boundary relative to video-stereo literature

Recent video-stereo systems can improve both geometry and temporal stability
by retaining matching cost volumes, learned monocular/video features, or
recurrent refinement states.  Examples include
[BiDAStereo](https://arxiv.org/abs/2403.10755),
[Temporally Consistent Stereo Matching](https://eccv.ecva.net/virtual/2024/poster/256),
and [Stereo Any Video](https://openaccess.thecvf.com/content/ICCV2025/html/Jing_Stereo_Any_Video_Temporally_Consistent_Stereo_Matching_ICCV_2025_paper.html).
Those mechanisms are useful scientific references but are not directly
transferable to ARGOS: their state contains learned stereo features and/or cost
volumes, whereas ARGOS must accept frozen heterogeneous black-box stereo
predictions.  The LRC audit therefore evaluates only an explicitly observable
two-view consistency cue and keeps all warping in the validated causal BiDA
module.

## Cost and caching policy

Right-reference inference approximately doubles frozen stereo inference.  A
small isolated reverse-cache namespace is permitted only if the smoke proves
the map convention works and the full audit cannot otherwise run practically.
It must use frame IDs, checkpoint hashes, source/cache resolution, flip-swap
metadata, atomic completion, and never overwrite `cache_scaredc_backbones`.
No reverse cache is created by the deterministic component itself.

## Current status

The deterministic component and synthetic tests are complete.  A temporary
ten-frame reverse-inference smoke on `S2M2-S/dataset_3_keyframe_1` completed
and then deleted its reverse cache and output directory as required.  The nine
causal current frames with valid GT/LRC support contained 162,810 pixels.  Raw
EPE was `.1251` cache pixels; LRC had Pearson correlation `.3266` with raw
absolute error and AUROC `.5982` for identifying raw Bad1 pixels.  Mean LRC
was `.1221` on raw-clean pixels and `.2540` on raw-Bad1 pixels.

This is sufficient only to justify the preregistered calibration audit.  It is
not a selector, final-test, backbone-generalization, temporal-memory, or OOD
result.  No threshold has been selected and no dense reverse cache remains
from the smoke.

## Completed calibration result — branch closed (NO-GO)

The complete frozen LRC audit then ran on *only* the preregistered calibration
split (`dataset_7_keyframe_1/2`) for S2M2-S, RAFT-Stereo and StereoAnywhere.
It included 9,661,747 pixels on the strict common GT/raw/BiDA/LRC support.
For raw stereo confidence, LRC was useful: raw-LRC versus raw-Bad1 had AUROC
`.6333` overall (per backbone `.6127` S2M2-S, `.6517` RAFT-Stereo, `.6355`
StereoAnywhere).  It was not a direct temporal-candidate signal:
`LRC(raw)-LRC(memory)` had memory-better AUROC `.4077` overall.  The latter
is therefore a direct-selection NO-GO.

The parameter-free one-way safety veto was then tested on all three frozen
utility-selector seeds.  It used no final-seen, unseen-backbone or OOD data,
and the same fixed frame-relative candidate quantiles `.50/.75/.90/.95` for
each seed.  At the most selective quantile `.95`, the calibration results
were:

| frozen selector seed | gain retained | authorized updates retained | harmful acceptance (base → veto) |
| --- | ---: | ---: | ---: |
| 0 | 72.8% | 37.5% | 33.5% → 19.4% |
| 1 | 53.1% | 18.0% | 32.4% → 19.1% |
| 2 | 19.7% | 6.8% | 24.1% → 15.8% |

The absolute raw-clean degradation rate did fall in all three seeds (for
example, seed 0 `.0459% → .0081%`), but this happened predominantly by
closing updates.  At less strict quantiles, seed 1 and seed 2 sometimes
*increased* conditional harmful acceptance while retaining more gain.  No
single preregistered quantile simultaneously retained at least 70% of gain,
improved safety, and retained a meaningful update fraction in all three
seeds.  Accordingly this veto is **NO-GO** and no final-seen, unseen-backbone
or OOD evaluation was opened.

This is a useful negative result: current-frame LRC is a transferable raw
error cue, but its frame-relative ranking is not stable enough to provide a
seed-robust safety authorization for the causal memory selector.  The source
runner is `scripts/run_lrc_safety_veto.py`; compact calibration artifacts are
under `results/lrc_safety_veto/seed_{0,1,2}/`.
