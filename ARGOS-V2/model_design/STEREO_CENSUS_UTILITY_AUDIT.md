# ARGOS v2 — causal stereo-census utility audit

## Question

The large-scale BiDA audit proves that causally aligned t-1 disparity has
complementary oracle information, while the existing universal selectors do
not reliably identify when it is safer than current raw stereo.  RGB L1 and
local ZNCC were already audited and were either unsafe or nearly uninformative.
This audit asks the smaller, preregistered question:

> Does an ordinal current-stereo matching cost distinguish raw from causally
> aligned memory better than L1/ZNCC, without using a stereo backbone's cost
> volume or identity?

The method is a deterministic cue only.  It neither updates a disparity,
trains a network, alters SEA-RAFT/BiDA, nor accesses future frames.

## Why ternary census is a distinct cue

For a positive left disparity candidate `d`, current right RGB is sampled at
`x_R = x_L - d(x_L)` using the already validated image/disparity convention:
bilinear `grid_sample`, `align_corners=True`, zero padding and explicit
in-bounds support.  On luminance patches around each left and reconstructed
right pixel, the cost compares ternary neighbour-to-centre relations:

```text
c(v, c) = +1 if v-c > 0.02; -1 if v-c < -0.02; 0 otherwise
CensusCost = mean(|c_left - c_right| / 2).
```

This creates an ordinal local image-match cue in `[0,1]`; it is not a learned
feature or a hidden stereo-network confidence.  A full candidate right-support
window is required, so zero padding cannot appear as a low-cost match.  Census
matching is a classical illumination-robust stereo evidence family; recent
domain-generalized stereo work explicitly uses a multi-scale census cost as a
backbone-free matching representation ([Cheng et al., 2021](https://arxiv.org/abs/2108.00335)).
The general need for viewpoint-consistent matching under domain shift is also
supported by [Zhang et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Revisiting_Domain_Generalized_Stereo_Matching_Networks_From_a_Feature_Consistency_CVPR_2022_paper.html).

## Frozen contract

Implementation: `external_components/stereo_photometric.py::ternary_census_cost`.

Inputs are only current rectified left/right RGB and one existing candidate
disparity at the cache grid.  For raw and BiDA-aligned memory candidates, the
paired comparison mask is:

```text
GT coverage > 0.50
& raw prediction valid
& sampled t-1 memory valid
& BiDA in-bounds support
& raw current-right support
& memory current-right support
& raw full census-window support
& memory full census-window support.
```

No RGB encoder, DINO feature, backbone identifier, cost volume,
backbone-specific confidence, future frame, memory age beyond causal t-1 or
post-processing is permitted.

## Predeclared audit ladder

1. Synthetic unit tests: sign, true-versus-wrong disparity and patch-edge
   support; smoke one SCARED-C sequence/backbone/ten pairs; then delete smoke
   output.
2. Validation-only sweep on SCARED-C `dataset_7_keyframe_1/2` and three seen
   backbones.  Kernels `{5,7,9}` and lower-cost memory margins
   `{0,.01,.02,.05,.10}` are fixed before results.
3. Select a deterministic census replacement only if positive gain,
   false-update <=2%, clean degradation <=1%, coverage >=.2%, and a
   meaningful improvement over the earlier L1/ZNCC ceiling appear.
4. Only if stage 3 passes, use the best **frozen** census maps as an input
   ablation to the pre-existing causal selector.  The BiDA flow, selector
   target, split, CNN capacity and raw-by-default rule must otherwise remain
   fixed.  No final, unseen or OOD data may be opened before that gate.

If stage 2 fails, census is documented as another deterministic-confidence
NO-GO and no selector is trained with it.

## Completed validation-only result — NO-GO

The full stage-2 sweep completed on all 3,819 causal pairs from SCARED-C
`dataset_7_keyframe_1/2` and all three seen backbones (9,535,997 pixels on the
strict maximum-9x9 census common support).  It was a frozen deterministic
audit: no final-test, unseen-backbone, OOD or training sample was opened.

The best unconstrained census row (`9x9`, margin `0`) gains `.000934` cache
pixels, i.e. only `4.02%` of the `.023239` raw-or-memory oracle gain, while
incurring `8.42%` false updates and `3.90%` clean degradation.  The selected
safe row (`5x5`, margin `.02`) has `.000244` gain, `1.05%` oracle recovery,
`1.02%` false updates, `.46%` clean degradation and `1.14%` coverage.  It
does meet the deliberately minimal deterministic safety constraint, but it is
not materially better than the existing L1/ZNCC safe ceiling and leaves nearly
all useful oracle evidence untouched.

Accordingly ternary census is **NO-GO** as a direct selector and as a new CNN
input ablation.  The result rules out the specific hypothesis that local
ordinal current-stereo matching resolves the causal raw-versus-memory choice.
The compact artifacts are under
`results/stereo_census_selector/validation/`; no final/unseen/OOD evaluation
was opened.
