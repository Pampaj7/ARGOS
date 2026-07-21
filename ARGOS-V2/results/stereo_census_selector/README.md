# ARGOS v2 stereo-census selector validation

This is a frozen, validation-only causal BiDA evidence audit, not a learned
model.  It compares current stereo reprojection census cost for the raw and
causally aligned t-1 candidates on SCARED-C `dataset_7_keyframe_1/2` and the
three seen backbones.  The full method, masks and predeclared sweep are in
`model_design/STEREO_CENSUS_UTILITY_AUDIT.md`.

At the strict common 9x9 support (9,535,997 pixels), raw EPE is `0.158415`,
the t-1 per-pixel oracle EPE is `0.135175`, and the available oracle gain is
`0.023239` cache pixels.  The best unconstrained census replacement (9x9,
zero margin) recovers only `4.02%` of that gain while causing `8.42%` false
updates and `3.90%` clean-pixel degradation.  The selected safe row (5x5,
margin `0.02`) has `1.05%` oracle recovery at `1.02%` false update, `0.46%`
clean degradation and `1.14%` coverage.

Verdict: **NO-GO** as a deterministic selector and therefore no CNN input
ablation, final-test, unseen-backbone or OOD evaluation was opened.  Compact
tables are under `validation/`; no dense prediction cache was written.
