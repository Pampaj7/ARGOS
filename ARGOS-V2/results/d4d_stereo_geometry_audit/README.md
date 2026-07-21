# D4D stereo-geometry consistency audit

This is a frozen data-contract check. `zivid_scale_*` rows are diagnostic
fixed copies only; they must never be used to rescale GT, train a model, or
report corrected D4D geometry. Candidate costs share exact
GT/candidate/right/census support per anchor. The SCARED-C control checks the
same code on established rectified stereo GT.

Across 156 D4D anchors, the finite-support aggregate selects Zivid × `0.25`
for both local RGB-L1 and ternary census; the fixed SCARED-C control selects
its stored GT scale `1.0` for both measures in 25/25 frames. This flags a
material D4D reference/image disparity-contract discrepancy, not a temporal
refinement finding. See `model_design/D4D_STEREO_GEOMETRY_CONSISTENCY_AUDIT.md`
for the restricted interpretation and the sparse-support caveat.
