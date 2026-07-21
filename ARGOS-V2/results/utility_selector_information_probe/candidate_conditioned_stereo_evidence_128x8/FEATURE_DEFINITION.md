# Candidate-conditioned stereo evidence

For the frozen raw and BiDA-aligned t-1 disparity candidates, the current
rectified right image is sampled at `x_right = x_left - (d + o)` for
`o = [-4,-2,-1,0,+1,+2,+4]` cache-grid pixels.  A deterministic 5x5 ternary
census mismatch against the current left image forms each local cost curve.

The `full` treatment appends 37 maps to the validated 13 universal inputs:
six direct cost/support/boundary maps, twelve per-candidate local statistics,
five raw-versus-memory statistic differences, and both seven-point curves.
All maps use fixed bounded normalisation; `normalization_statistics.json` in
each seed is descriptive, uses train crops only and is not fitted by the model.

These maps are inputs only.  They never alter the existing BiDA validity,
supervision, calibration or paired evaluation masks.
