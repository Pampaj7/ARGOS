# CODD-style Phase-1 cue contract

The phase uses raw current stereo `d_S`, causal BiDA-aligned prior fused state
`d_M`, frozen left/right ResNet-18 layer-1 feature maps, and no future frames.
All warped current-to-past quantities reuse `bidavideo.py`'s target-to-source
`grid + flow`, `align_corners=True` convention.

The 142 cue channels are: three L/R feature distances for each candidate and
their difference (9); two eight-channel scalar-disparity self-correlations
(16); current/aligned-previous eight-channel appearance self-correlations
(16); nine cross-disparity and nine cross-appearance correlations (18);
per-offset candidate support (6); 64 frozen current-context features; raw,
memory, signed/absolute disagreement (4); six motion/support maps; and current
RGB (3).  Costs are clipped to `[0,1]`, scalar disparity correlations to
`[0,1]`, cosine correlations to `[-1,1]`, flow to fixed `[-32,32]` cache-grid
pixels and RGB to `[-1,1]`.

These cues are evidence only: they never alter GT coverage, causal support,
the BiDA warp, or paired metric masks.
