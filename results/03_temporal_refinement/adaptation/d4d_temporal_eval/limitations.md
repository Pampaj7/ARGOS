# Limitations

- **No dense temporal GT**: all metrics are prediction-space / RAFT motion-compensated
  diagnostics. MC-inconsistency mixes residual disparity jitter with imperfectly-compensated
  real non-rigid motion — it is dominated by scene motion, so small corrections barely register.
  This is itself the finding, but it caps sensitivity.
- **Flow validity**: RAFT fwd-bwd occlusion masking used; high-motion regions (mc_highmotion
  ≈ 5px vs mc_lowmotion ≈ 0.2px) are less reliable and excluded via occlusion; residual flow
  error remains. Non-rigid deformation is not modelled.
- **Scope**: 6 clips (2/specimen, ≤120 frames each) — representative, not exhaustive.
- **Single adapted seed per config** (most-anchors seed). calib-2s has only 2 train anchors.
- Anchor MAE reused from the few-shot frozen-test (not recomputed here).
- No motion-lag pathology detected precisely because corrections are tiny; the metric exists
  (mc stratified by flow magnitude) but is uninformative at this correction scale.
