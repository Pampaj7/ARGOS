# Inference assumptions — D4D zero-shot

- **Rectified images**: use the D4D processed rectification (recomputed per session with the
  session calibration for context frames; the anchor frame matches the GT pipeline). Never
  re-rectify the already-rectified GT anchor differently.
- **S2M2 input resolution**: width 512, output disparity in native (894×714) coords, positive px.
- **Refiner grid**: target_scale=0.25 → 224×179 (894×714), disparity kept in native px ÷64
  (NO rescale to SCARED — that would be domain tuning).
- **Valid mask** (feature + metric): Zivid GT valid ∧ finite raw ∧ gt>0, downsampled valid-aware
  (same semantics as SCARED training targets). GT exists only at the anchor frame.
- **Causal context**: 3 preceding same-session stereo frames by timestamp; no future frames; no
  clip/session crossing; clamp-repeat only at true sequence start (recorded in context_manifest).
- **No D4D tuning**: selected checkpoints, thresholds, scales unchanged.
- **Skips**: anchors without a right stereo pair (raw S2M2 impossible) are skipped with reason.
