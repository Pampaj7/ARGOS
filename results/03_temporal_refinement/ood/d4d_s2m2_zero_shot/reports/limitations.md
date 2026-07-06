# Limitations — D4D zero-shot baseline

- **Sparse keyframe GT only** (Zivid at ~2 instants/clip). Geometric accuracy at anchors;
  NOT dense temporal-consistency. Temporal diagnostics (Phase 8) are prediction-space only.
- **MPC/CPV blocked** at D4D's odd grid height (179). Even-dim padding or a model fix would
  enable them; they are secondary branches, so not required for the primary result.
- **19 anchors** have no valid raw∩GT overlap after downsample → excluded from stats
  (kept in anchor_metrics with valid_px=0). 10 anchors skipped for missing right stereo pair.
- **Cross-camera / per-specimen calibration**: handled per session; D4D raw error regime
  differs from SCARED (lower), which is precisely the domain shift under study.
- **No SNR-stratified table** in this pass (shards do not carry SNR); available via the
  benchmark snr_mask if needed. Boundary/interior and raw-error bins ARE reported.
- No D4D tuning, no training — by design.
