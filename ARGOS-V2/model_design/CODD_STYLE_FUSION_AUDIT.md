# ARGOS v2 CODD-style fusion diagnostic audit

## Faithful CODD mechanics retained

This Phase-1 baseline follows CODD (WACV 2023), Sections 3.3.1, 3.3.2,
3.4 and Appendix A.3:

- two bounded per-pixel weights and the exact equation
  `d_F=(1-w_reset*w_fusion)d_S+(w_reset*w_fusion)d_M`;
- reset labels based on `e_M-e_S` with a no-supervision dead-band;
- fusion labels with the same comparison and tie regularisation to 0.5;
- Huber supervision of the fused disparity;
- reported CODD weights `alpha_disp=alpha_reset=alpha_fusion=1` and
  `alpha_reg=0.2`;
- reset regression at the cache grid and fusion regression at reduced
  resolution followed by bilinear upsampling;
- four-frame causal unrolling.  The first pair consumes raw `t-1`; every
  following pair consumes the preceding fused output, matching CODD's
  sequence-length motivation in Table 4.

CODD's reported `tau_reset=5` and `tau_fusion=1` are native disparity-pixel
thresholds.  SCARED-C images are 1280 pixels wide and ARGOS cache disparity is
at width 180, so Phase 1 uses `0.703125` and `0.140625` cache pixels,
respectively.  This preserves their physical disparity displacement rather
than incorrectly treating five 1280-grid pixels as five cache-grid pixels.

## Fixed ARGOS replacement for CODD motion

CODD uses learned RAFT3D per-pixel SE3 motion plus differentiable forward
rendering.  Phase 1 deliberately does **not** add this: its `d_M` is the
already validated frozen SEA-RAFT/BiDA causal pull warp of the preceding fused
state.  This makes the experiment a fusion/evidence diagnostic, not an
end-to-end CODD reproduction or a Phase-2 motion comparison.

## Input cues

CODD obtains learned left/right features from the particular frozen stereo
network.  ARGOS's disparity caches do not retain those internal maps, so no
honest backbone-internal feature reproduction is possible.  The closest common
learned feature source is a local, frozen ImageNet ResNet-18 layer-1 map.  It
is loaded from the pre-existing local checkpoint and never trained.  The
fusion head receives:

1. L1 left/right learned-feature matching costs at `d_S,d_S±1,d_M,d_M±1`;
2. W=3, dilation=2 local disparity and learned-appearance self-correlations;
3. W=3, dilation=2 cross-frame disparity and appearance correlations after
   the canonical BiDA feature warp;
4. frozen feature context and current RGB;
5. flow components/magnitude, FB confidence, warp support and aligned validity.

This is richer than the failed deterministic 37-channel census probe, but it
is not identical to CODD's stereo-network feature maps or its semantic/motion
network context.  That difference is structural and is reported rather than
hidden.

## Expected interpretation

Phase 1 passing the >50% strict raw-vs-raw-memory oracle-recovery gate would
show that dual soft fusion plus richer learned correspondence cues are worth
separating from motion.  Failure near the 27% hard-selector result means that
the missing information is not repaired by this common frozen feature space;
only then is a separately approved Phase-2 learned SE3/visibility study
scientifically motivated.
