# D4D keyframe GT — blockers & caveats (scientific honesty)

## Resolved (not blockers)
- **Zivid→stereo-left extrinsic**: RESOLVED via the MiRe45-bridged Polaris chain, validated
  by anatomical reprojection alignment (`transform_chain_hypotheses.md`). Not assumed.
- **Depth units/convention**: metres, optical-axis Z, NaN=invalid; backprojection with the
  Zivid colour K reproduces the `.ply` exactly.
- **TF direction**: child→parent (ROS), confirmed by the working chain (wrong direction put
  points behind the camera).
- **Rectification**: reproduced with OpenCV from `left/right(_rect).yaml`; fx=798.32 px,
  baseline=4.235 mm from the rectified P matrices.
- **Timestamp sync**: pose interpolation < 10 ms, nearest stereo frame 2–9 ms from scan.

## Remaining caveats
1. **Marker-identity assumption** (minor): the bridge treats `polaris_spectra_MiRe45` and
   `MiRe45` as the same physical marker. Empirically corroborated by correct reprojection;
   not independently documented in the dataset. A wrong identity would misalign anatomy — it
   does not.
2. **Cross-camera photometric metric is weak**: Zivid and endoscope are different cameras
   with different colour response/illumination, so absolute RGB MAE (~55–65) is an upper
   bound, not a tight alignment score. Structural/instrument alignment is the real evidence.
3. **Sparse GT only** — HARD scientific limit: Zivid provides geometry at ~2 instants per
   clip. Valid coverage per anchor is ~30 % of the endoscope frame (Zivid FOV is wider; only
   textured, in-FOV, front-facing, z-buffer-winning surface survives). This is genuine sparse
   keyframe GT, NOT dense per-frame GT.
4. **Non-rigid scene** — do NOT propagate a Zivid scan through the clip as GT. Tissue and
   instruments move between the two scans. Any temporal use is limited to a small validated
   neighbourhood around each anchor (see optional local-pseudo-GT analysis, not yet run).
5. **Right-image validation pending**: epipolar/vertical-disparity check between rectified
   left/right (feature matching) is implemented at the anchor level but a quantitative
   session-wide epipolar error table is a next step.
6. **specimen_2 partial; specimen_3/4/5 not extracted** — only specimen_1 converted.

## Not done (documented, not faked)
- No dense per-frame disparity GT produced.
- No rigid propagation presented as GT.
- Local pseudo-GT (±1–3 frames) analysis prepared but not run.
