# Figure assets

Real data, exported for the overview figure. Nothing here is illustrative:
every pixel comes from the frozen evaluation boundary.

- Source: `dataset_2_keyframe_4`, frames 1100-1104, five CONSECUTIVE frames.
- `frame_0..4.png`: left RGB, as consumed by the frozen stereo network.
- `disp_0..4.png`: the RAW disparity that network produced.
- `fused_0..4.png`: the disparity AFTER the module, same five frames.
- `fused_window.npz`: the fused array for frames 1088-1104. Frames before
  1100 are warm-up: the recurrence is bounded at H=4, so the state must be in
  steady state before the window we show.

Both disparity rows share ONE colour scale (11.9-19.2 px, 2nd-98th percentile
over raw and fused together). They must, because the honest fact is that the
two rows look nearly identical: mean |fused - raw| over the window is
0.108 px on disparities of 12-19 px. Normalising the rows separately would
manufacture a visible difference that is not there.

The shared scale matters: the five disparity maps must be comparable, because
the point they make is that the RGB barely changes between consecutive frames
while the disparity visibly wanders. That is the temporal inconsistency the
paper is about, shown rather than asserted.

Regenerate: see the snippet in the commit that added this directory.
