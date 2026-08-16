# Figure assets

Real data, exported for the overview figure. Nothing here is illustrative:
every pixel comes from the frozen evaluation boundary.

- Source: `dataset_2_keyframe_4`, frames 1100-1104, five CONSECUTIVE frames.
- `frame_0..4.png`: left RGB, as consumed by the frozen stereo network.
- `disp_0..4.png`: the raw disparity that network produced, viridis, on one
  shared scale (11.9-19.3 px, 2nd-98th percentile over the five frames).

The shared scale matters: the five disparity maps must be comparable, because
the point they make is that the RGB barely changes between consecutive frames
while the disparity visibly wanders. That is the temporal inconsistency the
paper is about, shown rather than asserted.

Regenerate: see the snippet in the commit that added this directory.
