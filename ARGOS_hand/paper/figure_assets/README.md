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

Both disparity rows share ONE colour scale, symmetric at +/-1.49 px. They
must, because the honest fact is that the two rows look nearly identical:
mean |fused - raw| over the window is 0.126 px. Normalising the rows
separately would manufacture a visible difference that is not there.

ONE fitted plane is removed from all ten disparity panels. Absolute disparity
here is a featureless gradient: the liver surface is smooth at 144x180, a 7 px
global ramp carrying about 1 px of anatomy. Other frames and the other two D2
sequences were checked and look the same, so this is the scene and not a bad
choice of window. The plane is fitted once on the first raw frame shown and
subtracted from every panel including the fused ones, which keeps the rows
comparable and leaves mean |fused - raw| unchanged.

The shared scale matters: the five disparity maps must be comparable, because
the point they make is that the RGB barely changes between consecutive frames
while the disparity visibly wanders. That is the temporal inconsistency the
paper is about, shown rather than asserted.

Regenerate: `scripts/run_figure_window.sh` re-drives the window on the GPU and
`scripts/export_figure_panels.py` renders the panels; `--verify` re-renders from
the stored window and diffs against the PNGs instead of overwriting them. The
renderer used to exist only inside a commit message, which is why the fused row
went on showing the 142-channel ablation after the proposed model changed. The
previous window is kept as `fused_window.142ch.npz`.

NOTE: `Temporal_consistency_taxonomy.pdf` is a hand-composed design file that
embeds these panels. Regenerating the PNGs does not update it -- the composed
figure has to be re-exported by hand.
