# S2M2 Streaming Result Analysis

Poor aggregate metrics are outlier-heavy but not isolated: removing the top three disparity-MAE sequences lowers weighted disparity MAE from 6.829 px to 5.388 px, while median sequence bad-3px remains 38.13%.

Outputs:

- `best_sequences.csv` and `worst_sequences.csv`: sequence rankings by disparity MAE, bad-3px, and depth MAE.
- `worst_frames.csv`: highest-error included frames ranked by disparity MAE and bad-3px.
- `core_stress_exclude_split.csv`: quantile-based proposed sequence split.
- `analysis_summary.json`: thresholds, correlations, contribution checks, and group counts.

Split rule:

- `exclude_or_diagnostic`: any error metric above its Tukey upper fence, or valid-pixel mean at/below q10.
- `stress_eval`: any error metric at/above q75, valid-pixel mean at/below q25, or skipped ratio at/above q75.
- `core_eval`: everything else.

Plots:

- `sequence_disp_mae_sorted.png`
- `sequence_bad3_sorted.png`
- `error_vs_valid_ratio.png`
- `frame_error_histogram.png`
