# Streaming S2M2-S Rectified Temporal-GT Evaluation

This run evaluates S2M2-S frame-by-frame on rectified SCARED temporal-GT and discards predictions by default.
No RAFT, StereoAnyVideo, ConvGRU, temporal smoothing, oracle selection, or optical flow is run by this script.

- Input root: `/dtu/p1/leopam/ARGOS/dataset/SCARED/curated/temporal_gt_rectified`
- Audit frame CSV: `/dtu/p1/leopam/ARGOS/dataset/SCARED/curated/audit/temporal_gt_rectified_integrity/frame_integrity.csv`
- Sequences: `1`
- Frames: `10`
- Evaluated frames: `10`
- Skipped frames: `0`
- Resize width: `512`
- Skip suspicious: `True`
- Minimum valid ratio: `0.05`
- Saved predictions: `False`
- Temporal metrics: `True`
- Sequence group: `all`
- Disparity MAE weighted: `1.1796637029088366`
- Disparity RMSE weighted: `1.514762986052594`
- Bad-1px weighted pct: `49.193677056602716`
- Bad-2px weighted pct: `16.879365436930517`
- Bad-3px weighted pct: `4.7599621257098015`
- Depth MAE weighted: `0.9322597970159586`
- Median runtime per evaluated frame ms: `64.02110343333334`
- Peak VRAM MB: `395.21337890625`
- Estimated cache storage saved GiB: `0.048828125`

Outputs:

- `frame_metrics.csv`: per-frame include/skip status and metrics.
- `sequence_metrics.csv`: per-sequence aggregate metrics.
- `aggregate_summary.json`: machine-readable aggregate summary.
- `diagnostics/<sequence_id>/`: compact contact sheets for selected evaluated frames.

Temporal metrics are same-pixel, non-motion-compensated frame-to-previous-included-frame differences.
- Temporal pairs: `9`
- Temporal pair coverage: `1.0`
- Temporal disparity diff mean: `0.39078475700484383`
- GT temporal disparity diff mean: `0.9186154339048598`
- Temporal motion mismatch mean: `0.6210023164749146`
