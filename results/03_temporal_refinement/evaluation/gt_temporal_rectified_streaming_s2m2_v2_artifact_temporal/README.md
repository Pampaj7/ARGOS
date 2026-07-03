# Streaming S2M2-S Rectified Temporal-GT Evaluation

This run evaluates S2M2-S frame-by-frame on rectified SCARED temporal-GT and discards predictions by default.
No RAFT, StereoAnyVideo, ConvGRU, temporal smoothing, oracle selection, or optical flow is run by this script.
This is a no-flow streaming evaluation: motion-compensated temporal metrics, ghosting, and occlusion metrics that require optical flow are not computed.

- Input root: `/dtu/p1/leopam/ARGOS/dataset/SCARED/curated/temporal_gt_rectified`
- Audit frame CSV: `/dtu/p1/leopam/ARGOS/dataset/SCARED/curated/audit/temporal_gt_rectified_integrity/frame_integrity.csv`
- Sequences: `27`
- Frames: `22950`
- Evaluated frames: `20621`
- Skipped frames: `2329`
- Resize width: `512`
- Skip suspicious: `True`
- Minimum valid ratio: `0.05`
- Saved predictions: `False`
- Temporal metrics: `True`
- Artifact metrics: `True`
- Sequence group: `all`
- Disparity MAE weighted: `6.829471384440535`
- Disparity RMSE weighted: `9.256683951829169`
- Bad-1px weighted pct: `72.96114795890018`
- Bad-2px weighted pct: `54.18712785214271`
- Bad-3px weighted pct: `42.715268635392626`
- Depth MAE weighted: `3.5437596988229942`
- Median runtime per evaluated frame ms: `62.450126046314836`
- Peak VRAM MB: `395.21337890625`
- Estimated cache storage saved GiB: `100.6884765625`

Outputs:

- `frame_metrics.csv`: per-frame include/skip status and metrics.
- `sequence_metrics.csv`: per-sequence aggregate metrics.
- `aggregate_summary.json`: machine-readable aggregate summary.
- `diagnostics/<sequence_id>/`: compact contact sheets for selected evaluated frames.

Temporal metrics are same-pixel, non-motion-compensated frame-to-previous-included-frame differences.
- Temporal pairs: `20594`
- Temporal pair coverage: `1.0`
- Temporal disparity diff mean: `1.6483188765956494`
- GT temporal disparity diff mean: `1.3769916053482976`
- Temporal motion mismatch mean: `2.0261270027574034`

Artifact metrics reuse the old v4 frame-only definitions where possible.
Same-pixel temporal/artifact proxies are reported separately from motion-compensated metrics.
Prediction arrays are not cached unless `--save-predictions true` is passed.
- Edge sharpness ratio raw mean: `1.0`
- Boundary disparity MAE mean px: `8.98199239037622`
- RGB/disparity edge correlation mean: `0.019890423106689495`
- Same-pixel temporal MAE mean px: `1.6483188765956494`
- Motion-compensated temporal metrics: `not_computed_no_flow`
