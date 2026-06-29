# Streaming S2M2-S Rectified Temporal-GT Evaluation

This run evaluates S2M2-S frame-by-frame on rectified SCARED temporal-GT and discards predictions by default.
No RAFT, StereoAnyVideo, ConvGRU, temporal smoothing, oracle selection, or optical flow is run by this script.

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
- Disparity MAE weighted: `6.829471384440535`
- Disparity RMSE weighted: `9.256683951829169`
- Bad-1px weighted pct: `72.96114795890018`
- Bad-2px weighted pct: `54.18712785214271`
- Bad-3px weighted pct: `42.715268635392626`
- Depth MAE weighted: `3.5437596988229942`
- Median runtime per evaluated frame ms: `56.97797401808202`
- Peak VRAM MB: `395.21337890625`
- Estimated cache storage saved GiB: `100.6884765625`

Outputs:

- `frame_metrics.csv`: per-frame include/skip status and metrics.
- `sequence_metrics.csv`: per-sequence aggregate metrics.
- `aggregate_summary.json`: machine-readable aggregate summary.
- `diagnostics/<sequence_id>/`: compact contact sheets for selected evaluated frames.
