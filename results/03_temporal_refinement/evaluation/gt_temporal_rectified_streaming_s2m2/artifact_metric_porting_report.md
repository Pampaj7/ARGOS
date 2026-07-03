# Artifact Metric Porting Report

Target evaluator: `scripts/temporal_refinement/eval_scripts/evaluate_s2m2_streaming_temporal_gt_rectified.py`

The new evaluator is streaming, S2M2-S-only, and no-flow by default. It does not run RAFT, StereoAnyVideo, ConvGRU, oracle selection, optical flow, or smoothing. Prediction arrays are not cached unless `--save-predictions true` is passed.

| old column | category | old function / location | required inputs | previous frame | flow | new column |
|---|---|---|---|---:|---:|---|
| `edge_sharpness_ratio_raw` | directly portable to streaming S2M2 | `artifact_metrics.frame_artifact_metrics`; called by `benchmark_scared_s2m2_temporal_baselines_v4_artifact_metrics.py:153` | current prediction, raw S2M2 prediction, GT disparity, valid mask, RGB | no | no | `edge_sharpness_ratio_raw` |
| `edge_sharpness_ratio_raw_edges` | directly portable to streaming S2M2 | same as above | current prediction, raw S2M2 prediction, GT disparity, valid mask, RGB | no | no | `edge_sharpness_ratio_raw_edges` |
| `boundary_disp_mae_px` | directly portable to streaming S2M2 | same as above | current prediction, GT disparity, valid mask | no | no | `boundary_disp_mae_px` |
| `boundary_disp_mae_px_p80` | directly portable to streaming S2M2 | same as above | current prediction, GT disparity, valid mask | no | no | `boundary_disp_mae_px_p80` |
| `rgb_disp_edge_corr` | directly portable to streaming S2M2 | same as above | current prediction, RGB, valid mask | no | no | `rgb_disp_edge_corr` |
| `rgb_disp_edge_corr_rgb_edges` | directly portable to streaming S2M2 | same as above | current prediction, RGB, valid mask | no | no | `rgb_disp_edge_corr_rgb_edges` |
| `raw_temporal_disp_diff_px` | directly portable to streaming S2M2 | `benchmark_scared_s2m2_temporal_baselines.py:584` `temporal_pair_metrics` | previous prediction, current prediction, previous/current masks | yes | no | `raw_temporal_disp_diff_px` |
| `lag_rate` | portable as same-pixel approximation | `artifact_metrics.pair_artifact_metrics` lag block | current prediction, previous GT, current GT, previous/current masks | yes | no | `lag_rate` |
| `lag_error_margin_px` | portable as same-pixel approximation | same lag block | current prediction, previous GT, current GT, previous/current masks | yes | no | `lag_error_margin_px` |
| `ghosting_score_px_tau2` | requires optical flow, so not computed in no-flow streaming | `artifact_metrics.pair_artifact_metrics`; called by `benchmark_scared_s2m2_temporal_baselines_v4_artifact_metrics.py:169` | previous prediction warped by flow, current raw prediction, current prediction, current GT/mask | yes | yes | `ghosting_score_px_tau2` = NaN |
| `ghosting_gt_error_px_tau2` | requires optical flow, so not computed in no-flow streaming | same as above | previous prediction warped by flow, current prediction, current GT/mask | yes | yes | `ghosting_gt_error_px_tau2` = NaN |
| `ghosting_score_px_tau5` | requires optical flow, so not computed in no-flow streaming | same as above | previous prediction warped by flow, current raw prediction, current prediction, current GT/mask | yes | yes | `ghosting_score_px_tau5` = NaN |
| `ghosting_gt_error_px_tau5` | requires optical flow, so not computed in no-flow streaming | same as above | previous prediction warped by flow, current prediction, current GT/mask | yes | yes | `ghosting_gt_error_px_tau5` = NaN |
| `occlusion_disp_mae_px` | requires optical flow, so not computed in no-flow streaming | `artifact_metrics.forward_backward_occlusion_mask` inside `pair_artifact_metrics` | forward flow, backward flow, current prediction, current GT/mask | yes | yes | `occlusion_disp_mae_px` = NaN |
| `motion_compensated_temporal_mae_px` | requires optical flow, so not computed in no-flow streaming | `benchmark_scared_s2m2_temporal_baselines.py:584` `temporal_pair_metrics` | previous prediction warped by forward flow, current prediction, masks | yes | yes | `motion_compensated_temporal_mae_px` = NaN |
| `flow_forward_runtime_ms` | method/runtime metadata only | `benchmark_scared_s2m2_temporal_baselines.py:329` `load_flow_runtime_metadata` and `online_runtime_fields` | flow cache summary/manifest | no | yes | `flow_forward_runtime_ms` = NaN |
| `flow_backward_runtime_ms` | method/runtime metadata only | same as above | flow cache summary/manifest | no | yes | `flow_backward_runtime_ms` = NaN |
| `flow_runtime_used_ms` | method/runtime metadata only | same as above | method metadata and flow runtime metadata | no | yes | `flow_runtime_used_ms` = NaN |

Additional no-flow streaming proxy columns:

| new column | source |
|---|---|
| `samepixel_temporal_mae_px` | same-pixel alias of `raw_temporal_disp_diff_px` |
| `samepixel_temporal_error_variation_px` | same-pixel version of temporal error variation already added in the streaming evaluator |
| `samepixel_temporal_motion_mismatch_px` | same-pixel version of predicted temporal delta minus GT temporal delta |

