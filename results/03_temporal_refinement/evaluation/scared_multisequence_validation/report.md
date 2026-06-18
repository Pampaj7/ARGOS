# SCARED Multi-Sequence Benchmark Validation

## StereoAnyVideo Runtime

Corrected true end-to-end runtime: `127.73 ms/frame`; peak VRAM `10542.4 MB`.
Timing includes image loading, resize, tensor transfer/stack, synchronized model forward, disparity resize back to original coordinates, and CPU conversion. Cached predictions are not timed. Model load is excluded; warm-up is excluded.

## S2M2-L Diagnosis

S2M2-L uses the same positive-disparity sign convention and resize-back-to-original coordinate policy as S2M2-S. No nonpositive/sign failure was found; poor multi-sequence result is concentrated in dataset_5 warped sequences, where L overestimates disparity/depth geometry relative to GT, so it appears to be a real domain/checkpoint failure rather than a loader scale bug.

## Frame-Weighted Summary

| method | depth_mae_mm | disp_mae_px | bad_2mm_pct | raw_temporal_diff | motion_compensated_temporal_mae | runtime_ms | peak_vram_mb |
| --- | --- | --- | --- | --- | --- | --- | --- |
| StereoAnyVideo | 2.8634 | 7.0171 | 39.6737 | 0.6958 | 0.5858 | 59.9916 | 6977.1122 |
| S2M2-S@512+EMA0.50 | 2.9755 | 7.4783 | 41.4374 | 0.8881 | 0.8341 | 60.9741 | 371.3345 |
| S2M2-S@512 | 3.0588 | 7.6098 | 41.4450 | 1.4092 | 1.3141 | 60.9741 | 371.3345 |
| Fast-FoundationStereo ONNX | 4.2377 | 12.5454 | 42.7982 | 3.0961 | 3.0613 | 45.3598 | nan |
| S2M2-L@736+EMA0.50 | 4.5513 | 13.2945 | 44.6050 | 1.5785 | 1.4738 | 187.7671 | 1672.4600 |
| S2M2-L@736 | 4.6033 | 13.5567 | 44.4617 | 3.0862 | 2.8414 | 187.7671 | 1672.4600 |
| ConvGRU V2 e40 | 4.6194 | 13.4794 | 45.0020 | 2.9602 | 2.6724 | 209.9492 | 2679.9346 |

## Sequence-Balanced Summary

| method | depth_mae_mm | disp_mae_px | bad_2mm_pct | raw_temporal_diff | motion_compensated_temporal_mae | runtime_ms | peak_vram_mb |
| --- | --- | --- | --- | --- | --- | --- | --- |
| StereoAnyVideo | 3.0729 | 6.0828 | 41.9389 | 0.5175 | 0.4770 | 60.0816 | 4565.8887 |
| S2M2-S@512+EMA0.50 | 3.3434 | 6.8923 | 45.0960 | 0.8998 | 0.9001 | 60.9741 | 371.3345 |
| S2M2-S@512 | 3.4188 | 7.0231 | 44.7542 | 1.7238 | 1.7248 | 60.9741 | 371.3345 |
| Fast-FoundationStereo ONNX | 5.4885 | 15.7787 | 47.3263 | 4.6786 | 4.7735 | 45.3598 | nan |
| S2M2-L@736+EMA0.50 | 6.1098 | 17.1684 | 50.6213 | 2.1123 | 2.0271 | 187.7671 | 1672.4600 |
| S2M2-L@736 | 6.1282 | 17.5199 | 50.0076 | 4.6775 | 4.4222 | 187.7671 | 1672.4600 |
| ConvGRU V2 e40 | 6.1880 | 17.4710 | 51.2255 | 4.3821 | 4.1251 | 209.7989 | 2679.9346 |

## Ranking Stability

{
  "StereoAnyVideo": {
    "best_motion_compensated_temporal_mae": 8,
    "best_depth_mae_mm": 3
  },
  "S2M2-S@512": {
    "best_depth_mae_mm": 1
  },
  "S2M2-S@512+EMA0.50": {
    "best_depth_mae_mm": 2,
    "best_motion_compensated_temporal_mae": 2
  },
  "S2M2-L@736": {},
  "S2M2-L@736+EMA0.50": {},
  "ConvGRU V2 e40": {
    "best_depth_mae_mm": 1
  },
  "Fast-FoundationStereo ONNX": {
    "best_depth_mae_mm": 3
  }
}

## Deployment Conclusion

S2M2-S@512 + EMA0.50 remains the best lightweight causal configuration under sequence-balanced evaluation: `True`.

## Reference Images

- `/dtu/p1/leopam/ARGOS/results/temporal evaluation/scared_multisequence_validation/reference_images/s2m2_l_normal_dataset_1_keyframe_2.png`
- `/dtu/p1/leopam/ARGOS/results/temporal evaluation/scared_multisequence_validation/reference_images/s2m2_l_worst_dataset_5_keyframe_1.png`
