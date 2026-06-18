# Temporal Evaluation With SCARED GT

Main table:

- `temporal_evaluation.md`
- `temporal_evaluation.csv`

Protocol: SCARED `dataset_9/keyframe_3`, rectified left/right frames with GT depth/disparity/mask. Metrics are computed only on frames with GT valid-pixel ratio >= `0.20` (`103` frames out of `130`).

This folder now contains one consolidated temporal-GT table. It includes:

- frame-wise SOTA stereo methods run independently per frame;
- video-native StereoAnyVideo;
- ARGOS Tiny U-Net and ConvGRU temporal refiners;
- SGBM as a fragile classical baseline.

## Regeneration

Frame-based methods were run into:

`frame_based_gt/native_frame_methods/`

The compact table is regenerated with:

```bash
/home/pampaj/Desktop/stereo/Fast-FoundationStereo/.conda/bin/python \
  scripts/temporal_refinement/build_temporal_gt_table.py
```

## Protocol Notes

- All rows use the same SCARED temporal-GT sequence and the same `valid_pixel_ratio >= 0.20` frame filter.
- Temporal diff is mean consecutive absolute disparity difference on adjacent GT-valid masks and positive predicted disparity.
- Frame-based rows are not video models; their temporal metric measures frame-to-frame output stability.
- StereoAnyVideo is video-native and non-causal in this setup.
- ConvGRU rows are causal ARGOS refiners applied on top of S2M2-L@736 predictions.
- Tiny U-Net uses a 5-frame window and is therefore non-causal in the current prototype.
- Temporal smoothness is reported together with GT error; smoother does not automatically mean geometrically better.

## Files

- `temporal_evaluation.csv`: compact main table.
- `temporal_evaluation.md`: readable main table.
- `frame_based_gt/`: regenerated frame-wise methods on the temporal-GT sequence.
- `gt_temporal_test_dataset_9_keyframe_3/`: detailed temporal/video/refiner metrics, JSON metadata, run log, predictions, and qualitative montages.
