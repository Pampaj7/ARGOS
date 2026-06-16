# Temporal Evaluation With SCARED GT

Protocol: SCARED `dataset_9/keyframe_3`, rectified left/right frames with GT depth/disparity/mask. Metrics are computed only on frames with GT valid-pixel ratio >= `0.20` (`103` frames out of `130`).

| Method | Training / Checkpoint | Input res. | Depth MAE ↓ | Bad-2 mm ↓ | Disp. MAE ↓ | Temporal diff ↓ | Runtime ↓ | VRAM ↓ | Causal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S2M2-L@736 | official pretrained L | 736 px width | 2.593 | 37.15% | 8.331 | 0.984 | 187.8 ms | 1.63 GB | yes |
| S2M2-S@512 | official pretrained S | 512 px width | 2.584 | 37.08% | 8.384 | 0.987 | 61.0 ms | 0.36 GB | yes |
| StereoAnyVideo@384x640 | official MIX checkpoint | 384x640 video | 2.587 | 36.69% | 8.250 | 0.925 | 146.3 ms | 9.89 GB | no |
| ConvGRU V2 e30 | ARGOS scheduled checkpoint epoch 30 | S2M2-L@736 input | 2.562 | 37.36% | 8.235 | 1.094 | 24.1 ms | 0.96 GB | yes |
| ConvGRU V2 e40 | ARGOS scheduled checkpoint epoch 40 | S2M2-L@736 input | 2.551 | 36.80% | 8.216 | 1.081 | 24.0 ms | 0.96 GB | yes |
| ConvGRU V2 e50 | ARGOS scheduled checkpoint epoch 50 | S2M2-L@736 input | 2.555 | 36.76% | 8.227 | 1.137 | 24.0 ms | 0.96 GB | yes |
| ConvGRU V2 latest | ARGOS scheduled latest checkpoint | S2M2-L@736 input | 2.570 | 36.84% | 8.279 | 1.176 | 24.0 ms | 0.96 GB | yes |
| Tiny U-Net e100 | ARGOS conservative checkpoint epoch 100 | S2M2-L@736 5-frame input | 2.609 | 37.72% | 8.321 | 0.974 | 24.0 ms | 0.96 GB | no |

## Protocol Notes

- All predictions in this table were regenerated on the same rectified GT-aligned frame sequence.
- The older no-GT temporal-cache tables were removed from this folder to avoid mixing protocols.
- StereoAnyVideo is video-native and run in chunks; S2M2 rows are frame-wise.
- ConvGRU/Tiny U-Net rows are ARGOS refiners applied on top of S2M2-L@736 predictions.
- Temporal smoothness is reported together with GT error; smoother does not automatically mean geometrically better.

## Files

- `temporal_evaluation.csv`: compact main table.
- `gt_temporal_test_dataset_9_keyframe_3/`: detailed per-frame metrics, JSON metadata, run log, predictions, and qualitative montages.
