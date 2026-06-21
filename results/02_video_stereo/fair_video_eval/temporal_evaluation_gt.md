# Temporal Evaluation With SCARED GT

Sequence: `dataset/SCARED/curated/temporal_gt/test_dataset_9_keyframe_3`

This table is regenerated on rectified frames with GT attached. The old long temporal cache is not scored against this GT because its frames are not pixel-aligned with the newly converted rectified SCARED frames.

| method | training_or_checkpoint | input_res | frames_with_gt_used | Depth MAE ↓ | Bad-2 mm ↓ | Disp. MAE ↓ | Temporal diff ↓ | Runtime ↓ | VRAM ↓ | causal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S2M2-L@736 | L | 736 | 103 | 2.593 | 37.144 | 8.331 | 0.984 | 149.946 | 1698.312 |  |
| S2M2-S@512 | S | 512 | 103 | 2.584 | 37.089 | 8.384 | 0.988 | 64.879 | 395.401 |  |
| StereoAnyVideo@384x640 |  | 384x640 | 103 | 2.588 | 36.694 | 8.251 | 0.925 | 59.208 | 10156.241 |  |

## Protocol

- GT source: raw `dataset_9.zip`, block `keyframe_3`, converted into rectified left/right plus depth/disparity/mask.
- Geometry metrics use only frames whose GT valid-pixel ratio passes the configured threshold.
- Disparity predictions are saved in original image coordinates before metric computation.
- StereoAnyVideo is run in chunks; it remains a video-native teacher/baseline, but chunking can slightly affect long-range temporal context.
- Temporal smoothness is not geometric correctness; here we report both GT errors and temporal differences.

## Methods Included

- `S2M2-L@736`
- `S2M2-S@512`
- `StereoAnyVideo@384x640`
