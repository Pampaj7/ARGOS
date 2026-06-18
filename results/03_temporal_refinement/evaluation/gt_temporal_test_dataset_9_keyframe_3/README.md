# Temporal Evaluation With SCARED GT

Sequence: `/home/pampaj/Desktop/ARGOS/dataset/SCARED/curated/temporal_gt/test_dataset_9_keyframe_3`

This table is regenerated on rectified frames with GT attached. The old long temporal cache is not scored against this GT because its frames are not pixel-aligned with the newly converted rectified SCARED frames.

| method | training_or_checkpoint | input_res | frames_with_gt_used | Depth MAE ↓ | Bad-2 mm ↓ | Disp. MAE ↓ | Temporal diff ↓ | Runtime ↓ | VRAM ↓ | causal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S2M2-L@736 | L | 736 | 103 | 2.593 | 37.149 | 8.331 | 0.984 | 187.767 | 1672.460 |  |
| S2M2-S@512 | S | 512 | 103 | 2.584 | 37.081 | 8.384 | 0.987 | 60.974 | 371.334 |  |
| StereoAnyVideo@384x640 |  | 384x640 | 103 | 2.587 | 36.693 | 8.250 | 0.925 | 146.295 | 10132.177 |  |
| ConvGRU-L736@temporal_refinement_train_convgru_l736_v2_scheduled:epoch_0030 | /home/pampaj/Desktop/ARGOS/results/03_temporal_refinement/training/convgru/temporal_refinement_train_convgru_l736_v2_scheduled/checkpoints/epoch_0030.pt |  | 103 | 2.562 | 37.361 | 8.235 | 1.094 | 24.051 | 982.904 | True |
| ConvGRU-L736@temporal_refinement_train_convgru_l736_v2_scheduled:epoch_0040 | /home/pampaj/Desktop/ARGOS/results/03_temporal_refinement/training/convgru/temporal_refinement_train_convgru_l736_v2_scheduled/checkpoints/epoch_0040.pt |  | 103 | 2.551 | 36.795 | 8.216 | 1.081 | 23.980 | 982.904 | True |
| ConvGRU-L736@temporal_refinement_train_convgru_l736_v2_scheduled:epoch_0050 | /home/pampaj/Desktop/ARGOS/results/03_temporal_refinement/training/convgru/temporal_refinement_train_convgru_l736_v2_scheduled/checkpoints/epoch_0050.pt |  | 103 | 2.555 | 36.756 | 8.227 | 1.137 | 24.042 | 982.904 | True |
| ConvGRU-L736@temporal_refinement_train_convgru_l736_v2_scheduled:latest | /home/pampaj/Desktop/ARGOS/results/03_temporal_refinement/training/convgru/temporal_refinement_train_convgru_l736_v2_scheduled/checkpoints/latest.pt |  | 103 | 2.570 | 36.839 | 8.279 | 1.176 | 23.954 | 982.904 | True |
| TinyUNet-L736@temporal_refinement_train_unet_s2m2l736_fastcache_v2_conservative:epoch_0100 | /home/pampaj/Desktop/ARGOS/results/03_temporal_refinement/training/unet/temporal_refinement_train_unet_s2m2l736_fastcache_v2_conservative/checkpoints/epoch_0100.pt |  | 103 | 2.609 | 37.720 | 8.321 | 0.974 | 24.035 | 982.064 | False |

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
- `ConvGRU-L736@temporal_refinement_train_convgru_l736_v2_scheduled:epoch_0030`
- `ConvGRU-L736@temporal_refinement_train_convgru_l736_v2_scheduled:epoch_0040`
- `ConvGRU-L736@temporal_refinement_train_convgru_l736_v2_scheduled:epoch_0050`
- `ConvGRU-L736@temporal_refinement_train_convgru_l736_v2_scheduled:latest`
- `TinyUNet-L736@temporal_refinement_train_unet_s2m2l736_fastcache_v2_conservative:epoch_0100`
