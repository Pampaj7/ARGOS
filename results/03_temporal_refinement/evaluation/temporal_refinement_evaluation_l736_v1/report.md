# Temporal Refinement Evaluation L736 V1

Frames: `126`
Sequences: `test_dataset_9_keyframe_3`
GT available: `False`

GT note: selected fast-cache validation rows have no SCARED GT/calibration; geometric depth metrics are reported as `NaN` rather than approximated.

## Summary

- `raw_s2m2_l736`: temporal_diff=1.2553, teacher_delta=0.7058, to_backbone=0.0000, to_SAV=1.4577, runtime=0.00 ms
- `stereoanyvideo`: temporal_diff=0.9672, teacher_delta=0.0000, to_backbone=1.4577, to_SAV=0.0000, runtime=0.00 ms
- `ema_alpha_0.3`: temporal_diff=0.8788, teacher_delta=0.6614, to_backbone=1.9293, to_SAV=2.6138, runtime=0.00 ms
- `prev_blend_alpha_0.3`: temporal_diff=1.1177, teacher_delta=0.7206, to_backbone=0.8231, to_SAV=1.7085, runtime=0.00 ms
- `ema_alpha_0.5`: temporal_diff=0.9991, teacher_delta=0.5882, to_backbone=0.9382, to_SAV=1.8267, runtime=0.00 ms
- `prev_blend_alpha_0.5`: temporal_diff=1.0905, teacher_delta=0.6099, to_backbone=0.5880, to_SAV=1.5820, runtime=0.00 ms
- `ema_alpha_0.7`: temporal_diff=1.0970, teacher_delta=0.5852, to_backbone=0.4409, to_SAV=1.5329, runtime=0.00 ms
- `prev_blend_alpha_0.7`: temporal_diff=1.1213, teacher_delta=0.5863, to_backbone=0.3528, to_SAV=1.4898, runtime=0.00 ms
- `ema_alpha_0.9`: temporal_diff=1.1961, teacher_delta=0.6453, to_backbone=0.1245, to_SAV=1.4502, runtime=0.00 ms
- `prev_blend_alpha_0.9`: temporal_diff=1.1984, teacher_delta=0.6449, to_backbone=0.1176, to_SAV=1.4493, runtime=0.00 ms
- `median3_noncausal`: temporal_diff=1.0257, teacher_delta=0.5303, to_backbone=0.1562, to_SAV=1.4244, runtime=0.00 ms
- `median3_causal`: temporal_diff=1.0195, teacher_delta=0.8176, to_backbone=1.0227, to_SAV=1.9009, runtime=0.00 ms
- `median5_noncausal`: temporal_diff=0.9447, teacher_delta=0.4916, to_backbone=0.2397, to_SAV=1.4117, runtime=0.00 ms
- `median5_causal`: temporal_diff=0.9316, teacher_delta=0.9191, to_backbone=1.8722, to_SAV=2.5900, runtime=0.00 ms
- `tiny_unet_conservative`: temporal_diff=1.2434, teacher_delta=0.6958, to_backbone=0.0860, to_SAV=1.4038, runtime=95.66 ms
- `tiny_unet_conservative:epoch_0010`: temporal_diff=1.2463, teacher_delta=0.6988, to_backbone=0.0961, to_SAV=1.4213, runtime=94.07 ms
- `tiny_unet_conservative:epoch_0020`: temporal_diff=1.2403, teacher_delta=0.6928, to_backbone=0.1104, to_SAV=1.3971, runtime=92.00 ms
- `tiny_unet_conservative:epoch_0030`: temporal_diff=1.2414, teacher_delta=0.6939, to_backbone=0.0887, to_SAV=1.3982, runtime=92.06 ms
- `tiny_unet_conservative:epoch_0040`: temporal_diff=1.2434, teacher_delta=0.6958, to_backbone=0.0860, to_SAV=1.4038, runtime=90.46 ms
- `tiny_unet_conservative:epoch_0050`: temporal_diff=1.2391, teacher_delta=0.6916, to_backbone=0.0934, to_SAV=1.3915, runtime=90.76 ms
- `tiny_unet_conservative:epoch_0060`: temporal_diff=1.2397, teacher_delta=0.6927, to_backbone=0.1128, to_SAV=1.3788, runtime=91.00 ms
- `tiny_unet_conservative:epoch_0070`: temporal_diff=1.2372, teacher_delta=0.6900, to_backbone=0.1019, to_SAV=1.3920, runtime=90.03 ms
- `tiny_unet_conservative:epoch_0080`: temporal_diff=1.2368, teacher_delta=0.6898, to_backbone=0.1141, to_SAV=1.3815, runtime=90.94 ms
- `tiny_unet_conservative:epoch_0090`: temporal_diff=1.2373, teacher_delta=0.6902, to_backbone=0.1037, to_SAV=1.3897, runtime=91.22 ms
- `tiny_unet_conservative:epoch_0100`: temporal_diff=1.2353, teacher_delta=0.6887, to_backbone=0.1226, to_SAV=1.3793, runtime=90.27 ms
- `convgru_v1_conservative`: temporal_diff=1.2410, teacher_delta=0.6932, to_backbone=0.0550, to_SAV=1.4355, runtime=67.58 ms
- `convgru_v1_conservative:latest`: temporal_diff=1.2281, teacher_delta=0.6815, to_backbone=0.0914, to_SAV=1.3950, runtime=67.14 ms
- `convgru_v1_conservative:epoch_0010`: temporal_diff=1.2287, teacher_delta=0.6859, to_backbone=0.1574, to_SAV=1.3980, runtime=67.72 ms
- `convgru_v1_conservative:epoch_0020`: temporal_diff=1.2314, teacher_delta=0.6856, to_backbone=0.1192, to_SAV=1.4074, runtime=67.11 ms
- `convgru_v1_conservative:epoch_0030`: temporal_diff=1.2217, teacher_delta=0.6775, to_backbone=0.1223, to_SAV=1.4056, runtime=67.30 ms
- `convgru_v1_conservative:epoch_0040`: temporal_diff=1.2245, teacher_delta=0.6796, to_backbone=0.1225, to_SAV=1.3892, runtime=67.21 ms
- `convgru_v1_conservative:epoch_0050`: temporal_diff=1.2309, teacher_delta=0.6841, to_backbone=0.0758, to_SAV=1.4140, runtime=66.74 ms
- `convgru_v1_conservative:epoch_0060`: temporal_diff=1.2316, teacher_delta=0.6853, to_backbone=0.0881, to_SAV=1.4115, runtime=66.59 ms
- `convgru_v1_conservative:epoch_0070`: temporal_diff=1.2327, teacher_delta=0.6855, to_backbone=0.0860, to_SAV=1.4062, runtime=66.91 ms
- `convgru_v1_conservative:epoch_0080`: temporal_diff=1.2331, teacher_delta=0.6861, to_backbone=0.0835, to_SAV=1.4176, runtime=67.08 ms
- `convgru_v1_conservative:epoch_0090`: temporal_diff=1.2279, teacher_delta=0.6812, to_backbone=0.1030, to_SAV=1.3980, runtime=67.30 ms
- `convgru_v1_conservative:epoch_0100`: temporal_diff=1.2281, teacher_delta=0.6815, to_backbone=0.0914, to_SAV=1.3950, runtime=67.28 ms
- `convgru_v2_scheduled`: temporal_diff=1.2496, teacher_delta=0.7055, to_backbone=0.1229, to_SAV=1.4451, runtime=67.05 ms
- `convgru_v2_scheduled:latest`: temporal_diff=1.1781, teacher_delta=0.6467, to_backbone=0.2557, to_SAV=1.3508, runtime=67.35 ms
- `convgru_v2_scheduled:epoch_0010`: temporal_diff=1.2328, teacher_delta=0.6886, to_backbone=0.1253, to_SAV=1.4281, runtime=68.01 ms
- `convgru_v2_scheduled:epoch_0020`: temporal_diff=1.2046, teacher_delta=0.6688, to_backbone=0.1543, to_SAV=1.3990, runtime=67.62 ms
- `convgru_v2_scheduled:epoch_0030`: temporal_diff=1.1545, teacher_delta=0.6290, to_backbone=0.3662, to_SAV=1.3274, runtime=67.34 ms
- `convgru_v2_scheduled:epoch_0040`: temporal_diff=1.1554, teacher_delta=0.6272, to_backbone=0.2656, to_SAV=1.3619, runtime=67.20 ms
- `convgru_v2_scheduled:epoch_0050`: temporal_diff=1.1580, teacher_delta=0.6282, to_backbone=0.3197, to_SAV=1.3132, runtime=66.95 ms
- `convgru_v2_scheduled:epoch_0060`: temporal_diff=1.1719, teacher_delta=0.6406, to_backbone=0.2320, to_SAV=1.3732, runtime=67.16 ms
- `convgru_v2_scheduled:epoch_0070`: temporal_diff=1.1720, teacher_delta=0.6442, to_backbone=0.2474, to_SAV=1.3734, runtime=67.10 ms
- `convgru_v2_scheduled:epoch_0080`: temporal_diff=1.1785, teacher_delta=0.6598, to_backbone=0.2949, to_SAV=1.3991, runtime=66.75 ms
- `convgru_v2_scheduled:epoch_0090`: temporal_diff=1.1782, teacher_delta=0.6625, to_backbone=0.2942, to_SAV=1.4245, runtime=67.04 ms
- `convgru_v2_scheduled:epoch_0100`: temporal_diff=1.1781, teacher_delta=0.6467, to_backbone=0.2557, to_SAV=1.3508, runtime=67.34 ms
