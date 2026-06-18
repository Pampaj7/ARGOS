# SCARED temporal GT fairness checks

Frames: `103` with GT valid-pixel ratio >= `0.2`.

Optical flow: OpenCV Farneback local fallback. The motion-compensated metric warps previous disparity toward the current frame before computing temporal MAE. This is an interpretability check, not a learned flow benchmark.

## Summary

| method | source | causal | depth_mae_mm | bad_2mm_pct | disp_mae_px | raw_temporal_diff | motion_compensated_temporal_mae | coverage_pct | runtime_ms | end_to_end_runtime_ms | peak_vram_mb | frames | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EMA alpha=0.7 | simple_baseline_from_s2m2_l736 | yes | 2.5432 | 36.8855 | 8.2491 | 0.9231 | 0.7428 | 100.0000 |  | 187.7671 | 1672.4600 | 103 |  |
| ConvGRU V2 e40 | temporal_gt_existing | yes | 2.5508 | 36.7953 | 8.2157 | 1.0851 | 0.7567 | 100.0000 | 23.9798 | 211.7469 | 2655.3643 | 103 |  |
| ConvGRU V2 e50 | temporal_gt_existing | yes | 2.5548 | 36.7564 | 8.2269 | 1.1396 | 0.7652 | 100.0000 | 24.0422 | 211.8094 | 2655.3643 | 103 |  |
| ConvGRU V2 e30 | temporal_gt_existing | yes | 2.5618 | 37.3613 | 8.2345 | 1.0922 | 0.7617 | 100.0000 | 24.0511 | 211.8183 | 2655.3643 | 103 |  |
| ConvGRU V2 latest | temporal_gt_existing | yes | 2.5700 | 36.8391 | 8.2792 | 1.1630 | 0.7654 | 100.0000 | 23.9543 | 211.7215 | 2655.3643 | 103 |  |
| median5 non-causal | simple_baseline_from_s2m2_l736 | no | 2.5812 | 37.0337 | 8.3002 | 0.8806 | 0.7464 | 100.0000 |  | 187.7671 | 1672.4600 | 103 | uses future frames |
| S2M2-S@512 | temporal_gt_existing | yes | 2.5840 | 37.0813 | 8.3835 | 0.9945 | 0.7727 | 100.0000 | 60.9741 | 60.9741 | 371.3345 | 103 |  |
| StereoAnyVideo@384x640 | temporal_gt_existing | no | 2.5874 | 36.6930 | 8.2502 | 0.9320 | 0.7313 | 100.0000 | 146.2953 | 146.2953 | 10132.1772 | 103 |  |
| Fast-FoundationStereo ONNX | frame_based_gt | yes | 2.5884 | 36.8269 | 8.2818 | 1.0093 | 0.8035 | 100.0000 | 45.3598 | 45.3598 |  | 103 |  |
| S2M2-L@736 | temporal_gt_existing | yes | 2.5926 | 37.1486 | 8.3305 | 0.9878 | 0.7569 | 100.0000 | 187.7671 | 187.7671 | 1672.4600 | 103 |  |
| DEFOM-Stereo ViT-L ETH3D | frame_based_gt | yes | 2.5929 | 37.0117 | 8.3129 | 1.0028 | 0.7812 | 100.0000 | 1644.0549 | 1644.0549 | 5373.8970 | 103 |  |
| S2M2-L full | frame_based_gt | yes | 2.5947 | 37.1816 | 8.3189 | 0.9922 | 0.7549 | 100.0000 | 491.8661 | 491.8661 | 2898.5044 | 103 |  |
| CREStereo | frame_based_gt | yes | 2.5985 | 37.0301 | 8.3386 | 1.0370 | 0.8263 | 100.0000 | 444.3919 | 444.3919 | 1292.8320 | 103 |  |
| S2M2-XL | frame_based_gt | yes | 2.6029 | 37.2626 | 8.2942 | 0.9860 | 0.7466 | 100.0000 | 895.0778 | 895.0778 | 5176.2407 | 103 |  |
| MonSter++ MixAll | frame_based_gt | yes | 2.6075 | 37.3706 | 8.3097 | 0.9920 | 0.7507 | 100.0000 | 1875.6214 | 1875.6214 | 4670.8428 | 103 |  |
| RT-MonSter++ zero-shot | frame_based_gt | yes | 2.6075 | 36.8133 | 8.3344 | 1.0555 | 0.8358 | 100.0000 | 132.1675 | 132.1675 | 2461.4258 | 103 |  |
| Tiny U-Net e100 | temporal_gt_existing | no | 2.6095 | 37.7203 | 8.3214 | 0.9771 | 0.7452 | 100.0000 | 24.0347 | 211.8018 | 2654.5244 | 103 |  |
| RAFT-Stereo Middlebury | frame_based_gt | yes | 2.6097 | 37.0248 | 8.3358 | 1.0341 | 0.8189 | 100.0000 | 912.1840 | 912.1840 | 3419.9790 | 103 |  |
| StereoAnywhere | frame_based_gt | yes | 2.6172 | 37.3571 | 8.3628 | 1.0126 | 0.8071 | 100.0000 | 1506.4115 | 1506.4115 | 8378.8784 | 103 |  |
| SGBM | frame_based_gt | yes | 2.7895 | 37.7945 | 7.9278 | 1.1299 | 2.6570 | 75.6112 | 95.7424 | 95.7424 |  | 103 | fragile classical baseline |

## Common Valid-Pixel Intersection

| method | depth_mae_mm | bad_2mm_pct | disp_mae_px | common_valid_pixels | common_coverage_pct | causal | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EMA alpha=0.7 | 2.6532 | 36.9326 | 7.7836 | 40568142 | 75.6112 | yes |  |
| ConvGRU V2 e40 | 2.6542 | 36.7869 | 7.7598 | 40568142 | 75.6112 | yes |  |
| ConvGRU V2 e50 | 2.6549 | 36.6515 | 7.7501 | 40568142 | 75.6112 | yes |  |
| ConvGRU V2 e30 | 2.6673 | 37.4839 | 7.7761 | 40568142 | 75.6112 | yes |  |
| ConvGRU V2 latest | 2.6768 | 36.7806 | 7.8268 | 40568142 | 75.6112 | yes |  |
| S2M2-S@512 | 2.6860 | 37.0140 | 7.8864 | 40568142 | 75.6112 | yes |  |
| median5 non-causal | 2.6877 | 37.0939 | 7.8317 | 40568142 | 75.6112 | no | uses future frames |
| DEFOM-Stereo ViT-L ETH3D | 2.6966 | 37.1470 | 7.8132 | 40568142 | 75.6112 | yes |  |
| Fast-FoundationStereo ONNX | 2.6967 | 37.0303 | 7.8198 | 40568142 | 75.6112 | yes |  |
| S2M2-L@736 | 2.6971 | 37.1941 | 7.8534 | 40568142 | 75.6112 | yes |  |
| S2M2-L full | 2.6973 | 37.1682 | 7.8412 | 40568142 | 75.6112 | yes |  |
| CREStereo | 2.6981 | 37.1463 | 7.8277 | 40568142 | 75.6112 | yes |  |
| StereoAnyVideo@384x640 | 2.7006 | 37.0869 | 7.8563 | 40568142 | 75.6112 | no |  |
| RAFT-Stereo Middlebury | 2.7059 | 37.0436 | 7.8241 | 40568142 | 75.6112 | yes |  |
| RT-MonSter++ zero-shot | 2.7083 | 37.3583 | 7.8231 | 40568142 | 75.6112 | yes |  |
| S2M2-XL | 2.7147 | 37.3361 | 7.8461 | 40568142 | 75.6112 | yes |  |
| StereoAnywhere | 2.7167 | 37.3599 | 7.8376 | 40568142 | 75.6112 | yes |  |
| MonSter++ MixAll | 2.7220 | 37.6994 | 7.8708 | 40568142 | 75.6112 | yes |  |
| Tiny U-Net e100 | 2.7224 | 37.9511 | 7.8583 | 40568142 | 75.6112 | no |  |
| SGBM | 2.7895 | 37.7945 | 7.9278 | 40568142 | 75.6112 | yes | fragile classical baseline |

## ConvGRU Checkpoints

| method | depth_mae_mm | disp_mae_px | raw_temporal_diff | motion_compensated_temporal_mae | coverage_pct | total_runtime_ms |
| --- | --- | --- | --- | --- | --- | --- |
| ConvGRU V2 e30 | 2.5618 | 8.2345 | 1.0922 | 0.7617 | 100.0000 | 211.8183 |
| ConvGRU V2 e40 | 2.5508 | 8.2157 | 1.0851 | 0.7567 | 100.0000 | 211.7469 |
| ConvGRU V2 e50 | 2.5548 | 8.2269 | 1.1396 | 0.7652 | 100.0000 | 211.8094 |
| ConvGRU V2 latest | 2.5700 | 8.2792 | 1.1630 | 0.7654 | 100.0000 | 211.7215 |
| S2M2-L@736 | 2.5926 | 8.3305 | 0.9878 | 0.7569 | 100.0000 | 187.7671 |

## Interpretation

- Raw S2M2-L motion-compensated temporal MAE: `0.7569` px.
- Best ConvGRU motion-compensated temporal MAE: `ConvGRU V2 e40` at `0.7567` px.
- Best simple trade-off among ConvGRU checkpoints by depth+temporal score: `ConvGRU V2 e40`.
- Median-5 is explicitly non-causal and uses future frames.
- Runtime for refiners is reported both as refiner overhead and estimated full S2M2-L+refiner pipeline runtime.
