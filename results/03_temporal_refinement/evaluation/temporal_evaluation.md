# SCARED temporal GT evaluation

Protocol: `dataset/SCARED/curated/temporal_gt/test_dataset_9_keyframe_3`, frames with GT valid-pixel ratio >= 0.20. Metrics are averaged over 103 valid-GT frames. Temporal diff is mean consecutive absolute disparity difference on the intersection of adjacent GT-valid masks and positive predicted disparity.

Frame-based methods are run independently per frame; StereoAnyVideo and ARGOS refiners use temporal context. Lower is better for all numeric metric columns.

| Method | Training / Checkpoint | Input res. | Depth MAE ↓ | Bad-2 mm ↓ | Disp. MAE ↓ | Temporal diff ↓ | Runtime ↓ | VRAM ↓ | Causal | Frames with GT | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ConvGRU V2 e40 | ARGOS scheduled checkpoint epoch 40 | S2M2-L@736 input | 2.5508 | 36.7953 | 8.2157 | 1.0811 | 23.9798 | 982.90 | yes | 103.00 |  |
| ConvGRU V2 e50 | ARGOS scheduled checkpoint epoch 50 | S2M2-L@736 input | 2.5548 | 36.7564 | 8.2269 | 1.1375 | 24.0422 | 982.90 | yes | 103.00 |  |
| ConvGRU V2 e30 | ARGOS scheduled checkpoint epoch 30 | S2M2-L@736 input | 2.5618 | 37.3613 | 8.2345 | 1.0945 | 24.0511 | 982.90 | yes | 103.00 |  |
| ConvGRU V2 latest | ARGOS scheduled latest checkpoint | S2M2-L@736 input | 2.5700 | 36.8391 | 8.2792 | 1.1759 | 23.9543 | 982.90 | yes | 103.00 |  |
| S2M2-S@512 | official pretrained S | 512 px width | 2.5840 | 37.0813 | 8.3835 | 0.9945 | 67.1840 |  | yes | 103.00 |  |
| StereoAnyVideo@384x640 | official MIX checkpoint | 384x640 video | 2.5874 | 36.6930 | 8.2502 | 0.9248 | 146.30 | 10132.18 | no | 103.00 |  |
| Fast-FoundationStereo ONNX | official ONNX checkpoint | ONNX script default | 2.5884 | 36.8269 | 8.2818 | 1.0093 | 46.0848 |  | yes | 103.00 |  |
| S2M2-L@736 | official pretrained L | 736 px width | 2.5926 | 37.1486 | 8.3305 | 0.9878 | 179.44 |  | yes | 103.00 |  |
| DEFOM-Stereo ViT-L ETH3D | official ETH3D checkpoint | native adapter | 2.5929 | 37.0117 | 8.3129 | 1.0028 | 1642.96 |  | yes | 103.00 |  |
| S2M2-L full | official pretrained L | full resolution | 2.5947 | 37.1816 | 8.3189 | 0.9922 | 491.27 |  | yes | 103.00 |  |
| CREStereo | official checkpoint | native adapter | 2.5985 | 37.0301 | 8.3386 | 1.0370 | 444.83 |  | yes | 103.00 |  |
| S2M2-XL | official pretrained XL | full resolution | 2.6029 | 37.2626 | 8.2942 | 0.9860 | 894.36 |  | yes | 103.00 |  |
| MonSter++ MixAll | official MixAll checkpoint | native adapter | 2.6075 | 37.3706 | 8.3097 | 0.9920 | 1874.80 |  | yes | 103.00 |  |
| RT-MonSter++ zero-shot | official zero-shot checkpoint | native adapter | 2.6075 | 36.8133 | 8.3344 | 1.0555 | 132.99 |  | yes | 103.00 |  |
| Tiny U-Net e100 | ARGOS conservative checkpoint epoch 100 | S2M2-L@736 5-frame input | 2.6095 | 37.7203 | 8.3214 | 0.9742 | 24.0347 | 982.06 | no | 103.00 |  |
| RAFT-Stereo Middlebury | official Middlebury checkpoint | native adapter | 2.6097 | 37.0248 | 8.3358 | 1.0341 | 912.24 |  | yes | 103.00 |  |
| StereoAnywhere | official checkpoint | native adapter | 2.6172 | 37.3571 | 8.3628 | 1.0126 | 1506.02 |  | yes | 103.00 |  |
| SGBM | OpenCV classical baseline | full resolution | 2.7895 | 37.7945 | 7.9278 | 1.1299 | 95.8858 |  | yes | 103.00 | Fragile classical baseline; about one quarter of pixels are excluded by the positive-disparity filter. |

Caveat: temporal smoothness is not geometric correctness. SGBM is retained only as a fragile classical baseline.
