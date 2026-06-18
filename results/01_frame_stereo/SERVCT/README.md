# SERV-CT Evaluation

Single baseline table for how the stereo methods perform on the SERV-CT honest-test split.

| Method | Training / Checkpoint | Input res. | Depth MAE ↓ | Bad-2 mm ↓ | Disp. MAE ↓ | Runtime ↓ | VRAM ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S2M2-S all-surgical finetuned | S all-surgical checkpoint | native/legacy_eval | 0.946 | 8.71 | 0.875 | 400.8 ms | 1.05 GB |
| S2M2-S honest finetuned | S honest holdout checkpoint | native/legacy_eval | 1.368 | 14.43 | 1.377 | 398.7 ms | 1.05 GB |
| S2M2-M pretrained | M pretrained | native/legacy_eval | 1.581 | 21.23 | 1.414 | 470.9 ms | 1.79 GB |
| S2M2-L pretrained | L pretrained | native/legacy_eval | 1.762 | 21.89 | 1.502 | 562.5 ms | 2.44 GB |
| Fast-FoundationStereo ONNX | ONNX foundation baseline | native/legacy_eval | 1.763 | 24.11 | 1.707 | 391.0 ms | 8.30 GB |
| S2M2-S pretrained | S pretrained | native/legacy_eval | 1.764 | 23.06 | 1.462 | 398.9 ms | 1.05 GB |
| DEFOM-Stereo ViT-L ETH3D | ViT-L ETH3D | native/legacy_eval | 1.794 | 21.42 | 1.683 | 1050.2 ms | 3.93 GB |
| MonSter++ MixAll i16 | MixAll i16 | native/legacy_eval | 1.799 | 22.37 | 1.734 | 1183.7 ms | 2.90 GB |
| MonSter++ MixAll | MixAll | native/legacy_eval | 1.833 | 22.70 | 1.789 | 1042.7 ms | 2.93 GB |
| RT-MonSter++ zero-shot | zero-shot | native/legacy_eval | 1.925 | 26.19 | 1.793 | 410.7 ms | 1.38 GB |
| DEFOM-Stereo ViT-S RVC | ViT-S RVC | native/legacy_eval | 2.122 | 24.04 | 2.277 | 563.5 ms | 2.20 GB |
| DEFOM-Stereo ViT-L KITTI | ViT-L KITTI | native/legacy_eval | 2.152 | 26.10 | 2.128 | 1047.9 ms | 3.93 GB |
| DEFOM-Stereo ViT-L Middlebury | ViT-L Middlebury | native/legacy_eval | 2.175 | 24.50 | 2.272 | 1047.1 ms | 3.93 GB |
| StereoAnywhere ViT-L | ViT-L | native/legacy_eval | 2.261 | 26.45 | 2.102 | 1311.0 ms | 4.02 GB |
| RAFT-Stereo RVC | RVC | native/legacy_eval | 2.277 | 22.05 | 2.317 |  |  |
| CREStereo | local checkpoint | native/legacy_eval | 2.287 | 26.70 | 2.038 | 379.9 ms | 0.84 GB |
| StereoAnywhere | default | native/legacy_eval | 2.445 | 30.69 | 2.712 | 889.3 ms | 3.35 GB |
| RAFT-Stereo Middlebury | Middlebury | native/legacy_eval | 2.683 | 25.03 | 2.267 | 548.4 ms | 1.92 GB |
| RAFT-Stereo ETH3D | ETH3D | native/legacy_eval | 2.794 | 29.68 | 2.202 | 548.7 ms | 1.92 GB |
| DEFOM-Stereo ViT-S SceneFlow | ViT-S SceneFlow | native/legacy_eval | 2.885 | 30.29 | 3.531 | 564.1 ms | 2.20 GB |
| RAFT-Stereo SceneFlow | SceneFlow | native/legacy_eval | 2.951 | 30.76 | 2.348 | 548.7 ms | 1.92 GB |
| DEFOM-Stereo ViT-L SceneFlow | ViT-L SceneFlow | native/legacy_eval | 5.419 | 36.98 | 8.373 | 1049.5 ms | 3.93 GB |
| S2M2-XL pretrained | XL pretrained | native/legacy_eval | 18.360 | 46.96 | 48.325 | 783.2 ms | 4.16 GB |
| SGBM | OpenCV SGBM | native/legacy_eval | 51.560 | 62.77 | 12.177 | 288.7 ms | 0.00 GB |

Notes:

- Dataset: `dataset/SERVCT/argos/servct_argos/honest_test`, 8 samples.
- Runtime is adapter end-to-end per frame on CUDA when available.
- `S2M2-S all-surgical finetuned` is an upper-bound/all-data adaptation, not the fair held-out protocol.
- Fair fine-tuned baseline is `S2M2-S honest finetuned`.
- Source: `results/01_frame_stereo/SERVCT/servct_unified_frame_benchmark_v1/servct_benchmark_full_with_runtime.csv`.
