| Method | Training | Depth MAE [mm] | Bad-2 mm [%] | Disp. MAE [px] | Runtime [ms] | FPS | Peak VRAM [GB] | Device |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S2M2-S all-surgical finetuned | S all-surgical checkpoint | 0.946 | 8.709 | 0.875 | 400.765 | 2.495 | 1.055 | cuda |
| S2M2-S honest finetuned | S honest holdout checkpoint | 1.368 | 14.434 | 1.377 | 398.737 | 2.508 | 1.055 | cuda |
| S2M2-M pretrained | M pretrained | 1.581 | 21.226 | 1.414 | 470.879 | 2.124 | 1.791 | cuda |
| S2M2-L pretrained | L pretrained | 1.762 | 21.889 | 1.502 | 562.505 | 1.778 | 2.441 | cuda |
| Fast-FoundationStereo ONNX | ONNX foundation baseline | 1.763 | 24.107 | 1.707 | 390.978 | 2.558 | 8.301 | cuda |
| S2M2-S pretrained | S pretrained | 1.764 | 23.056 | 1.462 | 398.851 | 2.507 | 1.055 | cuda |
| DEFOM-Stereo ViT-L ETH3D | ViT-L ETH3D | 1.794 | 21.422 | 1.683 | 1050.214 | 0.952 | 3.926 | cuda |
| MonSter++ MixAll i16 | MixAll i16 | 1.799 | 22.370 | 1.734 | 1183.663 | 0.845 | 2.896 | cuda |
| MonSter++ MixAll | MixAll | 1.833 | 22.704 | 1.789 | 1042.725 | 0.959 | 2.930 | cuda |
| RT-MonSter++ zero-shot | zero-shot | 1.925 | 26.193 | 1.793 | 410.665 | 2.435 | 1.383 | cuda |
| DEFOM-Stereo ViT-S RVC | ViT-S RVC | 2.122 | 24.041 | 2.277 | 563.542 | 1.774 | 2.203 | cuda |
| DEFOM-Stereo ViT-L KITTI | ViT-L KITTI | 2.152 | 26.104 | 2.128 | 1047.874 | 0.954 | 3.926 | cuda |
| DEFOM-Stereo ViT-L Middlebury | ViT-L Middlebury | 2.175 | 24.502 | 2.272 | 1047.093 | 0.955 | 3.926 | cuda |
| StereoAnywhere ViT-L | ViT-L | 2.261 | 26.450 | 2.102 | 1311.007 | 0.763 | 4.018 | cuda |
| RAFT-Stereo RVC | RVC | 2.277 | 22.053 | 2.317 |  |  |  |  |
| CREStereo | local checkpoint | 2.287 | 26.699 | 2.038 | 379.912 | 2.632 | 0.844 | cuda |
| StereoAnywhere | default | 2.445 | 30.687 | 2.712 | 889.337 | 1.124 | 3.354 | cuda |
| RAFT-Stereo Middlebury | Middlebury | 2.683 | 25.030 | 2.267 | 548.436 | 1.823 | 1.922 | cuda |
| RAFT-Stereo ETH3D | ETH3D | 2.794 | 29.676 | 2.202 | 548.745 | 1.822 | 1.922 | cuda |
| DEFOM-Stereo ViT-S SceneFlow | ViT-S SceneFlow | 2.885 | 30.294 | 3.531 | 564.069 | 1.773 | 2.203 | cuda |
| RAFT-Stereo SceneFlow | SceneFlow | 2.951 | 30.764 | 2.348 | 548.736 | 1.822 | 1.922 | cuda |
| DEFOM-Stereo ViT-L SceneFlow | ViT-L SceneFlow | 5.419 | 36.985 | 8.373 | 1049.480 | 0.953 | 3.926 | cuda |
| S2M2-XL pretrained | XL pretrained | 18.360 | 46.956 | 48.325 | 783.232 | 1.277 | 4.158 | cuda |
| SGBM | OpenCV SGBM | 51.560 | 62.765 | 12.177 | 288.661 | 3.464 | 0.000 | cpu |
