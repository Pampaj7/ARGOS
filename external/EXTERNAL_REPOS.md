# ARGOS v2 External Repositories

Cloned/audited for ARGOS v2 external-component audit. Do not modify these repositories in place.

| Repository | Local path | Commit | Branch/tag | License | Requirements | Checkpoints | Submodules |
|---|---|---|---|---|---|---|---|
| BiDAVideo / BiDAStabilizer | `external/bidavideo` | `dae817df1ceaafcb865ebd9c7aa44b16c535e856` | `main` | MIT | Python, PyTorch, hydra-core, einops, OpenCV, scipy, pytorch-lightning, moviepy | Stereo checkpoint optional; stabilizer checkpoint optional; SEA-RAFT path hard-coded to `third_party/SEA-RAFT/models/Tartan-C-T-TSKH-spring540x960-S.pth` | No |
| PPMStereo | `external/PPMStereo` | `d0ccf7705145502c1eea49e7be0ddeafbcfd6a08` | `main` | MIT | Python 3.8, PyTorch 2.3.1 CUDA 12.1, hydra-core, einops, OpenCV, scipy, flash-attn optional, unfoldNd | PPMStereo checkpoints under `ckpt/ppmstereo/`; RAFT/RAFT-Stereo checkpoints for auxiliary models | Yes: RAFT and RAFT-Stereo paths in `.gitmodules` |
| EndoStreamDepth | `external/EndoStreamDepth` | `5abe89d9c0e09f64fdc5276d21bb5a34aa815cc6` | `main` | Apache-2.0 | PyTorch, mamba-ssm/xLSTM stack, DINO/DepthAnything/DPT-related deps | Training/inference checkpoints configured through project configs; no checkpoints copied | No |
| SEA-RAFT | `external/SEA-RAFT` | `9137517ba24e628442aec097d3afe71d03503b75` | `main` | BSD-3-Clause | PyTorch, torchvision, numpy, scipy, OpenCV, h5py, tqdm, einops | Requires `.pth` checkpoint, e.g. `models/Tartan-C-T-TSKH-spring540x960-M.pth` | No |
| RAFT | `external/RAFT` | `2888e15a51fa41140771d3f498ed8023cff098d1` | `master` | BSD-3-Clause | PyTorch, torchvision, numpy, scipy, OpenCV; optional alt CUDA corr | Requires `.pth` checkpoint from `download_models.sh` | No |

## Notes

- No checkpoints, datasets, caches, binaries, or virtual environments were copied.
- `ARGOS-V2/external_code_backbone_needed/` contains only minimal clean-room utilities, notes, manifests, and license copies.
- Full source repositories remain reference-only unless an ARGOS v2 wrapper explicitly needs them.
