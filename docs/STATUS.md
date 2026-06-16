# ARGOS Status Log

## 2026-06-08

- Reorganized all local data under one folder per dataset:
  - SCARED: `dataset/SCARED/`;
  - SERV-CT: `dataset/SERVCT/`;
  - StereoMIS: `dataset/StereoMIS/`;
  - D4D/Dresden: `dataset/D4D/`;
  - EndoSLAM/support data: `dataset/EndoSLAM/`.
- Curated ready-to-use subsets now live inside their dataset folder:
  - `dataset/SCARED/curated/consecutive32/`;
  - `dataset/SCARED/curated/rect5/`;
  - `dataset/SCARED/curated/keyframes_gt_dataset8/`;
  - `dataset/SERVCT/argos/servct_argos/`.
- Ran S2M2-S honest SERV-CT fine-tuning on GPU:
  - train: Experiment_1 / Reference_CT;
  - test: Experiment_2 / Reference_CT;
  - mode: refiners-only, 250 steps, LR `2e-5`.
- Result on Experiment_2:
  - pretrained depth MAE: `1.7638 mm`;
  - fine-tuned depth MAE: `1.4580 mm`;
  - pretrained disparity MAE: `1.4615 px`;
  - fine-tuned disparity MAE: `1.4684 px`.
- Added before/after error montage and report:
  - `results/servct_s2m2_honest_finetune_gpu/report.md`;
  - `results/servct_s2m2_honest_finetune_gpu/s2m2_servct_before_after_error_montage.png`.
- Tested SERV-CT fine-tuned S2M2-S transfer to rectified SCARED dataset_8 keyframes:
  - pretrained S2M2-S depth MAE: `2.8372 mm`;
  - SERV-CT fine-tuned depth MAE: `2.8320 mm`;
  - pretrained disparity MAE: `4.2431 px`;
  - SERV-CT fine-tuned disparity MAE: `4.2832 px`;
  - conclusion: nearly neutral transfer, so mixed SERV-CT + SCARED tuning is likely needed.

## 2026-06-03

- Created `stereo/` workspace with multiple upstream stereo models.
- Evaluated Fast-FoundationStereo ONNX, S2M2-S/M, Stereo Anywhere, and SGBM on SERV-CT.
- Fine-tuned S2M2-S on SERV-CT with two regimes:
  - honest holdout: train Experiment_1 CT, eval Experiment_2 CT;
  - all surgical: train SERV-CT CT/RGB references, eval CT.
- Started full SCARED download in detached `screen` session `argos_scared_download`.
- Started extra download queue in detached `screen` session `argos_training_extras`.
- Cloned MonSter++ and prepared common SERV-CT evaluation script.
- Evaluated RT-MonSter++ `Zero_shot.pth` on SERV-CT:
  - disparity MAE: 1.60 px;
  - disparity RMSE: 2.78 px;
  - depth MAE: 2.05 mm;
  - depth RMSE: 3.42 mm;
  - frames: 16.
- Downloaded and evaluated MonSter++ `Mix_all_large.pth` on SERV-CT.
- Cloned and evaluated CREStereo on SERV-CT.
- Cloned RAFT-Stereo, downloaded Dropbox pretrained models, and evaluated RVC/Middlebury checkpoints on SERV-CT.
- Cloned IGEV++ and Selective-Stereo; both are waiting on Google Drive/manual checkpoint access.
- Downloaded DEFOM-Stereo checkpoints and evaluated VITS SceneFlow, VITS RVC, and VIT-L Middlebury on SERV-CT.
  - best DEFOM run so far: VIT-L Middlebury, disparity MAE 1.73 px, depth MAE 1.99 mm;
  - VIT-L SceneFlow was unstable on SERV-CT depth and is not used as a primary score.
- Added combined SERV-CT scoreboard:
  - `stereo/argos_baselines/docs/servct_scoreboard.md`;
  - `stereo/argos_baselines/images/servct_depth_mae_scoreboard.png`;
  - mirrored into ARGOS as `docs/SERVCT_BASELINE_SCOREBOARD.md`.
- Added paper/repo scaffolding:
  - `docs/EXPERIMENT_PROTOCOL.md`;
  - `docs/DATASETS.md`;
  - `docs/MODEL_ZOO.md`;
  - `docs/ROADMAP.md`;
  - `configs/servct_baselines.yaml`;
  - `configs/surgical_splits.yaml`.
- Added SERV-CT converter and verified it creates 16 unified ARGOS samples under `/home/pampaj/Desktop/stereo/argos_data/servct/`.
- Added SCARED converter placeholder that reports available archives until the full dataset layout is available.
- Added `scripts/run_all_servct_baselines.py` for lightweight regeneration of current scoreboard outputs.
- Integrated the ARGOS-Wound internal proposal direction into the main README:
  - v0 geometry-anchored stereo/RGB benchmark;
  - v1 active depth / ToF / LiDAR extension;
  - failure-aware ARGOS-Fuse direction;
  - open-wound ex-vivo acquisition and MICCAI 2027 positioning.
- Restarted long downloads in detached `screen` sessions after detecting stale network sockets:
  - `argos_scared_download`;
  - `argos_training_extras`;
  - `argos_monsterpp_large`.
- Initialized local ARGOS git repo with README, metrics, result images, and scripts. GitHub push is waiting for `gh auth login`.

## Current Tasks

- Finish SCARED download and write SCARED converter to dense disparity/depth training format.
- Extend the SCARED converter once the full archive layout is available.
- Add S2M2-L/XL when the extra training download queue reaches those weights.
- Add IGEV++/Selective-Stereo when checkpoints are available.
- Prepare a larger S2M2 fine-tuning run using SERV-CT + SCARED.
- Publish lightweight ARGOS repo to GitHub once authentication is available.

## 2026-06-09

- Benchmarked S2M2-S, S2M2-L, and S2M2-XL on rectified SCARED dataset_8 clean keyframes.
- Evaluated full resolution plus resized widths `1024`, `736`, and `512`, with disparity predictions rescaled back to original image coordinates.
- Result folder: `results/s2m2_size_tradeoff/`.
- Best current result:
  - XL full: `2.6963 mm` depth MAE, `4.1303 px` disparity MAE;
  - L full: `2.7100 mm` depth MAE, `4.1445 px` disparity MAE;
  - S full: `2.7452 mm` depth MAE, `4.1667 px` disparity MAE.
- Interpretation: XL wins, but the margin over L is tiny relative to cost; use XL as teacher candidate, L as balanced default baseline, and S/512 as the fastest real-time candidate.

- Extended the S2M2 benchmark into a resolution tradeoff study:
  - result folder: `results/s2m2_resolution_tradeoff/`;
  - best S/L under 500 ms: L full, `2.7100 mm` depth MAE, `485.10 ms`;
  - best S/L under 300 ms: L@736, `2.7425 mm` depth MAE, `185.86 ms`;
  - lowest VRAM S/L option: S@512, `2.8660 mm` depth MAE, `374.9 MB` VRAM;
  - current deployment recommendation: L@736, with XL full reserved as teacher/reference.

- Added video-stereo repository scouting under `external/video_stereo_repos/` and `results/video_stereo_repos/`.
- Cloned/linked:
  - TemporalStereo;
  - TC-Stereo / Temporally Consistent Stereo Matching;
  - DynamicStereo;
  - BiDAStereo;
  - StereoAnyVideo.
- Prepared a common 5-frame rectified SCARED dataset_8 smoke-test package with left/right images, GT disparity, GT depth, and valid masks.
- Smoke-test status:
  - StereoAnyVideo runs on ARGOS/SCARED custom stereo folders with local checkpoint and CUDA;
  - TemporalStereo needs isolated Python 3.8/PyTorch 1.10/CUDA 11.3 plus Apex/Detectron2/Cupy and checkpoint;
  - TC-Stereo needs separate env plus Dropbox checkpoint;
  - DynamicStereo needs Hydra/PyTorch3D/checkpoint and has CC-BY-NC-4.0 license;
  - BiDAStereo needs Hydra/PyTorch3D/checkpoint and likely high VRAM.
- StereoAnyVideo smoke metrics on the 5-frame SCARED package:
  - disparity MAE: `4.1624 px`;
  - depth MAE: `2.7090 mm`;
  - bad-2mm: `18.81%`;
  - comparable to S2M2-L full on this small smoke subset, but true temporal conclusions need consecutive video frames.

- Integrated StereoAnyVideo as the first video-stereo upper-bound baseline:
  - result folder: `results/stereoanyvideo_temporal_eval/`;
  - compared against S2M2-L@full, S2M2-L@736, and S2M2-S@512;
  - used `gt5` for GT accuracy and `consecutive32` for true temporal/flicker metrics.
- Key StereoAnyVideo temporal result on `consecutive32`:
  - mean consecutive disparity diff: `1.0214`;
  - S2M2-L@736: `1.6737`;
  - S2M2-S@512: `1.2217`;
  - S2M2-L@full: `8.6276`.
- Key accuracy result on `gt5`:
  - StereoAnyVideo@384x640 depth MAE: `2.7090 mm`;
  - S2M2-L@full depth MAE: `2.7100 mm`;
  - S2M2-L@736 depth MAE: `2.7425 mm`.
- Interpretation: StereoAnyVideo is a strong temporal teacher/reference and video quality upper bound. It is not yet the deployment default because the 32-frame run uses about `10.1 GB` peak VRAM at 384x640.

- Added ARGOS temporal refinement design:
  - result folder: `results/argos_temporal_refinement_design/`;
  - proposed first module: Tiny 2D U-Net residual refiner;
  - inputs: center RGB plus 5-frame S2M2 disparity window, optional edge/failure maps;
  - teacher: StereoAnyVideo disparity;
  - baseline input: S2M2-L@736, with S2M2-S@512 as fast variant;
  - no training started.
- Added reusable future structure under `scripts/temporal_refinement/` and starter config under `configs/temporal_refinement/`.

- Built the first temporal refinement debug cache:
  - result folder: `results/temporal_refinement_cache/debug_v1/`;
  - samples: `29` total;
  - `28` samples from `consecutive32` without GT;
  - `1` sample from `gt5` with GT;
  - tensor shapes: RGB `[1024,1280,3]`, S2M2 disparity window `[5,1024,1280]`;
  - all disparities stored in original image coordinates;
  - mean absolute S2M2-L@736 vs StereoAnyVideo difference: `1.6631 px`.

- Ran the first Tiny U-Net residual temporal-refiner debug experiment:
  - result folder: `results/temporal_refinement_debug_unet_v1/`;
  - trained only the ARGOS Tiny U-Net residual head;
  - S2M2-L@736 and StereoAnyVideo were frozen;
  - input crop: `256x512`, batch size `1`, CUDA on RTX 3090;
  - best epoch: `69`.
- Debug result:
  - validation teacher MAE before refinement: `0.8925 px`;
  - validation teacher MAE after refinement: `0.6369 px`;
  - consecutive32 S2M2 temporal diff: `0.8589 px`;
  - consecutive32 refined temporal diff: `0.8297 px`;
  - single-GT-crop depth MAE: `2.7915 mm`.
- Interpretation: the data path, residual model, teacher loss, checkpointing, metrics, and visualization work. The run is intentionally small and does not saturate the GPU; next training should use larger batch/crops and more supervised/temporal data.

- Ran Tiny U-Net temporal-refiner debug V2 with stronger temporal loss:
  - result folder: `results/temporal_refinement_debug_unet_v2_temporal_loss/`;
  - same cache and model as V1;
  - loss weights: teacher `0.7`, S2M2-window median temporal target `0.3`, residual L1 `0.05`, edge-aware smoothness `0.05`;
  - batch size `2`, crop `256x512`, CUDA on RTX 3090;
  - best epoch: `78`.
- V2 result:
  - validation teacher MAE after refinement: `0.6381 px` versus V1 `0.6369 px`;
  - consecutive32 refined temporal diff: `0.8324 px` versus V1 `0.8297 px`;
  - residual std: `0.5648` versus V1 `0.7347`;
  - single-GT-crop depth MAE: `2.7918 mm` versus V1 `2.7915 mm`.
- Interpretation: window-median temporal regularization makes the residual more conservative but does not improve actual consecutive-frame flicker. Next temporal refiner experiment should use true consecutive-pair/sequence loss or StereoAnyVideo temporal teacher-difference distillation.

- Ran Tiny U-Net temporal-refiner debug V3 with true consecutive-frame teacher-delta loss:
  - result folder: `results/temporal_refinement_debug_unet_v3_teacher_delta_loss/`;
  - training script: `scripts/temporal_refinement/train_debug_unet_refiner_pairs.py`;
  - dataset mode: consecutive cached pairs `(t-1, t)` with identical crop for both frames;
  - loss weights: absolute teacher `0.7`, teacher delta `0.3`, residual L1 `0.05`, edge smoothness `0.03`;
  - best epoch: `76`.
- V3 result:
  - validation teacher MAE after refinement: `0.5980 px` versus V1 `0.6369 px` and V2 `0.6381 px`;
  - consecutive32 refined temporal diff: `0.8238 px` versus V1 `0.8297 px` and V2 `0.8324 px`;
  - temporal delta MAE: `0.4647 px`;
  - residual std: `0.7676` versus V1 `0.7347` and V2 `0.5648`;
  - single-GT-crop depth MAE: `2.8060 mm`.
- Interpretation: true teacher-delta distillation is the first temporal-refiner debug objective that improves both teacher matching and temporal consistency. It is slightly more aggressive and mildly worsens the single GT crop, so the next run should keep this objective but add stronger residual/GT-aware regularization or test delta weight `0.5` with cheaper periodic evaluation.

- Built `results/temporal_refinement_cache/large_v1/` as the first larger-cache scaffold:
  - actual samples: `29`;
  - target samples: `1,000` minimum, `5,000` eventual;
  - source: all complete S2M2-L@736 and StereoAnyVideo predictions currently available locally;
  - samples by sequence: `consecutive32=28`, `gt5=1`;
  - limitation: raw SCARED video archives are present, but long-sequence S2M2/StereoAnyVideo predictions still need to be generated before the cache can truly scale.
- Patched `scripts/temporal_refinement/train_debug_unet_refiner_pairs.py` for longer runs:
  - `--eval-every`;
  - `--save-every`;
  - `--max-train-samples`;
  - `--max-val-samples`;
  - `--num-workers`;
  - `--crop-height` / `--crop-width`;
  - `--amp` / `--no-amp`;
  - `--resume`;
  - S2M2 anchor loss.
- Launched first longer Tiny U-Net run:
  - result folder: `results/temporal_refinement_train_unet_v1_large/`;
  - cache: `results/temporal_refinement_cache/large_v1/`;
  - crop: `384x640`;
  - batch size: `4`;
  - epochs: `100`;
  - eval every `5`;
  - save every `10`;
  - loss weights: absolute teacher `0.65`, teacher delta `0.30`, residual L1 `0.10`, edge smoothness `0.05`, S2M2 anchor `0.15`;
  - CUDA run uses about `3.45 GB` VRAM at launch.
- Completed first longer Tiny U-Net run:
  - best epoch: `100`;
  - validation teacher MAE before/after: `1.1287 px` -> `0.8164 px`;
  - S2M2 temporal diff: `0.8566 px`;
  - refined temporal diff: `0.7718 px`;
  - teacher temporal diff: `0.6961 px`;
  - temporal delta MAE: `0.4407 px`;
  - single-GT-crop depth MAE: `2.7775 mm`;
  - Bad-2mm: `19.76%`;
  - peak VRAM allocated: `2967 MB`;
  - runtime: `920 s`.
- Interpretation: the longer run is stability-biased. It significantly improves temporal diff and slightly improves the GT crop, but worsens teacher MAE compared with V3 debug. The stronger S2M2 anchor/residual control likely pulls the model away from StereoAnyVideo absolute predictions. Next run should warm up or reduce anchor weight while keeping teacher-delta loss.

- Built the first real long-sequence temporal-refinement source set:
  - inventory report: `results/temporal_refinement_cache/large_v2_source_inventory.md`;
  - extracted sequences: `results/scared_long_sequences/`;
  - source: SCARED `test_dataset_8.zip` and `test_dataset_9.zip`;
  - streams: `8` keyframe video streams;
  - frames per stream: `130`;
  - total stereo pairs: `1,040`;
  - estimated valid 5-frame windows: `1,008`;
  - image shape: `1024x1280`;
  - layout: SCARED `rgb.mp4` vertical stereo stack split into top-left and bottom-right images.
- Generated frozen predictions for the long SCARED streams:
  - S2M2-L@736 output: `results/scared_long_predictions/s2m2_l736/`;
  - S2M2 frames: `1,040`;
  - S2M2 peak VRAM: `1,672 MB`;
  - StereoAnyVideo@384x640 output: `results/scared_long_predictions/stereoanyvideo_384x640/`;
  - StereoAnyVideo frames: `1,040`;
  - StereoAnyVideo chunking: `64` frames with overlap `4`;
  - all dense disparity arrays are stored in original image disparity coordinates and ignored by Git.
- Built `results/temporal_refinement_cache/large_v2/`:
  - samples: `1,008`;
  - samples per sequence: `126`;
  - window size: `5`;
  - each sample stores center RGB, S2M2-L@736 disparity window, S2M2 center disparity, StereoAnyVideo center disparity, StereoAnyVideo window, and metadata;
  - payload size: about `53 GB`;
  - sanity montages: `20`;
  - mean absolute S2M2-vs-StereoAnyVideo center disparity difference: `55.32 px` overall.
- Sanity note for Large V2:
  - several extracted SCARED test streams show much larger S2M2-vs-StereoAnyVideo scale/shape disagreement than the earlier `consecutive32` and `gt5` subsets;
  - this is preserved in the cache and documented rather than corrected by hand;
  - before training on all `large_v2` samples as teacher supervision, inspect per-sequence montages and consider sequence filtering, teacher confidence masks, or GT/calibration validation.

- Built S2M2-S@512 multi-teacher cache and first slow training run:
  - S2M2-S@512 predictions: `results/scared_long_predictions/s2m2_s512/`;
  - frames: `1,040`;
  - peak S2M2-S inference VRAM: `372 MB`;
  - compressed sample cache: `results/temporal_refinement_cache/large_v2_s2m2s512/`;
  - samples: `1,008`;
  - cache size: `67 GB`;
  - first slow run output: `results/temporal_refinement_train_unet_s2m2s512_v1/`;
  - stopped after epoch `5` due I/O bottleneck, preserving logs/checkpoints.
- Slow run epoch-5 validation:
  - refined -> S2M2-L MAE: `0.6965 px`;
  - refined -> StereoAnyVideo MAE: `0.9246 px`;
  - teacher-delta MAE: `0.4631 px`;
  - refined temporal diff: `0.9623 px`.
- Built fast indexed per-frame cache:
  - result folder: `results/temporal_refinement_cache/large_v3_s2m2s512_fast/`;
  - format: per-frame float16 `.npy` disparity arrays plus `index.csv`;
  - samples: `1,008`;
  - cache size: `7.7 GB`;
  - speed report: `results/temporal_refinement_cache/large_v3_s2m2s512_fast/speed_report.md`.
- Fast-cache 2-epoch benchmark:
  - output: `results/temporal_refinement_train_unet_s2m2s512_fastcache_benchmark/`;
  - mean epoch time: `31.85 s`;
  - old compressed cache observed epoch time: about `390 s`;
  - speedup: about `12.2x`;
  - peak VRAM allocated: `2976 MB`;
  - refined -> S2M2-L MAE after 2 epochs: `0.6262 px`;
  - refined -> StereoAnyVideo MAE after 2 epochs: `0.9910 px`;
  - refined temporal diff: `0.9615 px`;
  - teacher-delta MAE: `0.4647 px`.
- Reorganized temporal-refinement training code:
  - unified trainer entrypoint: `scripts/temporal_refinement/train_refiner.py`;
  - generic implementation: `scripts/temporal_refinement/train_temporal_refiner_fastcache.py`;
  - shared training/eval utilities: `scripts/temporal_refinement/lib/training.py`;
  - legacy debug/special-case trainers moved under `scripts/temporal_refinement/legacy/`;
  - verified the unified loader works for both `s2m2_s512` and `s2m2_l736` backbones from the fast indexed cache.
- Added first causal ConvGRU temporal-refinement path:
  - model classes: `scripts/temporal_refinement/lib/models.py`;
  - trainer mode: `scripts/temporal_refinement/train_refiner.py --model convgru`;
  - input per timestep: RGB plus current frozen backbone disparity, no future-frame window;
  - training uses fixed-length clips, validation can use clips or full recurrent sequences;
  - added smoke/debug controls: `--max-train-samples`, `--max-val-samples`, `--sequence-length`, `--hidden-channels`, `--grad-clip-norm`;
  - tests: `tests/test_convgru_refiner.py`;
  - smoke output: `results/temporal_refinement_train_convgru_l736_smoke/`;
  - smoke metrics: refined -> backbone `0.0009 px`, refined -> StereoAnyVideo `0.2680 px`, refined temporal diff `0.4595 px`, teacher-delta MAE `0.2022 px`.

## GitHub

Local repo: `/home/lpampaloni/ARGOS`

Push command after authentication:

```bash
cd /home/lpampaloni/ARGOS
gh auth login
gh repo create ARGOS --public --source=. --remote=origin --push
```
