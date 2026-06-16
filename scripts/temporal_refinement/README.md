# Temporal Refinement

All ARGOS temporal-refinement code lives here.

This directory contains both:

- runnable CLI scripts for cache building, prediction, training, and evaluation;
- reusable refiner modules under `lib/`.

The refiner stack uses frozen S2M2 disparity predictions, StereoAnyVideo teacher predictions, and surgical GT where available.

Design document:

`results/argos_temporal_refinement_design/design.md`

## Structure

```text
scripts/temporal_refinement/
  lib/
    datasets.py
    losses.py
    metrics.py
    models.py
    training.py
  build_debug_cache.py
  build_large_cache.py
  build_large_v2_cache.py
  build_large_v3_s2m2s512_fast_cache.py
  predict_s2m2_long_sequences.py
  predict_stereoanyvideo_long_sequences.py
  train_refiner.py
  train_temporal_refiner_fastcache.py
  evaluate_temporal_refinement.py
  legacy/
```

There is no separate `argos/temporal_refinement/` package anymore; keeping the code here avoids duplicate homes for the same project area.

Principles:

- freeze upstream stereo models;
- train only small ARGOS refiner modules;
- cache S2M2 and StereoAnyVideo predictions before training;
- keep tensor interfaces generic;
- keep all generated dense arrays and checkpoints out of Git.

## Unified Trainer

Use one trainer for current ARGOS temporal-refinement runs:

```bash
PYTHONPATH=. python scripts/temporal_refinement/train_refiner.py \
  --cache-root results/temporal_refinement_cache/large_v3_s2m2s512_fast \
  --index-file index.csv \
  --out-dir results/temporal_refinement_train_unet_s2m2s512_example \
  --backbone-prefix s2m2_s512 \
  --spatial-teacher-prefix s2m2_l736 \
  --temporal-teacher-prefix sav \
  --spatial-target teacher \
  --epochs 100 \
  --batch-size 16 \
  --crop-height 384 \
  --crop-width 640 \
  --num-workers 8 \
  --persistent-workers \
  --prefetch-factor 4 \
  --eval-every 5 \
  --save-every 10 \
  --spatial-weight 0.45 \
  --abs-sav-weight 0.20 \
  --delta-sav-weight 0.25 \
  --res-weight 0.10 \
  --edge-weight 0.05
```

The same entrypoint can train different backbones from the indexed fast cache.

S2M2-S@512 backbone with S2M2-L spatial teacher:

```bash
--index-file index.csv \
--backbone-prefix s2m2_s512 \
--spatial-teacher-prefix s2m2_l736 \
--temporal-teacher-prefix sav \
--spatial-target teacher \
--residual-clamp-px 4.0
```

S2M2-L@736 conservative stabilizer:

```bash
--index-file index_s2m2l736.csv \
--backbone-prefix s2m2_l736 \
--spatial-teacher-prefix s2m2_l736 \
--temporal-teacher-prefix sav \
--spatial-target backbone \
--residual-clamp-px 2.0
```

Important CLI knobs:

- `--backbone-prefix`: frozen disparity stream used as refiner input, e.g. `s2m2_s512` or `s2m2_l736`;
- `--spatial-teacher-prefix`: spatial teacher stream, usually `s2m2_l736`;
- `--temporal-teacher-prefix`: temporal teacher stream, usually `sav`;
- `--spatial-target`: `teacher`, `backbone`, or `none`;
- `--warmup-epochs` plus `--warmup-*-weight`: optional scheduled loss weights;
- `--residual-clamp-px`: maximum residual amplitude predicted by the Tiny U-Net.

Implementation lives in `train_temporal_refiner_fastcache.py`; `train_refiner.py` is the stable short entrypoint. The default `--model tiny_unet` path is kept compatible with the existing pair/5-frame-window training runs.

## Causal ConvGRU Refiner

The unified trainer also supports a causal online ConvGRU refiner:

- input at each timestep: center RGB `[3,H,W]` plus current frozen backbone disparity `[1,H,W]`;
- recurrent hidden state is propagated one frame at a time;
- no future frame is loaded into the model input for a timestep;
- training uses fixed-length clips, usually `--sequence-length 5`;
- validation can optionally run recurrently over whole validation sequences with `--eval-full-sequences`;
- output remains a bounded residual disparity added to the frozen backbone disparity.

S2M2-L@736 conservative causal stabilizer smoke/short run:

```bash
PYTHONPATH=. python scripts/temporal_refinement/train_refiner.py \
  --model convgru \
  --cache-root results/temporal_refinement_cache/large_v3_s2m2s512_fast \
  --index-file index_s2m2l736.csv \
  --out-dir results/temporal_refinement_train_convgru_l736_example \
  --backbone-prefix s2m2_l736 \
  --spatial-teacher-prefix s2m2_l736 \
  --temporal-teacher-prefix sav \
  --spatial-target backbone \
  --epochs 100 \
  --batch-size 16 \
  --crop-height 384 \
  --crop-width 640 \
  --num-workers 8 \
  --persistent-workers \
  --prefetch-factor 4 \
  --sequence-length 5 \
  --hidden-channels 64 \
  --residual-clamp-px 2.0 \
  --spatial-weight 0.40 \
  --abs-sav-weight 0.35 \
  --delta-sav-weight 0.20 \
  --res-weight 0.20 \
  --edge-weight 0.05 \
  --grad-clip-norm 1.0 \
  --eval-every 5 \
  --save-every 10
```

S2M2-S@512 deployment-oriented causal refiner:

```bash
PYTHONPATH=. python scripts/temporal_refinement/train_refiner.py \
  --model convgru \
  --cache-root results/temporal_refinement_cache/large_v3_s2m2s512_fast \
  --index-file index.csv \
  --out-dir results/temporal_refinement_train_convgru_s2m2s512_example \
  --backbone-prefix s2m2_s512 \
  --spatial-teacher-prefix s2m2_l736 \
  --temporal-teacher-prefix sav \
  --spatial-target teacher \
  --epochs 100 \
  --batch-size 16 \
  --crop-height 384 \
  --crop-width 640 \
  --num-workers 8 \
  --persistent-workers \
  --prefetch-factor 4 \
  --sequence-length 5 \
  --hidden-channels 64 \
  --residual-clamp-px 4.0
```

Debug controls for quick experiments:

- `--max-train-samples` and `--max-val-samples` limit index rows before clip construction;
- `--base-channels` and `--hidden-channels` shrink the ConvGRU for smoke tests;
- `--score-spatial-weight` controls ConvGRU checkpoint selection as `teacher_delta_mae + weight * spatial_mae`;
- Tiny U-Net checkpoint selection keeps the historical score `spatial_mae + 0.5 * teacher_delta_mae`.

## Cache Utilities

`build_debug_cache.py` builds the first local cache for the Tiny U-Net prototype:

```bash
python scripts/temporal_refinement/build_debug_cache.py
```

Output:

`results/temporal_refinement_cache/debug_v1/`

The `.npz` samples are intentionally ignored by Git. The cache README, index, metadata, and sanity montages are small enough to track if needed.

Historical debug trainers were moved to `scripts/temporal_refinement/legacy/`. They are kept only to reproduce old V1/V2/V3 debug runs and should not be used for new experiments.

Legacy first Tiny U-Net residual-refiner debug experiment:

```bash
PYTHONPATH=. python scripts/temporal_refinement/legacy/train_debug_unet_refiner.py \
  --epochs 80 \
  --batch-size 1 \
  --crop-h 256 \
  --crop-w 512 \
  --out-dir results/temporal_refinement_debug_unet_v1
```

Output:

`results/temporal_refinement_debug_unet_v1/`

This trains only the ARGOS Tiny U-Net residual head. S2M2-L@736 and StereoAnyVideo stay frozen. The current debug run uses small crops and batch size 1, so it runs on CUDA but does not saturate the RTX 3090.

Temporal-loss v2 debug run:

```bash
PYTHONPATH=. python scripts/temporal_refinement/legacy/train_debug_unet_refiner.py \
  --epochs 80 \
  --batch-size 2 \
  --crop-h 256 \
  --crop-w 512 \
  --teacher-weight 0.7 \
  --temporal-weight 0.3 \
  --out-dir results/temporal_refinement_debug_unet_v2_temporal_loss
```

V2 adds a window-median temporal loss. It makes the residual smaller, but did not improve consecutive-frame temporal diff over V1. Future temporal losses should use consecutive sample pairs or StereoAnyVideo temporal teacher differences directly.

Teacher-delta v3 debug run:

```bash
PYTHONPATH=. python scripts/temporal_refinement/legacy/train_debug_unet_refiner_pairs.py \
  --epochs 80 \
  --batch-size 2 \
  --crop-h 256 \
  --crop-w 512 \
  --abs-weight 0.7 \
  --delta-weight 0.3 \
  --out-dir results/temporal_refinement_debug_unet_v3_teacher_delta_loss
```

V3 trains on consecutive cached pairs and distills StereoAnyVideo temporal dynamics. It is the first debug run to improve both teacher MAE and temporal diff over V1/V2.

Large-cache/training utilities:

```bash
cd /home/pampaj/Desktop/ARGOS
PYTHONPATH=/home/pampaj/Desktop/ARGOS \
/home/pampaj/Desktop/stereo/Fast-FoundationStereo/.conda/bin/python \
  scripts/temporal_refinement/build_large_cache.py
```

`build_large_cache.py` creates `results/temporal_refinement_cache/large_v1/` from all currently available complete S2M2-L@736 and StereoAnyVideo predictions. At the moment this is a seed cache, not a true 1,000-sample cache, because long SCARED video predictions still need to be generated.

Large V2 long-sequence cache utilities:

```bash
cd /home/pampaj/Desktop/ARGOS
PYTHONPATH=/home/pampaj/Desktop/ARGOS \
/home/pampaj/Desktop/stereo/Fast-FoundationStereo/.conda/bin/python \
  scripts/temporal_refinement/inventory_scared_sources.py

PYTHONPATH=/home/pampaj/Desktop/ARGOS \
/home/pampaj/Desktop/stereo/Fast-FoundationStereo/.conda/bin/python \
  scripts/temporal_refinement/extract_scared_long_sequences.py --max-per-sequence 130

PYTHONPATH=/home/pampaj/Desktop/ARGOS \
/home/pampaj/Desktop/stereo/Fast-FoundationStereo/.conda/bin/python \
  scripts/temporal_refinement/predict_s2m2_long_sequences.py \
  --sequences-root results/04_dataset_derivatives/SCARED/scared_long_sequences \
  --out-root results/04_dataset_derivatives/SCARED/scared_long_predictions/s2m2_l736

PYTHONPATH=/home/pampaj/Desktop/ARGOS \
/home/pampaj/Desktop/stereo/Fast-FoundationStereo/.conda/bin/python \
  scripts/temporal_refinement/predict_stereoanyvideo_long_sequences.py \
  --sequences-root results/04_dataset_derivatives/SCARED/scared_long_sequences \
  --out-root results/04_dataset_derivatives/SCARED/scared_long_predictions/stereoanyvideo_384x640 \
  --chunk-size 64 \
  --overlap 4

PYTHONPATH=/home/pampaj/Desktop/ARGOS \
/home/pampaj/Desktop/stereo/Fast-FoundationStereo/.conda/bin/python \
  scripts/temporal_refinement/build_large_v2_cache.py
```

Large V2 outputs:

- source inventory: `results/temporal_refinement_cache/large_v2_source_inventory.md`;
- extracted long stereo sequences: `results/04_dataset_derivatives/SCARED/scared_long_sequences/`;
- S2M2-L@736 predictions: `results/04_dataset_derivatives/SCARED/scared_long_predictions/s2m2_l736/`;
- StereoAnyVideo@384x640 predictions: `results/04_dataset_derivatives/SCARED/scared_long_predictions/stereoanyvideo_384x640/`;
- cache: `results/temporal_refinement_cache/large_v2/`.

Large V2 currently contains `1,008` 5-frame samples from `8` SCARED test keyframe video streams. Each sample stores full-resolution RGB, S2M2-L@736 disparity window, StereoAnyVideo teacher window, center-frame fields, and metadata in original image disparity coordinates. Payload arrays and extracted frames are ignored by Git.

Sanity note: several long test streams show large absolute differences between S2M2 and StereoAnyVideo after both are rescaled to original disparity coordinates. This is preserved deliberately for teacher-analysis rather than corrected by hand; check montages and GT/calibration before treating every teacher frame as reliable supervision.

S2M2-S@512 multi-teacher fast cache:

```bash
cd /home/pampaj/Desktop/ARGOS
PYTHONPATH=/home/pampaj/Desktop/ARGOS \
/home/pampaj/Desktop/stereo/Fast-FoundationStereo/.conda/bin/python \
  scripts/temporal_refinement/build_large_v3_s2m2s512_fast_cache.py

PYTHONPATH=/home/pampaj/Desktop/ARGOS \
/home/pampaj/Desktop/stereo/Fast-FoundationStereo/.conda/bin/python \
  scripts/temporal_refinement/train_refiner.py \
  --cache-root results/temporal_refinement_cache/large_v3_s2m2s512_fast \
  --index-file index.csv \
  --out-dir results/temporal_refinement_train_unet_s2m2s512_fastcache_benchmark \
  --backbone-prefix s2m2_s512 \
  --spatial-teacher-prefix s2m2_l736 \
  --temporal-teacher-prefix sav \
  --spatial-target teacher \
  --epochs 2 \
  --batch-size 4 \
  --crop-height 384 \
  --crop-width 640 \
  --num-workers 2
```

The fast cache stores per-frame float16 disparity arrays and reconstructs 5-frame windows from `index.csv`. It reduces cache size from `67 GB` to `7.7 GB` and epoch time from roughly `390 s` to `31.9 s` for the same crop/batch benchmark.

Speed report:

`results/temporal_refinement_cache/large_v3_s2m2s512_fast/speed_report.md`

The unified trainer supports longer-run controls:

- `--eval-every`
- `--save-every`
- `--num-workers`
- `--batch-size`
- `--crop-height` / `--crop-width`
- `--amp`
- `--warmup-epochs`
- `--warmup-*-weight`
- `--backbone-prefix`
- `--spatial-teacher-prefix`
- `--temporal-teacher-prefix`
- `--spatial-target`
