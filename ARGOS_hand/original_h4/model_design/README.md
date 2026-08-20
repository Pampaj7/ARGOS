# Canonical H4 model and training

This directory contains the dataset-2-selected bounded CODD-style H=4 baseline:
model sources, frozen alignment/data dependencies, the immutable checkpoint,
its launch entrypoint, and the retained `metrics/unified_metrics.py` utility.

The frozen checkpoint is `checkpoints/codd_style_h4_best_validation.pt`
(`99c5745c164fd4903b8aa8acf8f57efccacd9cdcd0d4ed4305cd10609324d725`).
`checkpoints/training_provenance.json` locks its producing configuration and
the hashes checked before training starts.

Run from this directory:

```bash
/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python train.py --dry-run
CUDA_VISIBLE_DEVICES=1 /dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python train.py
CUDA_VISIBLE_DEVICES=1 /dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python train.py --output training_runs/experiment_1 --no-resume
CUDA_VISIBLE_DEVICES=1 /dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python train.py --output training_runs/experiment_1 --resume
```

The real configuration is seed 0; train dataset IDs 1/3/6; validation ID 2;
S2M2-S, RAFT-Stereo and StereoAnywhere; recurrent four-frame clips; 12 epochs;
batch 4; AdamW `2e-4`, `1e-4` weight decay; coverage 0.5; native reset/fusion
thresholds 5/1 px; alpha regularizer 0.2; gradient clipping 1; CUDA AMP; and
learned cues enabled.  Only output location, workers and resume are selectable.

This is not the separate geometry-v1 multi-anchor configuration described in
the manuscript (`10` epochs, batch `12`, learning rate `2e-3`, anchors
`{1,2,4,8}`).  No geometry-v1 objective or scheduler is used here.

## Exact producing run

The frozen baseline is seed `0`, selected solely on validation dataset `2` at
epoch `12` (`val EPE = 0.2958238375`).  Dataset IDs `1/3/6` trained it;
dataset `7` was never used for training.  Training backbones were S2M2-S,
RAFT-Stereo and StereoAnywhere; CREStereo and Fast-FoundationStereo were
excluded.  There were 8,616 training frames / 8,606 causal pairs, 2,148
non-overlap H4 clips per backbone (6,444 clips/epoch), about 1,611 optimizer
steps per epoch, 19,332 steps over 12 epochs, and about 309,312 pair
exposures.  Validation had 4,249 frames / 4,246 pairs and 1,061 clips per
backbone (3,183 total).

The cache grid was `144x180`, with no crop.  The trainable fusion model has
177,338 parameters and 142 cues.  This section documents the ORIGINAL canonical
run, which the paper now reports as an ablation: the shipped head is the
38-cue, 154,874-parameter variant trained by the same recipe with
`disable_learned_stereo_evidence`, under
`training_runs/ablation_A2_no_learned_evidence*`.  Its objective is Huber fused disparity plus
reset, fusion and tie regularizers (not the geometry-v1 classification/ranking
objective).  The original timestamps imply about `3h06`; no exact duration was
logged.  Canonical metadata is locked in `checkpoints/training_provenance.json`.
