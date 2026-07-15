# ARGOS v2 — Raw Error Detection and Abstention

## Verdict

**GO** for the frozen raw-error detector authorizing the frozen A2 t-1
proposal.  At cache-grid coverage 0.50 on held-out seen keyframes 3/4, balanced
abstention improves EPE from **0.961605 to 0.934660** on all three seen
backbones.  It retains **63.2%** of the original A2 EPE gain while reducing the
historical false-update rate from **35.07% to 1.92%** and clean degradation from
**13.15% to 0.97%** on this exact common mask. Intervention precision is 76.28%,
3.33% of frames worsen, worst-frame degradation is 0.0401 px, and the maximum
sequence 95th-percentile degradation is 0.00158 px.

The frozen one-shot Fast-FoundationStereo evaluation also improves EPE from
**0.975741 to 0.952309**, with 1.88% false updates and 0.91% clean degradation.
No unseen data were used before architecture, loss, checkpoint, temperature,
and operating modes were frozen; no tuning occurred afterward.

## Frozen system

- Detector: S1 pixel-wise shared 1x1 CNN, 1,107 parameters, receptive field 1.
- Inputs: 17 universal channels from raw disparity, t-1 BiDA evidence, masks,
  flow/photometric evidence, and frozen A2 proposal/gates.
- Outputs: wrong-raw probability, positive expected error `mu`, positive
  uncertainty `sigma`.
- A2 and SEA-RAFT are frozen and executed without gradient graphs.
- Balanced: temperature 0.740366; `p>=0.50`, `mu>=0.25 px`, `sigma<=2.0 px`,
  valid support, finite proposal, `|update|<=3 px`.
- Ultra-safe differs only by `p>=0.95`.
- Rejection is bit-exact raw via `torch.where`.

## Split and masks

- Train: all accepted non-dataset-7 sequences; S2M2-S, RAFT-Stereo,
  StereoAnywhere, balanced per backbone.
- Calibration/checkpoint selection: `dataset_7_keyframe_1/2`.
- Final seen test: `dataset_7_keyframe_3/4`.
- Primary unseen: Fast-FoundationStereo on keyframes 3/4, one shot.
- GT cache resize: `resize(disparity*valid)/resize(valid)`, followed by width
  scaling. Primary fractional coverage is 0.50; sensitivity is recorded at
  0.05/0.25/0.50/0.90.
- All method comparisons use the same GT/raw/aligned/support common mask.

## Detector-only result

On the calibration split at epsilon 0.50: AUROC **0.8730**, AP **0.4031**,
Brier **0.0260**, ECE **0.0051**, expected-error MAE **0.1373**, Pearson
**0.4556**, Spearman **0.3110**, and uncertainty/error correlation **0.3869**.
The final seen sensitivity table is `detector_metrics.csv`; at coverage 0.50
and epsilon 0.50 it gives AUROC 0.8702 and AP 0.6126.  The mean regression MAE
on final seen is 1.0306, exposing a real sequence-level calibration shift even
though classification remains strong.

At the top 1% confidence coverage, raw-error precision is 62.6%, A2-helpful
precision is 65.0%, and mean available gain is 0.213 px. Curves are in
`risk_coverage.csv` and `precision_coverage.csv`.

## Geometry at coverage 0.50

| Method | EPE | Bad1 | Bad3 | Boundary EPE | new-Bad3 |
|---|---:|---:|---:|---:|---:|
| Raw | 0.961605 | 0.08150 | 0.05915 | 2.52873 | 0 |
| Original frozen A2 | 0.918975 | 0.07851 | 0.05716 | 2.36990 | 0.00050 |
| Heuristic BiDA gate | 0.950150 | 0.08211 | 0.05955 | 2.49684 | 0.00259 |
| Ultra-safe authorized A2 | 0.943572 | 0.08049 | 0.05787 | 2.45289 | 0.00022 |
| Balanced authorized A2 | **0.934660** | **0.07942** | **0.05742** | **2.41054** | **0.00036** |
| Oracle-authorized A2 | 0.896479 | 0.07609 | 0.05671 | 2.32102 | 0 |

Balanced recovers 41.4% of the oracle-authorization gain and 63.2% of the
original A2 gain. It is intentionally less aggressive than A2.

## Per-backbone balanced result

| Backbone | Raw EPE | Output EPE | Gain | False update | Clean degradation |
|---|---:|---:|---:|---:|---:|
| S2M2-S | 1.028950 | 1.009592 | 0.019358 | 1.96% | 1.02% |
| RAFT-Stereo | 0.925869 | 0.897324 | 0.028545 | 1.97% | 0.97% |
| StereoAnywhere | 0.929996 | 0.897062 | 0.032934 | 1.84% | 0.92% |
| Fast-FoundationStereo unseen | 0.975741 | 0.952309 | 0.023432 | 1.88% | 0.91% |

## Ablations and smoke

The controlled pilot is in `architecture_target_ablation.csv`. S1 beat the
larger local/multiscale detectors; A4 and 5:1 clean cost gave the best AP and
calibration. The real 24-pair overfit smoke reduced total loss from 0.68175 to
-0.11061, changed the gate, remained finite, reached validation AUROC up to
0.867 and Spearman up to 0.504, then its temporary directory was deleted.

## Runtime

The detector costs 0.045 ms/frame at batch 16 on H100. Validated cache-grid
SEA-RAFT and BiDA evidence reference timings are 5.78 and 7.54 ms/frame. The
detector is therefore negligible relative to temporal evidence construction.
See `runtime_summary.json`.

## Exact commands

```bash
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python

$PY -m pytest -q model_design/tests/test_bidavideo.py \
  model_design/tests/test_raw_error_detector.py \
  model_design/tests/test_abstention.py \
  model_design/tests/test_raw_error_dataset.py

CUDA_VISIBLE_DEVICES=1 $PY scripts/run_raw_error_abstention.py \
  --mode smoke --output /tmp/argos_raw_error_smoke --architecture s2 \
  --loss-mode a4 --channels 24 --batch-size 8 --workers 8 \
  --learning-rate 0.003 --device cuda:0

CUDA_VISIBLE_DEVICES=1 $PY scripts/run_raw_error_abstention.py \
  --mode train --output results/raw_error_abstention/full \
  --architecture s1 --loss-mode a4 --false-positive-cost 5 --channels 24 \
  --epochs 5 --batch-size 16 --workers 32 --max-train-pairs 256 \
  --max-validation-pairs 160 --learning-rate 0.002 --device cuda:0 --resume

CUDA_VISIBLE_DEVICES=1 $PY scripts/run_raw_error_abstention.py \
  --mode evaluate --output results/raw_error_abstention/full \
  --checkpoint results/raw_error_abstention/full/checkpoints/best_validation.pt \
  --batch-size 16 --workers 32 --max-validation-pairs 160 \
  --sample-pixels-per-frame 2048 --device cuda:0

# Executed once; a completion marker prevents accidental repetition.
CUDA_VISIBLE_DEVICES=1 $PY scripts/run_raw_error_abstention.py \
  --mode unseen --output results/raw_error_abstention/full \
  --checkpoint results/raw_error_abstention/full/checkpoints/best_validation.pt \
  --batch-size 16 --workers 32 --max-validation-pairs 160 --device cuda:0
```

No joint fine-tuning was run: the frozen composition already passes promotion,
and YAGNI avoids confounding the result.
