# ARGOS v2 — Multi-domain Raw Error Detector (M1)

Overall verdict: **NO-GO**.

All values below are cache-grid metrics at fractional GT coverage 0.50. No OOD sample was loaded before detector, temperature and thresholds were frozen.

| Dataset | Method | Raw EPE | Output EPE | Gain | False update | Clean degradation | Coverage | Precision |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CREStereo | m0_scared_only | 1.100315 | 1.062523 | +0.037792 | 2.08% | 1.00% | 6.51% | 81.68% |
| CREStereo | multidomain | 1.100315 | 1.086211 | +0.014105 | 0.89% | 0.23% | 2.52% | 86.34% |
| D4D-heldout-specimen3 | m0_scared_only | 7.300630 | 7.362655 | -0.062025 | 14.85% | 11.88% | 16.46% | 29.32% |
| D4D-heldout-specimen3 | multidomain | 7.300630 | 7.335185 | -0.034555 | 0.00% | 0.00% | 3.07% | 6.13% |
| Fast-FoundationStereo | m0_scared_only | 0.975741 | 0.952309 | +0.023432 | 1.82% | 0.86% | 5.64% | 78.37% |
| Fast-FoundationStereo | multidomain | 0.975741 | 0.965615 | +0.010126 | 0.79% | 0.19% | 2.19% | 85.24% |
| SCARED-C-test | m0_scared_only | 0.961605 | 0.934660 | +0.026946 | 1.92% | 0.97% | 5.62% | 76.28% |
| SCARED-C-test | multidomain | 0.961605 | 0.949089 | +0.012516 | 0.88% | 0.31% | 2.33% | 81.77% |
| SERV-CT-unseen | m0_scared_only | 1.046284 | 1.103232 | -0.056949 | 38.31% | 22.54% | 39.54% | 49.11% |
| SERV-CT-unseen | multidomain | 1.046284 | 1.414039 | -0.367755 | 49.08% | 40.53% | 51.02% | 24.30% |

## Frozen protocol

- Trainable component: unchanged S1 Raw Error Detector (1,107 parameters).
- Frozen: disparity caches, SEA-RAFT, canonical BiDA warp, A2 proposal and bounded update.
- D4D supervision: curated Zivid anchor only; no temporal propagation and no prediction-derived GT.
- M1 selection/calibration: held-out SCARED-C plus D4D specimen 2 only; SERV-CT and unseen backbones are final-only.

## Exact commands

```bash
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
$PY scripts/run_multidomain_raw_error_training.py --stage train --fold m1 --added-fraction 0.25 --output results/multidomain_raw_error --device cuda:0 --workers 16 --batch-size 16 --epochs 5 --samples-per-epoch 2048 --pixels-per-error-bin 512
$PY scripts/run_multidomain_raw_error_training.py --stage train --fold m1 --added-fraction 0.50 --output results/multidomain_raw_error --device cuda:1 --workers 16 --batch-size 16 --epochs 5 --samples-per-epoch 2048 --pixels-per-error-bin 512
$PY scripts/run_multidomain_raw_error_training.py --stage select --fold m1 --output results/multidomain_raw_error --device cuda:0 --workers 16 --batch-size 16
$PY scripts/run_multidomain_raw_error_training.py --stage final --fold m1 --output results/multidomain_raw_error --device cuda:0 --workers 16 --batch-size 16
```
