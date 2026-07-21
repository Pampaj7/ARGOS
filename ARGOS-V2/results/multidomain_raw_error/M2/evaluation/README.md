# ARGOS v2 — Multi-domain Raw Error Detector (M2)

Overall verdict: **NO-GO**.

All values below are cache-grid metrics at fractional GT coverage 0.50. No OOD sample was loaded before detector, temperature and thresholds were frozen.

| Dataset | Method | Raw EPE | Output EPE | Gain | False update | Clean degradation | Coverage | Precision |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CREStereo | m0_scared_only | 1.100315 | 1.062523 | +0.037792 | 2.08% | 1.00% | 6.51% | 81.68% |
| CREStereo | multidomain | 1.100315 | 1.075199 | +0.025116 | 0.20% | 0.15% | 2.08% | 91.21% |
| D4D-unseen | m0_scared_only | 2.905514 | 2.868071 | +0.037442 | 28.49% | 16.84% | 38.28% | 61.25% |
| D4D-unseen | multidomain | 2.905514 | 2.900132 | +0.005382 | 1.80% | 1.61% | 3.31% | 56.55% |
| Fast-FoundationStereo | m0_scared_only | 0.975741 | 0.952309 | +0.023432 | 1.82% | 0.86% | 5.64% | 78.37% |
| Fast-FoundationStereo | multidomain | 0.975741 | 0.958053 | +0.017688 | 0.19% | 0.14% | 1.97% | 86.90% |
| SCARED-C-test | m0_scared_only | 0.961605 | 0.934660 | +0.026946 | 1.92% | 0.97% | 5.62% | 76.28% |
| SCARED-C-test | multidomain | 0.961605 | 0.942083 | +0.019522 | 0.25% | 0.20% | 1.97% | 84.20% |
| SERV-CT-seen-calibration | m0_scared_only | 1.015624 | 1.061772 | -0.046148 | 28.92% | 17.78% | 33.38% | 53.55% |
| SERV-CT-seen-calibration | multidomain | 1.015624 | 1.017059 | -0.001435 | 1.63% | 1.23% | 2.35% | 40.96% |

## Frozen protocol

- Trainable component: unchanged S1 Raw Error Detector (1,107 parameters).
- Frozen: disparity caches, SEA-RAFT, canonical BiDA warp, A2 proposal and bounded update.
- D4D supervision: curated Zivid anchor only; no temporal propagation and no prediction-derived GT.
- M2 selection/calibration: held-out SCARED-C plus SERV-CT Experiment 2 only; all D4D specimens and unseen backbones are final-only.

## Exact commands

```bash
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
$PY scripts/run_multidomain_raw_error_training.py --stage train --fold m2 --added-fraction 0.25 --output results/multidomain_raw_error --device cuda:0 --workers 16 --batch-size 16 --epochs 5 --samples-per-epoch 2048 --pixels-per-error-bin 512
$PY scripts/run_multidomain_raw_error_training.py --stage train --fold m2 --added-fraction 0.50 --output results/multidomain_raw_error --device cuda:1 --workers 16 --batch-size 16 --epochs 5 --samples-per-epoch 2048 --pixels-per-error-bin 512
$PY scripts/run_multidomain_raw_error_training.py --stage select --fold m2 --output results/multidomain_raw_error --device cuda:0 --workers 16 --batch-size 16
$PY scripts/run_multidomain_raw_error_training.py --stage final --fold m2 --output results/multidomain_raw_error --device cuda:0 --workers 16 --batch-size 16
```
