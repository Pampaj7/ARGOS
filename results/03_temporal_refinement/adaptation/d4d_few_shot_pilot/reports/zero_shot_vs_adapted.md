# Zero-shot vs adapted (EGBM-v3-CARE-S, frozen test)

| metric | zero-shot | calib-only 2s | full 4s |
|--------|----------:|--------------:|--------:|
| MAE (raw=3.50) | 4.72 | 3.66 | 4.09 |
| new-Bad3 % | 23.5 | 1.9 | 3.4 |
| harmful rate | 0.83 | 0.34 | 0.61 |
| Δ raw<1px (0=no damage) | −1.44 | −0.05 | −0.23 |
| Δ raw>6px (higher=better) | +0.86 | +0.08 | +0.58 |
| selectivity | −0.58 | +0.03 | +0.35 |

Zero-shot destroys raw-good pixels; calibration-only removes ~96% of that damage with 10.9k
trainable params but suppresses most large-error gain; full retains large-error skill but
overfits at this data scale. See `figures/`.
