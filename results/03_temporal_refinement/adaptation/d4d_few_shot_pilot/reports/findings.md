# D4D few-shot pilot — findings

Frozen eval: `session_disjoint` val (selection) + test (30 anchors). Raw S2M2-S frozen-test
MAE = **3.50 px**. 84 runs, mean over available seeds. Grad isolation verified (zero frozen drift).

## Headline (EGBM-v3-CARE-S)

| config | MAE | ΔMAE | new-Bad3 | harmful | Δ(raw<1px) | Δ(raw>6px) | selectivity |
|--------|----:|-----:|---------:|--------:|-----------:|-----------:|------------:|
| zero-shot            | 4.72 | −1.22 | 23.5 | 0.83 | **−1.44** | +0.86 | **−0.58** |
| calibration-only 2s  | 3.66 | −0.16 | 1.9  | 0.34 | −0.05 | +0.08 | +0.03 |
| calibration-only 8s  | 3.50 | +0.00 | 0.0  | 0.04 | −0.00 | +0.00 | +0.00 |
| head-only 8s         | 3.53 | −0.03 | 0.1  | 0.25 | −0.01 | +0.05 | +0.04 |
| full 4s              | 4.09 | −0.59 | 3.4  | 0.61 | −0.23 | **+0.58** | **+0.35** |
| scratch 4s           | 3.47 | +0.03 | 1.2  | 0.13 | −0.03 | +0.14 | +0.11 |

## Answers

1. **Modes implemented**: zero-shot, calibration-only, head-only, full, scratch (both models).
2. **Trainable params**: EGBM calib 10,856 (0.25%) / head 94,577 (2.1%) / full 4.42M;
   v3.2c calib 65 / head 130 / full 194,818.
3. **Decision gate**: PASSED (calibration-only removed 96% of raw-good damage, flipped
   selectivity −0.58→+0.09, beat zero-shot combined score).
4. **1/2/4 sessions**: calibration-only already near-safe at 1–2 sessions (new-Bad3 ≈0,
   harmful 0.0–0.34); more sessions monotonically reduce residual harm.
5. **Pretrained vs scratch**: at 4–8 sessions scratch reaches the same *parity* (MAE ≈3.47–3.51)
   by learning ~identity; pretraining's edge is that a 10k-param calibration reaches the same
   safety far cheaper. Neither clearly beats raw MAE at ≤8 sessions.
6. **Raw-good preservation**: dramatically fixed — Δ(raw<1px) −1.44 → −0.05 with 2 sessions.
7. **Large-error recovery**: **partially retained.** calibration-only shrinks Δ(raw>6px)
   +0.86→+0.1 (it mostly SUPPRESSES correction). Only **full** fine-tuning keeps meaningful
   large-error gain (+0.58) and the best selectivity (+0.35) — but overfits at ≤11 anchors
   (worse MAE, more harm).
8. **v3.2c**: near-identity model; every mode collapses it to ~raw (MAE 3.50) — uninformative
   beyond confirming safe abstention.

## Conclusion (honest, nuanced)
The false-activation mechanism **is** fixable with a tiny (0.25%) calibration head using
1–2 D4D sessions: raw-good damage and new-Bad3 essentially vanish and selectivity flips
positive. **Hypothesis half-confirmed**: the *activation policy* transfers-with-calibration,
but the *large-error correction skill* is largely suppressed by calibration and only retained
under full fine-tuning, which needs more target data to avoid overfitting. At ≤8 sessions
**no mode clearly beats raw S2M2-S MAE** (raw is already 3.50 on frozen test) — adaptation
makes the refiner **safe** (stops harming) rather than clearly **beneficial**.

**Recommended full-paper sweep**: EGBM-v3-CARE-S, calibration-only vs full, at 8/16/all
sessions, with a selectivity-targeted objective (up-weight >6px gain) to convert safety into
net benefit. Start point: calibration-only (cheapest safe config). v3.2c can be dropped.
