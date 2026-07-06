# D4D zero-shot refiner findings (0%-target baseline)

156 keyframe anchors (137 with valid GT overlap used for stats), specimen_1/2/3.
Raw S2M2-S vs SCARED-trained refiners applied **zero-shot** (no D4D tuning).

## Primary table (valid+warning anchors)

| model | disp MAE | Bad-1 | Bad-3 | new-Bad3 | harmful rate | % anchors improved | mod. ratio | runtime |
|-------|---------:|------:|------:|---------:|-------------:|-------------------:|-----------:|--------:|
| raw S2M2-S      | **4.09** | 67.7 | 33.2 | 0.0  | 0.00 | —    | 0.00 | — |
| v3.2c           | 4.46 | 73.4 | 41.7 | 14.2 | 0.51 | 15 % | 0.32 | ~13 ms |
| EGBM-v1         | 4.81 | 79.8 | 45.6 | 21.6 | 0.73 | 9 %  | 0.80 | ~6 ms |
| EGBM-v2-CARE    | 4.82 | 80.8 | 45.0 | 21.4 | 0.68 | 9 %  | 0.90 | ~13 ms |
| EGBM-v3-CARE-S  | 4.66 | 76.2 | 44.0 | 19.0 | 0.72 | 9.5 %| 0.62 | ~16 ms |

MPC and CPV are **blocked** at D4D grid resolution (odd height 179; the large-proposal
`cat([f3, mem])` path assumes even dims — a SCARED-256×320 artifact). Documented in
`metrics/blocked_models.csv`; they are secondary large-correction branches, not the main
deployable comparison.

## Answers to the study questions

1. **Raw S2M2-S on D4D**: MAE 4.09 px, Bad-3 33 % — intermediate (worse than SERV-CT 1.28,
   better than SCARED 5.2). S2M2 is decent but not perfect on D4D keyframes.
2. **Zero-shot transfer**: **fails.** Every refiner increases MAE and Bad-3 and only 9–15 %
   of anchors improve.
3. **Improve vs degrade**: harmful correction rate 0.51–0.73; net negative for all.
4. **Consistency**: harm is consistent across all three specimens (specimen_1 −0.59,
   specimen_2 −0.09, specimen_3 −0.85 ΔMAE) — no specimen benefits.
5. **Temporal context / false-activation**: the refiners **false-activate where raw is
   already good**. Raw-error-bin (EGBM-v3-CARE-S):
   - raw error <1 px (49.6 k px): ΔMAE **−0.71** (raw 0.48 → refined 1.19) — damages good pixels.
   - 1–3 px: −0.25; 3–6 px: −0.10 (harms).
   - 6–12 px: **+0.41**; >12 px: **+1.64** (helps large errors).
   The learned correction policy only pays off in the large-error regime it was trained on
   (SCARED ~5 px), and D4D is dominated by good/moderate pixels → net harm.

## Conclusion
This is the **0%-target-data baseline**. SCARED-trained temporal refiners do **not** transfer
zero-shot to D4D: they over-correct raw-good pixels under domain shift. Consistent with the
SERV-CT OOD result on a third dataset with real structured-light GT and true causal context.
**Direct evidence that lightweight domain-specific adaptation is required** — the intended
paper role.
