# ARGOS v2 Q0 audit

The canonical predeclared audit, candidate contract, masks, targets, split,
metrics, baselines and promotion gates are in
`model_design/QUALITY_PREDICTION_AUDIT.md`.

The completed experiment followed that audit with one documented pilot
adaptation: the best capacity-controlled representation was the shared
pixel-wise Q0-1 encoder, so its otherwise identical heteroscedastic sigma head
was enabled. This 1,155-parameter configuration outperformed the larger Q0-5
Mini U-Net in the seen-only pilot and was frozen before held-out evaluation.

No selector, replacement, blend, residual refiner or unseen-backbone
evaluation was executed. All argmin results are diagnostics only.
