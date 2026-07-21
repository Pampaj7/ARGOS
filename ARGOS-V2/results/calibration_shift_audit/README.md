# ARGOS v2 D0 Calibration Shift Audit

Frozen, forward-only analysis of the SCARED-C-calibrated detector/A2/BiDA
composition. No checkpoint, threshold, model, loss, flow, or dataset asset was
modified. The final authorization remains the SCARED-C `balanced` mode;
`authorization_ultra_safe` is the already frozen SCARED-C ultra-safe mode and
is reported only as a non-tuned sensitivity diagnostic.

The scalar statistics cover **every eligible evaluated pixel**. Multivariate
PCA, t-SNE, nearest-neighbour, AUC/AP, correlation and logistic diagnostics
use a deterministic fixed-seed in-memory per-frame sample to make the audit
compact. No sample tensor or prediction map is persisted. StereoMIS has no
dense GT: it participates in feature-shift/no-reference statistics only.

Decision: **B — feature-space support detector is sufficient**. See `decision_report.md` and
`aggregate_summary.json` for the quantitative basis.
