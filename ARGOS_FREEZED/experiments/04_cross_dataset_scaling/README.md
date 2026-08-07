# ARGOS v2 cross-dataset scaling

This workspace preserves the original planning records and adds a controlled D2 temporal-audit sidecar plus a CPU raw-only SERV-CT static audit. D7, training, threshold tuning, recipe freezing, and dense cache generation remain prohibited. StereoMIS and D4D are fail-closed preflight-only until their own frozen external protocol passes.

SCARED-C is the only registered supervised source: 17 quality-gated processed pseudo-GT sequences / 16,921 frames, causal ages 1/2/4/8 after an 8-frame warmup. StereoMIS is no-reference (3 pilot sequences / 38,241 pairs at 60fps); SERV-CT is 16 static CT-GT pairs; D4D is 362 sparse anchors / 239 usable. Hamlyn and EndoSLAM are locally unavailable/unknown. See `manifests/`, `metrics/`, and the existing registry/audit/split records.
