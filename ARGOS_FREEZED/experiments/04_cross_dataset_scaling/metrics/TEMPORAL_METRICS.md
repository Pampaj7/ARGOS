# ARGOS v2 temporal audit metrics

These are evaluation-only diagnostics, never refiner inputs or selection thresholds. For age \(k\), `pull_warp` samples a past field at \(x+F_{t\rightarrow t-k}(x)\), with bilinear interpolation and in-bounds support.

- `GT-TCE_k = mean |(d_t-W(d_{t-k}))-(g_t-W(g_{t-k}))|` on current/warped positive finite benchmark-proxy disparity, flow consistency, and strict method support. `rGT-TCE_k` divides by the same-mask mean \(|g_t|+10^{-6}\). SCARED-C remains processed benchmark-proxy supervision, not clinical GT.
- `NR-TCE_k = trimmed_mean_10% |d_t-W(d_{t-k})|/(0.5(|d_t|+|W(d_{t-k})|)+10^{-6})` on finite positive, bidirectionally supported pixels. It is no-reference only and is reported with \(k/\mathrm{FPS}\).
- `StereoPhoto` is mean Charbonnier RGB residual between left RGB and right RGB pulled at \(x-d\) under the positive-left convention. `Delta StereoPhoto = raw - method`; a lower NR-TCE alone is not success.
- `TemporalPhoto` is the current/past left-RGB pull residual. It stratifies flow reliability and is not disparity accuracy evidence.

Metrics are never reported across unrelated FPS without physical-span labels. Track jitter is intentionally omitted until a validated long-track support contract exists.
