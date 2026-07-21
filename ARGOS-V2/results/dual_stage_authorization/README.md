# ARGOS v2 — Dual-stage authorization

## Outcome

**NO-GO under the predeclared promotion gate.** The frozen P4 utility veto is
useful as a conditional filter, but it is not yet safe enough to promote. It
retains 99.31% of the frozen Raw Error authorization gain and improves every
seen backbone, while reducing false updates by 19.3% and clean degradation by
26.2%. Nevertheless, final held-out SCARED-C remains above all three strict
safety limits: 1.531% false updates (target <1.25%), 0.726% clean degradation
(target <0.60%), and 79.46% intervention precision (target >80%).

The protocol therefore blocked Fast-FoundationStereo, CREStereo, and all OOD
diagnostics. No unseen or OOD result was inspected and no threshold was changed
after final-test opening.

## Frozen cascade

The selected C2 rule was fitted on `dataset_7_keyframe_1/2` only:

```text
veto       = raw_error_authorized AND (P4 predicted utility <= 0 px)
authorized = raw_error_authorized AND NOT veto
output     = where(authorized, frozen_A2, raw)
```

P4 cannot open an intervention. Rejected pixels are the raw tensor bit-exactly;
accepted pixels are the frozen A2 tensor bit-exactly. C1 uses a 2.0-px maximum
update veto. C3 was not run because C1 uniquely recovered only 0.42% of harmful
validation proposals, below the predeclared 5% complementarity threshold.

## Final held-out SCARED-C

Primary results use cache-grid disparity at width 180, GT coverage >0.50,
pixel-weighted aggregation, and one identical paired mask. There are 7,200,246
valid comparison pixels across 960 frames, three seen backbones, and
`dataset_7_keyframe_3/4`.

| Method | EPE | Gain vs raw | Coverage | Precision | False update | Clean degradation |
|---|---:|---:|---:|---:|---:|---:|
| Raw | 0.961605 | 0 | 0% | — | 0% | 0% |
| A2 unconditional | 0.918975 | 0.042630 | 39.96% | 61.87% | 34.14% | 12.96% |
| C0 Raw Error | 0.934660 | 0.026946 | 5.62% | 75.63% | 1.90% | 0.98% |
| P4 standalone prior | 0.948138 | 0.013467 | 3.18% | 84.11% | 1.19% | 0.40% |
| C1 magnitude veto | 0.946923 | 0.014682 | 5.07% | 74.35% | 1.88% | 0.97% |
| **C2 P4 utility veto** | **0.934845** | **0.026760** | **4.72%** | **79.46%** | **1.53%** | **0.73%** |
| Random matched veto | 0.940068 | 0.021538 | 4.49% | 75.65% | 1.52% | 0.79% |
| Oracle conditional veto | 0.929596 | 0.032009 | 4.66% | 91.38% | 1.15% | 0.23% |

C2 costs only 0.000186 px EPE relative to C0 while retaining 84.0% of C0
coverage. On the sampled conditional audit it rejects 27.42% of harmful
proposals and retains 91.52% of helpful proposals. Its matched-random baseline
is materially worse in EPE, showing that P4 contains real conditional signal.

| Backbone | Raw EPE | C0 EPE | C2 EPE | C2 precision | C2 false update |
|---|---:|---:|---:|---:|---:|
| S2M2-S | 1.028950 | 1.009592 | 1.009830 | 78.41% | 1.54% |
| RAFT-Stereo | 0.925869 | 0.897324 | 0.897360 | 79.44% | 1.59% |
| StereoAnywhere | 0.929996 | 0.897062 | 0.897347 | 80.53% | 1.46% |

All three backbones improve over raw and no catastrophic frame was observed,
but the safety gate fails in aggregate. Worst-frame degradation is 0.03808 px;
3.125% of frames worsen.

## Conditional validation audit

Among 25,187 nontrivial Raw Error-authorized validation samples, 56.91% are
helpful, 32.00% harmful, and 11.09% indifferent at epsilon 0.10 px. Mean utility
is +0.1819 px; update magnitude and utility correlate at +0.4367. The selected
C2 veto has 40.78% harmful recall, 65.21% harmful-veto precision, and retains
94.68% of useful proposals on validation. No candidate met every selection
constraint, so C2 was frozen as an explicitly ineligible highest-gain safety
Pareto point before final test.

## Runtime

P4 has 15,533 frozen parameters and costs 0.132 ms/frame on H100. Selected
cascade logic is approximately 0.011 ms/frame. The complete frozen proposal and
authorization stack contains 55,939 parameters; total trainable parameters are
zero. Peak allocated GPU memory for the evaluation path was 625,241,088 bytes.
The 72.3 ms/frame wall time includes SEA-RAFT, data I/O, all baselines, metrics,
and four coverage thresholds; it is not the latency of only the selected path.

## Files

- `frozen_manifest.json`: hashes, frozen policies, split state, promotion state.
- `threshold_selection.csv`: complete compact predeclared C1/C2 grid.
- `conditional_error_analysis.csv`: validation conditional distribution.
- `policy_comparison_validation.csv`: selected validation policies.
- `frame_metrics.csv`, `sequence_metrics.csv`, `per_backbone.csv`: final seen results.
- `risk_gain_coverage.csv`: diagnostic conditional retention curve.
- `aggregate_summary.json`, `safety_summary.json`, `runtime_summary.json`.
- `verdicts.json`: gate-by-gate decision and evaluation blockade.

No dense disparity, flow, or feature cache and no checkpoint were written.
Contact sheets were omitted because this experiment tests a logical policy and
the quantitative promotion gate failed before qualitative OOD diagnostics.

