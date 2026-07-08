# ARGOS v2 Next Actions

## Do Immediately

1. Treat `SOTA/ARGOS (5).pdf` as the current source of truth.
2. Do not run the old 18-run NVDS-lite matrix.
3. Fix or add a proper causal streaming/sliding evaluator for NVDS-lite; the current non-overlapping window reset is the main protocol gap.
4. Re-run the causality/gradient checks after any evaluator/model-interface change.
5. Run the minimal one-seed closure: raw, A, D, E, F, optional B.

## Do Not Do

- Do not implement causal BiDA propagation yet.
- Do not implement aligned-local-only before closing NVDS-lite unless the project explicitly abandons NVDS-lite as already insufficient.
- Do not mix stale identity-collapse logs with corrected runs.
- Do not use the D seed0 log-only result as a reusable checkpoint.
- Do not launch jobs from the login host as local detached processes.
- Do not evaluate D4D/SERV-CT until SCARED closure is internally valid.

## Next Command / Job To Run

After the streaming/sliding evaluator exists and validation scripts pass, launch a one-seed closure job through a compute-node/LSF-safe pattern, writing to a new explicit output root such as:

`results/03_temporal_refinement/nvds_lite_causal_pilot/minimal_closure_seed0/`

Configs:

```text
A D E F B(optional)
```

Use corrected defaults:

```text
clip_len=8
seed=0
lam_safe=0.2
lam_sparse=0.02
gate_bias=0.0
batch=4 initially
steps=1200 unless a shorter audited value is chosen before launch
```

## Expected Output

- per-config train logs;
- best checkpoint for each config;
- streaming/sliding validation metrics;
- gate/correction distribution;
- runtime/VRAM;
- compact run manifest.

## Decision After Output

If A does not clearly beat D/E/F without safety regression, close NVDS-lite as `NOT CONFIRMED` and move to aligned-local-only.

If A is positive but marginal, mark `MARGINAL` and do not expand until the result is checked for oversmoothing/lag.

If A clearly wins, promote only the minimal positive subset to three seeds and then consider D4D/SERV-CT decision gates.
