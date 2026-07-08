# ARGOS v2 Causal BiDA Adaptation Plan

## Faithful Causal BiDA

Purpose: isolate the cost/benefit of causalising the official BiDAStabilizer.

Mechanism:

```text
D_{t-1}, flow(t->t-1) -> warp previous disparity into t
previous hidden, flow(t->t-1) -> warp hidden into t
[prev_warped_disp, current_disp, current_disp] -> official local feature extractor
local feature + warped hidden -> official forward PropEnc block
hidden -> official-style residual head -> D_raw + residual
```

Differences from official:

- no future frame;
- no backward propagation;
- no embedded SEA-RAFT;
- ARGOS supplies cached target-to-source flow;
- positive disparity convention;
- optional residual bound and identity-safe zero output init for smoke/testing.

## Safe Causal BiDA

Purpose: add ARGOS v2 surgical safety without hiding the causalisation baseline.

Adds:

- reliability mask input;
- optional RGB/current-vs-warped-previous RGB gate features;
- bounded residual `r_max * tanh(delta)`;
- learned gate `g in [0,1]`;
- final correction `D_ref = D_raw + g * r_max * tanh(delta)`;
- diagnostics for gate, proposal residual, applied residual, warped previous disparity.

## What Is Not Done Yet

- no training script;
- no full SCARED run;
- no D4D/SERV-CT evaluation;
- no official checkpoint conversion;
- no offline bidirectional upper bound.

Those are next-ladder work, not part of this extraction/smoke task.
