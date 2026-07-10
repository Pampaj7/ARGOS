# Preliminary Architecture Notes From Component Probes

All conclusions here are preliminary. Use the corrected CSVs for decisions: common-mask cache metrics,
native-grid metrics, coverage sensitivity, and true memory oracle fields are now explicit.

Flow load status: SEA-RAFT=flow_cache_only, RAFT=flow_cache_only.

Mean raw current EPE: 5.974310153722763. Mean aligned previous EPE: 5.972244370977084. Aligned-better frames: 28/72.
Coverage thresholds evaluated: [0.05, 0.25, 0.5, 0.9].
PPMStereo ranking is intentionally not summarized here; tiny-mask minima are not architecture evidence.
Endo status: MambaModel import:pass; MambaModel instantiate:blocked; xLSTMModel import:blocked; temporal_consistency_loss import:pass; temporal_consistency_loss real S2M2 sequence:pass.

1. Which BiDAStabilizer tricks are genuinely useful?
   Preliminary reusable trick: exact `grid + flow`, `align_corners=True` alignment convention plus support and forward-backward consistency signals.
2. Does explicit flow-based alignment improve usable temporal evidence?
   Preliminary: useful evidence, but not a direct replacement for raw disparity in this probe.
3. RAFT or SEA-RAFT?
   Preliminary: compare `raft_vs_searaft.csv`; keep both until the corrected metrics are reviewed.
4. Should flow run online or be precomputed?
   Preliminary: cached low-resolution flow is enough for this probe layer.
5. Does PPMStereo scoring/top-k math evaluated with a deterministic untrained ARGOS feature adapter beat simple recent-frame selection?
   Preliminary only; do not conclude until common-mask, native-grid, coverage-sensitivity, and true-oracle CSVs have been reviewed.
6. Does redundancy-aware selection matter?
   Preliminary only; the invalid global oracle has been removed.
7. Does dynamic memory modulation matter?
   Preliminary only; full flash-attn readout remains reference-only.
8. What K is supported by evidence?
   Preliminary only; no K conclusion until the corrected CSVs are inspected.
9. Is the actual EndoStreamDepth Mamba state useful on ARGOS features?
   Not established because actual blocks are dependency/DPT coupled here.
10. Does it add value beyond explicit memory?
    Unknown; do not include it in the first serious model.
11. Are Mamba state and Pick-and-Play complementary or redundant?
    Undecided; neither earns inclusion from this probe.
12. Which imported mechanisms damage clean predictions?
    Preliminary only; no claim that aligned history or K>1 damages predictions until corrected common-mask/native/oracle results are inspected.
13. Which parts should be directly reused?
    BiDA warp convention, official SEA-RAFT wrapper, and PPMStereo scoring math for ablations.
14. Which parts need adapters?
    PPMStereo needs universal ARGOS feature adapters; EndoStreamDepth needs DPT-token adapters and missing deps.
15. Which parts should be cleanly reimplemented?
    Causal forward-only propagation, support/consistency gating, and bounded identity-preserving residual output.
16. Which parts should remain reference-only?
    Full BiDAStabilizer, PPMStereo flash-attn readout, and EndoStreamDepth Mamba/xLSTM blocks for now.
17. Recommended first ARGOS v2 model?
    Preliminary: causal BiDA-style alignment signals + current raw disparity + support/FB-consistency + safe bounded residual gate; memory depth remains open until corrected CSVs are inspected.
