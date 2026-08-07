# ARGOS v2 paper status

The final geometric method is the frozen immutable raw multi-anchor refiner. The bounded CODD-style H4 corrected-memory model remains the architectural baseline: short corrected recurrence helps, whereas indefinite recurrence drifts. Oracle analysis motivated retaining multiple immutable temporal ages; the learned shared retrieval and pairwise fusion realize part of that oracle opportunity without recurrent writeback.

The frozen model achieved FULL UNSEEN-BACKBONE GEOMETRY GO within SCARED-C on CREStereo and Fast-FoundationStereo, improving raw and H4 on all four sequences for both backbones while continuing to use CS4/CS8 substantially. The post-hoc hard-negative spatial critic remains a validation NO-GO for strict safety (about 20.1% harmful accepted updates at 2.06% coverage, +0.00598 EPE gain) and is excluded from geometry-v1. External-domain geometric evaluation remains unresolved.

Remaining paper work:

- multi-seed replication;
- unified five-backbone table;
- runtime breakdown;
- final architecture figure;
- recurrence-drift figure;
- oracle-realization figure;
- manuscript consolidation.
