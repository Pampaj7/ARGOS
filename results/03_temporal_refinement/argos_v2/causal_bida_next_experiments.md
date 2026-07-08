# ARGOS v2 Causal BiDA Next Experiments

Minimal ladder, not launched in this task:

1. Raw S2M2.
2. Current-only bounded refiner.
3. Aligned-local-only.
4. Faithful causal BiDA.
5. Faithful causal BiDA with state reset every frame/window.
6. Faithful causal BiDA with shuffled history.
7. Safe causal BiDA.
8. Optional offline official bidirectional upper bound.

## First Run Recommendation

Run one seed on SCARED only, with the same sequence-disjoint split and a true streaming evaluator. Promote to three seeds only if faithful causal BiDA beats aligned-local-only/current-only/shuffled without geometry or New-Bad3 regression.

## Stop Rule

If faithful causal BiDA does not beat aligned-local-only, do not add more safety machinery yet. The BiDA propagation hypothesis failed for this backbone/data setting.
