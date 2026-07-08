# ARGOS v2 Causal BiDA Next Decision

FaithfulCausalBiDA: `TEMPORAL_SMOOTHING_WITHOUT_STATE_USE`. SafeCausalBiDA: `WARM_START_RECOMMENDED`.

Do not run three seeds yet. First fix SafeCausalBiDA by warm-starting from the faithful checkpoint and verify whether state contribution remains negligible. If it does, promote aligned-local as the confirmed baseline and treat persistent propagation as unproven.
