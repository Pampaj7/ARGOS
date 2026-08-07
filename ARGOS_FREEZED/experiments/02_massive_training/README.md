# ARGOS v2 paper-grade training-budget campaign

This campaign changes only total optimization budget for the frozen `geometry_v1` architecture. Paired seeds are 20260722, 20260723, and 20260724; budgets are 1x/3x/6x = 10/30/60 epochs. Dataset 7 is held out from the frozen training and validation protocol of the campaign and remains fail-closed until the D2-only recipe is frozen.

All training code is experiment-local and imports `argos_freezed`, never ARGOS-V2 at runtime. ARGOS-V2 is used only by the pre-launch provenance/initialization audit. Stereo caches are reused; SEA-RAFT is frozen and inferred live. Evidence remains in RAM and is not persisted as nine dense caches.
