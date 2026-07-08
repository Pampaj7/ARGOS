# ARGOS v2 Shuffled-History Semantics Audit

The existing evaluator's `shuffled_history` mode is not a clean history ablation: it shuffles `previous_raw`, sets flow to zero, but leaves the hidden state built from the chronological past. That mixes corrupted local evidence with correct persistent state.

This diagnostic therefore reports both `faithful_shuffled_existing` and `faithful_shuffled_corrected`, where the previous raw is causally shuffled and the hidden state is reset for that step. No future frames are introduced.
