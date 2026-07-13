# ARGOS v2 Model Design

This folder owns reusable model-design logic for ARGOS v2.

Keep this separate from:

- `scripts/`: cache generation and dataset plumbing.

Current contents:

- `external_components/`: thin, source-attributed adapters for external mechanisms that may enter the model design (bidavideo, ppmstereo, ppmstereo_candidate_scoring, endostreamdepth, sea_raft).
- `BIDAVIDEO_AUDIT.md`: source-level audit of BiDAStabilizer, flow/warp conventions, non-causal paths, and ARGOS v2 adaptation decisions.
- `tests/test_bidavideo.py`: deterministic original-equivalence, geometry, mask, consistency, gradient, and frozen-flow tests.
- `models/learned_t1_refiner.py`: 39k-41k parameter causal CNN with separate raw-error, memory-trust, and bounded-correction heads.
- `losses/safety_losses.py`: geometry, selector supervision, clean preservation, safety ranking, and update regularization.
- `data/temporal_pair_dataset.py`: balanced mmap cache pairs with sequence-held-out splits and coverage-normalized GT resizing.
- `tests/test_learned_t1_refiner.py`: causal data, identity, bound, frozen-input, gradient, resize, and determinism tests.

Rule of thumb: if code is meant to be imported by a future ARGOS v2 model, it belongs here.
