# PPMStereo Notes

Source: `external/PPMStereo` at commit `d0ccf7705145502c1eea49e7be0ddeafbcfd6a08`.

Relevant files:

- `models/core/ppmstereo.py`
- `models/core/ppmtereo_update.py`

Useful components:

- `compute_qk_similarity`: query/key cosine similarity for memory relevance.
- Quality-aware memory assessment around `models/core/ppmstereo.py:504`.
- Top-K selection around `models/core/ppmstereo.py:509`.
- Dynamic memory modulation around `models/core/ppmstereo.py:540`.
- FlashAttention read-out around `models/core/ppmstereo.py:550`.
- Temporal positional encoding in `models/core/ppmtereo_update.py`.

ARGOS v2 action:

- Reuse the scoring idea only.
- Keep memory causal and past-only.
- Use universal signals, not PPMStereo cost volumes.

Risks:

- Full model is backbone/cost-volume-specific.
- Full implementation is not a simple disparity plugin.
- Official code has RAFT/RAFT-Stereo submodules and checkpoint assumptions.
