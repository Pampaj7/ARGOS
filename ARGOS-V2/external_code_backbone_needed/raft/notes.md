# RAFT Notes

Source: `external/RAFT` at commit `2888e15a51fa41140771d3f498ed8023cff098d1`.

Relevant files:

- `demo.py`
- `core/raft.py`
- `core/utils/utils.py`

Useful components:

- `InputPadder`
- `bilinear_sampler`
- `coords_grid`
- `upflow8`
- checkpoint loading pattern in `demo.py`

ARGOS v2 action:

- Reference optical-flow implementation.
- Use for comparison with BiDAStabilizer/SEA-RAFT behavior.
- Prefer SEA-RAFT by default unless it fails.

Risks:

- Requires checkpoint.
- Older implementation and optional CUDA correlation path.
