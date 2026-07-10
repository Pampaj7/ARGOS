# SEA-RAFT Notes

Source: `external/SEA-RAFT` at commit `9137517ba24e628442aec097d3afe71d03503b75`.

Relevant files:

- `custom.py`
- `demo.py`
- `core/raft.py`
- `core/utils/utils.py`

Useful components:

- `InputPadder`: pad/unpad to stride-compatible shape.
- `load_ckpt`: checkpoint loading.
- `calc_flow`: scale-aware inference helper in `demo.py`.
- Output dict contains `flow` and `info`; `info` can be explored as uncertainty/confidence.

ARGOS v2 action:

- Default optical-flow implementation.
- Wrap external repo with explicit checkpoint path.
- Do not copy full model into ARGOS v2.

Risks:

- Requires external checkpoint.
- `torchvision` may download ImageNet backbone weights if not cached.
