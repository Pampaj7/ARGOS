# Source provenance

Port source root: `/dtu/p1/leopam/ARGOS/ARGOS-V2/model_design`.

| Source path | SHA-256 | Port destination |
| --- | --- | --- |
| `models/raw_multi_anchor_refiner.py` | `ee58f36744c8174d0bdf439084c0ce616220e090120f517d0603df21dd75953b` | `src/argos_v2_hand/raw_multi_anchor.py` |
| `losses/raw_multi_anchor_losses.py` | `6a77cfbdec84d161c7146a332a5aa73a36f024b43664b014fcc5f19012c3f53b` | `src/argos_v2_hand/losses.py` |
| `models/codd_style_fusion.py` | `4b1a22c638214371055012087f93e9faf8a3b087a40a0b969341f33e05187a49` | `src/argos_v2_hand/codd.py` |
| `models/codd_bounded_memory.py` | `53a7468a93a78009d294eb0cfc30be825d4f6f7a726d2abcdc2822d1730f5b04` | `src/argos_v2_hand/state.py` |
| `losses/codd_fusion_losses.py` | `6558e6d7611ca43be9f10333fb9dcc2e851a7e6aaca46ad856ebf468756c2f0f` | `src/argos_v2_hand/losses.py` |
| `external_components/bidavideo.py` (tensor-only causal helpers) | `133a13f8a4dd89065f736484f1dba1811b40e0f1272d0bbec87d74074bf5c530` | `src/argos_v2_hand/alignment.py` |
| `external_components/stereo_photometric.py` | `7ab78bead1b38478ec61a1923101b7f421e4e475b184107a1758617ec9bb7924` | `src/argos_v2_hand/stereo.py` |

Imports were rewritten for package-local modules. External flow inference/checkpoint adapters were deliberately excluded; callers supply flow tensors. Full H4 cue construction requires a caller-supplied frozen ResNet-18 checkpoint via `FrozenResNet18Layer1(checkpoint=Path(...))`; this package carries no checkpoint.
