# ARGOS v2 External Component Audit

Audited on 2026-07-09 for ARGOS v2. Full cloned repositories live under `external/`; minimal export lives under `ARGOS-V2/external_code_backbone_needed/`.

| Repository | Source path | Component | Input | Output | Tensor dimensions | Causal status | Future frames | Backbone dependency | Dependencies | Usefulness for ARGOS v2 | Reuse recommendation | Risks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BiDAVideo | `models/core/bidastabilizer.py` | `flow_warp` | Tensor and optical flow | Warped tensor | `[B,C,H,W]`, flow `[B,2,H,W]` or `[B,H,W,2]` | Causal if called with past-only flow | No by itself | No | PyTorch | Core alignment convention | Clean reimplementation exported | Sign/flow direction must be tested |
| BiDAVideo | `models/core/bidastabilizer.py` | Local disparity feature extractor | aligned prev/current/next disparity | local feature map | input 3 channels, output `mid_channels` | Non-causal in original | Yes, uses `t+1` | No | PyTorch | Good architectural reference | Reimplement causally | Original uses future disparity |
| BiDAVideo | `models/core/bidastabilizer.py` | Forward/backward propagation | local features and flows | hidden states | `[B,C,H,W]` per frame | Non-causal as full block | Yes, backward pass | No | PyTorch, SEA-RAFT | Strong evidence for propagation | Keep forward-only idea | Full module cannot deploy online |
| BiDAVideo | `models/core/bidastabilizer.py` | Fusion/residual output | forward/backward features | disparity residual | `[B,1,H,W]` | Non-causal in original | Yes via backward feature | No | PyTorch | Residual output reference | Reimplement with bounded gate | Residual is unbounded |
| BiDAVideo | `train_utils/losses.py` | `consistency_loss` | RGB sequence, disparities, flow model | temporal loss | `[B,T,C,H,W]`, `[B,T,1,H,W]` | Non-causal training loss | Yes, bidirectional | Flow model | PyTorch, flow model | Useful as metric/loss reference | Adapt motion-aware, causal-safe | Can reward over-smoothing |
| PPMStereo | `models/core/ppmstereo.py` | Q/K similarity | query/key features | similarity score | cost feature tensors over time | Sequence-level | Potentially full sequence | Yes, cost-volume features | PyTorch, flash-attn | Core Pick signal | Clean memory scoring exported | Entangled with full model |
| PPMStereo | `models/core/ppmstereo.py` | Quality-aware top-K | confidence, similarity, redundancy | selected frame indices | scores `[B,1,T,T]` in original | Sequence-level | Potentially full sequence | Yes | PyTorch | Core selective memory idea | Clean top-K exported | Must make past-only in ARGOS |
| PPMStereo | `models/core/ppmstereo.py` | Dynamic memory modulation | selected key/value and scores | weighted key/value | `[B,C,K,H,W]` style | Sequence-level | Potentially full sequence | Yes | PyTorch, flash-attn | Good play-weight reference | Adapt cleanly | Cost feature assumptions |
| PPMStereo | `models/core/ppmstereo.py` | Memory read-out | query and selected memory | aggregated feature | flattened attention tokens | Sequence-level | Potentially full sequence | Yes | flash-attn | Conceptual reference | Reference only | Heavy dependency |
| EndoStreamDepth | `endostreamdepth/mamba.py` | `MambaModel.start_new_sequence` | none | reset state | per temporal layer | Causal | No | DPT-coupled model context | mamba-ssm | State lifecycle reference | Clean state helper exported | Heavy Mamba stack |
| EndoStreamDepth | `endostreamdepth/mamba.py` | `forward_single_frame` | one feature frame | updated feature/state | `[B,L,C]` | Causal | No | DPT feature dims | mamba-ssm | Streaming operator reference | Reference only initially | Dependency complexity |
| EndoStreamDepth | `endostreamdepth/model.py` | `dpt_features_to_mamba` | DPT features | temporal DPT features | `(B*T,C,h,w)` | Causal loop | No | DepthAnything/DPT | PyTorch, einops | Multi-scale insertion pattern | Reimplement only if needed | Monocular/depth-specific |
| EndoStreamDepth | `endostreamdepth/util/loss.py` | SiLog/edge/temporal losses | depth predictions and GT | loss scalars | depth `[B,H,W]` | Training only | No | Monocular depth | PyTorch | Loss references | Reference only | Depth units differ from disparity |
| SEA-RAFT | `core/utils/utils.py` | `InputPadder`, `load_ckpt`, resize helpers | image/flow tensors | padded/unpadded tensors | `[B,C,H,W]` | Causal pairwise | No | Flow model | PyTorch | Default flow wrapper ingredients | Wrapper later | Requires checkpoint |
| SEA-RAFT | `demo.py`, `custom.py` | inference pattern | image pair | flow and info | flow `[B,2,H,W]` | Causal pairwise | No | SEA-RAFT | PyTorch, OpenCV | Preferred flow default | Wrapper | Checkpoint and optional pretrained encoder |
| RAFT | `core/utils/utils.py` | `InputPadder`, `bilinear_sampler`, `coords_grid` | tensors | padded/sampled tensors | `[B,C,H,W]` | Causal pairwise | No | RAFT | PyTorch, scipy | Reference flow utilities | Reference only | Older baseline |
| RAFT | `demo.py` | inference pattern | image pair | flow | flow `[B,2,H,W]` | Causal pairwise | No | RAFT | PyTorch | Compare against SEA-RAFT | Wrapper only if needed | Requires checkpoint |

## Final Decision Report

1. Directly reusable: `causal_warp.py`, `pick_and_play.py` for prototype memory selection.
2. Needs a wrapper: SEA-RAFT inference with explicit checkpoint path; RAFT only for comparison.
3. Should be adapted: BiDAStabilizer forward propagation into causal forward-only state; PPMStereo memory scoring into past-only universal-feature scoring.
4. Conceptual reference only: full PPMStereo, full EndoStreamDepth, full BiDAStabilizer deployable path.
5. Clean reimplementation: bounded residual head, identity-preserving gate, causal selective memory over ARGOS cache signals.
6. Initial temporal operator: ConvGRU/gated recurrent state plus explicit top-K memory; add Mamba only after the minimal model plateaus.
7. Pick-and-Play memory and Mamba state are complementary only if separated by role: Pick-and-Play chooses evidence, Mamba/GRU updates state. Starting with both full-size is unnecessary.
8. Default optical flow: SEA-RAFT, with RAFT as reference comparator.
9. Non-causal components that must not enter deployable ARGOS v2: BiDAStabilizer future-frame local alignment, backward propagation, bidirectional consistency as inference logic, PPMStereo full-sequence memory.
10. Minimal first prototype: cached raw disparity + valid masks + target-to-source flow + causal warp + top-K past memory + small gated state + bounded residual output.
