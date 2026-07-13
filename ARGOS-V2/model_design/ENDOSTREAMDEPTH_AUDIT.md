# EndoStreamDepth audit for ARGOS v2

## Scope and provenance

This audit covers `SOTA/endostream.pdf` (MIDL 2026 submission, arXiv
2512.18159v2) and `external/EndoStreamDepth` at commit
`5abe89d9c0e09f64fdc5276d21bb5a34aa815cc6`. EndoStreamDepth is a monocular
DINOv2/DPT depth network. ARGOS v2 reuses only the scientific mechanism of
persistent causal multi-level state; it does not reuse monocular depth, DPT,
DepthAnything, or backbone tokens.

Three names are kept distinct throughout:

1. **faithful EndoStreamDepth**: released DPT decoder plus Mamba2/xLSTM state;
2. **ARGOS latent-state adaptation**: explicit state over universal stereo
   refinement evidence;
3. **generic gated-state baseline**: the ConvGRU operator used for E2-E5. It is
   not called Mamba or the full EndoStreamDepth architecture.

## Paper mechanism

At frame `t`, the paper encodes RGB `I_t`, combines the current feature with
hidden state `H_{t-1}`, applies a Mamba module, decodes the refined feature, and
stores `H_t` for frame `t+1`. Frames are processed sequentially, not jointly.
The hierarchical version inserts a separate temporal module at four DPT decoder
levels, finest to coarsest. Each module contains four Mamba blocks and every SSM
layer owns its recurrent state.

Training uses windows of five frames, batch size four, and four equally weighted
multi-scale SiLog losses plus log-depth metric loss, log-gradient edge loss, and
0.01 times a self-supervised temporal smoothness loss. The latter normalizes all
valid predicted depths in a video window by one median and mean absolute
deviation, then penalizes same-pixel differences between consecutive frames. It
does not use optical flow, so motion/deformation can be penalized as flicker.

The paper claims arbitrarily long streaming at inference, but training uses
five-frame windows. It does not specify selective batch reset, state detach,
truncated BPTT, serialization, scene-change forgetting, or sequence-ID checks.

## Released files and exact mechanics

### Network and multi-level insertion

- `external/EndoStreamDepth/endostreamdepth/model.py::EndoStreamDepth` builds a
  DINOv2 encoder, `original_dpt.py::DPTHead`, and one selected temporal operator.
- `EndoStreamDepth.dpt_features_to_mamba` reshapes decoder maps
  `[B*T,C,h,w] -> [B,T,h*w,C]`, loops over `T`, calls
  `forward_single_frame`, and stacks outputs back to `[B*T,C,h,w]`.
- `original_dpt.py::DPTHead.forward_with_mamba` inserts temporal processing
  after `path_4`, `path_3`, `path_2`, and `path_1`. The configured
  `mamba_in_dpt_layer=[0,1,2,3]` therefore uses all four decoder levels.
- DPT starts from patch-14 tokens. Its four encoder maps are resized by factors
  4, 2, 1 and 1/2, then fused coarse-to-fine. At C3VD resolution 518, released
  output supervision is 518, 259, 130 and 65 pixels. The default temporal input
  at every level is additionally average-pooled by `downsample_mamba=0.1`.
  Consequently the actual SSM token grid is configuration- and rounding-
  dependent, not simply the supervised output pyramid.
- Current decoder features are the only input to the original state. Cached
  disparity, optical flow, validity and stereo confidence do not exist there.

### Mamba state

- `endostreamdepth/mamba.py::MambaBlock` is LayerNorm, released `Mamba2`, a
  residual addition, then LayerNorm/MLP/residual.
- `mamba.py::MambaModel` constructs a separate four-block stack for each DPT
  insertion level. Add/modulation output projections are zero initialized.
- `mamba.py::InferenceParams` stores `seqlen_offset` and
  `key_value_memory_dict`; Mamba2 lazily populates convolution and SSM caches by
  layer index.
- `MambaModel.start_new_sequence` replaces all `InferenceParams` objects.
  `forward_single_frame` mutates these caches and advances `seqlen_offset` by
  the number of spatial tokens, not by one video frame.
- No sequence IDs are stored. Reset is all-or-nothing; selected batch elements
  cannot reset independently. State is neither exposed nor serialized by the
  EndoStreamDepth API.

### xLSTM state

- `endostreamdepth/xlstm_block.py::xLSTMModel` imports
  `xlstm.xlstm_large.model`, embeds DPT channels to the configured state width,
  calls `xLSTMLarge.backbone(x,self.state)`, and assigns the returned state to
  `self.state`.
- `start_new_sequence` sets `self.state=None`. Training pads spatial-token
  sequence length to a multiple of 64. Inference requests BF16 state and Triton
  sequence/step kernels.
- As with Mamba, state is hidden, has no sequence identity, selective reset,
  detach API, serialization contract, or variable-batch validation.

### Other recurrent reference

- `endostreamdepth/rnn_transformer.py::TransformerRNN` maintains a learned
  fixed-size token state. Each layer writes frame tokens into state with cross
  attention and reads state back into frame tokens. The source contains a TODO
  noting that it does not maintain a distinct state per layer. It is a repository
  ablation, not the paper's Mamba operator and not used as ARGOS E2-E5.

### Sequence lifecycle and causality

- `model.py::train_sequence` resets once per input clip, flattens RGB for the
  DINO encoder, then processes decoder features in chronological Python order.
- `model.py::forward` resets once per video and loops over frames in order.
- Neither path reads future frames in its recurrent operator. The training
  encoder processes `[B*T]` together, but it is frame-independent DINOv2; the
  temporal state still advances only from earlier to later indices.
- The source never calls `detach` on temporal state inside a clip. Therefore
  the intended training graph spans the complete five-frame window. A new clip
  reset truncates history. The repository does not implement longer TBPTT or
  configurable detach intervals.
- Calling `forward_single_frame` directly without `start_new_sequence` silently
  leaks state across videos. Batch-size changes and reordered sequences are not
  guarded.

## Loss implementation versus paper

- `model.py::train_sequence` applies `SiLogLoss`, finest-level log L1 and
  `GradientEdgeLoss`; all four SiLog levels have weight one.
- `endostreamdepth/util/loss.py::temporal_consistency_loss` implements the
  paper's per-window median/MAD normalization and consecutive same-coordinate
  L1. It is evaluated at the finest output and multiplied by 0.01.
- The code resets `loss_temp` inside the scale loop and only computes it in the
  `lvl == 0` branch, consistent with finest-scale intent but less clearly
  structured than Eq. 6.
- This temporal loss is unsuitable unchanged for ARGOS: it can reward frozen
  disparities under real camera/tissue motion. ARGOS retains supervised
  geometry and safety losses and does not add an unwarped smoothing objective.

## Dependency audit and E6 status

The repository setup pins PyTorch 2.4, torchvision 0.19, xformers 0.0.27,
FlashAttention, and builds a local Mamba package with CUDA extensions. ARGOS
currently uses PyTorch 2.5.1+cu121. `mamba_ssm`, `causal_conv1d`, `xlstm` and
`xformers` are not installed. Importing the vendored Mamba with `PYTHONPATH`
fails on missing `selective_scan_cuda`; xLSTM fails on missing `xlstm`.

Installing or rebuilding these tightly coupled CUDA/Triton packages would
mutate the validated ARGOS environment and is not justified before the generic
state ladder establishes value. E6 is therefore **blocked/reference-only**.
No ConvGRU or custom SSM will be mislabeled as Mamba/xLSTM.

## Reuse decision for ARGOS v2

Directly reusable scientific mechanics:

- strict one-frame-at-a-time causal update;
- reset at video boundaries;
- state at several decoder/evidence scales;
- zero-initialized temporal residual injection;
- finite-window BPTT followed by explicit state detach/reset.

Requires a clean ARGOS adapter:

- explicit immutable state object rather than module-global caches;
- sequence IDs, frame indices and crossing checks;
- selected-element reset, detach, serialization and statistics;
- universal disparity/validity/BiDA inputs;
- 1/4, 1/8 and 1/16 evidence states with documented shapes;
- reliability-driven forgetting and identity-preserving bounded disparity
  correction.

Reference-only:

- DINOv2 and DPT encoder/decoder;
- monocular metric-depth heads and multi-scale depth supervision;
- same-pixel temporal smoothing loss;
- Mamba2/xLSTM CUDA/Triton operators until their exact dependencies are
  available in a controlled environment;
- hidden global inference cache and unrestricted cross-sequence reuse.

The ARGOS E2-E5 operator is consequently a controlled generic ConvGRU baseline
inspired by EndoStreamDepth's causal, hierarchical state placement—not a claim
to reproduce the full model.
