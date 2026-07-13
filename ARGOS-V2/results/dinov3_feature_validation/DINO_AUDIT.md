# DINOv3 audit for ARGOS v2

## Scope and provenance

This audit covers `SOTA/dinov3.pdf`, official repository
`external/dinov3` at commit `346f38fee679c56a6888f91c51670fae61d364e0`,
and the local ViT-L/16 LVD-1689M checkpoint. The checkpoint is 1,213,050,671
bytes and has SHA-256
`8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035`.
No network loading is permitted by the ARGOS wrapper.

The PPMStereo and BiDAVideo audits remain authoritative for memory selection
and temporal alignment respectively. DINOv3 is studied only as a frozen RGB
representation; it is neither a stereo estimator nor a replacement for BiDA.

## Official architecture and loading path

- `external/dinov3/dinov3/hub/backbones.py::dinov3_vitl16` calls
  `_make_dinov3_vit`, constructs the official model, loads the supplied state
  dictionary with `strict=True`, and recognizes the `8aa4cbdd` LVD checkpoint.
- The exact architecture is patch size 16, embedding width 1024, 24 transformer
  blocks, 16 attention heads, MLP ratio 4, four storage/register tokens, RoPE
  coordinates normalized separately by axis, and 300M parameters (303,154,176
  in the released implementation).
- `external/dinov3/dinov3/models/vision_transformer.py::prepare_tokens_with_masks`
  orders tokens as CLS, four storage tokens, then row-major image patches.
- `DinoVisionTransformer.get_intermediate_layers` accepts explicit zero-based
  block indices, normalizes outputs, removes CLS/storage tokens, and with
  `reshape=True` returns `[B,1024,H/16,W/16]`.
- The official LVD transform in `external/dinov3/README.md` converts RGB to
  `[0,1]` and applies ImageNet mean `(0.485,0.456,0.406)` and standard deviation
  `(0.229,0.224,0.225)`.

ARGOS wraps this path in
`model_design/external_components/dinov3.py::FrozenDINOv3`. It verifies the
checkpoint hash and architecture, freezes every parameter, forces evaluation
mode even when a parent module trains, and uses `torch.inference_mode` by
default. BF16 CUDA autocast is configurable; outputs remain normalized official
patch features.

## Paper evidence relevant to the representation study

The paper introduces Gram anchoring and high-resolution adaptation to retain
localized, consistent patch features while scaling self-supervised training.
Its dense probes use frozen patch features, normally from the last layer, and
show strong semantic segmentation, monocular-depth, multi-view correspondence,
and video tracking results. It also reports that higher image resolution yields
finer feature maps and that the models operate across varying resolutions.

These results motivate, but do not answer, the ARGOS question. Semantic or
multi-view correspondence does not imply that a feature can identify whether a
warped cached stereo disparity is more accurate than the current disparity.
Therefore layer choice, resolution, ranking, raw/null abstention, calibration,
and safety require controlled SCARED-C measurements. Attractive PCA maps are
not promotion evidence.

## Geometry and tensor contract

Native SCARED-C left RGB is 1024x1280 (aspect 4:5). The controlled resolutions
are 256x320 and 320x400, both exact 4:5 and divisible by 16; no crop, pad, or
stretch is used. Their patch grids are 16x20 and 20x25. The generic wrapper can
fit-and-pad other aspect ratios and records the resize and padding metadata.

For current frame `t` and past frame `t-k`, DINO features are extracted
independently. The validated current-to-past BiDA flow is resized with
`model_design.external_components.bidavideo.resize_flow`, including separate
x/y component scaling, then passed to the canonical
`bidavideo.causal_warp`. Thus current patch locations sample past patches using
BiDA's `grid + flow`, zero padding, and `align_corners=True`. ARGOS does not
invent a second token-warp convention.

## Layer protocol

The controlled initial indices are block 5 (early/intermediate), 11 (middle),
23 (late), and multi-layer `(5,11,17,23)`. Every variant is projected to the
same compact descriptor width before the shared selector/ranker. The final layer
is included because the paper's dense probes use it, but it is not selected by
assumption.

## Relation to PPMStereo

Faithful PPMStereo keys come from learned stereo context features, while values
and quality depend on cost volumes, recurrent state, and its confidence network
(`model_design/PPMSTEREO_AUDIT.md`). Frozen DINO patch maps cannot reproduce
those internals. In ARGOS they are universal query/key context used alongside
aligned disparity, validity, flow consistency, photometric residual, and age.
The causal bank, redundancy mechanics, normalized play weights, and top-K logic
can remain PPMStereo-inspired, but the resulting learned selector is explicitly
an ARGOS adapter, not the full PPMStereo selector.

The raw/null option is mandatory. A softmax over `{raw,t-1,t-2,t-4,t-8}` may
abstain rather than forcing memory mass. Fast-FoundationStereo is excluded from
training, validation, layer/resolution choice, loss/threshold tuning, and
checkpoint selection.

## Cache policy and promotion gate

ViT-L patch maps are expensive: one FP16 1024-channel frame is approximately
0.625 MiB at 16x20 and 0.977 MiB at 20x25 per layer. Four layers multiply this
by four before metadata/filesystem overhead. Only the controlled probe subset is
computed initially. A resumable mmap cache may be built after measuring feature
quantization and projected storage, with frame IDs, source sequence, layers,
resolution, dtype, checkpoint hash, repository commit, normalization,
interpolation, atomic completion, and integrity metadata.

DINO is promoted only if controlled ranking or abstention improves materially
without relying on a larger trainable decoder, and this translates into safer
bounded correction. Long-memory integration is gated behind a successful t-1
safety result; unseen Fast-FoundationStereo is accessed only after all choices
are frozen.
