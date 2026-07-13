# PPMStereo audit for ARGOS v2

## Sources and scope

This audit covers the NeurIPS 2025 paper `SOTA/NeurIPS-2025-ppmstereo-pick-and-play-memory-construction-for-consistent-dynamic-stereo-matching-Paper-Conference.pdf` and repository `external/PPMStereo` at commit `d0ccf7705145502c1eea49e7be0ddeafbcfd6a08`.

The paper describes Pick-and-Play at the level of a long video memory. The released implementation evaluates all frames in an input clip jointly; it is not a persistent causal streaming memory bank. ARGOS v2 therefore distinguishes three things throughout:

1. **PPM-faithful mechanics**: equations and executable operations retained exactly where their inputs exist.
2. **ARGOS universal memory adapter**: a causal bank and scores built only from RGB, cached disparity, flow, validity and derived evidence.
3. **Learned ARGOS selector**: a small model trained to predict memory usefulness. It is not the original QAM.

## Exact implementation map

- `external/PPMStereo/models/core/ppmstereo.py::PPMStereo.forward` extracts stereo/context features, builds correlation volumes and performs multi-scale recurrent refinement.
- `PPMStereo.forward_update_block` contains quality assessment, top-K selection, strive-time updates, dynamic key modulation, FlashAttention read-out and residual-disparity updates.
- `PPMStereo.compute_qk_similarity` computes the released inter-frame similarity.
- `external/PPMStereo/models/core/ppmtereo_update.py::Attention_qk` projects context features to query/key.
- `ppmtereo_update.py::SequenceUpdateBlock3D.get_motion_and_value` obtains memory values from correlation/flow motion features.
- `SequenceUpdateBlock3D.get_uncertainty` is the two-convolution sigmoid confidence network.
- `ppmtereo_update.py::get_temporal_positional_encoding` provides sinusoidal temporal encoding.
- `ppmtereo_update.py::Aggregate` provides the value projection and a learnable residual scale `beta`, initialized to zero.
- `external/PPMStereo/models/ppm_stereo_model.py::PPMStereoModel` loads a checkpoint and evaluates overlapping clips with `kernel_size=20` by default.
- `external/PPMStereo/train_utils/losses.py::sequence_loss` implements training losses and metrics.
- `external/PPMStereo/models/core/ppmstereo_VDA.py::PPMStereo` repeats the PPM block while replacing/augmenting image representations with Video Depth Anything features. It does not change the core Pick-and-Play equations.

## What a faithful memory entry contains

At a given recurrent scale, a frame contributes:

- a key `k_t`, projected from the left-image context tensor `inp` by `Attention_qk.to_qk`;
- a value `v_t`, projected by `SequenceUpdateBlock3D.aggregator.to_v` from motion features derived from the current disparity/flow estimate, correlation-volume lookup and recurrent motion state;
- a sigmoid confidence map produced from concatenated recurrent state `net` and current value;
- a temporal positional encoding;
- an iteration-local selection count (`strive_time`).

The released code materializes query/key as `[B,C,T,H,W]`, similarity and selection masks as `[B,1,T,T]`, and values as `[B,C,T,H,W]`. It does not store RGB or a disparity map as the memory value and does not expose a persistent `MemoryEntry` object.

## Pick: quality, similarity, redundancy and top-K

### Paper definition

The confidence target is `exp(-|d_pred-d_gt|/sigma)` with `sigma=5`; spatial pooling produces frame confidence `S_c`. The paper average-pools and L2-normalizes query/key features, then computes cosine similarity `sim`. A selection-count regularizer `R[k]=exp(-t_k/T)` downweights repeatedly selected frames. The relevance score is `S_r=R*sim`, total quality is `S=S_c+S_r`, and the largest `K` scores are retained.

### Released-code definition

`PPMStereo.compute_qk_similarity` differs slightly from the prose: it uses `AdaptiveMaxPool2d((H//4,W//4))`, averages channels, flattens the pooled spatial map and calls `F.cosine_similarity`. In `forward_update_block`, `strive_time` is initialized to ones for every scale invocation. At recurrent iteration `n`:

```text
penalty = exp(-strive_time / (sum_candidates(strive_time) + T))
frame_confidence = spatial_mean(sigmoid_confidence_map)
frame_score = penalty * similarity + frame_confidence
indices = argsort(frame_score, descending=True)[..., :5]
strive_time[selected] += 1
```

Thus released “redundancy” is not pairwise spatial redundancy between two memories. It is an iteration-local usage penalty, separately maintained for each query frame. `top_k` is hard-coded to 5. Sorting is performed over every frame in the clip, with no causal mask, no invalid-frame exclusion and no explicit exclusion of the query frame itself.

The training implementation also differs from Equation 1: `train_utils/losses.py::sequence_loss` forms the confidence target as `exp(-0.9*abs(error)/7)+1e-2`. It adds an L1 confidence loss to the disparity L1 term at each recurrent prediction.

## Play: modulation and read-out

For each query frame, selected scores are divided by `selected_score.mean()` in the released code (a scalar mean over the selected tensor in that loop). The selected key is stored internally as `[key, positional_encoding]`; modulation evaluates

```text
modulated_key = key * normalized_selected_score + positional_encoding
```

The current query also receives its temporal encoding. Query-to-selected-key FlashAttention then weights selected motion values. Its result is added to the per-frame motion features through `update_block.aggregator.beta`; `beta` is initialized to zero. The recurrent update block consumes current correlation features, context/recurrent state and aggregated temporal motion features, predicts `delta_flow`, and iteratively adds that residual to disparity/flow.

The paper's normalized play weight `S_i/sum_j S_j` is therefore best understood as feature modulation, not a direct weighted average of disparity maps. An ARGOS weighted disparity aggregation is an adapted analogue, not the faithful read-out.

## Causality and backbone coupling

The original implementation is **non-causal**:

- `forward` receives the whole `[B,T,3,H,W]` stereo clip;
- `forward_sst_block` applies temporal attention over all `T` frames;
- `forward_update_block` computes a full `[T,T]` query-memory score matrix;
- `flash_attn_func(..., causal=False)` reads all selected frames;
- a frame can select itself, earlier frames and future frames;
- the wrapper's overlapping-window inference uses both sides of interior frames.

The original module is also intrinsically stereo-backbone-specific. Its keys depend on learned context features; values depend on correlation volumes, disparity state and recurrent hidden state; confidence is trained from those values; and the read-out feeds a PPMStereo GRU decoder. These are not available in heterogeneous frozen disparity caches.

## Reuse decision for ARGOS v2

Directly reusable, with source attribution:

- pooled cosine query/key similarity as a reference diagnostic;
- quality-plus-relevance score composition;
- selection-count/strive-time penalty concept;
- deterministic top-K construction;
- normalized per-entry play weights;
- zero/near-zero initialization of the temporal residual path.

Requires an ARGOS-specific adaptation:

- a persistent append/reset causal bank;
- exact past-frame ages and sequence-boundary reset;
- validity-aware, spatially corresponding similarity/redundancy;
- BiDA alignment of every candidate into the current grid;
- universal quality from warp support, forward-backward confidence, photometric evidence and disparity agreement;
- aggregation of disparity/evidence rather than cost-volume motion values;
- an optional learned selector and bounded residual refiner.

Cannot honestly be called the full PPMStereo selector after removing internals:

- QAM confidence learned from correlation/recurrent value features;
- cost-volume value memory;
- query/key projections from PPMStereo context features;
- FlashAttention cost-feature read-out;
- GRU-integrated iterative residual stereo decoding;
- full-clip non-causal selection.

Accordingly, `model_design/external_components/ppmstereo.py` names its exact formula helpers **faithful reference** and its deployable mechanisms **ARGOS universal causal adapter**. BiDA warping is imported from `model_design/external_components/bidavideo.py`; it is not reimplemented.

## Losses in the original training

`train_utils/losses.py::sequence_loss` masks invalid/large (`>=700`) disparities and applies exponentially weighted L1 loss over recurrent predictions. The adjusted gamma is `0.9**(15/(N-1))`. When uncertainty outputs are present it adds L1 distance to the code-level confidence target above. An optional initialized-flow prediction receives an additional L1 term. There are no explicit clean-preservation, ranking, abstention, redundancy or safety losses in the released implementation.
