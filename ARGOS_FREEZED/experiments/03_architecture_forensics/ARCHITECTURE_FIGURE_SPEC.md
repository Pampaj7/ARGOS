# Figure specification

Draw a left-to-right causal pipeline. Use blue for frozen external components (stereo and SEA-RAFT), gray for immutable raw memory, orange for learned 60,739-parameter retrieval/fusion, and green for masks/fallback.

1. Top-left: current rectified stereo pair; arrow to a box labelled “frozen stereo estimator”; output `d_t^S, M_t`.
2. Left-middle: raw bank with four labelled records `CS1=d_{t-1}^S`, `CS2=d_{t-2}^S`, `CS4=d_{t-4}^S`, `CS8=d_{t-8}^S` and their source left RGB/provenance.
3. Center: four independent parallel branches. Each branch shows two SEA-RAFT arrows (`current -> source` and `source -> current`), then a BiDA-style pull warp onto the current grid. Label each branch `F_{t->t-k}`, `\tilde d_t^(k)`, validity/support/FB confidence.
4. Middle-right: four tensors labelled `[B, K, 17, H, W]`, entering one shared-weight CNN block marked `shared over k`.
5. Right: four score maps; a pixelwise `argmax_k` selects one age and one aligned candidate. Do not draw multiple candidates being averaged.
6. Output: selected candidate and raw disparity enter `pairwise soft fusion`; then `acceptance (p>=0.9, s>=0.1)` and a green `exact raw fallback` mux.
7. Bottom feedback: a solid arrow from `current raw d_t^S` only to `append_raw` and then bank. Draw a red crossed arrow from `d_t^out` to the bank labelled `no fused-state writeback`.
8. Show a dashed causal boundary behind the current frame and no future-frame boxes.

Do not include a critic, H4 state, backbone ID, internal cost volume/features, flow composition, or a claim of safety/OOD/general backbone agnosticism.
