# ARGOS v2 geometry_v1: forensic architecture reconstruction

## 1. Evidence discipline

This report separates facts from interpretation. `[IMPLEMENTED]` means the statement is read directly from frozen code; `[TESTED]` means a targeted frozen test enforces it; `[CONFIGURED]` means it is recorded in a frozen manifest/configuration; `[EMPIRICALLY VALIDATED]` means it is a reported result of a prior validated experiment; `[INTERPRETATION]` is a mechanical consequence or scientific reading. Frozen verification passed with 24 immutable hashes. The source checkpoint is `raw_multi_anchor_best_validation.pt`, SHA-256 `40526a32ef6e9a62a3ea2b59e6751a60c441b8190f9b96522e3b12b35895d5cd`; SEA-RAFT is `Tartan-C-T-TSKH-spring540x960-S.pth`, SHA-256 `1a21575ed6ca2c6945fb8e25c4169d241cf59ee5d12b8802c01c965206268cac`. `[TESTED][CONFIGURED]`

The primary frozen symbols are in `src/argos_freezed/pipeline.py`, `memory_bank.py`, `alignment/bida_pull_warp.py`, `alignment/sea_raft_adapter.py`, and `models/raw_multi_anchor_refiner.py`. Training provenance is `ARGOS-V2/model_design/losses/raw_multi_anchor_losses.py` and `scripts/run_raw_multi_anchor_temporal_refiner.py`. `[IMPLEMENTED]`

## 2. End-to-end causal execution

At time index (t), an external frozen stereo estimator first consumes the current rectified pair and produces (d_t^{S}) and a validity mask (M_t). The temporal package receives `current_left_rgb`, `current_right_rgb`, (d_t^{S}), (M_t), and a `RawAnchorBank`. The right image is shape-checked but not used after this point; it is present to preserve the stereo-interface boundary. `[IMPLEMENTED]`

For each (kin{1,2,4,8}), the bank looks up the raw source at (t-k). A missing source produces a zero-validity/zero-support dummy candidate whose disparity tensor has the current raw shape. A present source supplies detached raw disparity (d_{t-k}^{S}), validity, and left RGB. `[IMPLEMENTED]`

For a present source, frozen SEA-RAFT is evaluated twice: (F_{tightarrow t-k}) from current left RGB to source left RGB and (F_{t-kightarrow t}) in the reverse order. No intermediate flow is composed. The source disparity is pulled to the current grid by BiDA-style `grid_sample`, then validity, support, forward-backward error/confidence, photometric helper evidence, and disparity disagreement are computed. Only aligned disparity, aligned validity, coordinate support, and continuous FB confidence enter `MultiAnchorEvidence`. `[IMPLEMENTED][TESTED]`

The four candidates are stacked as ([1,K,H,W]), (K=4). Evidence construction creates ([1,K,17,H,W]). The shared CNN runs on ([B K,17,H,W]); it returns one utility logit, one signed delta, and one fusion weight per candidate and pixel. Unavailable scores are set to (-infty). A pixelwise max chooses exactly one age. The selected candidate is proposed as (d^{prop}=d^{S}+w(d^{align}-d^{S})). Acceptance requires availability, sigmoid utility probability at least (0.9), and score at least (0.1) px. Rejected pixels use `torch.where(..., raw)`, so their output is exactly the input raw tensor. `[IMPLEMENTED][TESTED][CONFIGURED]`

Only after output construction does the bank append the current raw prediction with `append_raw`. The output, proposal, or any fused value is never written to long-term state. `[IMPLEMENTED][TESTED]`

## 3. Raw immutable memory

`RawAnchor` contains `disparity: [1,1,H,W]`, `validity: [1,1,H,W]`, `left_rgb: [1,3,H,W]`, `frame_id: str`, `frame_index: int`, optional `timestamp`, and the exact provenance string `independent_frozen_stereo`. `append_raw` requires finite positive disparity on valid pixels, matching shapes, batch one, strictly increasing frame indices, and that provenance. It stores detached clones, not references. `[IMPLEMENTED][TESTED]`

Lookup is by integer index (t-k). Eviction removes indices below (t-max(A)=t-8); with contiguous input the bank can retain the current eight-frame history plus the current frame. Missing ages return `None`; there is no interpolation. Duplicate or out-of-order indices are rejected. Timestamp never controls lookup or eviction. `[IMPLEMENTED]`

The invariant is

\[
\mathcal M_t=\{(d_j^S,M_j,I_j,j,\tau_j):j\le t,\;j\text{ retained}\},
\qquad d_j^S=\operatorname{Stereo}(I_j^L,I_j^R),
\]

and not (operatorname{StereoRefine}(I_j,d_j^S,mathcal M_{j-1})). Therefore an error introduced at (j) is not recursively reintroduced as the source for (j+1); this is the mechanical reason the architecture avoids corrected-state drift. `[IMPLEMENTED][INTERPRETATION]`

## 4. Ages and causal semantics

Age (k) means source frame index (t-k), not an elapsed timestamp and not a recurrent state. Each pixel has an independently selected age (k_t^star(x)). The model does not average several retrieved anchors: the max is over candidate scores and only one candidate is gathered. Age is represented as the scalar (k/8), spatially expanded as channel 7. It therefore enters the shared evidence encoder, not a separate age head. `[IMPLEMENTED]`

## 5. SEA-RAFT

The adapter is the vendored SEA-RAFT model loaded with `use_var=True`, `var_min=0`, `var_max=10`, `pretrain=resnet18`, `initial_dim=64`, `block_dims=[64,128,256]`, `radius=4`, `dim=128`, `num_blocks=2`, and `iters=4`. Inputs are float32 `[B,3,H,W]`; the vendored `InputPadder` pads both images, the model runs under `torch.inference_mode()`, and the final flow is unpadded. Output is `[B,2,H,W]` in pixel units. `[IMPLEMENTED]`

`infer(target,source)` returns target-to-source flow. Thus the current-to-anchor call is (F_{tightarrow t-k}), and the reverse call is (F_{t-kightarrow t}). There are two SEA-RAFT evaluations per available age, at most eight per current frame. No gradients, weight updates, flow composition, or future frame access are permitted. `[IMPLEMENTED][TESTED]`

## 6. BiDA-style pull warp

For source tensor (S\in\mathbb R^{B\times C\times H\times W}), target-to-source flow (F=(F_x,F_y)), and target-grid integer coordinate ((x,y)),

\[
q_x(x,y)=x+F_x(x,y),\qquad q_y(x,y)=y+F_y(x,y),
\]
\[
G_x=2q_x/(W-1)-1,\qquad G_y=2q_y/(H-1)-1,
\qquad \widetilde S(x,y)=\operatorname{bilinear}_{\text{zero},\,\texttt{align\_corners=True}}(S,G(x,y)).
\]

The support mask is (B(x,y)=1) iff (0\le q_x\le W-1) and (0\le q_y\le H-1). A source-validity tensor is sampled with the same grid and is valid only when the sampled value is at least (0.999). The combined warp validity is (B\land V_S^{sampled}). `[IMPLEMENTED][CONFIGURED]`

Disparity is a scalar field and is sampled directly; no horizontal-flow correction is applied. This is a pull, not a push: each current pixel asks which source location supplies its value. `[IMPLEMENTED][INTERPRETATION]`

## 7. Forward-backward consistency

The reverse flow is pulled onto the current grid using the forward flow:

\[
\bar F_{t-k\rightarrow t}=\operatorname{Warp}(F_{t-k\rightarrow t},F_{t\rightarrow t-k}),
\quad e=\|F_{t\rightarrow t-k}+\bar F_{t-k\rightarrow t}\|_2,
\]
\[
\theta=0.5+0.01(\|F_{t\rightarrow t-k}\|_2+\|\bar F_{t-k\rightarrow t}\|_2),
\quad C=\exp(-e/\max(\theta,10^{-6}))B.
\]

The binary FB-valid mask is (B\land(e\le\theta)). Confidence (C) is continuous, nominally in ([0,1]), and zero outside support. The frozen candidate availability does **not** use the binary FB-valid mask; it is `candidate_valid & warp_support`. FB confidence remains a feature and returned diagnostic. `[IMPLEMENTED]`

## 8. Exact 17-channel evidence

For candidate (k), (r=d_t^S), (c=\widetilde d_t^{(k)}), (a_k) its availability, (m) the candidate median over available ages, (q) its MAD, and (n=\sum_k a_k), the exact ordered tensor is

\[
x_t^{(k)}=[r/64,c/64,\operatorname{clip}(c-r,-16,16)/16,|c-r|/16,
\mu_5(|c-r|)/8,\sigma_5(c-r)/8,k/8,
a_k, s_k,C_k,n/K,m/64,q/8,|c-m|/8,|r-m|/8,A/K,p_k].
\]

Here raw/candidate/median are clipped to ([0,64]); local mean/std are 5x5 average-pool with stride 1 and padding 2, with variance clamped at zero before square root and local outputs clipped to 8; (A=\sum_k[a_k\land |c_k-m|\le q+0.10]); (p_k) is provenance, zero for canonical raw anchors. The tensor is `torch.stack(maps, dim=2)` with shape `[B,K,17,H,W]`. `[IMPLEMENTED]`

The helper computes gradients, photometric residual, signed/absolute disparity disagreement, and flow magnitude, but those values are not in `maps`; the exact 17-channel list above is authoritative. `[IMPLEMENTED]`

## 9. Shared encoder and parameterization

Candidates are flattened into the batch dimension, so every age uses identical weights. There is no cross-age convolution. Cross-age information enters only through median/MAD/witness/agreement evidence and the later pixelwise max. `[IMPLEMENTED]`

| Layer | Input | Operation | Parameters | Output |
|---|---|---|---:|---|
| input | `[BK,17,H,W]` | 3x3 Conv, 17→32, padding 1, no bias | 4,896 | `[BK,32,H,W]` |
| norm/act | `[BK,32,H,W]` | GroupNorm(8,32), SiLU | 64 | same |
| block 1 | `[BK,32,H,W]` | Conv3x3-GN-SiLU-Conv3x3-GN + residual, then SiLU | 18,560 | same |
| block 2 | same | same shared-block form | 18,560 | same |
| block 3 | same | same shared-block form | 18,560 | same |
| head | `[BK,32,H,W]` | 1x1 Conv 32→3 | 99 | `[BK,3,H,W]` |

The head is zero-initialized with bias ([-2,0,-3]). Total (4896+64+3(18560)+99=60,739), independently recomputed by instantiating `RawMultiAnchorRefiner(32,3)`. `[IMPLEMENTED][TESTED]` The theoretical local receptive field is 15 pixels. `[CONFIGURED]`

## 10. Retrieval

For head channels (z_0,z_1,z_2),

\[
p_k=\sigma(z_{0k}),\qquad \delta_k=8\tanh(z_{1k}),\qquad w_k=\sigma(z_{2k}),
\qquad s_k=p_k\delta_k.
\]

Unavailable candidates receive (s_k=-\infty). The selected index is

\[
k_t^\star(x)=\arg\max_{k\in\{1,2,4,8\}}s_t^{(k)}(x).
\]

`torch.max` supplies the first index on an exact tie. There is no softmax over ages and (p_k) is not a calibrated probability of correctness; it is a sigmoid utility-logit output. Retrieval and acceptance are distinct: retrieval always selects the max finite score when one exists, while acceptance additionally checks (p) and the score threshold. `[IMPLEMENTED]`

## 11. Pairwise fusion, acceptance, fallback

After gathering the selected aligned anchor (widetilde d_t^{(k^star)}), probability, and raw predicted weight,

\[
d_t^{prop}=d_t^S+w_t(\widetilde d_t^{(k^\star)}-d_t^S)
=(1-w_t)d_t^S+w_t\widetilde d_t^{(k^\star)}.
\]

The effective weight is (w_t) only if accepted, otherwise zero. With (p_t=p_{k^star}), (s_t=s_{k^star}), and (V_t) the gathered availability,

\[
a_t=V_t\land[p_t\ge0.9]\land[s_t\ge0.1\text{ px}],
\qquad d_t^{out}=\operatorname{where}(a_t,d_t^{prop},d_t^S).
\]

Thus rejected pixels are bit-identical to raw. A missing/invalid selected candidate is rejected; an unavailable-only pixel has all scores (-\infty), a zero gathered effective weight, and raw fallback. An accepted update can still have a small numerical displacement because the predicted fusion weight is in ((0,1)); exact fallback does not certify accepted updates as safe. `[IMPLEMENTED][TESTED][INTERPRETATION]`

## 12. Returned result

For batch one all disparity-like fields are `[1,1,H,W]` float32 except `selected_anchor_age`, which is an integer-valued `[1,1,H,W]` tensor. `selected_anchor_age` is gathered age, `selected_aligned_anchor` is the pre-fusion selected candidate, `selection_score` is pre-acceptance (s), and `fusion_weight` is post-acceptance effective weight. `accepted_mask` is bool. `support_mask` is gathered availability; `validity_mask` is current validity intersected with gathered support; `forward_backward_consistency` is gathered continuous (C); `update_magnitude=|d^{out}-d^S|). `proposal_disparity` is pre-acceptance and can differ from output on rejected pixels. Metadata records project/version/ages/missing ages/direct-flow/no-composition/no-recurrence/no-fused-writeback. Only raw disparity, validity, RGB, and provenance can enter memory. `[IMPLEMENTED]`

## 13. Canonical training objective

For raw (r), candidate (c_k), ground truth (g), coverage (G), raw validity (M), and availability (A_k),

\[
e_r=|r-g|,\quad e_k=|c_k-g|,\quad \Delta_k=e_r-e_k,
\]
\[
V_k=[G>0.5]\land M\land A_k\land\operatorname{finite}(\Delta_k),
\qquad h_k=V_k\land[\Delta_k>0.10].
\]

Let (N_+=\sum h_k), (N_-\) be valid non-helpful count, and (\rho=\operatorname{clip}(N_-/\max(N_+,1),1,50)). With clipped (widehat\Delta=\operatorname{clip}(\Delta,-8,8)),

\[
\mathcal L_{cls}=\operatorname{MaskedMean}_{V}\operatorname{BCEWithLogits}(z_0,h;\operatorname{pos\_weight}=\rho),
\]
\[
\mathcal L_{reg}=\operatorname{MaskedMean}_{V}\bigl[\operatorname{SmoothL1}(\delta,\widehat\Delta)\,\operatorname{where}(h,\rho,1)\bigr].
\]

For ordered candidate pairs with both valid and (|\Delta_i-\Delta_j|>0.10), (o_{ij}=\operatorname{sign}(\Delta_i-Delta_j)),

\[
\mathcal L_{rank}=\operatorname{MaskedMean}\max(0,0.10-o_{ij}(\delta_i-\delta_j)).
\]

The differentiable candidate fusion prediction is (q_k=r+w_k(c_k-r)). Then

\[
\mathcal L_{fuse}=\operatorname{MaskedMean}_{V}[\operatorname{SmoothL1}(q_k,g)\operatorname{where}(h,\rho,1)],
\quad
\mathcal L_{harm}=\operatorname{MaskedMean}_{V\land[\Delta_k\le0.10]}w_k,
\]
\[
\mathcal L=1.0\mathcal L_{cls}+0.5\mathcal L_{reg}+0.25\mathcal L_{rank}+1.0\mathcal L_{fuse}+0.2\mathcal L_{harm}.
\]

The last two terms are enabled for soft training; the validated runner disables them for its `hard` configuration. Masks with no elements contribute an exact zero via `value.sum()*0`. Alignment and SEA-RAFT are outside the learned graph in the runner's precomputed bank; raw stereo predictions and flow are not optimized. `[IMPLEMENTED]`

## 14. Training sample construction

The validated runner uses training sequence IDs 1,3,6 and validation ID 2, with seen backbones S2M2-S, RAFT-Stereo, and StereoAnywhere. It loads validated raw caches, computes direct SEA-RAFT age flows for ages 1,2,4,8, aligns raw banks, and retains frames from index 8 onward. Training crops are (64\times80), stored in float16 for continuous arrays and uint8 for masks, then converted to float32 on device. A seeded NumPy generator selects crop origins; no image augmentation was found. `[IMPLEMENTED]`

The sampler groups by `(backbone, sequence)`, shuffles each group with a seed plus epoch, repeats shorter groups to the maximum group length, and interleaves groups. The model sees only the universal evidence tensor, not a backbone ID; the same evidence schema and shared weights therefore process all three frozen stereo sources. `[IMPLEMENTED]`

The source runner defaults to batch size 12, 10 epochs, AdamW (lr=2\times10^{-3}), weight decay (10^{-4}), cosine annealing to (0.05lr), gradient clipping 5, and CUDA AMP when the device is CUDA. Those values describe the validated runner defaults, not additional inference architecture. `[CONFIGURED][IMPLEMENTED]`

## 15. H4 comparison

| Property | bounded corrected-memory H4 | immutable raw multi-anchor geometry_v1 |
|---|---|---|
| memory | corrected disparity state | independently generated raw disparity + raw provenance |
| recurrence | corrected state recurrent within a four-frame horizon | no corrected-state recurrence |
| writeback | corrected state | raw-only `append_raw` |
| ages | bounded H4 sequence | CS1/CS2/CS4/CS8 |
| alignment | BiDA-style causal alignment | same direct current-to-anchor alignment |
| fusion | CODD-style soft fusion against corrected memory | pairwise soft fusion against one selected raw anchor |
| drift | reset/bound limits exposure | no recursive corrected error source by construction |
| role | strong architectural baseline | main frozen geometric method |

H4 is isolated from the main package and is not imported by `pipeline`, `memory_bank`, or `models`. `[TESTED]` The side-by-side interpretation is architectural; prior result files report it as the bounded corrected-memory baseline. `[EMPIRICALLY VALIDATED][INTERPRETATION]`

## 16. Computational structure

For (K=4), the bank stores up to eight past raw anchors plus the current raw frame. Each available age costs two SEA-RAFT passes, one disparity pull warp, one backward-flow pull for consistency, and one photometric helper warp. The CNN sees (K) candidates as a shared batch: (K) feature maps, (K) score maps, (K) fusion maps, and one selected candidate per pixel. The learned adapter storage is 60,739 parameters. `[IMPLEMENTED]`

Ignoring flow-network internals, evidence and CNN work are (O(KHW)), memory is (O(KHW)), and flow cost is (O(K C_{SEA}(H,W))). A measured runtime/FPS is not present in the inspected sources and is therefore not claimed. `[INTERPRETATION]`

## 17. Invariants and claims

The enforced invariants are listed in `INVARIANTS.csv`: past-only lookup, direct current-to-anchor flow, no composition, raw-only detached clones, no recurrent/fused writeback, exact ages, (-\infty) invalid masking, exact fallback, positive-left disparity, no backbone ID/internal stereo features, no critic/H4 import, frozen weights, deterministic inference, and original-runner parity. `[TESTED]`

Supported claims are causal online operation, immutable raw temporal memory, a frozen stereo interface, substantial long-range-anchor use, and same-domain transfer to the evaluated unseen estimators within SCARED-C. Prior transfer artifacts report FULL UNSEEN-BACKBONE GEOMETRY GO. `[EMPIRICALLY VALIDATED]`

Unsupported claims are universal backbone agnosticism, external-domain/OOD generalization, clinical or risk-controlled safety, guaranteed non-degradation, and real-time deployment without a dedicated benchmark. Exact fallback only protects rejected pixels; it does not make accepted updates safe. `[INTERPRETATION]`
