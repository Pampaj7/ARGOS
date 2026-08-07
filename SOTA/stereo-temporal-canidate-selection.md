# Causal Per-Pixel Selection Between Current and Temporally Propagated Stereo Disparity: A Deep Literature Review and Novelty Audit

**Date:** 2026-07-19
**Status:** Draft (pre-citation). Evidence base: four researcher notes (`outputs/.drafts/stereo-temporal-candidate-selection-research-T1..T4.md`), ~140 primary sources at abstract/HTML level; three papers read in full (TC-Stereo, CODD, TemporalStereo). PDF-only method details are marked blocked throughout.
**Method caveat:** most mechanism claims below rest on abstracts, project pages, and HTML full texts — not full PDF reads. Where a claim depends on a paper body that was not read, it is flagged. Absence-of-evidence claims ("no work found doing X") rest on ~100 distinct search queries across four researchers and are inference, not proof.

---

## Executive summary

The problem — a causal, lightweight, per-pixel policy that selects between (a) the current-frame stereo disparity and (b) a flow-propagated, BiDA-aligned previous-frame disparity, with an abstain-to-current default, optimized for *net geometric utility* rather than confidence ranking — is **partially addressed in adjacent settings, and its exact combination appears open**. The dominant novelty threat is **CODD (Li et al., WACV 2023)**, which operates on the same high-level candidate pair, learns per-pixel fusion/reset weights **supervised on the relative error of the two candidates** (with margins and a dead-band), defaults to the current estimate when the temporal candidate is rejected, publishes the same per-pixel oracle experiment (their "empirical best fusion", Tab. 5), and demonstrates cross-backbone applicability. What CODD lacks, and what was not found together in the other surveyed work, is: a hard selection/abstention *decision* (vs. bounded soft blending), a risk–coverage / net-utility / harm-rate evaluation against a keep-current baseline, a calibrated safe operating threshold, and a formulation of the problem as selective prediction / learning-to-defer. The decision-theoretic literature (Okati et al. 2021; Jitkrittum et al. 2023) already proves — at example level — that the optimal such policy thresholds the *predicted error difference* between the two candidates and that confidence-only deferral is provably suboptimal precisely when the second candidate is a "specialist" (good only on a subset of inputs), which is exactly the character of a temporally propagated disparity. No surveyed work was found that instantiates this at pixel level for dense regression, and no surveyed work was found that evaluates temporal-candidate selection with net-utility-at-safe-threshold metrics. The clinical decision-curve-analysis literature independently documents the "good AUROC, negative net benefit vs. do-nothing" trap and prescribes the fix (net-benefit-vs-threshold curves against a never-intervene baseline) — a fix not found in the surveyed vision literature for this temporal-selection setting.

**Verdict in one line:** partially addressed (CODD covers the candidate pair and relative-error supervision; DVSNet/GRFP/Accel/LTMU cover fragments of the gating mechanics; L2D theory covers the formulation at example level) — but the specific problem *as a risk-controlled selective decision with abstention, evaluated on net geometric utility, causal and backbone-independent,* appears unsolved.

---

## 1. Ranked list of closest prior work

Ranking criterion: directness to "predict per pixel which of {current, propagated} disparity is more accurate, without GT, with abstention, causally."

### Tier 1 — same candidate pair, learned per-pixel arbitration

**1. CODD — "Temporally Consistent Online Depth Estimation in Dynamic Scenes" (Li, Ye, Wang, Creighton, Taylor, Venkatesh, Unberath; WACV 2023; arXiv:2111.09337).** [Read in full via ar5iv HTML.]
- Same high-level candidate pair: current per-frame stereo disparity d_S vs. previous fused estimate aligned to the current frame (d_M). The construction is materially different from the proposed setup: CODD aligns the temporal candidate via a learned per-pixel SE3/scene-flow field rather than optical flow + BiDA-style alignment.
- A tiny fusion network (0.2M params, causal, ~0.3 ms) regresses per-pixel reset and fusion weights; output = (1 − w_r·w_f)·d_S + w_r·w_f·d_M. Structural default is the current estimate (w_r→0 recovers d_S).
- **Relative-utility supervision:** the weights are supervised on the sign of e_M − e_S with margins (τ_reset = 5 px; τ_fusion = 1 px, with a regularize-to-0.5 dead-band when candidates are equally good). This is per-pixel comparative supervision, not generic confidence — the central conceptual move of the proposed project already exists here.
- **Oracle published:** "empirical best fusion" (per-pixel GT-better candidate) improves EPE 0.595→0.455 and TEPE 0.741→0.529 on FlyingThings3D (their Tab. 5) — the same headroom argument driving this project.
- **Blind fusion harms — published evidence:** their per-pixel Kalman baseline improves temporal consistency but worsens EPE; the motion-only (propagated) candidate is worse than per-frame stereo on EPE.
- Cross-backbone: fusion+motion nets were applied to frozen HITNet, STTR, PSMNet, and GwcNet. CODD mostly improves temporal metrics across backbones, but not every metric/backbone improves; its Appendix C.1/Table 7 reports exceptions including STTR EPE and temporal δ3px and a GwcNet temporal metric. This weakens any blanket improvement claim.
- What it does **not** do: no hard selection; no explicit abstain class (output always a convex blend; their stated limitation: cannot fix pixels where both candidates are wrong — but equally, it cannot fully *refuse* the temporal candidate other than asymptotically); no risk–coverage / net-utility / harm-rate evaluation; no calibrated operating threshold; objective framed as temporal consistency ("accuracy on par"), not accuracy gain; weight losses disabled on sparse-GT KITTI (a practical warning about supervision under sparse GT); alignment requires a trained scene-flow network.

**2. Khan et al. — "Temporally Consistent Online Depth Estimation Using Point-Based Fusion" (CVPR 2023; arXiv:2304.07435).** [Abstract + snippets; method PDF blocked.]
- Causal; global point-cloud memory reprojected to the current frame vs. current per-frame depth. Abstract/project-level evidence supports learned image-space fusion and the stated consistency-vs-correction tension. The more specific α-mask/gating-direction details came from snippets/code-level summaries and were not verified from a non-PDF method source, so they should be treated as unverified unless checked in the paper body or code.
- Abstract states the exact tension: the method "must choose between enforcing consistency and correcting errors from previous estimations."
- Decision direction is inverted relative to ours: change detection admits *current over history* to protect consistency, rather than admitting *history over current* to improve accuracy. No relative-error supervision claimed in the abstract; no abstention semantics; no risk evaluation. Depth-source agnostic (works over any per-frame depth), which parallels our backbone-independence goal.

### Tier 2 — learned propagate-vs-recompute / warped-previous-vs-current gating in other dense tasks

**3. DVSNet — "Dynamic Video Segmentation Network" (Xu et al., CVPR 2018; arXiv:1804.00931).**
- A decision network predicts the expected quality (confidence score) of the flow-warped segmentation per frame *region*; regions above threshold take the cheap warp path, below take full re-segmentation.
- This is a learned, quality-predicting, causal propagate-vs-recompute selector with a recompute (≈ keep-current) default — the right supervision idea (predict the propagated path's quality before using it), wrong granularity (region), absolute rather than relative target, and motivated by compute savings, not accuracy or harm control.

**4. GRFP — "Semantic Video Segmentation by Gated Recurrent Flow Propagation" (Nilsson & Sminchisescu, CVPR 2018; arXiv:1612.08871) and Accel (Jain et al., CVPR 2019).**
- GRFP: flow-warped previous predictions are gated per pixel by estimated warp/flow reliability before fusion with the current prediction (STGRU), end-to-end. Accel: a warped-keyframe branch and a current-frame branch fused by a learned 1×1 convolution per pixel.
- Structurally the closest mechanism family: two candidates (warped memory vs. current output), learned per-pixel arbitration, causal (GRFP's core unit). But: soft blending, trained implicitly through task loss (no explicit relative-error labels), classification not regression, no abstention semantics, no harm/risk evaluation.

**5. LTMU — "High-Performance Long-Term Tracking with Meta-Updater" (Dai et al., CVPR 2020).**
- A learned, binary, causal, abstaining gate: "is the tracker ready for updating in the current frame?", from sequential geometric/discriminative/appearance cues, explicitly to avoid harmful updates; transferable across trackers.
- Right decision semantics (learned WHETHER with abstain-by-default, harm-motivated, estimator-agnostic), wrong granularity (frame) and wrong object (model update, not estimate replacement).

### Tier 3 — formulation-level matches from decision theory / L2D

**6. Okati, De & Gomez-Rodriguez — "Differentiable Learning Under Triage" (NeurIPS 2021; arXiv:2103.08902).**
- Proves the optimal triage policy under a deferral budget is a deterministic threshold on the per-instance *difference* between the two agents' errors. This is exactly the mathematical object our selector must estimate — published, but at example level, for human-AI triage, not dense vision.

**7. Jitkrittum et al. — "When Does Confidence-Based Cascade Deferral Suffice?" (NeurIPS 2023; arXiv:2307.02764).**
- Characterizes optimal two-model deferral; proves confidence-only deferral is suboptimal when the downstream model is a specialist, under label noise, or under shift. The propagated disparity is a textbook specialist (excellent on static well-tracked regions, harmful at occlusions/motion) — a theoretical argument, made by us as an inference, that generic-confidence selectors are the wrong object for this problem. Example-level, classification.

**8. DeferredSeg (arXiv:2604.12411, 2026).**
- The first found pixel-wise learning-to-defer system: routes each pixel to base segmentor or expert; pixel-wise surrogate collaboration loss + spatial-coherence loss on the deferral mask; multi-expert extension. Dense classification, expert = human (no harm from deferral target), offline. No dense-regression L2D found anywhere.

### Tier 4 — learned multi-candidate disparity selection (non-temporal) and two-source depth fusion

**9. SGM-Forest — "Learning to Fuse Proposals from Multiple Scanline Optimizations in SGM" (Schönberger, Sinha, Pollefeys; ECCV 2018).**
- Learned per-pixel classification selecting among multiple scanline disparity proposals. Genuine learned per-pixel candidate selection in stereo — so "learned per-pixel disparity candidate selection" per se is *not* novel. Trained as per-proposal correctness (absolute), no abstention, no temporal setting, no net-utility evaluation.

**10. Spyropoulos & Mordohai — "Ensemble Classifier for Combining Stereo Matching Algorithms" (3DV 2015) and Poggi & Mattoccia — "Deep Stereo Fusion" (3DV 2016).**
- Classifiers deciding per pixel whether each of several stereo matchers is correct, using cross-candidate agreement/disagreement features (2015), and a CNN selecting the best disparity among multiple maps using disparity-domain cues only (2016 — consistent with our finding that photometric cues are weak). Targets are per-candidate absolute correctness, argmaxed at inference; not pairwise relative targets; not temporal; no abstention. [Both at record/abstract level; PDFs blocked.]

**11. ELFNet (ICCV 2023; arXiv:2308.00728) and Marin et al. ToF-stereo fusion (ECCV 2016).**
- Two-candidate depth arbitration via two independently estimated (evidential or modality-specific) uncertainties. Comparing two marginal absolute confidences ≠ predicting the signed error difference; calibration error of each map contaminates the comparison. No abstention, no temporal setting, no risk evaluation.

**12. ManyDepth (CVPR 2021) / ProDepth (ECCV 2024).**
- Monocular multi-frame line with per-pixel *fallback to the single-frame estimate* where multi-frame (cost-volume) evidence is unreliable — abstain-to-safer-candidate in spirit. ManyDepth implements this via a training-time reliability/consistency mechanism (mask details partially from secondary sources). ProDepth has an arXiv abstract that supports a high-level probabilistic fusion/modulation framing, but method details were not read. These works are not stereo and are not inference-time learned relative-utility selectors between finished disparity maps.

### Tier 5 — memory-staleness gating and evaluation machinery

**13. QDMN (ECCV 2022)** — learned per-frame quality gate on what enters VOS memory (absolute quality, write-time). **RealBasicVSR (CVPR 2022)** — sharpest published statement that propagation is *conditionally harmful* ("severe degradations could be exaggerated through propagation"); mitigation is global pre-cleaning, not per-pixel selection. **RMem (CVPR 2024)** — restricting memory size alone improves VOS (memory can be net harmful). **Skip-Conv / DeltaCNN (CVPR 2021/2022)** — hard per-pixel reuse-vs-recompute gates, but criterion = input change, objective = FLOPs.

**14. Risk-control and evaluation machinery:** Chow (1970) cost-sensitive rejection; Geifman & El-Yaniv (2017)/SelectiveNet (2019) selective prediction with risk–coverage; Zaoui et al. (2020) selective regression (example-level, variance-based); Learn-then-Test (2021), Conformal Risk Control (2022), and **Conformal Decision Theory (Lekeufack et al., ICRA 2024)** — candidate machinery for calibrating decision thresholds and controlling user-defined risks. Applying these guarantees to dense per-pixel video switching would require a formal risk definition, calibration unit, aggregation rule, and validity conditions. **Decision-curve analysis** (Vickers & Elkin 2006; Vickers et al., BMJ 2016) documents that good-AUC models can have net benefit below the do-nothing policy at operating thresholds, and prescribes net-benefit-vs-threshold curves against never-intervene as the primary evaluation; Jaeger et al. (ICLR 2023) advocates AURC/risk–coverage over fragmented AUROC protocols for failure detection.

Also relevant but more distant: TC-Stereo (ECCV 2024 — temporal candidate as confidence-filtered semi-dense *initialization*, refined away rather than selected against; its peak-ratio filter is a per-pixel abstention mechanism on temporal hints); TemporalStereo (IROS 2023 — temporal disparities as extra cost-volume hypotheses, arbitration implicit in cost aggregation); XR-Stereo (WACV 2024 — warm-start only); DROID-SLAM (learned per-pixel confidence weights on temporal correspondence residuals, soft, inside BA); Mostegel et al. (CVPR 2016) and Poggi et al. (ECCV 2020) — GT-free confidence supervision from self-contradiction / online self-adaptation, the key recipe for supervising a selector without GT.

---

## 2. Comparison table

Legend — Rel-util: is the learned target comparative (sign/magnitude of error difference between candidates) rather than absolute confidence? Abst: explicit abstain/keep-current action? Risk-eval: evaluates net utility, harm rate, risk–coverage, or calibrated threshold?

| # | Work (year) | Task | Candidates | Rel-util target | Granularity | Abst | Causal | Supervision | Inference cues | Risk-eval | Cross-model | Directness to our problem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | CODD (2023) | online stereo depth | current stereo vs SE3-aligned previous fused | **Yes** (margin losses on e_M−e_S) | pixel | implicit (reset→current; dead-band) | yes | GT disparity of both candidates | LR-feature confidence, self/cross-correlation, flow mag+conf, visibility, semantics | no (EPE/TEPE only) | yes (4 backbones) | **Highest** — same pair, comparative supervision, oracle published |
| 2 | Khan et al. (2023) | online video depth | propagated point-cloud history vs current depth | no (change mask) | region/pixel | no | yes | blocked (PDF) | change detection | no | depth-source agnostic | High — same tension, inverted decision direction |
| 3 | DVSNet (2018) | video semantic seg | flow-warped prev seg vs recompute | no (absolute expected quality of warp path) | region | recompute default | yes | expected-confidence routing (supervision details not verified from non-PDF source) | flow, features | no | — | High — learned quality-predicting propagate-vs-recompute |
| 4 | GRFP / Accel (2018/19) | video semantic seg | warped prev prediction vs current | no (implicit task loss) | pixel | no | GRFP core: yes | task loss | flow reliability | no | no | High — per-pixel two-candidate fusion structure |
| 5 | LTMU (2020) | long-term tracking | update template vs keep old | no (binary update-safety) | frame | **yes** | yes | tracking outcome labels | sequential geometric/appearance cues | drift-motivated only | yes (plug-in) | Medium-high — abstaining learned gate, wrong granularity/object |
| 6 | Okati et al. (2021) | human-AI triage | model vs human per instance | **Yes** (threshold on error difference — proven optimal) | example | budgeted deferral | n/a | both agents' errors | learned predictor | no | theory | Formulation match at example level |
| 7 | Jitkrittum et al. (2023) | model cascades | model 1 vs model 2 | **Yes** (optimal rule uses both correctness probs) | example | deferral | n/a | both models' correctness | post-hoc features | no | theory | Formulation match; specialist analysis fits our temporal candidate |
| 8 | DeferredSeg (2026) | medical image seg | segmentor vs human expert | partial (collaboration surrogate) | **pixel** | **yes** (defer) | no | expert labels | features + spatial coherence | no | no | Pixel-wise L2D exists — for classification, expert=human |
| 9 | SGM-Forest (2018) | stereo (single frame) | multiple scanline proposals | no (per-proposal correctness) | pixel | no | n/a | GT disparity | cost/proposal features | no | generalizes across datasets | Learned per-pixel disparity candidate selection, non-temporal |
| 10 | Spyropoulos 3DV15 / DSF 3DV16 | stereo (single frame) | multiple algorithms' maps | no (per-candidate correctness; relational input features) | pixel | no | n/a | GT disparity | cross-candidate agreement; disparity-domain only (DSF) | no | multi-algorithm | Ancestor of candidate selection; relational cues precedent |
| 11 | ELFNet (2023) / Marin (2016) | stereo / ToF+stereo | two heterogeneous depth sources | no (two absolute uncertainties compared) | pixel | no | n/a | GT / per-branch | evidential params / modality confidences | no | two fixed models | Two-candidate fusion via marginal confidences |
| 12 | ManyDepth / ProDepth (2021/24) | mono multi-frame depth | multi-frame vs single-frame | no (disagreement mask / prob. fusion) | pixel | fallback-to-mono | yes | self-supervised | teacher disagreement / uncertainty | no | no | Fallback-to-safer-candidate precedent (training-time / probabilistic) |
| 13 | QDMN (2022) | VOS | store frame in memory vs not | no (absolute mask quality) | frame | store/don't | yes | quality proxy | predicted mask quality | no | plug-in | Write-time absolute-quality memory gate |
| 14 | CDT (2024) / DCA (2006) / LTT-CRC (2021-22) | decisions / clinical / calibration | act vs default | n/a (risk calibration & net-benefit eval) | decision | **yes** | CDT: online | calibration data | any score | **Yes — the machinery itself** | model-agnostic | Supplies the missing safety layer & evaluation protocol |

(Additional rows implicit: TC-Stereo, TemporalStereo, XR-Stereo — temporal-as-initialization/hypothesis, no selection head; RealBasicVSR/RMem — harm evidence, global mitigation; Skip-Conv/DeltaCNN — hard per-pixel gates for compute; DROID-SLAM — soft learned per-pixel residual weighting.)

---

## 3. Strongest novelty threats

1. **CODD is the dominant threat.** A reviewer can legitimately say: "the candidate pair, the per-pixel comparative supervision on relative error with margins/dead-band, the keep-current default, the oracle motivation, and the cross-backbone demonstration are all in CODD (WACV 2023)." Any paper on this problem that does not lead with a precise differentiation from CODD is at risk of a novelty reject. The defensible deltas are: hard selection with an explicit abstain action (CODD's output is always a convex blend); risk-controlled operating thresholds and net-utility/harm-rate evaluation (absent in CODD); accuracy-first rather than consistency-first objective; optical-flow+BiDA alignment without a trained scene-flow network; a selector claimed and evaluated as backbone-*independent* rather than backbone-*transferable-by-retraining*; and behavior under sparse/no GT (where CODD's weight losses were disabled).
2. **The formulation is not new at example level.** Okati et al. and Jitkrittum et al. already establish "threshold the predicted error difference" as the optimal two-candidate policy and prove confidence-only selection suboptimal for specialists. A framing that claims to *invent* relative-utility deferral will be rejected; the claim must be its instantiation, supervision, and safety evaluation for dense causal regression.
3. **Learned per-pixel candidate selection in stereo is 10+ years old.** SGM-Forest, Spyropoulos & Mordohai, Deep Stereo Fusion, Hu & Mordohai's multi-hypothesis evaluation protocol (2012). "We learn which disparity candidate is better per pixel" is not, alone, a contribution.
4. **Pixel-wise L2D now exists (DeferredSeg, 2026).** The "first pixel-wise learning-to-defer" claim is taken for classification. A dense-regression, harmful-expert, causal variant remains open but must be positioned against it.
5. **Fragment coverage from T4 literatures.** DVSNet (learned quality-predicting propagate-vs-recompute), GRFP/Accel (per-pixel warped-vs-current fusion), LTMU (learned abstaining update gate), QDMN (quality-gated memory writes), Skip-Conv/DeltaCNN (hard per-pixel reuse decisions), Khan et al. (gated overwrite between history and current). Individually none solves the problem; collectively they mean nearly every *mechanism component* has precedent. Novelty must be claimed at the level of the formulation + supervision target + safety evaluation, not the gate architecture.

---

## 4. The precise unresolved gap

Synthesis (inference from absence across ~100 queries; each fragment's existence is paper-supported):

No work found in the surveyed sources combines **all** of:
1. **Two finished dense regression candidates** — current stereo disparity and a temporally propagated, aligned previous disparity (CODD has this);
2. **A pairwise/relative decision target** — sign (and ideally magnitude) of the per-pixel error difference (CODD supervises soft weights this way; Okati proves it optimal at example level; no one trains an explicit pixel-wise comparator head with this target);
3. **A selective switch policy with abstention-as-default** — operationally a two-output decision {switch to propagated, keep current}; a three-label training target could distinguish 'current confidently better' from 'unsafe/uncertain, abstain', but both map to the same inference output. No dense-regression instance with this explicit semantics was found;
4. **Risk-controlled operation** — a calibrated threshold for a formally defined harm risk (e.g., fraction of switched pixels made worse by more than δ), using machinery such as Learn-then-Test / Conformal Risk Control / Conformal Decision Theory. This was not found in surveyed temporal-fusion or disparity-selection sources;
5. **Net-utility evaluation against the keep-current baseline** — decision-curve-style net-benefit vs. threshold and AURC, rather than AUROC (the "good AUROC, negative net utility" trap is documented in clinical DCA and in failure-detection critiques, and no stereo/video-depth selection paper reports such curves; SGM-Forest/DSF/ELFNet/CODD all report only final map error, which conflates selector and candidate quality);
6. **Causal, lightweight, backbone-independent deployment** (CODD is causal+lightweight and backbone-transferable; the plug-in stabilizer precedent exists in NVDS/BiDAStabilizer but for smoothing, not selection).

Secondary open sub-questions the literature does not answer:
- **GT-free comparative supervision:** self-contradiction/online-adaptation confidence labeling (Mostegel 2016; Poggi ECCV 2020) was not found in this survey extended to *pairwise* "A-better-than-B" pseudo-labels for two disparity candidates.
- **Sparse-GT supervision:** CODD explicitly disabled its comparative weight losses on KITTI's sparse GT — how to supervise a relative-utility head under sparse/pseudo GT is unresolved.
- **Why photometric cues fail and what replaces them:** the candidate-selection ancestors that work use disparity-domain and cross-candidate relational features (Spyropoulos 2015; DSF 2016), consistent with our observations; a principled cue analysis for the temporal setting is missing.
- **Harm-rate reporting:** no field surveyed reports a per-pixel "harm rate of using memory"; the metric itself would be a contribution (T4 synthesis, inference).

---

## 5. Verdict

**Partially addressed in adjacent settings; the exact problem appears open.**
- *Already solved?* No. No surveyed work delivers a causal, abstaining, risk-controlled, per-pixel relative-utility selector for current-vs-propagated disparity, evaluated on net geometric utility.
- *Partially addressed?* Substantially. CODD covers the candidate pair, comparative soft supervision, causality, lightweight deployment, and the oracle motivation. Decision theory covers the optimal-policy form. DVSNet/GRFP/Accel/LTMU/QDMN cover gate mechanics at various granularities. Risk-control and net-benefit machinery exists off the shelf.
- *Honest framing:* the contribution space that remains is (i) the **problem formulation** as pixel-wise selective deferral between estimators with a harmful (not merely useless) alternative; (ii) the **explicit pairwise decision head with abstention**; (iii) **risk-calibrated operation**; (iv) the **evaluation protocol** (net utility vs. keep-current, harm rate, risk–coverage); (v) **backbone-independence and GT-free/sparse-GT supervision**. Claiming more than this is not defensible against CODD.

---

## 6. Appropriate scientific terminology

- The overall problem: **selective temporal fusion** or **risk-controlled temporal candidate selection**; the decision itself is an instance of **learning to defer between two estimators** (cascade-deferral / triage form) at pixel granularity, where the deferral target can be harmful.
- The learned quantity: **relative utility** / **predicted error difference** (Okati's triage object), *not* "confidence" or "uncertainty" — reserve those for single-map absolute notions.
- The abstain behavior: **selective prediction with a reject option** (Chow; El-Yaniv & Wiener), default action = keep current.
- The safety layer: **distribution-free risk control / conformal decision-making**; the evaluation: **risk–coverage (AURC)** and **net benefit / decision-curve analysis** against the never-switch baseline; per-pixel **harm rate** for switched pixels.
- Terms to avoid as primary framing: "temporal fusion" (implies blending), "confidence estimation" (implies absolute single-map target), "uncertainty estimation" (implies distributional target), "smoothing/stabilization" (implies consistency objective). These name the adjacent literatures the paper must be distinguished from.

---

## 7. Three promising methodological directions (synthesis)

1. **A pixel-wise pairwise comparator with cost-sensitive deferral training.** Directly regress/classify sign(e_cur − e_prop) with a margin dead-band (CODD's label construction), but as an explicit decision head with three actions, trained with a consistent cost-sensitive L2D surrogate (Mozannar & Sontag 2020) and one-vs-all calibration (Verma & Nalisnick 2022) so thresholds are meaningful, using cross-candidate relational features (agreement/disagreement statistics, flow validity, LR-consistency, track-record/age features à la ORB-SLAM culling) rather than photometric cues — the feature choice is supported by Spyropoulos 2015 / DSF 2016 and by CODD's explicit-cue ablation. Add spatial-coherence regularization on the switch mask (DeferredSeg).
2. **Risk-calibrated switching with an explicit dense-video risk definition.** Treat the switch threshold as the free parameter and investigate Learn-then-Test / Conformal Risk Control for a risk such as 'fraction of switched pixels made worse by > δ'; investigate Conformal Decision Theory for online, non-IID adaptation along the video. This would require defining the calibration unit (pixel, frame, or sequence), aggregation rule, and monotonicity/validity assumptions. Evaluate with net-benefit-vs-threshold curves (DCA) and AURC, with keep-current as the zero line. This layer was not found in the surveyed stereo/video-depth literature.
3. **GT-free comparative pseudo-labels for cross-sequence robustness.** Extend self-contradiction confidence learning (Mostegel 2016) and online self-adapting confidence (Poggi ECCV 2020) to pairwise labels: where the two candidates disagree, use held-out geometric evidence (multi-view/temporal round-trip consistency, future-frame verification in an offline labeling pass) to decide which candidate the evidence contradicts, yielding "A-better-than-B" labels without GT. This addresses both the sparse-GT problem CODD hit on KITTI and the backbone-independence goal (labels are generated per deployment domain, selector stays a lightweight head over backbone-agnostic cues).

---

## 8. Recommended paper framing

**Title-shape:** "Should I Trust Yesterday's Depth? Risk-Controlled Pixel-Wise Deferral Between Current and Propagated Stereo Disparity."

**Framing:** Do *not* frame as a new temporal stereo method or as confidence estimation. Frame as: (1) an **empirical exposé** — across N backbones and sequences, a per-pixel oracle over {current, propagated} yields consistent gains (situating vs. CODD Tab. 5 and extending it to flow+BiDA alignment and surgical/real domains), while blind fusion and AUROC-good selectors produce negative net utility at safe thresholds (connecting to the DCA literature's AUC-vs-net-benefit divergence — the first demonstration of this trap in dense geometry); (2) a **problem formalization** — pixel-wise learning to defer between two estimators with a potentially harmful deferral target, optimal policy = threshold on predicted error difference (importing Okati/Jitkrittum to dense regression); (3) a **minimal method** — lightweight comparator head + calibrated risk-controlled threshold with abstention; (4) an **evaluation protocol contribution** — net utility vs. keep-current, harm rate, risk–coverage, cross-backbone and cross-sequence, which subsequent temporal-fusion papers can adopt. Position CODD as the closest prior in the second paragraph of the intro, not in related work only. The surgical application is a deployment domain, not the novelty claim.

---

## Open questions and known weaknesses of this review

- **Blocked or only partially read details:** Khan et al. method internals; ManyDepth mask formulation specifics; ProDepth method details (arXiv abstract available; paper body not read); Spyropoulos/DSF loss details; DynamicStereo method section; VeloDepth (3DV 2026, correction-based propagation — snippet only).
- **Name resolutions and dead ends (from researcher notes):** "XStereo" → XR-Stereo (arXiv:2309.04183); "Marin et al. SGM meta-selection", "FMFNet", and "Jiang et al. selective regression under distribution shift" could not be located and are likely misremembered names; PDC-Net's arXiv ID was not independently verified.
- **Absence claims are bounded by search:** ~100 queries across four researchers, terminology-varied; still, a workshop paper or 2025–26 preprint doing pixel-wise relative-error selection could exist under unusual terminology. Recommended pre-submission checks: citation-chase CODD's citing papers (Semantic Scholar), DeferredSeg's citing papers, and "video depth" + "learning to defer" quarterly.
- **CODD's exact KITTI behavior and the oracle numbers** were taken from the ar5iv HTML full text (T1, read-in-full) — high confidence, but per-number verification against the published WACV version is advised before quoting in a paper.


---

## Consolidated Sources / URL Map

### Closest temporal stereo and video-depth sources
1. Li et al., **Temporally Consistent Online Depth Estimation in Dynamic Scenes (CODD)**, WACV 2023, arXiv:2111.09337 — https://arxiv.org/abs/2111.09337 ; HTML used: https://ar5iv.labs.arxiv.org/html/2111.09337 ; project: https://mli0603.github.io/codd
2. Zeng et al., **Temporally Consistent Stereo Matching (TC-Stereo)**, ECCV 2024, arXiv:2407.11950 — https://arxiv.org/abs/2407.11950 ; HTML: https://arxiv.org/html/2407.11950v1 ; code: https://github.com/jiaxiZeng/Temporally-Consistent-Stereo-Matching
3. Zhang, Poggi, Mattoccia, **TemporalStereo: Efficient Spatial-Temporal Stereo Matching Network**, IROS 2023, arXiv:2211.13755 — https://arxiv.org/abs/2211.13755 ; HTML: https://ar5iv.labs.arxiv.org/html/2211.13755 ; code: https://github.com/youmi-zym/TemporalStereo
4. Khan et al., **Temporally Consistent Online Depth Estimation Using Point-Based Fusion**, CVPR 2023, arXiv:2304.07435 — https://arxiv.org/abs/2304.07435 ; code: https://github.com/facebookresearch/TemporallyConsistentDepth ; PDF not parsed: https://openaccess.thecvf.com/content/CVPR2023/papers/Khan_Temporally_Consistent_Online_Depth_Estimation_Using_Point-Based_Fusion_CVPR_2023_paper.pdf
5. Cheng, Yang, Li, **Stereo Matching in Time: 100+ FPS Video Stereo Matching for Extended Reality (XR-Stereo)**, WACV 2024, arXiv:2309.04183 — https://arxiv.org/abs/2309.04183 ; WACV page: https://openaccess.thecvf.com/content/WACV2024/html/Cheng_Stereo_Matching_in_Time_100_FPS_Video_Stereo_Matching_for_WACV_2024_paper.html ; code: https://github.com/za-cheng/XR-Stereo
6. Jing et al., **Match Stereo Videos via Bidirectional Alignment / BiDAStereo**, ECCV 2024 / TPAMI 2026 line — https://arxiv.org/abs/2403.10755 ; https://arxiv.org/abs/2409.20283 ; project: https://tomtomtommi.github.io/BiDAStereo/ ; code: https://github.com/MatchLab-Imperial/bidavideo
7. Jing et al., **Stereo Any Video: Temporally Consistent Stereo Matching**, ICCV 2025 — https://arxiv.org/abs/2503.05549 ; project: https://tomtomtommi.github.io/StereoAnyVideo/
8. Karaev et al., **DynamicStereo**, CVPR 2023 — https://arxiv.org/abs/2305.02296 ; project: https://dynamic-stereo.github.io/
9. Watson et al., **The Temporal Opportunist: Self-Supervised Multi-Frame Monocular Depth (ManyDepth)**, CVPR 2021 — https://arxiv.org/abs/2104.14540 ; PDF not parsed: https://openaccess.thecvf.com/content/CVPR2021/papers/Watson_The_Temporal_Opportunist_Self-Supervised_Multi-Frame_Monocular_Depth_CVPR_2021_paper.pdf
10. **ProDepth** (ECCV 2024), probabilistic mono/multi-frame depth fusion/modulation (abstract read; method body not read) — https://arxiv.org/abs/2407.09303
11. Luo et al., **Consistent Video Depth Estimation**, SIGGRAPH 2020 — https://arxiv.org/abs/2004.15021 ; project: https://roxanneluo.github.io/Consistent-Video-Depth-Estimation/
11. Kopf, Rong, Huang, **Robust Consistent Video Depth Estimation**, CVPR 2021 — https://arxiv.org/abs/2012.05901 ; project: https://robust-cvd.github.io/
12. Wang et al., **Neural Video Depth Stabilizer / NVDS+**, ICCV 2023 / TPAMI 2024 — https://arxiv.org/abs/2307.08695 ; code: https://github.com/RaymondWang987/NVDS
13. Yasarla et al., **MAMo: Leveraging Memory and Attention for Monocular Video Depth Estimation**, ICCV 2023 — https://arxiv.org/abs/2307.14336
14. Yasarla et al., **FutureDepth**, ECCV 2024 — https://arxiv.org/abs/2403.12953
15. Chen et al., **Video Depth Anything**, CVPR 2025 — https://arxiv.org/abs/2501.12375 ; project: https://videodepthanything.github.io/ ; code: https://github.com/DepthAnything/Video-Depth-Anything
16. Hu et al., **DepthCrafter**, 2024 — https://arxiv.org/abs/2409.02095 ; project: https://depthcrafter.github.io
17. Tosi et al., **A Survey on Deep Stereo Matching in the Twenties**, IJCV 2025 — https://link.springer.com/article/10.1007/s11263-024-02331-0

### Stereo confidence, uncertainty, and candidate selection
18. Poggi et al., **On the Confidence of Stereo Matching in a Deep-Learning Era**, TPAMI 2021 — https://arxiv.org/abs/2101.00431
19. Hu & Mordohai, **A Quantitative Evaluation of Confidence Measures for Stereo Vision**, TPAMI 2012 — https://researchwith.stevens.edu/en/publications/a-quantitative-evaluation-of-confidence-measures-for-stereo-visio ; DOI: https://doi.org/10.1109/TPAMI.2012.46
20. Spyropoulos & Mordohai, **Ensemble Classifier for Combining Stereo Matching Algorithms**, 3DV 2015 — https://researchwith.stevens.edu/en/publications/ensemble-classifier-for-combining-stereo-matching-algorithms/
21. Spyropoulos & Mordohai, **Correctness Prediction, Accuracy Improvement and Generalization of Stereo Matching Using Supervised Learning**, IJCV 2016 — https://researchwith.stevens.edu/en/publications/correctness-prediction-accuracy-improvement-and-generalization-of/ ; PDF not parsed: https://mordohai.github.io/public/Spyropoulos_LearningForStereo_IJCV15.pdf
22. Schönberger, Sinha, Pollefeys, **Learning to Fuse Proposals from Multiple Scanline Optimizations in Semi-Global Matching (SGM-Forest)**, ECCV 2018 — https://openaccess.thecvf.com/content_ECCV_2018/html/Johannes_Schoenberger_Learning_to_Fuse_ECCV_2018_paper.html
23. Poggi & Mattoccia, **Deep Stereo Fusion**, 3DV 2016 — https://cris.unibo.it/handle/11585/589263
24. Poggi & Mattoccia, **Learning from Scratch a Confidence Measure**, BMVC 2016 — https://www.bmva-archive.org.uk/bmvc/2016/papers/paper046/index.html
25. Tosi et al., **Beyond Local Reasoning for Stereo Confidence Estimation with Deep Learning**, ECCV 2018 — https://openaccess.thecvf.com/content_ECCV_2018/html/Fabio_Tosi_Beyond_local_reasoning_ECCV_2018_paper.html
26. Mostegel et al., **Using Self-Contradiction to Learn Confidence Measures in Stereo Vision**, CVPR 2016 — https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Mostegel_Using_Self-Contradiction_to_CVPR_2016_paper.html
27. Poggi et al., **Self-adapting Confidence Estimation for Stereo**, ECCV 2020 — https://arxiv.org/abs/2008.06447
28. Jie et al., **Left-Right Comparative Recurrent Model for Stereo Matching**, CVPR 2018 — https://openaccess.thecvf.com/content_cvpr_2018/html/Jie_Left-Right_Comparative_Recurrent_CVPR_2018_paper.html
29. Marin, Zanuttigh, Mattoccia, **Reliable Fusion of ToF and Stereo Depth Driven by Confidence Measures**, ECCV 2016 — https://cris.unibo.it/handle/11585/589254
30. Chen, Wang, Mordohai, **Learning the Distribution of Errors in Stereo Matching for Joint Disparity and Uncertainty Estimation (SEDNet)**, CVPR 2023 — https://arxiv.org/abs/2304.00152 ; code: https://github.com/lly00412/SEDNet
31. Lou et al., **ELFNet: Evidential Local-global Fusion for Stereo Matching**, ICCV 2023 — https://arxiv.org/abs/2308.00728 ; CVF: https://openaccess.thecvf.com/content/ICCV2023/html/Lou_ELFNet_Evidential_Local-global_Fusion_for_Stereo_Matching_ICCV_2023_paper.html
32. Kendall & Gal, **What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?**, NeurIPS 2017 — https://arxiv.org/abs/1703.04977
33. Ilg et al., **Uncertainty Estimates and Multi-Hypotheses Networks for Optical Flow**, ECCV 2018 — https://openaccess.thecvf.com/content_ECCV_2018/html/Eddy_Ilg_Uncertainty_Estimates_and_ECCV_2018_paper.html

### Selective prediction, learning to defer, risk control, and net utility
34. El-Yaniv & Wiener, **On the Foundations of Noise-free Selective Classification**, JMLR 2010 — https://jmlr.csail.mit.edu/papers/v11/el-yaniv10a.html
35. Geifman & El-Yaniv, **Selective Classification for Deep Neural Networks**, NeurIPS 2017 — https://arxiv.org/abs/1705.08500
36. Geifman & El-Yaniv, **SelectiveNet**, ICML 2019 — https://arxiv.org/abs/1901.09192
37. Zaoui, Denis & Hebiri, **Regression with Reject Option**, NeurIPS 2020 — https://papers.nips.cc/paper/2020/hash/e8219d4c93f6c55c6b10fe6bfe997c6c-Abstract.html
38. Cortes, DeSalvo & Mohri, **Learning with Rejection**, 2016 — https://cs.nyu.edu/~mohri/pub/rej.pdf
39. Madras, Pitassi & Zemel, **Predict Responsibly: Improving Fairness and Accuracy by Learning to Defer**, NeurIPS 2018 — https://arxiv.org/abs/1711.06664
40. Mozannar & Sontag, **Consistent Estimators for Learning to Defer to an Expert**, ICML 2020 — https://arxiv.org/abs/2006.01862
41. Verma & Nalisnick, **Calibrated Learning to Defer with One-vs-All Classifiers**, ICML 2022 — https://proceedings.mlr.press/v162/verma22c.html
42. Okati, De & Gomez-Rodriguez, **Differentiable Learning Under Triage**, NeurIPS 2021 — https://arxiv.org/abs/2103.08902
43. Jitkrittum et al., **When Does Confidence-Based Cascade Deferral Suffice?**, NeurIPS 2023 — https://arxiv.org/abs/2307.02764
44. Sun et al., **DeferredSeg: Deferral-aware Medical Image Segmentation**, arXiv 2026 — https://arxiv.org/abs/2604.12411
45. Angelopoulos et al., **Learn then Test**, 2021 — https://arxiv.org/abs/2110.01052
46. Angelopoulos et al., **Conformal Risk Control**, 2022 — https://arxiv.org/abs/2208.02814
47. Lekeufack et al., **Conformal Decision Theory: Safe Autonomous Decisions from Imperfect Predictions**, ICRA 2024 — https://arxiv.org/abs/2310.05921
48. Vickers & Elkin, **Decision Curve Analysis**, Medical Decision Making 2006 — https://journals.sagepub.com/doi/pdf/10.1177/0272989X06295361
49. Vickers, Van Calster & Steyerberg, **Net benefit approaches to the evaluation of prediction models**, BMJ 2016 — https://www.bmj.com/content/352/bmj.i6
50. Jaeger et al., **A Call to Reflect on Evaluation Practices for Failure Detection**, ICLR 2023 — https://arxiv.org/abs/2211.15259
51. Janner et al., **When to Trust Your Model: Model-Based Policy Optimization**, NeurIPS 2019 — https://arxiv.org/abs/1906.08253

### Temporal-memory rejection in VOS, tracking, SLAM, VSR, video segmentation, and flow
52. Liu et al., **Learning Quality-aware Dynamic Memory for Video Object Segmentation (QDMN)**, ECCV 2022 — https://arxiv.org/abs/2207.07922
53. Cheng & Schwing, **XMem**, ECCV 2022 — https://arxiv.org/abs/2207.07115 ; project: https://hkchengrex.com/XMem/
54. Oh et al., **Video Object Segmentation using Space-Time Memory Networks (STM)**, ICCV 2019 — https://arxiv.org/abs/1904.00607
55. Yang et al., **Associating Objects with Transformers for VOS (AOT)**, NeurIPS 2021 — https://arxiv.org/abs/2106.02638
56. Zhou et al., **RMem: Restricted Memory Banks Improve Video Object Segmentation**, CVPR 2024 — https://openaccess.thecvf.com/content/CVPR2024/html/Zhou_RMem_Restricted_Memory_Banks_Improve_Video_Object_Segmentation_CVPR_2024_paper.html
57. Zhang et al., **UpdateNet: Learning the Model Update for Siamese Trackers**, ICCV 2019 — https://arxiv.org/abs/1908.00855
58. Dai et al., **High-Performance Long-Term Tracking with Meta-Updater (LTMU)**, CVPR 2020 — https://openaccess.thecvf.com/content_CVPR_2020/html/Dai_High-Performance_Long-Term_Tracking_With_Meta-Updater_CVPR_2020_paper.html ; arXiv: https://arxiv.org/abs/2004.00305
59. Teed & Deng, **DROID-SLAM**, NeurIPS 2021 — https://arxiv.org/abs/2108.10869 ; code: https://github.com/princeton-vl/droid-slam
60. Bescos et al., **DynaSLAM**, RA-L 2018 — https://arxiv.org/abs/1806.05620
61. Mur-Artal & Tardós, **ORB-SLAM2**, T-RO 2017 — https://arxiv.org/abs/1610.06475
62. Chan et al., **BasicVSR / IconVSR**, CVPR 2021 — https://arxiv.org/abs/2012.02181
63. Chan et al., **BasicVSR++**, CVPR 2022 — https://arxiv.org/abs/2104.13371
64. Chan et al., **RealBasicVSR**, CVPR 2022 — https://arxiv.org/abs/2111.12704
65. Isobe et al., **Video Super-Resolution with Recurrent Structure-Detail Network (RSDN)**, ECCV 2020 — https://arxiv.org/abs/2008.00455
66. Xie et al., **Mitigating Artifacts in Real-World Video Super-Resolution Models**, AAAI 2023 — https://arxiv.org/abs/2212.07339
67. Habibian et al., **Skip-Convolutions for Efficient Video Processing**, CVPR 2021 — https://arxiv.org/abs/2104.11487
68. Parger et al., **DeltaCNN**, CVPR 2022 — https://arxiv.org/abs/2203.03996
69. Zhu et al., **Deep Feature Flow for Video Recognition**, CVPR 2017 — https://arxiv.org/abs/1611.07715
70. Jain et al., **Accel: Corrective Fusion Network for Efficient Semantic Segmentation on Video**, CVPR 2019 — https://openaccess.thecvf.com/content_CVPR_2019/papers/Jain_Accel_A_Corrective_Fusion_Network_for_Efficient_Semantic_Segmentation_on_CVPR_2019_paper.pdf ; project: https://www.samvitjain.com/accel/
71. Li et al., **Low-Latency Video Semantic Segmentation**, CVPR 2018 — https://arxiv.org/abs/1804.00389
72. Xu et al., **Dynamic Video Segmentation Network (DVSNet)**, CVPR 2018 — https://arxiv.org/abs/1804.00931
73. Nilsson & Sminchisescu, **Semantic Video Segmentation by Gated Recurrent Flow Propagation (GRFP)**, CVPR 2018 — https://arxiv.org/abs/1612.08871
74. Meister et al., **UnFlow**, AAAI 2018 — https://ojs.aaai.org/index.php/AAAI/article/view/12276
75. Wang et al., **Occlusion Aware Unsupervised Learning of Optical Flow**, CVPR 2018 — https://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Occlusion_Aware_Unsupervised_CVPR_2018_paper.html
76. Hur & Roth, **MirrorFlow**, ICCV 2017 — https://openaccess.thecvf.com/content_iccv_2017/html/Hur_MirrorFlow_Exploiting_Symmetries_ICCV_2017_paper.html
77. Jonschkowski et al., **What Matters in Unsupervised Optical Flow**, ECCV 2020 — https://arxiv.org/abs/2006.04902
78. Zhao et al., **MaskFlownet**, CVPR 2020 — https://www.microsoft.com/en-us/research/publication/maskflownet-asymmetric-feature-matching-with-learnable-occlusion-mask/

### Verification and blocked-source notes
- PDF-only method details were not parsed unless an HTML/OpenAccess abstract page existed; this affects Khan et al. CVPR 2023 method details, ManyDepth mask details, VeloDepth, several DSF/Spyropoulos internals, and some VSR/SLAM implementation thresholds. ProDepth has an arXiv abstract (https://arxiv.org/abs/2407.09303), but its method body was not read.
- Subagent verifier failed due account rate limit; this cited version was assembled by the lead from the four research notes. See provenance sidecar.
