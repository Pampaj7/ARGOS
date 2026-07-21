# ARGOS v2 — Multi-domain Raw Error Detector Audit

## Scientific scope

This experiment changes only the training distribution of the existing S1 Raw
Error Detector. SEA-RAFT, causal BiDA t-1 alignment, A2, its bounded residual,
cache conventions and authorization formulation remain unchanged. The main M1
question is whether supervised exposure to SCARED-C plus sparse, genuine D4D
geometry improves safety on completely untouched SERV-CT.

No support guard, additional representation, larger detector, temporal memory,
or joint A2 fine-tuning belongs to this experiment.

## Genuine supervision inventory

### SCARED-C

The existing `RawErrorDataset` and `TemporalPairDataset` are reused. Targets are
the previously validated corrected temporal pseudo-disparity, resized to the
144x180 cache grid with

```text
resize(disparity * valid) / resize(valid)
```

and disparity magnitude multiplied by `180 / W_native`. Primary GT coverage is
0.50. This is internal processed supervision, not official native structured-
light GT.

The existing split is preserved:

- train: 13 accepted non-dataset-7 sequences;
- calibration: `dataset_7_keyframe_1/2`;
- final seen test: `dataset_7_keyframe_3/4`;
- training backbones: S2M2-S, RAFT-Stereo and StereoAnywhere.

### D4D

Canonical source:

```text
dataset/D4D/processed/keyframe_stereo_gt_curated/
```

`DATASET_CARD.md`, `d4d_keyframe_gt.py` and the benchmark validation establish:

- 362 converted anchors;
- 239 usable curated-pose anchors (237 valid, 2 warning);
- genuine disparity comes from Zivid structured-light scans projected into the
  rectified left endoscope frame;
- GT exists only at acquisition anchors, with roughly 12–42% spatial coverage;
- the non-rigid sequence makes propagation of anchor GT to neighbouring frames
  scientifically invalid.

The validated S2M2 zero-shot package contains 156 usable four-frame causal
windows from specimens 1–3. Each `.npz` contains S2M2 raw disparity on
`[t-3,t-2,t-1,t]`, but `valid_mask` and GT are nonzero only at `t`. Context RGB
is loaded and rectified from the original session using the existing D4D helper.
The detector therefore receives one supervised causal pair `t-1 -> t` per
anchor. Context frames are evidence, never labels.

Training reads `gt_disp[3]` and `valid_mask[3]` from these already validated
shards, then applies the same coverage-weighted resize as the existing OOD
evaluator. The shard GT is the curated Zivid anchor in the disparity-unit and
rectification contract of the stored S2M2 prediction. Its provenance is checked
against `keyframe_stereo_gt_curated`; prediction-derived fields are rejected.

The following are explicitly forbidden as supervision:

- `stereo_depth`;
- IGEV++ predictions;
- the S2M2 raw prediction itself;
- temporally propagated Zivid disparity;
- zero-filled invalid pixels.

M1 uses the available specimen-disjoint partition:

- train: specimen 1 (72 validated S2M2 anchor windows);
- calibration/model selection: specimen 2 (33 windows);
- final D4D test: specimen 3 (51 windows).

No specimen, session or clip crosses these partitions. Only S2M2-S predictions
currently exist in the validated D4D package; no additional backbone prediction
cache is fabricated for this experiment.

### SERV-CT

The validated SERV adapter contains 16 CT-derived disparity frames in two
eight-frame weak-sparse replays. GT is positive left-reference disparity at the
cache grid. There are 14 causal adjacent pairs in total.

For M1, neither experiment may be constructed, loaded, calibrated or inspected
before the detector, temperature and operating modes are frozen. Both are then
used exactly once as the fully unseen domain.

For the optional M2 fold only:

- train: `honest_train__Experiment_1` (7 causal pairs);
- calibration: `honest_test__Experiment_2` (7 causal pairs);
- D4D specimens 1–3 remain completely untouched until M2 is frozen.

SERV-CT is too small and static to support a claim of dense temporal training.
M2 is therefore a limited domain-exposure control, not a temporal-learning
experiment and not an independent SERV-CT test.

The validated D4D and SERV-CT packages expose S2M2-S predictions only. A joint
`unseen domain + unseen backbone` geometry test is therefore unavailable without
generating new stereo predictions, which this task forbids. Fast-FoundationStereo
and CREStereo remain genuine unseen-backbone checks inside held-out SCARED-C.

## Predeclared M1 ladder

- M0/D0: reuse the validated SCARED-C-only detector and results;
- M1/D1: 75% SCARED-C, 25% D4D anchor samples;
- M1/D2: 50% SCARED-C, 50% D4D anchor samples.

Sampling is explicit rather than dataset concatenation. Within SCARED-C it is
balanced over backbone and sequence; within D4D it is balanced over sessions.
Pixel supervision is balanced over clean (`<=0.5 px`), moderate (`0.5–3 px`)
and large-error (`>3 px`) strata. Indifferent classification targets within
0.1 px of the 0.5 px boundary remain excluded.

The existing S1 detector is used unchanged: 17 universal inputs, 24 hidden
channels, three 1x1 heads, 1,107 trainable parameters. Loss mode A4 and
false-positive cost 5 are retained. Only detector parameters receive gradients.

## Selection and calibration

Checkpoint selection uses the domain-equal mean of SCARED-C calibration loss
and added-domain calibration loss. Ratio selection uses only held-out SCARED-C
and D4D specimen 2 safety/geometry. Temperature and balanced/ultra-safe
thresholds are fitted only on balanced samples from those same calibration
domains.

M1 artifacts are serialized and hashed before any SERV-CT, unseen-backbone or
D4D specimen-3 final evaluator is constructed. Fast-FoundationStereo and
CREStereo never enter training, checkpoint selection or calibration.

## Interpretation limits

- D4D training is sparse anchor-local supervision, not dense D4D temporal GT.
- Domain diversity is partially confounded with S2M2-only availability on D4D;
  this is recorded rather than hidden.
- Lower temporal inconsistency is not evidence of better geometry.
- Zero authorization on SERV-CT is safety-only, not a strong generalization GO.
- M2 and a capacity control are run only if M1 leaves the scientific hypothesis
  unresolved; M3 is outside the minimal required test.

## Tiny overfit smoke (2026-07-15)

The real-data M1 smoke used one SCARED-C backbone/sequence, four D4D specimen-1
anchors, twelve epochs and 32 explicitly balanced samples per epoch. The first
epoch loss was 5.3260 and the final/best epoch mean was 2.2577 (57.61% lower).
Gradients were non-zero, all values remained finite, peak allocated GPU memory
was 214,242,816 bytes and the detector retained exactly 1,107 trainable
parameters. The temporary smoke directory was deleted after this check.
