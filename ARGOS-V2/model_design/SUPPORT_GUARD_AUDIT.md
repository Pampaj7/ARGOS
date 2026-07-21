# ARGOS v2 — Support Guard Audit

## Frozen composition

D1 adds one Boolean support mask after the already frozen balanced raw-error
authorization. SEA-RAFT, canonical causal BiDA t-1 evidence, A2 proposal,
Raw Error Detector, temperature and balanced thresholds are unchanged.

```text
a_final = a_error AND a_support
d_out = where(a_final, d_raw + update_A2, d_raw)
```

`torch.where` makes every rejected pixel bit-exact raw. The support guard never
changes or rescales the A2 proposal.

## Evidence and D0 motivation

D0 found that the detector penultimate 24-channel feature is almost entirely
in the SCARED-C reference support for CREStereo (92.8% nearest-neighbour
overlap), but outside it for SERV-CT (0%) and D4D (7.8%). Mean shrinkage
Mahalanobis scores were 21.3, 581.8 and 168.8 respectively. The already frozen
ultra-safe probability threshold still authorized 43.3% of SERV-CT and 28.5%
of D4D, ruling out a scalar probability-threshold explanation.

## Exact representation

The primary and only fitted representation is the output of
`RawErrorDetector.encoder`, immediately before its three prediction heads:

```text
[B,24,144,180] float feature map
```

The encoder consumes the existing 17 normalized universal channels documented
in `raw_error_detector.py`. No backbone identity, authorization bit, dataset
identity, cost volume, hidden stereo state, RGB foundation feature, or future
frame enters support fitting.

## Leakage boundary

The frozen raw-error split is reused verbatim:

- support fit: the 13 `train_sequences` and three seen backbones only;
- method/threshold selection: `dataset_7_keyframe_1/2` only;
- seen test: `dataset_7_keyframe_3/4` only;
- Fast-FoundationStereo, CREStereo, SERV-CT, D4D, structured-light anchors and
  StereoMIS are unavailable until the support reference, method and thresholds
  have been serialized and hashed.

Fit and threshold APIs reject any provenance other than `SCARED-C/training`
and `SCARED-C/calibration`. The runner verifies the original frozen artifact
hashes before every phase.

## Minimal method ladder

- G0: no guard;
- G1: sum of squared per-channel standardized distances;
- G2: Ledoit-Wolf shrinkage Mahalanobis distance in standardized feature space;
- G3: mean distance to five nearest vectors in a deterministic compact bank;
- G4: permitted only if G2/G3 validation scores show material complementarity;
  otherwise skipped and recorded as YAGNI.

Statistics use balanced deterministic sampling over backbone, sequence and
clean/error status. The compact k-NN bank is sampled only from this balanced
training representation. No dense prediction or feature cache is written.

Both predeclared granularities are selected with the same SCARED-C-only score
thresholds: pixel masking, and one frame decision obtained by comparing the
median valid pixel score with that threshold. Frame aggregation introduces no
new fitted parameter and avoids salt-and-pepper rejection.

## Threshold protocol

Candidate acceptance quantiles are fixed before OOD evaluation:

```text
90%, 95%, 97.5%, 99%, 99.5%
```

Selection uses only held-out SCARED-C calibration geometry plus transparent
feature perturbations derived from SCARED-C features (scale, standardized
offset and noise). These perturbations test whether the score reacts to a
support shift; they are not claimed to model SERV-CT or D4D.

## Metrics and interpretation

All geometry methods share the exact same GT/raw/warp mask. Frame-level score
summaries are reported alongside the operational pixel mask. A guard is useful
only if it retains source-domain gain and unseen-backbone transfer while
materially reducing the already documented SERV-CT/D4D safety failures. A
near-zero intervention system is not promoted.
