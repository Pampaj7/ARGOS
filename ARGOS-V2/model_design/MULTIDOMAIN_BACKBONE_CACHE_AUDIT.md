# ARGOS v2 — multi-domain backbone-cache audit

## Purpose

The previous multi-domain raw-error study is not a clean test of
domain-and-backbone generalization: its D4D and SERV-CT samples expose only
the precomputed **S2M2-S** disparity.  Consequently an apparent domain result
is confounded with the absence of RAFT-Stereo and StereoAnywhere examples
outside SCARED-C.

This document authorizes a minimal prerequisite only: create separate,
immutable prediction caches for the already validated frozen RAFT-Stereo and
StereoAnywhere wrappers on the genuinely supervised OOD frames.  This is data
preparation, not a new architecture or a change to BiDA/A2/detectors.

## Ground-truth provenance

| Domain | Supervision that may be used | What must not be used as GT |
|---|---|---|
| SCARED-C | existing corrected temporal pseudo-GT | — |
| D4D | curated, rectified Zivid anchor disparity and validity only | `stereo_depth`, IGEV/IGEV++, or any stereo prediction |
| SERV-CT | existing CT-derived disparity from the prepared manifest | any temporal replay assumption as additional GT |

D4D context frames remain unsupervised.  The only supervised pair is causal
`t-1 -> t` at a curated anchor.  Stereo images are rectified using the same
left/right YAML maps as the validated `run_d4d_context_shards.py` pipeline.
The target grid is 144x180 and all cached disparity is positive left disparity
in cache-grid pixels:

`d_cache = resize(d_native, 144x180) * 180 / W_native`.

## Cache contract

The new cache root is intentionally separate from SCARED-C caches:

```text
cache_multidomain_backbones/<backbone>/<domain>/
    disparity.npy       [T,144,180] float16
    valid_mask.npy      [T,144,180] uint8
    frame_ids.npy       [T] string
    frame_manifest.csv  exact source image / sequence / temporal order
    metadata.json
    .complete
```

Each cache is published atomically only after its compact integrity check
passes.  It records source paths, rectification policy, wrapper checkpoint,
frame IDs, and a partial checkpoint hash.  It never overwrites existing
completed caches unless `--force` is explicitly supplied.

## Leakage and scientific use

* Building a prediction cache never reads GT values; GT paths are retained only
  as provenance in the original manifests.
* A cache may contain a final-test frame, but it confers no label or model
  update.  A later training experiment must still enforce specimen/sequence
  splits and may load only its declared training partition.
* Fast-FoundationStereo and CREStereo remain unseen and are not created by this
  prerequisite.
* SEA-RAFT and canonical BiDA are not involved in cache building and are
  unchanged.

## Acceptance checks

For a smoke cache, verify: (1) left and right source paths differ; (2) source
images have matching shape after rectification; (3) output is finite and
positive wherever prediction-valid; (4) every output has the canonical shape
and unit conversion; and (5) frame identity and temporal ordering match the
source manifest.  A successful smoke directory is deleted before any full run.

## Interpretation boundary

Creating these caches does **not** promote multi-domain training, invalidate
earlier results, or claim that a detector is OOD-safe.  It removes a concrete
backbone confound so the next leave-one-domain-out A2/proposal-authorizer study
can be scientifically interpretable.

## First validation finding (M1 calibration only)

The cache itself passes all tensor and source-pair checks, but cache-grid D4D
specimen-2 validation exposes a separate issue that must not be hidden by
pooling backbones.  On the identical Zivid-valid pixels:

| frozen backbone | raw EPE (px) | raw median (px) | GT median (px) | raw/GT correlation |
|---|---:|---:|---:|---:|
| S2M2-S | 0.874 | 15.31 | 16.27 | 0.959 |
| RAFT-Stereo | 12.452 | 3.95 | 16.27 | 0.953 |
| StereoAnywhere | 7.936 | 7.59 | 16.27 | 0.006 |

The RAFT prediction is highly correlated with GT but has an approximately
global scale mismatch on this D4D camera; StereoAnywhere is also unsuitable
as a geometrically comparable D4D candidate under its frozen zero-shot
checkpoint.  The frozen A2 proposal is bounded and cannot repair a 8--12 px
global backbone failure.  Therefore this data cannot fairly answer whether
multi-domain *authorization* transfers: it makes the detector absorb an
unfixable proposal/backbone-scale error rather than a causal safety signal.

This is a protocol finding, not an invitation to calibrate per-domain using
Zivid GT.  No such calibration was applied.  The correct immediate result is
to retain per-backbone reporting and stop the M1 detector study before SERV-CT
or unseen-backbone evaluation when its seen-domain gate fails.
