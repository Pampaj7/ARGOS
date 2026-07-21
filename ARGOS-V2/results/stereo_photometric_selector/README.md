# ARGOS v2 — Frozen stereo-photometric causal-memory selector

This directory records a no-training control for causal BiDA memory.  It uses
the current left/right image pair only to compare the stereo reprojection cost
of two existing disparity candidates: cached raw disparity and the validated
causally aligned t-1 memory.  It never alters a candidate, fits a network, or
writes a dense cache.

## Validation-only choice

`validation/` is the completed sweep on SCARED-C `dataset_7_keyframe_1/2`,
all three seen backbones, 3,819 causal pairs.  It evaluates RGB local-L1 and
ZNCC costs at odd local windows 15/21/31 and margins 0/.002/.005/.01/.02.
The common paired mask includes both candidates' right-image support in
addition to canonical BiDA/GT validity.

The selected safe row is ZNCC 21x21 with margin .002:

| metric | value |
|---|---:|
| raw EPE | 0.158868 px |
| selected EPE | 0.158631 px |
| gain | 0.000237 px |
| oracle recovery | 1.01% |
| coverage | 1.69% |
| false update | 1.55% |
| clean degradation | 0.75% |

This is **NO-GO as a deterministic selector**: it meets the conservative
safety constraint but recovers negligible oracle utility.  The final held-out
and unseen backbones are intentionally not evaluated for this policy.

The implementation and sign/mask contract are documented in
`model_design/STEREO_PHOTOMETRIC_UTILITY_AUDIT.md`; synthetic tests are in
`model_design/tests/test_stereo_photometric.py`.
