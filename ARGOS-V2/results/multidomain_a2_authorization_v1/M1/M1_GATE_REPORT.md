# ARGOS v2 — M1 gate report: multi-domain A2 plus Raw Error Detector

## Protocol frozen before final-only data

- Proposal: unchanged A2 architecture, trained with balanced SCARED-C plus
  curated D4D Zivid-anchor supervision (S2M2-S only).
- Authorization: unchanged 1,107-parameter S1 Raw Error Detector, trained
  from scratch on the frozen proposal.
- D4D train/calibration/test split: specimen_1 / specimen_2 / specimen_3.
- SCARED-C train/calibration/final split: existing validated sequence split.
- Fast-FoundationStereo, CREStereo, SERV-CT, StereoMIS and D4D specimen_3
  were not loaded for selection or evaluation.

## Selection outcome

The selected D2 candidate (50% D4D sampling) had no eligible calibrated
operating point.  Its least-violating balanced policy was frozen only to make
the failure reproducible; it is **not** a promoted deployment artifact.

At fractional GT coverage 0.50 on the selection domains:

| Domain | Raw EPE | A2 EPE | Authorized EPE | A2 gain retained | Coverage | False update | Clean degradation |
|---|---:|---:|---:|---:|---:|---:|---:|
| SCARED-C validation | 0.191870 | 0.175457 | 0.186953 | 29.96% | 0.588% | 0.063% | 0.028% |
| D4D specimen_2 | 0.882704 | 0.550069 | 0.882704 | 0.00% | 0.00% | 0.00% | 0.00% |

The predeclared SCARED requirement is at least 70% A2-gain retention with
safe, nonzero coverage.  D4D requires nonzero conservative coverage and no
harm.  Both cannot be met by this raw-error authorization target.

## Decision

**NO-GO for `A2 multi-domain proposal + Raw Error Detector` as a paper
configuration.**  The failure is informative: the detector answers whether
the *raw* prediction is wrong, while the deployment decision is whether the
specific temporal proposal is beneficial.  This is not evidence against the
causal BiDA signal or the multi-domain A2 proposal itself.

No final-only/unseen dataset was evaluated after the failed seen-domain gate.
