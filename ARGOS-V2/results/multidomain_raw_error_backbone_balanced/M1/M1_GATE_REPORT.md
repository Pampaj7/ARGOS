# ARGOS v2 — M1 backbone-balanced gate report

## Protocol

The architecture was unchanged: frozen stereo cache, SEA-RAFT, causal BiDA,
and frozen A2; only the 1,107-parameter S1 Raw Error Detector was trained.
SERV-CT, Fast-FoundationStereo, CREStereo, and D4D specimen-3 labels were not
loaded for training, calibration, checkpoint choice, or threshold choice.

Training used three seen backbones in both domains.  D1 used 75% SCARED-C / 25%
D4D specimen-1; D2 used 50% / 50%.  Selection used only held-out SCARED-C and
D4D specimen-2.

## Frozen seen-domain selection result

| candidate | SCARED gain (px) | SCARED coverage | D4D gain (px) | D4D coverage | eligible |
|---|---:|---:|---:|---:|---|
| D1 (75/25) | +0.002805 | 1.251% | -0.133598 | 6.747% | no |
| D2 (50/50) | +0.000018 | 0.0033% | +0.000014 | 0.0112% | no |

D1 fails because it worsens the added seen domain.  D2 meets its safety
numbers only by collapsing to an almost exact identity mapping; it retains
0.19% of the available SCARED A2 gain and has no meaningful coverage.

The fallback code serialized D2 only to preserve deterministic provenance, not
because it passed a promotion gate.  `ratio_selection.json` records both failed
eligibility checks and the frozen artifact hashes.

## Stop decision

**NO-GO for the detector-only M1 experiment.**  Per the preregistered protocol,
there is no SERV-CT, Fast-FoundationStereo, CREStereo, or final-test evaluation
for this failed candidate.  Running them would not be a frozen validation of a
promoted configuration.

The per-backbone D4D scale audit in
`model_design/MULTIDOMAIN_BACKBONE_CACHE_AUDIT.md` identifies why pooling the
new data is not diagnostic: RAFT-Stereo and StereoAnywhere have large frozen
geometric failures on D4D that a bounded A2 proposal cannot repair.  No
domain-specific scale fitting or use of Zivid GT for prediction calibration was
performed.
