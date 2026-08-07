# raw_multi_anchor_spatial_safety_critic

ARGOS v2. Veto-only spatial comparative error critic over the frozen raw
multi-anchor proposal. Accepts the frozen proposal or returns raw bit-exact;
never changes the anchor/weight/proposal, never reopens a pixel, never enters
the anchor bank. Seen backbones only (S2M2-S, RAFT-Stereo, StereoAnywhere).
Split by dataset ID: train {1,3,6}, validation {2}, test {7}.

## Result (one line)

**CONDITIONAL GO.** Geometry **GO** (multi-anchor beats bounded CODD-style H=4 on
D7, all backbones/sequences). Safety **CONDITIONAL**: the critic dominates the
ungated proposal on every safety axis, beats H4 (+0.0101 vs +0.0068), transfers
D2→D7 without coverage collapse (7.9% cov vs the scalar gate's 1.1%), and lifts
harm AUROC 0.57→0.69 — but does **not** reach the ≤10% harmful-update bound
(24.7% at the frozen point). Binding limitation: harm calibration, not geometry,
transfer, or coverage.

Full report: **`SPATIAL_SAFETY_CRITIC_AUDIT.md`**. Machine-readable:
`verdicts.json`, `aggregate_summary.json`.

## Pipeline (reproduce)

```
# Phase 1 audit (existing families + protocol)
python scripts/run_raw_multi_anchor_spatial_safety_critic.py --mode audit
python scripts/audit_spatial_critic_existing.py

# Phase 2 train (per family; workers=32 batch=16 validated, workers=8 for cal/eval)
python scripts/run_raw_multi_anchor_spatial_safety_critic.py --mode train --families plane_sweep --epochs 12

# Phase 3 calibrate on dataset 2 (writes freeze manifest)
python scripts/run_raw_multi_anchor_spatial_safety_critic.py --mode calibrate \
    --families geometry,temporal,stereo,plane_sweep --workers 8

# Phase 4 freeze (stereo primary, enrich + hash manifest)
python scripts/freeze_spatial_critic.py

# Phase 5 open dataset 7 once (gated on freeze manifest)
python scripts/run_raw_multi_anchor_spatial_safety_critic.py --mode evaluate --split test \
    --families geometry,temporal,stereo,plane_sweep --workers 8

# verdict assembly
python scripts/assemble_spatial_critic_verdict.py
```

Frozen refiner SHA-256 `40526a...5d5c`; freeze manifest SHA-256 `c620ced...5d24`;
repo commit `998d771`; seed 20260722. Dataset 7 stays closed until the freeze
manifest exists (enforced in `evaluate`; tested in
`model_design/tests/test_spatial_critic_freeze_guard.py`).

## Tests

```
python -m pytest model_design/tests/test_spatial_error_critic.py \
                 model_design/tests/test_spatial_critic_freeze_guard.py -q
```
