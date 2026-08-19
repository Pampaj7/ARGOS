# ARGOS transfer closure v7

Static paper-ready closure of completed evidence only. No inference, tuning, calibration, or reselection was run.

| Claim | Verdict | Evidence boundary |
|---|---|---|
| Unseen-backbone transfer | **PASS** | Frozen H4 on SCARED-C D7: both unseen backbones improve and the attested gate passes. |
| External OOD transfer | **NOT_CONFIRMED** | D4D has no reference; DRENDS is a single ToF-reference pilot with a severe refined tail. |
| Joint unseen-backbone + external OOD | **UNAVAILABLE** | No completed evaluation intersects both axes. |

D7 is historical non-blind confirmation, not a fresh blind holdout. Its source manifest had a D2-valued `scope` metadata defect; the D7 attestation verifies the D7/H4 payload scope without modifying the immutable manifest. The run is frozen H4 only: no tuning or reselection. Oracle rows are excluded.

## Table conventions

`paper_transfer_table.csv` is the compact citable table. D7 uses macro-sequence EPE over four sequences. D4D entries are equal-frame no-reference diagnostics, not accuracy claims. DRENDS is a spatial ToF-reference pilot: central errors improve, but refined RMSE worsens and maxima hit 1,000 px / 10,000 mm, so it cannot support an external-OOD pass. SERV-CT temporal H4 is not applicable because static stereo pairs have no temporal adjacency.

## Provenance (verified SHA-256)

| Artifact | SHA-256 |
|---|---|
| `.../experimental_closure/d7_confirmation/run_manifest.json` | `4646d72ddfa079326ef07cdeddd41d0685a6ea7d757cb37a3da938343d1c3f2b` |
| `.../experimental_closure/d7_confirmation/summary.csv` | `5020bb6e0a6aa85d1fe239f14c08837c79164717d5847e197886514b6a6a8ec4` |
| `.../experimental_closure/protocol/d7_scope_attestation_v1.json` | `06202d38afd1014c2701271f14ff233dc685b0abbb48e09c0a935e9ba262e335` |
| `.../canonical_h4_ood_v7/run_manifest.json` | `c15144e61080c93e644377cda0a8fca2d0c39cbe9682924596bb0453cfd4528c` |
| `.../canonical_h4_ood_v7/external_ood_attestation.json` | `7c1f4e400e35cac48951cf1959b286d76dffdd4f865c0f873ebe5d38de1c92c7` |
| `.../canonical_h4_ood_v7/d4d_no_reference_summary.csv` | `7ba05dfe70626bc692cd02dc6254e019bbffdd353872655e8947d47ab68afdd7` |
| `.../canonical_h4_ood_v7/drends_aggregate_metrics.csv` | `9d65d8a4fb14e847b0bc6b56d04ef8d3cbff56d672994d04ee6030643e4d87fb` |

All full paths and hashes are in `verdicts.json`.
