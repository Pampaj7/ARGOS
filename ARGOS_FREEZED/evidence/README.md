# Compact validated evidence snapshot

These files are byte-identical copies from `/dtu/p1/leopam/ARGOS/ARGOS-V2/results/raw_multi_anchor_unseen_backbone_transfer/`. Their hashes are recorded in `FREEZE_MANIFEST.json`.

- `aggregate_summary.json`: `d03d1fcf21bff4755242e674fe4c0079c899e866080c901e73c7247c45c80f61`
- `verdicts.json`: `f004369204ec35ee0159d7448b315c5cf11bc93601295879fec605e78a1ad190`
- `checkpoint_hashes.json`: `36654b86d81863a099ad625b6e17eb72c88b7055bc65c9cfd3db8162bdf3e0f3`
- `reproduction_check.json`: `08a35bc9e1a4b50869b640386d189f562cd78301ec53fe2adfc8371d60a04838`
- `paper_ready_tables.tex`: `b7ecf781783b7cedd4cea9b98f08cee04c0a37d6fb0243512bb26d10ad9ff8ab`

The validated verdict is **FULL UNSEEN-BACKBONE GEOMETRY GO** within SCARED-C. Dataset 7 was held out from the frozen training and calibration protocol, but is not a project-wide pristine test set. CREStereo improved from raw/H4 EPE 0.578407/0.555541 to 0.543687; Fast-FoundationStereo improved from 0.538815/0.523738 to 0.520947. All four sequences improved for both backbones. Strict common-support coverage was 0.973082; empty-support frames were excluded from pixel aggregation and retained as explicit frame diagnostics in the original artifacts.

Selected CS4+CS8 frequency was 0.756593 for CREStereo and 0.784907 for Fast-FoundationStereo; accepted-use CS4+CS8 frequency was approximately 53.4% and 55.8%, respectively. The live SEA-RAFT reproduction passed at 2e-4 EPE tolerance.

Supported claims are improved geometry, a backbone-independent input interface, same-domain transfer to the two evaluated stereo estimators excluded from training, and continued long-range anchor use. This evidence does not establish safety, risk control, OOD robustness, universal backbone independence, or clinical readiness.

Large frame-level CSVs were not copied. They remain at:

- `/dtu/p1/leopam/ARGOS/ARGOS-V2/results/raw_multi_anchor_unseen_backbone_transfer/frame_metrics.csv`
- `/dtu/p1/leopam/ARGOS/ARGOS-V2/results/raw_multi_anchor_unseen_backbone_transfer/per_sequence_metrics.csv`
- `/dtu/p1/leopam/ARGOS/ARGOS-V2/results/raw_multi_anchor_unseen_backbone_transfer/per_backbone_metrics.csv`
