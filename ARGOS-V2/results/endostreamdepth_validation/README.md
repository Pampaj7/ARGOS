# ARGOS v2 latent-state validation

Decision: **NO-GO for the tested E2-E5 generic ConvGRU adaptation**. This is
not a claim about the faithful EndoStreamDepth Mamba/xLSTM operator, which is
reference-only because the validated environment lacks its compiled
`selective_scan_cuda`/Mamba and `xlstm` dependencies.

The controlled ladder gives small held-out validation gains: 0.00192 px without
BiDA (E2), 0.00232 px for single-scale BiDA state (E3), 0.00267 px for
multi-scale state (E4), and 0.00269 px with reset/forgetting (E5). However, the
mandatory state-usage test fails. On three held-out streams from the three seen
backbones, true-history and zero-state outputs differ by only about
`1.7e-6`--`1.8e-6` px; zero-state EPE is equal or slightly better. Shuffled and
wrong-sequence histories are likewise indistinguishable. The state tensors have
non-zero norms, but the correction decoder ignores their history.

Therefore the apparent gain is attributable to the current-frame/BiDA CNN path,
not exploitation of persistent long-range state. Per the PONYTAIL stage gate,
full seen safety, Fast-FoundationStereo, CREStereo, native-grid evaluation and
E6 dependency installation were not run. The validated learned BiDA t-1 A2
model remains the reference; the explicit PPMStereo study also remains NO-GO,
for unsafe selection rather than state non-use.

Key artifacts:

- audit: `model_design/ENDOSTREAMDEPTH_AUDIT.md`;
- ladder histories/checkpoints: `ladder/E2` through `ladder/E5`;
- three-backbone diagnostics: `state_diagnostics_s2m2_seq1`,
  `state_diagnostics_raft_seq2`, and `state_diagnostics_sa_seq4`;
- consolidated decision: `aggregate_summary.json`;
- runtime/state footprint: `runtime_summary.json`;
- explicit stopped safety stage: `safety_summary.json`.

Exact smoke command:

```bash
/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python scripts/run_endostreamdepth_validation.py --mode smoke --variant E3 --output /tmp/argos_endostream_smoke --clip-length 8 --burn-in 1 --steps 30 --batch-size 1 --workers 0 --learning-rate 0.003 --feature-channels 16 --state-channels 8 --device cuda:1 --no-resume
```

Exact three-backbone training template (`E2`, `E3`, `E4`, `E5` were run):

```bash
/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python scripts/run_endostreamdepth_validation.py --mode train --variant E5 --output results/endostreamdepth_validation/ladder/E5 --backbones S2M2-S RAFT-Stereo StereoAnywhere --clip-length 8 --burn-in 1 --max-train-clips-per-sequence 8 --max-validation-clips-per-sequence 8 --epochs 4 --batch-size 4 --workers 4 --feature-channels 32 --state-channels 16 --learning-rate 0.002 --device cuda:1
```

Exact state diagnostic template:

```bash
/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python scripts/run_endostreamdepth_validation.py --mode diagnose --variant E5 --checkpoint results/endostreamdepth_validation/ladder/E5/checkpoints/best_validation.pt --output results/endostreamdepth_validation/state_diagnostics_raft_seq2 --backbones RAFT-Stereo --validation-sequences dataset_7_keyframe_2 --clip-length 8 --max-diagnostic-clips 4 --workers 0 --device cuda:1
```

Exact tests:

```bash
/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python -m pytest -q model_design/tests/test_endostreamdepth.py model_design/tests/test_temporal_clip_dataset.py model_design/tests/test_bidavideo.py
```
