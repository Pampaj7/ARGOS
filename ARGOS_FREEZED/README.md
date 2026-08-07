# ARGOS v2 — frozen geometry-v1

`ARGOS_FREEZED` is the canonical immutable geometry-v1 implementation used for future ARGOS v2 paper experiments. It is a sibling of the validated development/provenance repository `ARGOS-V2`, not a child of it.

The method consumes frozen stereo RGB, raw positive-left disparity, and validity; estimates direct current-to-anchor SEA-RAFT flow; performs causal BiDA-style pull alignment to immutable raw CS1/CS2/CS4/CS8 anchors; builds universal shared candidate evidence; retrieves one anchor; applies pairwise soft fusion; falls back exactly to raw when no update is accepted; and writes only the independently generated raw state to memory. It has 60,739 learned parameters. It has no recurrent corrected state, fused writeback, backbone identifier, internal stereo feature, H4 import, or spatial critic.

Install and verify:

```bash
cd /dtu/p1/leopam/ARGOS/ARGOS_FREEZED
/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/pip install -e .
/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python scripts/verify_freeze.py
/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python scripts/smoke_test.py
```

The main checkpoint is `checkpoints/raw_multi_anchor_best_validation.pt`, SHA-256 `40526a32ef6e9a62a3ea2b59e6751a60c441b8190f9b96522e3b12b35895d5cd`.

The immutable boundary comprises the package, scientific configuration, checkpoint and provenance manifests. New work belongs only in `experiments/<experiment_id>/`; runners verify the freeze before execution. The compact validated transfer evidence is under `evidence/`. H4 is documentation/provenance only under `baselines/`.

Limitations: evidence covers same-domain SCARED-C transfer to the evaluated backbones. It does not establish universal backbone independence, external-domain/OOD generalization, clinical safety, risk-controlled intervention, or real-time operation.
