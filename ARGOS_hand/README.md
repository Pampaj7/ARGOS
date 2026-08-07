# ARGOS v2 hand package

Standalone, code-only PyTorch port of the ARGOS v2 temporal training heads: immutable raw multi-anchor retrieval/fusion, bounded H4 CODD-style fusion, causal BiDA tensor alignment, stereo photometric evidence, and reset policy helpers.

It accepts prepared tensors only. It intentionally contains no datasets, cache paths, checkpoints, runners, D7 logic, critics, or results. Full H4 cue construction (`build_codd_cues(..., include_learned_stereo_evidence=True)`) requires a caller-supplied `FrozenResNet18Layer1` built from an external frozen ResNet-18 checkpoint; this package carries no checkpoint.

Install and run the CPU smoke test:

```bash
python -m pip install -e .
pytest -q
```

See `PROVENANCE.md` for exact source files and hashes.
