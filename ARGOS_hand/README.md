# ARGOS v2 hand package

Two independent trees, kept side by side on purpose.

```
ARGOS_hand/
├── src/argos_v2_hand/   port: standalone, prepared-tensor-only PyTorch heads
├── tests/               CPU smoke test for the port
├── PROVENANCE.md        source files + hashes for the port
└── original_h4/         verbatim ARGOS-V2 H4 training/inference path
```

## `src/argos_v2_hand/` — the port

Standalone, code-only PyTorch port of the ARGOS v2 temporal training heads:
immutable raw multi-anchor retrieval/fusion, bounded H4 CODD-style fusion,
causal BiDA tensor alignment, stereo photometric evidence, and reset policy
helpers.

It accepts prepared tensors only. It intentionally contains no datasets, cache
paths, checkpoints, runners, D7 logic, critics, or results. Full H4 cue
construction (`build_codd_cues(..., include_learned_stereo_evidence=True)`)
requires a caller-supplied `FrozenResNet18Layer1` built from an external frozen
ResNet-18 checkpoint; this package carries no checkpoint.

Install and run the CPU smoke test:

```bash
python -m pip install -e .
pytest -q
```

See `PROVENANCE.md` for exact source files and hashes.

## `original_h4/` — the H4 working copy

Started as byte-identical copies of the authoritative ARGOS-V2 H4 model, loss,
alignment, dataset helpers, and runner files. **It is now hand-edited** and no
longer tracked against `SOURCE_HASHES.sha256` — treat every file here as
working code, not a verified snapshot. The authoritative, unmodified source
stays in ARGOS-V2; diff against it there if you need to know what changed.

Nothing here is installed or imported by the port in `src/`.
