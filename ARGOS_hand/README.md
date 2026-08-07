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

## `original_h4/` — the original code

Byte-identical copies of the authoritative ARGOS-V2 H4 model, loss, alignment,
dataset helpers, and runner files live under `original_h4/`. Nothing there is
installed or imported by the port; it is a source snapshot for direct review.
The authoritative paths and hashes remain in ARGOS-V2.
