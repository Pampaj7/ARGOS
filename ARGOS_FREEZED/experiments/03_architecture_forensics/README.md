# ARGOS v2 geometry_v1 architecture forensics

This is a read-only reconstruction of the frozen ARGOS v2 geometric method.  It does not modify the frozen workspace or ARGOS-V2, does not train, does not run a full evaluation, and does not open dataset 7.  All statements are tagged in the main report as `[IMPLEMENTED]`, `[TESTED]`, `[CONFIGURED]`, `[EMPIRICALLY VALIDATED]`, or `[INTERPRETATION]`.

Primary source: `/dtu/p1/leopam/ARGOS/ARGOS_FREEZED`.  Provenance was checked against the corresponding ARGOS-V2 model, alignment, runner, loss, and targeted SOTA notes.  The frozen verifier passed before this audit: `PASS ARGOS v2 geometry_v1 (24 immutable hashes)`.

The main deliverable is [ARCHITECTURE_FORENSICS.md](ARCHITECTURE_FORENSICS.md), with machine-readable tables and paper-ready LaTeX fragments beside it.  `docs/METHOD_LATEX.tex` was not modified.

Dataset 7 was not opened.  No training or full evaluation was run.
