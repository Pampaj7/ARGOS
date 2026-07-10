# ARGOS v2 Model Design

This folder owns reusable model-design logic for ARGOS v2.

Keep this separate from:

- `component_probes/`: experiments, CSVs, contact sheets, and probe runners.
- `scripts/`: cache generation and dataset plumbing.
- `external_code_backbone_needed/`: audited external-code export and notes.

Current contents:

- `external_components/`: thin, source-attributed adapters for external mechanisms that may enter the model design.

Rule of thumb: if code is meant to be imported by a future ARGOS v2 model, it belongs here. If it only measures a probe, it stays in `component_probes/`.

