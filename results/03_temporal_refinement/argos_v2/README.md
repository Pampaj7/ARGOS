# ARGOS v2 Audit Package

Generated: 2026-07-07

## Status

This is an audit-only package. No training jobs were launched and no new architecture was implemented as part of this audit package.

## Key Verdict

NVDS-lite is not closed from existing evidence. The corrected code has the right flow/mask conventions and passes causality/gradient diagnostics, but all training runs are partial, stale, failed, or missing reusable artifacts.

The smallest correct next action is a one-seed minimal NVDS-lite closure after fixing/adding proper causal streaming/sliding evaluation.

## Files

- `sota_context_extracted.md`: extracted ARGOS v2 SOTA/source-of-truth summary.
- `nvds_lite_audit.md`: current NVDS-lite code/result state.
- `nvds_lite_run_manifest.csv`: row-level run classification.
- `code_component_audit.csv`: component-level code audit.
- `current_jobs_audit.md`: active and historical job/log state.
- `minimal_nvds_lite_closure_plan.md`: smallest experiment set to close NVDS-lite.
- `aligned_local_readiness.md`: readiness for the next aligned-local-only architecture.
- `next_actions.md`: ordered checklist.

## Source of Truth Note

The prompt referenced `sota/`, but the repository contains `SOTA/ARGOS (5).pdf`. That PDF is treated as the ARGOS v2 technical source of truth.

## Current Git State

- Branch: `fable-wildcard-cfr`
- Commit: `b1b80a41d185775cb75e56ad3d707ed41e727790`
- Important modified file outside this audit: `results/03_temporal_refinement/ood/d4d_full_dataset/extraction_log.txt`
- Many untracked experiment scripts/results exist from previous work; do not commit blindly.

## Streaming Evaluator and Aligned-Local Certification

Implemented in this package:

- true causal streaming reference evaluator: `scripts/temporal_refinement/eval_scripts/evaluate_argos_v2_streaming.py`;
- aligned-local-only baselines: `AlignedLocalOnlyFaithful`, `AlignedLocalOnlySafe`;
- synthetic and tiny SCARED smoke tests;
- no long training, no D4D/SERV-CT evaluation.

Key reports:

- `streaming_evaluator_audit.md`
- `streaming_evaluator_implementation_report.md`
- `streaming_evaluator_test_report.md`
- `streaming_evaluator_validation.json`
- `aligned_local_architecture.md`
- `aligned_local_implementation_report.md`
- `aligned_local_test_report.md`
- `one_seed_ladder_ready.md`



## ARGOS v2 One-Seed Ladder (2026-07-07)
Completed on H100 LSF job `28884920`. Classification: `ALIGNMENT_CONFIRMED_PROPAGATION_NOT_CONFIRMED`. Main report: `one_seed_ladder_report.md`.
