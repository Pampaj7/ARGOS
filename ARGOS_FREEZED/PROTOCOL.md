# Frozen protocol

Training dataset IDs are 1, 3, and 6; validation/checkpoint selection uses ID 2; reported frozen test uses ID 7 only after all choices are frozen. Dataset 7 is held out from the reported frozen training and validation protocol, not a project-wide pristine test set.

The geometry checkpoint is selected on validation only. Before any experiment, `scripts/verify_freeze.py` must pass and the experiment manifest must record its hash. Test mode requires a frozen validation decision. Test data must never influence seeds, thresholds, architecture, evidence schema, losses, or checkpoint selection.

Evaluation uses strict common support across GT coverage, raw validity, H4 support, and all multi-anchor supports for any three-way comparison. Empty-support frames contribute zero pixels, remain visible in frame diagnostics, and do not create fabricated values. Valid caches must pass path/hash, dimensions, chronological frame order, disparity sign, finite-value, metadata, and support checks before reuse.

Flow is frozen SEA-RAFT target-to-source `current_to_anchor`, inferred directly for each original anchor. Pull alignment uses bilinear sampling, zero padding, and `align_corners=True`; no flow composition is permitted. Disparity is finite positive-left pixels at 144x180 and is not rescaled during same-grid warping.

Inference runs in eval/inference mode with frozen weights and deterministic seeds. Byte-copied model and extracted alignment tensors require exact agreement; the documented 2e-4 tolerance applies only to EPE reproduction where FP32 SEA-RAFT batching differs. Aggregates are valid-pixel weighted, with per-backbone, per-sequence, and per-seed statistics retained.

Allowed claims: causal online geometry refinement, backbone-independent input interface, same-domain transfer to the evaluated unseen stereo estimators, meaningful long-range anchor use, and improved geometry. Prohibited claims: universal backbone agnosticism, external-domain/OOD robustness, clinical safety, risk-controlled intervention, or real-time deployment without measurement.
