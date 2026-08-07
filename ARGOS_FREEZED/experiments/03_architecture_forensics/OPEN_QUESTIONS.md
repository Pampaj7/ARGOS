# Open or deliberately bounded details

- The public frozen `step` API accepts `current_right_rgb` for the stereo-interface contract, but the temporal adapter does not use it after the stereo estimator has produced `current_raw_disparity`; the upstream stereo preprocessing is outside this package. [IMPLEMENTED]
- The frozen source exposes `timestamp` in `RawAnchor`, but eviction and lookup are index-based; timestamp is provenance only. [IMPLEMENTED]
- `forward_backward_consistency.valid` is computed but is not itself included in `MultiAnchorEvidence.available`; candidate availability is `candidate_valid & warp_support`, while the continuous FB confidence is an input/output diagnostic. [IMPLEMENTED]
- `photometric_residual`, disparity disagreement, flow magnitude, and gradient intermediates are computed by helper code but are not among the 17 channels passed to the CNN. [IMPLEMENTED]
- `torch.nanmedian` on a pixel with no available candidate is protected by `torch.nan_to_num` in the frozen model. The exact backend warning behavior, if any, is not part of the scientific contract. [IMPLEMENTED]
- A formal closed-form FLOP count and measured end-to-end runtime are not present in the inspected frozen sources. [UNKNOWN — NOT PRESENT IN THE INSPECTED SOURCES]
- The source loss file is the canonical loss implementation; no additional hidden loss term was found in the inspected runner. [IMPLEMENTED]
- The exact random augmentation policy is not used by the frozen inference package. Training-time sample construction in the validated runner performs a deterministic crop selected by a seeded NumPy generator; no image augmentation was found. [IMPLEMENTED]
