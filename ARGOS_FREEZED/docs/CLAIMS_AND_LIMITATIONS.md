# Claims and limitations

## Supported

- causal, online processing using past raw anchors only;
- frozen stereo input interface without backbone IDs or internal features;
- direct current-to-anchor flow with no composition;
- same-domain SCARED-C transfer to CREStereo and Fast-FoundationStereo, which were excluded from training;
- continued useful selection of long-range CS4/CS8 anchors;
- improved geometry relative to raw and bounded H4 on strict common support.

## Not supported

- universal backbone independence or agnosticism;
- external-domain or OOD geometric robustness;
- clinical safety;
- strict risk-controlled intervention;
- real-time deployment without a dedicated runtime benchmark.

Safety diagnostics describe observed failures only. The negative hard-negative critic result is not a component or guarantee of geometry-v1.
