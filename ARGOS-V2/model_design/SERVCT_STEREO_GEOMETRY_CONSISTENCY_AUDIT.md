# ARGOS v2 — SERV-CT stereo-geometry consistency audit

After the D4D reference-scale discrepancy, SERV-CT was checked independently
before treating it as a cross-domain geometry benchmark.  This is a frozen
data-validity audit, not a learned model experiment.

`scripts/run_servct_stereo_geometry_audit.py` evaluates the declared
CT-derived positive-left disparity against the matching rectified left/right
images at the canonical 144x180 grid.  It compares only fixed diagnostic GT
scales `{.25,.50,.75,1.0}` on the exact common GT/right/census support.  No GT,
cache, prediction, or training target is changed.

## Result

All 16 available SERV-CT frames have dense support (median 22,903 cache-grid
pixels).  Ternary census selects the declared scale `1.0` in **16/16** frames;
local RGB-L1 selects `1.0` in 13/16 and `.75` in the remaining 3.  Aggregate
costs are also lowest at scale `1.0`:

| GT scale | local RGB-L1 | ternary census |
|---:|---:|---:|
| .25 | .04939 | .18439 |
| .50 | .03930 | .17550 |
| .75 | .02992 | .15052 |
| **1.0** | **.02681** | **.06072** |

Thus, unlike D4D, the declared SERV-CT disparity convention is consistent
with its rectified stereo image correspondence.  The prior SERV-CT failures
of detector/authorization policies are genuine cross-domain calibration and
safety failures, not an identified global GT/image scale mismatch.

Compact reproducible results are in
`results/servct_stereo_geometry_audit/`.
