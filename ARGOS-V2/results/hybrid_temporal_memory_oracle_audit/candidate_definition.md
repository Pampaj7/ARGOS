# ARGOS v2 hybrid candidate contract

`C0` is current raw stereo. `CS{1,2,4,8}` are immutable raw disparities aligned directly from their source frame. `CF{1,2}` are disparities produced by the canonical no-learned-stereo-evidence H=4 refiner, then aligned directly from their source frame. Every temporal candidate stores age, raw/corrected provenance, source frame, backbone, BiDA warp support, validity and FB confidence. Current-to-anchor SEA-RAFT flow is inferred directly; consecutive flow chains are never composed.

Strict comparisons intersect all candidate supports. Availability-aware comparisons retain C0 exactly and allow each temporal candidate only on its own valid support.
