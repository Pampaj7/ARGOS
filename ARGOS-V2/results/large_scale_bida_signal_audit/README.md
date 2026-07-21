# ARGOS v2 large-scale causal BiDA signal audit

Training-free evaluation over every causal pair in all 17 accepted SCARED-C sequences. It uses only frozen disparity caches, frozen SEA-RAFT, and canonical causal BiDA warping.

**Classification: STRONG SIGNAL**

| Role | Backbone | Raw EPE | Memory EPE | Oracle EPE | Gain | Relative gain | Better pixels | Seq. CI95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| seen | RAFT-Stereo | 0.227736 | 0.227749 | 0.200271 | 0.027464 | 12.06% | 49.80% | [0.022343, 0.046819] |
| seen | S2M2-S | 0.283570 | 0.284168 | 0.250638 | 0.032932 | 11.61% | 49.84% | [0.021778, 0.046972] |
| seen | StereoAnywhere | 0.288453 | 0.287372 | 0.212112 | 0.076341 | 26.47% | 49.74% | [0.036318, 0.317968] |
| unseen | CREStereo | 0.233381 | 0.233410 | 0.207401 | 0.025980 | 11.13% | 49.67% | [0.022162, 0.038762] |
| unseen | Fast-FoundationStereo | 0.231000 | 0.231081 | 0.208208 | 0.022792 | 9.87% | 49.81% | [0.019785, 0.031745] |

The unit of uncertainty is the complete sequence, not the pixel. Geometry is reported on the cache grid at GT coverage >0.50 using coverage-normalized GT resize. Temporal-difference metrics are reported separately and are not interpreted as accuracy. Exact shard commands, PIDs as recorded during execution, and aggregation command are in `run.log`.

Audit trace: a first complete run was superseded before interpretation because its temporal-difference field accidentally used error difference. Geometry/oracle values were recomputed as well; this report is the corrected run, where temporal difference is `abs(raw_t - aligned_raw_t-1)`.
