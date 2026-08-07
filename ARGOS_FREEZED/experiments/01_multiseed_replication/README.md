# Planned multi-seed replication

Scientific question: **Does the frozen immutable raw multi-anchor architecture reproduce its geometric gain across multiple independently trained retrieval/fusion seeds?**

This directory is prepared only; no run has been launched. Training IDs are 1, 3, and 6; validation ID is 2; test ID is 7. Choices and validation decisions must be frozen before test is opened.

Seeds `20260722`, `20260723`, and `20260724` are deterministic and were selected before any future dataset-7 evaluation. `20260722` is the existing ARGOS v2 convention; the next two consecutive integers add independent initializations without outcome-based selection.

Only the small 60,739-parameter shared retrieval/fusion module may train. Stereo estimators, SEA-RAFT, evidence schema, architecture, losses, thresholds, anchor ages, validation rule, and test protocol remain frozen. Run implementation is intentionally deferred until the experiment is approved; YAGNI avoids duplicating the validated trainer now.
