# ARGOS v2 geometry-v1 architecture

1. A frozen stereo estimator supplies left/right RGB, raw positive-left disparity, and its validity mask through a backbone-independent interface.
2. Frozen SEA-RAFT estimates target-to-source flow for each current/anchor pair.
3. Every CS1, CS2, CS4, and CS8 flow is direct current-to-anchor; flows are never composed and future frames are inaccessible.
4. BiDA-style pull warping samples the source at `grid + flow`, bilinearly with zero padding and `align_corners=True`.
5. The memory bank stores immutable independently generated raw disparity, raw validity, RGB provenance, frame ID/index, and age metadata only.
6. Seventeen universal candidate features encode raw/candidate geometry, residual statistics, support, forward-backward confidence, age, consensus, and raw provenance.
7. A single shared 60,739-parameter candidate encoder scores all anchors. There is no backbone ID or internal stereo evidence.
8. Invalid anchors receive negative-infinity selection scores. Learned retrieval chooses one valid anchor per output location.
9. Pairwise soft fusion interpolates between current raw and the chosen aligned raw anchor only when the frozen probability and utility criteria accept it.
10. No valid/accepted candidate yields the input raw disparity exactly.
11. After output construction, only the current raw stereo state is appended. There is no corrected-state recurrence or fused-state writeback.
12. The spatial critic and hard-negative branches are excluded. Bounded recurrent H4 is an isolated architectural baseline, not part of inference.
