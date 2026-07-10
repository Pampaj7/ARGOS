# Repository Component Map

| Mechanism | Repository | Source file | Original shape | Wrapper shape | Direct original code used | Causal | Future frames | Probe result |
|---|---|---|---|---|---|---|---|---|
| BiDA flow warp | `external/bidavideo` | `train_utils/losses.py::flow_warp` | tensor `[B,C,H,W]`, flow `[B,2,H,W]` | same | yes | yes if past-only | no | executed |
| BiDA propagation/fusion | `external/bidavideo` | `models/core/bidastabilizer.py::forward` | sequence `[B,T,C,H,W]` | none | inspected | no | yes | reference-only, backward/future pass |
| RAFT flow | `external/RAFT` | `demo.py`, `core/raft.py` | RGB pair `[B,3,H,W]` | same at 144x180 | yes | pairwise causal | no | see `raft_vs_searaft.csv` |
| SEA-RAFT flow | `external/SEA-RAFT` | `custom.py`, `core/raft.py` | RGB pair `[B,3,H,W]` | same at 144x180 | yes | pairwise causal | no | see `raft_vs_searaft.csv` |
| PPMStereo QK similarity | `external/PPMStereo` | `ppmstereo.py::compute_qk_similarity` | `[B,C,T,H,W]` | adapter features `[B,6,T,H,W]` | compared against original method | past-only in probe | no | executed |
| PPMStereo top-k/modulation | `external/PPMStereo` | `ppmstereo.py` lines 504-541 | score `[B,1,T,T]` | candidate score `[B,M]` | math preserved | past-only in probe | no | executed, readout diagnostic only |
| PPMStereo flash-attn readout | `external/PPMStereo` | `ppmstereo.py::forward_update_block` | cost/update features | none | no | sequence-level | possible | reference-only, cost-volume coupled |
| EndoStreamDepth Mamba/xLSTM | `external/EndoStreamDepth` | `mamba.py`, `xlstm_block.py` | DPT tokens `[B,L,C]` | none | import/instantiate tested | yes | no | dependency-coupled |
| Endo temporal loss | `external/EndoStreamDepth` | `util/loss.py::temporal_consistency_loss` | `[B,T,H,W]` | cache disparity `[B,T,144,180]` | yes | training loss | no | executed |
