# Limitations
- Pilot scale: 1 seed, 1500 steps, single predefined TGM weight (1.0); clip-len 8 only
  (16 not run — pilot gate reached first). Frozen SCARED val for selection, frozen test reported.
- Ablation coupling: `current_frame`/`shuffled` use spatial loss (not TGM) — they isolate
  memory/order usage, not TGM×order jointly; shuffled being worst still proves order matters.
- SCARED spatial/safety mild regression under TGM (new-Bad3 +2pt) — a selectivity/safety term
  could recover it.
- D4D temporal metrics are prediction-space / RAFT motion-compensated diagnostics (no dense
  temporal GT); improvements are small (~2%) and dominated by real non-rigid motion.
- SERV-CT static safety check deferred (single-frame domain; VDPP causal degenerates there).
- D4D transfer evaluated on the tgm checkpoint only (the hypothesis model).
