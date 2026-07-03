# Distillation Target GT Consistency Audit

This audit used existing `.npz` targets and rectified GT only. It did not run S2M2, SAV, RAFT, DINO, or training.

## Finding

Status: `PASS`.

- Raw MAE vs valid-masked downsampled GT: `12.298818`
- Oracle-all MAE vs valid-masked downsampled GT: `5.731828`
- Oracle violation rate: `0.00%`
- Delta reconstruction MAE: `0.000000`

low-res oracle target is GT-consistent

## Recommendation

none for target consistency
