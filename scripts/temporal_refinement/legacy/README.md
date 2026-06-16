# Legacy Temporal-Refinement Trainers

These scripts are kept only to reproduce early ARGOS temporal-refinement debug runs.

Use the unified trainer for new experiments:

```bash
scripts/temporal_refinement/train_refiner.py
```

Legacy files:

- `train_debug_unet_refiner.py`: first single-frame Tiny U-Net debug trainer.
- `train_debug_unet_refiner_pairs.py`: first consecutive-pair teacher-delta debug trainer.
- `train_multiteacher_s2m2s_refiner.py`: old compressed `.npz` multi-teacher trainer.
- `train_multiteacher_s2m2s_refiner_fastcache.py`: old S2M2-S-only fast-cache trainer.

Do not add new training features here. Put reusable training code under
`scripts/temporal_refinement/lib/` and expose it through `train_refiner.py`.
