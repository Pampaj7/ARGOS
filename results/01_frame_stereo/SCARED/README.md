# SCARED Evaluation

Main table:

- `scared_evaluation.md`
- `scared_evaluation.csv`
- `evidence.csv`

Current main protocol: `scared_warped_gt_108`.

This uses `dataset/SCARED/curated/warped_gt_108/metadata.csv`, with 108
pre-rectified warped SCARED samples and GT depth/disparity/masks. Per-method
outputs are in `warped_gt_108/`.

The older 5-frame keyframe benchmark is preserved in `unified_keyframes/`, but
it is no longer the main SCARED table.

Regenerate the 108-frame table with:

```bash
python3 scripts/scared/run_all_scared_baselines.py \
  --metadata-csv dataset/SCARED/curated/warped_gt_108/metadata.csv \
  --protocol-name scared_warped_gt_108 \
  --dataset-label dataset/SCARED/curated/warped_gt_108/metadata.csv \
  --eval-subdir warped_gt_108
```

External-model adapters are run with `scripts/scared/eval_scared_external_native.py`
using the same metadata CSV, then aggregated with `--aggregate-only`.
