# Temporal Refinement Playground

This package is a modular research playground for surgical temporal stereo refinement.

The goal is to test architectural combinations from YAML configs without rewriting the training or validation path.

## Main Concepts

- `config.py`: typed dataclasses for complete experiments.
- `registry.py`: explicit registries for interchangeable modules.
- `modules.py`: stereo sources, motion estimators, warpers, memories, uncertainty, teachers, semantic priors, and fusion heads.
- `model.py`: assembles a `PlaygroundModel` entirely from config.
- `losses.py`: shared temporal/geometric/distillation loss scaffold.
- `runner.py`: synthetic smoke tests, reference images, metrics, and runtime/memory reports.

All fusion heads return the same `FusionOutput` fields:

- `fused_disparity`
- `source_weights`
- `alpha_map`
- `reset_map`
- `residual_map`
- `uncertainty_map`
- `hidden_state`
- `diagnostics`

## Smoke Test

```bash
.miniconda/envs/argos/bin/python scripts/temporal_refinement/run_playground.py \
  --device cpu \
  --out-dir results/03_temporal_refinement/playground/tmp_smoke
```

If CUDA is visible in the current environment, use `--device cuda`.

The smoke runner loads every YAML in:

```text
configs/temporal_refinement/playground/
```

and writes:

```text
results/03_temporal_refinement/playground/tmp_smoke/
  metrics.csv
  memory_runtime_report.csv
  report.md
  <experiment>/
    config.yaml
    metrics.csv
    report.md
    reference_images/synthetic_reference.png
```

`results/03_temporal_refinement/playground/` is ignored by Git and should be treated as disposable smoke-test output. Delete it after validating a change unless you explicitly need to preserve a report.

## Adding A Module

1. Add a class in `modules.py`.
2. Register it with the relevant registry, for example:

```python
@FUSION_HEADS.register("my_new_fusion")
class MyNewFusion(BaseFusionHead):
    ...
```

3. Reference it from YAML:

```yaml
fusion:
  name: my_new_fusion
  my_parameter: 123
```

4. Run:

```bash
.miniconda/envs/argos/bin/python -m pytest -q tests/test_temporal_refinement_playground.py
```
