# Temporal-module comparison

`run_comparison.py` owns the datasets, fixed-H4 protocol, metrics and output
files.  The temporal module is the only replaceable part:

```python
def factory(*, device: str): ...
adapter.describe() -> dict             # module/checkpoint/policy/code provenance
adapter.start(frame) -> result         # exact raw identity at a true boundary
adapter.step(frame) -> result          # current + causal past only
```

`frame` never contains GT, coverage, a future frame or a backbone identifier.
Both calls return a mapping with `disparity`, `support`, `reset`, `state_age`
and `diagnostics`; `step` may additionally return `aligned_memory`.  The
driver validates the fixed H=4 phase: four fusion updates, then re-anchor from
the preceding raw prediction.  It owns all support masks and metrics, so a new
temporal module is compared on the same protocol.

`diagnostics.update_magnitude` is mandatory on every `start()` and `step()`
result: it must be a finite scalar in the model input disparity grid.  The
frozen driver consumes it for D4D diagnostics; the wrapper only gives contract
context if that frozen evaluation fails.

The saved forward is not just the head checkpoint: it depends on the frozen
checkpoint/policy, fusion head, bounded policy, frozen ResNet, and the
validated BiDA/SEA-RAFT alignment path.

```bash
cd /dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
CUDA_VISIBLE_DEVICES=1 /dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python \
  model_design/comparison/run_comparison.py --dataset scared-d2

CUDA_VISIBLE_DEVICES=1 /dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python \
  model_design/comparison/run_comparison.py --dataset scared-d7

CUDA_VISIBLE_DEVICES=1 /dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python \
  model_design/comparison/run_comparison.py --dataset d4d

/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python \
  model_design/comparison/run_comparison.py --dataset servct
```

`CUDA_VISIBLE_DEVICES=1` maps physical GPU1 to the required logical `cuda:0`.
Default results are written below
`/dtu/p1/leopam/ARGOS/ARGOS_hand/results/temporal_module_comparison/<dataset>/`.
An existing output path is refused.

The definitive evaluator uses the same causal driver but writes only official
JSON reports and compact summaries (never dense predictions):

```bash
cd /dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
CUDA_VISIBLE_DEVICES=1 /dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python \
  model_design/comparison/definitive_evaluation.py --dataset scared-d2
```

Its default root is
`/dtu/p1/leopam/ARGOS/ARGOS_hand/results/definitive_temporal_evaluation/`.

## Definitive Evaluation

`run_definitive_evaluation.py` is the reusable command for any compatible
temporal module. It preserves `definitive_evaluation.py` as the frozen engine,
then writes complete long-form CSVs and one wide row per dataset/split ×
temporal model × backbone. It never pools D2/D7, GT and no-reference metrics,
and never writes dense predictions.

```bash
CUDA_VISIBLE_DEVICES=1 /dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python \
  model_design/comparison/run_definitive_evaluation.py \
  --datasets scared-d2 scared-d7 d4d servct \
  --output ../results/definitive_temporal_evaluation_csv/canonical_h4
```

Physical GPU1 is logical `cuda:0`.  `--module import.path:factory` replaces
only the temporal module.  Use `--scared-backbones` for SCARED-C and
`--external-backbones` for D4D/SERV-CT; defaults are the compatible caches.
Existing outputs are refused and published only after completion.

For a CPU-only audit of an already complete run, without re-evaluation:

```bash
/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python \
  model_design/comparison/run_definitive_evaluation.py \
  --compile-from ../results/definitive_temporal_evaluation \
  --output ../results/definitive_temporal_evaluation_csv/canonical_h4_compiled
```

The output contains `definitive_table.csv` plus `tables/<dataset>.csv`: the
wide paper-ready tables. Metadata marks training-seen/unseen backbones,
in-domain/OOD datasets, and applicability. Every aggregate scalar from
`unified_metrics.py` has a collision-free
`{section}__{metric_path}__{method}__{statistic}` column. SCARED rows use
macro sequence means, summed counts, and support-weighted micro metrics.
Long-form source CSVs remain available: `scared_aggregate_metrics.csv`,
`scared_per_sequence_metrics.csv`, `scared_per_frame_metrics.csv`,
`d4d_no_reference_metrics.csv`, `d4d_no_reference_summary.csv`,
`applicability.csv`, and a hash-checked `run_manifest.json`. Every SCARED
scalar is long-form (`metric_path`, `statistic`, `value`), including support
counts, sequence summaries and boolean flags. D4D summaries are explicitly
equal-frame no-reference diagnostics; they are never GT geometry. SERV-CT
temporal H4 remains `NOT_APPLICABLE`.

SCARED-D2 uses the paper strict all-anchor support after warm-up 8;
SCARED-D7 is intentionally the separate H4-only support protocol.  D4D uses
complete curated four-frame windows, past-to-present, and reports only
no-reference prediction-space diagnostics; unavailable source windows are
recorded and excluded.  SERV-CT is static-only and writes `NOT_APPLICABLE`.

### DRENDS pilot

DRENDS is an OOD `tof_reference_nonindependent` pilot, initially for available on-the-fly
`RAFT-Stereo` only. Its tables are labeled `APPLICABLE_WITH_CAVEAT`. It validates the curated chronological manifest, rejects
timing-offset frames before the initial reset, derives disparity from the
rectified focal-baseline and ToF depth, scales disparity at 1280×720 → 180×144,
and writes no prediction cache. The ToF reference is temporally smoothed and
is not independent stereo disparity GT.

```bash
CUDA_VISIBLE_DEVICES=1 /dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python \
  model_design/comparison/run_definitive_evaluation.py \
  --datasets drends --drends-recordings Vid14_Pancreas_High --max-frames 64 \
  --output ../results/definitive_temporal_evaluation_csv/canonical_h4_drends_pilot
```

## Script roles

- `canonical_h4_provenance.py`: required immutable checkpoint/policy provenance.
- `frozen_transfer_eval.py`: golden SCARED oracle, not the forward comparison requirement.
- `run_codd_style_bounded_memory_validation.py`: old SCARED evaluator, not required by the forward runner.
- `run_codd_style_fusion_probe.py`: canonical retraining only.
- `run_h4_augmented_fusion_probe.py`: augmented H4 retraining only.
- `run_codd_style_fusion_mechanism_audit.py`: diagnostics only.
- `scripts/argos_v2/` loaders: required by the current SCARED comparison base.
