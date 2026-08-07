#!/usr/bin/env bash
# Fair continuous-streaming control: every pair, state reset exactly every 4.
set -euo pipefail
ROOT=/dtu/p1/leopam/ARGOS/ARGOS-V2
PYTHON=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
OUT=$ROOT/results/codd_style_fusion_mechanism_audit
cd "$ROOT"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$ROOT" "$PYTHON" scripts/run_codd_style_fusion_mechanism_audit.py \
  --mode audit --evaluation-mode reset_every4_all_pairs --seed 0 --device cuda:0 --workers 20 --preload-workers 20 --output-root "$OUT" \
  >"$OUT/canonical_reset_every4.log" 2>&1
PYTHONPATH="$ROOT" "$PYTHON" scripts/run_codd_style_fusion_mechanism_audit.py --mode combine --output-root "$OUT"
