#!/usr/bin/env bash
set -euo pipefail

ROOT=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
PYTHON=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
OUT="$ROOT/results/h4_augmented/seed_0"
FINAL="$ROOT/model_design/checkpoints/h4_augmented/best_validation.pt"
RUNNER="$ROOT/scripts/run_h4_augmented_fusion_probe.py"

if [[ -e "$FINAL" ]]; then
  echo "final artifact exists; refusing to overwrite: $FINAL" >&2
  exit 1
fi
mkdir -p "$OUT"
if [[ -s "$OUT/train.pid" ]] && kill -0 "$(<"$OUT/train.pid")" 2>/dev/null; then
  echo "already running: $(<"$OUT/train.pid")" >&2
  exit 0
fi

COMMON=(--mode train --profile h4_augmented --output "$OUT" --device cuda:0 --seed 0 --epochs 150 --patience 10 --batch-size 32 --workers 20 --preload-workers 20 --learning-rate 2e-4 --weight-decay 1e-4 --clip-length 4 --coverage-threshold .50 --tau-reset-native-px 5.0 --tau-fusion-native-px 1.0 --alpha-reg .2 --memory-state recurrent)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$ROOT" "$PYTHON" "$RUNNER" "${COMMON[@]}" --dry-run >"$OUT/preflight.json"
sha256sum "$RUNNER" "$ROOT/model_design/checkpoints/codd_style_h4_best_validation.pt" >"$OUT/source_snapshot.sha256"

cd "$ROOT"
nohup setsid env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$ROOT" "$PYTHON" "$RUNNER" "${COMMON[@]}" >"$OUT/train.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" >"$OUT/train.pid"
printf '{"state":"launched","pid":%s,"physical_gpu":0}\n' "$pid" >"$OUT/status.json"
echo "launched pid=$pid log=$OUT/train.log"
