#!/usr/bin/env bash
set -euo pipefail
ROOT=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4
PYTHON=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
OUT="$ROOT/results/h4_scared_drends_augmented/seed_0"
FINAL="$ROOT/model_design/checkpoints/h4_scared_drends_augmented/best_validation.pt"
EVAL=/dtu/p1/leopam/ARGOS/ARGOS_hand/results/definitive_evaluation/h4_scared_drends_augmented
RUNNER="$ROOT/scripts/run_h4_scared_drends_augmented.py"
[[ ! -e "$FINAL" && ! -e "$EVAL" ]] || { echo "final/evaluation collision; refusing overwrite" >&2; exit 1; }
mkdir -p "$OUT"
if [[ -s "$OUT/train.pid" ]] && kill -0 "$(<"$OUT/train.pid")" 2>/dev/null; then echo "already running $(<"$OUT/train.pid")"; exit 0; fi
COMMON=(--mode train --output "$OUT" --cache-root "$ROOT/cache_drends_backbones" --device cuda:0 --seed 0 --epochs 150 --patience 10 --batch-size 32 --workers 20 --preload-workers 20 --learning-rate 2e-4 --weight-decay 1e-4 --clip-length 4 --coverage-threshold .50 --tau-reset-native-px 5.0 --tau-fusion-native-px 1.0 --alpha-reg .2 --memory-state recurrent)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$ROOT" "$PYTHON" "$RUNNER" "${COMMON[@]}" --mode dry-run >"$OUT/preflight.json"
sha256sum "$RUNNER" "$ROOT/model_design/comparison/h4_scared_drends_augmented.py" "$ROOT/model_design/checkpoints/codd_style_h4_best_validation.pt" >"$OUT/source_snapshot.sha256"
cd "$ROOT"
nohup setsid env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$ROOT" "$PYTHON" -c 'import subprocess,sys; root=sys.argv[1]; common=sys.argv[2:]; subprocess.run([sys.executable, root+"/scripts/run_h4_scared_drends_augmented.py", *common], check=True); subprocess.run([sys.executable, root+"/model_design/comparison/run_definitive_evaluation.py", "--datasets", "scared-d2", "scared-d7", "d4d", "servct", "drends", "--module", "model_design.comparison.h4_scared_drends_augmented:factory", "--output", "/dtu/p1/leopam/ARGOS/ARGOS_hand/results/definitive_evaluation/h4_scared_drends_augmented", "--device", "cuda:0"], check=True)' "$ROOT" "${COMMON[@]}" >"$OUT/train.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" >"$OUT/train.pid"
printf '{"state":"launched","pid":%s,"physical_gpu":0,"detached":true}\n' "$pid" >"$OUT/status.json"
echo "launched pid=$pid gpu=0 log=$OUT/train.log"
