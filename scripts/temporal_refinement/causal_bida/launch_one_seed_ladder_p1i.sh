#!/usr/bin/env bash
set -euo pipefail
cd /dtu/p1/leopam/ARGOS

PY=.miniconda/envs/argos/bin/python
OUT=results/03_temporal_refinement/argos_v2/one_seed_ladder
LOGS="$OUT/logs"
mkdir -p "$LOGS" "$OUT/resolved_configs" "$OUT/checkpoints" "$OUT/diagnostics"

echo "ARGOS_V2_LADDER_START $(date)"
hostname
nvidia-smi

run_cfg() {
  local cfg="$1"
  local gpu="$2"
  echo "START $cfg gpu=$gpu $(date)"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/temporal_refinement/causal_bida/train_argos_v2.py \
    --config "$cfg" \
    --output-root "$OUT" \
    --device cuda \
    --amp \
    --resume \
    > "$LOGS/${cfg}.out" 2> "$LOGS/${cfg}.err"
  echo "DONE $cfg gpu=$gpu $(date)"
}

run_eval_cfg() {
  local cfg="$1"
  local gpu="$2"
  local ckpt="$3"
  local ckpt_args=()
  if [[ -n "$ckpt" ]]; then
    ckpt_args=(--checkpoint "$ckpt")
  fi
  echo "START_EVAL $cfg gpu=$gpu $(date)"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/temporal_refinement/causal_bida/train_argos_v2.py \
    --config "$cfg" \
    --output-root "$OUT" \
    --device cuda \
    "${ckpt_args[@]}" \
    > "$LOGS/${cfg}.out" 2> "$LOGS/${cfg}.err"
  echo "DONE_EVAL $cfg gpu=$gpu $(date)"
}

run_eval_cfg raw_s2m2 0 ""

run_cfg current_only 0 &
p1=$!
run_cfg aligned_local_faithful 1 &
p2=$!
wait "$p1" "$p2"

run_cfg faithful_causal_bida 0 &
p1=$!
run_cfg safe_causal_bida 1 &
p2=$!
wait "$p1" "$p2"

FAITHFUL_CKPT="$OUT/faithful_causal_bida_seed0/checkpoints/best.pt"
run_eval_cfg faithful_causal_bida_state_reset 0 "$FAITHFUL_CKPT" &
p1=$!
run_eval_cfg faithful_causal_bida_shuffled_history 1 "$FAITHFUL_CKPT" &
p2=$!
wait "$p1" "$p2"

"$PY" scripts/temporal_refinement/causal_bida/summarize_one_seed_ladder.py --root "$OUT" \
  > "$LOGS/summary.out" 2> "$LOGS/summary.err"

echo "ARGOS_V2_LADDER_DONE $(date)"
