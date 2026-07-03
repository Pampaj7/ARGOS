#!/bin/bash
# Persistent chain: wait for v4_tiny training to finish, then launch EGBM training.
# Sequential on purpose -- running both on the shared p1i GPU at once caused an OOM crash.
set -uo pipefail
cd /dtu/p1/leopam/ARGOS

V4_JOBID=28865878
V4_DIR=results/03_temporal_refinement/training/modern_refiner_v4_tiny
CHAIN_LOG=logs/chain_v4tiny_then_egbm.log

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$CHAIN_LOG"; }

log "chain started, waiting on v4_tiny job $V4_JOBID"
while true; do
  STAT=$(bjobs -noheader -o stat "$V4_JOBID" 2>/dev/null)
  case "$STAT" in
    RUN|PEND|PSUSP|USUSP|SSUSP|WAIT) sleep 60 ;;
    *) break ;;
  esac
done
log "v4_tiny job $V4_JOBID reached terminal state: $STAT"

if [ -s "$V4_DIR/aggregate_summary.json" ]; then
  log "v4_tiny finished with a summary -- proceeding to EGBM training"
else
  log "WARNING: v4_tiny has no aggregate_summary.json (crashed or still writing) -- proceeding to EGBM anyway after a short grace wait"
  sleep 30
fi

log "launching EGBM staged training"
export ESUB_BYPASS=1 ESUB_QUIET=1
bsub -q p1i -app h100app -gpu "num=2:mode=shared" -n 4 -R "span[hosts=1]" -J egbm_full -I bash -c '
set -euo pipefail
cd /dtu/p1/leopam/ARGOS
export PYTHONPATH="$(pwd)"
export OMP_NUM_THREADS=4
source .miniconda/etc/profile.d/conda.sh
conda activate argos
echo "NODE=$(hostname)"
python scripts/temporal_refinement/train_experimental_refiner_vx.py \
  --num-workers 24 --overwrite
echo TRAIN_EXIT=$?
' >> "$CHAIN_LOG" 2>&1
log "EGBM training bsub -I returned exit=$?"
log "chain complete"
