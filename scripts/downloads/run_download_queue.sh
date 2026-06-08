#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${ARGOS_REPO_DIR:-/home/lpampaloni/ARGOS}"
STEREO_ROOT="${ARGOS_STEREO_ROOT:-${REPO_DIR}/stereo}"
PYTHON="${ARGOS_DOWNLOAD_PYTHON:-${STEREO_ROOT}/download_jobs/.venv/bin/python}"
LOG_DIR="${ARGOS_DOWNLOAD_LOG_DIR:-${STEREO_ROOT}/download_jobs}"

mkdir -p "${LOG_DIR}"

export ARGOS_FORCE_DOWNLOAD="${ARGOS_FORCE_DOWNLOAD:-1}"
export ARGOS_SCARED_DIR="${ARGOS_SCARED_DIR:-${STEREO_ROOT}/Fast-FoundationStereo/data/surgical_stereo/scared}"
export ARGOS_S2M2_WEIGHTS_DIR="${ARGOS_S2M2_WEIGHTS_DIR:-${STEREO_ROOT}/s2m2/weights/pretrain_weights}"
export ARGOS_ENDOSLAM_DIR="${ARGOS_ENDOSLAM_DIR:-${STEREO_ROOT}/datasets/EndoSLAM}"
export ARGOS_MONSTERPP_CHECKPOINT_DIR="${ARGOS_MONSTERPP_CHECKPOINT_DIR:-${STEREO_ROOT}/MonSter-plusplus/MonSter++/checkpoints}"

timestamp() {
  date '+%F %T'
}

run_step() {
  local name="$1"
  local script="$2"
  local log_file="${LOG_DIR}/${name}.log"

  echo "[$(timestamp)] starting ${name}" | tee -a "${LOG_DIR}/download_queue.log"
  "${PYTHON}" "${REPO_DIR}/${script}" 2>&1 | tee -a "${log_file}"
  echo "[$(timestamp)] finished ${name}" | tee -a "${LOG_DIR}/download_queue.log"
}

run_step scared_full scripts/downloads/download_scared_full.py
run_step training_extras scripts/downloads/download_training_extras.py
run_step monsterpp_large scripts/downloads/download_monsterpp_large.py

echo "[$(timestamp)] download queue complete" | tee -a "${LOG_DIR}/download_queue.log"
