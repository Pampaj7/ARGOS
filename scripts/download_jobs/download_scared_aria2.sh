#!/usr/bin/env bash
set -euo pipefail

INPUT="../../scripts/download_jobs/scared_remaining_aria2.txt"
LOG_PREFIX="$(date '+%F %T')"
echo "[$LOG_PREFIX] Starting SCARED aria2 segmented download"

attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "[$(date '+%F %T')] aria2 attempt ${attempt}"
  ../../external/frame_stereo_repos/Fast-FoundationStereo/.conda/bin/aria2c \
    --input-file="$INPUT" \
    --continue=true \
    --max-connection-per-server=16 \
    --split=16 \
    --min-split-size=1M \
    --max-concurrent-downloads=3 \
    --file-allocation=none \
    --auto-file-renaming=false \
    --allow-overwrite=false \
    --max-tries=0 \
    --retry-wait=10 \
    --summary-interval=30 \
    --console-log-level=notice || true

  if ! find ../../dataset/SCARED \
      -maxdepth 1 -name '*.aria2' -print -quit | grep -q .; then
    break
  fi

  echo "[$(date '+%F %T')] Some aria2 state files remain; refreshing signed URLs and resuming"
  sleep 20
done

echo "[$(date '+%F %T')] SCARED aria2 segmented download complete"
