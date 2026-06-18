#!/usr/bin/env bash
set -euo pipefail

INPUT="/dtu/p1/leopam/ARGOS/scripts/download_jobs/d4d_aria2_input.txt"
DEST_DIR="$(dirname "$0")/../../dataset/D4D/raw/source"

echo "Starting D4D aria2 segmented download"

# We use the system aria2c if available, else fallback to the conda one
ARIA2C="aria2c"
if ! command -v aria2c &> /dev/null; then
    ARIA2C="/dtu/p1/leopam/ARGOS/external/frame_stereo_repos/Fast-FoundationStereo/.conda/bin/aria2c"
fi

$ARIA2C \
  --dir="$DEST_DIR" \
  --input-file="$INPUT" \
  --continue=true \
  --max-connection-per-server=16 \
  --split=16 \
  --min-split-size=1M \
  --max-concurrent-downloads=3 \
  --file-allocation=none \
  --auto-file-renaming=false \
  --allow-overwrite=false \
  --check-certificate=false \
  --summary-interval=60 \
  --console-log-level=notice

echo "D4D download complete"
