#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
INPUT="$SCRIPT_DIR/d4d_aria2_input.txt"
DEST_DIR="$ROOT_DIR/dataset/D4D/raw/source"

echo "Starting D4D aria2 segmented download"

# We use the system aria2c if available, else fallback to the conda one
ARIA2C="aria2c"
if ! command -v aria2c &> /dev/null; then
    ARIA2C="$ROOT_DIR/external/frame_stereo_repos/Fast-FoundationStereo/.conda/bin/aria2c"
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
