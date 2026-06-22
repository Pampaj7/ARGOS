#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="/dtu/p1/leopam/ARGOS/dataset/D4D/raw/source"
DEST_DIR="/dtu/p1/leopam/ARGOS/dataset/D4D/raw/extracted"

mkdir -p "$DEST_DIR"

for TAR in "$SOURCE_DIR"/specimen_*.tar.gz; do
    echo "Extracting $(basename "$TAR")..."
    tar -xf "$TAR" -C "$DEST_DIR"
done

echo "Extraction complete."
