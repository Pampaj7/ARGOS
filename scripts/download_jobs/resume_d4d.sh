#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="/dtu/p1/leopam/ARGOS/dataset/D4D/raw/source"

echo "Resuming interrupted downloads with wget -c"

wget -c https://opara.zih.tu-dresden.de/bitstreams/f6020d36-cc3b-4f5f-8614-14e0e5146e6a/download -O "$DEST_DIR/specimen_2.tar.gz"
wget -c https://opara.zih.tu-dresden.de/bitstreams/cede63c4-cc80-4e7c-a937-34d5a045b62b/download -O "$DEST_DIR/specimen_3.tar.gz"
wget -c https://opara.zih.tu-dresden.de/bitstreams/d5de8092-c385-4d1f-a782-88bc7979475d/download -O "$DEST_DIR/specimen_4.tar.gz"
wget -c https://opara.zih.tu-dresden.de/bitstreams/e93ffb45-83a4-4995-a64b-908f6ce7c1d0/download -O "$DEST_DIR/archive_listing.txt"

echo "Resume complete."
