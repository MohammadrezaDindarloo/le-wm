#!/bin/bash
# Download cube dataset from HuggingFace and extract to ~/.stable_worldmodel/ogbench/

set -e

CACHE_DIR="${STABLEWM_HOME:-$HOME/.stable_worldmodel}"
DATASET_DIR="$CACHE_DIR/ogbench"
TARGET="$DATASET_DIR/cube_single_expert.h5"

if [ -f "$TARGET" ]; then
    echo "Dataset already exists at $TARGET, skipping download."
    exit 0
fi

mkdir -p "$DATASET_DIR"

echo "Downloading cube_single_expert.tar.zst from HuggingFace (~46 GB)..."
# Stream download directly through zstd decompress + tar extract
HF_URL="https://huggingface.co/datasets/quentinll/lewm-cube/resolve/main/cube_single_expert.tar.zst"

# Download to a temporary file first, then extract
TMP_ARCHIVE="$CACHE_DIR/cube_single_expert.tar.zst"

if [ ! -f "$TMP_ARCHIVE" ]; then
    wget --progress=dot:giga -O "$TMP_ARCHIVE" "$HF_URL"
else
    echo "Archive already downloaded at $TMP_ARCHIVE, proceeding to extraction."
fi

echo "Extracting archive..."
cd "$DATASET_DIR"
tar --zstd -xvf "$TMP_ARCHIVE"

echo "Extraction done. Checking for .h5 file..."
ls -lh "$DATASET_DIR/"*.h5

echo "Done! Dataset available at $DATASET_DIR"
