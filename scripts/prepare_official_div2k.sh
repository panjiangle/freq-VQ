#!/usr/bin/env bash
set -euo pipefail

# Download the publisher-provided DIV2K HR and x4 bicubic LR archives.
# No degradation or resizing is performed here.
ROOT=${1:-/root/autodl-tmp/freq-vq/datasets/DF2K_official}
RAW="$ROOT/raw"
mkdir -p "$RAW" "$ROOT/train_HR" "$ROOT/train_LR_x4" "$ROOT/val_HR" "$ROOT/val_LR_x4"
BASE=https://data.vision.ee.ethz.ch/cvl/DIV2K

download_extract() {
  local archive=$1
  local dest=$2
  local url="$BASE/$archive.zip"
  if [ ! -f "$RAW/$archive.done" ]; then
    echo "Downloading $url"
    wget -q --show-progress -c --tries=10 --timeout=30 -O "$RAW/$archive.zip" "$url"
    unzip -tq "$RAW/$archive.zip" >/dev/null
    unzip -q -o "$RAW/$archive.zip" -d "$RAW"
    rm "$RAW/$archive.zip"
    touch "$RAW/$archive.done"
  fi
  local source_dir="$RAW/$archive"
  if [ "$archive" = DIV2K_train_LR_bicubic_X4 ]; then
    source_dir="$RAW/DIV2K_train_LR_bicubic/X4"
  elif [ "$archive" = DIV2K_valid_LR_bicubic_X4 ]; then
    source_dir="$RAW/DIV2K_valid_LR_bicubic/X4"
  fi
  if [ ! -d "$source_dir" ]; then
    echo "Expected archive directory absent: $source_dir" >&2
    exit 1
  fi
  find "$source_dir" -maxdepth 1 -type f -name '*.png' -print0 |
    while IFS= read -r -d '' path; do
      local name
      name=$(basename "$path")
      ln -sfn "$path" "$dest/DIV2K_$name"
    done
  echo "$archive: $(find "$dest" -maxdepth 1 -type l | wc -l) links"
}

download_extract DIV2K_train_HR "$ROOT/train_HR"
download_extract DIV2K_train_LR_bicubic_X4 "$ROOT/train_LR_x4"
download_extract DIV2K_valid_HR "$ROOT/val_HR"
download_extract DIV2K_valid_LR_bicubic_X4 "$ROOT/val_LR_x4"
echo "DIV2K official pairs are ready."
