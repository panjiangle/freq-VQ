#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
DATA=$(cd "$REPO/../datasets/DF2K_official" && pwd)
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/freqvq/bin/python}
cd "$REPO"
for split in train_HR train_LR_x4 val_HR val_LR_x4; do
  test -d "$DATA/$split" || { echo "Missing data: $DATA/$split" >&2; exit 1; }
done
test "$(find "$DATA/train_HR" -maxdepth 1 -type l | wc -l)" -eq 3450
test "$(find "$DATA/train_LR_x4" -maxdepth 1 -type l | wc -l)" -eq 3450
test "$(find "$DATA/val_HR" -maxdepth 1 -type l | wc -l)" -eq 100
test "$(find "$DATA/val_LR_x4" -maxdepth 1 -type l | wc -l)" -eq 100
"$PYTHON_BIN" scripts/validate_cloud_data.py
PYTHONPATH="$REPO" "$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
mkdir -p "$REPO/cloud_logs"
PYTHONPATH="$REPO" "$PYTHON_BIN" -u -m basicsr.train \
  -opt options/train_FeMaSR_HQ_wavelet2.yml --auto_resume 2>&1 | tee -a "$REPO/cloud_logs/hq_train.log"
