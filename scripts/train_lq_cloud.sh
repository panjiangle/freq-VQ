#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/freqvq/bin/python}
cd "$REPO"
test -f experiments/008_FeMaSR_HQ_wavelet2/models/net_g_latest.pth || {
  echo 'HQ checkpoint is missing; finish HQ training first.' >&2
  exit 1
}
test -f experiments/pretrained_models/FeMaSR_HRP_model_d.pth
"$PYTHON_BIN" scripts/validate_cloud_data.py
PYTHONPATH="$REPO" "$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
mkdir -p "$REPO/cloud_logs"
PYTHONPATH="$REPO" "$PYTHON_BIN" -u -m basicsr.train \
  -opt options/train_FeMaSR_LQ_wavelet2.yml --auto_resume 2>&1 | tee -a "$REPO/cloud_logs/lq_train.log"
