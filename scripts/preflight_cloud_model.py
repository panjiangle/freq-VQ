"""Initialize the configured training model without running any training step."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from basicsr.models import build_model
from basicsr.utils.options import parse_options
import torch

config = sys.argv[1] if len(sys.argv) > 1 else 'options/train_FeMaSR_HQ_wavelet2.yml'
sys.argv = ['preflight', '-opt', config]
opt, _ = parse_options(str(ROOT), is_train=True)
opt['root_path'] = str(ROOT)
model = build_model(opt)
print(f'Initialized {opt["name"]} on {model.device}; no training step was run.', flush=True)
if torch.cuda.is_available():
    print(f'GPU memory allocated: {torch.cuda.memory_allocated() / 2**30:.2f} GiB', flush=True)
