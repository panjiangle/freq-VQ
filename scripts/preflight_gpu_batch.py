"""Measure one HQ backward pass without changing weights or starting training."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from basicsr.models import build_model
from basicsr.utils.options import parse_options
import torch

batch = int(sys.argv[1]) if len(sys.argv) > 1 else 8
sys.argv = ['preflight', '-opt', 'options/train_FeMaSR_HQ_wavelet2.yml']
opt, _ = parse_options(str(ROOT), is_train=True)
opt['root_path'] = str(ROOT)
model = build_model(opt)
for param in model.net_d.parameters():
    param.requires_grad_(False)
hr = torch.rand(batch, 3, 256, 256, device='cuda')
out, codebook_loss, semantic_loss, _ = model.net_g(hr)
pixel_loss = model.cri_pix(out, hr)
perceptual_loss, _ = model.cri_perceptual(out, hr)
gan_loss = model.cri_gan(model.net_d(out), True, is_disc=False)
total = pixel_loss + perceptual_loss + 0.1 * gan_loss + codebook_loss + 0.1 * semantic_loss
total.backward()
torch.cuda.synchronize()
print(f'HQ synthetic batch={batch} backward succeeded; no optimizer step ran.')
print(f'Peak allocated GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB')
print(f'Peak reserved GPU memory: {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB')
