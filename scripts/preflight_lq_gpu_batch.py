"""Measure one LQ backward pass with an untrained teacher, without updates."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import torch
from basicsr.archs.femasr_arch import FeMaSRNet
from basicsr.losses.losses import LPIPSLoss

batch = int(sys.argv[1]) if len(sys.argv) > 1 else 8
common = dict(codebook_params=[[32, 512, 512]], gt_resolution=256,
              wavelet_vq=True, detail_codebook_size=512)
teacher = FeMaSRNet(**common, LQ_stage=False).cuda().eval()
student = FeMaSRNet(**common, LQ_stage=True, scale_factor=4).cuda().train()
for name, module in student.named_modules():
    if any(word in name for word in ('quantize', 'decoder', 'after_quant_group', 'out_conv')):
        for param in module.parameters():
            param.requires_grad_(False)
perceptual = LPIPSLoss().cuda()
hr = torch.rand(batch, 3, 256, 256, device='cuda')
lr = torch.rand(batch, 3, 64, 64, device='cuda')
with torch.no_grad():
    _, _, _, indices = teacher(hr)
out, codebook_loss, _, _ = student(lr, indices)
loss, _ = perceptual(out, hr)
(loss + codebook_loss + (out - hr).abs().mean()).backward()
torch.cuda.synchronize()
print(f'LQ synthetic batch={batch} backward succeeded; no optimizer step ran.')
print(f'Peak allocated GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB')
print(f'Peak reserved GPU memory: {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB')
