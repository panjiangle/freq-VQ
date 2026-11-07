"""Export an HQ reconstruction from the trained two-codebook tokenizer.

This uses the same center 512x512 HR crop as the paired DIV2K validation set.
The HQ network receives HR, so this image is a reconstruction, not SR from LR.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Use this checkout's modified architecture, even if another BasicSR is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from basicsr.archs.femasr_arch import FeMaSRNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hr', type=Path, required=True)
    parser.add_argument('--weight', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--crop-size', type=int, default=512)
    args = parser.parse_args()

    image = cv2.imread(str(args.hr), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.hr)
    height, width = image.shape[:2]
    crop = args.crop_size
    if min(height, width) < crop:
        raise ValueError(f'Image {width}x{height} is smaller than crop {crop}')
    top = (height - crop) // 2
    left = (width - crop) // 2
    image = image[top:top + crop, left:left + crop]

    model = FeMaSRNet(
        codebook_params=[[32, 512, 512]],
        detail_codebook_size=512,
        wavelet_vq=True,
        gt_resolution=256,
        LQ_stage=False,
        use_quantize=True,
        # Inference does not need the VGG loss network saved in the checkpoint.
        use_semantic_loss=False,
    )
    state = torch.load(args.weight, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(state['params'], strict=False)
    if missing or any(not key.startswith(('vgg_feat_extractor.', 'conv_semantic.'))
                      for key in unexpected):
        raise RuntimeError(f'Incompatible checkpoint: missing={missing}, unexpected={unexpected}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    tensor = torch.from_numpy(np.ascontiguousarray(image[:, :, ::-1]))
    tensor = tensor.permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    with torch.inference_mode():
        output = model.test(tensor).clamp_(0, 1)
    output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output = np.round(output[:, :, ::-1] * 255).clip(0, 255).astype(np.uint8)

    args.out.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.out / 'HR_center512.png'), image):
        raise OSError('Failed to write HR reference')
    if not cv2.imwrite(str(args.out / 'reconstruction_center512.png'), output):
        raise OSError('Failed to write reconstruction')
    if not cv2.imwrite(str(args.out / 'HR_vs_reconstruction.png'),
                       np.concatenate([image, output], axis=1)):
        raise OSError('Failed to write comparison')
    print(f'Wrote HQ reconstruction to {args.out}')


if __name__ == '__main__':
    main()
