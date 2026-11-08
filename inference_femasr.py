import argparse
import cv2
import glob
import os
from tqdm import tqdm
import torch

from basicsr.utils import img2tensor, tensor2img, imwrite 
from basicsr.archs.femasr_arch import FeMaSRNet 
from basicsr.utils.download_util import load_file_from_url 

pretrain_model_url = {
    'x4': 'https://github.com/chaofengc/FeMaSR/releases/download/v0.1-pretrain_models/FeMaSR_SRX4_model_g.pth',
    'x2': 'https://github.com/chaofengc/FeMaSR/releases/download/v0.1-pretrain_models/FeMaSR_SRX2_model_g.pth',
}


def main():
    """Inference demo for FeMaSR 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder')
    parser.add_argument('-w', '--weight', type=str, default=None, help='path for model weights')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument('-s', '--out_scale', type=int, default=4, help='The final upsampling scale of the image')
    parser.add_argument('--suffix', type=str, default='', help='Suffix of the restored image')
    parser.add_argument('--max_size', type=int, default=600, help='Max image size for whole image inference, otherwise use tiled_test')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    if args.weight is None:
        weight_path = load_file_from_url(pretrain_model_url[f'x{args.out_scale}'])
    else:
        weight_path = args.weight
    
    # WAVELET-VQ CHANGE: infer the two-codebook architecture from an LQ checkpoint.
    # An HQ checkpoint has a different encoder and is not valid for SR inference.
    checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
    params = checkpoint['params']
    params = {(key[7:] if key.startswith('module.') else key): value
              for key, value in params.items()}
    ll_key = 'quantize_group.0.embedding.weight'
    detail_key = 'quantize_group.1.embedding.weight'
    wavelet_vq = detail_key in params and 'before_quant_group.1.weight' not in params
    if wavelet_vq:
        ll_codes, code_dim = params[ll_key].shape
        detail_codes, detail_dim = params[detail_key].shape
        if code_dim != 512 or detail_dim != code_dim:
            raise ValueError('Wavelet VQ checkpoint requires two 512-dimensional codebooks')
        sr_model = FeMaSRNet(
            codebook_params=[[32, ll_codes, code_dim]],
            detail_codebook_size=detail_codes,
            wavelet_vq=True,
            LQ_stage=True,
            scale_factor=args.out_scale,
        ).to(device)
    else:
        sr_model = FeMaSRNet(codebook_params=[[32, 1024, 512]],
                             LQ_stage=True, scale_factor=args.out_scale).to(device)
    try:
        sr_model.load_state_dict(params, strict=wavelet_vq)
    except RuntimeError as exc:
        if wavelet_vq:
            raise RuntimeError('Cannot load wavelet checkpoint for SR. Use an LQ-stage '
                               'net_g checkpoint; the HQ tokenizer has a different encoder.') from exc
        raise
    sr_model.eval()
    
    os.makedirs(args.output, exist_ok=True)
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    pbar = tqdm(total=len(paths), unit='image')
    for idx, path in enumerate(paths):
        img_name = os.path.basename(path)
        pbar.set_description(f'Test {img_name}')

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img_tensor = img2tensor(img).to(device) / 255.
        img_tensor = img_tensor.unsqueeze(0)

        max_size = args.max_size ** 2 
        h, w = img_tensor.shape[2:]
        if h * w < max_size: 
            output = sr_model.test(img_tensor)
        else:
            output = sr_model.test_tile(img_tensor)
        output_img = tensor2img(output)

        save_path = os.path.join(args.output, f'{img_name}')
        imwrite(output_img, save_path)
        pbar.update(1)
    pbar.close()


if __name__ == '__main__':
    main()
