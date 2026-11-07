"""Focused checks for the optional two-level wavelet VQ path."""

import os
import sys
import tempfile
from types import SimpleNamespace

# 本机可能装有另一份 editable basicsr；始终测试当前仓库的实现。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import torch

from basicsr.archs.femasr_arch import FeMaSRNet, haar_dwt2d, haar_idwt2d
from basicsr.models.femasr_model import FeMaSRModel


def test_haar_round_trip():
    x = torch.randn(2, 5, 32, 32, requires_grad=True)
    ll1, d1 = haar_dwt2d(x)
    ll2, d2 = haar_dwt2d(ll1)
    restored = haar_idwt2d(haar_idwt2d(ll2, d2), d1)
    assert restored.shape == x.shape
    assert torch.allclose(restored, x, atol=1e-6)
    restored.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_hq_lq_teacher_and_validation_images():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params = dict(codebook_params=[[32, 512, 512]], gt_resolution=256,
                  wavelet_vq=True, detail_codebook_size=512,
                  use_semantic_loss=False)
    hq = FeMaSRNet(**params, LQ_stage=False).to(device)
    hr = torch.randn(1, 3, 256, 256, device=device)
    reconstructed, hq_loss, _, teacher_indices = hq(hr)
    assert reconstructed.shape == hr.shape
    assert len(teacher_indices) == 7
    assert all(index.shape[-2:] == (8, 8) for index in teacher_indices[:4])
    assert all(index.shape[-2:] == (16, 16) for index in teacher_indices[4:])
    from_indices = hq.decode_indices(teacher_indices)
    assert from_indices.shape == reconstructed.shape
    assert torch.isfinite(from_indices).all()
    # 对未训练的随机解码器，归一化会放大查表的浮点舍入差；在 IDWT 输出处比较。
    with torch.no_grad():
        latent = torch.randn(1, 512, 32, 32, device=device)
        reconstructed_latent, _, latent_indices = hq._quantize_wavelet(latent)
        looked_up_ll2 = hq.quantize_group[0].get_codebook_entry(latent_indices[0])
        looked_up_details = [hq.quantize_group[1].get_codebook_entry(index)
                             for index in latent_indices[1:]]
        looked_up_ll1 = haar_idwt2d(looked_up_ll2, looked_up_details[:3])
        looked_up_latent = haar_idwt2d(looked_up_ll1, looked_up_details[3:])
        assert torch.allclose(reconstructed_latent, looked_up_latent, atol=1e-6)
    (reconstructed.mean() + hq_loss).backward()
    assert hq.quantize_group[0].embedding.weight.grad is not None
    assert hq.quantize_group[1].embedding.weight.grad is not None
    teacher_indices = [index.detach() for index in teacher_indices]
    del hq, reconstructed, from_indices, hq_loss
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    lq = FeMaSRNet(**params, LQ_stage=True, scale_factor=4).to(device)
    lr = torch.randn(1, 3, 64, 64, device=device)
    sr, lq_loss, _, predicted_indices = lq(lr, teacher_indices)
    assert sr.shape == (1, 3, 256, 256)
    assert len(predicted_indices) == 7
    (sr.mean() + lq_loss).backward()
    assert lq.before_quant_group[0].weight.grad is not None
    assert lq.multiscale_encoder.in_conv.weight.grad is not None

    with torch.no_grad():
        preview_sr, visuals = lq.test(torch.randn(1, 3, 48, 48, device=device),
                                      collect_wavelet=True)
    assert preview_sr.shape == (1, 3, 192, 192)
    assert set(visuals) == {'LL2', 'LH2', 'HL2', 'HH2'}
    with tempfile.TemporaryDirectory() as temporary_dir:
        visual_writer = SimpleNamespace(opt={
            'path': {'visualization': temporary_dir},
            'val': {'wavelet_vis_save_raw': False},
        })
        FeMaSRModel._save_wavelet_visuals(
            visual_writer, visuals, 'smoke', 1, 'sample')
        output_dir = os.path.join(temporary_dir, 'wavelet_bands', 'smoke', '1', 'sample')
        assert len([name for name in os.listdir(output_dir) if name.endswith('.png')]) == 10
        assert cv2.imread(os.path.join(output_dir, 'LL2_D2_before_after.png')).shape == (512, 1024, 3)
        assert cv2.imread(os.path.join(output_dir, 'LL2_D2_signed_channel0.png')).shape == (512, 1024, 3)


def test_official_single_codebook_path():
    net = FeMaSRNet(codebook_params=[[32, 1024, 512]], gt_resolution=256,
                    LQ_stage=False, use_semantic_loss=False)
    assert len(net.quantize_group) == 1
    assert net.wavelet_vq is False
    with torch.no_grad():
        output, _, _, indices = net(torch.randn(1, 3, 256, 256))
    assert output.shape == (1, 3, 256, 256)
    assert len(indices) == 1 and indices[0].shape == (1, 1, 32, 32)


def test_validation_cycle_writes_wavelet_artifacts():
    with tempfile.TemporaryDirectory() as temporary_dir:
        opt = {
            'name': 'wavelet-smoke', 'num_gpu': int(torch.cuda.is_available()),
            'is_train': False, 'dist': False,
            'network_g': {
                'type': 'FeMaSRNet', 'codebook_params': [[32, 512, 512]],
                'gt_resolution': 256, 'wavelet_vq': True, 'detail_codebook_size': 512,
            },
            'val': {'metrics': None, 'wavelet_vis_max_samples': 1,
                    'wavelet_vis_save_raw': False},
            'path': {'visualization': temporary_dir},
        }
        model = FeMaSRModel(opt)

        class OneSampleLoader:
            dataset = SimpleNamespace(opt={'name': 'smoke'})

            def __len__(self):
                return 1

            def __iter__(self):
                image = torch.rand(1, 3, 256, 256)
                yield {'lq': image, 'gt': image, 'lq_path': ['sample.png']}

        model.nondist_validation(OneSampleLoader(), 5, None, False, None)
        sheet_path = os.path.join(temporary_dir, 'wavelet_bands', 'smoke',
                                  '5', 'sample', 'LL2_D2_before_after.png')
        assert os.path.isfile(sheet_path)


def test_single_codebook_hq_checkpoint_is_rejected_for_lq():
    with tempfile.TemporaryDirectory() as temporary_dir:
        checkpoint_path = os.path.join(temporary_dir, 'old_hq.pth')
        torch.save({'params': {
            'quantize_group.0.embedding.weight': torch.empty(1024, 512),
        }}, checkpoint_path)
        opt = {
            'num_gpu': 0, 'is_train': False, 'dist': False,
            'network_g': {
                'type': 'FeMaSRNet', 'codebook_params': [[32, 512, 512]],
                'gt_resolution': 256, 'wavelet_vq': True,
                'detail_codebook_size': 512, 'LQ_stage': True,
            },
            'val': {'metrics': None},
            'path': {'pretrain_network_hq': checkpoint_path, 'strict_load': False},
        }
        try:
            FeMaSRModel(opt)
        except ValueError as error:
            assert 'compatible wavelet codebook' in str(error)
        else:
            raise AssertionError('An old single-codebook HQ checkpoint was accepted')


if __name__ == '__main__':
    test_haar_round_trip()
    test_hq_lq_teacher_and_validation_images()
    test_official_single_codebook_path()
    test_validation_cycle_writes_wavelet_artifacts()
    test_single_codebook_hq_checkpoint_is_rejected_for_lq()
    print('Wavelet VQ smoke checks passed')
