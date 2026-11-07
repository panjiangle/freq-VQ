"""Check that the paired loader preserves existing LR pixels and matches IDs."""
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from basicsr.data.paired_image_dataset import PairedImageDataset


def main():
    with tempfile.TemporaryDirectory() as root:
        hr_dir = os.path.join(root, 'hr')
        lr_dir = os.path.join(root, 'lr')
        os.makedirs(hr_dir)
        os.makedirs(lr_dir)
        hr = np.zeros((264, 264, 3), dtype=np.uint8)
        lr = np.full((66, 66, 3), 113, dtype=np.uint8)
        cv2.imwrite(os.path.join(hr_dir, 'DIV2K_0001.png'), hr)
        cv2.imwrite(os.path.join(lr_dir, 'DIV2K_0001x4.png'), lr)
        opt = dict(dataroot_gt=hr_dir, dataroot_lq=lr_dir,
                   io_backend=dict(type='disk'), scale=4, phase='train', gt_size=256,
                   use_resize_crop=False, paired_crop_only=True, paired_by_stem=True,
                   align_gt_to_lq=True, use_flip=False, use_rot=False)
        sample = PairedImageDataset(opt)[0]
        assert sample['gt'].shape == (3, 256, 256)
        assert sample['lq'].shape == (3, 64, 64)
        assert np.isclose(sample['lq'].numpy().mean(), 113 / 255)
        cv2.imwrite(os.path.join(hr_dir, 'DIV2K_0002.png'), hr)
        try:
            PairedImageDataset(opt)
        except ValueError as exc:
            assert 'do not match' in str(exc)
        else:
            raise AssertionError('An unpaired HR image was silently accepted')
    print('Official pair loader checks passed')


if __name__ == '__main__':
    main()
