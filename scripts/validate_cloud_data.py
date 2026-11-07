"""Validate existing HR/x4 LR pairs without modifying image pixels."""
from pathlib import Path
import struct
import sys


ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / 'datasets' / 'DF2K_official')


def png_size(path):
    with path.open('rb') as handle:
        header = handle.read(24)
        handle.seek(-12, 2)
        trailer = handle.read(12)
    if header[:8] != b'\x89PNG\r\n\x1a\n' or header[12:16] != b'IHDR':
        raise ValueError(f'Invalid PNG header: {path}')
    if trailer != b'\x00\x00\x00\x00IEND\xaeB`\x82':
        raise ValueError(f'Incomplete PNG: {path}')
    return struct.unpack('>II', header[16:24])


def check(split, expected):
    hr = ROOT / f'{split}_HR'
    lr = ROOT / f'{split}_LR_x4'
    hrs = {p.stem: p for p in hr.glob('*.png')}
    lrs = {p.stem[:-2]: p for p in lr.glob('*x4.png')}
    if len(hrs) != expected or len(lrs) != expected or hrs.keys() != lrs.keys():
        raise ValueError(f'{split}: expected {expected} aligned IDs; got HR={len(hrs)}, LR={len(lrs)}')
    for key, hpath in hrs.items():
        hw, hh = png_size(hpath)
        lw, lh = png_size(lrs[key])
        if not (0 <= hw - 4 * lw < 4 and 0 <= hh - 4 * lh < 4):
            raise ValueError(f'{key}: HR={hw}x{hh}, LR={lw}x{lh}')
    print(f'{split}: {expected} existing HR/LR x4 pairs checked', flush=True)


if __name__ == '__main__':
    check('train', 3450)
    check('val', 100)
