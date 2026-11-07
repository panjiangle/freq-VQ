"""Fetch published Flickr2K HR/x4 LR pairs from a research mirror.

The original publisher host is unavailable from this instance. This script
downloads existing paired PNGs; it never synthesizes LR images.
Source: https://huggingface.co/datasets/bezdarnost/DF2K-bicubic
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlopen
import os
import struct
import sys
import time


ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else '/root/autodl-tmp/freq-vq/datasets/DF2K_official')
BASE = 'https://hf-mirror.com/datasets/bezdarnost/DF2K-bicubic/resolve/main'
HR = ROOT / 'raw' / 'Flickr2K_HR'
LR = ROOT / 'raw' / 'Flickr2K_LR_x4'
for folder in (HR, LR, ROOT / 'train_HR', ROOT / 'train_LR_x4'):
    folder.mkdir(parents=True, exist_ok=True)


def png_size(path):
    with path.open('rb') as handle:
        header = handle.read(24)
    if header[:8] != b'\x89PNG\r\n\x1a\n' or header[12:16] != b'IHDR':
        raise ValueError(f'Invalid PNG: {path}')
    return struct.unpack('>II', header[16:24])


def fetch(path, url):
    if path.exists():
        try:
            png_size(path)
            return
        except ValueError:
            path.unlink()
    part = path.with_suffix('.part')
    for attempt in range(5):
        try:
            with urlopen(url, timeout=90) as response, part.open('wb') as target:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    target.write(chunk)
            png_size(part)
            part.replace(path)
            return
        except Exception:
            part.unlink(missing_ok=True)
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))


def get_pair(number):
    stem = f'{number:06d}'
    hr = HR / f'{stem}.png'
    lr = LR / f'{stem}x4.png'
    fetch(hr, f'{BASE}/hr/{hr.name}')
    fetch(lr, f'{BASE}/x4/{lr.name}')
    hw, hh = png_size(hr)
    lw, lh = png_size(lr)
    if not (0 <= hw - 4 * lw < 4 and 0 <= hh - 4 * lh < 4):
        raise ValueError(f'Unaligned pair {stem}: HR={hw}x{hh}, LR={lw}x{lh}')
    for target, source in ((ROOT / 'train_HR' / f'Flickr2K_{hr.name}', hr),
                           (ROOT / 'train_LR_x4' / f'Flickr2K_{lr.name}', lr)):
        if not target.exists():
            target.symlink_to(source)
    return stem


def main():
    failures = []
    with ThreadPoolExecutor(max_workers=48) as pool:
        futures = {pool.submit(get_pair, i): i for i in range(1, 2651)}
        for count, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as exc:
                failures.append((futures[future], repr(exc)))
                print(f'FAILED {futures[future]}: {exc}', flush=True)
            if count % 100 == 0:
                print(f'Checked {count}/2650 pairs; failed={len(failures)}', flush=True)
    if failures:
        raise RuntimeError(f'{len(failures)} Flickr2K pairs failed: {failures[:10]}')
    print('Flickr2K mirror HR/LR x4: 2650 aligned pairs ready', flush=True)


if __name__ == '__main__':
    main()
