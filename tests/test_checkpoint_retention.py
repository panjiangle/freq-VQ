"""Ensure cloud checkpoint pruning keeps resumable state and best/latest files."""
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from basicsr.models.femasr_model import FeMaSRModel


class DummyModel:
    def __init__(self, root):
        self.opt = {'logger': {'keep_last_checkpoints': 3},
                    'path': {'models': str(root / 'models'),
                             'training_states': str(root / 'states')}}
        self.net_g = SimpleNamespace(wavelet_vq=True)
        self.net_d = object()

    def get_bare_model(self, net):
        return net

    def save_network(self, net, name, iteration):
        (Path(self.opt['path']['models']) / f'{name}_{iteration}.pth').touch()

    def save_training_state(self, epoch, iteration):
        (Path(self.opt['path']['training_states']) / f'{iteration}.state').touch()


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / 'models').mkdir()
        (root / 'states').mkdir()
        for iteration in (5000, 10000, 15000):
            for name in ('g', 'd'):
                (root / 'models' / f'net_{name}_{iteration}.pth').touch()
            (root / 'states' / f'{iteration}.state').touch()
        (root / 'models' / 'net_g_latest.pth').touch()
        (root / 'models' / 'net_g_best_.pth').touch()
        FeMaSRModel.save(DummyModel(root), 1, 20000)
        assert not (root / 'states' / '5000.state').exists()
        assert not (root / 'models' / 'net_g_5000.pth').exists()
        assert sorted(int(path.stem) for path in (root / 'states').glob('*.state')) == [10000, 15000, 20000]
        assert (root / 'models' / 'net_g_latest.pth').exists()
        assert (root / 'models' / 'net_g_best_.pth').exists()
    print('Checkpoint retention checks passed')


if __name__ == '__main__':
    main()
