import importlib
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (
    REPO_ROOT / 'src',
    REPO_ROOT / 'torchlogix' / 'src',
    REPO_ROOT / 'da4ml' / 'src',
):
    sys.path.insert(0, str(path))

from da4ml_torch.tools import torchlogix_to_da4ml as tool


def test_find_named_checkpoint_directory(tmp_path):
    models_root = tmp_path / 'models'
    model_dir = models_root / 'toy_model'
    model_dir.mkdir(parents=True)
    checkpoint = model_dir / 'best_model.pth'
    checkpoint.write_bytes(b'checkpoint')

    assert tool.find_named_checkpoint('toy_model', models_root) == checkpoint.resolve()


def test_auto_sequential_checkpoint_fallback_loads_missing_top_class(tmp_path):
    module_dir = tmp_path / 'module_src'
    checkpoint_dir = tmp_path / 'checkpoints'
    module_dir.mkdir()
    checkpoint_dir.mkdir()
    module_path = module_dir / 'toy_missing_model.py'
    module_path.write_text(
        '\n'.join(
            [
                'import torch',
                '',
                'class ToyMissingModel(torch.nn.Module):',
                '    def __init__(self):',
                '        super().__init__()',
                '        self.layers = torch.nn.Sequential(torch.nn.Identity())',
                '',
                '    def forward(self, x):',
                '        return self.layers(x)',
                '',
            ]
        )
    )

    sys.path.insert(0, str(module_dir))
    try:
        module = importlib.import_module('toy_missing_model')
        checkpoint = checkpoint_dir / 'toy.pt'
        torch.save(module.ToyMissingModel(), checkpoint)
    finally:
        sys.path.remove(str(module_dir))
        sys.modules.pop('toy_missing_model', None)

    payload, unsafe, notes = tool._torch_load(checkpoint, sequential_fallback='auto')

    assert unsafe is True
    assert isinstance(payload, tool.AutoSequentialCheckpointModule)
    assert notes
    x = torch.tensor([[1.0, 2.0]])
    torch.testing.assert_close(payload(x), x)


def test_auto_sequential_checkpoint_fallback_can_be_disabled(tmp_path):
    module_dir = tmp_path / 'module_src'
    checkpoint_dir = tmp_path / 'checkpoints'
    module_dir.mkdir()
    checkpoint_dir.mkdir()
    (module_dir / 'toy_missing_model_never.py').write_text(
        'import torch\n'
        'class ToyMissingModelNever(torch.nn.Module):\n'
        '    def __init__(self):\n'
        '        super().__init__()\n'
        '        self.identity = torch.nn.Identity()\n'
        '    def forward(self, x):\n'
        '        return self.identity(x)\n'
    )

    sys.path.insert(0, str(module_dir))
    try:
        module = importlib.import_module('toy_missing_model_never')
        checkpoint = checkpoint_dir / 'toy.pt'
        torch.save(module.ToyMissingModelNever(), checkpoint)
    finally:
        sys.path.remove(str(module_dir))
        sys.modules.pop('toy_missing_model_never', None)

    with pytest.raises(Exception):
        tool._torch_load(checkpoint, sequential_fallback='never')


def test_enrich_report_estimates_adds_target_timing():
    report = tool.enrich_report_estimates({'cost': 123, 'clock_period': 5.0, 'latency': 7})

    assert report['rough_LUT_estimate'] == 123
    assert report['target_Fmax(MHz)'] == 200.0
    assert report['target_latency(ns)'] == 35.0
    assert report['timing_estimate_source'] == 'target_clock_metadata'
