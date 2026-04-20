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


def test_auto_sequential_checkpoint_fallback_is_rejected_for_conversion():
    loaded = tool.LoadedModel(
        model=tool.AutoSequentialCheckpointModule(),
        model_name='toy',
        architecture=None,
        checkpoint=None,
        checkpoint_kind='full_module_auto_sequential',
        training_config=None,
        unsafe_pickle_load=True,
        load_notes=(),
    )

    with pytest.raises(ValueError, match='cannot reconstruct a custom forward'):
        tool.require_real_forward_for_conversion(loaded)


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


def test_infer_input_shape_for_vector_binarized_conv_model():
    from torchlogix.layers import FixedBinarization, LogicConv2d

    model = torch.nn.Sequential(
        FixedBinarization(torch.zeros(105, 100)),
        LogicConv2d(in_dim=(13, 8), channels=100, num_kernels=4, receptive_field_size=3, tree_depth=1),
    )

    assert tool.infer_input_shapes(model) == ((1, 105),)


def test_infer_input_shape_for_channel_binarized_conv_model():
    from torchlogix.layers import FixedBinarization, LogicConv2d

    model = torch.nn.Sequential(
        FixedBinarization(torch.zeros(3, 4), feature_dim=1),
        LogicConv2d(in_dim=(5, 5), channels=12, num_kernels=4, receptive_field_size=3, tree_depth=1),
    )

    assert tool.infer_input_shapes(model) == ((1, 3, 5, 5),)


def test_enrich_report_estimates_adds_target_timing():
    report = tool.enrich_report_estimates({'cost': 123, 'clock_period': 5.0, 'latency': 7})

    assert report['rough_LUT_estimate'] == 123
    assert report['rough_ASIC_estimate'] == 123
    assert report['target_Fmax(MHz)'] == 200.0
    assert report['target_latency(ns)'] == 35.0
    assert report['timing_estimate_source'] == 'target_clock_metadata'


def test_estimate_lines_include_asic_metadata():
    report = tool.enrich_report_estimates({'cost': 123})

    lines = tool._estimate_lines(report)

    assert 'Rough LUT estimate: 123' in lines
    assert 'Rough ASIC estimate: 123' in lines


def test_validate_rtl_compares_against_torch_model_output():
    class DoubleModel(torch.nn.Module):
        def forward(self, x):
            return x * 2

    class DoubleRtl:
        def predict(self, data, n_threads=1):
            return data.astype('float32') * 2

    summary = tool.validate_rtl(
        DoubleModel(),
        DoubleRtl(),
        shapes=((1, 4),),
        n_samples=8,
        input_kif=(0, 1, 0),
        mode='binary',
        value_range=(0.0, 1.0),
        n_threads=1,
        atol=1e-5,
    )

    assert summary.passed
    assert summary.mismatches == 0


def test_validate_rtl_reports_model_rtl_mismatches():
    class DoubleModel(torch.nn.Module):
        def forward(self, x):
            return x * 2

    class OffsetRtl:
        def predict(self, data, n_threads=1):
            return data.astype('float32') * 2 + 1

    summary = tool.validate_rtl(
        DoubleModel(),
        OffsetRtl(),
        shapes=((1, 4),),
        n_samples=8,
        input_kif=(0, 1, 0),
        mode='binary',
        value_range=(0.0, 1.0),
        n_threads=1,
        atol=1e-5,
    )

    assert not summary.passed
    assert summary.mismatches > 0


def test_make_vivado_script_compatible_removes_global_retiming(tmp_path):
    script = tmp_path / 'build_vivado_prj.tcl'
    script.write_text(
        'synth_design -top $top_module -mode out_of_context -global_retiming on \\\n'
        '    -flatten_hierarchy full\n'
    )

    patched = tool.make_vivado_script_compatible(tmp_path)

    assert patched == (script,)
    assert '-global_retiming' not in script.read_text()
