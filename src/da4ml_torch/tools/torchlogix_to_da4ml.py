from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULTS: dict[str, Any] = {
    'checkpoint': None,
    'outdir': None,
    'project_name': 'model',
    'architecture': None,
    'training_config': 'auto',
    'model_factory': None,
    'model_kwargs': {},
    'input_shape': None,
    'input_kif': None,
    'hw_config': [1, -1, -1],
    'solver_options': {},
    'flavor': 'verilog',
    'latency_cutoff': 5.0,
    'clock_period': 5.0,
    'clock_uncertainty': 10.0,
    'part_name': 'xcvu13p-flga2577-2-e',
    'validate_comb_samples': 0,
    'validate_rtl_samples': 0,
    'validation_input_mode': 'auto',
    'validation_input_range': [0.0, 1.0],
    'validation_atol': 1e-5,
    'n_threads': 4,
    'compile_rtl': False,
    'synthesis_tool': 'none',
    'report_formats': ['json', 'md'],
    'overwrite': False,
}

CHECKPOINT_PATTERNS = (
    'best_model.pt',
    'best_model.pth',
    'bestmodel.pt',
    'bestmodel.pth',
    'bestmodelpth',
    '*best*.pt',
    '*best*.pth',
)


@dataclass(frozen=True)
class LoadedModel:
    model: torch.nn.Module
    architecture: str | None
    checkpoint: Path | None
    checkpoint_kind: str
    training_config: Path | None
    unsafe_pickle_load: bool


@dataclass(frozen=True)
class ValidationSummary:
    samples: int
    total_outputs: int
    mismatches: int
    max_abs_diff: float
    passed: bool


def _path_or_none(value: str | Path | None) -> Path | None:
    if value in (None, '', 'none', 'None'):
        return None
    return Path(value).expanduser()


def _json_value(value: str | dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    possible_path = Path(value).expanduser()
    if possible_path.exists():
        with possible_path.open() as f:
            return json.load(f)
    return json.loads(value)


def _parse_shape_item(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        parts = value.replace('x', ',').split(',')
        return tuple(int(part.strip()) for part in parts if part.strip())
    return tuple(int(part) for part in value)


def parse_input_shapes(value: str | Sequence[Any] | None) -> tuple[tuple[int, ...], ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return tuple(_parse_shape_item(item) for item in value.split(';') if item.strip())
    if value and all(isinstance(item, int) for item in value):
        return (_parse_shape_item(value),)
    return tuple(_parse_shape_item(item) for item in value)


def _serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_serializable(item) for item in value]
    if isinstance(value, list):
        return [_serializable(item) for item in value]
    if isinstance(value, dict):
        return {key: _serializable(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_json_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.expanduser().open() as f:
        return json.load(f)


def merge_config(args: argparse.Namespace) -> dict[str, Any]:
    config = DEFAULTS.copy()
    config.update(_load_json_config(args.config))

    for key, value in vars(args).items():
        if key == 'config' or value is None:
            continue
        config[key] = value

    config['checkpoint'] = _path_or_none(config.get('checkpoint'))
    config['outdir'] = _path_or_none(config.get('outdir'))
    training_config = config.get('training_config')
    if training_config != 'auto':
        config['training_config'] = _path_or_none(training_config)

    config['model_kwargs'] = _json_value(config.get('model_kwargs'))
    config['solver_options'] = _json_value(config.get('solver_options'))
    config['input_shape'] = parse_input_shapes(config.get('input_shape'))
    config['input_kif'] = tuple(config['input_kif']) if config.get('input_kif') is not None else None
    config['hw_config'] = tuple(config['hw_config'])
    config['validation_input_range'] = tuple(config['validation_input_range'])
    if isinstance(config.get('report_formats'), str):
        config['report_formats'] = [config['report_formats']]
    return config


def find_default_checkpoint(models_root: Path = Path('models')) -> Path | None:
    if not models_root.exists():
        return None
    matches: list[Path] = []
    for pattern in CHECKPOINT_PATTERNS:
        matches.extend(models_root.rglob(pattern))
    unique_matches = sorted(set(path.resolve() for path in matches if path.is_file()))
    if not unique_matches:
        return None
    if len(unique_matches) > 1:
        formatted = '\n'.join(f'  - {path}' for path in unique_matches)
        raise RuntimeError(f'Multiple candidate checkpoints found. Pass --checkpoint explicitly:\n{formatted}')
    return unique_matches[0]


def _torch_load(path: Path) -> tuple[Any, bool]:
    for candidate in (Path.cwd(), path.resolve().parent):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
    try:
        return torch.load(path, map_location='cpu', weights_only=True), False
    except TypeError:
        return torch.load(path, map_location='cpu'), True
    except Exception:
        return torch.load(path, map_location='cpu', weights_only=False), True


def _is_state_dict(value: Any) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    return all(torch.is_tensor(item) or isinstance(item, torch.nn.Parameter) for item in value.values())


def _extract_state_dict(payload: Any) -> tuple[dict[str, torch.Tensor] | None, str]:
    if isinstance(payload, torch.nn.Module):
        return None, 'full_module'
    if _is_state_dict(payload):
        return dict(payload), 'state_dict'
    if isinstance(payload, dict):
        for key in ('state_dict', 'model_state_dict', 'model'):
            nested = payload.get(key)
            if isinstance(nested, torch.nn.Module):
                return None, 'full_module'
            if _is_state_dict(nested):
                return dict(nested), key
    return None, type(payload).__name__


def _clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {
        key: value
        for key, value in state_dict.items()
        if '_export_lut_ids' not in key
    }
    prefixes = ('module.', '_orig_mod.')
    for prefix in prefixes:
        if cleaned and all(key.startswith(prefix) for key in cleaned):
            cleaned = {key[len(prefix):]: value for key, value in cleaned.items()}
    return cleaned


def _extract_thresholds(state_dict: dict[str, torch.Tensor] | None) -> torch.Tensor | None:
    if state_dict is None:
        return None
    for key, value in state_dict.items():
        if key.endswith('thresholds') and torch.is_tensor(value):
            return value.detach().cpu()
    for key, value in state_dict.items():
        if key.endswith('raw_diffs') and torch.is_tensor(value):
            return torch.cumsum(value.detach().cpu(), dim=-1)
    return None


def _resolve_object(spec: str) -> Any:
    module_name, sep, attr = spec.partition(':')
    if not sep:
        module_name, _, attr = spec.rpartition('.')
    if not module_name or not attr:
        raise ValueError(f'Expected import spec "module:object", got {spec!r}.')
    module = importlib.import_module(module_name)
    obj = module
    for part in attr.split('.'):
        obj = getattr(obj, part)
    return obj


def _training_config_path(checkpoint: Path | None, configured: str | Path | None) -> Path | None:
    if configured == 'auto':
        if checkpoint is None:
            return None
        candidate = checkpoint.parent / 'training_config.json'
        return candidate if candidate.exists() else None
    return _path_or_none(configured)


def _training_model_kwargs(config: dict[str, Any], thresholds: torch.Tensor | None, model_cls: type) -> dict[str, Any]:
    n_input_bits = getattr(model_cls, 'n_input_bits', None)
    if thresholds is None and n_input_bits is not None:
        thresholds = torch.zeros(n_input_bits, dtype=torch.float32)

    kwargs = {
        'connections': config.get('connections', 'fixed'),
        'connections_kwargs': {
            'init_method': config.get('connections_init_method', 'random'),
            'temperature': config.get('connections_temperature', 0.001),
            'gumbel': config.get('connections_gumbel', True),
        },
        'parametrization': config.get('parametrization', 'raw'),
        'parametrization_kwargs': {
            'temperature': config.get('parametrization_temperature', 1.0),
            'forward_sampling': config.get('forward_sampling', 'soft'),
            'weight_init': config.get('weight_init', 'residual'),
            'residual_probability': config.get('residual_probability', 0.951),
        },
        'device': 'cpu',
        'lut_rank': config.get('lut_rank', 2),
    }

    if thresholds is not None:
        kwargs.update(
            {
                'thresholds': thresholds,
                'binarization': config.get('binarization', 'fixed'),
                'binarization_kwargs': {
                    'one_per': config.get('binarization_per', 'global'),
                    'temperature_sampling': config.get('binarization_temperature', 0.001),
                    'temperature_softplus': config.get('binarization_temperature_softplus', 0.01),
                    'forward_sampling': config.get('binarization_forward_sampling', 'soft'),
                },
            }
        )
    return kwargs


def instantiate_torchlogix_model(
    architecture: str,
    state_dict: dict[str, torch.Tensor] | None,
    training_config: dict[str, Any] | None,
    model_kwargs: dict[str, Any],
) -> torch.nn.Module:
    import torchlogix.models

    if architecture not in torchlogix.models.__dict__:
        raise ValueError(f'Unknown TorchLogix architecture {architecture!r}. Use --list-architectures to inspect options.')

    model_cls = torchlogix.models.__dict__[architecture]
    base_config = training_config or {}
    thresholds = _extract_thresholds(state_dict)
    kwargs = _training_model_kwargs(base_config, thresholds, model_cls)
    kwargs.update(model_kwargs)
    model = model_cls(**kwargs)

    if state_dict is not None:
        model.load_state_dict(_clean_state_dict(state_dict))
    return model


def load_model(config: dict[str, Any]) -> LoadedModel:
    checkpoint = config['checkpoint'] or find_default_checkpoint()
    payload: Any = None
    unsafe_pickle_load = False
    state_dict: dict[str, torch.Tensor] | None = None
    checkpoint_kind = 'factory'

    if checkpoint is not None:
        checkpoint = checkpoint.resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
        payload, unsafe_pickle_load = _torch_load(checkpoint)
        if isinstance(payload, torch.nn.Module):
            model = payload
            return LoadedModel(model, None, checkpoint, 'full_module', None, unsafe_pickle_load)
        state_dict, checkpoint_kind = _extract_state_dict(payload)

    if config.get('model_factory'):
        factory = _resolve_object(config['model_factory'])
        model = factory(**config['model_kwargs'])
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f'Model factory {config["model_factory"]!r} returned {type(model)}, not torch.nn.Module.')
        if state_dict is not None:
            model.load_state_dict(_clean_state_dict(state_dict))
        return LoadedModel(model, config.get('architecture'), checkpoint, checkpoint_kind, None, unsafe_pickle_load)

    training_config_path = _training_config_path(checkpoint, config.get('training_config'))
    training_config = _load_json_config(training_config_path)
    architecture = config.get('architecture') or training_config.get('architecture')
    if architecture is None:
        if state_dict is None:
            raise ValueError('No model source was provided. Pass --checkpoint, --model-factory, or --architecture.')
        raise ValueError(
            'Checkpoint contains a state_dict, so the model architecture is required. '
            'Pass --architecture, --training-config, or --model-factory.'
        )

    model = instantiate_torchlogix_model(
        architecture=architecture,
        state_dict=state_dict,
        training_config=training_config,
        model_kwargs=config['model_kwargs'],
    )
    return LoadedModel(model, architecture, checkpoint, checkpoint_kind, training_config_path, unsafe_pickle_load)


def prepare_model_for_export(model: torch.nn.Module) -> None:
    model.cpu()
    model.eval()
    set_logic_export_mode(model, True)


def set_logic_export_mode(model: torch.nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if hasattr(module, 'set_export_mode'):
            module.set_export_mode(enabled)


def _temporarily_disable_export_for_torch_validation(model: torch.nn.Module) -> bool:
    changed = False
    for module in model.modules():
        if hasattr(module, 'set_export_mode') and getattr(module, 'export_mode', False):
            module.set_export_mode(False)
            changed = True
    return changed


def _is_threshold_binarization(module: torch.nn.Module) -> bool:
    try:
        from torchlogix.layers.binarization import FixedBinarization, LearnableBinarization, SoftBinarization
    except Exception:
        return False
    return isinstance(module, (FixedBinarization, LearnableBinarization, SoftBinarization))


def _is_dummy_binarization(module: torch.nn.Module) -> bool:
    try:
        from torchlogix.layers.binarization import DummyBinarization
    except Exception:
        return False
    return isinstance(module, DummyBinarization)


def has_threshold_binarization(model: torch.nn.Module) -> bool:
    return any(_is_threshold_binarization(module) for module in model.modules())


def infer_input_shapes(model: torch.nn.Module) -> tuple[tuple[int, ...], ...] | None:
    try:
        from torchlogix.layers import LogicConv2d, LogicDense
        from torchlogix.layers.binarization import Binarization
    except Exception:
        return None

    first_binarization: Binarization | None = None
    for module in model.modules():
        if module is model:
            continue
        if isinstance(module, Binarization):
            first_binarization = module
            continue
        if isinstance(module, LogicConv2d):
            if first_binarization is not None and not _is_dummy_binarization(first_binarization):
                thresholds = first_binarization.get_thresholds()
                n_bits = int(thresholds.shape[-1])
                raw_channels = module.channels // n_bits if module.channels % n_bits == 0 else module.channels
                return ((1, raw_channels, *module.in_dim),)
            return ((1, module.channels, *module.in_dim),)
        if isinstance(module, LogicDense):
            if first_binarization is not None and not _is_dummy_binarization(first_binarization):
                thresholds = first_binarization.get_thresholds()
                n_bits = int(thresholds.shape[-1])
                raw_dim = module.in_dim // n_bits if module.in_dim % n_bits == 0 else module.in_dim
                return ((1, raw_dim),)
            return ((1, module.in_dim),)
    return None


def choose_input_kif(configured: tuple[int, int, int] | None, model: torch.nn.Module) -> tuple[int, int, int]:
    if configured is not None:
        return configured
    if has_threshold_binarization(model):
        return (0, 1, 8)
    return (0, 1, 0)


def make_symbolic_inputs(
    shapes: tuple[tuple[int, ...], ...],
    input_kif: tuple[int, int, int],
    hw_config: tuple[int, int, int],
    solver_options: dict[str, Any],
) -> Any:
    from da4ml.trace import FixedVariableArrayInput

    inputs = tuple(
        FixedVariableArrayInput(shape, hw_config, solver_options).quantize(*input_kif)
        for shape in shapes
    )
    return inputs[0] if len(inputs) == 1 else inputs


def trace_to_comb(
    model: torch.nn.Module,
    symbolic_inputs: Any,
    hw_config: tuple[int, int, int],
    solver_options: dict[str, Any],
):
    from da4ml.trace import HWConfig, comb_trace
    from da4ml_torch.parser import TorchParser

    parser = TorchParser(model, HWConfig(*hw_config), solver_options)
    inp, out = parser.trace(inputs=symbolic_inputs)
    return comb_trace(inp, out)


def _validation_mode(configured: str, model: torch.nn.Module) -> str:
    if configured != 'auto':
        return configured
    return 'uniform' if has_threshold_binarization(model) else 'binary'


def _make_one_validation_input(
    shape: tuple[int, ...],
    n_samples: int,
    input_kif: tuple[int, int, int],
    mode: str,
    value_range: tuple[float, float],
) -> np.ndarray:
    data_shape = (n_samples, *shape[1:])
    if mode == 'binary':
        return np.random.randint(0, 2, data_shape).astype(np.bool_)

    k, i, f = input_kif
    step = 2.0**-f
    low, high = value_range
    q_low = int(np.ceil(low / step))
    q_high = int(np.floor(high / step))
    if k:
        q_low = max(q_low, int(-2**i / step))
    else:
        q_low = max(q_low, 0)
    q_high = min(q_high, int((2**i - step) / step))
    if q_low > q_high:
        raise ValueError(f'Validation range {value_range} is incompatible with input KIF {input_kif}.')
    return (np.random.randint(q_low, q_high + 1, data_shape) * step).astype(np.float32)


def make_validation_inputs(
    shapes: tuple[tuple[int, ...], ...],
    n_samples: int,
    input_kif: tuple[int, int, int],
    mode: str,
    value_range: tuple[float, float],
) -> tuple[np.ndarray, ...]:
    return tuple(_make_one_validation_input(shape, n_samples, input_kif, mode, value_range) for shape in shapes)


def _flatten_model_output(output: Any, n_samples: int) -> np.ndarray:
    if isinstance(output, (list, tuple)):
        parts = [_flatten_model_output(item, n_samples) for item in output]
        return np.concatenate(parts, axis=1)
    if torch.is_tensor(output):
        return output.detach().cpu().numpy().reshape(n_samples, -1)
    return np.asarray(output).reshape(n_samples, -1)


def validate_comb(
    model: torch.nn.Module,
    comb: Any,
    shapes: tuple[tuple[int, ...], ...],
    n_samples: int,
    input_kif: tuple[int, int, int],
    mode: str,
    value_range: tuple[float, float],
    n_threads: int,
    atol: float,
) -> ValidationSummary:
    data = make_validation_inputs(shapes, n_samples, input_kif, mode, value_range)
    torch_inputs = [
        torch.from_numpy(item.astype(np.float32) if item.dtype == np.bool_ else item)
        for item in data
    ]
    restore_export = _temporarily_disable_export_for_torch_validation(model)
    try:
        with torch.no_grad():
            torch_out = _flatten_model_output(model(*torch_inputs), n_samples)
    finally:
        if restore_export:
            set_logic_export_mode(model, True)

    comb_input: Any = data[0] if len(data) == 1 else data
    comb_out = np.asarray(comb.predict(comb_input, n_threads=n_threads)).reshape(n_samples, -1)
    diff = np.abs(comb_out - torch_out)
    mismatches = int(np.sum(diff > atol))
    total = int(diff.size)
    return ValidationSummary(
        samples=n_samples,
        total_outputs=total,
        mismatches=mismatches,
        max_abs_diff=float(np.max(diff)) if diff.size else 0.0,
        passed=mismatches == 0,
    )


def validate_rtl(
    rtl_model: Any,
    comb: Any,
    shapes: tuple[tuple[int, ...], ...],
    n_samples: int,
    input_kif: tuple[int, int, int],
    mode: str,
    value_range: tuple[float, float],
    n_threads: int,
    atol: float,
) -> ValidationSummary:
    data = make_validation_inputs(shapes, n_samples, input_kif, mode, value_range)
    comb_input: Any = data[0] if len(data) == 1 else data
    comb_out = np.asarray(comb.predict(comb_input, n_threads=n_threads)).reshape(n_samples, -1)
    rtl_out = np.asarray(rtl_model.predict(comb_input, n_threads=n_threads)).reshape(n_samples, -1)
    diff = np.abs(comb_out - rtl_out)
    mismatches = int(np.sum(diff > atol))
    total = int(diff.size)
    return ValidationSummary(
        samples=n_samples,
        total_outputs=total,
        mismatches=mismatches,
        max_abs_diff=float(np.max(diff)) if diff.size else 0.0,
        passed=mismatches == 0,
    )


def run_synthesis(tool: str, outdir: Path) -> None:
    if tool == 'none':
        return
    if tool == 'vivado':
        executable = shutil.which('vivado')
        if executable is None:
            raise RuntimeError('Vivado was requested but "vivado" was not found on PATH.')
        command = [executable, '-mode', 'batch', '-source', 'build_vivado_prj.tcl']
    elif tool == 'quartus':
        executable = shutil.which('quartus_sh')
        if executable is None:
            raise RuntimeError('Quartus was requested but "quartus_sh" was not found on PATH.')
        command = [executable, '-t', 'build_quartus_prj.tcl']
    else:
        raise ValueError(f'Unsupported synthesis tool: {tool}')
    subprocess.run(command, cwd=outdir, check=True)


def da4ml_project_report(outdir: Path) -> dict[str, Any]:
    from da4ml._cli.report import load_project

    report = load_project(outdir)
    if report is None:
        raise RuntimeError(f'Unable to load DA4ML report from {outdir}.')
    return report


def _write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    preferred = [
        'flavor',
        'part_name',
        'cost',
        'latency',
        'latency_cutoff',
        'clock_period',
        'actual_period',
        'Fmax(MHz)',
        'latency(ns)',
        'LUT',
        'FF',
        'DSP',
        'RAMB18',
        'Block RAM Tile',
        'Total On-Chip Power (W)',
    ]
    keys = [key for key in preferred if key in report]
    if not keys:
        keys = sorted(report)
    with path.open('w') as f:
        f.write('| metric | value |\n')
        f.write('|---|---|\n')
        for key in keys:
            f.write(f'| {key} | {report[key]} |\n')


def write_reports(outdir: Path, formats: Sequence[str]) -> dict[str, Any]:
    report = da4ml_project_report(outdir)
    analysis_dir = outdir / 'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        if fmt == 'json':
            with (analysis_dir / 'da4ml_report.json').open('w') as f:
                json.dump(report, f, indent=2)
        elif fmt == 'md':
            _write_markdown_report(analysis_dir / 'da4ml_report.md', report)
        else:
            raise ValueError(f'Unsupported report format: {fmt}')
    return report


def default_outdir(checkpoint: Path | None, architecture: str | None) -> Path:
    name = architecture or (checkpoint.stem if checkpoint is not None else 'torchlogix_model')
    return Path('build') / 'da4ml' / name


def convert(config: dict[str, Any]) -> dict[str, Any]:
    import da4ml_torch.layers  # noqa: F401
    from da4ml.codegen import RTLModel

    loaded = load_model(config)
    prepare_model_for_export(loaded.model)

    input_shapes = config['input_shape'] or infer_input_shapes(loaded.model)
    if input_shapes is None:
        raise ValueError('Unable to infer input shape. Pass --input-shape, for example --input-shape 1,1,28,28.')

    input_kif = choose_input_kif(config['input_kif'], loaded.model)
    outdir = config['outdir'] or default_outdir(loaded.checkpoint, loaded.architecture)
    outdir = outdir.resolve()
    if outdir.exists() and any(outdir.iterdir()) and not config['overwrite']:
        raise FileExistsError(f'Output directory is not empty: {outdir}. Pass --overwrite to reuse it.')
    outdir.mkdir(parents=True, exist_ok=True)

    symbolic_inputs = make_symbolic_inputs(input_shapes, input_kif, config['hw_config'], config['solver_options'])
    comb = trace_to_comb(loaded.model, symbolic_inputs, config['hw_config'], config['solver_options'])

    metadata = {
        'source_framework': 'torchlogix',
        'source_checkpoint': str(loaded.checkpoint) if loaded.checkpoint else None,
        'checkpoint_kind': loaded.checkpoint_kind,
        'architecture': loaded.architecture,
        'training_config': str(loaded.training_config) if loaded.training_config else None,
        'input_shape': input_shapes,
        'input_kif': input_kif,
        'hw_config': config['hw_config'],
        'unsafe_pickle_load': loaded.unsafe_pickle_load,
    }

    rtl_model = RTLModel(
        comb,
        config['project_name'],
        outdir,
        flavor=config['flavor'],
        latency_cutoff=config['latency_cutoff'],
        print_latency=True,
        clock_uncertainty=config['clock_uncertainty'] / 100,
        clock_period=config['clock_period'],
        part_name=config['part_name'],
    )
    rtl_model.write(metadata)

    validation_mode = _validation_mode(config['validation_input_mode'], loaded.model)
    validation: dict[str, Any] = {}
    if config['validate_comb_samples']:
        summary = validate_comb(
            loaded.model,
            comb,
            input_shapes,
            int(config['validate_comb_samples']),
            input_kif,
            validation_mode,
            config['validation_input_range'],
            config['n_threads'],
            config['validation_atol'],
        )
        validation['comb'] = _serializable(summary.__dict__)
        if not summary.passed:
            raise RuntimeError(f'Combinational validation failed: {summary.mismatches}/{summary.total_outputs} mismatches.')

    if config['compile_rtl'] or config['validate_rtl_samples']:
        rtl_model.compile(nproc=config['n_threads'])

    if config['validate_rtl_samples']:
        summary = validate_rtl(
            rtl_model,
            comb,
            input_shapes,
            int(config['validate_rtl_samples']),
            input_kif,
            validation_mode,
            config['validation_input_range'],
            config['n_threads'],
            config['validation_atol'],
        )
        validation['rtl'] = _serializable(summary.__dict__)
        if not summary.passed:
            raise RuntimeError(f'RTL validation failed: {summary.mismatches}/{summary.total_outputs} mismatches.')

    run_synthesis(config['synthesis_tool'], outdir)
    report = write_reports(outdir, config['report_formats'])

    effective_config = config.copy()
    effective_config['outdir'] = outdir
    effective_config['input_shape'] = input_shapes
    effective_config['input_kif'] = input_kif
    effective_config['validation_input_mode'] = validation_mode

    conversion_summary = {
        'outdir': outdir,
        'project_name': config['project_name'],
        'architecture': loaded.architecture,
        'checkpoint': loaded.checkpoint,
        'training_config': loaded.training_config,
        'input_shape': input_shapes,
        'input_kif': input_kif,
        'comb_shape': comb.shape,
        'comb_cost': comb.cost,
        'comb_latency': comb.latency,
        'validation': validation,
        'report': report,
    }

    analysis_dir = outdir / 'analysis'
    with (analysis_dir / 'conversion_summary.json').open('w') as f:
        json.dump(_serializable(conversion_summary), f, indent=2)
    with (analysis_dir / 'effective_config.json').open('w') as f:
        json.dump(_serializable(effective_config), f, indent=2)

    return conversion_summary


def list_architectures() -> None:
    import torchlogix.models

    names = sorted(
        name for name, value in torchlogix.models.__dict__.items()
        if isinstance(value, type) and issubclass(value, torch.nn.Module)
    )
    for name in names:
        print(name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Convert a TorchLogix PyTorch checkpoint to a DA4ML RTL project.')
    parser.add_argument('--config', type=Path, default=None, help='JSON conversion config. CLI options override config values.')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=None,
        help='TorchLogix checkpoint. Defaults to a unique best-model file under models/.',
    )
    parser.add_argument('--outdir', type=Path, default=None, help='DA4ML project output directory. Default: build/da4ml/<model>.')
    parser.add_argument('--project-name', default=None, help='Generated top-level RTL project/module name.')
    parser.add_argument('--architecture', default=None, help='TorchLogix architecture name, for state_dict checkpoints.')
    parser.add_argument(
        '--training-config',
        default=None,
        help='TorchLogix training_config.json path, "auto", or "none". Default: auto beside checkpoint.',
    )
    parser.add_argument(
        '--model-factory',
        default=None,
        help='Import path for a callable returning torch.nn.Module, e.g. package.module:make_model.',
    )
    parser.add_argument('--model-kwargs', default=None, help='JSON object or path with kwargs for --model-factory/architecture.')
    parser.add_argument(
        '--input-shape',
        default=None,
        help='Symbolic input shape, e.g. 1,1,28,28. Use semicolons for multiple inputs.',
    )
    parser.add_argument(
        '--input-kif',
        type=int,
        nargs=3,
        default=None,
        metavar=('K', 'I', 'F'),
        help='Input precision. Auto: binary or unsigned 1.8 for thresholded inputs.',
    )
    parser.add_argument(
        '--hw-config',
        type=int,
        nargs=3,
        default=None,
        metavar=('ACCUM', 'ADDER', 'CUTOFF'),
        help='DA4ML HWConfig tuple.',
    )
    parser.add_argument('--solver-options', default=None, help='JSON object or path passed to the DA4ML solver.')
    parser.add_argument('--flavor', choices=('verilog', 'vhdl'), default=None, help='RTL flavor.')
    parser.add_argument(
        '--latency-cutoff',
        type=float,
        default=None,
        help='DA4ML latency cutoff for pipelining. Use <=0 for combinational RTL.',
    )
    parser.add_argument('--clock-period', type=float, default=None, help='Target clock period in ns.')
    parser.add_argument('--clock-uncertainty', type=float, default=None, help='Clock uncertainty in percent.')
    parser.add_argument('--part-name', default=None, help='FPGA part name used in generated synthesis scripts.')
    parser.add_argument(
        '--validate-comb-samples',
        type=int,
        default=None,
        help='Random samples for PyTorch vs DA4ML comb validation.',
    )
    parser.add_argument(
        '--validate-rtl-samples',
        type=int,
        default=None,
        help='Random samples for DA4ML comb vs compiled RTL validation.',
    )
    parser.add_argument(
        '--validation-input-mode',
        choices=('auto', 'binary', 'uniform'),
        default=None,
        help='Random validation input mode.',
    )
    parser.add_argument(
        '--validation-input-range',
        type=float,
        nargs=2,
        default=None,
        metavar=('LOW', 'HIGH'),
        help='Uniform validation range.',
    )
    parser.add_argument('--validation-atol', type=float, default=None, help='Absolute tolerance for validation comparisons.')
    parser.add_argument('--n-threads', type=int, default=None, help='Threads for DA4ML simulation and RTL compilation.')
    parser.add_argument('--compile-rtl', action='store_true', default=None, help='Compile the RTL emulator with Verilator/GHDL.')
    parser.add_argument(
        '--synthesis-tool',
        choices=('none', 'vivado', 'quartus'),
        default=None,
        help='Optionally run FPGA synthesis after writing RTL.',
    )
    parser.add_argument(
        '--report-format',
        dest='report_formats',
        action='append',
        choices=('json', 'md'),
        default=None,
        help='Analysis report format. Repeatable.',
    )
    parser.add_argument('--overwrite', action='store_true', default=None, help='Allow writing into a non-empty output directory.')
    parser.add_argument(
        '--list-architectures',
        action='store_true',
        default=None,
        help='List available TorchLogix architecture names and exit.',
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_architectures:
        list_architectures()
        return 0

    config = merge_config(args)
    summary = convert(config)
    print(f'Wrote DA4ML RTL project: {summary["outdir"]}')
    print(f'Wrote analysis reports: {summary["outdir"] / "analysis"}')
    print(f'Comb shape: {summary["comb_shape"][0]} inputs -> {summary["comb_shape"][1]} outputs')
    print(f'Estimated DA4ML cost: {summary["comb_cost"]:.0f} LUTs')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
