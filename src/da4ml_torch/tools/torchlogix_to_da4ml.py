from __future__ import annotations

import argparse
import importlib
import json
import pickle
import re
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
    'model': None,
    'model_name': None,
    'models_root': 'models',
    'checkpoint': None,
    'outdir': None,
    'project_name': None,
    'architecture': None,
    'training_config': 'auto',
    'model_factory': None,
    'model_kwargs': {},
    'sequential_fallback': 'auto',
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

CHECKPOINT_SUFFIXES = ('.pt', '.pth', '.ckpt', '.pkl', '.pickle')


@dataclass(frozen=True)
class LoadedModel:
    model: torch.nn.Module
    model_name: str | None
    architecture: str | None
    checkpoint: Path | None
    checkpoint_kind: str
    training_config: Path | None
    unsafe_pickle_load: bool
    load_notes: tuple[str, ...] = ()


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


def _path_value(value: str | Path) -> Path:
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
    config['models_root'] = _path_value(config.get('models_root', 'models'))
    training_config = config.get('training_config')
    if training_config != 'auto':
        config['training_config'] = _path_or_none(training_config)

    config['model_kwargs'] = _json_value(config.get('model_kwargs'))
    config['solver_options'] = _json_value(config.get('solver_options'))
    config['input_shape'] = parse_input_shapes(config.get('input_shape'))
    config['input_kif'] = tuple(config['input_kif']) if config.get('input_kif') is not None else None
    config['hw_config'] = tuple(config['hw_config'])
    config['validation_input_range'] = tuple(config['validation_input_range'])
    if config.get('sequential_fallback') not in ('auto', 'never'):
        raise ValueError('sequential_fallback must be "auto" or "never".')
    if isinstance(config.get('report_formats'), str):
        config['report_formats'] = [config['report_formats']]
    return config


def _unique_checkpoint(matches: Sequence[Path], requested: str) -> Path | None:
    unique_matches = sorted(set(path.resolve() for path in matches if path.is_file()))
    if not unique_matches:
        return None
    if len(unique_matches) > 1:
        formatted = '\n'.join(f'  - {path}' for path in unique_matches)
        raise RuntimeError(f'Multiple candidate checkpoints found for {requested!r}. Pass --checkpoint explicitly:\n{formatted}')
    return unique_matches[0]


def find_default_checkpoint(models_root: Path = Path('models')) -> Path | None:
    if not models_root.exists():
        return None
    matches: list[Path] = []
    for pattern in CHECKPOINT_PATTERNS:
        matches.extend(models_root.rglob(pattern))
    return _unique_checkpoint(matches, 'default model')


def _checkpoints_in_dir(path: Path) -> list[Path]:
    matches: list[Path] = []
    for pattern in CHECKPOINT_PATTERNS:
        matches.extend(path.glob(pattern))
    if not matches:
        matches.extend(item for item in path.iterdir() if item.is_file() and item.suffix in CHECKPOINT_SUFFIXES)
    return matches


def find_named_checkpoint(model_ref: str | Path, models_root: Path = Path('models')) -> Path | None:
    ref = Path(model_ref).expanduser()
    if ref.exists():
        if ref.is_file():
            return ref.resolve()
        return _unique_checkpoint(_checkpoints_in_dir(ref), str(model_ref))

    rooted_ref = models_root / ref
    if rooted_ref.exists():
        if rooted_ref.is_file():
            return rooted_ref.resolve()
        return _unique_checkpoint(_checkpoints_in_dir(rooted_ref), str(model_ref))

    matches: list[Path] = []
    if ref.suffix:
        matches.extend(models_root.rglob(ref.name))
    else:
        for suffix in CHECKPOINT_SUFFIXES:
            matches.extend(models_root.rglob(f'{ref.name}{suffix}'))
        for directory in models_root.rglob(ref.name):
            if directory.is_dir():
                matches.extend(_checkpoints_in_dir(directory))
    return _unique_checkpoint(matches, str(model_ref))


class AutoSequentialCheckpointModule(torch.nn.Module):
    """Fallback top-level module for simple checkpoints whose model class is unavailable."""

    def forward(self, x: Any) -> Any:
        value: Any = x
        for child in self.children():
            value = _call_sequential_child(child, value)
        return value


def _call_sequential_child(child: torch.nn.Module, value: Any) -> Any:
    if isinstance(child, torch.nn.ModuleList):
        for module in child:
            value = _call_sequential_child(module, value)
        return value
    if isinstance(child, torch.nn.ModuleDict):
        for module in child.values():
            value = _call_sequential_child(module, value)
        return value
    if isinstance(value, tuple):
        return child(*value)
    return child(value)


_AUTO_SEQUENTIAL_CLASS_CACHE: dict[tuple[str, str], type[AutoSequentialCheckpointModule]] = {}


def _auto_sequential_class(module_name: str, class_name: str) -> type[AutoSequentialCheckpointModule]:
    key = (module_name, class_name)
    if key not in _AUTO_SEQUENTIAL_CLASS_CACHE:
        _AUTO_SEQUENTIAL_CLASS_CACHE[key] = type(
            class_name,
            (AutoSequentialCheckpointModule,),
            {'__module__': module_name},
        )
    return _AUTO_SEQUENTIAL_CLASS_CACHE[key]


class _AutoSequentialUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        try:
            return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError):
            return _auto_sequential_class(module, name)


class _AutoSequentialPickleModule:
    Unpickler = _AutoSequentialUnpickler
    Pickler = pickle.Pickler
    dump = staticmethod(pickle.dump)
    dumps = staticmethod(pickle.dumps)
    load = staticmethod(pickle.load)
    loads = staticmethod(pickle.loads)


def _torch_load_with_auto_sequential(path: Path) -> Any:
    return torch.load(path, map_location='cpu', weights_only=False, pickle_module=_AutoSequentialPickleModule)


def _torch_load(path: Path, sequential_fallback: str = 'auto') -> tuple[Any, bool, tuple[str, ...]]:
    for candidate in (Path.cwd(), path.resolve().parent):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
    try:
        return torch.load(path, map_location='cpu', weights_only=True), False, ()
    except TypeError as weights_arg_error:
        try:
            return torch.load(path, map_location='cpu'), True, ()
        except Exception as full_load_error:
            if sequential_fallback == 'never':
                raise full_load_error from weights_arg_error
            try:
                payload = _torch_load_with_auto_sequential(path)
            except Exception as fallback_error:
                raise RuntimeError(
                    'Unable to load checkpoint as a full PyTorch module or an auto-sequential fallback. '
                    'If this checkpoint is a state_dict, pass --architecture or --model-factory. If it is a custom '
                    'module with branching or multiple inputs, pass --model-factory so the original forward method is available.'
                ) from fallback_error
            return payload, True, (
                'Loaded with auto-sequential checkpoint fallback because the original top-level model class was unavailable. '
                'This is valid for single-input child-module chains; use --model-factory for custom branching forward methods.',
            )
    except Exception as weights_only_error:
        try:
            return torch.load(path, map_location='cpu', weights_only=False), True, ()
        except Exception as full_load_error:
            if sequential_fallback == 'never':
                raise full_load_error from weights_only_error
            try:
                payload = _torch_load_with_auto_sequential(path)
            except Exception as fallback_error:
                raise RuntimeError(
                    'Unable to load checkpoint as weights_only, a full PyTorch module, or an auto-sequential fallback. '
                    'If this checkpoint is a state_dict, pass --architecture or --model-factory. If it is a custom '
                    'module with branching or multiple inputs, pass --model-factory so the original forward method is available.'
                ) from fallback_error
            return payload, True, (
                'Loaded with auto-sequential checkpoint fallback because the original top-level model class was unavailable. '
                'This is valid for single-input child-module chains; use --model-factory for custom branching forward methods.',
            )


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


def _name_from_model_ref(model_ref: str | Path | None) -> str | None:
    if model_ref in (None, ''):
        return None
    ref = Path(model_ref)
    return ref.stem if ref.suffix else ref.name


def _sanitize_project_name(name: str) -> str:
    sanitized = re.sub(r'\W+', '_', name).strip('_')
    if not sanitized:
        sanitized = 'model'
    if sanitized[0].isdigit():
        sanitized = f'model_{sanitized}'
    return sanitized


def _resolve_model_name(config: dict[str, Any], checkpoint: Path | None, architecture: str | None) -> str | None:
    configured = config.get('model_name')
    if configured:
        return str(configured)
    ref_name = _name_from_model_ref(config.get('model'))
    if ref_name:
        return ref_name
    if architecture:
        return architecture
    if checkpoint is not None:
        return checkpoint.stem
    if config.get('model_factory'):
        return config['model_factory'].replace(':', '.').rsplit('.', 1)[-1]
    return None


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
    checkpoint = config['checkpoint']
    if checkpoint is None and config.get('model'):
        checkpoint = find_named_checkpoint(config['model'], config['models_root'])
        if checkpoint is None and not config.get('model_factory'):
            raise FileNotFoundError(
                f'No checkpoint found for model {config["model"]!r} under {config["models_root"]}. '
                'Pass --checkpoint or --model-factory.'
            )
    if checkpoint is None:
        checkpoint = find_default_checkpoint(config['models_root'])
    payload: Any = None
    unsafe_pickle_load = False
    load_notes: tuple[str, ...] = ()
    state_dict: dict[str, torch.Tensor] | None = None
    checkpoint_kind = 'factory'
    model_name = _resolve_model_name(config, checkpoint, config.get('architecture'))

    if checkpoint is not None:
        checkpoint = checkpoint.resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
        payload, unsafe_pickle_load, load_notes = _torch_load(checkpoint, config['sequential_fallback'])
        if isinstance(payload, torch.nn.Module):
            model = payload
            kind = 'full_module_auto_sequential' if isinstance(model, AutoSequentialCheckpointModule) else 'full_module'
            return LoadedModel(model, model_name, None, checkpoint, kind, None, unsafe_pickle_load, load_notes)
        state_dict, checkpoint_kind = _extract_state_dict(payload)

    if config.get('model_factory'):
        factory = _resolve_object(config['model_factory'])
        model = factory(**config['model_kwargs'])
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f'Model factory {config["model_factory"]!r} returned {type(model)}, not torch.nn.Module.')
        if state_dict is not None:
            model.load_state_dict(_clean_state_dict(state_dict))
        return LoadedModel(model, model_name, config.get('architecture'), checkpoint, checkpoint_kind, None, unsafe_pickle_load, load_notes)

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
    model_name = _resolve_model_name(config, checkpoint, architecture)
    return LoadedModel(model, model_name, architecture, checkpoint, checkpoint_kind, training_config_path, unsafe_pickle_load, load_notes)


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


def make_vivado_script_compatible(outdir: Path) -> tuple[Path, ...]:
    patched: list[Path] = []
    for script_name in ('build_vivado_prj.tcl', 'build_vivado_hls_prj.tcl'):
        script = outdir / script_name
        if not script.exists():
            continue
        text = script.read_text()
        updated = text.replace(' -global_retiming on', '')
        if updated != text:
            script.write_text(updated)
            patched.append(script)
    return tuple(patched)


def da4ml_project_report(outdir: Path) -> dict[str, Any]:
    from da4ml._cli.report import load_project

    report = load_project(outdir)
    if report is None:
        raise RuntimeError(f'Unable to load DA4ML report from {outdir}.')
    return report


def enrich_report_estimates(report: dict[str, Any]) -> dict[str, Any]:
    report = report.copy()
    if 'cost' in report:
        report.setdefault('rough_LUT_estimate', report['cost'])
    if 'clock_period' in report:
        clock_period = float(report['clock_period'])
        if clock_period > 0:
            report.setdefault('target_Fmax(MHz)', 1000.0 / clock_period)
    period = report.get('actual_period', report.get('clock_period'))
    latency = report.get('latency')
    if latency is not None and period is not None:
        report.setdefault('target_latency(ns)', float(latency) * float(report.get('clock_period', period)))
        if 'actual_period' in report:
            report.setdefault('synth_latency(ns)', float(latency) * float(period))
    if 'actual_period' in report:
        report.setdefault('timing_estimate_source', 'synthesis_report')
    elif 'clock_period' in report:
        report.setdefault('timing_estimate_source', 'target_clock_metadata')
    else:
        report.setdefault('timing_estimate_source', 'logic_metadata')
    return report


def _write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    preferred = [
        'flavor',
        'part_name',
        'cost',
        'rough_LUT_estimate',
        'comb_latency',
        'latency',
        'latency_cutoff',
        'clock_period',
        'target_Fmax(MHz)',
        'target_latency(ns)',
        'actual_period',
        'Fmax(MHz)',
        'synth_latency(ns)',
        'latency(ns)',
        'timing_estimate_source',
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
    report = enrich_report_estimates(da4ml_project_report(outdir))
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


def default_outdir(checkpoint: Path | None, architecture: str | None, model_name: str | None) -> Path:
    name = model_name or architecture or (checkpoint.stem if checkpoint is not None else 'torch_model')
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
    model_name = loaded.model_name or _resolve_model_name(config, loaded.checkpoint, loaded.architecture) or 'torch_model'
    project_name = config['project_name'] or _sanitize_project_name(model_name)
    outdir = config['outdir'] or default_outdir(loaded.checkpoint, loaded.architecture, model_name)
    outdir = outdir.resolve()
    if outdir.exists() and any(outdir.iterdir()) and not config['overwrite']:
        raise FileExistsError(f'Output directory is not empty: {outdir}. Pass --overwrite to reuse it.')
    outdir.mkdir(parents=True, exist_ok=True)

    symbolic_inputs = make_symbolic_inputs(input_shapes, input_kif, config['hw_config'], config['solver_options'])
    comb = trace_to_comb(loaded.model, symbolic_inputs, config['hw_config'], config['solver_options'])

    metadata = {
        'source_framework': 'torch',
        'model_name': model_name,
        'source_checkpoint': str(loaded.checkpoint) if loaded.checkpoint else None,
        'checkpoint_kind': loaded.checkpoint_kind,
        'architecture': loaded.architecture,
        'training_config': str(loaded.training_config) if loaded.training_config else None,
        'input_shape': input_shapes,
        'input_kif': input_kif,
        'hw_config': config['hw_config'],
        'unsafe_pickle_load': loaded.unsafe_pickle_load,
        'load_notes': loaded.load_notes,
        'comb_shape': comb.shape,
        'comb_cost': comb.cost,
        'comb_latency': comb.latency,
    }

    rtl_model = RTLModel(
        comb,
        project_name,
        outdir,
        flavor=config['flavor'],
        latency_cutoff=config['latency_cutoff'],
        print_latency=True,
        clock_uncertainty=config['clock_uncertainty'] / 100,
        clock_period=config['clock_period'],
        part_name=config['part_name'],
    )
    rtl_model.write(metadata)
    make_vivado_script_compatible(outdir)

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
        'project_name': project_name,
        'model_name': model_name,
        'architecture': loaded.architecture,
        'checkpoint': loaded.checkpoint,
        'checkpoint_kind': loaded.checkpoint_kind,
        'training_config': loaded.training_config,
        'input_shape': input_shapes,
        'input_kif': input_kif,
        'comb_shape': comb.shape,
        'comb_cost': comb.cost,
        'comb_latency': comb.latency,
        'validation': validation,
        'report': report,
        'load_notes': loaded.load_notes,
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
    parser = argparse.ArgumentParser(description='Convert a supported PyTorch/TorchLogix model to a DA4ML RTL project.')
    parser.add_argument('--config', type=Path, default=None, help='JSON conversion config. CLI options override config values.')
    parser.add_argument(
        '--model',
        default=None,
        help='Model name under --models-root or a checkpoint path. This is the usual one-argument model selector.',
    )
    parser.add_argument('--model-name', default=None, help='Human-readable model name used for default output naming.')
    parser.add_argument('--models-root', type=Path, default=None, help='Directory searched by --model. Default: models/.')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=None,
        help='PyTorch/TorchLogix checkpoint. Defaults to --model lookup or a unique best-model file under --models-root.',
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
        '--sequential-fallback',
        choices=('auto', 'never'),
        default=None,
        help='When a full-module checkpoint references a missing top-level class, auto-load it as an ordered child-module chain.',
    )
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


def _estimate_lines(report: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    source = report.get('timing_estimate_source', 'logic_metadata')
    if 'rough_LUT_estimate' in report:
        lines.append(f'Rough LUT estimate: {float(report["rough_LUT_estimate"]):.0f}')
    if 'latency' in report:
        lines.append(f'Pipeline latency: {report["latency"]} cycles')
    if 'target_Fmax(MHz)' in report:
        lines.append(f'Target Fmax: {float(report["target_Fmax(MHz)"]):.2f} MHz')
    if 'target_latency(ns)' in report:
        lines.append(f'Target latency: {float(report["target_latency(ns)"]):.2f} ns')
    if 'Fmax(MHz)' in report:
        lines.append(f'Synthesis Fmax: {float(report["Fmax(MHz)"]):.2f} MHz')
    if 'synth_latency(ns)' in report:
        lines.append(f'Synthesis latency: {float(report["synth_latency(ns)"]):.2f} ns')
    if any(key in report for key in ('LUT', 'FF', 'DSP', 'RAMB18', 'Block RAM Tile')):
        resource_parts = [f'{key}={report[key]}' for key in ('LUT', 'FF', 'DSP', 'RAMB18', 'Block RAM Tile') if key in report]
        lines.append('Synthesis resources: ' + ', '.join(resource_parts))
    lines.append(f'Estimate source: {source}')
    return lines


def print_estimate(report: dict[str, Any]) -> None:
    for line in _estimate_lines(report):
        print(line)


def build_estimate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Print rough timing/resource estimates from DA4ML RTL project directories.')
    parser.add_argument('paths', type=Path, nargs='+', help='DA4ML RTL project directories.')
    parser.add_argument('--format', choices=('text', 'json', 'md'), default='text', help='Output format.')
    parser.add_argument('--output', '-o', type=Path, default=None, help='Optional output file. Defaults to stdout.')
    return parser


def _format_estimate_output(paths: Sequence[Path], reports: Sequence[dict[str, Any]], fmt: str) -> str:
    if fmt == 'json':
        payload = [
            {
                'path': str(path),
                'report': _serializable(report),
            }
            for path, report in zip(paths, reports)
        ]
        return json.dumps(payload[0] if len(payload) == 1 else payload, indent=2)
    if fmt == 'md':
        rows = ['| project | metric | value |', '|---|---|---|']
        for path, report in zip(paths, reports):
            for line in _estimate_lines(report):
                metric, _, value = line.partition(': ')
                rows.append(f'| {path} | {metric} | {value} |')
        return '\n'.join(rows)

    chunks: list[str] = []
    for path, report in zip(paths, reports):
        if len(paths) > 1:
            chunks.append(f'{path}:')
        chunks.extend(_estimate_lines(report))
    return '\n'.join(chunks)


def estimate_main(argv: Sequence[str] | None = None) -> int:
    parser = build_estimate_parser()
    args = parser.parse_args(argv)
    reports = [enrich_report_estimates(da4ml_project_report(path)) for path in args.paths]
    output = _format_estimate_output(args.paths, reports, args.format)
    if args.output is None:
        print(output)
    else:
        args.output.write_text(output + '\n')
    return 0


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
    print_estimate(summary['report'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
