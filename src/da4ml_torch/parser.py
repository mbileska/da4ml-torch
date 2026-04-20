import numpy as np
import torch
import torch.nn.functional as F
from da4ml.converter.plugin import DAISTracerPluginBase, _flatten_arr
from da4ml.trace import FixedVariableArray
from torch.fx import Node, Tracer

from .layers import _registered_modules

SUPPORTED_METHODS = {
    'apply',
    'as_new', 
    'flatten', 
    'from_kif', 
    'from_lhs', 
    'matmul', 
    'quantize', 
    'ravel', 
    'relu', 
    'reshape', 
    'rmatmul', 
    'size',
    'to_bool', 
    'transpose',
    'view',
}

SUPPORTED_FUNCTIONS = {
    'relu',
    'reshape',
    'flatten',
    'matmul',
    'cat',
    'concat',
    'concatenate',
}

OPERATOR_MAP = {
    "mul": "__mul__",
    "add": "__add__",
    "sub": "__sub__",
    "and_": "__and__",
    "or_": "__or__",
}

PASSTHROUGH_FUNCTIONS = {
    "getattr",
}


def _resolve_fx_value(value, env):
    if isinstance(value, Node):
        return env[value.name]
    if isinstance(value, tuple):
        return tuple(_resolve_fx_value(v, env) for v in value)
    if isinstance(value, list):
        return [_resolve_fx_value(v, env) for v in value]
    if isinstance(value, dict):
        return {k: _resolve_fx_value(v, env) for k, v in value.items()}
    if isinstance(value, slice):
        return slice(
            _resolve_fx_value(value.start, env),
            _resolve_fx_value(value.stop, env),
            _resolve_fx_value(value.step, env),
        )
    return value


def _find_fixed_variable_array(value):
    if isinstance(value, FixedVariableArray):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _find_fixed_variable_array(item)
            if found is not None:
                return found
        return None
    if isinstance(value, dict):
        for item in value.values():
            found = _find_fixed_variable_array(item)
            if found is not None:
                return found
        return None
    return None


class DATracer(Tracer):
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str):
        if type(m) in _registered_modules:
            return True
        return super().is_leaf_module(m, module_qualified_name)


class TorchParser(DAISTracerPluginBase):
    def trace(
        self,
        verbose: bool = False,
        inputs: tuple[FixedVariableArray, ...] | FixedVariableArray | None = None,
        inputs_kif: tuple[int, int, int] | None = None,
        dump: bool = False,
    ):
        assert inputs is not None
        if isinstance(inputs, FixedVariableArray):
            inputs = (inputs,)
        self.model: torch.nn.Module
        tracer = DATracer()
        graph = tracer.trace(self.model)
        modules = dict(self.model.named_modules())
        env: dict[str, object] = {}
        inp_nodes = [n for n in graph.nodes if n.op == 'placeholder']
        out_nodes = [n for n in graph.nodes if n.op == 'output']
        assert len(out_nodes) == 1, f'only one output node is supported, but found {len(out_nodes)}'
        assert len(inputs) == len(inp_nodes), (
            f'inputs length {len(inputs)} does not match with graph input length {len(inp_nodes)}'
        )
        for node, inp in zip(inp_nodes, inputs):
            env[node.name] = inp

        for node in graph.nodes:
            args: tuple[Node, ...] = node.args  # type: ignore
            kwargs: dict[str, Node] = node.kwargs  # type: ignore
            target: str = node.target  # type: ignore
            match node.op:
                case 'call_module':
                    module = modules[target]
                    assert type(module) in _registered_modules, f'{type(module)} is not supported'
                    replay_cls = _registered_modules[type(module)]
                    replay = replay_cls(module)
                    _args = _resolve_fx_value(args, env)
                    _lwargs = _resolve_fx_value(kwargs, env)
                    env[node.name] = replay(*_args, **_lwargs)
                case 'call_function':
                    _args = _resolve_fx_value(args, env)
                    _kwargs = _resolve_fx_value(kwargs, env)
                    first_fva_arg = _find_fixed_variable_array(_args)
                    op_name = target.__name__
                    if op_name in PASSTHROUGH_FUNCTIONS:
                        env[node.name] = target(*_args, **_kwargs)
                        continue
                    if op_name == 'getitem':
                        env[node.name] = _args[0][_args[1]]
                        continue
                    if op_name in {'cat', 'concat', 'concatenate'}:
                        tensors = _args[0]
                        axis = _kwargs.pop('dim', None)
                        if axis is None:
                            axis = _kwargs.pop('axis', 0)
                        env[node.name] = np.concatenate(tensors, axis=axis, **_kwargs)
                        continue
                    if op_name in OPERATOR_MAP:
                        if first_fva_arg is None:
                            env[node.name] = target(*_args, **_kwargs)
                            continue
                        method_name = OPERATOR_MAP[op_name]
                        if not hasattr(first_fva_arg, method_name):
                            raise NotImplementedError(
                                f"Operator '{op_name}' not implemented for FixedVariableArray"
                            )
                        method = getattr(first_fva_arg, method_name)
                        env[node.name] = method(_args[1])
                        continue
                    if first_fva_arg is not None:
                        if op_name not in SUPPORTED_FUNCTIONS:
                            raise NotImplementedError(
                                f"Function '{op_name}' not supported"
                            )
                        if not hasattr(first_fva_arg, op_name):
                            raise NotImplementedError(
                                f"Function '{op_name}' declared supported but not implemented"
                            )
                        method = getattr(first_fva_arg, op_name)
                        env[node.name] = method(*_args[1:], **_kwargs)
                    else:
                        env[node.name] = target(*_args, **_kwargs)
                case 'call_method':
                    _args = _resolve_fx_value(args, env)
                    _kwargs = _resolve_fx_value(kwargs, env)
                    obj = _args[0]
                    if isinstance(obj, FixedVariableArray):
                        if target not in SUPPORTED_METHODS:
                            raise NotImplementedError(
                                f"Method '{target}' is not supported for FixedVariableArray"
                            )
                        if target == "view":
                            env[node.name] = obj.reshape(*_args[1:], **_kwargs)
                            continue
                        if target == "size":
                            env[node.name] = obj.shape[_args[1]] if len(_args) == 2 else obj.shape
                            continue
                        if not hasattr(obj, target):
                            raise NotImplementedError(
                                f"Method '{target}' declared supported but not implemented"
                            )
                        if target == "transpose": # handle PyTorch transpose API difference
                            if len(_args) == 3:
                                dim0, dim1 = _args[1], _args[2]
                                axes = list(range(obj.ndim))
                                axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
                                env[node.name] = obj.transpose(tuple(axes))
                                continue
                            env[node.name] = obj.transpose(*_args[1:], **_kwargs)
                            continue
                    method = getattr(obj, target)
                    env[node.name] = method(*_args[1:], **_kwargs)
                case 'get_attr':
                    attr = getattr(self.model, target)
                    if isinstance(attr, torch.Tensor):
                        attr = attr.detach().cpu().numpy()
                    env[node.name] = attr
                case 'placeholder':
                    pass
                case 'output':
                    pass
                case _:
                    raise NotImplementedError(f'unknown node op: {node.op}')

        inp_tensors = tuple(env[n.name] for n in inp_nodes)
        out_tensors = tuple(env[str(out_name)] for out_name in out_nodes[0].args)
        if not dump:
            return _flatten_arr(inp_tensors), _flatten_arr(out_tensors)
        return env
