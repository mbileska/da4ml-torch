from __future__ import annotations

import numpy as np
import torch
from da4ml.trace import FixedVariableArray
from da4ml.trace.fixed_variable_array import mmm

from da4ml_torch.brevitas_compat import QuantIdentity, QuantLinear, QuantReLU, fixed_point_kif

from ._base import ReplayBase

_ACTUAL_BREVITAS_HANDLES: tuple[type, ...] = ()
try:
    import brevitas.nn as _brevitas_nn

    _ACTUAL_BREVITAS_HANDLES = (
        _brevitas_nn.QuantIdentity,
        _brevitas_nn.QuantLinear,
        _brevitas_nn.QuantReLU,
    )
except Exception:
    pass


def _to_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _get_nested_attr(obj, path: str, default=None):
    cur = obj
    for part in path.split('.'):
        if not hasattr(cur, part):
            return default
        cur = getattr(cur, part)
    return cur


def _activation_scale(module) -> float:
    value = _get_nested_attr(module, 'act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value', None)
    if value is None:
        return 1.0
    if torch.is_tensor(value):
        return float(value.detach().cpu().reshape(()))
    return float(value)


def _module_bit_width(module, attr_name: str = 'bit_width') -> int | None:
    value = getattr(module, attr_name, None)
    if callable(value):
        value = value()
    if torch.is_tensor(value):
        value = int(value.detach().cpu().reshape(()))
    return None if value is None else int(value)


def _quantize_fixed_array(
    inputs: FixedVariableArray,
    bit_width: int | None,
    signed: bool,
    scale_value: float,
    relu: bool = False,
) -> FixedVariableArray:
    k, i, f = fixed_point_kif(bit_width, signed, scale_value)
    if relu:
        return inputs.relu(i=i, f=f)
    return inputs.quantize(k=k, i=i, f=f)


class ReplayQuantIdentity(ReplayBase):
    handles = (QuantIdentity,) + tuple(
        handle
        for handle in _ACTUAL_BREVITAS_HANDLES
        if getattr(handle, '__name__', '') == 'QuantIdentity'
    )

    def call(self, inputs: FixedVariableArray):
        bit_width = _module_bit_width(self.module)
        signed = bool(getattr(self.module, 'signed', True))
        return _quantize_fixed_array(inputs, bit_width, signed, _activation_scale(self.module))


class ReplayQuantReLU(ReplayBase):
    handles = (QuantReLU,) + tuple(
        handle
        for handle in _ACTUAL_BREVITAS_HANDLES
        if getattr(handle, '__name__', '') == 'QuantReLU'
    )

    def call(self, inputs: FixedVariableArray):
        bit_width = _module_bit_width(self.module)
        signed = bool(getattr(self.module, 'signed', False))
        return _quantize_fixed_array(inputs, bit_width, signed, _activation_scale(self.module), relu=True)


class ReplayQuantLinear(ReplayBase):
    handles = (QuantLinear,) + tuple(
        handle
        for handle in _ACTUAL_BREVITAS_HANDLES
        if getattr(handle, '__name__', '') == 'QuantLinear'
    )

    def call(self, inputs: FixedVariableArray):
        weight = getattr(self.module, 'weight')
        if hasattr(self.module, 'quantized_weight'):
            weight = self.module.quantized_weight()
        else:
            bit_width = _module_bit_width(self.module, 'weight_bit_width')
            if bit_width is not None:
                weight_np = _to_numpy(weight)
                scale = float(np.max(np.abs(weight_np))) if weight_np.size else 1.0
                k, i, f = fixed_point_kif(bit_width, True, scale)
                step = 2.0**-f
                low = -float(2**i)
                high = float(2**i) - step
                weight_np = np.clip(np.floor(weight_np / step + 0.5) * step, low, high)
                weight = weight_np

        weight_np = _to_numpy(weight).astype(np.float32)
        out = FixedVariableArray(mmm(inputs._vars, weight_np.T), inputs.solver_options)
        bias = getattr(self.module, 'bias', None)
        if bias is not None:
            out = out + _to_numpy(bias).astype(np.float32)
        return out
