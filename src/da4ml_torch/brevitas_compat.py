from __future__ import annotations

from math import ceil, log2
from typing import Any

import torch


def _scalar_float(value: Any, default: float = 1.0) -> float:
    if value is None:
        return default
    if torch.is_tensor(value):
        return float(value.detach().cpu().reshape(()))
    return float(value)


def fixed_point_kif(bit_width: int | None, signed: bool, scale_value: Any = None) -> tuple[int, int, int]:
    if bit_width is None:
        return (1 if signed else 0, 1, 8)

    k = 1 if signed else 0
    scale = abs(_scalar_float(scale_value, 1.0))
    if scale <= 0:
        i = 0
    else:
        i = max(0, int(ceil(log2(scale + 1e-12))))
    f = max(0, int(bit_width) - k - i)
    return k, i, f


def quantize_tensor(
    x: torch.Tensor,
    bit_width: int | None,
    signed: bool,
    scale_value: Any = None,
    round_mode: str = 'TRN',
) -> torch.Tensor:
    if bit_width is None:
        return x

    k, i, f = fixed_point_kif(bit_width, signed, scale_value)
    if k + i + f <= 0:
        return torch.zeros_like(x)

    step = 2.0**-f
    if round_mode.upper() == 'RND':
        q = torch.floor(x / step + 0.5)
    else:
        q = torch.floor(x / step)

    levels = float(2 ** (k + i + f))
    bias = float(k * 2 ** (i + f))
    return (torch.remainder(q + bias, levels) - bias) * step


class _ScalingImpl(torch.nn.Module):
    def __init__(self, value: float = 1.0):
        super().__init__()
        self.register_buffer('value', torch.tensor(float(value), dtype=torch.float32))


class _TensorQuant(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scaling_impl = _ScalingImpl()


class _FusedActivationQuantProxy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tensor_quant = _TensorQuant()


class _ActQuant(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fused_activation_quant_proxy = _FusedActivationQuantProxy()


class QuantIdentity(torch.nn.Module):
    def __init__(self, bit_width: int | None = None, signed: bool = True, **_: Any):
        super().__init__()
        self.bit_width = bit_width
        self.signed = signed
        self.act_quant = _ActQuant()

    @property
    def scale_value(self) -> torch.Tensor:
        return self.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return quantize_tensor(x, self.bit_width, self.signed, self.scale_value)

    def symbolic_input_kif(self) -> tuple[int, int, int]:
        return fixed_point_kif(self.bit_width, self.signed, self.scale_value)


class QuantReLU(torch.nn.Module):
    def __init__(self, bit_width: int | None = None, signed: bool = False, **_: Any):
        super().__init__()
        self.bit_width = bit_width
        self.signed = signed
        self.act_quant = _ActQuant()

    @property
    def scale_value(self) -> torch.Tensor:
        return self.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return quantize_tensor(torch.relu(x), self.bit_width, self.signed, self.scale_value)


class QuantLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, weight_bit_width: int | None = None, **_: Any):
        super().__init__(in_features, out_features, bias=bias)
        self.weight_bit_width = weight_bit_width

    def quantized_weight(self) -> torch.Tensor:
        scale = torch.max(torch.abs(self.weight.detach())) if self.weight.numel() else torch.tensor(1.0, device=self.weight.device)
        return quantize_tensor(self.weight, self.weight_bit_width, True, scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.quantized_weight(), self.bias)


class QuantConv2d(torch.nn.Conv2d):
    def __init__(self, *args: Any, weight_bit_width: int | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.weight_bit_width = weight_bit_width

    def quantized_weight(self) -> torch.Tensor:
        scale = torch.max(torch.abs(self.weight.detach())) if self.weight.numel() else torch.tensor(1.0, device=self.weight.device)
        return quantize_tensor(self.weight, self.weight_bit_width, True, scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv2d(
            x,
            self.quantized_weight(),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class QuantizedTowardsModel2(torch.nn.Module):
    n_input_features = 14

    def __init__(self, weight_bit_width: int, act_bit_width: int):
        super().__init__()
        self.quant_inp = QuantIdentity(bit_width=act_bit_width)
        self.fc1 = QuantLinear(14, 128, weight_bit_width=weight_bit_width)
        self.relu1 = QuantReLU(bit_width=act_bit_width)
        self.fc2 = QuantLinear(128, 3, weight_bit_width=weight_bit_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant_inp(x)
        x = self.relu1(self.fc1(x))
        return self.fc2(x)


def make_quantized_towards_model2(model_name: str) -> QuantizedTowardsModel2:
    configs = {
        'towards-model-2-brevitas-w5a10': (5, 10),
        'towards-model-2-brevitas-w4a8': (4, 8),
        'towards-model-2-brevitas-w2a6': (2, 6),
        'towards-model-2-brevitas-w2a4': (2, 4),
    }
    try:
        weight_bit_width, act_bit_width = configs[model_name]
    except KeyError as exc:
        raise ValueError(f'Unsupported Brevitas registry model {model_name!r}.') from exc
    return QuantizedTowardsModel2(weight_bit_width=weight_bit_width, act_bit_width=act_bit_width)
