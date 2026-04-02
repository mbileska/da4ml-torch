import numpy as np
from da4ml.trace import FixedVariableArray

from ._base import ReplayBase
from torchlogix.layers import GroupSum, LogicConv2d, LogicDense, OrPooling2d


class ReplayLogicConv2d(ReplayBase):
    handles = (LogicConv2d,)

    def call(self, inputs: FixedVariableArray):
        self.module: LogicConv2d
        solver_options = inputs.solver_options
        inputs_np = np.array(inputs)
        result_np = self.module(inputs_np)
        return FixedVariableArray(result_np, solver_options=solver_options)


class ReplayLogicDense(ReplayBase):
    handles = (LogicDense,)

    def call(self, inputs: FixedVariableArray):
        self.module: LogicDense
        solver_options = inputs.solver_options
        inputs_np = np.array(inputs)
        result_np = self.module(inputs_np)
        return FixedVariableArray(result_np, solver_options=solver_options)


def im2col(inp, px_in):
    inp_col = np.lib.stride_tricks.sliding_window_view(  # type: ignore
        inp,  # type: ignore
        px_in,
        axis=tuple(range(len(px_in))),  # type: ignore
    )
    inp_col = np.moveaxis(inp_col, len(px_in), -1).reshape(*inp_col.shape[: len(px_in)], -1)
    return inp_col


class ReplayOrPooling2d(ReplayBase):
    handles = (OrPooling2d,)

    def call(self, inputs: FixedVariableArray):
        self.module: OrPooling2d

        if not isinstance(inputs, FixedVariableArray):
            return FixedVariableArray(self.module(np.array(inputs)))

        kernel_size = self.module.kernel_size
        stride = self.module.stride
        padding = self.module.padding

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        raw = inputs._vars
        if any(padding):
            raw = np.pad(
                raw,
                ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
                mode='constant',
                constant_values=0,
            )

        windows = np.lib.stride_tricks.sliding_window_view(raw, kernel_size, axis=(-2, -1))
        windows = windows[:, :, :: stride[0], :: stride[1], :, :]
        return np.max(FixedVariableArray(windows, solver_options=inputs.solver_options), axis=(-2, -1))


class ReplayGroupSum(ReplayBase):
    handles = (GroupSum,)

    def call(self, inputs: FixedVariableArray):
        self.module: GroupSum
        solver_options = inputs.solver_options
        inputs_np = np.array(inputs)
        result_np = self.module(inputs_np)
        return FixedVariableArray(result_np, solver_options=solver_options)
