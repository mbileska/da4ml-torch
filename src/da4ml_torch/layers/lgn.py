import numpy as np
from da4ml.trace import FixedVariableArray
from torchlogix.layers import GroupSum, LogicConv2d, LogicDense, OrPooling2d

from ._base import ReplayBase

_LUT_OPERATORS = [
    lambda a, b: a & ~a,
    lambda a, b: a & b,
    lambda a, b: a & ~b,
    lambda a, b: a,
    lambda a, b: b & ~a,
    lambda a, b: b,
    lambda a, b: a ^ b,
    lambda a, b: a | b,
    lambda a, b: ~(a | b),
    lambda a, b: ~(a ^ b),
    lambda a, b: ~b,
    lambda a, b: ~(b & ~a),
    lambda a, b: ~a,
    lambda a, b: ~(a & ~b),
    lambda a, b: ~(a & b),
    lambda a, b: a | ~a,
]


def apply_lut_vectorized(a: FixedVariableArray, b: FixedVariableArray, lut_ids: np.ndarray):
    solver_options = a.solver_options
    a_arr, b_arr = np.array(a), np.array(b)
    result = np.empty_like(a_arr)
    for lut_id, op in enumerate(_LUT_OPERATORS):
        mask = lut_ids == lut_id
        result[..., mask] = op(a_arr[..., mask], b_arr[..., mask])
    return FixedVariableArray(result, solver_options=solver_options)


def _as_numpy_int(tensor) -> np.ndarray:
    if hasattr(tensor, 'detach'):
        tensor = tensor.detach().cpu().numpy()
    return np.asarray(tensor, dtype=np.int64)


def _dense_connection_indices(module: LogicDense) -> np.ndarray:
    connections = module.connections
    indices = connections.indices
    if not hasattr(connections, 'weights'):
        return _as_numpy_int(indices)

    choice = connections.weights.detach().cpu().argmax(dim=0)
    lut_axis = np.arange(connections.lut_rank)[:, None]
    out_axis = np.arange(connections.out_dim)[None, :]
    return _as_numpy_int(indices)[choice.numpy(), lut_axis, out_axis]


def _conv_lut_ids(module: LogicConv2d, level: int) -> np.ndarray:
    _, tree_ids = module.get_luts_and_ids()
    ids = [_as_numpy_int(level_ids) for level_ids in tree_ids[level]]
    return np.stack(ids, axis=0).T.flatten()


class ReplayLogicConv2d(ReplayBase):
    handles = (LogicConv2d,)

    def call(self, inputs: FixedVariableArray):
        self.module: LogicConv2d
        if self.module.lut_rank != 2:
            raise NotImplementedError('DA4ML TorchLogix conversion currently supports LogicConv2d with lut_rank=2.')

        if self.module.padding > 0:
            inputs = np.pad(
                inputs,
                ((0, 0), (0, 0), (self.module.padding, self.module.padding), (self.module.padding, self.module.padding)),
                mode='constant',
                constant_values=0,
            )

        conn_0 = _as_numpy_int(self.module.connections.indices[0])
        h_idx = conn_0[..., 0]
        w_idx = conn_0[..., 1]
        c_idx = conn_0[..., 2]

        selected = inputs[:, c_idx, h_idx, w_idx]
        a = selected[:, 0]
        b = selected[:, 1]

        batch, n_kernels, n_positions, n_nodes = a.shape
        a_flat = a.transpose((0, 2, 1, 3)).reshape(batch * n_positions, n_kernels * n_nodes)
        b_flat = b.transpose((0, 2, 1, 3)).reshape(batch * n_positions, n_kernels * n_nodes)

        result = apply_lut_vectorized(a_flat, b_flat, _conv_lut_ids(self.module, 0))
        result = result.reshape(batch, n_positions, n_kernels, n_nodes).transpose((0, 2, 1, 3))

        for level in range(1, self.module.tree_depth):
            conn_level = _as_numpy_int(self.module.connections.indices[level])
            selected = result[..., conn_level]

            a = selected[..., 0, :]
            b = selected[..., 1, :]

            batch, n_kernels, n_positions, n_nodes = a.shape
            a_flat = a.transpose((0, 2, 1, 3)).reshape(batch * n_positions, n_kernels * n_nodes)
            b_flat = b.transpose((0, 2, 1, 3)).reshape(batch * n_positions, n_kernels * n_nodes)

            result = apply_lut_vectorized(a_flat, b_flat, _conv_lut_ids(self.module, level))
            result = result.reshape(batch, n_positions, n_kernels, n_nodes).transpose((0, 2, 1, 3))

        out_spatial = tuple(
            (in_dim + 2 * self.module.padding - rf_size) // self.module.stride + 1
            for in_dim, rf_size in zip(self.module.in_dim, self.module.receptive_field_size)
        )
        return result[..., 0].reshape(batch, self.module.num_kernels, *out_spatial)


class ReplayLogicDense(ReplayBase):
    handles = (LogicDense,)

    def call(self, inputs: FixedVariableArray):
        self.module: LogicDense
        if self.module.lut_rank != 2:
            raise NotImplementedError('DA4ML TorchLogix conversion currently supports LogicDense with lut_rank=2.')

        connection_indices = _dense_connection_indices(self.module)
        selected = inputs[..., connection_indices]

        a = selected[..., 0, :]
        b = selected[..., 1, :]

        _, lut_ids = self.module.get_luts_and_ids()
        return apply_lut_vectorized(a, b, _as_numpy_int(lut_ids))


def im2col(inp, px_in):
    inp_col = np.lib.stride_tricks.sliding_window_view(
        inp,
        px_in,
        axis=tuple(range(len(px_in))),
    )
    inp_col = np.moveaxis(inp_col, len(px_in), -1).reshape(*inp_col.shape[: len(px_in)], -1)
    return inp_col


class ReplayOrPooling2d(ReplayBase):
    handles = (OrPooling2d,)

    def call(self, inputs: FixedVariableArray):
        self.module: OrPooling2d

        assert inputs.ndim == 4, 'Input tensor must be 4d'
        ker_size = self.module.kernel_size
        if not isinstance(ker_size, tuple):
            ker_size = (ker_size, ker_size)
        stride = self.module.stride
        if stride is None:
            stride = ker_size
        if not isinstance(stride, tuple):
            stride = (stride, stride)
        padding = self.module.padding

        if padding > 0:
            inputs = np.pad(
                inputs,
                ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                mode='constant',
                constant_values=0,
            )

        ch = inputs.shape[1]
        inp = np.moveaxis(inputs, 1, -1)
        inp = im2col(inp[0], ker_size)
        inp = inp.reshape(inp.shape[:-1] + (-1, ch))[None]
        out = np.any(inp, axis=-2)
        out: FixedVariableArray = np.moveaxis(out, -1, 1)
        out = out[:, :, :: stride[0], :: stride[1]]
        return out


class ReplayGroupSum(ReplayBase):
    handles = (GroupSum,)

    def call(self, inputs: FixedVariableArray):
        x = inputs
        x = x.reshape(*x.shape[:-1], self.module.k, x.shape[-1] // self.module.k)
        return (np.sum(x, -1) + self.module.beta) / self.module.tau
